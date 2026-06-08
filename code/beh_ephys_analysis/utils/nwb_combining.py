import json
import logging
import os
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
from dateutil.tz import tzlocal
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject
from hdmf_zarr import NWBZarrIO

from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename

from .beh_functions import get_session_tbl, get_unit_tbl, session_dirs


logger = logging.getLogger(__name__)


ARRAY_LIKE_UNIT_COLUMNS = {
    'waveform_mean',
    'waveform_sd',
    'peak_wf',
    'peak_wf_aligned',
    'peak_wf_opt',
    'peak_wf_opt_aligned',
    'peak_waveform_raw',
    'peak_waveform_raw_aligned',
    'peak_waveform_raw_fake',
    'peak_waveform_raw_fake_aligned',
    'wf_2d',
    'mat_wf_opt',
    'mat_wf_raw',
    'mat_wf_raw_aligned',
    'mat_wf_raw_fake',
    'mat_wf_raw_fake_aligned',
}


def _is_missing_value(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _serialize_table_value(value):
    """Convert dataframe cell values into NWB-safe scalar payloads."""
    if _is_missing_value(value):
        return np.nan
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray, dict)):
        try:
            if isinstance(value, np.ndarray):
                value = value.tolist()
            return json.dumps(value)
        except Exception:
            return str(value)
    return value


def _is_container_like(value):
    return isinstance(value, (list, tuple, np.ndarray, dict))


def _serialize_text_payload(value):
    """Serialize any value into text with stable handling for missing values."""
    if _is_missing_value(value):
        return ''
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    return str(value)


def _infer_text_serialized_columns(df, columns):
    """Columns with any container-like value should be serialized as text for all rows."""
    text_cols = set()
    for col in columns:
        series = df[col]
        for value in series.values:
            if _is_missing_value(value):
                continue
            if _is_container_like(value):
                text_cols.add(col)
                break
    return text_cols


def _infer_ragged_array_columns(df, columns):
    """Columns with array-like payloads but inconsistent shapes across rows."""
    ragged_cols = set()
    for col in columns:
        shapes = set()
        saw_array_like = False
        for value in df[col].values:
            if _is_missing_value(value):
                continue
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    continue
            if not isinstance(value, (list, tuple, np.ndarray)):
                continue
            try:
                arr = np.asarray(value)
            except Exception:
                continue
            saw_array_like = True
            shapes.add(tuple(arr.shape))
            if len(shapes) > 1:
                ragged_cols.add(col)
                break
        if saw_array_like and col in ragged_cols:
            continue
    return ragged_cols


def _is_empty_table(table):
    if table is None:
        return True
    try:
        return len(table) == 0
    except Exception:
        return True


def _load_unit_table_with_fallback(session_id, data_type, summary):
    """Load unit table and gracefully fallback between summary modes."""
    summary_flags = [summary]
    if summary is True:
        summary_flags.append(False)

    for summary_flag in summary_flags:
        try:
            unit_tbl = get_unit_tbl(session_id, data_type=data_type, summary=summary_flag)
        except Exception:
            unit_tbl = None

        if unit_tbl is None:
            continue

        if isinstance(unit_tbl, pd.DataFrame):
            if not _is_empty_table(unit_tbl):
                return unit_tbl, summary_flag
            continue

        try:
            unit_tbl_df = pd.DataFrame(unit_tbl)
        except Exception:
            continue
        if not _is_empty_table(unit_tbl_df):
            return unit_tbl_df, summary_flag

    return None, summary


def _serialize_times_like(value):
    if _is_missing_value(value):
        return np.array([])
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value, dtype=float)
        except Exception:
            return np.array([])
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return np.asarray(parsed, dtype=float)
            return np.asarray([float(parsed)], dtype=float)
        except Exception:
            return np.array([])
    try:
        return np.asarray([float(value)], dtype=float)
    except Exception:
        return np.array([])


def _infer_units_column_formats(source_nwb):
    """Infer expected per-column payload format from source NWB units table."""
    formats = {}
    if source_nwb is None or getattr(source_nwb, 'units', None) is None:
        return formats

    try:
        unit_cols = list(source_nwb.units.colnames)
    except Exception:
        return formats

    for col in unit_cols:
        if col in ('spike_times', 'obs_intervals'):
            continue
        try:
            col_data = source_nwb.units[col].data
            if len(col_data) == 0:
                continue
            sample = col_data[0]
        except Exception:
            continue

        if isinstance(sample, np.ndarray):
            formats[col] = 'ndarray'
        elif isinstance(sample, list):
            formats[col] = 'list'
        elif isinstance(sample, tuple):
            formats[col] = 'tuple'
        elif isinstance(sample, dict):
            formats[col] = 'dict'
        elif isinstance(sample, str):
            formats[col] = 'str'
        elif np.isscalar(sample):
            formats[col] = 'scalar'
        else:
            formats[col] = 'object'
    return formats


def _coerce_unit_value_to_format(value, expected_format, col_name=None):
    """Coerce unit table values to match source NWB column payload format."""
    if col_name in ARRAY_LIKE_UNIT_COLUMNS:
        if _is_missing_value(value):
            return np.array([])
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (list, tuple)):
                    return np.asarray(parsed)
            except Exception:
                pass
            return np.array([])
        try:
            return np.asarray(value)
        except Exception:
            return np.array([])

    # Preserve already array-like payloads instead of converting to strings.
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)) and expected_format != 'str':
        return value

    if expected_format == 'ndarray':
        if _is_missing_value(value):
            return np.array([])
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (list, tuple)):
                    return np.asarray(parsed)
            except Exception:
                pass
            return np.array([])
        try:
            return np.asarray(value)
        except Exception:
            return np.array([])

    if expected_format == 'list':
        if _is_missing_value(value):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return [value]
        return [value]

    if expected_format == 'tuple':
        if _is_missing_value(value):
            return tuple()
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (list, tuple)):
                    return tuple(parsed)
            except Exception:
                pass
            return (value,)
        return (value,)

    if expected_format == 'dict':
        if _is_missing_value(value):
            return '{}'
        if isinstance(value, dict):
            return json.dumps(value)
        if isinstance(value, str):
            return value
        return str(value)

    if expected_format == 'str':
        if _is_missing_value(value):
            return ''
        if isinstance(value, str):
            return value
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist())
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value)
        return str(value)

    if expected_format == 'scalar':
        if _is_missing_value(value):
            return np.nan
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.reshape(-1)[0].item() if isinstance(value.reshape(-1)[0], np.generic) else value.reshape(-1)[0]
            return json.dumps(value.tolist())
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value)
        return value

    return _serialize_table_value(value)


def build_nwb_from_session_and_unit_tables(
    session_id,
    data_type='curated',
    save_file=None,
    hopkins=False,
    summary=True,
):
    """
    Create a new NWB file from session and unit tables.

    - Trials: copy all columns from get_session_tbl(session_id)
    - Units: copy all columns from get_unit_tbl(session_id, data_type)
    """
    unit_tbl, unit_summary_used = _load_unit_table_with_fallback(session_id, data_type=data_type, summary=summary)
    if _is_empty_table(unit_tbl):
        logger.warning(f"No ephys to combine with for {session_id}.")
        return None, None

    if unit_summary_used != summary:
        logger.info(
            "Falling back to summary=%s unit table for %s.",
            unit_summary_used,
            session_id,
        )

    session_tbl = get_session_tbl(session_id)

    session_dir = session_dirs(session_id, hopkins=hopkins)
    source_nwb = None
    src_candidates = [
        session_dir.get(f'nwb_dir_{data_type}'),
        session_dir.get('nwb_dir_curated'),
        session_dir.get('nwb_dir_raw'),
    ]
    for src_path in src_candidates:
        if src_path is None or not os.path.exists(src_path):
            continue
        try:
            if src_path.endswith('.zarr'):
                with NWBZarrIO(src_path, mode='r') as io:
                    source_nwb = io.read()
            else:
                source_nwb = load_nwb_from_filename(src_path)
            break
        except Exception:
            source_nwb = None

    if source_nwb is not None:
        session_description = source_nwb.session_description
        session_start_time = source_nwb.session_start_time
        identifier = source_nwb.identifier
    else:
        session_description = f'Generated from session/unit tables for {session_id}'
        session_start_time = session_dir.get('datetime')
        if session_start_time is None:
            session_start_time = datetime.now(tzlocal())
        elif getattr(session_start_time, 'tzinfo', None) is None:
            session_start_time = session_start_time.replace(tzinfo=tzlocal())
        identifier = str(uuid4())

    new_nwb = NWBFile(
        session_description=session_description + " combined behavior and ephys data",
        identifier=identifier,
        session_start_time=session_start_time,
        session_id=session_id,
        institution='Allen Institute for Neural Dynamics',
    )

    if source_nwb is not None and source_nwb.subject is not None:
        try:
            new_nwb.subject = Subject(
                subject_id=getattr(source_nwb.subject, 'subject_id', 'unknown'),
                description=getattr(source_nwb.subject, 'description', ''),
                species=getattr(source_nwb.subject, 'species', None),
                sex=getattr(source_nwb.subject, 'sex', None),
                age=getattr(source_nwb.subject, 'age', None),
                weight=getattr(source_nwb.subject, 'weight', None),
            )
        except Exception:
            pass

    if session_tbl is not None and len(session_tbl) > 0:
        trial_df = session_tbl.reset_index(drop=True).copy()
        trial_cols = [col for col in trial_df.columns if col not in ('start_time', 'stop_time')]
        trial_text_cols = _infer_text_serialized_columns(trial_df, trial_cols)
        for col in trial_cols:
            new_nwb.add_trial_column(name=col, description=f'Copied from session_tbl column: {col}')

        for _, row in trial_df.iterrows():
            if 'start_time' in trial_df.columns and not _is_missing_value(row['start_time']):
                start_time = float(row['start_time'])
            elif 'goCue_start_time' in trial_df.columns and not _is_missing_value(row['goCue_start_time']):
                start_time = float(row['goCue_start_time'])
            else:
                start_time = 0.0

            if 'stop_time' in trial_df.columns and not _is_missing_value(row['stop_time']):
                stop_time = float(row['stop_time'])
            elif 'reward_outcome_time' in trial_df.columns and not _is_missing_value(row['reward_outcome_time']):
                stop_time = float(row['reward_outcome_time'])
            else:
                stop_time = start_time

            if stop_time < start_time:
                stop_time = start_time

            trial_kwargs = {}
            for col in trial_cols:
                if col in trial_text_cols:
                    trial_kwargs[col] = _serialize_text_payload(row[col])
                else:
                    trial_kwargs[col] = _serialize_table_value(row[col])
            new_nwb.add_trial(start_time=start_time, stop_time=stop_time, **trial_kwargs)
    else:
        logger.warning(f"No session table found for {session_id}; combining ephys only.")

    unit_df = unit_tbl.reset_index(drop=True).copy()
    unit_cols = [col for col in unit_df.columns if col not in ('id', 'spike_times', 'obs_intervals')]
    unit_col_formats = _infer_units_column_formats(source_nwb)
    ragged_array_cols = _infer_ragged_array_columns(unit_df, [col for col in unit_cols if col in ARRAY_LIKE_UNIT_COLUMNS])
    unit_text_cols = _infer_text_serialized_columns(
        unit_df,
        [col for col in unit_cols if col not in ARRAY_LIKE_UNIT_COLUMNS],
    )
    unit_text_cols.update(ragged_array_cols)
    for col in unit_cols:
        new_nwb.add_unit_column(name=col, description=f'Copied from unit_tbl column: {col}')

    for row_idx, row in unit_df.iterrows():
        unit_kwargs = {}

        if 'id' in unit_df.columns and not _is_missing_value(row['id']):
            try:
                unit_kwargs['id'] = int(row['id'])
            except Exception:
                unit_kwargs['id'] = row_idx
        else:
            unit_kwargs['id'] = row_idx

        if 'spike_times' in unit_df.columns:
            unit_kwargs['spike_times'] = _serialize_times_like(row['spike_times'])
        else:
            unit_kwargs['spike_times'] = np.array([])

        if 'obs_intervals' in unit_df.columns and not _is_missing_value(row['obs_intervals']):
            obs_val = row['obs_intervals']
            if isinstance(obs_val, np.ndarray):
                obs_val = obs_val.tolist()
            if isinstance(obs_val, tuple):
                obs_val = list(obs_val)
            unit_kwargs['obs_intervals'] = obs_val

        for col in unit_cols:
            if col in unit_text_cols:
                unit_kwargs[col] = _serialize_text_payload(row[col])
                continue
            expected_format = unit_col_formats.get(col)
            unit_kwargs[col] = _coerce_unit_value_to_format(row[col], expected_format, col_name=col)

        new_nwb.add_unit(**unit_kwargs)

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with NWBHDF5IO(save_file, mode='w') as io:
            io.write(new_nwb)
        logger.info(f'Generated new NWB from tables: {save_file}')
    else:
        logger.info('Generated NWB in memory only (no file written; pass save_file to write).')

    return save_file, new_nwb
