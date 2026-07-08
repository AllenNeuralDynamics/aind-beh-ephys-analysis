"""
Enhanced NWB builder that merges custom and kilosort unit tables.

This module provides a function to build an NWB file with units from both:
1. Custom pickle files (opto-tagging, CCF coordinates, etc.)
2. Kilosort NWB files (raw ephys metrics)

The columns are merged using the mappings defined in column_names_map.json.
"""
import json
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from uuid import uuid4
from dateutil.tz import tzlocal
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject

import sys
sys.path.insert(0, '/root/capsule/code/beh_ephys_analysis')
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from utils.beh_functions import get_session_tbl, get_unit_tbl, session_dirs, parseSessionID
from utils.pupil_utils import load_pupil
from pathlib import Path
from hdmf.common import DynamicTable, VectorData
from aind_dynamic_foraging_behavior_video_analysis.ephys.tongue_ephys import load_intermediate_data

logger = logging.getLogger(__name__)

# Lick/tongue movement data
LICK_DATA_DIR = Path('/root/capsule/data/all_tongue_movements_04022026')
LICK_PARQUET = LICK_DATA_DIR / 'all_tongue_movements_04022026.parquet'
KEYPOINT_TRACKING_DIR = Path('/root/capsule/data/keypoint_tracking_bottomview_LCrecordings_20260403')

# Load column mappings and descriptions
COLUMN_MAP_PATH = '/root/capsule/code/data_management/column_names_map.json'
COLUMN_DESC_PATH = '/root/capsule/code/data_management/column_names_description.json'

with open(COLUMN_MAP_PATH, 'r') as f:
    COLUMN_MAP = json.load(f)

with open(COLUMN_DESC_PATH, 'r') as f:
    COLUMN_DESCRIPTIONS = json.load(f)

# Known array columns (must be arrays, not scalars, even if all values are null)
KNOWN_ARRAY_COLUMNS = {
    'waveform_mean', 'waveform_sd',  # 2D waveform arrays
    'peak_of_optimized_waveform', 'peak_of_aligned_optimized_waveform',  # 1D arrays
    '2D_matrix_of_optimized_waveform', '2D_matrix_of_raw_waveform',  # 2D arrays
    '2D_matrix_of_aligned_raw_waveform', '2D_matrix_of_fake_raw_waveform',
    '2D_matrix_of_aligned_fake_raw_waveform',
    'waveform_on_peak_channel_of_raw_waveform', 'waveform_on_peak_channel_of_aligned_raw_waveform',
    'peak_waveform_fake_raw', 'peak_waveform_aligned_fake_raw',
}

def pupil_data_to_timeseries(pupil_data):
    """
    Convert a pupil data dict to a pynwb TimeSeries.

    Args:
        pupil_data: dict with keys 'pupil_times' (1D array, seconds) and
                    'pupil_diameter' (1D array, pixels)

    Returns:
        pynwb.TimeSeries with name 'pupil_diameter'
    """
    return TimeSeries(
        name='pupil_diameter',
        data=np.array(pupil_data['pupil_diameter'], dtype=np.float64),
        timestamps=np.array(pupil_data['pupil_times'], dtype=np.float64),
        unit='pixels',
        description='Pupil diameter measured from DLC tracking, aligned to session time.',
    )


def load_licks(session_id):
    """
    Load tongue/lick movements for a session from the parquet data asset.

    Matches session_id to the video session by animal ID and closest datetime,
    then returns only rows where has_lick=True as a pynwb DynamicTable.

    Args:
        session_id: session identifier string, e.g. 'behavior_791691_2025-06-27_13-54-30'

    Returns:
        hdmf DynamicTable with one row per lick, or None if no match found.
    """
    if not LICK_PARQUET.exists():
        logger.warning(f"Lick parquet not found at {LICK_PARQUET}")
        return None

    all_lick_df = pd.read_parquet(LICK_PARQUET)
    session_video_list = all_lick_df['session'].unique().tolist()

    animal_id, session_time, _ = parseSessionID(session_id)
    if animal_id is None:
        logger.warning(f"Could not parse session_id: {session_id}")
        return None

    candidate_sessions = [s for s in session_video_list if str(s).startswith(f'behavior_{animal_id}')]
    if not candidate_sessions:
        logger.info(f"No lick data found for animal {animal_id}")
        return None

    time_diffs = [abs((parseSessionID(s)[1] - session_time).total_seconds()) for s in candidate_sessions]
    best_idx = int(np.argmin(time_diffs))
    if time_diffs[best_idx] > 60:
        logger.info(f"Closest lick session is {time_diffs[best_idx]:.0f}s away — skipping")
        return None

    matched_session = candidate_sessions[best_idx]
    logger.info(f"Matched lick session: {matched_session}")

    licks = all_lick_df[(all_lick_df['session'] == matched_session) & (all_lick_df['has_lick'])].copy().reset_index(drop=True)
    if len(licks) == 0:
        logger.info("No lick rows found after filtering has_lick=True")
        return None

    # Columns to include in the DynamicTable (drop session identifier and redundant flags)
    exclude_cols = {'session', 'has_lick'}
    col_descriptions = {
        'movement_id': 'Unique tongue movement identifier',
        'start_time': 'Movement onset time relative to session start (s)',
        'end_time': 'Movement offset time relative to session start (s)',
        'duration': 'Movement duration (s)',
        'lick_time': 'Time of lick contact relative to session start (s)',
        'lick_count': 'Number of lick contacts within this movement',
        'lick_latency': 'Latency from go cue to lick contact (s)',
        'trial': 'Trial number',
        'cue_response': 'Whether the animal responded to the cue',
        'rewarded': 'Whether the trial was rewarded',
        'event': 'Lick event type (left_lick_time / right_lick_time)',
        'peak_velocity': 'Peak tongue velocity (px/s)',
        'mean_velocity': 'Mean tongue velocity (px/s)',
        'total_distance': 'Total tongue path distance (px)',
        'excursion_angle_deg': 'Tongue excursion angle (degrees)',
        'goCue_start_time_in_session': 'Go cue time for this trial relative to session start (s)',
        'movement_latency_from_go': 'Latency from go cue to movement onset (s)',
        'movement_number_in_trial': 'Index of this movement within the trial',
    }

    keep_cols = [c for c in licks.columns if c not in exclude_cols]

    table = DynamicTable(
        name='licks',
        description='Tongue lick movements detected from video DLC tracking, one row per lick.',
    )

    for col in keep_cols:
        series = licks[col]
        # Normalise nullable integer / boolean dtypes to plain numpy
        if hasattr(series, 'to_numpy'):
            arr = series.to_numpy(dtype=object, na_value=np.nan)
            # Cast to float if numeric, keeping NaN for missing values
            try:
                arr = arr.astype(np.float64)
            except (ValueError, TypeError):
                arr = arr.astype(object)
        else:
            arr = np.array(series, dtype=object)

        table.add_column(
            name=col,
            description=col_descriptions.get(col, col),
            data=arr.tolist(),
        )

    logger.info(f"Built lick DynamicTable with {len(licks)} rows and {len(keep_cols)} columns")
    return table


def _df_to_dynamic_table(df, name, description):
    """Convert a DataFrame to an hdmf DynamicTable, coercing nullable dtypes to plain numpy."""
    table = DynamicTable(name=name, description=description)
    for col in df.columns:
        arr = df[col].to_numpy(dtype=object, na_value=np.nan)
        try:
            arr = arr.astype(np.float64)
        except (ValueError, TypeError):
            arr = arr.astype(object)
        table.add_column(name=col, description=col, data=arr.tolist())
    return table


def load_keypoint_tracking(session_id):
    """
    Load tongue keypoint tracking data for a session from the bottomview DLC asset.

    Matches session_id to the nearest session directory under KEYPOINT_TRACKING_DIR
    by animal ID and datetime, then calls load_intermediate_data and returns
    DynamicTables for the movement summary (movs) and per-frame kinematics (kins).

    Args:
        session_id: session identifier string

    Returns:
        Tuple (movs_table, kins_table), or (None, None) if no match found.
    """
    if not KEYPOINT_TRACKING_DIR.exists():
        logger.warning(f"Keypoint tracking directory not found: {KEYPOINT_TRACKING_DIR}")
        return None, None

    animal_id, session_time, _ = parseSessionID(session_id)
    if animal_id is None:
        logger.warning(f"Could not parse session_id: {session_id}")
        return None, None

    candidate_dirs = [d for d in KEYPOINT_TRACKING_DIR.iterdir()
                      if d.is_dir() and d.name.startswith(f'behavior_{animal_id}')]
    if not candidate_dirs:
        logger.info(f"No keypoint tracking data found for animal {animal_id}")
        return None, None

    time_diffs = [abs((parseSessionID(d.name)[1] - session_time).total_seconds()) for d in candidate_dirs]
    best_idx = int(np.argmin(time_diffs))
    if time_diffs[best_idx] > 60:
        logger.info(f"Closest keypoint session is {time_diffs[best_idx]:.0f}s away — skipping")
        return None, None

    matched_dir = candidate_dirs[best_idx]
    logger.info(f"Matched keypoint tracking session: {matched_dir.name}")

    try:
        data = load_intermediate_data(matched_dir)
    except Exception as e:
        logger.warning(f"Failed to load intermediate data from {matched_dir}: {e}")
        return None, None

    movs_table = _df_to_dynamic_table(
        data['movs'],
        name='tongue_movements',
        description='Tongue movement summary table from DLC keypoint tracking (one row per movement).',
    )
    kins_table = _df_to_dynamic_table(
        data['kins'],
        name='tongue_kinematics',
        description='Per-frame tongue kinematics from DLC keypoint tracking (x, y, velocity, confidence).',
    )

    logger.info(f"Built tongue_movements ({len(data['movs'])} rows) and tongue_kinematics ({len(data['kins'])} rows) tables")
    return movs_table, kins_table


def merge_unit_tables(session_id, data_type='curated', return_nwb=False):
    """
    Merge unit tables from custom pickle and NWB kilosort data.

    Args:
        session_id: Session identifier
        data_type: 'curated' or 'raw'
        return_nwb: If True, return (merged_df, ephys_nwb). If False, return just merged_df

    Returns:
        If return_nwb=False: Merged DataFrame with mapped column names, or None if merge fails
        If return_nwb=True: Tuple of (merged_df, ephys_nwb) or (None, None) if merge fails
    """
    # 1. Load custom unit table (use summary version)
    custom_unit_tbl = get_unit_tbl(session_id, data_type=data_type, summary=True)
    if custom_unit_tbl is None or len(custom_unit_tbl) == 0:
        logger.warning(f"No custom unit table found for {session_id}")
        return (None, None) if return_nwb else None

    logger.info(f"Loaded {len(custom_unit_tbl)} units from custom table")

    # 2. Load NWB kilosort data
    session_dir = session_dirs(session_id)
    nwb_path = session_dir.get(f'nwb_dir_{data_type}')
    if nwb_path is None or not os.path.exists(nwb_path):
        logger.warning(f"NWB file not found at {nwb_path}")
        return (None, None) if return_nwb else None

    ephys_nwb = load_nwb_from_filename(nwb_path)
    if ephys_nwb.units is None:
        logger.warning(f"No units in NWB file for {session_id}")
        return (None, None) if return_nwb else None

    nwb_unit_tbl = ephys_nwb.units.to_dataframe()
    logger.info(f"Loaded {len(nwb_unit_tbl)} units from NWB")

    # 3. Verify and align by unit_id / ks_unit_id
    custom_unit_ids = set(custom_unit_tbl['unit_id'].values)

    # Determine which ID column the NWB file uses
    if 'ks_unit_id' in nwb_unit_tbl.columns:
        nwb_id_col = 'ks_unit_id'
    elif 'unit_id' in nwb_unit_tbl.columns:
        nwb_id_col = 'unit_id'
    else:
        logger.error(f"NWB units table has neither 'ks_unit_id' nor 'unit_id'. Columns: {list(nwb_unit_tbl.columns)}")
        return None

    logger.info(f"Using NWB ID column: '{nwb_id_col}' for alignment")
    nwb_unit_ids = set(nwb_unit_tbl[nwb_id_col].values)
    common_ids = custom_unit_ids & nwb_unit_ids

    if len(common_ids) == 0:
        logger.error(f"No common units found between custom and NWB tables!")
        logger.error(f"  Custom unit_ids ({len(custom_unit_ids)}): {sorted(list(custom_unit_ids))[:10]}")
        logger.error(f"  NWB {nwb_id_col} ({len(nwb_unit_ids)}): {sorted(list(nwb_unit_ids))[:10]}")
        return None

    if len(custom_unit_tbl) != len(common_ids):
        only_custom = custom_unit_ids - nwb_unit_ids
        logger.warning(
            f"Row count mismatch: {len(custom_unit_tbl)} custom units "
            f"but only {len(common_ids)} found in NWB. "
            f"Missing from NWB: {sorted(list(only_custom))[:10]}"
        )

    # Align tables
    custom_aligned = custom_unit_tbl[custom_unit_tbl['unit_id'].isin(common_ids)].sort_values('unit_id').reset_index(drop=True)
    nwb_aligned = nwb_unit_tbl[nwb_unit_tbl[nwb_id_col].isin(common_ids)].sort_values(nwb_id_col).reset_index(drop=True)

    logger.info(f"Aligned {len(custom_aligned)} common units")

    # 4. Apply column mappings and merge
    merged_df = pd.DataFrame()
    unit_columns_custom = COLUMN_MAP['unit_columns_custom']
    unit_columns_ks = COLUMN_MAP['unit_columns_ks']

    # Add custom columns with mapped names
    for orig_col, mapped_name in unit_columns_custom.items():
        if orig_col not in custom_aligned.columns:
            continue

        # Skip duplicates (will be taken from NWB)
        if 'duplicate as' in mapped_name:
            continue

        # Handle similar columns - use descriptive name
        if 'similar to' in mapped_name:
            clean_name = mapped_name.split(';')[0].strip()
            merged_df[clean_name] = custom_aligned[orig_col].values
        else:
            merged_df[mapped_name] = custom_aligned[orig_col].values

    # Add NWB columns with mapped names
    for orig_col, mapped_name in unit_columns_ks.items():
        if orig_col not in nwb_aligned.columns:
            continue

        # Skip if already exists
        if mapped_name in merged_df.columns:
            continue

        merged_df[mapped_name] = nwb_aligned[orig_col].values

    logger.info(f"Merged table has {len(merged_df)} rows and {len(merged_df.columns)} columns")

    if return_nwb:
        return merged_df, ephys_nwb
    else:
        return merged_df


def build_combined_nwb(session_id, data_type='curated', save_file=None):
    """
    Build a complete NWB file with available data modalities.

    Combines whichever data is available:
    - Behavior trials (from session table)
    - Ephys units (merged custom + kilosort)
    - Acquisition TimeSeries (lick times, reward times, etc.)

    Args:
        session_id: Session identifier
        data_type: 'curated' or 'raw'
        save_file: Path to save NWB file (if None, returns in-memory only)

    Returns:
        Tuple of (save_path, nwb_object, data_modalities_dict)
        data_modalities_dict has keys:
            'behavior_trials': bool - whether trial data is included
            'ephys_units': bool - whether ephys units are included
            'lick_times': bool - whether lick acquisition is included
            'reward_times': bool - whether reward acquisition is included
            'FP': bool - whether fiber photometry is included
            'beh_version': str - 'raw', 'processed', or 'none'
            'nwb_created': str - ISO timestamp when NWB object was created
            'nwb_saved': str or None - ISO timestamp when NWB was saved to file (None if not saved)
    """
    logger.info(f"Building combined NWB for {session_id}")

    # Track which data modalities are included
    data_modalities = {
        'behavior_trials': False,
        'ephys_units': False,
        'lick_times': False,
        'reward_times': False,
        'FP': False,  # Fiber photometry
        'pupil': False,
        'lick_video': False,
        'keypoint_tracking': False,
        'beh_version': 'none',  # 'raw', 'processed', or 'none'
        'nwb_created': None,  # Timestamp when NWB object was created
        'nwb_saved': None,  # Timestamp when NWB file was saved (if save_file provided)
    }

    # 1. Merge unit tables (optional - may not exist for all sessions)
    # Get both merged units and ephys NWB for metadata
    merge_result = merge_unit_tables(session_id, data_type, return_nwb=True)
    if merge_result[0] is None:
        logger.warning(f"No unit tables to merge for {session_id} - will create NWB with behavior/acquisition only")
        merged_units = None
        ephys_nwb = None
    else:
        merged_units, ephys_nwb = merge_result
        logger.info(f"Merged {len(merged_units)} units")
        data_modalities['ephys_units'] = True

    # 2. Load session/trial table (optional - may not exist for all sessions)
    # Try raw version first, then processed version
    session_tbl = get_session_tbl(session_id, load_raw=True)
    if session_tbl is not None and len(session_tbl) > 0:
        logger.info(f"Loaded {len(session_tbl)} trials from raw behavior NWB")
        data_modalities['beh_version'] = 'raw'
    else:
        # Try processed version
        session_tbl = get_session_tbl(session_id, load_raw=False)
        if session_tbl is not None and len(session_tbl) > 0:
            logger.info(f"Loaded {len(session_tbl)} trials from processed behavior NWB")
            data_modalities['beh_version'] = 'processed'
        else:
            logger.warning(f"No session table found for {session_id} - will create NWB without behavior trials")
            session_tbl = None
            data_modalities['beh_version'] = 'none'

    # 2b. Load behavior NWB for acquisition data
    session_dir = session_dirs(session_id)
    behavior_nwb_path = os.path.join(session_dir['beh_fig_dir'], session_id + '.nwb')
    behavior_nwb = None
    if os.path.exists(behavior_nwb_path):
        behavior_nwb = load_nwb_from_filename(behavior_nwb_path)
        logger.info(f"Loaded behavior NWB from {behavior_nwb_path}")
    else:
        logger.warning(f"Behavior NWB not found at {behavior_nwb_path}")

    # 3. Get session metadata - use from ephys NWB if available, otherwise behavior NWB
    # Priority: ephys_nwb > behavior_nwb > defaults
    source_nwb = ephys_nwb if ephys_nwb is not None else behavior_nwb

    if source_nwb is not None:
        session_description = source_nwb.session_description
        session_start_time = source_nwb.session_start_time
        source_session_id = source_nwb.session_id if hasattr(source_nwb, 'session_id') else session_id
        logger.info(f"Using metadata from {'ephys' if ephys_nwb is not None else 'behavior'} NWB")
    else:
        # Fallback to defaults
        session_description = f"Combined behavior and ephys data for {session_id}"
        session_dir = session_dirs(session_id)
        session_start_time = session_dir.get('datetime')
        if session_start_time is None:
            session_start_time = datetime.now(tzlocal())
        elif getattr(session_start_time, 'tzinfo', None) is None:
            session_start_time = session_start_time.replace(tzinfo=tzlocal())
        source_session_id = session_id
        logger.info("Using default metadata (no source NWB available)")

    # 4. Create NWB file
    creation_time = datetime.now(tzlocal())
    new_nwb = NWBFile(
        session_description=session_description,
        identifier=f"{session_id}_merged_{creation_time.strftime('%Y%m%d_%H%M%S')}",
        session_start_time=session_start_time,
        session_id=source_session_id,
        institution='Allen Institute for Neural Dynamics',
    )

    # Track creation time
    data_modalities['nwb_created'] = creation_time.isoformat()
    logger.info("Created NWB file")

    # 5. Add trials with descriptions (if session table exists)
    if session_tbl is not None:
        trial_df = session_tbl.reset_index(drop=True).copy()
        trial_cols = [col for col in trial_df.columns if col not in ('start_time', 'stop_time')]
        trial_descriptions = COLUMN_DESCRIPTIONS.get('behavior_trial_columns', {})

    for col in trial_cols:
        description = trial_descriptions.get(col, f'Trial column: {col}')
        new_nwb.add_trial_column(name=col, description=description)

    for _, row in trial_df.iterrows():
        start_time = float(row.get('start_time', 0.0))
        stop_time = float(row.get('stop_time', start_time))
        if stop_time < start_time:
            stop_time = start_time

        trial_kwargs = {}
        for col in trial_cols:
            if col not in row.index:
                continue
            val = row[col]

            # Convert Python None to appropriate type (like reference behavior NWB)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                # Check a non-null value in this column to determine type
                non_null_vals = trial_df[col].dropna()
                if len(non_null_vals) > 0:
                    sample_val = non_null_vals.iloc[0]
                    if isinstance(sample_val, np.ndarray):
                        # Array column - use empty array
                        val = np.array([])
                    else:
                        # Scalar column - use np.nan
                        val = np.nan
                else:
                    # All values are None - default to np.nan
                    val = np.nan

            trial_kwargs[col] = val

        new_nwb.add_trial(start_time=start_time, stop_time=stop_time, **trial_kwargs)

        logger.info(f"Added {len(trial_df)} trials with {len(trial_cols)} columns")
        data_modalities['behavior_trials'] = True
    else:
        logger.info("No behavior trials to add")

    # 5b. Add acquisition TimeSeries from behavior NWB (length > 1)
    if behavior_nwb and behavior_nwb.acquisition:
        for acq_name, acq_data in behavior_nwb.acquisition.items():
            if hasattr(acq_data, 'timestamps') and len(acq_data.timestamps) > 1:
                # Copy TimeSeries to new NWB
                from pynwb import TimeSeries
                new_ts = TimeSeries(
                    name=acq_name,
                    data=acq_data.data[:],
                    timestamps=acq_data.timestamps[:],
                    unit=acq_data.unit if hasattr(acq_data, 'unit') else 'N/A',
                    description=acq_data.description if hasattr(acq_data, 'description') else ''
                )
                new_nwb.add_acquisition(new_ts)
                logger.info(f"Added acquisition TimeSeries: {acq_name} ({len(acq_data.timestamps)} timestamps)")

                # Track lick, reward, and fiber photometry modalities
                if 'lick' in acq_name.lower():
                    data_modalities['lick_times'] = True
                if 'reward' in acq_name.lower():
                    data_modalities['reward_times'] = True
                # Photometry channels start with 'G' or 'Iso' (but not FIP which is timing signal)
                if (acq_name.startswith('G') or acq_name.startswith('Iso')) and not acq_name.startswith('FIP'):
                    data_modalities['FP'] = True

    # 6. Add merged units with descriptions (if units exist)
    if merged_units is not None:
        unit_df = merged_units.reset_index(drop=True).copy()

        # Predefined NWB columns that should NOT be added via add_unit_column()
        predefined_cols = ['spike_times', 'electrodes', 'obs_intervals', 'electrode_group']

        # Separate custom columns from predefined ones
        unit_cols = [col for col in unit_df.columns if col not in predefined_cols]
        unit_descriptions = COLUMN_DESCRIPTIONS.get('unit_columns', {})

        # Create reverse mapping: mapped_name -> original_name for looking up descriptions
        mapped_to_original = {}
        for orig_col, mapped_name in COLUMN_MAP['unit_columns_custom'].items():
            if 'duplicate as' in mapped_name:
                continue
            if 'similar to' in mapped_name:
                clean_name = mapped_name.split(';')[0].strip()
                mapped_to_original[clean_name] = orig_col
            else:
                mapped_to_original[mapped_name] = orig_col

        for orig_col, mapped_name in COLUMN_MAP['unit_columns_ks'].items():
            if mapped_name not in mapped_to_original:
                mapped_to_original[mapped_name] = orig_col

        # Only add custom columns (not predefined ones)
        for col in unit_cols:
            # Look up description using original column name
            orig_col = mapped_to_original.get(col, col)
            description = unit_descriptions.get(orig_col, col)
            if description == 'to be filled':
                description = col
            new_nwb.add_unit_column(name=col, description=description)

        for idx, row in unit_df.iterrows():
            unit_kwargs = {}

            # Handle predefined columns (spike_times, electrodes, etc.)
            if 'spike_times' in unit_df.columns:
                spike_times = row['spike_times']
                if isinstance(spike_times, (list, np.ndarray)):
                    unit_kwargs['spike_times'] = np.array(spike_times, dtype=np.float64)
                else:
                    unit_kwargs['spike_times'] = np.array([], dtype=np.float64)
            else:
                unit_kwargs['spike_times'] = np.array([], dtype=np.float64)

            # Handle electrodes if present (pass as parameter, not as custom column)
            # Skip if no electrode table exists in the NWB
            if 'electrodes' in unit_df.columns and new_nwb.electrodes is not None:
                val = row['electrodes']
                if isinstance(val, pd.DataFrame):
                    unit_kwargs['electrodes'] = list(val.index)
                elif val is not None and not pd.isna(val):
                    unit_kwargs['electrodes'] = val

            # Add custom columns (convert None to appropriate type)
            for col in unit_cols:
                val = row[col]

                # Convert Python None to np.nan for numeric types, empty array for array types
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    # Check if this is a known array column
                    if col in KNOWN_ARRAY_COLUMNS:
                        val = np.array([])  # Empty array for known array columns
                    else:
                        # Check a non-null value in this column to determine type
                        non_null_vals = unit_df[col].dropna()
                        if len(non_null_vals) > 0:
                            sample_val = non_null_vals.iloc[0]
                            if isinstance(sample_val, np.ndarray):
                                # Array column - use empty array with same ndim
                                if sample_val.ndim == 1:
                                    val = np.array([])
                                else:
                                    # For 2D arrays, need to match shape
                                    shape = list(sample_val.shape)
                                    shape[0] = 0  # Empty in first dimension
                                    val = np.empty(shape)
                            else:
                                # Scalar column - use np.nan
                                val = np.nan
                        else:
                            # All values are None and not a known array - default to np.nan
                            val = np.nan

                unit_kwargs[col] = val

            new_nwb.add_unit(**unit_kwargs)

        logger.info(f"Added {len(unit_df)} units with {len(unit_cols)} columns")
    else:
        logger.info("No units to add - behavior/acquisition only NWB")

    # 7. Add pupil data to behavior processing module (if available)
    pupil_data = load_pupil(session_id)
    if pupil_data is not None:
        if 'behavior' not in new_nwb.processing:
            new_nwb.create_processing_module(
                name='behavior',
                description='Processed behavioral data',
            )
        new_nwb.processing['behavior'].add(pupil_data_to_timeseries(pupil_data))
        data_modalities['pupil'] = True
        logger.info("Added pupil diameter TimeSeries to behavior processing module")
    else:
        logger.info("No pupil data available for this session")

    # 7b. Add lick DynamicTable to behavior processing module (if available)
    lick_table = load_licks(session_id)
    if lick_table is not None:
        if 'behavior' not in new_nwb.processing:
            new_nwb.create_processing_module(
                name='behavior',
                description='Processed behavioral data',
            )
        new_nwb.processing['behavior'].add(lick_table)
        data_modalities['lick_video'] = True
        logger.info(f"Added lick DynamicTable to behavior processing module")
    else:
        logger.info("No lick video data available for this session")

    # 7c. Add keypoint tracking tables to behavior processing module (if available)
    movs_table, kins_table = load_keypoint_tracking(session_id)
    if movs_table is not None and kins_table is not None:
        if 'behavior' not in new_nwb.processing:
            new_nwb.create_processing_module(
                name='behavior',
                description='Processed behavioral data',
            )
        new_nwb.processing['behavior'].add(movs_table)
        new_nwb.processing['behavior'].add(kins_table)
        data_modalities['keypoint_tracking'] = True
        logger.info("Added tongue_movements and tongue_kinematics tables to behavior processing module")
    else:
        logger.info("No keypoint tracking data available for this session")

    # 8. Log data modalities included
    included_modalities = [k for k, v in data_modalities.items() if v]
    logger.info(f"Data modalities included: {', '.join(included_modalities) if included_modalities else 'none'}")

    # 9. Save if requested
    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        save_time = datetime.now(tzlocal())
        with NWBHDF5IO(save_file, mode='w') as io:
            io.write(new_nwb)
        data_modalities['nwb_saved'] = save_time.isoformat()
        logger.info(f"Saved combined NWB to {save_file}")
    else:
        logger.info("Generated NWB in memory only (no file written)")

    return save_file, new_nwb, data_modalities


if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)

    sessions = [
        'behavior_754897_2025-03-13_11-20-42',
    ]

    for session in sessions:
        print(f"\n{'='*80}")
        print(f"Testing: {session}")
        print(f"{'='*80}\n")

        # Test the full build_combined_nwb function
        save_path, nwb = build_combined_nwb(session, data_type='curated', save_file=None)
        if nwb is not None:
            print(f"\n✓ Success! Combined NWB created")
            print(f"  Trials: {len(nwb.trials)} rows")
            print(f"  Units: {len(nwb.units)} rows")

            # Show sample columns
            trials_df = nwb.trials.to_dataframe()
            units_df = nwb.units.to_dataframe()
            print(f"\n  Trial columns ({len(trials_df.columns)}): {list(trials_df.columns)[:5]}...")
            print(f"  Unit columns ({len(units_df.columns)}): {list(units_df.columns)[:5]}...")
        else:
            print("\n✗ Build failed")
