import os
import sys
# Resolve code/beh_ephys_analysis (the folder containing `utils`) relative to this
# file's location, so imports work no matter where the repo is checked out.
import os
import sys
_anchor = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath(os.getcwd())
while _anchor != os.path.dirname(_anchor):
    _beh_ephys_root = os.path.join(_anchor, "code", "beh_ephys_analysis")
    if os.path.isdir(os.path.join(_beh_ephys_root, "utils")):
        if _beh_ephys_root in sys.path:
            sys.path.remove(_beh_ephys_root)
        sys.path.insert(0, _beh_ephys_root)
        break
    _anchor = os.path.dirname(_anchor)
from utils.capsule_migration import CAPSULE_ROOT

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import spikeinterface as si
import spikeinterface.preprocessing as spre

from joblib import Parallel, delayed

from utils.beh_functions import session_dirs, get_unit_tbl, get_session_tbl
from utils.plot_utils import combine_pdf_big
from utils.ephys_functions import load_drift


def waveform_check(session, data_type='curated', opto_only=True, units=None):
    if get_session_tbl(session) is None:
        print(f'{session}: No session table found, skipping.')
        return
    sample_num = 10
    session_dir = session_dirs(session)
    analyzer_path = session_dir[f'postprocessed_dir_{data_type}']
    analyzer = si.load(analyzer_path, load_extensions=False)
    sparsity = analyzer.sparsity
    waveform_extractor = analyzer.get_extension("waveforms")
    random_spikes = analyzer.get_extension("random_spikes")
    unit_tbl = get_unit_tbl(session, data_type=data_type, summary=True)
    compressed_folder = session_dir['session_dir_raw']
    stream_name = 'ProbeA'
    recording_zarr = [
        os.path.join(compressed_folder, f)
        for f in os.listdir(compressed_folder)
        if stream_name in f and 'LFP' not in f
    ][0]
    recording = si.read_zarr(recording_zarr)
    recording_ps = spre.phase_shift(recording, margin_ms=100.0)
    recording_bp = spre.bandpass_filter(recording_ps)
    recording_cr = spre.common_reference(recording_bp)

    channel_locations = analyzer.get_channel_locations()
    all_channels = analyzer.sparsity.channel_ids
    right_left = channel_locations[:, 0] < 20

    if units is None:
        if opto_only:
            if 'opto_pass' not in unit_tbl.columns:
                print('No tagged units found in unit table.')
                return
            if unit_tbl['opto_pass'].isnull().all() or not unit_tbl['tagged'].any():
                print(f'{session}: No tagged units found.')
                return
            else:
                units = unit_tbl.query('opto_pass == True')['unit_id'].values
        else:
            units = unit_tbl.unit_id.values

    for unit in units:
        out_path = os.path.join(
            session_dir[f'ephys_fig_dir_{data_type}'], 'waveforms_check',
            f'unit_{unit}_waveforms.pdf'
        )
        if os.path.exists(out_path):
            print(f'Unit {unit} already processed. Skipping...')
            continue

        waveforms = waveform_extractor.get_waveforms_one_unit(unit_id=unit)
        unit_channels_ind = sparsity.unit_id_to_channel_indices[unit]
        right_left_unit = right_left[unit_channels_ind]

        spike_times = unit_tbl[unit_tbl['unit_id'] == unit].spike_times.values[0]
        isi_v_ratio = unit_tbl[unit_tbl['unit_id'] == unit].isi_violations_ratio.values[0]
        amp = unit_tbl[unit_tbl['unit_id'] == unit].amp.values[0]
        peak = unit_tbl[unit_tbl['unit_id'] == unit].peak.values[0]
        snr = unit_tbl[unit_tbl['unit_id'] == unit].snr.values[0]

        mean_wf = np.squeeze(np.mean(waveforms, axis=0))
        wf_time = np.linspace(-3, 4, np.shape(waveforms)[1])
        peak_channel_id = np.argmax(np.ptp(mean_wf, axis=0))
        peak_waveform_mean = mean_wf[:, peak_channel_id]
        peak_waveforms = np.squeeze(waveforms[:, :, peak_channel_id])
        waveforms_sd = np.std(peak_waveforms, axis=0)
        waveforms_sem = waveforms_sd / np.sqrt(waveforms.shape[0])
        count = np.shape(waveforms)[0]

        # raw traces: channels on the same side as peak, closest 7
        peak_left_right = right_left_unit[peak_channel_id]
        unit_channels_major_ind = unit_channels_ind[right_left_unit == peak_left_right]
        unit_channels_major_ind = unit_channels_major_ind[
            np.argsort(np.abs(unit_channels_major_ind - unit_channels_ind[peak_channel_id]))[:7]
        ]
        unit_channels_major_ind = np.sort(unit_channels_major_ind)
        recording_unit = recording_cr.select_channels(all_channels[unit_channels_major_ind])
        spike_samples = analyzer.sorting.get_unit_spike_train(unit_id=unit)

        unit_drift = load_drift(session, unit, data_type=data_type)
        if unit_drift is not None:
            if unit_drift['ephys_cut'] is not None:
                if unit_drift['ephys_cut'][0] is not None:
                    spike_samples = spike_samples[spike_times >= unit_drift['ephys_cut'][0]]
                    spike_times = spike_times[spike_times >= unit_drift['ephys_cut'][0]]
                if unit_drift['ephys_cut'][1] is not None:
                    spike_samples = spike_samples[spike_times <= unit_drift['ephys_cut'][1]]
                    spike_times = spike_times[spike_times <= unit_drift['ephys_cut'][1]]

        pre_sample = int(30000 * 0.01)
        post_sample = int(30000 * 0.01)
        valid = (spike_samples >= pre_sample) & (spike_samples <= (recording_unit.get_num_frames() - post_sample))
        spike_times = spike_times[valid]
        spike_samples = spike_samples[valid]
        spike_inds = np.random.randint(0, len(spike_times), size=sample_num)

        bins = 6
        fig = plt.figure(figsize=(15, 3 + sample_num * 1.5))
        gs = gridspec.GridSpec(1 + sample_num, 2 + bins, height_ratios=[1] + [0.5] * sample_num)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(wf_time, peak_waveform_mean, color='k', label='mean')
        ax.fill_between(wf_time, peak_waveform_mean - waveforms_sd,
                        peak_waveform_mean + waveforms_sd, color='k', alpha=0.2, label='sd', edgecolor='none')
        ax.fill_between(wf_time, peak_waveform_mean - waveforms_sem,
                        peak_waveform_mean + waveforms_sem, color='k', alpha=0.4, label='sem', edgecolor='none')
        ax.set_title(f'A: {amp:.1f}, P: {peak:.1f}')

        edges = np.linspace(0, count, bins + 1, dtype=int)
        upper_lim = np.max(peak_waveforms)
        lower_lim = np.min(peak_waveforms)
        for i in range(bins):
            ax = fig.add_subplot(gs[0, i + 1])
            ax.plot(wf_time, peak_waveforms[edges[i]:edges[i + 1], :].T, color='k', alpha=0.5, linewidth=0.2)
            ax.set_ylim(lower_lim, upper_lim)
            ax.set_title(f'Waveform {i + 1}')

        corr = [np.corrcoef(peak_waveforms[i, :], peak_waveform_mean)[0, 1] for i in range(count)]
        ax = fig.add_subplot(gs[0, -1])
        ax.hist(x=corr, bins=50, color='k', edgecolor='none', alpha=0.5)
        ax.set_title(f'Corr, Mean: {np.mean(corr):.2f}')

        for sample_ind in range(sample_num):
            curr_sample_spike = spike_inds[sample_ind]
            start_frame = spike_samples[curr_sample_spike] - pre_sample
            stop_frame = spike_samples[curr_sample_spike] + post_sample
            curr_signal = recording_unit.get_traces(
                start_frame=start_frame, end_frame=stop_frame, return_scaled=True
            )
            ax = fig.add_subplot(gs[sample_ind + 1, :])
            ax.imshow(curr_signal.T,
                      extent=(-pre_sample / 30, post_sample / 30, 0, len(unit_channels_major_ind)),
                      aspect='auto', cmap='gray', interpolation='none')
            spike_samples_in_window = spike_samples[
                (spike_samples >= start_frame) & (spike_samples <= stop_frame)
            ]
            spike_times_in_window = (spike_samples_in_window - spike_samples[curr_sample_spike]) / 30
            for spike_time in spike_times_in_window:
                ax.axvline(x=spike_time, color='r', linewidth=0.5)

        plt.suptitle(f'{session} Unit {unit} - ISI Violations Ratio: {isi_v_ratio:.3f}, SNR: {snr:.2f}')
        plt.tight_layout()
        os.makedirs(os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], 'waveforms_check'), exist_ok=True)
        plt.savefig(out_path)
        plt.close()

    print(f'{session}: Done plotting waveforms for all units.')


def compare_mean_go_cue_waveforms(session, data_type='curated', versions=('bandpass', 'raw')):
    """Compare full-session mean waveforms against go-cue aligned mean waveforms.

    Args:
        session: Session id like 'behavior_754897_2025-03-14_11-28-53'.
        data_type: Data type key used in session dirs (typically 'curated').
        versions: Iterable of waveform versions to plot. Supported values are
            'bandpass' and 'raw'.
    """
    session_dir = session_dirs(session)

    analyzer_all = si.load(session_dir[f'postprocessed_dir_{data_type}'], load_extensions=False)

    all_raw_file = os.path.join(session_dir[f'ephys_dir_{data_type}'], 'raw_fake.zarr')
    go_cue_raw_file = os.path.join(session_dir[f'ephys_dir_{data_type}'], 'gocue_raw_fake.zarr')
    go_cue_bp_file = os.path.join(session_dir[f'ephys_dir_{data_type}'], 'gocue_bp.zarr')

    # Validate required analyzer files for selected versions.
    required_files = []
    if 'bandpass' in versions:
        required_files.append(go_cue_bp_file)
    if 'raw' in versions:
        required_files.extend([all_raw_file, go_cue_raw_file])

    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        print(f'{session}: Missing waveform analyzer files: {missing}')
        return

    analyzer_gocue_bp = si.load(go_cue_bp_file, load_extensions=False) if 'bandpass' in versions else None
    analyzer_gocue_raw = si.load(go_cue_raw_file, load_extensions=False) if 'raw' in versions else None
    analyzer_all_raw = si.load(all_raw_file, load_extensions=False) if 'raw' in versions else None

    units_focus = analyzer_gocue_bp.sorting.get_unit_ids() if analyzer_gocue_bp is not None else analyzer_gocue_raw.sorting.get_unit_ids()
    if len(units_focus) == 0:
        print(f'{session}: No units found for go-cue waveform comparison.')
        return

    ncols = 3
    nrows = int(np.ceil(len(units_focus) / ncols))
    wf_all = analyzer_all.get_extension('templates')

    output_dir = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], 'waveforms_check_go_cue')
    os.makedirs(output_dir, exist_ok=True)
    out_path_base = os.path.join(output_dir, 'go_cue_waveforms.pdf')

    version_configs = {
        'bandpass': {
            'gocue_ext': analyzer_gocue_bp.get_extension('templates') if analyzer_gocue_bp is not None else None,
            'all_ext': wf_all,
            'all_time': (-3, 4),
            'gocue_time': (-5, 10),
            'title': 'bandpass',
        },
        'raw': {
            'gocue_ext': analyzer_gocue_raw.get_extension('templates') if analyzer_gocue_raw is not None else None,
            'all_ext': analyzer_all_raw.get_extension('templates') if analyzer_all_raw is not None else None,
            'all_time': (-1.5, 2.5),
            'gocue_time': (-5, 10),
            'title': 'raw',
        },
    }

    for version in versions:
        if version not in version_configs:
            print(f"{session}: Unsupported version '{version}', skipping.")
            continue

        cfg = version_configs[version]
        if cfg['gocue_ext'] is None or cfg['all_ext'] is None:
            print(f"{session}: Missing analyzer extension for version '{version}', skipping.")
            continue

        fig = plt.figure(figsize=(5 * ncols, 3 * nrows))
        gs = gridspec.GridSpec(nrows, ncols, figure=fig)

        for ind, unit in enumerate(units_focus):
            all_wf = cfg['all_ext'].get_unit_template(unit_id=unit, operator='average')
            gocue_wf = cfg['gocue_ext'].get_unit_template(unit_id=unit, operator='average')
            ptp_channel = np.argmax(np.ptp(all_wf, axis=0))

            ax = fig.add_subplot(gs[ind // ncols, ind % ncols])
            time_all = np.linspace(cfg['all_time'][0], cfg['all_time'][1], all_wf.shape[0])
            time_go_cue = np.linspace(cfg['gocue_time'][0], cfg['gocue_time'][1], gocue_wf.shape[0])
            ax.plot(time_all, all_wf[:, ptp_channel], c='k', label='all_mean_wf')
            ax.plot(time_go_cue, gocue_wf[:, ptp_channel], c='g', label='post_go_cue')
            ax.set_title(f'unit {unit}')
            if (ind // ncols) == (nrows - 1):
                ax.set_xlabel('ms')
            if ind == 0:
                ax.legend()

        plt.suptitle(f"Waveforms for {session} - {cfg['title']}", fontsize=16)
        plt.tight_layout()
        out_path = out_path_base.replace('.pdf', f'_{version}.pdf')
        plt.savefig(out_path)
        plt.close(fig)
        print(f'{session}: Saved {out_path}')





if __name__ == '__main__':
    data_type = 'curated'  # 'raw' or 'curated'
    session_df = pd.read_csv(CAPSULE_ROOT + '/code/data_management/session_assets.csv')
    session_list = session_df['session_id'].dropna().tolist()

    def process(session, data_type='curated'):
        session_dir = session_dirs(session)
        print(f'Processing session: {session}')
        if session_dir[f'curated_dir_{data_type}'] is not None:
            try:
                waveform_check(session, data_type=data_type, opto_only=True)
                # compare_mean_go_cue_waveforms(session, data_type=data_type, versions=['bandpass', 'raw'])
            except Exception as e:
                print(f'Failed to process session {session}: {e}')
            print(f'Finished session: {session}')
        else:
            print(f'No curated data for session: {session}, skipping.')

    # from joblib import Parallel, delayed
    Parallel(n_jobs=5)(delayed(process)(session, data_type=data_type) for session in session_list)
    # process('behavior_717121_2024-06-15_10-00-58')
    # for session in session_list:
    #     process(session)
