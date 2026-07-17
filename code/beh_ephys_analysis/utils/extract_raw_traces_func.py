import os, sys
from .capsule_migration import CAPSULE_ROOT, capsule_directories


def extract_trial_df(session, unit_id, roi_site):
    """Extract per-event raw-trace cutouts and spike times for an ROI's antidromic events.

    Uses the same loading pattern as the F_cross_correlation example: reads the
    raw HDF5 recording via ``read_hdf5(session_dir['raw_rec'])`` and applies
    ``spre.bandpass_filter(300–6000 Hz) + spre.common_reference`` (later
    spikeinterface API), then reads traces with ``get_traces(..., return_in_uV=True)``.
    Peak channel is chosen from ``unit_tbl['waveform_mean']`` (peak-to-peak amplitude
    per channel) rather than from a WaveformExtractor.
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import spikeinterface as si
    import spikeinterface.preprocessing as spre
    import utils.analysis_funcs as af
    from utils.beh_functions import session_dirs, get_unit_tbl

    data_type = 'curated'
    session_dir = session_dirs(session)
    subject_id = session.split('_')[1]

    # spike times
    root_folder = capsule_directories()['derived_dir']
    spike_times_folder = rf'{root_folder}/{subject_id}/{session}/ephys/{data_type}/processed'
    with open(os.path.join(spike_times_folder, 'spiketimes.pkl'), 'rb') as f:
        spiketimes = pickle.load(f)
    unit_spike_times = spiketimes[unit_id]
    print('unit_spike_times loaded')

    # antidromic event times for the ROI site
    event_csv = (
        str(capsule_directories()['derived_dir'])
        + f'/{subject_id}/{session}/ephys/opto/curated/{session}_opto_session.csv'
    )
    event_ids = pd.read_csv(event_csv)
    print('event_csv_file loaded')

    roi_rows = event_ids.query('emission_location == @roi_site')
    antidromic_stim_times = []
    for _, row in roi_rows.iterrows():
        for pulse_num in range(int(row.num_pulses)):
            time_shift = pulse_num * (row.duration + row.pulse_interval) / 1000
            antidromic_stim_times.append(row.time + time_shift)
    print('antidromic_stim_times retrieved')

    # load + preprocess raw recording (later-SI API)
    recording_zarr = session_dir['raw_rec']
    recording = si.read_zarr(recording_zarr)
    recording = recording.select_segments(session_dir['seg_id']-1)
    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_filtered = spre.common_reference(recording_filtered)
    rec_times = recording.get_times()
    fs = recording.get_sampling_frequency()
    start_time_session = recording.get_start_time()
    channel_ids = np.asarray(recording_filtered.channel_ids)

    # pick peak channel from the unit's waveform_mean (samples × channels)
    unit_tbl = get_unit_tbl(session, data_type=data_type)
    unit_row = unit_tbl[unit_tbl['unit_id'] == unit_id]
    if unit_row.empty:
        raise ValueError(f'{session}: unit {unit_id!r} not in curated unit table')
    unit_waveform = np.asarray(unit_row['waveform_mean'].values[0])
    peak_ind = int(np.argmax(np.max(unit_waveform, axis=0) - np.min(unit_waveform, axis=0)))
    best_channel = channel_ids[peak_ind]

    cutout_pre = 0.1
    cutout_post = 0.07
    n_pre = int(cutout_pre * fs)
    n_post = int(cutout_post * fs)

    trial_data = []
    int_event_locked_timestamps = []
    for i, evt in enumerate(antidromic_stim_times):
        evt_frame = np.searchsorted(rec_times, evt)
        start_frame = evt_frame - n_pre
        end_frame = evt_frame + n_post
        trace = recording_filtered.get_traces(
            segment_index=0,
            start_frame=start_frame,
            end_frame=end_frame,
            return_in_uV=True,
        )[:, peak_ind]
        aligned_times = (np.arange(end_frame - start_frame) - n_pre) / fs

        this_locked = af.event_locked_timestamps(
            unit_spike_times, [evt], np.array([-cutout_pre, cutout_post]), time_shift=0
        )
        int_event_locked_timestamps.append(this_locked[0])
        trial_data.append({
            'trial': i,
            'x': aligned_times,
            'y': trace,
            'unit_spike_times': this_locked[0],
            'channel': best_channel,
        })

    return pd.DataFrame(trial_data), int_event_locked_timestamps
