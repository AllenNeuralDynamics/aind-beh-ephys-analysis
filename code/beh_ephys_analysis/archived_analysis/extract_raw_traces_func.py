def extract_trial_df(session, unit_id, roi_site):
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import spikeinterface as si
    import spikeinterface.preprocessing as spre
    from utils.beh_functions import session_dirs
    import utils.analysis_funcs as af

    def load_and_preprocess_recording(session_folder, stream_name):
        ephys_path = os.path.dirname(session_folder)
        compressed_folder = os.path.join(ephys_path, 'ecephys_compressed')
        recording_zarr = [os.path.join(compressed_folder, f) for f in os.listdir(compressed_folder) if stream_name in f][0]
        recording = si.read_zarr(recording_zarr)
        recording_processed = spre.phase_shift(recording)
        recording_processed = spre.highpass_filter(recording_processed)
        recording_processed = spre.common_reference(recording_processed)
        
        return recording_processed

    data_type = 'curated'
    session_dir = session_dirs(session)
    filename = rf"{session}_unit_{unit_id}_{roi_site}"

    # spike times
    subject_id = session.split('_')[1]
    session_id = '_'.join(session.split('_')[1:3])

    root_folder = '/root/capsule/data/LC-NE_scratch_data_06-14-2025'
    spike_times_folder = rf'{root_folder}/{subject_id}/{session}/ephys/{data_type}/processed'
    with open(os.path.join(spike_times_folder, 'spiketimes.pkl'), 'rb') as f:
        spiketimes = pickle.load(f)
    unit_spike_times = spiketimes[unit_id]
    print("unit_spike_times loaded")

    event_csv = f'/root/capsule/data/LC-NE_scratch_data_06-14-2025/{subject_id}/{session}/ephys/opto/curated/{session}_opto_session.csv'
    event_ids = pd.read_csv(event_csv)
    print('event_csv_file loaded')

    roi_rows = event_ids.query("emission_location ==@roi_site")
    roi_event_times = []
    for i, row in roi_rows.iterrows():
        roi_event_time = row.time
        duration = row.duration
        num_pulses = row.num_pulses
        pulse_interval = row.pulse_interval
        for pulse_num in range(num_pulses):
            time_shift = pulse_num * (duration + pulse_interval) / 1000
            this_event_time = roi_event_time + time_shift
            roi_event_times.append(this_event_time)
    antidromic_stim_times = roi_event_times
    print('antidromic_stim_times retrieved')

    # filter good channels
    recording_processed = load_and_preprocess_recording(session_dir['session_dir'], 'ProbeA')
    we = si.load(session_dir[f'postprocessed_dir_{data_type}'], load_extensions=False)
    print('we loaded')
    # good_channel_ids = recording_processed.channel_ids[
    #     np.in1d(recording_processed.channel_ids, we.channel_ids)
    # ]
    extremum_channel_ids = si.get_template_extremum_channel(we)
    # recording_processed_good = recording_processed.select_channels(good_channel_ids)
    rec_times = recording_processed.get_times()


    cutout_pre = 0.1
    cutout_post = 0.07
    trial_data = []
    int_event_locked_timestamps = []

    # plt.figure()
    for i, evt in enumerate(antidromic_stim_times):
        evt_index = np.searchsorted(rec_times, evt)
        start_frame = evt_index - int(cutout_pre * we.sampling_frequency)
        end_frame = evt_index + int(cutout_post * we.sampling_frequency)
        target_unit = unit_id
        best_channel = extremum_channel_ids[target_unit]
        trace = recording_processed.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=[best_channel]
        )[:, 0]
        times_for_evt = rec_times[start_frame:end_frame]
        aligned_times = times_for_evt - evt
        # plt.plot(aligned_times, trace, alpha=0.5)
        this_time_range = np.array([-cutout_pre, cutout_post])
        this_locked = af.event_locked_timestamps(
            unit_spike_times, [evt], this_time_range, time_shift=0
        )
        int_event_locked_timestamps.append(this_locked[0])
        trial_data.append({
            "trial": i,
            "x": aligned_times,
            "y": trace,
            "unit_spike_times": this_locked[0],
        })

    trial_df = pd.DataFrame(trial_data)
    return trial_df, int_event_locked_timestamps
    