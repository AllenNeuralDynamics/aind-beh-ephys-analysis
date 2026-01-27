# %%
import sys
import os
import numpy as np
import json
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw
from utils.hdf5_extractor import read_hdf5
from utils.beh_functions import session_dirs, parseSessionID, get_unit_tbl
# import h5py 
import matplotlib.pyplot as plt
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename

# %%
def make_sorting_analyzer(session):
    session_dir = session_dirs(session)

    # %%
    rec_file = session_dir['raw_rec']
    nwb_file = session_dir['nwb_dir_raw']
    sorted_data_dir = session_dir['session_dir']
    # 'session_dir': '/root/capsule/data/behavior_ZS061_2021-03-28_16-35-51_raw_data/ecephys/neuralynx/session'

    # %%
    rec = read_hdf5(rec_file)

    # %%
    start_time = rec.get_start_time()
    end_time = rec.get_end_time()

    recording_raw_fake = spre.bandpass_filter(rec, freq_min=50, freq_max=8000)
    recording_raw_fake = spre.common_reference(recording_raw_fake)

    # %%
    sorting = se.read_nwb_sorting(
        nwb_file, sampling_frequency=rec.sampling_frequency, t_start=start_time
    )
    sorting = sorting.rename_units(sorting.get_property('unit_id'))
    unit_ids = sorting.unit_ids

    # gather all spike times
    all_spike_times = []
    all_labels = []
    for unit_id in unit_ids:
        file = os.path.join(sorted_data_dir, f"{unit_id}.txt")
        timestamps = list(np.loadtxt(file)/1000000)
        all_spike_times.extend(timestamps)
        all_labels.extend(list(np.full(len(timestamps), fill_value=unit_id)))


    # %%
    # convert to timestamps
    rec_timestamps = rec.get_times()
    all_spike_times = np.asarray(all_spike_times)

    idx = np.searchsorted(rec_timestamps, all_spike_times)
    idx[idx == len(rec_timestamps)] = len(rec_timestamps) - 1  # clip overflow
    # Correct for the fact that closest could be on the left or right
    left_idx = np.clip(idx - 1, 0, len(rec_timestamps)-1)
    # Pick whichever neighbor is closer
    choose_left = np.abs(all_spike_times - rec_timestamps[left_idx]) < np.abs(all_spike_times - rec_timestamps[idx])
    frame_ind = np.where(choose_left, left_idx, idx)

    # %%
    
    sparsity_params_file = '/root/capsule/code/beh_ephys_analysis/params.json'
    with open(sparsity_params_file, 'r') as f:
        postprocessing_params = json.load(f)
    job_kwargs = postprocessing_params.pop("job_kwargs")
    job_kwargs["n_jobs"] = 8
    si.set_global_job_kwargs(**job_kwargs)

    sparsity_params = postprocessing_params.pop("sparsity")
    quality_metrics_names = postprocessing_params.pop("quality_metrics_names")
    quality_metrics_params = postprocessing_params.pop("quality_metrics")

    analyzer_dict = postprocessing_params.copy()
    analyzer_dict.pop("duplicate_threshold")
    analyzer_dict.pop("return_scaled")

    # %%
    sorting = si.NumpySorting.from_samples_and_labels(
        samples_list= np.array(frame_ind),
        labels_list= np.array(all_labels),
        sampling_frequency=rec.sampling_frequency,
    )
    units = list(sorting.unit_ids)
    group = [int(unit_id[2])-1 for unit_id in units]
    sorting.set_property('group', group)
    # sorting_saved = sorting.save(folder=os.path.join(sorted_data_dir,"highres_timestamps_sorting"), overwrite=True)
    # # sorting_loaded = si.load("..")

    # %%
    si.set_global_job_kwargs(n_jobs=0.7, mp_context='spawn', progress_bar=True)

    # %%
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording_raw_fake,
        # radius_um=30,
        method="by_property",
        by_property="group",
        # sparsity=False
    )

    # %%
    _ = analyzer.compute(analyzer_dict)

    # %%
    _ = analyzer.compute(
        "quality_metrics",
        metric_names=quality_metrics_names,
        qm_params=quality_metrics_params
    )
    waveform_zarr_folder = f'{session_dir[f"ephys_dir_curated"]}/analyzer.zarr'
    if os.path.exists(waveform_zarr_folder):
        print("Zarr folder already exists, deleting.")
        shutil.rmtree(waveform_zarr_folder)
    # save
    analyzer_saved_zarr = analyzer.save_as(format='zarr', folder = waveform_zarr_folder)

# append raw waveform to opto tag summary
def append_raw_wf(session, clip_win=(-2,3)):
    session_dir = session_dirs(session)
    sorting_analyzer_file = os.path.join(session_dir['ephys_dir_curated'], 'analyzer.zarr')
    if os.path.exists(sorting_analyzer_file):
        analyzer = si.load(sorting_analyzer_file)
    else:
        print("Analyzer file not found!")
        return
    templates = analyzer.load_extension(extension_name="templates")
    unit_tbl = get_unit_tbl(session, data_type='curated')
    peak_waveform_raw_fake_aligned = []
    peak_raw_fake = []
    for unit_id in unit_tbl.unit_id.values:
        curr_template_raw = templates.get_unit_template(unit_id)
        peak_chan_id = np.argmax(np.ptp(curr_template_raw, axis=0))
        if curr_template_raw[96, peak_chan_id] < 0:
            sign = -1
        else:
            sign = 1
        curr_temp = np.full((clip_win[1]-clip_win[0])*32, np.nan)
        if sign == 1:
            peak_sample = np.argmax(curr_template_raw[:, peak_chan_id])
        else:
            peak_sample = np.argmin(curr_template_raw[:, peak_chan_id])
        peak_raw_fake.append(curr_template_raw[peak_sample, peak_chan_id])
        # pre_peak
        pre_samples = min(peak_sample, 32*np.abs(clip_win[0]))
        curr_temp[32*(np.abs(clip_win[0])) - pre_samples:32*(np.abs(clip_win[0]))] = curr_template_raw[peak_sample - pre_samples:peak_sample, peak_chan_id]
        # post_peak
        post_samples = min(curr_template_raw.shape[0]-peak_sample, 32*clip_win[1])
        curr_temp[32*(np.abs(clip_win[0])):32*(np.abs(clip_win[0])) + post_samples] = curr_template_raw[peak_sample:peak_sample + post_samples, peak_chan_id]

        peak_waveform_raw_fake_aligned.append(curr_temp)
    unit_tbl['peak_waveform_raw_fake_aligned'] = peak_waveform_raw_fake_aligned
    unit_tbl['peak_raw_fake'] = peak_raw_fake

    # save back to pkl
    file = os.path.join(session_dir['opto_dir_curated'], f'{session}_curated_soma_opto_tagging_summary_raw_wf.pkl')
    with open(file, 'wb') as f:
        import pickle
        pickle.dump(unit_tbl, f)
    print(f'{session} saved to {file}')

    # plot
    n_rows = int(np.ceil(len(unit_tbl)/4))
    fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(4*4,4*n_rows))
    raw_time = np.arange(clip_win[0], clip_win[1], 1/32)
    filtered_time = (np.arange(0, len(unit_tbl['peak_wf_aligned'].values[0])) - 100)/32
    for i, unit_id in enumerate(unit_tbl['unit_id'].values):
        ax = axes.flatten()[i]
        # template_raw
        curr_template_raw = unit_tbl.loc[unit_tbl['unit_id']==unit_id, 'peak_waveform_raw_fake_aligned'].values[0]
        ax.plot(raw_time, curr_template_raw, color='gray', label='raw wf')
        ax.axhline(y=0, color='black', linestyle='--')
        ax.axhline(unit_tbl.loc[unit_tbl['unit_id']==unit_id, 'peak_raw_fake'].values[0], color='red', linestyle='--')

        # template_filtered
        curr_template_filtered = unit_tbl.loc[unit_tbl['unit_id']==unit_id, 'peak_wf_aligned'].values[0]
        ax.plot(filtered_time, curr_template_filtered, color='blue', label='filtered wf')

        # template_denoised
        if i==0:
            ax.legend()
        ax.set_title(f'{unit_id}')
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir['ephys_dir_curated'], 'waveforms_re_filtered.pdf'))

if __name__ == "__main__":
    import pandas as pd
    session_list = pd.read_csv('/root/capsule/code/data_management/hopkins_session_assets.csv')
    session_list = session_list['session_id'].tolist()
    # make_sorting_analyzer('behavior_ZS059_2021-03-27_16-03-00')
    for session in session_list:
        session_dir = session_dirs(session)
        # waveform_zarr_folder = f'{session_dir[f"ephys_dir_curated"]}/analyzer.zarr'
        # # check is already processed
        # if os.path.exists(waveform_zarr_folder):
        #     print(f"Zarr folder already exists for session {session}, skipping.")
        #     continue
        # check if nwb units exist
        if session_dir['nwb_dir_raw'] is None:
            print(f"NWB file not found for session {session}, skipping.")
            continue
        else:
            nwb=load_nwb_from_filename(session_dir['nwb_dir_raw'])
            if nwb.units is None:
                print(f"No units found in NWB for session {session}, skipping.")
                continue
        # check if raw recording exists
        if session_dir['raw_rec'] is None:
            print(f"Raw recording file not found for session {session}, skipping.")
            continue

        print(f"Processing session {session}")
        try:
            # make_sorting_analyzer(session)
            append_raw_wf(session, clip_win=(-2,4))
        except Exception as e:
            print(f"Error processing session {session}: {e}")




