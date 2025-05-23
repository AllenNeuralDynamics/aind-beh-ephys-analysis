# %%
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import json
from harp.clock import decode_harp_clock, align_timestamps_to_anchor_points
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from utils.beh_functions import parseSessionID, session_dirs
from utils.plot_utils import shiftedColorMap, template_reorder
from utils.ephys_functions import load_drift
from open_ephys.analysis import Session##
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import load_nwb
from aind_ephys_utils import align
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from spikeinterface.core.sorting_tools import random_spikes_selection
import pickle
import datetime
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
from tqdm import tqdm
import shutil
from utils.beh_functions import get_unit_tbl
from joblib import Parallel, delayed

def load_and_preprocess_recording(session_folder, stream_name):
    si.set_global_job_kwargs(n_jobs=5, progress_bar=True)
    ephys_path = os.path.dirname(session_folder)
    compressed_folder = os.path.join(ephys_path, 'ecephys_compressed')
    recording_zarr = [os.path.join(compressed_folder, f) for f in os.listdir(compressed_folder) if stream_name in f and 'LFP' not in f][0]
    
    recording = si.read_zarr(recording_zarr)
    # preprocess
    recording_processed = spre.phase_shift(recording)
    recording_processed = spre.highpass_filter(recording_processed)
    # recording_processed = spre.bandpass_filter(recording_processed, freq_min=300, freq_max=6000)    
    recording_processed = spre.common_reference(recording_processed)
    return recording_processed
# %%
def opto_wf_preprocessing(session, data_type, target, load_sorting_analyzer = True):
    max_spikes_per_unit_spontaneous = 250
    # %%
    session_dir = session_dirs(session)
    output_file = os.path.join(session_dir['processed_dir'], f"{session}_process_record.txt")

    # Redirect stdout to the file
    if not os.path.exists(output_file):
        log_file = open(output_file, "w") 
    else: 
        log_file = open(output_file, "a")
    sys.stdout = log_file
    print('--------------------------Waveform Preprocessing--------------------------')
    print(f"Session: {session} processed at {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    # opto info
    opto_df = pd.read_csv(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_session_{target}.csv'), index_col=0)
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_info_{target}.json')) as f:
        opto_info = json.load(f)
    powers = opto_info['powers']
    sites = opto_info['sites']
    num_pulses = opto_info['num_pulses']
    duration = opto_info['durations']
    pulse_offset = np.array([1/freq for freq in opto_info['freqs']])
    total_duration = (pulse_offset) * num_pulses
    session_rec = Session(session_dir['session_dir'])
    recording = session_rec.recordnodes[0].recordings[0]
    timestamps = recording.continuous[0].timestamps
    laser_onset_samples = opto_df['laser_onset_samples'].values

    # %%
    if load_sorting_analyzer:
        waveform_zarr_folder = f'{session_dir[f"opto_dir_{data_type}"]}/opto_waveforms.zarr'
        if not os.path.exists(waveform_zarr_folder):
            print("Analyzer doesn't exist, computing first.")
            load_sorting_analyzer = False
    if not load_sorting_analyzer:
        # recording info
        sorting = si.load(session_dir[f'curated_dir_{data_type}'])
        spike_vector = sorting.to_spike_vector()
        unit_ids = sorting.unit_ids
        num_units = len(sorting.unit_ids)
        print(f"Total {len(sorting.unit_ids)} units")

        # %%
        # Spike indices
        spike_indices = spike_vector["sample_index"]

        response_window = opto_info['resp_win']+0.005 # to take care of potential late responses
        response_window_samples = int(response_window * sorting.sampling_frequency)

        num_cases = len(opto_info['powers']) * len(opto_info['sites']) * len(opto_info['durations'])

        unit_index_offset = num_units
        all_unit_ids = [f"{u} spont_pre" for u in unit_ids]
        all_unit_ids += [f"{u} spont_post" for u in unit_ids]

        all_spikes_in_responses = []
        spike_indices_removed = []
        conditions = ['powers', 'sites', 'durations', 'pre_post']

        for power_ind, curr_power in enumerate(opto_info['powers']):
            for site_ind, curr_site in enumerate(opto_info['sites']):  
                for curr_pre_post_ind, curr_pre_post in enumerate(opto_info['pre_post']):                                                           
                    for duration_ind, curr_duration in enumerate(opto_info['durations']):
                        spikes_in_response = []
                        onset_offset_indices = []
                        if len(opto_df.query('site == @curr_site and power == @curr_power and duration == @curr_duration and pre_post == @curr_pre_post')) > 0:
                            for freq_ind, curr_freq in enumerate(opto_info['freqs']):
                                for num_pulse_ind, curr_num_pulses in enumerate(opto_info['num_pulses']):
                                    onset_samples = opto_df.query(
                                        'site == @curr_site and power == @curr_power and duration == @curr_duration and freq == @curr_freq and num_pulses == @curr_num_pulses and pre_post == @curr_pre_post'
                                        )['laser_onset_samples'].values
                                    if len(onset_samples) == 0:
                                        continue
                                    pulse_offset = 1/curr_freq
                                    pulse_offset_samples = int(pulse_offset * sorting.sampling_frequency)
                                    for pulse_ind in range(curr_num_pulses):
                                        for onset_sample in onset_samples:
                                            # response window
                                            onset_response = onset_sample + pulse_ind * pulse_offset_samples
                                            onset_offset_indices.append(onset_response)
                                            offset_response = onset_response + response_window_samples
                                            onset_offset_indices.append(offset_response)
                                    start_stop_indices = np.searchsorted(spike_indices, np.array(onset_offset_indices))
                                    for i, (start, stop) in enumerate(zip(start_stop_indices[::2], start_stop_indices[1::2])):
                                        sv = spike_vector[start:stop]
                                        if len(sv) > 0:
                                            spike_indices_removed.append(np.arange(start, stop))
                                        sv_copy = sv.copy()
                                        sv_copy["unit_index"] = sv_copy["unit_index"] + unit_index_offset + num_units
                                        spikes_in_response.append(sv_copy)
                                        # num_cases += 1
                            
                            spikes_in_response = np.concatenate(spikes_in_response)
                            print('# of responding units: ', len(np.unique(spikes_in_response['unit_index'])))
                            unit_index_offset += num_units
                            all_spikes_in_responses.append(spikes_in_response)
                            new_unit_ids = [f"{u} emission_location:{curr_site}, power:{curr_power}, duration:{curr_duration}, freq:{curr_freq}, pre_post:{curr_pre_post}" for u in unit_ids]
                            print(f"emission_location:{curr_site}, power:{curr_power}, duration:{curr_duration}, freq:{curr_freq}, pre_post:{curr_pre_post}")
                            all_unit_ids += new_unit_ids
                                    
        all_spikes_in_responses = np.concatenate(all_spikes_in_responses)
        spike_indices_removed = np.concatenate(spike_indices_removed)
        all_spikes_not_in_responses = np.delete(spike_vector, spike_indices_removed)
        all_spikes_not_in_responses_pre = all_spikes_not_in_responses[all_spikes_not_in_responses["sample_index"] < (opto_df.query('pre_post == "pre"')['laser_onset_samples'].max() + 5 * 60 * sorting.sampling_frequency)]
        all_spikes_not_in_responses_post = all_spikes_not_in_responses[all_spikes_not_in_responses["sample_index"] > (opto_df.query('pre_post == "post"')['laser_onset_samples'].min() - 5 * 60 * sorting.sampling_frequency)]
        # select random spontaneous spikes for each unit divided by first and second half of the session
        sorting_no_responses_pre = si.NumpySorting(
            all_spikes_not_in_responses_pre, 
            unit_ids=[f'{unit_id}_pre' for unit_id in unit_ids], 
            sampling_frequency=sorting.sampling_frequency
        )
        random_spike_indices = random_spikes_selection(sorting_no_responses_pre, method="uniform",
                                                    max_spikes_per_unit=max_spikes_per_unit_spontaneous)
        selected_spikes_no_responses_pre = all_spikes_not_in_responses_pre[random_spike_indices]



        sorting_no_responses_post = si.NumpySorting(
            all_spikes_not_in_responses_post, 
            unit_ids=[f'{unit_id}_post' for unit_id in unit_ids], 
            sampling_frequency=sorting.sampling_frequency
        )
        random_spike_indices = random_spikes_selection(sorting_no_responses_post, method="uniform",
                                                    max_spikes_per_unit=max_spikes_per_unit_spontaneous)
        selected_spikes_no_responses_post = all_spikes_not_in_responses_post[random_spike_indices]
        selected_spikes_no_responses_post_copy = selected_spikes_no_responses_post.copy()
        selected_spikes_no_responses_post_copy["unit_index"] = selected_spikes_no_responses_post["unit_index"] + num_units



        all_spikes = np.concatenate([selected_spikes_no_responses_pre, selected_spikes_no_responses_post_copy, all_spikes_in_responses])

        # sort by segment+index
        all_spikes = all_spikes[
            np.lexsort((all_spikes["sample_index"], all_spikes["segment_index"]))
        ]

        sorting_all = si.NumpySorting(
            all_spikes, 
            unit_ids=all_unit_ids, 
            sampling_frequency=sorting.sampling_frequency
        )

        # %%
        print("original:", len(spike_vector))
        print("sampled:", len(sorting_all.to_spike_vector()))

        # %%
        # filter good channels
        recording_processed = load_and_preprocess_recording(session_dir['session_dir'], 'ProbeA')
        we = si.load(session_dir[f'postprocessed_dir_{data_type}'], load_extensions=False)
        good_channel_ids = recording_processed.channel_ids[
            np.in1d(recording_processed.channel_ids, we.channel_ids)
        ]

        recording_processed_good = recording_processed.select_channels(good_channel_ids)
        print(f"Num good channels: {recording_processed_good.get_num_channels()}")

        # %%
        num_cases = len(sorting_all.unit_ids) // num_units - 2
        sparsity_mask_all = np.tile(we.sparsity.mask, (num_cases + 2, 1))
        del we
        sparsity_all = si.ChannelSparsity(
            sparsity_mask_all,
            unit_ids=sorting_all.unit_ids,
            channel_ids=recording_processed_good.channel_ids
        )

        # %%
        # create analyzer
        analyzer_all = si.create_sorting_analyzer(
            # sorting_all.select_units(ROI_unit_ids),
            sorting_all,
            recording_processed_good,
            sparsity=sparsity_all
        )

        # %%
        min_spikes_per_unit = 5
        keep_unit_ids = []
        count_spikes = sorting_all.count_num_spikes_per_unit()
        for unit_id, count in count_spikes.items():
            if count >= min_spikes_per_unit:
                keep_unit_ids.append(unit_id)

        analyzer = analyzer_all.select_units(keep_unit_ids)
        print(f"Number of units with at least {min_spikes_per_unit} spikes: {len(analyzer.unit_ids)}")

        # %%
        _ = analyzer.compute("random_spikes", method="all")
        _ = analyzer.compute(["waveforms", "templates"])
        waveform_zarr_folder = f'{session_dir[f"opto_dir_{data_type}"]}/opto_waveforms.zarr'
        if os.path.exists(waveform_zarr_folder):
            print("Zarr folder already exists, deleting.")
            shutil.rmtree(waveform_zarr_folder)
        analyzer_saved_zarr = analyzer.save_as(format='zarr', folder = waveform_zarr_folder)
        print(f'Saved analyzer to {waveform_zarr_folder}')
    else:
        waveform_zarr_folder = f'{session_dir[f"opto_dir_{data_type}"]}/opto_waveforms.zarr'
        if os.path.exists(waveform_zarr_folder):
            analyzer = si.load(waveform_zarr_folder, load_extensions=False)
            print(f'Loaded analyzer from {waveform_zarr_folder}')
        else:
            print("Analyzer doesn't exist, please compute first.")
            return None

    # %%
    conditions = ['power', 'site', 'duration', 'pre_post']
    columns = ['unit_id'] + conditions + ['template', 'peak_channel', 'peak_waveform', 'spont', 'count']
    waveform_metrics = pd.DataFrame(columns=columns)

    template_ext = analyzer.get_extension("templates")
    extreme_channel_indices = si.get_template_extremum_channel(analyzer, mode = "at_index", outputs = "index")
    extreme_channels = si.get_template_extremum_channel(analyzer) 

    for ind_id, unit_id in enumerate(analyzer.unit_ids):
        unit_id_name, case = unit_id.split(" ", 1)
        
        unit_template = template_ext.get_unit_template(unit_id)       
        peak_waveform = unit_template[:,extreme_channel_indices[unit_id]]
        # plt.figure()
        # plt.plot(list(unit_waveform[:,extreme_channel_indices["14 spont"]]))
        # plt.show()

        new_row = dict(unit_id=unit_id_name, template=unit_template, peak_channel=extreme_channels[unit_id], peak_waveform=peak_waveform, count=len(analyzer.sorting.get_unit_spike_train(unit_id)))
        if "spont" in case:
            pre_post = case.split("_")[1]                
            new_row["spont"] = 1
            new_row["pre_post"] = pre_post
            new_row["power"] = np.nan
            new_row["site"] = np.nan
            new_row["duration"] = np.nan

        else:
            split_strs = case.split(",")
            site_str, power_str, duration_str, freq_str, pre_post_str = split_strs

            site = site_str.split(":")[1]
            power = power_str.split(":")[1]
            duration = duration_str.split(":")[1]
            pre_post = pre_post_str.split(":")[1]
            # site = site_str.split(":")[1]
            new_row["spont"] = 0
            new_row["pre_post"] = pre_post
            new_row["power"] = power
            new_row["site"] = site
            new_row["duration"] = duration
            # new_row["site"] = power
        waveform_metrics = pd.concat([waveform_metrics, pd.DataFrame([new_row])], ignore_index=True)


    # %%
    # calculate correlation and Euclidean distance, normalized by power
    waveform_metrics['unit_id'].unique()
    waveform_metrics['euclidean_norm'] = np.nan
    waveform_metrics['correlation'] = np.nan
    sparsity = analyzer.sparsity
    all_channels = sparsity.channel_ids
    # loop through all rows in the dataframe
    for index, row in waveform_metrics.iterrows():
        # get the template and peak waveform for the current row
        if row['spont'] == 1:
            continue
        else:
            # 
            template = row['template']
            peak_channel = row['peak_channel']
            unit_id = row['unit_id']
            if unit_id == '62':
                print(unit_id)
            pre_post = row['pre_post']
            
            spont_unit = waveform_metrics.query("spont == 1 and unit_id == @unit_id and pre_post == @pre_post")
            if len(spont_unit) == 0:
                template_spont = np.nan
                waveform_spont = np.nan
                channel_spont = np.nan
            else:
                template_spont = spont_unit['template'].values[0]
                peak_waveform_spont = spont_unit['peak_waveform'].values[0]
                peak_channel_spont = spont_unit['peak_channel'].values[0]
            
            peak_channel_ind = np.argmin(np.min(template_spont, 0))
            peak_waveform_resp = template[:, peak_channel_ind]
            waveform_metrics.at[index, 'peak_waveform'] = peak_waveform_resp
            peak_samp_ind = np.argmin(peak_waveform_spont)

            # correlation 
            correlation = np.corrcoef(peak_waveform_resp.reshape(-1), peak_waveform_spont.reshape(-1))[0, 1]
            waveform_metrics.at[index, 'correlation'] = correlation
            # euclidean distance
            focus_ind = np.array(range(np.max(np.array([peak_samp_ind-20, 0])), np.min(np.array([peak_samp_ind+40, len(peak_waveform_spont)]))))

            euc_dist = np.linalg.norm(
                peak_waveform_resp[focus_ind]
                - peak_waveform_spont[focus_ind])
            # energy
            energy = np.linalg.norm(peak_waveform_spont[focus_ind])
            euc_dist_norm = euc_dist / energy
            waveform_metrics.at[index, 'euclidean_norm'] = euc_dist_norm

    with open(f"{session_dir[f'opto_dir_{data_type}']}/{session}_opto_waveform_metrics.pkl", 'wb') as f:
        pickle.dump(waveform_metrics, f)
    print(f"Saved waveform metrics to {session_dir[f'opto_dir_{data_type}']}/{session}_opto_waveform_metrics.pkl")


    # %%
    unit_ids = waveform_metrics['unit_id'].unique()
    gs = gridspec.GridSpec(40, 10)
    fig = plt.figure(figsize=(20, 50))
    count = 0
    for unit_id in unit_ids:
        spont_post = waveform_metrics.query('spont == 1 and unit_id == @unit_id and pre_post == "post"')
        spont_pre = waveform_metrics.query('spont == 1 and unit_id == @unit_id and pre_post == "pre"')
        if len(spont_post) > 0:
            wf_spont = spont_post['peak_waveform'].values[0]
        elif len(spont_pre) > 0:
            wf_spont = spont_pre['peak_waveform'].values[0]
        else:
            continue

        resp = waveform_metrics.query('spont == 0 and unit_id == @unit_id and pre_post == "post"')
        if len(resp) > 0:
            wf_resp = resp['peak_waveform'].values
            wf_powers = resp['power'].values
            wf_powers = [float(wf_power) for wf_power in wf_powers]
            ax = fig.add_subplot(gs[count])
            ax.plot(wf_spont, color='k', linewidth=2, alpha = 0.5)
            for wf, power in zip(wf_resp, wf_powers):
                ax.plot(wf, color=colormaps['hot'](power/max(wf_powers)), linewidth=1)
            ax.set_frame_on(False)
            count = count + 1
            ax.set_title(unit_id)
        else:
            continue

    plt.tight_layout()
    fig.savefig(fname=f"{session_dir[f'opto_dir_{data_type}']}/{session}_opto_waveforms.pdf")
    print(f"Output saved to {output_file}")
    sys.stdout = sys.__stdout__
    # Close the file
    log_file.close()

def waveform_recompute_session(session, data_type, load_sorting_analyzer=True, opto_only=True, plot=True, save=True):
    session_dir = session_dirs(session)
    bin_size = 10*60*30000 # 10 minutes,in samples
    max_spikes_per_unit_spontaneous = 500
    waveform_zarr_folder = f'{session_dir[f"ephys_dir_{data_type}"]}/waveforms_session.zarr'
    if load_sorting_analyzer:
        if not os.path.exists(waveform_zarr_folder):
            print("Analyzer doesn't exist, computing first.")
            load_sorting_analyzer = False
    
    if not load_sorting_analyzer:
        shutil.rmtree(waveform_zarr_folder, ignore_errors=True)
        # recording info
        sorting = si.load(session_dir[f'curated_dir_{data_type}'])
        sorting_analyzer = si.load(session_dir[f'postprocessed_dir_{data_type}'], load_extensions=False)
        # amplitudes = sorting_analyzer.get_extension('spike_amplitudes').get_data(outputs="by_unit")[0]
        if opto_only:
            unit_tbl = get_unit_tbl(session, data_type)
            unit_ids = list(unit_tbl.query('opto_pass == 1')['unit_id'].values)
            unit_ids = [int(unit_id) for unit_id in unit_ids]
            if len(unit_ids) == 0:
                print(f'No opto units in session {session}')
                all_tagged_units = list(unit_tbl.query('opto_pass == True and default_qc == True')['unit_id'].values)
                all_tagged_units = [int(unit_id) for unit_id in all_tagged_units]
                waveform_recompute = pd.DataFrame({'unit_id': all_tagged_units, 'amplitude_opt': None, 'peak_wf_opt': None, 'mat_wf_opt': None})
                if save: 
                    if 'peak_wf_opt' in unit_tbl.columns:
                        unit_tbl = unit_tbl.drop(waveform_recompute.columns.tolist(), axis=1) 
                    unit_tbl['unit_id']= unit_tbl['ks_unit_id']
                    unit_tbl = unit_tbl.merge(waveform_recompute, on='unit_id', how='left')
                    unit_tbl_file = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_{data_type}_soma_opto_tagging_summary.pkl')
                    with open(unit_tbl_file, 'wb') as f:
                        pickle.dump(unit_tbl, f)
                return waveform_recompute
            sorting = sorting.select_units(unit_ids)
            sorting_analyzer = sorting_analyzer.select_units(unit_ids)                
        spike_vector = sorting.to_spike_vector()
        unit_ids = sorting.unit_ids
        num_units = len(sorting.unit_ids)
        # %%
        # Spike indices
        spike_indices = spike_vector["sample_index"]
        # unit_ids
        spike_unit_ids = spike_vector["unit_index"]
        # session_length
        session_length = spike_indices[-1]
        # devide session into 5 equal parts
        edges = np.arange(0, session_length+bin_size, bin_size)
        new_unit_ids = []
        new_vector = []
        for bin_ind in range(len(edges)-1):
            start = int(edges[bin_ind])
            end = int(edges[bin_ind+1])
            indices = np.where((spike_indices > start) & (spike_indices <= end))[0]
            curr_vector = spike_vector[indices].copy()
            curr_vector_copy = curr_vector.copy()
            curr_vector_copy['unit_index'] = curr_vector_copy['unit_index'] + bin_ind*num_units
            curr_unit_ids = [f'{unit_id}-{bin_ind}' for unit_id in unit_ids]
            new_unit_ids += curr_unit_ids
            new_vector.append(curr_vector_copy)
        spikes_binned = np.concatenate(new_vector)
        # %% make new sorting
        sorting_binned = si.NumpySorting(spikes_binned, 
                                    unit_ids=new_unit_ids,
                                    sampling_frequency=30000)


        # %%
        random_spike_indices = random_spikes_selection(sorting_binned, method="uniform",
                                                    max_spikes_per_unit=max_spikes_per_unit_spontaneous)
        selected_spikes_binned = spikes_binned[random_spike_indices]
        sorting_binned_selected = si.NumpySorting(selected_spikes_binned, 
                                    unit_ids=new_unit_ids,
                                    sampling_frequency=30000)

        # %%
        recording_processed = load_and_preprocess_recording(session_dir['session_dir'], 'ProbeA')
        good_channel_ids = recording_processed.channel_ids[
            np.in1d(recording_processed.channel_ids, sorting_analyzer.channel_ids)
        ]

        recording_processed_good = recording_processed.select_channels(good_channel_ids)
        print(f"Num good channels: {recording_processed_good.get_num_channels()}")

        # %%
        sparsity_mask_all = np.tile(sorting_analyzer.sparsity.mask, (len(edges)-1, 1))
        sparsity_all = si.ChannelSparsity(
            sparsity_mask_all,
            unit_ids=sorting_binned_selected.unit_ids,
            channel_ids=recording_processed_good.channel_ids
        )
        

        # %%
        analyzer_binned = si.create_sorting_analyzer(
            # sorting_all.select_units(ROI_unit_ids),
            sorting_binned_selected,
            recording_processed_good,
            sparsity=sparsity_all
        )

        # %%
        min_spikes_per_unit = 50
        keep_unit_ids = []
        count_spikes = sorting_binned_selected.count_num_spikes_per_unit()
        for unit_id, count in count_spikes.items():
            if count >= min_spikes_per_unit:
                keep_unit_ids.append(unit_id)
        analyzer_binned = analyzer_binned.select_units(keep_unit_ids)

        # %%
        _ = analyzer_binned.compute("random_spikes", method="all", max_spikes_per_unit=max_spikes_per_unit_spontaneous)
        _ = analyzer_binned.compute("templates", ms_before=1.5, ms_after=2.5)
        analyzer_binned.save_as(format='zarr', folder = waveform_zarr_folder)
        print(f'Saved as {waveform_zarr_folder}')
    else:
        analyzer_binned = si.load(waveform_zarr_folder, load_extensions=True)
        print(f'Loaded {waveform_zarr_folder}')


    # take re-computed waveforms and pick templates
    analyzer_binned = si.load(waveform_zarr_folder, load_extensions=True)
    analyzer = si.load(session_dir[f'postprocessed_dir_{data_type}'], load_extensions=False)

    unit_ids = analyzer.sorting.get_unit_ids()
    unit_ids_binned = analyzer_binned.sorting.get_unit_ids()
    bin_inds = list(set([int(curr_id.split('-')[-1]) for curr_id in unit_ids_binned])) 
    all_channels = analyzer.sparsity.channel_ids
    if all_channels[0].startswith('AP'):
        all_channels_int = np.array([int(channel.split('AP')[-1]) for channel in all_channels])
    else:
        all_channels_int = np.array([int(channel.split('CH')[-1]) for channel in all_channels])
    unit_spartsiity = analyzer.sparsity.unit_id_to_channel_ids
    channel_locations = analyzer.get_channel_locations()
    unit_locations = analyzer.get_extension("unit_locations").get_data(outputs="by_unit")
    right_left = channel_locations[:, 0]<20 

    # %%
    qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_file) as f:
        qm_dict = json.load(f)
    start = qm_dict['ephys_cut'][0]
    end = qm_dict['ephys_cut'][1]

    # %%
    # re-organize templates so that left and right separate
    colors = ["blue", "white", "red"]
    b_w_r_cmap = LinearSegmentedColormap.from_list("b_w_r", colors)
    # load wfs
    waveform_info_file = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_waveform_params.json')
    with open(waveform_info_file) as f:
        waveform_info = json.load(f)
    y_neighbors_to_keep = waveform_info['y_neighbors_to_keep']
    samples_to_keep = waveform_info['samples_to_keep']
    orginal_loc = True
    
    channel_loc_dict = {channel: channel_loc for channel, channel_loc in zip(all_channels_int, channel_locations)}

    wf_ext = analyzer_binned.get_extension("waveforms")
    temp_ext = analyzer_binned.get_extension("templates")
    temp_ext_mean = analyzer.get_extension("templates")

    extreme_channel_indices_binned = si.get_template_extremum_channel(analyzer_binned, mode = "at_index", outputs = "index")
    extreme_channels_binned = si.get_template_extremum_channel(analyzer_binned) 

    extreme_channel_indices = si.get_template_extremum_channel(analyzer, mode = "at_index", outputs = "index")
    extreme_channels = si.get_template_extremum_channel(analyzer) 

    # recompute extreme channel binned so that they are from the same column
    units_in_binned = [unit_id.split('-')[0] for unit_id in unit_ids_binned]
    for unit_id in unit_ids:
        curr_peak_ori = extreme_channel_indices[unit_id]
        for curr_bin_ind in bin_inds:
            curr_unit_id = f'{unit_id}-{curr_bin_ind}'
            if curr_unit_id in unit_ids_binned:
                curr_peak_opt = extreme_channel_indices_binned[curr_unit_id]
                if right_left[curr_peak_ori] != right_left[curr_peak_opt]:
                    # recompute if changed column
                    channel_inds_same_col = np.where(right_left == right_left[curr_peak_ori])[0]
                    temp_same = temp_ext.get_unit_template(curr_unit_id, operator='average')[:, channel_inds_same_col]
                    peak_ind = np.argmax(np.abs(temp_same[30, :]))
                    extreme_channel_indices_binned[curr_unit_id] = channel_inds_same_col[peak_ind]

    # %% get unit templates and peak wf for each bin
    unit_tbl = get_unit_tbl(session, data_type)
    all_wfs = []
    all_amps_ori = []
    all_wfs_ori = []
    all_peak_wf = []
    all_peak_wf_ori = []
    all_counts = np.zeros((len(unit_ids), len(bin_inds)))
    presence_score = np.ones((len(unit_ids), len(bin_inds)))
    euc_dis = np.zeros((len(unit_ids), len(bin_inds)))
    all_units = []
    if opto_only:
        all_tagged_units = list(unit_tbl.query('opto_pass == True and default_qc == True')['unit_id'].values)
        all_tagged_units = [int(unit_id) for unit_id in all_tagged_units]
    else:
        all_tagged_units = list(unit_tbl.query('default_qc == True')['unit_id'].values)
        all_tagged_units = [int(unit_id) for unit_id in all_tagged_units]

    for unit_ind, unit_id in enumerate(all_tagged_units):
        curr_wfs_bins = []
        curr_wfs_bins_ori = []
        curr_peak_bins = []
        curr_peak_bins_ori = []
        drift = load_drift(session, unit_id, data_type=data_type)
        temp_mean_curr = temp_ext_mean.get_unit_template(unit_id, operator='average') 
        peak_mean_curr = temp_mean_curr[:,extreme_channel_indices[unit_id]]
        amp_curr = peak_mean_curr[90]
        for curr_ind in bin_inds:
            curr_unit_id = f'{unit_id}-{curr_ind}'
            curr_time_bin = [curr_ind * 10 * 60 + start, (curr_ind + 1) * 10 * 60 + start]
            if drift is not None:
                if drift['ephys_cut'] is not None:
                    if drift['ephys_cut'][0] is not None:
                        if drift['ephys_cut'][0] > curr_time_bin[0] + 5 * 60:
                            presence_score[unit_ind, curr_ind] = 0
                    if drift['ephys_cut'][1] is not None:
                        if drift['ephys_cut'][1] < curr_time_bin[1] - 5 * 60:
                            presence_score[unit_ind, curr_ind] = 0
            if curr_unit_id in unit_ids_binned:
                curr_temp = temp_ext.get_unit_template(curr_unit_id, operator='average')
                # curr_std  = temp_ext.get_unit_template(curr_unit_id, operator='std')
                curr_wfs = wf_ext.get_waveforms_one_unit(curr_unit_id)
                all_counts[unit_ind, curr_ind] = len(curr_wfs)
                reordered_template = template_reorder(curr_temp, right_left, all_channels_int, 
                                                    sample_to_keep = samples_to_keep, y_neighbors_to_keep = y_neighbors_to_keep, orginal_loc = True, 
                                                    peak_ind=extreme_channel_indices_binned[curr_unit_id])
                curr_wfs_bins.append(reordered_template)
                orginal_template = template_reorder(curr_temp, right_left, all_channels_int, 
                                                    sample_to_keep = samples_to_keep, y_neighbors_to_keep = y_neighbors_to_keep, orginal_loc = True, 
                                                    peak_ind=extreme_channel_indices[unit_id])
                # curr_peak_wf_curr_bin = curr_temp[:, all_channels_int[extreme_channel_indices_binned[curr_unit_id]]]
                # curr_peak_wf_curr_ori = curr_temp[:, all_channels_int[extreme_channel_indices[unit_id]]]
                curr_peak_wf_curr_bin = curr_temp[:, extreme_channel_indices_binned[curr_unit_id]]
                curr_peak_wf_curr_ori = curr_temp[:, extreme_channel_indices[unit_id]]
                # euclidean distance
                euc_dist = np.linalg.norm(
                    curr_peak_wf_curr_bin
                    - peak_mean_curr[60:150])
                # energy
                energy = np.linalg.norm(peak_mean_curr[60:150])
                euc_dist_norm = euc_dist / energy
                euc_dis[unit_ind, curr_ind] = euc_dist_norm

                if euc_dist_norm>1:
                    curr_peak_wf_curr_bin = peak_mean_curr[60:150]
                

                curr_wfs_bins_ori.append(orginal_template)
                curr_peak_bins.append(curr_peak_wf_curr_bin)
                curr_peak_bins_ori.append(curr_peak_wf_curr_ori)
            else:
                curr_wfs_bins.append(np.array([]))
                curr_wfs_bins_ori.append(np.array([]))
                curr_peak_bins.append(np.array([]))
                curr_peak_bins_ori.append(np.array([]))

        non_empty_shape = next(arr.shape for arr in curr_wfs_bins if arr.size > 0)
        curr_wfs_bins = [arr if arr.size > 0 else np.full(non_empty_shape, np.nan) for arr in curr_wfs_bins]
        curr_wfs_bins_ori = [arr if arr.size > 0 else np.full(non_empty_shape, np.nan) for arr in curr_wfs_bins_ori]

        non_empty_shape = next(arr.shape for arr in curr_peak_bins if arr.size > 0)
        curr_peak_bins = [arr if arr.size > 0 else np.full(non_empty_shape, np.nan) for arr in curr_peak_bins]
        curr_peak_bins_ori = [arr if arr.size > 0 else np.full(non_empty_shape, np.nan) for arr in curr_peak_bins_ori]

        all_wfs.append(curr_wfs_bins)  
        all_wfs_ori.append(curr_wfs_bins_ori)
        all_peak_wf.append(curr_peak_bins)
        all_peak_wf_ori.append(curr_peak_bins_ori)
        all_amps_ori.append(amp_curr)


    # %% pick the best bin to get representative wf
    all_peak_wf_opt = []
    all_mat_wfs_opt = []
    all_amp_opt = []
    for unit_ind, unit in enumerate(all_tagged_units):
        mat_wfs = all_wfs[unit_ind]
        peak_wfs = all_peak_wf[unit_ind]
        presence = presence_score[unit_ind]
        count = all_counts[unit_ind]
        # find both presence and count
        if any(p and c > 100 for p, c in zip(presence, count)):
            val_bins = np.where((presence > 0) & (count > 0))[0]
            curr_peaks = np.array([peak_wf[30] if peak_wf.size > 0 else 0 for peak_wf in peak_wfs])
            max_ind = np.argmax(np.abs(curr_peaks[val_bins]))
            unit_amp = curr_peaks[val_bins[max_ind]]
            unit_peak = peak_wfs[val_bins[max_ind]]
            unit_peak_mat = mat_wfs[val_bins[max_ind]]
        elif any(c > 100 for c in count):
            val_bins = np.where(count > 0)[0]
            curr_peaks = np.array([peak_wf[30] for peak_wf in peak_wfs if peak_wf.size > 0])
            max_ind = np.argmax(np.abs(curr_peaks[val_bins]))
            unit_amp = curr_peaks[val_bins[max_ind]]
            unit_peak = peak_wfs[val_bins[max_ind]]
            unit_peak_mat = mat_wfs[val_bins[max_ind]]
        else:
            unit_amp = 0
            unit_peak = np.zeros(len(all_channels_int))
            unit_peak_mat = np.zeros((len(all_channels_int), 2 * y_neighbors_to_keep + 1)) 
        all_amp_opt.append(unit_amp)
        all_peak_wf_opt.append(unit_peak)
        all_mat_wfs_opt.append(unit_peak_mat)
    
    peak_ind = [np.argmin(wf) for wf in all_peak_wf_opt]
    all_peak_wf_opt_aligned = [
        np.concatenate((
            np.full(max(30 - peak_ind_curr, 0), np.nan),
            temp[max(peak_ind_curr - 30, 0) : min(peak_ind_curr + 60, temp.shape[0])],
            np.full(max((peak_ind_curr + 60) - temp.shape[0], 0), np.nan)
        ))
        for temp, peak_ind_curr in zip(all_peak_wf_opt, peak_ind)
    ]

    waveform_recompute = pd.DataFrame({'unit_id': all_tagged_units, 'amplitude_opt': all_amp_opt, 'peak_wf_opt': all_peak_wf_opt, 'mat_wf_opt': all_mat_wfs_opt, 'peak_wf_opt_aligned': all_peak_wf_opt_aligned})
    if save: 
        if 'peak_wf_opt' in unit_tbl.columns:
            # find common columns
            common_cols = list(set(unit_tbl.columns) & set(waveform_recompute.columns))
            unit_tbl = unit_tbl.drop(common_cols, axis=1) 
        unit_tbl['unit_id']= unit_tbl['ks_unit_id']
        unit_tbl = unit_tbl.merge(waveform_recompute, on='unit_id', how='left')
        unit_tbl_file = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_{data_type}_soma_opto_tagging_summary.pkl')
        with open(unit_tbl_file, 'wb') as f:
            pickle.dump(unit_tbl, f)


    # %%
    if plot: 
        fig = plt.figure(figsize=(20, 4*len(all_tagged_units)))
        gs = gridspec.GridSpec(2*len(all_tagged_units)+2, len(bin_inds)+1, figure=fig)
        for unit_ind, unit_id in enumerate(all_tagged_units):
            if unit_ind == 379:
                print('stop')
            ax_wf_ori = fig.add_subplot(gs[unit_ind*2, len(bin_inds)])
            ax_wf_reo = fig.add_subplot(gs[unit_ind*2+1, len(bin_inds)])
            temp_mean_curr = temp_ext_mean.get_unit_template(unit_id, operator='average') 
            peak_mean_curr = temp_mean_curr[:,extreme_channel_indices[unit_id]]
            ax_wf_ori.plot(peak_mean_curr[60:150], color='r', alpha=0.5, linewidth=1)  
            ax_wf_reo.plot(all_peak_wf_opt[unit_ind], color='r', alpha=0.5, linewidth=1)

            all_temps_bins = all_wfs[unit_ind]
            all_temps_bins_ori = all_wfs_ori[unit_ind]  
            all_peak_bins = all_peak_wf[unit_ind]
            all_peak_bins_ori = all_peak_wf_ori[unit_ind]  
            # find min and mac of all templates
            min_reo = np.nanmin(np.array(all_temps_bins))
            max_reo = np.nanmax(np.array(all_temps_bins))
            min_ori = np.nanmin(np.array(all_temps_bins_ori))
            max_ori = np.nanmax(np.array(all_temps_bins_ori))
            min_all = min(min_reo, min_ori)
            max_all = max(max_reo, max_ori)
            if max_all == 0:
                max_all = np.nanmax(peak_mean_curr)
            if min_all == 0:
                min_all = np.nanmin(peak_mean_curr)    
            shifted_cmap = shiftedColorMap(b_w_r_cmap, min_all, max_all, 'shifted_b_w_r')
            for curr_ind in bin_inds:
                # original template
                ax = fig.add_subplot(gs[unit_ind*2, curr_ind])
                curr_temp = all_temps_bins_ori[curr_ind]
                im = ax.imshow(curr_temp, extent=[samples_to_keep[0], 
                                                samples_to_keep[0] + 2 * (samples_to_keep[1] - samples_to_keep[0]), 
                                                2 * y_neighbors_to_keep + 1, 0], 
                                                cmap=shifted_cmap, interpolation='nearest',
                                                vmin=min_all, vmax=max_all, aspect='auto')
                ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
                ax.axvline(samples_to_keep[1] - samples_to_keep[0], color='black', linestyle='--', linewidth=0.5)
                if curr_ind == 0:
                    ax.set_title(f'Unit_id: {unit_id} b: {curr_ind} c: {all_counts[unit_ind, curr_ind]} e: {euc_dis[unit_ind, curr_ind]:.2f}')
                    fig.colorbar(im, ax=ax)
                    ax.set_ylabel('Original Template')
                else:
                    ax.set_title(f'b: {curr_ind} c: {all_counts[unit_ind, curr_ind]} e: {euc_dis[unit_ind, curr_ind]:.2f}')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax_wf_ori.plot(all_peak_bins_ori[curr_ind], color='black', alpha=0.1)

                # reordered template
                ax = fig.add_subplot(gs[unit_ind*2+1, curr_ind])
                curr_temp = all_temps_bins[curr_ind]
                im = ax.imshow(curr_temp, extent=[samples_to_keep[0], 
                                                samples_to_keep[0] + 2 * (samples_to_keep[1] - samples_to_keep[0]), 
                                                2 * y_neighbors_to_keep + 1, 0], 
                                                cmap=shifted_cmap, interpolation='nearest',
                                                vmin=min_all, vmax=max_all, aspect='auto')
                ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
                ax.axvline(samples_to_keep[1] - samples_to_keep[0], color='black', linestyle='--', linewidth=0.5)
                if curr_ind == 0:
                    ax.set_title(f'Unit_id: {unit_id} bin: {curr_ind} present: {presence_score[unit_ind, curr_ind]}')
                    fig.colorbar(im, ax=ax)
                    ax.set_ylabel('Reordred Template')
                else:
                    ax.set_title(f'bin: {curr_ind} present: {presence_score[unit_ind, curr_ind]}')
                ax.spines['top'].set_visible(False) 
                ax.spines['right'].set_visible(False)
                ax_wf_reo.plot(all_peak_bins[curr_ind], color='black', alpha=0.1)

        ax = fig.add_subplot(gs[2*len(all_tagged_units):, 0:2])
        ax.scatter(all_amp_opt, all_amps_ori)
        all_max = max(max(all_amps_ori), max(all_amp_opt))
        all_min = min(min(all_amps_ori), min(all_amp_opt))
        ax.plot([all_min, all_max], [all_min, all_max], color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Amplitude - recomputed')
        ax.set_ylabel('Amplitude - original')

        plt.tight_layout()
        fig.savefig(fname=f'{session_dir[f"ephys_dir_{data_type}"]}/waveforms_recompute.pdf')
        print('Saved figure')

    return waveform_recompute
    

    
if __name__ == "__main__": 
    # session = 'behavior_751004_2024-12-20_13-26-11'
    # suppress warnings from spikeinterface package
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='spikeinterface')

    data_type = 'curated'
    target = 'soma'
    load_sorting_analyzer = True
    session = 'behavior_751181_2025-02-27_11-24-47'
    # opto_wf_preprocessing(session, data_type, target, load_sorting_analyzer = load_sorting_analyzer)
 
    session_assets = pd.read_csv('/root/capsule/code/data_management/session_assets.csv')
    session_list = session_assets['session_id'].values
    ind = [i for i, session in enumerate(session_list) if session == 'behavior_751769_2025-01-16_11-32-05']
    ind = ind[0]
    # print(session_list[-2])
    # session = 'behavior_751004_2024-12-21_13-28-28'
    # waveform_recompute_session(session, data_type, load_sorting_analyzer=True, opto_only=True, plot=True, save=False)
    def process(session):
        print(session)
        session_dir = session_dirs(session)
        # if os.path.exists(os.path.join(session_dir['beh_fig_dir'], f'{session}.nwb')):   
        if session_dir['curated_dir_curated'] is not None:
            data_type = 'curated'
            # opto_wf_preprocessing(session, data_type, target, load_sort ing_analyzer = load_sorting_analyzer)
            outcome = waveform_recompute_session(session, data_type, load_sorting_analyzer= True, opto_only=True, plot=True, save=True)
            del outcome
            # elif session_dir['nwb_dir_raw'] is not None:
            #     data_type = 'raw'
            #     opto_wf_p reprocessing(session, data_type, target, load_sorting_analyzer = load_sorting_analyzer)
    
    

    # Parallel(n_jobs=4)(delayed(process)(session) for session in  session_list[10:17]) 
    # for session in session_list[16:17]: 
    #     process(session) 
    process('behavior_754897_2025-03-15_11-32-18') 
