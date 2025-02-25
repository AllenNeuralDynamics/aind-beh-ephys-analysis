# %%
import os
import sys
sys.path.append('/root/capsule/aind-beh-ephys-analysis/code/beh_ephys_analysis')
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re 
from utils.beh_functions import session_dirs
from utils.plot_utils import shiftedColorMap, template_reorder, plot_raster_bar,merge_pdfs
from open_ephys.analysis import Session
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
from aind_ephys_utils import align
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
import pickle

from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import load_nwb
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename

 
# Create a white-to-bright red colormap
colors = [(1, 1, 1), (1, 0, 0)]  # white to red
my_red = LinearSegmentedColormap.from_list("white_to_red", colors)

# %%
# session = 'behavior_758017_2025-02-04_11-57-38'
# session_dir = session_dirs(session)
# data_type = 'raw'
# target = 'soma'

# resp_thresh = 0.8
# lat_thresh = 0.015
# pulse_width = 4
# %%
def max_index_ndarray(arr):
    """
    Find the index of the maximum value in an N-dimensional array.

    Parameters:
        arr (numpy.ndarray): Input N-dimensional array.

    Returns:
        tuple: Indices of the maximum value in each dimension.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    # Find the flattened index of the max value
    max_flat_index = np.argmax(arr)

    # Convert flattened index back to multi-dimensional index
    max_nd_index = np.unravel_index(max_flat_index, arr.shape)

    return max_nd_index 
# %%
def opto_plotting_unit(unit_id, spike_times, spike_amplitude, waveform, opto_wf, qc_dict, crosscorr, resp_p, resp_lat, opto_df, opto_info, waveform_params, dim_1 = 'powers', resp_thresh=0.8, lat_thresh=0.015, plot = False):
    # calculate baseline
    baseline = qc_dict['firing_rate']*opto_info['resp_win']
    # conditions
    # opto tagging
    pass_opto = False
    fig = None
    # find resp_p > thresh and resp_lat < lat_thresh
    resp_p = np.array(resp_p.tolist())
    resp_lat = np.array(resp_lat.tolist())
    resp_pass_ind = np.where((resp_p > resp_thresh + baseline) & (resp_lat < lat_thresh) & (resp_lat > 3/1000))
    opto_tagging_dict = {'unit_id': unit_id, 
                 'resp_p': None, 
                 'resp_lat': None, 
                 'powers': None,
                 'sites': None, 
                 'num_pulses': None, 
                 'durations': None,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                 'freqs': None, 
                 'stim_times': None, 
                 'opto_pass': False}
    if len(resp_pass_ind[0]) > 0:
        opto_tagging_dict['opto_pass'] = True
        # get the response p and latencies
        opto_tagging_dict['resp_p'] = resp_p[resp_pass_ind]
        opto_tagging_dict['resp_lat'] = resp_lat[resp_pass_ind]
        power_ind, site_ind, num_pulse_ind, duration_ind, freq_ind, stim_time_ind = resp_pass_ind[:-1]
        opto_tagging_dict['powers'] = np.array(opto_info['powers'])[power_ind]
        opto_tagging_dict['sites'] = np.array(opto_info['sites'])[site_ind]
        opto_tagging_dict['num_pulses'] = np.array(opto_info['num_pulses'])[num_pulse_ind]
        opto_tagging_dict['durations'] = np.array(opto_info['durations'])[duration_ind]
        opto_tagging_dict['freqs'] = np.array(opto_info['freqs'])[freq_ind]
        opto_tagging_dict['stim_times'] = np.array(['pre', 'post'])[stim_time_ind]
        # get all similarities
        opto_tagging_df = pd.DataFrame(opto_tagging_dict)
        # group by site, power, duration, pre_post and take maximum of resp_p and resp_lat
        opto_tagging_df = opto_tagging_df.groupby(['unit_id', 'sites', 'powers', 'num_pulses', 'durations', 'freqs', 'stim_times']).agg({'resp_p': 'max', 'resp_lat': 'min'}).reset_index()
        
        euc_dist = []
        corr = []
        for _, row in opto_tagging_df.iterrows():
            wf_curr = opto_wf.query(
                'unit_id == @unit_id and site == @site and power == @power and pre_post == @pre_post',
                local_dict={
                    'site': row['sites'],
                    'power': row['powers'],
                    'duration': row['durations'],
                    'pre_post': row['stim_times'],
                    'unit_id': unit_id
                }
            )
            # Ensure wf_curr is not empty before extracting values
            if not wf_curr.empty:
                euc_dist_curr, corr_curr = wf_curr.iloc[0][['euclidean_norm', 'correlation']]
            else:
                euc_dist_curr, corr_curr = None, None  # Handle missing values

            euc_dist.append(euc_dist_curr)
            corr.append(corr_curr)
        opto_tagging_df['euclidean_norm'] = euc_dist
        opto_tagging_df['correlation'] = corr
        opto_tagging_dict = {key: opto_tagging_df[key].values for key in opto_tagging_df.columns}
    opto_tagging_dict['unit_id'] = unit_id
    opto_tagging_dict.update(qc_dict)

    
    # plot
    if plot:
        fig = plt.figure(figsize=(12, 8))
        # select the first dimension to separate by subplots
        gs_all = gridspec.GridSpec(2, 1, height_ratios=[1, 5])
        # plot basic firing rates first
        gs_basic = gridspec.GridSpecFromSubplotSpec(2, 3, width_ratios=[1, 1, 2], subplot_spec=gs_all[0])
        ax = fig.add_subplot(gs_basic[0, 2])
        bins = np.linspace(np.min(spike_times), np.max(spike_times), 200)
        ax.hist(spike_times, bins=bins, color='black', alpha=0.5)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks([])
        ax.set_title(f'Firing Rate: {qc_dict["firing_rate"]:.2f} Hz')
        ax = fig.add_subplot(gs_basic[1, 2])
        ax.scatter(spike_times, -spike_amplitude, color='b', alpha=0.5, s=1, edgecolors='none')
        ax.set_xlim(bins[0], bins[-1])
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.set_title('Spike Amplitude (uV)')
        ax.set_xlabel('Time (s)')
        # long crosscorr
        # no laser
        ax = fig.add_subplot(gs_basic[0, 0])
        if unit_id in crosscorr['long']['unit_ids_nolaser']:
            unit_ind = np.where(crosscorr['long']['unit_ids_nolaser'] == unit_id)[0][0]
            bin_size = np.mean(np.diff(crosscorr['long']['time_bins']))
            ax.bar(0.5*crosscorr['long']['time_bins'][1:] + 0.5*crosscorr['long']['time_bins'][:-1], crosscorr['long']['correlogram_nolaser'][unit_ind, unit_ind, :], width=bin_size, color='black', alpha=0.75)
        ax.set_xlim(crosscorr['long']['time_bins'][0], crosscorr['long']['time_bins'][-1])
        ax.set_title('Long Crosscorr')
        ax.set_ylabel('Session')
        # turn off x ticks
        ax.set_xticks([])
        # laser
        ax = fig.add_subplot(gs_basic[1, 0])
        if unit_id in crosscorr['long']['unit_ids_laser']:
            unit_ind = np.where(crosscorr['long']['unit_ids_laser'] == unit_id)[0][0]
            bin_size = np.mean(np.diff(crosscorr['long']['time_bins']))
            ax.bar(0.5*crosscorr['long']['time_bins'][1:] + 0.5*crosscorr['long']['time_bins'][:-1], crosscorr['long']['correlogram_laser'][unit_ind, unit_ind, :], width=bin_size, color='red', alpha=0.75)
        ax.set_xlim(crosscorr['long']['time_bins'][0], crosscorr['long']['time_bins'][-1])
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel('Opto')

        # short crosscorr
        # no laser
        ax = fig.add_subplot(gs_basic[0, 1])
        if unit_id in crosscorr['short']['unit_ids_nolaser']:
            unit_ind = np.where(crosscorr['short']['unit_ids_nolaser'] == unit_id)[0][0]
            bin_size = np.mean(np.diff(crosscorr['short']['time_bins']))
            ax.bar(0.5*crosscorr['short']['time_bins'][1:] + 0.5*crosscorr['short']['time_bins'][:-1], crosscorr['short']['correlogram_nolaser'][unit_ind, unit_ind, :], width=bin_size, color='black', alpha=0.5)
        ax.set_xlim(crosscorr['short']['time_bins'][0], crosscorr['short']['time_bins'][-1])
        ax.set_title('Short Crosscorr')
        ax.set_xticks([])
        # laser
        ax = fig.add_subplot(gs_basic[1, 1])
        if unit_id in crosscorr['short']['unit_ids_laser']:
            unit_ind = np.where(crosscorr['short']['unit_ids_laser'] == unit_id)[0][0]
            bin_size = np.mean(np.diff(crosscorr['short']['time_bins']))
            ax.bar(0.5*crosscorr['short']['time_bins'][1:] + 0.5*crosscorr['short']['time_bins'][:-1], crosscorr['short']['correlogram_laser'][unit_ind, unit_ind, :], width=bin_size, color='red', alpha=0.5)
        ax.set_xlim(crosscorr['short']['time_bins'][0], crosscorr['short']['time_bins'][-1])
        ax.set_xlabel('Lag (ms)')
        plt.tight_layout()



        gs = gridspec.GridSpecFromSubplotSpec(1, len(opto_info[dim_1]), width_ratios=[1]*len(opto_info[dim_1]), subplot_spec=gs_all[1])
        max_p_all = []
        max_lat_all = []
        pulse_width = opto_df['duration'].mode()[0]

        colors = ["blue", "white", "red"]
        b_w_r_cmap = LinearSegmentedColormap.from_list("b_w_r", colors)
        
        for power_ind, curr_power in enumerate(opto_info[dim_1]):
            gs_sub_raster = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[power_ind], height_ratios=[1, 1], width_ratios=[1,2])
            curr_resp_p = resp_p[power_ind, :, :, np.sort(opto_info[opto_info['dimensions'][3]+'s'])==pulse_width, :]
            if np.isnan(curr_resp_p).all():
                continue
            curr_resp_lat = resp_lat[power_ind, :, :, np.array(opto_info[opto_info['dimensions'][3]+'s'])==pulse_width, :]
            curr_resp_p = np.array(curr_resp_p.tolist())
            curr_resp_lat = np.array(curr_resp_lat.tolist())
            # remove rows that are all nan along last axis
            mask = np.squeeze(~np.isnan(curr_resp_p).all(axis=-1, keepdims=True))
            curr_resp_lat = curr_resp_lat[:, :, :, :, mask, :]
            curr_resp_p = curr_resp_p[:, :, :, :, mask, :]
            # if multi-sites, use sites as frist dimention to plot, else if multi-time, use multi-time as first dimension
            # if curr_resp_p.shape[1] > 1:
            #     curr_resp_p = np.squeeze(curr_resp_p)
            #     curr_resp_lat = np.squeeze(curr_resp_lat)
            # elif any(dim > 1 for dim in list(curr_resp_p.shape)[2:-1]):
            # if more than 2 dimensions>1, remain the first dimension and max over the rest
            if np.sum(np.array([dim > 1 for dim in list(curr_resp_p.shape)[0:-1]]))>1:
                dims = tuple(i for i in range(curr_resp_p.ndim) if i != 0 and i != 1 and i !=curr_resp_p.ndim) # exclude site and power and pulses
                curr_resp_p_colormap = np.max(curr_resp_p, axis = dims) # shape is 2d, sites x pulses
                curr_resp_p_colormap = np.squeeze(curr_resp_p_colormap)
            else:
                curr_resp_p_colormap = np.squeeze(curr_resp_p)

            if curr_resp_p_colormap.ndim == 1:
                curr_resp_p_colormap = curr_resp_p_colormap[:, np.newaxis].T
            # plot response p as heatmap
            ax = fig.add_subplot(gs_sub_raster[0,0])
            ax.imshow(curr_resp_p_colormap, cmap=my_red, aspect='auto', vmin=0, vmax=1)
            # find max response p along last dimention (pulses) then find max conditions
            max_ind = max_index_ndarray(curr_resp_p)
            site_ind = max_ind[1]
            max_site = opto_info['sites'][site_ind]
            max_p_all.append(curr_resp_p[max_ind])
            max_lat_all.append(curr_resp_lat[max_ind])

            p_train_to_plot = curr_resp_p[max_ind[:-1]]
            lat_train_to_plot = curr_resp_lat[max_ind[:-1]]

            if np.shape(curr_resp_p)[-2]==1:
                curr_pre_post = 'post'
            elif max_ind[-2] == 0:
                curr_pre_post = 'pre'
            else:
                curr_pre_post = 'post'
            
            laser_times_curr = np.sort(np.concatenate([opto_df.query('site == @max_site and power == @curr_power and pre_post == "pre"')['time'].values, 
                                        opto_df.query('site == @max_site and power == @curr_power and pre_post == "post"')['time'].values], axis = 0), )[::-1] 
            raster_df = align.to_events(spike_times, laser_times_curr, (-0.5, 1.5), return_df=True)
            ax = fig.add_subplot(gs_sub_raster[0,1])
            plot_raster_bar(raster_df, ax)
            ax.set_title(f'Power: {curr_power}; Site: {max_site}')
            ax.set_xlim(-0.5, 1)
            ax.set_ylim(0, len(laser_times_curr)+1)
            ax.set_yticks([])
            # ax.axis('off')
            ax.set(xlabel='Time (s)')
            for spine in ax.spines.values():
                spine.set_visible(False)
            for j in range(opto_info['num_pulses'][0]):
                x = j * 1/opto_info['freqs'][0]
                rect = patches.Rectangle((x, 0), opto_info['durations'][0]/1000, len(laser_times_curr), color='r', alpha=0.5, edgecolor=None)
                ax.add_patch(rect)

            # plot waveform
            gs_sub_waveform = gridspec.GridSpecFromSubplotSpec(6, 2, subplot_spec=gs[power_ind], hspace=0.75)
            ax = fig.add_subplot(gs_sub_waveform[3,0])
            wf_resp = opto_wf.query('unit_id == @unit_id and site == @max_site and power == @curr_power and duration == @pulse_width and pre_post == @curr_pre_post')
            wf_spont = opto_wf.query('unit_id == @unit_id and pre_post == @curr_pre_post and spont == 1')

            if len(wf_resp) > 0:
                ax.plot(wf_spont['peak_waveform'].values[0], color='black', alpha = 0.5)
            if len(wf_resp) > 0:
                # plot the response waveform
                ax.plot(wf_resp['peak_waveform'].values[0], color='red')
                ax.set_title(f'Euc: {wf_resp["euclidean_norm"].values[0]:.2f} Corr: {wf_resp["correlation"].values[0]:.2f}')
            
            ax = fig.add_subplot(gs_sub_waveform[3,1])
            shifted_cmap = shiftedColorMap(b_w_r_cmap, np.nanmin(waveform), np.nanmax(waveform), 'shifted_b_w_r');
            cax = ax.imshow(waveform[:, 0:int(0.5*np.shape(waveform)[1])], extent = [waveform_params['samples_to_keep'][0], waveform_params['samples_to_keep'][1], 2*waveform_params['y_neighbors_to_keep']+1, 0], cmap=shifted_cmap, aspect='auto');
            fig.colorbar(cax, ax=ax)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
            ax.axvline(waveform_params['samples_to_keep'][1]-waveform_params['samples_to_keep'][0], color='black', linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            # plot best response p and latency
            # location_max = np.argmax(np.mean(curr_resp_p_max, axis = -1))
            # max_site = opto_info['sites'][location_max]
            ax = fig.add_subplot(gs_sub_waveform[4,:])
            ax.plot(p_train_to_plot)
            ax.set_ylim(baseline, 1)
            ax.axhline(resp_thresh, color='black', linestyle='--', linewidth=0.5)
            ax.set_title(f'P(resp) Site: {max_site}')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax = fig.add_subplot(gs_sub_waveform[5,:])
            ax.plot(1000*lat_train_to_plot)
            ax.set_ylim(0, 1000*opto_info['resp_win'])
            ax.axhline(1000*lat_thresh, color='black', linestyle='--', linewidth=0.5)
            ax.set_title(f'Latency (ms)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        max_p_all = np.max(np.array(max_p_all)) - baseline
        max_lat_all = np.max(np.array(max_lat_all))
        pass_qc_curr = (qc_dict['isi_violations_ratio'] < 0.5) & \
                (qc_dict['firing_rate'] > 0.1) & \
                (qc_dict['presence_ratio'] > 0.95) & \
                (qc_dict['amplitude_cutoff'] < 0.05) & \
                (qc_dict['decoder_label'] != 'noise') & \
                (qc_dict['decoder_label'] != 'artifact')
        plt.suptitle(f"Unit {unit_id} Depth: {qc_dict['depth']:.2f} RespWin: {opto_info['resp_win']} s  pResp: {max_p_all:.2f} Lat: {max_lat_all:.2f} pass_qc {pass_qc_curr}")
        plt.tight_layout()
    return fig, opto_tagging_dict
#%%
def opto_plotting_session(session, data_type, target, resp_thresh=0.8, lat_thresh=0.015, plot = False, target_unit_ids=None):
    session_dir = session_dirs(session)
    sorting = si.load_extractor(session_dir[f'curated_dir_{data_type}'])
    we = si.load_sorting_analyzer_or_waveforms(session_dir[f'postprocessed_dir_{data_type}'])
    spike_amplitude = we.load_extension('spike_amplitudes').get_data(outputs="by_unit")[0]
    unit_ids = sorting.get_unit_ids()

    # load quality metrics from nwb
    # load quality metrics
    if os.path.exists(session_dir[f'nwb_dir_{data_type}']):
        nwb = load_nwb(session_dir[f'nwb_dir_{data_type}'])
        unit_qc = nwb.units[:][['ks_unit_id', 'isi_violations_ratio', 'firing_rate', 'presence_ratio', 'amplitude_cutoff', 'decoder_label', 'depth']]
    else:
        print('No nwb file found.')
    
    # change all strings in unit_qc to float 
    #     qm = pd.read_csv(session_dir['qm_dir'], index_col=0)
    #     unit_qc = qm[:][['isi_violations_ratio', 'firing_rate', 'presence_ratio', 'amplitude_cutoff']]
    #     unit_qc['ks_unit_id'] = unit_qc.index
    #     sorting = si.load_extractor(session_dir['curated_dir'])
    #     label = sorting.get_property('decoder_label')
    #     unit_qc['decoder_label'] = label
    unit_qc = unit_qc.apply(pd.to_numeric, errors='ignore') 
    pass_qc = (unit_qc['isi_violations_ratio'] < 0.5) & \
            (unit_qc['firing_rate'] > 0.1) & \
            (unit_qc['presence_ratio'] > 0.95) & \
            (unit_qc['amplitude_cutoff'] < 0.05) & \
            (unit_qc['decoder_label'] != 'noise') & \
            (unit_qc['decoder_label'] != 'artifact')
    pass_qc = pass_qc.values
    pass_qc = {unit_id: pass_qc_curr for unit_id, pass_qc_curr in zip(unit_ids, pass_qc)}
    print(f'{sum(pass_qc.values())} out of {len(pass_qc)} units pass quality control')


    # load waveforms info
    # we = si.load_sorting_analyzer_or_waveforms(session_dir[f'postprocessed_dir_{data_type}'])
    # unit_locations = we.load_extension("unit_locations").get_data(outputs="by_unit")
    # channel_locations = we.get_channel_locations()
    # right_left = channel_locations[:, 0]<20
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], session+'_waveform_params.json')) as f:
        waveform_params = json.load(f)
    print(waveform_params)

    # load opto responses

    with open(os.path.join(session_dir[f'ephys_processed_dir_{data_type}'], 'spiketimes.pkl'), 'rb') as f:
        spiketimes = pickle.load(f)
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_responses_{target}.pkl'), 'rb') as f:
        opto_responses = pickle.load(f)
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_waveforms_{target}.pkl'), 'rb') as f:
        waveforms = pickle.load(f)
    
    # load crosscorr
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_correlogram_laser_short.pkl'), 'rb') as f:
        crosscorr_short = pickle.load(f)
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_correlogram_laser_long.pkl'), 'rb') as f:
        crosscorr_long = pickle.load(f)
    crosscorr = {'short': crosscorr_short, 'long': crosscorr_long}
    # load waveforms from csv
    opto_wf_pkl = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_waveform_metrics.pkl')
    with open (opto_wf_pkl, 'rb') as f:
        opto_wf = pickle.load(f)
    
    opto_wf = opto_wf.apply(pd.to_numeric, errors='ignore')

    opto_df = pd.read_csv(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_session_{target}.csv'))
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_info_{target}.json')) as f:
        opto_info = json.load(f)
    
    # opto_info = {key: np.sort(value) for key, value in opto_info.items()}
    pulse_width = opto_df['duration'].mode()[0]
    colors = ["blue", "white", "red"]
    b_w_r_cmap = LinearSegmentedColormap.from_list("b_w_r", colors)
    opto_pass = []
    if target_unit_ids is None:
        target_unit_ids = unit_ids
    target_pass_qc = []
    opto_tagging_df_sess = pd.DataFrame()
    for unit_id in target_unit_ids:
    #     if pass_qc[unit_id]:   
        qc_dict = unit_qc.loc[unit_qc['ks_unit_id'] == unit_id].iloc[0].to_dict()
        fig, opto_tagging_dict_curr = opto_plotting_unit(unit_id, spiketimes[unit_id], spike_amplitude[unit_id], 
                                                        waveforms[unit_id], opto_wf, qc_dict, crosscorr,
                                                        opto_responses['resp_p'][unit_id], opto_responses['resp_lat'][unit_id], 
                                                        opto_df, opto_info, 
                                                        waveform_params, 
                                                        dim_1 = 'powers', resp_thresh=resp_thresh, lat_thresh=lat_thresh, plot=plot)
        if fig is not None:
            fig.savefig(os.path.join(session_dir[f'opto_dir_fig_{data_type}'], f'unit_{unit_id}_pulse_width_{pulse_width}_opto_tagging.pdf'))
        plt.close('all')
        opto_tagging_df_sess = pd.concat([opto_tagging_df_sess, pd.DataFrame([opto_tagging_dict_curr])], ignore_index=True)
        target_pass_qc.append(pass_qc[unit_id])
        opto_pass_curr = True
        if opto_tagging_dict_curr['resp_p'] is None:
            opto_pass_curr = False
        opto_pass.append(opto_pass_curr)

    merge_pdfs(session_dir[f'opto_dir_fig_{data_type}'], os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_tagging.pdf'))
    # both pass qc and opto tagging
    unit_count_pass = np.sum(np.array(target_pass_qc) & np.array(opto_pass))
    print(f'{unit_count_pass} out of {len(target_pass_qc)} units pass quality control and opto tagging')
    opto_tagging_df_sess['default_qc'] = target_pass_qc
    opto_tagging_df_sess['opto_pass'] = opto_pass
    # target_qc['target_pass_qc'] = target_pass_qc
    return opto_tagging_df_sess

# %%
if __name__ == "__main__":
    session = 'behavior_758017_2025-02-04_11-57-38'
    target = 'soma'
    data_type = 'raw' 
    resp_thresh = 0.3
    lat_thresh = 0.02 
    session_dir = session_dirs(session)
    # merge_pdfs(session_dir[f'opto_dir_fig_{data_type}'], os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_tagging.pdf'))

    opto_tagging_df_sess = opto_plotting_session(session, data_type, target, resp_thresh=resp_thresh, lat_thresh=lat_thresh, target_unit_ids= None, plot = False)
    opto_tagging_df_sess.to_csv(os.path.join(session_dirs(session)[f'opto_dir_{data_type}'], f'{session}_opto_tagging_metrics.csv'), index=False)



# %%
