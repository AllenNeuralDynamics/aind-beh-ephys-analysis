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
def opto_plotting_unit(unit_id, spike_times, waveform, qc_dict, resp_p, resp_lat, opto_df, opto_info, waveform_params, dim_1 = 'powers', resp_thresh=0.8, lat_thresh=0.015, plot = False):
    # calculate baseline
    baseline = qc_dict['firing_rate']*opto_info['resp_win']

    # opto tagging
    pass_opto = False
    fig = None
    # find resp_p > thresh and resp_lat < lat_thresh
    resp_p = np.array(resp_p.tolist())
    resp_lat = np.array(resp_lat.tolist())
    resp_pass_ind = np.where((resp_p > resp_thresh) & (resp_lat < lat_thresh))[0]
    if len(resp_pass_ind) > 0:
        pass_opto = True

    
    # plot
    if plot:
        fig = plt.figure(figsize=(12, 8))
        # select the first dimension to separate by subplots
        gs = gridspec.GridSpec(1, len(opto_info[dim_1]), width_ratios=[1]*len(opto_info[dim_1]))
        max_p_all = []
        max_lat_all = []
        pulse_width = opto_df['duration'].mode()[0]

        colors = ["blue", "white", "red"]
        b_w_r_cmap = LinearSegmentedColormap.from_list("b_w_r", colors)
        
        for power_ind, curr_power in enumerate(opto_info[dim_1]):
            gs_sub_raster = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[power_ind], height_ratios=[1, 1], width_ratios=[1,2])
            curr_resp_p = resp_p[power_ind, :, :, np.sort(opto_info[opto_info['dimensions'][3]+'s'])==pulse_width, :]
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
            ax = fig.add_subplot(gs_sub_waveform[3,:])
            shifted_cmap = shiftedColorMap(b_w_r_cmap, np.nanmin(waveform), np.nanmax(waveform), 'shifted_b_w_r');
            cax = ax.imshow(waveform, extent = [waveform_params['samples_to_keep'][0], waveform_params['samples_to_keep'][0]+2*(waveform_params['samples_to_keep'][1]-waveform_params['samples_to_keep'][0]), 2*waveform_params['y_neighbors_to_keep']+1, 0], cmap=shifted_cmap, aspect='auto');
            fig.colorbar(cax, ax=ax)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
            ax.axvline(waveform_params['samples_to_keep'][1]-waveform_params['samples_to_keep'][0], color='black', linestyle='--', linewidth=0.5)
            ax.set_title(f'depth: {qc_dict["depth"]:.2f}')
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
        plt.suptitle(f"Unit {unit_id} RespWin: {opto_info['resp_win']} s pResp: {max_p_all:.2f} Lat: {max_lat_all:.2f} pass_qc {pass_qc_curr}")
        plt.tight_layout()
    return fig, pass_opto
#%%
def opto_plotting_session(session, data_type, target, resp_thresh=0.8, lat_thresh=0.015, plot = False):
    session_dir = session_dirs(session)
    sorting = si.load_extractor(session_dir[f'curated_dir_{data_type}'])
    unit_ids = sorting.get_unit_ids()

    # load quality metrics from nwb
    # load quality metrics
    if os.path.exists(session_dir[f'nwb_dir_{data_type}']):
        nwb = load_nwb(session_dir[f'nwb_dir_{data_type}'])
        unit_qc = nwb.units[:][['ks_unit_id', 'isi_violations_ratio', 'firing_rate', 'presence_ratio', 'amplitude_cutoff', 'decoder_label', 'depth']]
    else:
        print('No nwb file found.')
    
    # change all strings in unit_qc to float
    unit_qc = unit_qc.apply(pd.to_numeric, errors='coerce')   
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

    # prepare for heatmap
    with open(os.path.join(session_dir[f'ephys_processed_dir_{data_type}'], 'spiketimes.pkl'), 'rb') as f:
        spiketimes = pickle.load(f)
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_responses_{target}.pkl'), 'rb') as f:
        opto_responses = pickle.load(f)
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_waveforms_{target}.pkl'), 'rb') as f:
        waveforms = pickle.load(f)

    opto_df = pd.read_csv(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_session_{target}.csv'))
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_info_{target}.json')) as f:
        opto_info = json.load(f)
    pulse_width = opto_df['duration'].mode()[0]
    colors = ["blue", "white", "red"]
    b_w_r_cmap = LinearSegmentedColormap.from_list("b_w_r", colors)
    opto_pass = []
    for unit_id in unit_ids:
    #     if pass_qc[unit_id]:   
        qc_dict = unit_qc.loc[unit_qc['ks_unit_id'] == unit_id].iloc[0].to_dict()
        fig, opto_pass_curr = opto_plotting_unit(unit_id, spiketimes[unit_id], waveforms[unit_id], qc_dict, 
                                        opto_responses['resp_p'][unit_id], opto_responses['resp_lat'][unit_id], opto_df, opto_info, 
                                        waveform_params, dim_1 = 'powers', resp_thresh=resp_thresh, lat_thresh=lat_thresh, plot=plot)
        if fig is not None:
            fig.savefig(os.path.join(session_dir[f'opto_dir_fig_{data_type}'], f'unit_{unit_id}_pulse_width_{pulse_width}_opto_tagging.pdf'))
        plt.close('all')
        opto_pass.append(opto_pass_curr)

    merge_pdfs(session_dir[f'opto_dir_fig_{data_type}'], os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_tagging.pdf'))
    # both pass qc and opto tagging
    unit_count_pass = np.sum(np.array(list(pass_qc.values())) & np.array(opto_pass))
    print(f'{unit_count_pass} out of {len(pass_qc)} units pass quality control and opto tagging')
    pass_dict = {'pass_qc': pass_qc.values(), 'opto_tagging': opto_pass, 'unit_id': unit_ids}
    return pass_dict

# %%
if __name__ == "__main__":
    session = 'behavior_758017_2025-02-04_11-57-38'
    target = 'soma'
    data_type = 'raw' 
    resp_thresh = 0.6
    lat_thresh = 0.02
    

    opto_plotting_session(session, target, data_type, resp_thresh=resp_thresh, lat_thresh=lat_thresh, plot = True)




# %%
