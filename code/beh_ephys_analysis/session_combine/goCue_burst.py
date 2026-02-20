# %%
import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from scipy.io import loadmat
from scipy.stats import zscore
import ast
from pathlib import Path
import glob
import json
import seaborn as sns
from PyPDF2 import PdfMerger
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import re
from utils.beh_functions import session_dirs, parseSessionID, load_model_dv, makeSessionDF, get_session_tbl, get_unit_tbl, get_history_from_nwb
from utils.ephys_functions import*
from utils.ccf_utils import ccf_pts_convert_to_mm
import pickle
import scipy.stats as stats
import spikeinterface as si
import shutil
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import r2_score
import warnings
from scipy.stats import gaussian_kde
warnings.filterwarnings('ignore')
from matplotlib import cm
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.stats import ttest_ind
from sklearn.cross_decomposition import CCA
from trimesh import load_mesh
from joblib import Parallel, delayed
from utils.plot_utils import shiftedColorMap, template_reorder, plot_raster_bar,merge_pdfs, combine_pdf_big

def burst_analysis(session, data_type, units = None):
    print(f'Processing session {session} for data type {data_type}')
    unit_tbl = get_unit_tbl(session, data_type)
    session_df = get_session_tbl(session)
    session_dir = session_dirs(session, data_type)
    save_path = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], 'burst')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    if units is None:
        units = unit_tbl['unit_id'].tolist()
    for unit_id in units:
        # print(unit_id)
        if not unit_tbl[unit_tbl['unit_id']==unit_id]['tagged_loc'].values[0]:
            continue
        spike_times = unit_tbl[unit_tbl['unit_id']==unit_id]['spike_times'].values[0]
        pre_event = -0.1
        post_event= 0.1
        session_df_curr = session_df.copy()
        spike_times_curr = spike_times.copy()
        unit_drift = load_drift(session, unit_id, data_type=data_type)

        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_df_curr = session_df_curr[session_df_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_df_curr = session_df_curr[session_df_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        if len(session_df_curr) <=20:
            print(f'Skipping {unit_id} due to insufficient trials after drift cut.')
            continue
        # align to go cue sorted by choice time, separate by choice or not
        # from start to end
        fig = plt.figure(figsize=(14,10))
        gs = gridspec.GridSpec(2,8)
        lick_lat = session_df_curr['reward_outcome_time'].values - session_df_curr['goCue_start_time'].values
        lick_lat[session_df_curr['animal_response'].values==2] = np.nan
        pre_event = -0.1
        post_event= 0.1
        align_time = session_df_curr['goCue_start_time'].values
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[0, 0])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Go Cue')

        align_time = session_df_curr['reward_outcome_time'].values
        lick_lat[session_df_curr['animal_response'].values==2] = np.nan
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[0, 1])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Choice')


        # align to go cue and sort by frist spike time
        align_time = session_df_curr['goCue_start_time'].values
        spike_df = align.to_events(spike_times_curr, align_time, (0, 10), return_df=True)
        # for each value in event_index, get the first spike time
        first_spike_times = np.full(len(session_df_curr), np.nan)
        for i in range(len(session_df_curr)):
            spikes_in_trial = spike_df[spike_df['event_index']==i]['time']
            if len(spikes_in_trial) > 0:
                first_spike_times[i] = spikes_in_trial.min()
        # first_spike_times = np.full(len(session_df_curr), np.nan)
        align_time_firstspike_sort = align_time[np.argsort(first_spike_times)]
        ax = fig.add_subplot(gs[0, 2])
        df = align.to_events(spike_times_curr, align_time_firstspike_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('First spike time sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.set_title('Aligned to Go Cue')

        # align to first spike time
        first_spike_times_abs = first_spike_times + session_df_curr['goCue_start_time'].values
        first_spike_times_abs_sorted = first_spike_times_abs[np.argsort(first_spike_times)]
        df = align.to_events(spike_times_curr, first_spike_times_abs_sorted, (-0.05, 0.03), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax = fig.add_subplot(gs[0, 3])
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(-0.05, 0.03)
        ax.set_ylabel('Sorted by First Spike Time to go cue')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.set_title('Aligned to First Spike Time')

        # isi distribution
        ax = fig.add_subplot(gs[1, 3:5])
        isi_spikes = np.log(np.diff(spike_times_curr))
        edges = np.linspace(np.nanmin(isi_spikes), np.nanmax(isi_spikes), 50)
        ax.hist(isi_spikes, bins=edges, color='k', alpha = 0.5, density=True)
        # set xlabel to log
        ax.axvline(np.log(0.002), color='k', linestyle='--')
        ax.set_xlabel('log(Inter-spike interval (s))')
        ax.set_ylabel('Density')
        ax.set_title('log(ISI) Distribution')

        # color by isi
        first_spike_times_abs = first_spike_times + session_df_curr['goCue_start_time'].values
        first_spike_times_abs_sorted = first_spike_times_abs[np.argsort(first_spike_times)]
        df = align.to_events(spike_times_curr, first_spike_times_abs_sorted, (-0.05, 0.03), return_df=True)

        isi_list = df.copy()
        isi_list['isi'] = np.nan
        # infer time interval of previous spike for each spike
        for ind, row in isi_list.iterrows():
            event_index = row['event_index']
            time = row['time']
            prev_spikes = spike_times_curr[spike_times_curr < (time + first_spike_times_abs_sorted[int(event_index)])]
            if len(prev_spikes) == 0:
                isi_list.at[ind, 'isi'] = np.nan
            else:
                isi_list.at[ind, 'isi'] = time + first_spike_times_abs_sorted[[int(event_index)]] - prev_spikes[-1]

        isi_color_code = np.log(isi_list['isi'].values)
        up_bound = np.percentile(isi_color_code[~np.isnan(isi_color_code)], 95)
        low_bound = np.percentile(isi_color_code[~np.isnan(isi_color_code)], 5)
        isi_color_code = (isi_color_code - low_bound) / (up_bound - low_bound)
        isi_color_code[isi_color_code>1] = 1
        isi_color_code[isi_color_code<0] = 0

        ax.hist(np.log(isi_list['isi'].values), bins=50, color='r', alpha=0.5, density=True)
        ax.axvline(low_bound, color='b', linestyle='--')
        ax.axvline(up_bound, color='r', linestyle='--')

        ax= fig.add_subplot(gs[0,4])
        sc = ax.scatter(df.time, df.event_index, c=isi_color_code, marker= '|', s=4, zorder = 2, cmap='Reds_r')
        ax.set_xlim(-0.02, 0.03)
        ax.set_ylabel('Sorted by First Spike Time to go cue')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.set_title('Aligned to First Spike Time')

        # add colorbar
        ax = fig.add_subplot(gs[1,5])
        cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)

        # align to go cue and sort by frist spike time
        align_time = session_df_curr['goCue_start_time'].values
        spike_df = align.to_events(spike_times_curr, align_time, (0, 100), return_df=True)
        # for each value in event_index, get the first spike time
        first_spike_times = spike_df.groupby('event_index')['time'].min().values
        align_time_firstspike_sort = align_time[np.argsort(first_spike_times)]
        df = align.to_events(spike_times_curr, align_time_firstspike_sort, (pre_event, post_event), return_df=True)
        isi_list = df.copy()
        isi_list['isi'] = np.nan
        # infer time interval of previous spike for each spike
        for ind, row in isi_list.iterrows():
            event_index = row['event_index']
            time = row['time']
            prev_spikes = spike_times_curr[spike_times_curr < (time + align_time_firstspike_sort[int(event_index)])]
            if len(prev_spikes) == 0:
                isi_list.at[ind, 'isi'] = np.nan
            else:
                isi_list.at[ind, 'isi'] = time + align_time_firstspike_sort[[int(event_index)]] - prev_spikes[-1]

        isi_color_code = np.log(isi_list['isi'].values)
        isi_color_code = (isi_color_code - low_bound) / (up_bound - low_bound)
        isi_color_code[isi_color_code>1] = 1
        isi_color_code[isi_color_code<0] = 0

        # plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax = fig.add_subplot(gs[0, 5])
        ax.scatter(df.time, df.event_index, c=isi_color_code, marker= '|', s=3, zorder = 2, cmap = 'Reds_r')
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('First spike time sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_xlim(-0.02, 0.03)
        ax.set_title('Aligned to Go Cue')


        pre_event = -1.7
        post_event = 0.1
        align_time = session_df_curr['goCue_start_time'].values
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[1, 0])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=3, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Go Cue')


        align_time = session_df_curr['reward_outcome_time'].values
        lick_lat[session_df_curr['animal_response'].values==2] = np.nan
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[1, 1])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=3, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Choice')


        # align to go cue and sort by frist spike time
        post_event = 1
        pre_event = -0.5
        align_time = session_df_curr['goCue_start_time'].values
        spike_df = align.to_events(spike_times_curr, align_time, (0, 100), return_df=True)
        # for each value in event_index, get the first spike time
        first_spike_times = spike_df.groupby('event_index')['time'].min()
        # fill in nan for trials with no spikes
        first_spike_times_full = np.full(len(session_df_curr), 100)
        first_spike_times_full[first_spike_times.index] = first_spike_times.values
        first_spike_times = first_spike_times_full.copy()
        align_time_firstspike_sort = align_time[np.argsort(first_spike_times)]
        ax = fig.add_subplot(gs[1, 2])
        df = align.to_events(spike_times_curr, align_time_firstspike_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=4, zorder = 2)
        trial_ind = np.argsort(first_spike_times)
        lick_time = session_df_curr['reward_outcome_time'].values[trial_ind] - session_df_curr['goCue_start_time'].values[trial_ind]
        lick_time[session_df_curr['animal_response'].values[trial_ind]==2] = np.nan
        ax.scatter(lick_time, np.arange(len(session_df_curr)), c='b', label='Lick Time', marker= '|', s=4)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('First spike time sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_firstspike_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Go Cue') 

        # first lick time vs first spike time
        ax = fig.add_subplot(gs[0, 6:8])
        lick_time = session_df_curr['reward_outcome_time'].values - session_df_curr['goCue_start_time'].values
        lick_time[session_df_curr['animal_response'].values==2] = np.nan

        ax.scatter(first_spike_times, lick_time, c='k', marker='o', s=3)
        ax.set_xlabel('First Spike Time (s)')
        ax.set_ylabel('First Lick Time (s)')

        plt.suptitle(f'Session {session}, Unit {unit_id}')
        plt.tight_layout()
        fig.savefig(os.path.join(save_path, f'{session}_{unit_id}_burst_selected.pdf'), dpi=300)
        plt.close(fig)

    # print(f'{session} Combining PDFs...')
    # combine_pdf_big(save_path, os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], f'{session}_bursting.pdf'))
    print(f'{session} Done!')


# %%
if __name__ == '__main__':
    dfs = [pd.read_csv('/root/capsule/code/data_management/session_assets.csv'),
        pd.read_csv('/root/capsule/code/data_management/hopkins_session_assets.csv')]
    df = pd.concat(dfs)
    session_list = df['session_id'].values.tolist()
    session_list = [session for session in session_list if str(session).startswith('behavior') and 'ZS' not in session]
    def save_process(session, data_type):
        try:
            burst_analysis(session, data_type, units = None)
        except Exception as e:
            print(f'Error processing session {session} for data type {data_type}: {e}')
    Parallel(n_jobs=4)(delayed(save_process)(session, 'curated') for session in session_list)
