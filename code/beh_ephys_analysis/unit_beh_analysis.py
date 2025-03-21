# %%
import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
from harp.clock import decode_harp_clock, align_timestamps_to_anchor_points
from open_ephys.analysis import Session
import datetime
from aind_ephys_rig_qc.temporal_alignment import search_harp_line
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from scipy.io import loadmat
from scipy.stats import zscore
import ast
from utils.plot_utils import combine_pdf_big

from open_ephys.analysis import Session
from pathlib import Path
import glob

import json
import seaborn as sns
from PyPDF2 import PdfMerger
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import re
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from hdmf_zarr.nwb import NWBZarrIO
from utils.beh_functions import session_dirs, parseSessionID, load_model_dv, makeSessionDF, get_session_tbl, get_unit_tbl, get_history_from_nwb
from utils.ephys_functions import*
import pandas as pd
import pickle
import scipy.stats as stats
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial
import time

def plot_unit_beh_session(session, data_type = 'curated', align_name = 'go_cue', curate_time=True, 
                        model_name = 'stan_qLearning_5params', 
                        formula = 'spikes ~ 1 + outcome + choice + Qchosen',
                        pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
                        units  = None):
    # %%
    # load behavior data
    session_dir = session_dirs(session, model_name = model_name) 
    session_df = makeSessionDF(session, model_name = model_name)
    tblTrials = get_session_tbl(session)
    pdf_dir = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], align_name)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
    # remove all files in pdf_dir if exists
    if os.path.exists(pdf_dir):
        files = glob.glob(pdf_dir + '/*')
        if len(files) > 0:
            for f in files:
                os.remove(f)      
    # %%
    qm_dir = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_dir, 'r') as f:
        qm = json.load(f)

    # %% load units 
    unit_tbl = get_unit_tbl(session, data_type)

    # %%
    fs = 14
    fsLegend = 8

    # %%
    colors = ["blue", "white", "red"]
    custom_cmap = LinearSegmentedColormap.from_list("blue_white_red", colors)
    
    def plot_unit(unit_id):
        unit_drift = load_drift(session, unit_id, data_type=data_type)
        spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]
        qc_pass = unit_tbl.query('unit_id == @unit_id')['default_qc'].values[0]
        opto_pass = unit_tbl.query('unit_id == @unit_id')['opto_pass'].values[0]
        session_df_curr = session_df.copy()
        spike_times_curr = spike_times.copy()
        tblTrials_curr = tblTrials.copy()
        if curate_time:
            if unit_drift is not None:
                if unit_drift['ephys_cut'][0] is not None:
                    spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                    session_df_curr = session_df_curr[session_df_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                    tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
                if unit_drift['ephys_cut'][1] is not None:
                    spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                    session_df_curr = session_df_curr[session_df_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                    tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        if len(session_df_curr) == 0:
            # return None and exit function
            print(f'No session data for unit {unit_id}')
            return None
        else:
            print(f'Plotting unit {unit_id}')      
            if align_name == 'go_cue':
                align_time = session_df_curr['go_cue_time'].values
                align_time_all = tblTrials_curr['goCue_start_time'].values
            elif align_name == 'response':
                align_time = session_df_curr['choice_time'].values
                align_time_all = tblTrials_curr['reward_outcome_time'].values
            spike_matrix, slide_times = get_spike_matrix(spike_times_curr, align_time, 
                                                        pre_event=pre_event, post_event=post_event, 
                                                        binSize=binSize, stepSize=stepSize)
            spike_matrix_LM, slide_times_LM = get_spike_matrix(spike_times_curr, align_time, 
                                                        pre_event=-2, post_event=2.5, 
                                                        binSize=0.5, stepSize=0.2)
            spike_matrix_all, slide_times = get_spike_matrix(spike_times_curr, align_time_all, 
                                                        pre_event=pre_event, post_event=post_event, 
                                                        binSize=binSize, stepSize=stepSize)

            fig = plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(2, 7, height_ratios=[3, 1], wspace=0.35, hspace=0.2)
            # plot session
            ax = fig.add_subplot(gs[0, 0]) 
            choice_history, reward_history, p_reward, autowater_offered, random_number, trial_time = get_history_from_nwb(session_df_curr)
            _, axes = plot_foraging_session(  # noqa: C901
                                            choice_history,
                                            reward_history,
                                            p_reward = p_reward,
                                            autowater_offered = autowater_offered,
                                            ax = ax,
                                            legend=False,
                                            vertical=True,
                                            ) 
            for ax in axes:
                ax.set_ylim(0, len(session_df_curr))
            ax.set_ylim(0, len(session_df_curr))
            # from start to end
            ax = fig.add_subplot(gs[0, 1])  
            df = align.to_events(spike_times_curr, align_time, (pre_event, post_event), return_df=True)
            plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
            ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
            ax.set_xlim(pre_event, post_event)
            ax.set_ylabel('Trial number')
            ax.tick_params(axis='both', which='major')
            ax.set_ylim(0, len(session_df_curr))


            # waveform
            ax = fig.add_subplot(gs[1, 0])  
            waveform = unit_tbl.query('unit_id == @unit_id')['waveform_mean'].values[0]
            peakChannel = np.argmin(np.min(waveform, axis=0))
            peakWaveform = waveform[:,peakChannel]
            peakSD = unit_tbl.query('unit_id == @unit_id')['waveform_mean'].values[0][:,peakChannel]
            timeWF = np.array(range(len(peakWaveform)))-90
            ax.plot(timeWF, peakWaveform, color = 'k')
            ax.fill_between(timeWF, peakWaveform - peakSD/np.sqrt(499), peakWaveform + peakSD/np.sqrt(499), color = 'k', alpha = 0.1)
            ax.axhline(y=0, color = 'r', ls = '--')
            ax.set_xlabel('Time (ms)', fontsize = fs)
            ax.set_ylabel(r'$\mu$-Plot')

            # reward and no reward
            outcome_int = [int(item) for item in session_df_curr['outcome'].tolist()]
            bins = [-1, 0.5, 1.5]
            labels = ['no reward', 'reward']
            fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                            spike_matrix, 
                                            slide_times,
                                            align_time, 
                                            outcome_int, # sorted by certain value
                                            bins,
                                            labels, 
                                            custom_cmap,
                                            fig,
                                            gs[0, 2],
                                            tb=pre_event,
                                            tf=post_event,
                                            )
            ax1.set_title('Reward vs No Reward', fontsize = fs+2)

            # left and right
            side_int = [int(item) for item in session_df_curr['choice'].tolist()]
            bins = [-1.5, 0.5, 1.5]
            labels = ['left', 'right']
            fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                            spike_matrix,
                                            slide_times, 
                                            align_time, 
                                            side_int, # sorted by certain value
                                            bins,
                                            labels,
                                            custom_cmap,
                                            fig,
                                            gs[0, 3],
                                            tb=pre_event,
                                            tf=post_event,
                                            )
            ax1.set_title('Right vs Left', fontsize = fs+2)

            # rpe
            target_var = 'pe'
            bin_counts = 4
            bins = np.quantile(session_df_curr[target_var].values, np.linspace(0, 1, bin_counts+1))
            # bins = [-1.0001, -0.5, 0, 0.5, 1.0001]
            bins[0] = bins[0] - 0.0001
            bins[-1] = bins[-1] + 0.0001
            labels = ['1', '2', '3', '4']
            
            fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                            spike_matrix, 
                                            slide_times,
                                            align_time,  
                                            session_df_curr[target_var].values, # sorted by certain value
                                            bins,
                                            labels,
                                            custom_cmap,
                                            fig,
                                            gs[0, 4],
                                            tb=pre_event,
                                            tf=post_event,
                                            )
            # ax.set_yticks([])
            # ax.set_ylabel(label, fontsize = fs)
            ax1.set_title(target_var, fontsize = fs+2)


            # Qchosen
            target_var = 'Qchosen'
            bin_counts = 4
            bins = np.quantile(session_df_curr[target_var].values, np.linspace(0, 1, bin_counts+1))
            # bins = [-1.0001, -0.5, 0, 0.5, 1.0001]
            bins[0] = bins[0] - 0.0001
            bins[-1] = bins[-1] + 0.0001
            labels = ['1', '2', '3', '4']
            fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                            spike_matrix, 
                                            slide_times,
                                            align_time, 
                                            session_df_curr[target_var].values, # sorted by certain value
                                            bins,
                                            labels,
                                            custom_cmap,
                                            fig,
                                            gs[0, 5],
                                            tb=pre_event,
                                            tf=post_event,
                                            )
            ax1.set_title(target_var, fontsize = fs+2)

            # stay vs switch
            target_var = 'svs'
            # bins = np.quantile(session_df_curr[target_var].values, np.linspace(0, 1, bin_counts+1))
            bins = [-1.0001, 0.5, 1.0001]
            labels = ['stay', 'switch']
            # ax = fig.add_subplot(gs[1, 1])
            fig, ax = plot_rate(
                                spike_matrix,
                                slide_times, 
                                session_df_curr[target_var].values,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 1],
                                tb=pre_event,
                                tf=post_event,
                                )

            ax.set_yticks([])
            ax.set_title(target_var, fontsize = fs+2)

            # right rwd vs no rwd
            target_var = 'outcome'
            spike_matrix_curr = spike_matrix[session_df_curr['choice'].values == 1, :]
            focus_var = session_df_curr[session_df_curr['choice'].values == 1][target_var].values
            bins = [-1.0001, 0.5, 1.0001]
            labels = ['no rwd', 'rwd']
            fig, ax = plot_rate(
                                spike_matrix_curr,
                                slide_times,
                                focus_var,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 2],
                                tb=pre_event,
                                tf=post_event,
                                )
            ax.set_title('Right: rwd nrwd', fontsize = fs+2)

            # left rwd vs no rwd
            target_var = 'outcome'
            spike_matrix_curr = spike_matrix[session_df_curr['choice'].values == 0, :]
            focus_var = session_df_curr[session_df_curr['choice'].values == 0][target_var].values
            bins = [-1.0001, 0.5, 1.0001]
            labels = ['no rwd', 'rwd']
            fig, ax = plot_rate(
                                spike_matrix_curr,
                                slide_times,
                                focus_var,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 3],
                                tb=pre_event,
                                tf=post_event,
                                )
            ax.set_title('Left: rwd nrwd', fontsize = fs+2)

            # go vs miss
            map_value = tblTrials_curr['animal_response'].values!=2
            bins = [-0.5, 0.5, 1.5]
            labels = ['miss', 'go']
            fig, ax = plot_rate(
                                spike_matrix_all,
                                slide_times,
                                map_value,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 4],
                                tb=pre_event,
                                tf=post_event,
                                )


            
            if len(session_df_curr) > 100 and np.sum((spike_times_curr>=session_df_curr['go_cue_time'].values[0]) & (spike_times_curr<=session_df_curr['go_cue_time'].values[-1]))/(session_df_curr['go_cue_time'].values[-1] - session_df_curr['go_cue_time'].values[0]) > 0.1:
                # rwd history
                # fr with regression with rwd history
                target_var = 'outcome'
                vector = session_df_curr[target_var].values
                align_time = session_df_curr['choice_time'].values
                _, events_id, _ = align.to_events(spike_times_curr, align_time, (0, 1.5), return_df=False) 
                spike_counts = [np.sum(events_id==curr_id) for curr_id in range(len(align_time))]
                spike_counts = stats.zscore(np.array(spike_counts))
                trials_back = [0, 2]
                coeffs, pvals, tvals, conf_int = regression_rwd(spike_counts, vector, trials_back = trials_back)

                ax = fig.add_subplot(gs[1, 5])
                ax.plot(range(trials_back[0], trials_back[1] + 1), coeffs, c = 'k', lw = 2)
                ax.fill_between(range(trials_back[0], trials_back[1] + 1), conf_int[:, 0], conf_int[:, 1], color = 'k', alpha = 0.25, edgecolor = None)
                ax.axhline(y=0, color = 'r', ls = '--')
                ax.scatter(np.array(range(trials_back[0], trials_back[1] + 1))[pvals<0.05], coeffs[pvals<0.05], c = 'r', s = 10, zorder = 2)
                ax.set_title('Spikes~rwd history', fontsize = fs+2)
                ax.set_xlabel('Trials back')

                # only on left trials
                trials_back = [0, 2]
                if np.sum(session_df_curr['choice'].values == 0) >= 40:
                    coeffs, pvals, tvals, conf_int = regression_rwd(spike_counts, vector, trials_back = trials_back, sub_selection=session_df_curr['choice'].values == 0)
                    ax = fig.add_subplot(gs[1, 6])
                    ax.plot(range(trials_back[0], trials_back[1] + 1), coeffs, c = 'm', lw = 2, label = 'left')
                    ax.fill_between(range(trials_back[0], trials_back[1] + 1), conf_int[:, 0], conf_int[:, 1], color = 'm', alpha = 0.25, edgecolor = None)
                    ax.scatter(np.array(range(trials_back[0], trials_back[1] + 1))[pvals<0.05], coeffs[pvals<0.05], c = 'r', s = 10)
                    ax.axhline(y=0, color = 'r', ls = '--')
                    ax.set_title('Spikes~rwd hist L/R', fontsize = fs+2)
                    ax.set_xlabel('Trials back')
                    ax.legend()

                # only on right trials
                if np.sum(session_df_curr['choice'].values == 1) >= 40:
                    coeffs, pvals, tvals, conf_int = regression_rwd(spike_counts, vector, trials_back = trials_back, sub_selection=session_df_curr['choice'].values == 1)
                    ax.plot(range(trials_back[0], trials_back[1] + 1), coeffs, c = 'c', lw = 2, label = 'right')
                    ax.fill_between(range(trials_back[0], trials_back[1] + 1), conf_int[:, 0], conf_int[:, 1], color = 'c', alpha = 0.25, edgecolor = None)
                    ax.scatter(np.array(range(trials_back[0], trials_back[1] + 1))[pvals<0.05], coeffs[pvals<0.05], c = 'r', s = 10)
                    ax.axhline(y=0, color = 'r', ls = '--')
                    ax.set_title('Spikes~rwd hist L/R', fontsize = fs+2)
                    ax.set_xlabel('Trials back')
                    ax.legend()

                # plot regresssions
                gs = gridspec.GridSpec(3, 7, height_ratios=[1, 1, 1], wspace=0.3, hspace=0.3)
                ax = fig.add_subplot(gs[0,-1])
                try: 
                    regressors, TvCurrU, PvCurrU, EvCurrU = fitSpikeModelG(session_df_curr, spike_matrix_LM, formula)
                    TvCurrUSig = TvCurrU.copy()
                    TvCurrUSig[PvCurrU>=0.05] = np.nan
                    cmap = plt.get_cmap('viridis')
                    colors = cmap(np.linspace(0, 1, len(regressors)))
                    for regress in range(1, len(regressors)):
                        ax.plot(slide_times_LM, TvCurrU[:, regress], lw = 2, color = colors[regress,], label = regressors[regress])
                        ax.plot(slide_times_LM, TvCurrUSig[:, regress], lw = 4, color = colors[regress,])
                    ax.legend(fontsize = fsLegend)
                    ax.set_xlabel(f'Time from {align_name} (s)')
                    ax.set_title('T-stats', fontsize = fs)

                    ax = fig.add_subplot(gs[1,-1])
                    for regress in range(1, len(regressors)):
                        ax.plot(slide_times_LM, -np.log10(PvCurrU[:, regress]), lw = 1, color = colors[regress,], label = regressors[regress])

                    plt.axhline(y = -np.log10(0.05), color='r', ls = '--')
                    ax.legend(fontsize = fsLegend)
                    ax.set_xlabel(f'Time from {align_name} (s)')
                    ax.set_title('p-value', fontsize = fs)
                except:
                    print(f'Failed to fit model for unit {unit_id}')
                plt.suptitle(f'Unit{str(unit_id)} Aligned to {align_name} default qc: {qc_pass} maybe opto: {opto_pass}', fontsize = 20) 
            # plt.tight_layout()  
            return fig

# %%
    log_record_file = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], align_name, f'{session}_unit_beh.log')
    def process(unit_id): 
        try:
            fig = plot_unit(unit_id) 
            if fig is not None:
                fig.savefig(fname=os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], align_name, f'unit_{unit_id}_goCue.pdf'))
            plt.close(fig)
            # write to log
            with open(log_record_file, 'a') as f:
                f.write(f'Unit {unit_id} plotted\n')
            # pause for 1 second
        except:
            with open(log_record_file, 'a') as f:
                f.write(f'Unit {unit_id} failed\n')
        time.sleep(1)

    # def process(unit_id):
    #     fig = plot_unit(unit_id) 
    #     if fig is not None: 
    #         fig.savefig(fname=os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], f'unit_{unit_id}_{align_name}.pdf'))
    #     plt.close(fig)
    #     time.sleep(1)



    # with Pool(processes=4) as pool:  # Ensures cleanup
    #     result = pool.map(process, unit_tbl['unit_id'].values)
    if units is None:
        units= unit_tbl['unit_id'].values

    Parallel(n_jobs=-1)(
        delayed(process)(unit_id)
        for unit_id in units
    )
    # for unit_id in unit_tbl['unit_id'].values:
    #     process(unit_id)

    output_pdf = os.path.join(session_dirs(session)[f'ephys_dir_{data_type}'],f'{session}_unit_beh_{align_name}.pdf')

    if os.path.exists(pdf_dir):
        print(f'Combining {session}')
        combine_pdf_big(pdf_dir, output_pdf)
    
    plt.close('all')


if __name__ == '__main__': 

    df = pd.read_csv('/root/capsule/code/data_management/session_assets.csv')
    session_ids = df['session_id'].values
    model_name = 'stan_qLearning_5params'
    data_type = 'curated'
    curate_time = True
    align_name = 'go_cue'
    formula = 'spikes ~ 1 + outcome + choice + Qchosen'
    for session in session_ids:
        print(session)
        session_dir = session_dirs(session)
        if os.path.exists(os.path.join(session_dir['beh_fig_dir'], f'{session}.nwb')):
            if session_dir['curated_dir_curated'] is not None:
                if not os.path.exists(os.path.join(session_dirs(session)['ephys_dir_curated'],f'{session}_unit_beh_{align_name}.pdf')):
                    plot_unit_beh_session(session, data_type = 'curated', align_name = align_name, curate_time=curate_time, 
                                model_name = model_name, formula=formula,
                                pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
                                units=None)
            elif session_dir['curated_dir_raw'] is not None:
                if not os.path.exists(os.path.join(session_dirs(session)['ephys_dir_raw'],f'{session}_unit_beh_{align_name}.pdf')):
                    plot_unit_beh_session(session, data_type = 'raw', align_name = align_name, curate_time=curate_time, 
                                    model_name = model_name, formula=formula,
                                    pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
                                    units=None)
    # session = 'behavior_717121_2024-06-15_10-00-58'
    # plot_unit_beh_session(session, data_type = data_type, align_name = align_name, curate_time=curate_time, 
    #                     model_name = model_name, formula=formula,
    #                     pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #                     units=None)

