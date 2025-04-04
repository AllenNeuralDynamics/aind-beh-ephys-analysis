# %%
# %matplotlib inline
import os
import sys

from scipy.sparse import data
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
from session_preprocessing import ephys_opto_preprocessing
from opto_tagging import opto_plotting_session
from open_ephys.analysis import Session
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
from aind_ephys_utils import align
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
import pickle
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import load_nwb
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
# def load_motion_info(folder):
from spikeinterface.preprocessing.motion import load_motion_info
from PIL import Image

# %%
def plot_session_opto_drift(session, data_type, plot=True, update_csv = False):
    session_dir = session_dirs(session)

    # %%
    # ephys_opto_preprocessing(session, 'curated', 'soma')

    # %%
    motion_info = load_motion_info(f'/root/capsule/data/{session}_sorted/preprocessed/motion/experiment1_Record Node 104#Neuropix-PXI-100.ProbeA_recording1')

    # %%
    # sampling
    bin_sampling = 10
    temp_bins_sampling = np.arange(motion_info['motion'].temporal_bins_s[0][0], motion_info['motion'].temporal_bins_s[0][-1], bin_sampling)
    probe_location = np.linspace(2500, 0,  96)

    drift_sampling = np.zeros(shape=(len(probe_location), len(temp_bins_sampling)))
    for i, t in enumerate(temp_bins_sampling): 
        for j, p in enumerate(probe_location):
            drift_sampling[j, i] = motion_info['motion'].get_displacement_at_time_and_depth([t], [p])

    # %%
    # fast dynamics
    bin_short = 100
    temp_bins = np.arange(motion_info['motion'].temporal_bins_s[0][0], motion_info['motion'].temporal_bins_s[0][-1], bin_short)
    drift = np.zeros(shape=(len(probe_location), len(temp_bins)))

    drift = np.full((len(probe_location), len(temp_bins)), np.nan)
    for i, t in enumerate(temp_bins): 
        for j, p in enumerate(probe_location):
            curr_ind = np.where((temp_bins_sampling >= t-bin_short/2) & (temp_bins_sampling < t+bin_short/2))[0]
            if curr_ind.size > 0:
                drift[j, i] = np.mean(drift_sampling[j, curr_ind])



    # %%
    # slow dynamics
    bin_long = 300 # seconds
    temp_bins_slow = temp_bins

    drift_slow_pre = np.full((len(probe_location), len(temp_bins_slow)), np.nan)
    drift_slow_post = np.full((len(probe_location), len(temp_bins_slow)), np.nan)
    drift_slow = np.full((len(probe_location), len(temp_bins_slow)), np.nan)
    for i, t in enumerate(temp_bins_slow): 
        for j, p in enumerate(probe_location):
            curr_ind = np.where((temp_bins_sampling >= t-bin_long) & (temp_bins_sampling < t))[0]
            if curr_ind.size > 0:
                drift_slow_pre[j, i] = np.mean(drift_sampling[j, curr_ind])
            curr_ind = np.where((temp_bins_sampling > t) & (temp_bins_sampling <= t+bin_long))[0]
            if curr_ind.size > 0:
                drift_slow_post[j, i] = np.mean(drift_sampling[j, curr_ind])
            curr_ind = np.where((temp_bins_sampling >= t-bin_long/2) & (temp_bins_sampling <= t+bin_long/2))[0]
            if curr_ind.size > 0:
                drift_slow[j, i] = np.mean(drift_sampling[j, curr_ind])
    drift_slow_pre[:,0] = drift_slow_pre[:,1]
    drift_slow_post[:,-1] = drift_slow_post[:,-2]

    # %%
    range_max = np.max(np.abs(drift))
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    plt.subplot(1, 5, 1)
    sns.heatmap(drift, cmap='seismic', center=0, vmin=-range_max, vmax=range_max)
    plt.xticks(np.linspace(0, len(temp_bins), 10), [f"{x:.1f}" for x in list(np.linspace(temp_bins[0], temp_bins[-1], 10))]);
    plt.yticks(np.linspace(0, len(probe_location), 5), [f"{x:.1f}" for x in list(np.linspace(probe_location[0], probe_location[-1], 5))]);
    plt.xlabel('Time (s)')
    plt.ylabel('Depth (um)')
    plt.title('Fast drift')

    plt.subplot(1, 5, 2)
    sns.heatmap(drift_slow, cmap='seismic', center=0, vmin=-range_max, vmax=range_max)
    plt.xticks(np.linspace(0, len(temp_bins_slow), 10), [f"{x:.1f}" for x in list(np.linspace(temp_bins_slow[0], temp_bins_slow[-1], 10))]);
    plt.yticks(np.linspace(0, len(probe_location), 5), [f"{x:.1f}" for x in list(np.linspace(probe_location[0], probe_location[-1], 5))]);
    plt.xlabel('Time (s)')
    plt.ylabel('Depth (um)')
    plt.title('Slow drift')

    plt.subplot(1, 5, 3)
    range_max = np.max(np.abs(np.diff(drift)))
    sns.heatmap(np.diff(drift), cmap='seismic', center=0, vmin=-range_max, vmax=range_max)
    plt.xticks(np.linspace(0, len(temp_bins), 10), [f"{x:.1f}" for x in list(np.linspace(temp_bins[0], temp_bins[-1], 10))]);
    plt.yticks(np.linspace(0, len(probe_location), 5), [f"{x:.1f}" for x in list(np.linspace(probe_location[0], probe_location[-1], 5))]);
    plt.xlabel('Time (s)')
    plt.ylabel('Depth (um)')
    plt.title('Fast drift derivative')

    range_max = np.max(np.abs(drift_slow_post - drift_slow_pre))
    plt.subplot(1, 5, 4)
    sns.heatmap(drift_slow_post - drift_slow_pre, cmap='seismic', center=0, vmin=-range_max, vmax=range_max)
    plt.xticks(np.linspace(0, len(temp_bins_slow), 10), [f"{x:.1f}" for x in list(np.linspace(temp_bins_slow[0], temp_bins_slow[-1], 10))]);
    plt.yticks(np.linspace(0, len(probe_location), 5), [f"{x:.1f}" for x in list(np.linspace(probe_location[0], probe_location[-1], 5))]);
    plt.xlabel('Time (s)')
    plt.ylabel('Depth (um)')
    plt.title('Slow drift derivative')

    drift_diff = drift - drift_slow_pre
    range_max = np.max(np.abs(drift_diff))
    plt.subplot(1, 5, 5)
    sns.heatmap(drift_diff, cmap='seismic', center=0, vmin=-range_max, vmax=range_max)
    plt.xticks(np.linspace(0, len(temp_bins_slow), 10), [f"{x:.1f}" for x in list(np.linspace(temp_bins_slow[0], temp_bins_slow[-1], 10))]);
    plt.yticks(np.linspace(0, len(probe_location), 5), [f"{x:.1f}" for x in list(np.linspace(probe_location[0], probe_location[-1], 5))]);
    plt.xlabel('Time (s)')
    plt.ylabel('Depth (um)')
    plt.title('Fast drift - slow drift')

    plt.tight_layout()
    plt.savefig(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_motion_drift.pdf'))

    # %%
    if os.path.exists(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_metrics.pkl')):
        # load from pickle
        with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_metrics.pkl'), 'rb') as f:
            unit_tbl = pickle.load(f)
    else: 
        unit_tbl = opto_plotting_session(session, data_type, 'soma', plot=False, resp_thresh=0.3, lat_thresh=0.025) 
        unit_tbl.to_pickle(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_metrics.pkl'))


    # %%
    sorting_analyzer = si.load_sorting_analyzer(session_dir[f'postprocessed_dir_{data_type}']) 
    sorting = si.load_extractor(session_dir[f'curated_dir_{data_type}'])

    # %%
    unit_locations = sorting_analyzer.get_extension('unit_locations').get_data(outputs="by_unit")
    spike_amplitudes = sorting_analyzer.get_extension('spike_amplitudes').get_data(outputs="by_unit")[0]

    # # %%
    # spike_pcs = sorting_analyzer.get_extension('principal_components')

    # # %%
    # pcs_unit = spike_pcs.get_projections_one_unit(10)

    # %%
    def plot_drift(unit_id, plot=plot):
        spike_times = unit_tbl[unit_tbl['unit_id'] == unit_id]['spike_times'].values[0]
        spike_amplitude = spike_amplitudes[unit_id]
        # firing rate
        temp_bins_spike = np.zeros(len(temp_bins)+1)
        temp_bins_spike[1:] = temp_bins+bin_short/2
        temp_bins_spike[0] = temp_bins[0] - bin_short/2
        temp_bins_slow_spike = temp_bins_slow
        spike_counts_fast, _ = np.histogram(spike_times, bins=temp_bins_spike)
        spike_counts_fast = spike_counts_fast/bin_short
        spike_counts_slow_pre = np.full(len(temp_bins_slow_spike), np.nan)
        spike_counts_slow_post = np.full(len(temp_bins_slow_spike), np.nan)
        spike_counts_slow = np.full(len(temp_bins_slow_spike), np.nan)
        for i, t in enumerate(temp_bins_slow_spike):
            if t - bin_long> temp_bins[0]:
                spike_counts_slow_pre[i] = np.sum((spike_times >= t - bin_long) & (spike_times < t))
                spike_counts_slow_pre[i] = spike_counts_slow_pre[i] / bin_long
            if t + bin_long < temp_bins[-1]:
                spike_counts_slow_post[i]= np.sum((spike_times >= t) & (spike_times < t + bin_long))
                spike_counts_slow_post[i] = spike_counts_slow_post[i] / bin_long
            if t - bin_long/2 > temp_bins[0] and t + bin_long/2 < temp_bins[-1]:
                spike_counts_slow[i] = np.sum((spike_times >= t - bin_long/2) & (spike_times < t + bin_long/2))
                spike_counts_slow[i] = spike_counts_slow[i] / bin_long
            
        # drift no need to calculate again
        closest_ybin = np.argmin(np.abs(probe_location - unit_locations[unit_id][1]))

        # amplitude
        amplitude_fast = np.full((len(temp_bins)), np.nan)
        amplitude_slow_pre = np.full((len(temp_bins_slow)), np.nan)
        amplitude_slow_post = np.full((len(temp_bins_slow)), np.nan)
        amplitude_slow = np.full((len(temp_bins)), np.nan)
        
        for i, t in enumerate(temp_bins):
            inds = np.where((spike_times >= t-bin_short) & (spike_times < t+bin_short))[0]
            if inds.size > 5:
                temp = spike_amplitude[inds]
                cut_off = np.percentile(temp, [5, 95])
                temp = temp[(temp >= cut_off[0]) & (temp <= cut_off[1])]
                if temp.size > 0:
                    amplitude_fast[i] = np.mean(temp)

        for i, t in enumerate(temp_bins_slow):
            inds = np.where((spike_times >= t-bin_long - bin_short) & (spike_times < t - bin_short))[0]
            if inds.size > 5:
                temp = spike_amplitude[inds]
                cut_off = np.percentile(temp, [5, 95])
                temp = temp[(temp >= cut_off[0]) & (temp <= cut_off[1])]
                if temp.size > 0:
                    amplitude_slow_pre[i] = np.mean(temp)
            inds = np.where((spike_times >= t + bin_short) & (spike_times < t + bin_short + bin_long))[0]
            if inds.size > 5:
                temp = spike_amplitude[inds]
                cut_off = np.percentile(temp, [5, 95])
                temp = temp[(temp >= cut_off[0]) & (temp <= cut_off[1])]
                if temp.size > 0:
                    amplitude_slow_post[i] = np.mean(temp)
            
            inds = np.where((spike_times >= t-bin_long/2) & (spike_times < t+bin_long/2))[0]
            if inds.size > 5:
                temp = spike_amplitude[inds]
                cut_off = np.percentile(temp, [5, 95])
                temp = temp[(temp >= cut_off[0]) & (temp <= cut_off[1])]
                if temp.size > 0:
                    amplitude_slow[i] = np.mean(temp)

        ## linear regression for slower dynamics
        # Combine x and y into a feature matrix
        X_lr = np.column_stack((drift_slow[closest_ybin], amplitude_slow))

        # Create and fit the linear regression model
        model = LinearRegression()
        nan_inds_slow = np.where(~np.isnan(spike_counts_slow) & np.all(~np.isnan(X_lr), axis=1))[0]

        model.fit(X_lr[nan_inds_slow], spike_counts_slow[nan_inds_slow])
        r_squared = model.score(X_lr[nan_inds_slow], spike_counts_slow[nan_inds_slow])

        # sd.mean
        sd = np.std(spike_counts_slow[nan_inds_slow])/np.nanmean(spike_counts_slow)
        ## random forest model for fast dynamics
        # Feature matrix
        z_rf_fast = np.abs((spike_counts_fast - spike_counts_slow_pre)/(0.5*(spike_counts_fast + spike_counts_slow_pre))) # firing rate change
        X_rf_fast = np.abs(np.column_stack((drift[closest_ybin]-drift_slow_pre[closest_ybin], # drift
                            amplitude_fast - amplitude_slow_pre,  # amplitude change
                            (drift[closest_ybin]-drift_slow_pre[closest_ybin])*spike_counts_slow))) # interaction between drift and firing rate
        nan_inds = np.where(~np.isnan(z_rf_fast) & np.all(~np.isnan(X_rf_fast), axis=1))[0]
        # Train Random Forest model
        model_rf_fast = RandomForestRegressor(n_estimators=10, random_state=42)
        model_rf_fast.fit(X_rf_fast[nan_inds], z_rf_fast[nan_inds])
        # Predictions
        z_pred_rf_fast = model_rf_fast.predict(X_rf_fast)
        # Evaluate model
        r2_rf_fast = r2_score(z_rf_fast[nan_inds], z_pred_rf_fast[nan_inds])
        print(f"Random Forest R² Score fast: {r2_rf_fast:.4f}")
        # Feature Importance
        importances_fast = model_rf_fast.feature_importances_
        print(f"Feature Importance fast - motion: {importances_fast[0]:.2f}, amp: {importances_fast[1]:.2f}, motion*fr: {importances_fast[2]:.2f}")
        
        ## Random forest mode for slower dynamics
        z_rf = spike_counts_slow_post - spike_counts_slow_pre/(0.5*(spike_counts_slow_post+spike_counts_slow_pre)) # firing rate change
        X_rf = np.column_stack((drift_slow_post[closest_ybin]-drift_slow_pre[closest_ybin], # drift
                            amplitude_slow_post - amplitude_slow_pre,  # amplitude change
                            (drift_slow_post[closest_ybin]-drift_slow_pre[closest_ybin])*spike_counts_slow)) # interaction between drift and firing rate
        nan_inds_slow_rf = np.where(~np.isnan(z_rf) & np.all(~np.isnan(X_rf), axis=1))[0]
        # Train Random Forest model
        model_rf = RandomForestRegressor(n_estimators=10, random_state=42)
        model_rf.fit(X_rf[nan_inds_slow_rf], z_rf[nan_inds_slow_rf])
        # Predictions
        z_pred_rf = model_rf.predict(X_rf)
        # Evaluate model
        r2_rf = r2_score(z_rf[nan_inds_slow_rf], z_pred_rf[nan_inds_slow_rf])
        print(f"Random Forest R² Score slow: {r2_rf:.4f}")
        # Feature Importance
        importances = model_rf.feature_importances_
        print(f"Feature Importance slow - motion: {importances[0]:.2f}, amp: {importances[1]:.2f}, motion*fr: {importances[2]:.2f}")

        ## linear regression with for faster timescale
        model_fast = LinearRegression()
        model_fast.fit(X_rf_fast[nan_inds], z_rf_fast[nan_inds])
        z_pred_fast = model_fast.predict(X_rf_fast[nan_inds])
        r_squared_fast = model_fast.score(X_rf_fast[nan_inds], z_rf_fast[nan_inds])


        
        if plot: 
            fig = plt.figure(figsize=(20, 20))
            plt.rcParams.update({'font.size': 8})
            gs = gridspec.GridSpec(5, 3)
            ax = plt.subplot(gs[0, 0])
            plt.hist(spike_times, bins=temp_bins);
            plt.title('Firing rate')

            ax = plt.subplot(gs[1, 0])
            plt.plot(temp_bins, spike_counts_fast, label='fast')
            plt.plot(temp_bins_slow_spike, spike_counts_slow, label='slow', c = 'r')
            plt.legend()

            ax = plt.subplot(gs[2, 0])
            plt.plot(temp_bins, (spike_counts_fast-spike_counts_slow_pre)/(0.5*(spike_counts_fast+spike_counts_slow_pre)), label='fast-slow')
            plt.plot(temp_bins_slow_spike, (spike_counts_slow_post - spike_counts_slow_pre)/(0.5*(spike_counts_slow_pre + spike_counts_slow_post)), label = 'diff(slow)', c='r')
            plt.legend()

            ax = plt.subplot(gs[3, 0])
            plt.plot(temp_bins, np.abs(spike_counts_fast - spike_counts_slow_pre)/(0.5*(spike_counts_fast+spike_counts_slow_pre)), label='abs(fast-slow)')
            plt.plot(temp_bins_slow_spike, np.abs(spike_counts_slow_post - spike_counts_slow_pre)/(0.5*(spike_counts_slow_pre + spike_counts_slow_post)), label = 'abs(diff(slow))', c='r')
            plt.legend()
            # drift
            ax = plt.subplot(gs[0, 1])
            plt.title('Esitimated motion')
            ax = plt.subplot(gs[1, 1])
            plt.plot(temp_bins, drift[closest_ybin, :], label='fast')
            plt.plot(temp_bins_slow, drift_slow[closest_ybin, :], label='slow', c = 'r')
            plt.plot(temp_bins, np.zeros(len(temp_bins)), 'k--')
            plt.legend()
            # plt.plot(motion_info['motion'].get_displacement_at_time_and_depth(times_s, locations_um))
            ax = plt.subplot(gs[2, 1])
            plt.plot(temp_bins, (drift[closest_ybin, :] - drift_slow[closest_ybin,:]), label='fast-slow')
            plt.plot(temp_bins, drift_slow_post[closest_ybin, :] - drift_slow[closest_ybin,:], label='diff(slow)', c = 'r')
            plt.legend()


            ax = plt.subplot(gs[3, 1])
            plt.plot(temp_bins, np.abs(drift[closest_ybin, :] - drift_slow[closest_ybin,:]), label='abs(fast-slow)')
            plt.plot(temp_bins, np.abs(drift_slow_post[closest_ybin, :] - drift_slow[closest_ybin,:]), label='abs(diff(slow))', c = 'r')
            plt.legend()

            ax = plt.subplot(gs[0, 2])
            plt.scatter(spike_times, spike_amplitude, c='k', s=0.5, alpha=0.25)
            plt.title('Spike amplitude')
            plt.legend()


            ax = plt.subplot(gs[1, 2])
            plt.plot(temp_bins, amplitude_fast, label='fast')
            plt.plot(temp_bins, amplitude_slow, label='slow', c = 'r')
            plt.legend()

            ax = plt.subplot(gs[2, 2])
            plt.plot(temp_bins, amplitude_fast - amplitude_slow_pre, label='fast-slow')
            # ax = ax.twinx()
            plt.plot(temp_bins_slow, amplitude_slow_post - amplitude_slow_pre, label='diff(slow)', c='r')
            plt.legend()

            ax = plt.subplot(gs[3, 2])
            plt.plot(temp_bins, np.abs(amplitude_fast - amplitude_slow_pre), label='abs(fast-slow)')
            plt.plot(temp_bins_slow, np.abs(amplitude_slow_post - amplitude_slow_pre), label = 'abs(diff(slow))', c='r')
            plt.legend()

            gs_model = gridspec.GridSpecFromSubplotSpec(1, 5, gs[4, :])
            ax = plt.subplot(gs_model[0])
            plt.plot(temp_bins, spike_counts_slow, label='data', c='r')
            plt.plot(temp_bins[nan_inds_slow], model.predict(X_lr[nan_inds_slow]), label='prediction', c='b')
            plt.legend()
            plt.xlabel('Time (s)')
            plt.title(f"Slow FR: LR R²: {r_squared:.2f}")

            ax = plt.subplot(gs_model[1])
            plt.plot(temp_bins, z_rf, label='data', c='r')
            plt.plot(temp_bins, z_pred_rf, label='prediction', c='b')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.title(f"Slow diff(FR): RF R²: {r2_rf:.2f}")

            ax = plt.subplot(gs_model[2])
            plt.plot(temp_bins, z_rf_fast, label='data', c='r')
            plt.plot(temp_bins[nan_inds], z_pred_fast, label='prediction', c='b')
            plt.legend()
            plt.xlabel('Time (s)')
            plt.title(f"Fast abs(diff(FR)): LR R²: {r_squared_fast:.2f}")

            # # Print coefficients and intercept
            # print(f"Coefficients: {model.coef_}")
            # print(f"Intercept: {model.intercept_}")

            # # Predict new values
            # X_new = np.array([[6, 7], [7, 8]])
            # z_pred = model.predict(X_new)
            # print(f"Predicted values: {z_pred}")

            ax = plt.subplot(gs_model[3])
            plt.plot(temp_bins, z_rf_fast, label='data', c='r')
            plt.plot(temp_bins[nan_inds], z_pred_rf_fast[nan_inds], label='prediction', c='b')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.title(f"Fast abs(diff(FR)) : RF R²: {r2_rf:.2f}")

            ax = plt.subplot(gs_model[4])
            plt.hist(spike_counts_slow, bins=20, alpha=0.5, label='data', color='r')
            plt.title(f"SD/mean: {sd:.2f}")
            plt.xlabel('Time (s)')
            plt.legend()
    
            plt.suptitle(f'{unit_id} y loc {unit_locations[unit_id][1] :.2f} um')
            plt.rcParams.update({'font.size': 8})
            plt.tight_layout()

        return {'unit_id': unit_id,
                'r_squared_slow': r_squared,
                'r_squared_fast': r_squared_fast,
                'sd/mean': sd, 
                'r_squared_rf': r2_rf, 
                'importances': importances, 
                'regressors': ['drift', 'amplitude', 'drift*fr'],
                'ephys_cut': [np.nan, np.nan],
                'drift_unit': False}

    # %%
    units_to_plot = []
    p_thresh = 0.4
    lat_thresh = 0.02
    for i, row in unit_tbl.iterrows():
        if row['resp_p'] is not None:
            opto_pass = len(np.where((np.array(row['resp_p']) > p_thresh) & (np.array(row['resp_lat']) < lat_thresh) & (np.array(row['resp_lat']) > 0.004))[0]) > 0
            quality_pass = (row['isi_violations_ratio'] < 0.05) & (row['decoder_label'] != 'noise') & (row['decoder_label'] != 'artifact')
            if opto_pass & quality_pass:
                if np.where(np.array(row['resp_p']) > p_thresh)[0].size > 0:
                    units_to_plot.append(row['unit_id'])

    # %%
    drift_dir = os.path.join(session_dir['opto_dir_curated'], 'drift')
    os.makedirs(name=drift_dir, exist_ok=True)
    # os.mkdir(drift_dir, exist_ok=True)?
    opto_drift_tbl = pd.DataFrame()
    for unit in units_to_plot:                             
        unit_drift_dict = plot_drift(unit, plot=True)
        opto_drift_tbl = pd.concat([opto_drift_tbl, pd.DataFrame([unit_drift_dict])], ignore_index=True)
        if plot:
            plt.savefig(os.path.join(drift_dir, f'{unit}_drift.pdf'))
            plt.show()
    if update_csv:
        opto_drift_tbl.to_csv(os.path.join(session_dir['opto_dir_curated'], f'{session}_opto_drift_tbl.csv'))
    if plot:
        merge_pdfs(input_dir=drift_dir, output_filename=os.path.join(session_dir['opto_dir_curated'], f'{session}_drift.pdf'))
    return opto_drift_tbl

if __name__ == '__main__':
    session = 'behavior_751004_2024-12-19_11-50-37'
    data_type = 'curated' 
    plot_session_opto_drift(session, data_type)



