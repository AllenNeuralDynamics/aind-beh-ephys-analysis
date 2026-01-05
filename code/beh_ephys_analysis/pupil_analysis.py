# %%
# %%
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from utils.beh_functions import parseSessionID, session_dirs, get_unit_tbl, get_session_tbl, makeSessionDF
from utils.ephys_functions import plot_rate, get_spike_matrix, correlate_nan, load_drift
from utils.lick_utils import load_licks
from utils.photometry_utils import bin_timeseries_around_align
from utils.pupil_utils import load_pupil
from aind_ephys_utils import align
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib import colormaps
import pickle
import shutil
import datetime
import seaborn as sns
import matplotlib.cm as cm
from trimesh import load_mesh
from utils.plot_utils import combine_pdf_big
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from scipy.stats import pearsonr

# %%

def pupil_analysis_session(session, plot_licks=False):
    session_dir = session_dirs(session_id=session)
    beh_df = get_session_tbl(session)
    session_df = makeSessionDF(session)
    pupil_data = load_pupil(session)
    if pupil_data is None:
        print(f'No pupil data for session {session}')
        return
    lick_data = load_licks(session, plot=plot_licks)

    out, bins = bin_timeseries_around_align(
        ts={'time': pupil_data['pupil_times'], 'value': pupil_data['pupil_diameter_z']},
        align_times=session_df['go_cue_time'].values,
        step_size=0.1,
        bin_size=0.15,
        t_start=-1,
        t_stop=5
    )
    mat_licks, bins_licks = bin_timeseries_around_align(
        ts={'time': pupil_data['pupil_times'], 'value': pupil_data['pupil_diameter_z']},
        align_times=lick_data['lick_trains_all']['train_starts'],
        step_size=0.1,
        bin_size=0.15,
        t_start=-1,
        t_stop=5
    )
    mat_licks_L, bins_licks_L = bin_timeseries_around_align(
        ts={'time': pupil_data['pupil_times'], 'value': pupil_data['pupil_diameter_z']},
        align_times=lick_data['lick_trains_L']['train_starts'],
        step_size=0.1,
        bin_size=0.15,
        t_start=-1,
        t_stop=5
    )

    mat_licks_R, bins_licks_R = bin_timeseries_around_align(
        ts={'time': pupil_data['pupil_times'], 'value': pupil_data['pupil_diameter_z']},
        align_times=lick_data['lick_trains_R']['train_starts'],
        step_size=0.1,
        bin_size=0.15,
        t_start=-1,
        t_stop=5
    )

    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(3,4)
    colormap = LinearSegmentedColormap.from_list([0, 0.5, 1], ['blue', 'white', 'red'])
    _, ax = plot_rate(out, bins, session_df['svs'].values, [-1, 0.5, 2], ['stay', 'switch'], colormap, fig, gs[0,0])
    ax.set_title('stay vs switch')
    _, ax = plot_rate(out, bins, session_df['outcome'].values, [-1, 0.5, 2], ['no reward', 'reward'], colormap, fig, gs[0,1])
    ax.set_title('no reward vs reward')
    _, ax = plot_rate(out, bins, session_df['choice'].values, [-1, 0.5, 2], ['left', 'right'], colormap, fig, gs[0,2])
    ax.set_title('left vs right')
    # aligned to licks
    _, ax = plot_rate(mat_licks, bins_licks, lick_data['lick_trains_all']['side'], [-1, 0.5, 2], ['L', 'R'], colormap, fig, gs[1,0])
    ax.set_title('All licks left vs right')
    _, ax = plot_rate(mat_licks_L, bins_licks_L, lick_data['lick_trains_L']['in_trial'].astype(int), [-1, 0.5, 2], ['out of trial', 'in trial'], colormap, fig, gs[1,1])
    ax.set_title('Left licks out of trial vs in trial')
    _, ax = plot_rate(mat_licks_R, bins_licks_R, lick_data['lick_trains_R']['in_trial'].astype(int), [-1, 0.5, 2], ['out of trial', 'in trial'], colormap, fig, gs[1,2])
    ax.set_title('Right licks out of trial vs in trial')

    # reward vs no reward on right side
    _, ax = plot_rate(out[session_df['choice']==1, :], bins, session_df[session_df['choice']==1]['outcome'], [-1, 0.5, 2], ['no reward', 'reward'], colormap, fig, gs[0, 3])
    ax.set_title('Right choice: no reward vs reward')
    # reward vs no reward on left side
    _, ax = plot_rate(out[session_df['choice']==0, :], bins, session_df[session_df['choice']==0]['outcome'], [-1, 0.5, 2], ['no reward', 'reward'], colormap, fig, gs[1, 3])
    ax.set_title('Left choice: no reward vs reward')

    # plot raw pupil trace
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(pupil_data['pupil_times'], pupil_data['pupil_diameter_z'], color='black')
    ax.set_title('Raw pupil diameter (z-scored)')
    # scatter go cue times
    ax.scatter(session_df['go_cue_time'], np.zeros_like(session_df['go_cue_time'])+3, color='red', label='go cue')
    ax.set_xlabel(xlabel='Time (s)')
    ax.set_xlim(session_df['go_cue_time'][0]-5, session_df['go_cue_time'][0]-5+2*60)
    ax.legend()
    ax.set_title('Pupil diameter (z-scored)-Early')

    ax = fig.add_subplot(gs[2, 2:])
    ax.plot(pupil_data['pupil_times'], pupil_data['pupil_diameter_z'], color='black')
    ax.set_title('Raw pupil diameter (z-scored)')
    # scatter go cue times
    ax.scatter(session_df['go_cue_time'], np.zeros_like(session_df['go_cue_time'])+3, color='red', label='go cue')
    ax.set_xlabel(xlabel='Time (s)')
    ax.set_xlim(session_df['go_cue_time'].values[-1]-5-2*60, session_df['go_cue_time'].values[-1]-5)
    ax.legend()
    ax.set_title('Pupil diameter (z-scored)-Late')


    plt.suptitle(f'{session} pupil analysis')
    plt.tight_layout()

    fig.savefig(os.path.join(session_dir['beh_fig_dir'], f'{session}_pupil_analysis.pdf'))
    # plt.close('all')


# plot unit vs pupil correlation
# def plot_unit_pupil_correlation(session, opto_only=True, bin_size=0.1):
def plot_unit_pupil_correlation(session, bin_size = 5, step_size = 0.1, opto_only=True, bin_size_short=0.5, win_length=10, plot = True):
    trial_duration = [-0.01, 3.5]
    session_dir = session_dirs(session_id=session)
    unit_tbl = get_unit_tbl(session, data_type='curated')
    sesssion_df = get_session_tbl(session)
    pupil_data = load_pupil(session)
    if pupil_data is None:
        print(f'No pupil data for session {session}')
        return None
    if opto_only:
        if unit_tbl is not None:
            if 'opto_pass' in unit_tbl.columns:
                unit_tbl = unit_tbl[unit_tbl['opto_pass']]
        else:
            print(f'No units for session {session}')
            return None

    if len(unit_tbl) == 0:
        print(f'No units for session {session}')
        return None

    pupil_times = pupil_data['pupil_times']
    pupil_values = pupil_data['pupil_diameter_z']
    # set those in trials not pass quality to nan

    gs = gridspec.GridSpec(len(unit_tbl), 3, width_ratios=[1,0.2,0.2])

    # load session quality metrics
    session_qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(session_qm_file, 'r') as f:
        session_qm = json.load(f)
    rec_start = session_qm['ephys_cut'][0]
    rec_end = session_qm['ephys_cut'][1]

    results = []

    for i, row in unit_tbl.iterrows():
        spike_times = row['spike_times']
        drift = load_drift(session, row['unit_id'], 'curated')
        start_unit = rec_start
        end_unit = rec_end
        session_tbl_curr = sesssion_df.copy()
        if drift is not None:
            if drift['ephys_cut'][0] is not None:
                start_unit = max(start_unit, drift['ephys_cut'][0])
                if session_tbl_curr is not None:
                    session_tbl_curr = session_tbl_curr[session_tbl_curr['goCue_start_time'] >= drift['ephys_cut'][0]]
            if drift['ephys_cut'][1] is not None:
                end_unit = min(end_unit, drift['ephys_cut'][1])
                if session_tbl_curr is not None:
                    session_tbl_curr = session_tbl_curr[session_tbl_curr['goCue_start_time'] <= drift['ephys_cut'][1]]

        pupil_values_no_trial = pupil_values.copy()
        if session_tbl_curr is not None and len(session_tbl_curr) > 0:
            trial_starts = session_tbl_curr['goCue_start_time'].values + trial_duration[0]
            trial_ends = session_tbl_curr['goCue_start_time'].values + trial_duration[1]
            mask_tmp = np.ones_like(pupil_values, dtype=bool)
            for start, end in zip(trial_starts, trial_ends):
                # if any part of the bin overlaps with trial, set to nan
                mask_tmp[(pupil_times >= start) & (pupil_times <= end)] = False
            pupil_values_no_trial[~mask_tmp] = np.nan

        # correlation bewtween spike rate and pupil size
        start_unit = max(start_unit, np.nanmin(pupil_times))
        end_unit = min(end_unit, np.nanmax(pupil_times))

        # including trial time
        # calculate spike counts in bins with each step at step_size
        bin_starts = np.arange(start_unit, end_unit-bin_size, step_size)
        bin_ends = bin_starts + bin_size
        left = np.searchsorted(spike_times, bin_starts, side="left")
        right = np.searchsorted(spike_times, bin_ends, side="left")  # [start, end)
        spike_counts = right - left
        spike_counts = spike_counts / bin_size  # convert to rate

        pupil_dia_binned, bin_centers = bin_timeseries_around_align(ts={'time': pupil_times, 'value': pupil_values},
                                                        align_times=np.array([start_unit]),
                                                        step_size=step_size,
                                                        bin_size=bin_size,
                                                        t_start=0,
                                                        t_stop=end_unit - start_unit)
        pupil_dia_binned = pupil_dia_binned.flatten()

        # excluding trial time
        spike_counts_no_trial = spike_counts.copy()
        if session_tbl_curr is not None and len(session_tbl_curr) > 0:
            trial_starts = session_tbl_curr['goCue_start_time'].values + trial_duration[0]
            trial_ends = session_tbl_curr['goCue_start_time'].values + trial_duration[1]
            # if part of the bin overlaps with trial, recompute spike count as part of the bin that does exclude trial
            for start, end in zip(trial_starts, trial_ends):
                # if any part of the bin overlaps with trial, set to nan
                # find bins that is completely in the trial
                mask = (bin_starts > start) & (bin_ends < end)
                spike_counts_no_trial[mask] = np.nan
                # find bins that partially overlap with trial that started before the trial
                mask_partial = (bin_starts < start) & (bin_ends > start) & (bin_ends < end)
                for idx in np.where(mask_partial)[0]:
                    # recompute spike count for this bin
                    bin_time = start - bin_starts[idx]
                    if bin_time >= 0.5*bin_size:
                        left_idx = np.searchsorted(spike_times, bin_starts[idx], side="left")
                        right_idx = np.searchsorted(spike_times, start, side="left")
                        spike_count_partial = right_idx - left_idx
                        spike_counts_no_trial[idx] = spike_count_partial / bin_time
                    else:
                        spike_counts_no_trial[idx] = np.nan
                # find bins that partially overlap with trial that ended after the trial
                mask_partial = (bin_starts > start) & (bin_starts < end) & (bin_ends > end)
                for idx in np.where(mask_partial)[0]:
                    # recompute spike count for this bin
                    bin_time = bin_ends[idx] - end
                    if bin_time >= 0.5*bin_size:
                        left_idx = np.searchsorted(spike_times, end, side="left")
                        right_idx = np.searchsorted(spike_times, bin_ends[idx], side="left")
                        spike_count_partial = right_idx - left_idx
                        spike_counts_no_trial[idx] = spike_count_partial / bin_time
                    else:
                        spike_counts_no_trial[idx] = np.nan
        
        pupil_dia_binned_no_trial, bin_centers_no_trial = bin_timeseries_around_align(ts={'time': pupil_times, 'value': pupil_values_no_trial},
                                                        align_times=np.array([start_unit]),
                                                        step_size=step_size,
                                                        bin_size=bin_size,
                                                        t_start=0,
                                                        t_stop=end_unit - start_unit)
        pupil_dia_binned_no_trial = pupil_dia_binned_no_trial.flatten()


        # non-overlapping bins
        bin_starts = np.arange(start_unit, end_unit-bin_size, bin_size)
        bin_ends = bin_starts + bin_size
        left = np.searchsorted(spike_times, bin_starts, side="left")
        right = np.searchsorted(spike_times, bin_ends, side="left")  # [start, end)
        spike_counts_non_overlap = right - left
        spike_counts_non_overlap = spike_counts_non_overlap / bin_size  # convert to rate    

        # remove a linear fit of time from spike_counts_non_overlap
        time_vec = bin_starts + 0.5*bin_size
        coeffs = np.polyfit(time_vec, spike_counts_non_overlap, 1)
        trend = np.polyval(coeffs, time_vec)
        spike_counts_non_overlap = spike_counts_non_overlap - trend


        pupil_dia_binned_non_overlap, bin_centers_non_overlap = bin_timeseries_around_align(ts={'time': pupil_times, 'value': pupil_values},
                                                        align_times=np.array([start_unit]),
                                                        step_size=bin_size,
                                                        bin_size=bin_size,
                                                        t_start=0,
                                                        t_stop=end_unit - start_unit)
        pupil_dia_binned_non_overlap = pupil_dia_binned_non_overlap.flatten()

        # cross-correlation
        corrs_go = []
        lags_go = []
        lag=int(np.round(win_length/step_size))
        for lag_i in range(-lag, lag+1):
            pupil_dia_binned_shifted, _ = bin_timeseries_around_align(ts={'time': pupil_times, 'value': pupil_values},
                                                            align_times=np.array([start_unit + lag_i*step_size]),
                                                            step_size=bin_size,
                                                            bin_size=bin_size,
                                                            t_start=0,
                                                            t_stop=end_unit - start_unit)
            pupil_dia_binned_shifted = pupil_dia_binned_shifted.flatten()
            corr = correlate_nan(spike_counts_non_overlap, pupil_dia_binned_shifted, lag=0)[0]

            corrs_go.append(corr)
            lags_go.append(lag_i)


        # cross-correlation excluding trial time
        spike_counts_no_trial_non_overlap = spike_counts_non_overlap.copy()
        if session_tbl_curr is not None and len(session_tbl_curr) > 0:
            trial_starts = session_tbl_curr['goCue_start_time'].values + trial_duration[0]
            trial_ends = session_tbl_curr['goCue_start_time'].values + trial_duration[1]
            # if part of the bin overlaps with trial, recompute spike count as part of the bin that does exclude trial
            for start, end in zip(trial_starts, trial_ends):
                # if any part of the bin overlaps with trial, set to nan
                # find bins that is completely in the trial
                mask = (bin_starts > start) & (bin_ends < end)
                spike_counts_no_trial_non_overlap[mask] = np.nan
                # find bins that partially overlap with trial that started before the trial
                mask_partial = (bin_starts < start) & (bin_ends > start) & (bin_ends < end)
                for idx in np.where(mask_partial)[0]:
                    # recompute spike count for this bin
                    bin_time = start - bin_starts[idx]
                    if bin_time >= 0.5*bin_size:
                        left_idx = np.searchsorted(spike_times, bin_starts[idx], side="left")
                        right_idx = np.searchsorted(spike_times, start, side="left")
                        spike_count_partial = right_idx - left_idx
                        spike_counts_no_trial_non_overlap[idx] = spike_count_partial / bin_time
                    else:
                        spike_counts_no_trial_non_overlap[idx] = np.nan
                # find bins that partially overlap with trial that ended after the trial
                mask_partial = (bin_starts > start) & (bin_starts < end) & (bin_ends > end)
                for idx in np.where(mask_partial)[0]:
                    # recompute spike count for this bin
                    bin_time = bin_ends[idx] - end
                    if bin_time >= 0.5*bin_size:
                        left_idx = np.searchsorted(spike_times, end, side="left")
                        right_idx = np.searchsorted(spike_times, bin_ends[idx], side="left")
                        spike_count_partial = right_idx - left_idx
                        spike_counts_no_trial_non_overlap[idx] = spike_count_partial / bin_time
                    else:
                        spike_counts_no_trial_non_overlap[idx] = np.nan
        
        pupil_dia_binned_no_trial_non_overlap = bin_timeseries_around_align(ts={'time': pupil_times, 'value': pupil_values_no_trial},
                                                        align_times=np.array([start_unit]),
                                                        step_size=bin_size,
                                                        bin_size=bin_size,
                                                        t_start=0,
                                                        t_stop=end_unit - start_unit)       

        corrs_no_go = []
        lags_no_go = []

        for lag_i in range(-lag, lag+1):
            # remove trial periods
            pupil_dia_binned_shifted, bin_centers_tmp = bin_timeseries_around_align(ts={'time': pupil_times, 'value': pupil_values_no_trial},
                                                            align_times=np.array([start_unit + lag_i*step_size]),
                                                            step_size=bin_size,
                                                            bin_size=bin_size,
                                                            t_start=0,
                                                            t_stop=end_unit - start_unit)
            pupil_dia_binned_shifted = pupil_dia_binned_shifted.flatten()

            corr = correlate_nan(spike_counts_no_trial_non_overlap, pupil_dia_binned_shifted, lag=0)[0]
            corrs_no_go.append(corr)
            lags_no_go.append(lag_i)

        # autocorr of pupil
        pupil_dia_binned_non_overlap_autocorr, bin_centers_tmp = bin_timeseries_around_align(ts={'time': pupil_times, 'value': pupil_values},
                                                    align_times=np.array([start_unit]),
                                                    step_size=bin_size_short,
                                                    bin_size=bin_size_short,
                                                    t_start=0,
                                                    t_stop=end_unit - start_unit)
        pupil_dia_binned_non_overlap_autocorr = pupil_dia_binned_non_overlap_autocorr.flatten()
        corrs_pupil = correlate_nan(pupil_dia_binned_non_overlap_autocorr, pupil_dia_binned_non_overlap_autocorr, lag=int(np.round(win_length/bin_size_short)))
        lags_pupil = np.array(range(len(corrs_pupil)))
        
        pupil_dia_binned_no_trial_non_overlap_autocorr, bin_centers_tmp = bin_timeseries_around_align(ts={'time': pupil_times, 'value': pupil_values_no_trial},
                                                    align_times=np.array([start_unit]),
                                                    step_size=bin_size_short,
                                                    bin_size=bin_size_short,
                                                    t_start=0,
                                                    t_stop=end_unit - start_unit)
        pupil_dia_binned_no_trial_non_overlap_autocorr = pupil_dia_binned_no_trial_non_overlap_autocorr.flatten()
        corrs_pupil_no_trial = correlate_nan(pupil_dia_binned_no_trial_non_overlap_autocorr, pupil_dia_binned_no_trial_non_overlap_autocorr, lag=int(np.round(win_length/bin_size_short)))
        lags_pupil_no_trial = np.array(range(len(corrs_pupil_no_trial)))


        # correlation of spike rate and pupil size in baseline
        time, baseline, _ = align.align_to_events(spike_times, session_tbl_curr['goCue_start_time'].values,
                                (-1, -0.01), bin_size=0.9)
        baseline = baseline[0, :]/0.9  # convert to rate
        pupil_baseline, _ = bin_timeseries_around_align({'time': pupil_times, 'value': pupil_values},
                                                    session_tbl_curr['goCue_start_time'].values,
                                                    step_size=0.9,
                                                    bin_size=0.9,
                                                    t_start=-1,
                                                    t_stop=0)
        pupil_baseline = pupil_baseline.flatten()
        # compute correlation and significance
        if len(baseline) > 1:
            valid_mask = ~np.isnan(baseline) & ~np.isnan(pupil_baseline)
            corr_baseline = np.corrcoef(baseline[valid_mask], pupil_baseline[valid_mask])[0,1]
            p_sig = pearsonr(baseline[valid_mask], pupil_baseline[valid_mask])[1]
        else:
            corr_baseline = np.nan
            p_sig = np.nan

        # correlation of pupil dilation with baseline spike rate
        # dilation: maximum pupil diameter within 2s from go cue

        dilation = []

        for trial_ind, go_cue in enumerate(session_tbl_curr['goCue_start_time'].values):
            mask = (pupil_times >= go_cue) & (pupil_times <= go_cue + 2)
            if np.any(mask):
                dilation.append(
                    np.nanmax(pupil_values[mask]) - pupil_baseline[trial_ind]
                )
            else:
                dilation.append(np.nan)

        dilation = np.array(dilation)

        valid_mask = ~np.isnan(baseline) & ~np.isnan(dilation)
        if np.sum(valid_mask) > 1:
            corr_dilation = np.corrcoef(baseline[valid_mask], dilation[valid_mask])[0,1]
            p_sig_dilation = pearsonr(baseline[valid_mask], dilation[valid_mask])[1]
        else:
            corr_dilation = np.nan
            p_sig_dilation = np.nan

        # correlation of pupil dilation with spike rate inresponse to go cue (0-0.3s after go cue)
        response_spike_rate = []
        for trial_ind, go_cue in enumerate(session_tbl_curr['goCue_start_time'].values):
            mask = (spike_times >= go_cue) & (spike_times <= go_cue + 0.3)
            spike_count = np.sum(mask)
            response_spike_rate.append(spike_count / 0.3)
        response_spike_rate = np.array(response_spike_rate)
        valid_mask = ~np.isnan(response_spike_rate) & ~np.isnan(dilation)
        if np.sum(valid_mask) > 1:
            corr_response = np.corrcoef(response_spike_rate[valid_mask], dilation[valid_mask])[0,1]
            p_sig_response = pearsonr(response_spike_rate[valid_mask], dilation[valid_mask])[1]
        else:
            corr_response = np.nan
            p_sig_response = np.nan
            
        
        result = {'unit': row['unit_id'],
                'session': session,
                'corr_baseline': corr_baseline,
                'p_sig': p_sig,
                'crosscorr_with_trials': corrs_go,
                'crosscorr_no_trials': corrs_no_go,
                'crosscorr_lags': np.array(lags_go)*step_size,
                'auto_corr_pupil_with_trials': corrs_pupil,
                'auto_corr_pupil_no_trials': corrs_pupil_no_trial,
                'auto_corr_pupil_lags': np.array(lags_pupil)*bin_size_short,
                'baseline_corr': corr_baseline,
                'baseline_p_sig': p_sig,
                'dilation_bl_corr': corr_dilation,
                'dilation_bl_p_sig': p_sig_dilation,
                'dilation_response_corr': corr_response,
                'dilation_response_p_sig': p_sig_response,
                'error_prop': pupil_data['error_prop'],
                }

        results.append(result)

        if plot:

            fig = plt.figure(figsize=(20, 13))
            gs = gridspec.GridSpec(4, 5, height_ratios=[0.5, 0.5, 1, 1])
            ax = fig.add_subplot(gs[0, :])
            # left y axis
            ax.plot(bin_centers+start_unit, pupil_dia_binned, color='black', linewidth=1)
            ax.set_ylabel('Pupil diameter (z-scored)', color='black')
            # right y axis
            ax2 = ax.twinx()
            ax2.plot(bin_centers+start_unit, spike_counts, color='blue', linewidth=1)
            ax2.set_ylabel('Spike rate', color='blue')
            ax.set_xlabel(xlabel='Time (s)')
            ax.set_title('Whole session')

            # plot baseline 
            ax = fig.add_subplot(gs[1, :])
            mask = ~np.isnan(spike_counts_no_trial) & ~np.isnan(pupil_dia_binned_no_trial)
            ax.plot(bin_centers+start_unit, pupil_dia_binned_no_trial, color='black', label='pupil', linewidth=1)
            ax.set_ylabel('Pupil diameter (z-scored)', color='black')
            ax2 = ax.twinx()
            ax2.plot(bin_centers+start_unit, spike_counts_no_trial, color='blue', label='spike counts', linewidth=1)
            ax2.set_ylabel('Spike rate', color='blue')
            ax.set_xlabel(xlabel='Time (s)')
            ax.set_title('Excluding trial periods')

            # cross correlation between pupil and spikes
            ax = fig.add_subplot(gs[2, 0])
            ax.plot(np.array(lags_go)*step_size, corrs_go, label='with trials')
            ax.plot(np.array(lags_no_go)*step_size, corrs_no_go, label='no trials')
            ax.set_title('Cross-correlation between pupil and spikes')
            ax.set_xlabel('Lag (s)')
            ax.set_ylabel('Correlation')
            ax.legend()
            # autocorr of pupil
            ax = fig.add_subplot(gs[2, 1])
            ax.plot(np.array(lags_pupil)*bin_size_short, corrs_pupil, label='with trials')
            ax.plot(np.array(lags_pupil)*bin_size_short, corrs_pupil_no_trial, label='no trials')
            ax.set_title('Autocorrelation of pupil')
            ax.set_xlabel('Lag (s)')
            ax.set_ylabel('Correlation')
            ax.legend()
            # scatter baseline spikes and pupil
            ax = fig.add_subplot(gs[2,2])
            ax.scatter(baseline, pupil_baseline, facecolors='none', edgecolors='black')
            ax.set_title(f'Bl correlation: {corr_baseline:.2f} (p={p_sig:.2f})')
            ax.set_xlabel('Baseline spike rate')
            ax.set_ylabel('Baseline pupil diameter (z-scored)')

            # scatter dilation and baseline spikes
            ax = fig.add_subplot(gs[2,3])
            ax.scatter(baseline, dilation, facecolors='none', edgecolors='black')
            ax.set_title(f'Dila vs unit bl: {corr_dilation:.2f} (p={p_sig_dilation:.2f})')
            ax.set_xlabel('Baseline spike rate')
            ax.set_ylabel('Pupil dilation')

            # scatter dilation and response spikes
            ax = fig.add_subplot(gs[2,4])
            ax.scatter(response_spike_rate, dilation, facecolors='none', edgecolors='black')
            ax.set_title(f'Dila vs unit resp: {corr_response:.2f} p:{p_sig_response:.2f})')
            ax.set_xlabel('Response spike rate')
            ax.set_ylabel('Pupil dilation')

            # bin trials by baseline spike rate and plot psth
            pupil_mat, slide_times = bin_timeseries_around_align(
                ts={'time': pupil_times, 'value': pupil_values},
                align_times=session_tbl_curr['goCue_start_time'].values,
                step_size=0.1,
                bin_size=0.2,
                t_start=-2,
                t_stop=4
            )

            pupil_mat_dilation = pupil_mat - pupil_baseline[:, np.newaxis]
            bins = 3
            edges = np.percentile(response_spike_rate, np.linspace(0, 100, bins+1))
            edges[0] -= 1e-6  # include min value
            edges[-1] += 1e-6  # include max value
            colormap_curr = colormaps['cool']
            fig, ax = plot_rate(pupil_mat_dilation, slide_times, response_spike_rate, edges, range(bins), colormap_curr, fig, gs[3, -1])
            ax.set_title('Pupil psth by response spike rate')

            # bin trials by baseline spike rate and plot psth
            edges = np.percentile(baseline, np.linspace(0, 100, bins+1))
            edges[0] -= 1e-6  # include min value
            edges[-1] += 1e-6
            fig, ax = plot_rate(pupil_mat_dilation, slide_times, baseline, edges, range(bins), colormap_curr, fig, gs[3, -2])
            ax.set_title('Pupil psth by baseline spike rate')

            # bin trials by pupil basline and plot spike psth
            spike_mat, slide_times = get_spike_matrix(
                spike_times,
                session_tbl_curr['goCue_start_time'].values,
                pre_event=-2,
                post_event=4,
                binSize=0.2,
                stepSize=0.1,
            )
            edges = np.percentile(pupil_baseline[~np.isnan(pupil_baseline)], np.linspace(0, 100, bins+1))
            edges[0] -= 1e-6  # include min value
            edges[-1] += 1e-6   
            fig, ax = plot_rate(spike_mat, slide_times, pupil_baseline, edges, range(bins), colormap_curr, fig, gs[3, 2])
            ax.set_title('Spike psth by pupil baseline')

            

            plt.suptitle(f'Session {session} Unit {row["unit_id"]} pupil correlation analysis')
            plt.tight_layout()
            os.makedirs(name=session_dir['ephys_fig_dir_curated'] + '/unit_pupil', exist_ok=True)
            fig.savefig(os.path.join(session_dir['ephys_fig_dir_curated'], 'unit_pupil', f'{session}_unit_{row["unit_id"]}_pupil_correlation_analysis.pdf'))
    # save results
    results_df = pd.DataFrame(results)
    # save to .pickle
    with open(os.path.join(session_dir['ephys_processed_dir_curated'], f'{session}_unit_pupil_correlation_results.pickle'), 'wb') as f:
        pickle.dump(results_df, f)

    return results_df
# plt.close('all')


# %%
# parallel processing across sessions
from joblib import Parallel, delayed
data_df = pd.read_csv('/root/capsule/code/data_management/hopkins_session_assets.csv')
sessions = data_df['session_id'].values.tolist()
def safe_plot(session):
    # try:
    #     # pupil_analysis_session(session, plot_licks=True)
    plot_unit_pupil_correlation(session, bin_size=5, step_size=0.1, win_length=20, plot=True)

    #     print(f'Success {session}')
    # except Exception as e:
    #     print(f'Error processing session {session}: {e}')
# Parallel(n_jobs=-1)(delayed(safe_plot)(session) for session in sessions)
# plot_unit_pupil_correlation('behavior_ZS061_2021-03-20_18-03-26')
plot_unit_pupil_correlation('behavior_ZS060_2021-03-29_16-21-37', bin_size=5, step_size=0.1, win_length=20, plot=True)

 