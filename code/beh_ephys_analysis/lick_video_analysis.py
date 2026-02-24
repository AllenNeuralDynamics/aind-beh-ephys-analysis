# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from scipy.io import loadmat
from scipy.stats import zscore
import ast
from utils.plot_utils import combine_pdf_big
from utils.beh_functions import session_dirs, parseSessionID, load_model_dv, makeSessionDF, get_session_tbl, get_unit_tbl, get_history_from_nwb
from utils.ephys_functions import*
from utils.lick_utils import load_licks, clean_up_licks
from utils.combine_tools import apply_qc, to_str_intlike

from open_ephys.analysis import Session
from pathlib import Path
import glob

import json
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import re
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from hdmf_zarr.nwb import NWBZarrIO

import pandas as pd
import pickle
import scipy.stats as stats
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial
import time
import shutil 
from aind_ephys_utils import align
from PIL import Image
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, log_loss

# %%
def assign_lick_label(start_time, end_time, peak_time, licks_L, licks_R):
    has_lick = 0
    if np.any((licks_L >= start_time) & (licks_L <= end_time)):
        has_lick = -1
    if np.any((licks_R >= start_time) & (licks_R <= end_time)):
        has_lick = 1
    min_time_diff_L_ind = np.argmin(np.abs(licks_L - (start_time+peak_time)))
    min_time_diff_R_ind = np.argmin(np.abs(licks_R - (start_time+peak_time)))
    min_time_diff_L = np.abs(licks_L[min_time_diff_L_ind] - (start_time+peak_time))
    min_time_diff_R = np.abs(licks_R[min_time_diff_R_ind] - (start_time+peak_time))
    if min_time_diff_L < min_time_diff_R:
        min_time_diff = licks_L[min_time_diff_L_ind] - (start_time+peak_time)
    else:
        min_time_diff = licks_R[min_time_diff_R_ind] - (start_time+peak_time)
    return has_lick, min_time_diff
def pairplot_color_code(start, max_point, color_code, feature_name, fig, subplot_gs,
                        v_range=None, center_xys=None, bins_xy=50, bins_c=50,
                        equal_aspect=True):
    """
    start, max_point: shape (2, N) arrays: [x_like, y_like] in your code it's [Y, X]
    color_code: length N
    center_xy: tuple (center_y, center_x) in your coordinate convention
    """

    # ---- layout ----
    gs_sub = gridspec.GridSpecFromSubplotSpec(
        6, 4, subplot_spec=subplot_gs,
        height_ratios=[1, 1, 1, 1, 0.5, 0.5],
        hspace=0.4, wspace=0.4
    )
    ax_main  = fig.add_subplot(gs_sub[1:4, 0:3])
    ax_xhist = fig.add_subplot(gs_sub[0,   0:3], sharex=ax_main)
    ax_yhist = fig.add_subplot(gs_sub[1:4, 3],   sharey=ax_main)

    # ---- main plot ----
    # trajectories
    for i in range(start.shape[1]):
        ax_main.plot([start[0, i], max_point[0, i]],
                     [start[1, i], max_point[1, i]],
                     color='k', alpha=0.1, linewidth=0.05)

    # color range
    if v_range is None:
        color_code = np.asarray(color_code, float)
        v_range = (np.nanquantile(color_code, 0.05), np.nanquantile(color_code, 0.95))

    ax_main.scatter(start[0, :], start[1, :], color='gray', s=2, label='Start Point')

    sg = ax_main.scatter(
        max_point[0, :], max_point[1, :],
        c=color_code, s=2, vmin=v_range[0], vmax=v_range[1]
    )

    if center_xys is not None:
        ax_main.scatter(center_xys[:, 0], center_xys[:, 1], color='red', marker='x', s=25, label='Center of Mass')

    ax_main.set_xlabel("Y")
    ax_main.set_ylabel("X")
    ax_main.set_title(f'Lick locations colored by {feature_name}')

    # ---- lock limits (CRITICAL for alignment) ----
    # Compute limits directly from the plotted data to avoid autoscale surprises.
    x = np.asarray(max_point[0, :], float)
    y = np.asarray(max_point[1, :], float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # optional tiny padding (consistent across axes)
    pad_x = 0.02 * (xmax - xmin) if xmax > xmin else 1.0
    pad_y = 0.02 * (ymax - ymin) if ymax > ymin else 1.0

    ax_main.set_xlim(xmin - pad_x, xmax + pad_x)
    ax_main.set_ylim(ymin - pad_y, ymax + pad_y)

    if equal_aspect:
        # use box adjustment so limits don't change after aspect set
        ax_main.set_aspect('equal', adjustable='box')

    # Capture final limits after all adjustments
    x_lims = ax_main.get_xlim()
    y_lims = ax_main.get_ylim()

    # ---- marginals (draw AFTER limits are fixed) ----
    ax_xhist = fig.add_subplot(gs_sub[0,   0:3], sharex=ax_main)
    ax_yhist = fig.add_subplot(gs_sub[1:4, 3],   sharey=ax_main)
    ax_xhist.hist(x, bins=bins_xy, color='black', alpha=0.6)
    ax_yhist.hist(y, bins=bins_xy, orientation='horizontal', color='black', alpha=0.6)

    # Re-apply limits because hist can autoscale
    ax_xhist.set_xlim(x_lims)
    ax_yhist.set_ylim(y_lims)

    # Prevent future autoscale drift
    ax_xhist.set_autoscale_on(False)
    ax_yhist.set_autoscale_on(False)

    # Clean marginal ticks/labels
    ax_xhist.tick_params(axis='x', labelbottom=False)
    ax_xhist.tick_params(axis='y', left=False, labelleft=False)
    ax_yhist.tick_params(axis='y', labelleft=False)
    ax_yhist.tick_params(axis='x', bottom=False, labelbottom=False)

    # ---- horizontal colorbar (own axis) ----
    ax_cbar = fig.add_subplot(gs_sub[4, 0:3])
    cbar = fig.colorbar(sg, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(feature_name)

    # ---- histogram of color_code ----
    ax_hist_color = fig.add_subplot(gs_sub[5, 0:3])
    cc = np.asarray(color_code, float)
    cc = cc[np.isfinite(cc)]
    bins = np.linspace(np.min(cc), np.max(cc), bins_c)
    ax_hist_color.hist(color_code, bins=bins, color='skyblue', edgecolor=None)
    ax_hist_color.axvline(v_range[0], color='b', linestyle='--')
    ax_hist_color.axvline(v_range[1], color='y', linestyle='--')
    ax_hist_color.set_yscale('log')
    ax_hist_color.set_title(f'Histogram of {feature_name}')

    return ax_main, ax_xhist, ax_yhist, ax_cbar, ax_hist_color



def classify_with_gmm(
    X,
    y,
    *,
    seed=0,
    plot=False,
    plot_boundary=True,
    confidence_alpha=True,
    ax=None
):
    """
    X : (N,2) positions
    y : (N,) labels (0, 1 / np.nan for unlabeled)
    plot : bool
        Whether to plot classification result.
    plot_boundary : bool
        Whether to draw decision boundary.
    confidence_alpha : bool
        If True, point transparency reflects confidence.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    unlabeled = (y == -1) if y.dtype.kind != 'f' else ~np.isfinite(y)
    labeled = ~unlabeled

    # Fit GMM
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=seed
    )
    gmm.fit(X)

    probs = gmm.predict_proba(X)

    # Align components to labeled meaning
    if labeled.sum() > 0:
        comp_labels = gmm.predict(X[labeled])
        true_labels = y[labeled].astype(int)

        mean_label_per_comp = []
        for k in range(2):
            if np.any(comp_labels == k):
                mean_label_per_comp.append(
                    np.mean(true_labels[comp_labels == k])
                )
            else:
                mean_label_per_comp.append(0)

        comp_to_class = np.argsort(mean_label_per_comp)
        probs = probs[:, comp_to_class]

    proba1 = probs[:, 1]
    pred = (proba1 >= 0.5).astype(int)
    confidence = np.maximum(proba1, 1 - proba1)

    # Metrics
    metrics = {}
    if labeled.sum() > 0:
        acc = accuracy_score(y[labeled], pred[labeled])
        ll = log_loss(y[labeled], proba1[labeled])
        metrics = {
            "accuracy": acc,
            "logloss": ll,
            "n_labeled": int(labeled.sum())
        }

    # ------------------ Plot ------------------
    fig = None
    if plot:
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(30, 15))
        ax = axes[0]
        # Decision boundary
        if plot_boundary:
            x_min, x_max = X[:,0].min(), X[:,0].max()
            y_min, y_max = X[:,1].min(), X[:,1].max()
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 200),
                np.linspace(y_min, y_max, 200)
            )
            grid = np.column_stack([xx.ravel(), yy.ravel()])
            grid_probs = gmm.predict_proba(grid)

            if labeled.sum() > 0:
                grid_probs = grid_probs[:, comp_to_class]

            Z = grid_probs[:,1].reshape(xx.shape)
            ax.contour(xx, yy, Z, levels=[0.5], linewidths=2)

        # Scatter points
        for cls, color in zip([0,1], ["tab:blue", "tab:orange"]):
            mask = pred == cls
            alpha_vals = confidence[mask] if confidence_alpha else 0.8
            ax.scatter(
                X[mask,0],
                X[mask,1],
                c=color,
                alpha=alpha_vals,
                label=f"Class {cls}"
            )

        # Mark labeled points
        if labeled.sum() > 0:
            ax.scatter(
                X[labeled,0],
                X[labeled,1],
                facecolors='none',
                edgecolors='black',
                s=80,
                linewidths=1.5,
                label="Labeled"
            )

        # Plot GMM means
        means = gmm.means_
        ax.scatter(
            means[:,0],
            means[:,1],
            c="red",
            marker="X",
            s=150,
            label="GMM Means"
        )
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.legend()
        ax.set_aspect("equal")

        # Another scatter plot for colorcode of confidence
        ax = axes[1]
        sc = ax.scatter(
            X[:,0],
            X[:,1],
            c=confidence,
            cmap='viridis',
            vmin=0.5, vmax=1.0,
            label="Confidence"
        )
        # circle the labeled points
        if labeled.sum() > 0:
            ax.scatter(
                X[labeled,0],
                X[labeled,1],
                facecolors='none',
                edgecolors='black',
                s=80,
                linewidths=1.5,
                label="Labeled"
            )
        fig.colorbar(sc, ax=ax, label='Confidence')
        # plot center of the two clusters
        centers = gmm.means_
        ax.scatter(
            centers[:,0],
            centers[:,1],
            c="red",
            marker="X",
            s=150,
            label="GMM Means"
        )
        plt.suptitle("GMM Classification with Confidence")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.legend()
        ax.set_aspect("equal")

        if ax is None:
            plt.show()

    return {
        "pred": pred,
        "proba1": proba1,
        "confidence": confidence,
        "metrics_on_labeled": metrics,
        "gmm": gmm,
        'fig': fig
    }


def plot_licks_from_video(session, plot=True, cutoff_percentile=0.95):
    video_data_file = '/root/capsule/data/all_tongue_movements_04022026/all_tongue_movements_04022026.parquet'
    all_lick_df = pd.read_parquet(video_data_file)
    session_video_list = all_lick_df['session'].unique().tolist()
    # load data
    session_df = get_session_tbl(session)
    animal_id, session_time_curr, _  = parseSessionID(session)
    # find the corresponding session in the session list withs same animalsl_id and closest session time
    session_list_curr = [s for s in session_video_list if str(s).startswith(f'behavior_{animal_id}')]
    if len(session_list_curr) == 0:
        print(f"No session found for {session}. Skipping.")
        return
    session_index = np.argmin([abs((parseSessionID(s)[1] - session_time_curr).total_seconds()) for s in session_list_curr])
    if np.min([abs((parseSessionID(s)[1] - session_time_curr).total_seconds()) for s in session_list_curr])>60:
        print(f"No closely matched session found for {session}. Skipping.")
        return
    session_video = session_list_curr[session_index]
    session_licks = all_lick_df[all_lick_df['session'] == session_video].copy().reset_index(drop=True)
    session_dir = session_dirs(session)

    if session_dir['aniID'].startswith('ZS'):
        raw_nwb_file = [f for f in os.listdir(session_dir['raw_dir']) if f.endswith('.nwb.zarr')][0]
        raw_nwb = load_nwb_from_filename(os.path.join(session_dir['raw_dir'], raw_nwb_file))
    else:
        if not os.path.exists(session_dir['nwb_beh']):
            print(f"Session {session} does not have a behavioral NWB file. Skipping.")
        else:
            raw_nwb = load_nwb_from_filename(session_dir['nwb_beh'])
    raw_df = raw_nwb.intervals['trials'].to_dataframe()
    licks_L = raw_nwb.acquisition['left_lick_time'].timestamps[:]
    licks_R = raw_nwb.acquisition['right_lick_time'].timestamps[:]
    all_licks = np.sort(np.concatenate([licks_L, licks_R]))
    licks_L_cleaned, licks_R_cleaned, fig = clean_up_licks(licks_L, licks_R, crosstalk_thresh=100/1000, rebound_thresh=50/1000, plot=False)
    session_licks['start_time_session'] = session_licks['start_time'] + session_df['goCue_start_time'].values[0]
    session_licks['end_time_session'] = session_licks['end_time'] + session_df['goCue_start_time'].values[0]
    session_licks['goCue_start_time_in_session'] = session_licks['goCue_start_time_in_session'] + session_df['goCue_start_time'].values[0]
    session_licks['mean_speed'] = session_licks['total_distance'].values/session_licks['duration'].values

    session_licks['has_lick_side'] = 0
    session_licks['min_time_diff'] = np.inf
    for idx, row in session_licks.iterrows():
        has_lick, min_time_diff = assign_lick_label(row['start_time_session'], row['end_time_session'], row['time_to_endpoint'], licks_L_cleaned, licks_R_cleaned)
        session_licks.at[idx, 'has_lick_side'] = has_lick
        session_licks.at[idx, 'min_time_diff'] = min_time_diff

    center_x = session_licks['max_y_from_jaw_x'].mean()
    center_y_L = session_licks[session_licks['has_lick_side'] == -1]['max_y_from_jaw'].mean()
    center_y_R = session_licks[session_licks['has_lick_side'] == 1]['max_y_from_jaw'].mean()
    center_y = (center_y_L + center_y_R) / 2

    center_x_start = session_licks[session_licks['has_lick_side'] != 0]['startpoint_x'].mean()
    center_y_start_L = session_licks[session_licks['has_lick_side'] == -1]['startpoint_y'].mean()
    center_y_start_R = session_licks[session_licks['has_lick_side'] == 1]['startpoint_y'].mean()
    center_y_start = (center_y_start_L + center_y_start_R) / 2

    session_licks['dis_center_start'] = np.sqrt(
        np.sum(
            (session_licks[['startpoint_x', 'startpoint_y']].values
            - np.array([center_x_start, center_y_start]))**2,
            axis=1
        )
    )
    session_licks['dis_center_max'] = np.sqrt(
        np.sum(
            (session_licks[['max_x_from_jaw', 'max_y_from_jaw']].values
            - np.array([center_x, center_y]))**2,
            axis=1
        )
    )

    # data selection
    thresholds = {'mean_speed': [0, 2500],
                'dis_center_max': [0, 150],
                'duration': [0.006, 0.3],
                'total_distance': [0, 300],
                'dis_center_start': [0, 100],
                'peak_velocity': [0, session_licks[session_licks['has_lick']]['peak_velocity'].quantile(cutoff_percentile)]}
    thresholds = {'mean_speed': [0, session_licks[session_licks['has_lick']]['mean_speed'].quantile(cutoff_percentile)],
                'dis_center_max': [0, session_licks[session_licks['has_lick']]['dis_center_max'].quantile(cutoff_percentile)],
                'duration': [0.02, session_licks[session_licks['has_lick']]['duration'].quantile(cutoff_percentile)],
                'total_distance': [0, session_licks[session_licks['has_lick']]['total_distance'].quantile(cutoff_percentile)],
                'dis_center_start': [0, session_licks[session_licks['has_lick']]['dis_center_start'].quantile(cutoff_percentile)],
                'peak_velocity': [0, session_licks[session_licks['has_lick']]['peak_velocity'].quantile(cutoff_percentile)]}
    filter = np.full(len(session_licks), True)
    for feature, (lower, upper) in thresholds.items():
        filter &= (session_licks[feature] >= lower) & (session_licks[feature] <= upper)
    session_licks['filter'] = filter
    # save the filtered data

    # classify with GMM
    X = np.stack([session_licks['max_y_from_jaw'].values, session_licks['max_y_from_jaw_x'].values], axis=1)[session_licks['filter'].values].T
    y = session_licks['has_lick_side'].values[session_licks['filter'].values]
    y = y.astype(float)
    y[y == 0] = np.nan
    y[y == -1] = 0

    gmm_result = classify_with_gmm(X.T, y, plot=True)
    gmm_fig = gmm_result['fig']
    save_file_gmm = os.path.join(session_dir['beh_fig_dir'], f'{session}_lick_gmm_classification.png')
    gmm_fig.savefig(save_file_gmm, dpi=300)

    # append the GMM classification result to the session_licks dataframe
    session_licks.loc[session_licks['filter'], 'gmm_pred'] = gmm_result['pred']
    session_licks.loc[session_licks['filter'], 'gmm_proba1'] = gmm_result['confidence']

    save_lick_csv = os.path.join(session_dir['beh_fig_dir'], f"{session}_filtered_video_licks.csv")
    session_licks.to_csv(save_lick_csv, index=False)

    # plot selection
    if plot:
        g = sns.pairplot(
            data=session_licks[['duration', 'total_distance', 'mean_speed',
                                'dis_center_max', 'dis_center_start',
                                'peak_velocity', 'filter']],
            hue='filter',
            diag_kind='hist',
            diag_kws={'alpha': 0.5},
            plot_kws={'alpha': 0.3},
        )

        for ax in g.diag_axes:
            ax.set_yscale('log')
            ax.set_xscale('log')

        for ax in g.axes.flatten():
            ax.set_xscale('log')
            ax.set_yscale('log')


        save_file_selection = os.path.join(session_dir['beh_fig_dir'], f"{session}_lick_feature_relationships.png")
        g.fig.savefig(save_file_selection, dpi=300)
        plt.close(g.fig)

        # plot detected vs not licks
        g = sns.pairplot(
            data=session_licks[['duration', 'total_distance', 'mean_speed',
                                'dis_center_max', 'dis_center_start',
                                'peak_velocity', 'has_lick_side']],   
            hue='has_lick_side',
            diag_kind='hist',
            diag_kws={'alpha': 0.5},
            plot_kws={'alpha': 0.3},
            palette="viridis"
        )
        for ax in g.diag_axes:
            ax.set_yscale('log')
            ax.set_xscale('log')

        for ax in g.axes.flatten():
            ax.set_xscale('log')
            ax.set_yscale('log')

        save_file_has_lick = os.path.join(session_dir['beh_fig_dir'], f"{session}_lick_feature_relationships_by_lick_side.png")
        g.fig.savefig(save_file_has_lick, dpi=300)
        plt.close(g.fig)

        # plot lick raster
        fig = plt.figure(figsize=(18, 5))
        gs = gridspec.GridSpec(1, 5)
        color_map = LinearSegmentedColormap.from_list('custom_colormap', ['blue', 'gray', 'red'])
        bin_num = 3
        edges = np.linspace(0, len(session_df), bin_num)
        _, ax, _ = plot_raster_rate(licks_L_cleaned, session_df['goCue_start_time'].values, np.arange(len(session_df)), edges, ['early', 'late'], color_map, fig, gs[0])
        ax.set_title('Left Licks')
        _, ax, _ = plot_raster_rate(licks_R_cleaned, session_df['goCue_start_time'].values, np.arange(len(session_df)), edges, ['early', 'late'], color_map, fig, gs[1])
        ax.set_title('Right Licks')

        all_licks = np.sort(np.concatenate([licks_L_cleaned, licks_R_cleaned]))
        _, ax, _ = plot_raster_rate(all_licks, session_df['goCue_start_time'].values, np.arange(len(session_df)), edges, ['early', 'late'], color_map, fig, gs[2])
        ax.set_title('All Licks')

        _, ax, _ = plot_raster_rate(session_licks[session_licks['has_lick_side']!=0]['start_time_session'].values, session_df['goCue_start_time'].values, np.arange(len(session_df)), edges, ['early', 'late'], color_map, fig, gs[3])
        ax.set_title('Detected video Licks')
        _, ax, _ = plot_raster_rate(session_licks['start_time_session'].values, session_df['goCue_start_time'].values, np.arange(len(session_df)), edges, ['early', 'late'], color_map, fig, gs[4])
        ax.set_title('Video Licks')

        plt.tight_layout()

        save_file_raster = os.path.join(session_dir['beh_fig_dir'], f"{session}_lick_raster_comparison.png")
        fig.savefig(save_file_raster, dpi=300)
        plt.close(fig)

        # plot feature in space
        fig = plt.figure(figsize=(24, 24))
        subplot_gs = gridspec.GridSpec(2, 2, hspace=0.2, wspace=0.2)
        start = np.array([session_licks['startpoint_y'].values,
                        session_licks['startpoint_x'].values])[:, session_licks['filter'].values]

        max_point = np.array([session_licks['max_y_from_jaw'].values,
                            session_licks['max_y_from_jaw_x'].values])[:, session_licks['filter'].values]
        center_xys = np.array([[center_y, center_x], 
                            [center_y_start, center_x_start]])

        color_code = session_licks['has_lick_side'].values[session_licks['filter'].values]
        sort_ind = np.argsort(color_code)
        curr_ax, ax_main, ax_xhist, ax_yhist, ax_hist_color = pairplot_color_code(start[:, sort_ind], max_point[:, sort_ind], color_code[sort_ind], 'Lick_side', fig, subplot_gs[0], center_xys=center_xys)

        feature = 'duration'
        color_code = session_licks[feature].values[session_licks['filter'].values]
        sort_ind = np.argsort(color_code)
        curr_ax, ax_main, ax_xhist, ax_yhist, ax_hist_color = pairplot_color_code(start[:, sort_ind], max_point[:, sort_ind], color_code[sort_ind], feature, fig, subplot_gs[1], center_xys=center_xys)

        feature = 'peak_velocity'
        color_code = session_licks[feature].values[session_licks['filter'].values]
        sort_ind = np.argsort(color_code)
        curr_ax, ax_main, ax_xhist, ax_yhist, ax_hist_color = pairplot_color_code(start[:, sort_ind], max_point[:, sort_ind], color_code[sort_ind], feature, fig, subplot_gs[2], center_xys=center_xys)

        feature = 'mean_speed'
        color_code = session_licks[feature].values[session_licks['filter'].values]
        sort_ind = np.argsort(color_code)
        curr_ax, ax_main, ax_xhist, ax_yhist, ax_hist_color = pairplot_color_code(start[:, sort_ind], max_point[:, sort_ind], color_code[sort_ind], feature, fig, subplot_gs[3], center_xys=center_xys)

        plt.suptitle(session)

        plt.tight_layout()

        save_file_space = os.path.join(session_dir['beh_fig_dir'], f'{session}_lick_feature_space.png')
        fig.savefig(save_file_space, dpi=300)
        # close fig
        plt.close(fig)



        # combine all pdfs into one
        png_files = [save_file_selection, save_file_has_lick, save_file_raster, save_file_space, save_file_gmm]
        output_pdf = os.path.join(session_dir['beh_fig_dir'], f'{session}_lick_video_analysis_combined.pdf')
        images = [Image.open(f).convert("RGB") for f in png_files]

        images[0].save(
            output_pdf,
            save_all=True,
            append_images=images[1:]
        )
        plt.close('all')



if __name__ == "__main__":
    dfs = [pd.read_csv('/root/capsule/code/data_management/session_assets.csv'),
        pd.read_csv('/root/capsule/code/data_management/hopkins_session_assets.csv')]
    df = pd.concat(dfs)
    session_list = df['session_id'].values.tolist()
    session_list = [session for session in session_list if str(session).startswith('behavior') and 'ZS' not in session]
    # for session in session_list:
    #     print(f"Processing session {session}...")
    #     plot_licks_from_video(session, plot=True, cutoff_percentile=0.95)
    # plot_licks_from_video('behavior_791691_2025-06-25_14-06-10')
    # parallel processing
    num_cores = 4
    Parallel(n_jobs=num_cores)(
        delayed(plot_licks_from_video)(session, plot=True, cutoff_percentile=0.95)
        for session in session_list
    )

