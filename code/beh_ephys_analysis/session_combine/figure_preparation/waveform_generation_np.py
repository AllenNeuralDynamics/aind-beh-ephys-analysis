"""
Step 3a of figure preparation pipeline: Generate waveform comparison data for Neuropixels recordings.

Prerequisites:
    MUST run FIRST:
    1. make_combined_unit_tbl.py (Step 1) - Creates combined_unit_tbl.pkl

    Data requirements:
    - combined_unit_tbl.pkl with waveform data (wf, wf_raw, peak, peak_raw columns)
    - Waveforms should be populated during session preprocessing
    - NeuropixelsProbe (np) recordings only

Pipeline Position:
    Script #3 in sequence.txt (line 3)
    Can run IN PARALLEL with:
    - antidromic_generation.py
    - waveform_generation_tt.py
    - basic_ephys_generation.py
    - acg_generation.py
    - response_tstats_generation.py
    - outcome_window_generation_parallel.py
    (All these scripts only need combined_unit_tbl.pkl from Step 1)

Purpose:
    Extracts waveform features from NeuropixelsProbe recordings to characterize cell types:
    - Peak-to-trough width, half-width (distinguishes fast-spiking vs regular-spiking)
    - Waveform symmetry metrics (pre/post peak slopes, integral areas)
    - Amplitude features (peak, trough locations and magnitudes)
    - PCA-based dimensionality reduction of waveform shapes

    These features enable cell-type classification and quality assessment.

Input:
    - Combined unit table from Step 1 (with wf, wf_raw, peak, peak_raw columns)

Output:
    - combined_waveform_np_tbl.pkl: Waveform features for np-probe units
    - Includes: peak-to-trough width, half-width, symmetry metrics, trough ratios,
      waveform PCA components, all features for cell-type classification
    - Quality control plots showing waveform distributions

Usage:
    Run after antidromic_generation.py completes. Processes NeuropixelsProbe data only.
    Extracts features from waveforms already in combined_unit_tbl.
"""
import sys
import os
# Resolve code/beh_ephys_analysis (the folder containing `utils`) relative to this
# file's location, so imports work no matter where the repo is checked out.
import os
import sys
_beh_ephys_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _beh_ephys_root not in sys.path:
    sys.path.insert(0, _beh_ephys_root)
from utils.capsule_migration import CAPSULE_ROOT
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
from utils.ccf_utils import ccf_pts_convert_to_mm, pir_to_lps, project_to_plane
from utils.combine_tools import apply_qc, to_str_intlike, spatial_dependence_summary, binary_shift_P_vs_U, welch_shift_P_vs_U
from utils.capsule_migration import capsule_directories
import pandas as pd
import pickle
import scipy.stats as stats
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial
import time
import spikeinterface as si
import shutil 
import seaborn as sns
import math  
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import zscore
from trimesh import load_mesh
from scipy.optimize import minimize
from scipy.linalg import null_space
from joblib import Parallel, delayed
from matplotlib.colors import Normalize
from scipy.stats import rankdata
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
capsule_dirs = capsule_directories()

criteria_name = 'waveform_all'
waveform_version = '_raw' # 'wf_2D' for 'wf_2D_raw'
sampling_frequency = 30000  # Hz - Neuropixels sampling rate
ms_per_sample = 1000 / sampling_frequency  # Convert samples to milliseconds (0.0333 ms/sample)
target_folder = os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'waveforms_np')
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# load constraints and data
with open(os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'combined_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
    combined_tagged_units = pickle.load(f)
with open(os.path.join(CAPSULE_ROOT + '/code/beh_ephys_analysis/session_combine/metrics', f'{criteria_name}.json'), 'r') as f:
    constraints = json.load(f)

combined_tagged_units_filtered, combined_tagged_units, fig, axes = apply_qc(combined_tagged_units, constraints)
fig.savefig(os.path.join(target_folder, 'quality_metrics.pdf'))

# Extract waveform features for all units that passed QC
wf_norm = []  # Normalized waveforms (divided by peak amplitude)
wf_2D_norm = []  # 2D waveforms, normalize by peak amplitude

# Width features - distinguish fast-spiking (narrow) from regular-spiking (wide) neurons
# All width features are in milliseconds (converted from samples using ms_per_sample)
half_w = []  # Half-width: time between half-peak crossings before and after peak (ms)
trough_w = []  # Trough-to-trough width: time from pre-peak trough to post-peak trough (ms)
pre_half = []  # Pre-peak half-width: time from pre-peak half-crossing to peak (ms)
post_half = []  # Post-peak half-width: time from peak to post-peak half-crossing (ms)
post_w = []  # Post-peak trough width: time from peak to post-peak trough (ms)
pre_w = []  # Pre-peak trough width: time from pre-peak trough to peak (ms)

# Trough ratio features - characterize waveform shape asymmetry
trough_post_ratio_1D = []  # Post-peak trough amplitude / peak amplitude (positive if same sign)
trough_pre_ratio_1D = []  # Pre-peak trough amplitude / peak amplitude (positive if same sign)

# Slope and integral features - characterize rise/fall kinetics (slopes per ms, integrals in ms)
post_slope = []  # Post-peak slope: ((peak - post_trough)/peak) / time_ms (normalized amplitude change per ms)
post_trough_slope = []  # Post-trough slope: trough_post_ratio / time_ms (see trough_post_ratio_1D above, per ms)
post_space = []  # Post-peak integral: trough_post_ratio × time_ms (normalized amplitude × ms)
post_space_raw = []  # Post-peak raw area: sum of positive waveform values after peak (unnormalized, no time unit)
pre_slope = []  # Pre-peak slope: ((peak - pre_trough)/peak) / time_ms (normalized amplitude change per ms)
pre_space = []  # Pre-peak integral: trough_pre_ratio × time_ms (normalized amplitude × ms)
pre_space_raw = []  # Pre-peak raw area: sum of positive waveform values before peak (unnormalized, no time unit)

# Symmetry features - quantify waveform asymmetry (difference between post- and pre-peak)
symmetry_inte = []  # Integral asymmetry: post_space - pre_space (positive = more post-peak area)
symmetry_half = []  # Half-width asymmetry: post_half - pre_half (ms, positive = slower repolarization)
symmetry_slope = []  # Slope asymmetry: post_slope - pre_slope (positive = steeper post-peak)
symmetry_inte_div = []  # Integral ratio: post_space / pre_space
symmetry_half_div = []  # Half-width ratio: post_half / pre_half
symmetry_slope_div = []  # Slope ratio: pre_slope / post_slope (inverted for convention)
symmetry_space_raw_div = []  # Raw area ratio: post_space_raw / pre_space_raw
symmetry_trough_dis = []  # Trough distance: post_trough_ind - pre_trough_ind (ms)

# Summary features - combined metrics
trough_sum = []  # Combined trough metric: (pre_space_raw + post_space_raw) / peak
slope_sum = []  # Combined slope metric: post_slope + pre_slope


wf_norm = combined_tagged_units_filtered[f'wf{waveform_version}']/np.abs(combined_tagged_units_filtered[f'peak{waveform_version}'])
# wf_2D_norm = combined_tagged_units_filtered[f'wf_2d{waveform_version}']/np.abs(combined_tagged_units_filtered[f'peak{waveform_version}'])
for rows in combined_tagged_units_filtered.iterrows():
    wf = rows[1][f'wf{waveform_version}']
    wf_bl = np.nanmean(wf[:5])
    peak = rows[1]['peak_raw'] - wf_bl
    wf = wf - wf_bl
    peak_ind = np.argmin(wf)

    if np.abs(wf[0]-wf[-1])>50:
        half_w.append(np.nan)  # half width in samples
        trough_w.append(np.nan)  # trough width in samples
        post_w.append(np.nan)  # post trough width in samples
        pre_w.append(np.nan)
        pre_half.append(np.nan)
        post_half.append(np.nan)
        trough_post_ratio_1D.append(np.nan)
        trough_pre_ratio_1D.append(np.nan)
        post_slope.append(np.nan)
        post_trough_slope.append(np.nan)
        post_space.append(np.nan)
        pre_slope.append(np.nan)
        pre_space.append(np.nan)
        symmetry_inte.append(np.nan)
        symmetry_half.append(np.nan)
        symmetry_slope.append(np.nan)
        symmetry_inte_div.append(np.nan)
        symmetry_half_div.append(np.nan)
        symmetry_slope_div.append(np.nan)
        symmetry_space_raw_div.append(np.nan)
        symmetry_trough_dis.append(np.nan)
        post_space_raw.append(np.nan)
        pre_space_raw.append(np.nan)
        continue

    if peak<0:
        curr_trough_loc = np.argmax(wf[peak_ind:])+1
        post_trough = np.max(wf[peak_ind:])
        post_trough_ind = np.argmax(wf[peak_ind:])+1
        pre_trough = np.max(wf[:peak_ind])
        pre_trough_ind = peak_ind - np.argmax(wf[:peak_ind])  
    else:
        curr_trough_loc = np.argmin(wf[peak_ind:])+1
        post_trough = np.min(wf[peak_ind:])
        post_trough_ind = np.argmin(wf[peak_ind:])+1
        pre_trough = np.min(wf[:peak_ind])
        pre_trough_ind = peak_ind - np.argmin(wf[:peak_ind])  
    curr_trough_post = post_trough/peak # positive if same sign, negative if opposite sign
    curr_trough_pre = pre_trough/peak # positive if same sign, negative if opposite sign
    # Convert sample indices to ms for slope and integral calculations
    post_trough_ind_ms = post_trough_ind * ms_per_sample
    pre_trough_ind_ms = pre_trough_ind * ms_per_sample
    curr_trough_loc_slope_post = ((peak - post_trough)/peak)/post_trough_ind_ms # normalized amplitude change per ms
    curr_trough_slope = (post_trough/peak)/post_trough_ind_ms  # trough_post_ratio per ms
    curr_trough_loc_inte_post = ((post_trough)/peak)*post_trough_ind_ms  # normalized amplitude × ms
    curr_trough_loc_slope_pre = ((peak - pre_trough)/peak)/pre_trough_ind_ms # normalized amplitude change per ms
    curr_trough_loc_inte_pre = ((pre_trough)/peak)*pre_trough_ind_ms # normalized amplitude × ms
    pre_peak_wf = wf[:peak_ind]
    post_peak_wf = wf[peak_ind:]
    curr_pre_space_raw = np.sum(pre_peak_wf[pre_peak_wf>0])
    curr_post_space_raw = np.sum(post_peak_wf[post_peak_wf>0])

    curr_symmetry_inte = curr_trough_loc_inte_post - curr_trough_loc_inte_pre  # positive if same sign, negative if opposite sign
    curr_symmetry_slope = curr_trough_loc_slope_post - curr_trough_loc_slope_pre  # positive if same sign, negative if opposite sign
    
    curr_trough_sum = (curr_pre_space_raw + curr_post_space_raw)/peak
    curr_slope_sum = curr_trough_loc_slope_post + curr_trough_loc_slope_pre
    

    # find samples where the waveform crosses the half peak threshold to infer half width
    half_peak = peak / 2
    wf_half_crossings = np.where(np.diff(np.sign(wf - half_peak))!=0)[0]
    if len(wf_half_crossings) < 2:
        post_crossing = np.nan
        pre_crossing = np.nan
    else: 
        if np.all(wf_half_crossings < peak_ind) or np.all(wf_half_crossings > peak_ind):
            post_crossing = np.nan
            pre_crossing = np.nan
        else:
            post_crossing = np.min(wf_half_crossings[wf_half_crossings > peak_ind]) - peak_ind
            pre_crossing = peak_ind - np.max(wf_half_crossings[wf_half_crossings < peak_ind])

    half_w.append((post_crossing + pre_crossing) * ms_per_sample)  # half width in ms
    trough_w.append((post_trough_ind + pre_trough_ind) * ms_per_sample)  # trough width in ms
    post_w.append(post_trough_ind * ms_per_sample)  # post trough width in ms
    pre_w.append(pre_trough_ind * ms_per_sample)  # pre-peak trough width in ms
    pre_half.append(pre_crossing * ms_per_sample)  # pre-peak half-width in ms
    post_half.append(post_crossing * ms_per_sample)  # post-peak half-width in ms

    trough_post_ratio_1D.append(curr_trough_post)
    trough_pre_ratio_1D.append(curr_trough_pre)

    post_slope.append(curr_trough_loc_slope_post)
    post_trough_slope.append(curr_trough_slope)
    post_space.append(curr_trough_loc_inte_post)
    pre_slope.append(curr_trough_loc_slope_pre)
    pre_space.append(curr_trough_loc_inte_pre)
    post_space_raw.append(curr_post_space_raw)
    pre_space_raw.append(curr_pre_space_raw)

    symmetry_inte.append(curr_trough_loc_inte_post - curr_trough_loc_inte_pre)
    symmetry_half.append((post_crossing - pre_crossing) * ms_per_sample)  # in ms
    symmetry_slope.append(curr_trough_loc_slope_post - curr_trough_loc_slope_pre)

    symmetry_inte_div.append(curr_trough_loc_inte_post/curr_trough_loc_inte_pre)
    symmetry_half_div.append(post_crossing/pre_crossing)  # ratio, dimensionless
    symmetry_slope_div.append(curr_trough_loc_slope_pre/curr_trough_loc_slope_post)
    symmetry_space_raw_div.append(curr_post_space_raw/curr_pre_space_raw)
    symmetry_trough_dis.append((post_trough_ind - pre_trough_ind) * ms_per_sample)  # in ms

    trough_sum.append(curr_trough_sum)
    slope_sum.append(curr_slope_sum)

wf_features = pd.DataFrame({'unit_id': combined_tagged_units_filtered['unit'],
                            'session': combined_tagged_units_filtered['session'],
                            'amp': combined_tagged_units_filtered['amp'],
                            'peak': combined_tagged_units_filtered['peak'],
                            'half_w': half_w,
                            'trough_w': trough_w,
                            'pre_half': pre_half,
                            'post_half': post_half,
                            'post_w': post_w,
                            'pre_w':pre_w,
                            'trough_post_ratio_1D': trough_post_ratio_1D,
                            'trough_pre_ratio_1D': trough_pre_ratio_1D,
                            'post_slope': post_slope,
                            'post_trough_slope': post_trough_slope,
                            'post_space': post_space,
                            'pre_slope': pre_slope,
                            'pre_space': pre_space,
                            'symmetry_inte': symmetry_inte,
                            'symmetry_half': symmetry_half,
                            'symmetry_slope': symmetry_slope,
                            'symmetry_inte_div': symmetry_inte_div,
                            'symmetry_half_div': symmetry_half_div,
                            'symmetry_slope_div': symmetry_slope_div,
                            'symmetry_space_raw_div': symmetry_space_raw_div,
                            'symmetry_trough_dis': symmetry_trough_dis,
                            'trough_sum': trough_sum,
                            'slope_sum': slope_sum,
                            'y_loc': combined_tagged_units_filtered['y_loc'],
                            'probe': combined_tagged_units_filtered['probe'],
                            })
wf_features['symmetry_inte_div_log'] = np.log(wf_features['symmetry_inte_div'] + 1e-6)
wf_features['symmetry_slope_div_log'] = np.log(wf_features['symmetry_slope_div'] + 1e-6)
wf_features['symmetry_half_div_log'] = np.log(wf_features['symmetry_half_div'] + 1e-6)
wf_features.rename(columns={'unit_id': 'unit'}, inplace=True)
# compute pcs
focus_features = [
                'post_w', 
                'trough_post_ratio_1D', 
                'post_trough_slope', 
                'pre_slope',
                'symmetry_slope_div_log',
                'symmetry_trough_dis', 
                'symmetry_inte_div_log', 
                ]


feature_mat = wf_features[focus_features]
# remove rows with nan
nan_ind = np.isnan(feature_mat).any(axis=1)
feature_mat = feature_mat[~nan_ind]
# zscore
feature_mat = zscore(feature_mat, axis=0, nan_policy='omit')

pca = PCA(n_components=3)
pca_result = pca.fit_transform(feature_mat)

pc_filtered = np.full((len(wf_features), 3), np.nan)
pc_filtered[~nan_ind] = pca_result

wf_features['wf_pc_1'] = np.nan
wf_features['wf_pc_2'] = np.nan
wf_features['wf_pc_3'] = np.nan
wf_features.loc[~nan_ind, 'wf_pc_1'] = pc_filtered[:, 0]
wf_features.loc[~nan_ind, 'wf_pc_2'] = pc_filtered[:, 1]
wf_features.loc[~nan_ind, 'wf_pc_3'] = pc_filtered[:, 2]

# save combined features with PCs
wf_features.to_csv(os.path.join(target_folder, f'combined_features.csv'), index=False)