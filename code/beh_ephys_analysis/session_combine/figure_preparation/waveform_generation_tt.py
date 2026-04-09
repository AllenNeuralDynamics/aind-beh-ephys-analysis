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

criteria_name = 'waveform_TT'
waveform_version = '_raw' # 'wf_2D' for 'wf_2D_raw'
target_folder = os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'waveforms_tt')
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# load constraints and data
with open(os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'combined_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
    combined_tagged_units = pickle.load(f)
with open(os.path.join('/root/capsule/code/beh_ephys_analysis/session_combine/metrics', f'{criteria_name}.json'), 'r') as f:
    constraints = json.load(f)

combined_tagged_units_filtered, combined_tagged_units, fig, axes = apply_qc(combined_tagged_units, constraints)
fig.savefig(os.path.join(target_folder, 'quality_metrics.pdf'))

def zero_crossings_linear(x, y):
    """
    Find x positions where y crosses zero using linear interpolation.

    Parameters
    ----------
    x, y : 1D arrays of same length

    Returns
    -------
    x0 : 1D array
        x positions of zero crossings
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be 1D arrays of the same length")

    # Find sign changes (exclude exact zeros handled below)
    sign_change = (y[:-1] * y[1:] < 0)

    x0 = x[:-1][sign_change] - y[:-1][sign_change] * (
        x[1:][sign_change] - x[:-1][sign_change]
    ) / (
        y[1:][sign_change] - y[:-1][sign_change]
    )

    # Handle exact zeros (y == 0)
    exact = np.where(y == 0)[0]
    if exact.size > 0:
        x0 = np.sort(np.concatenate([x0, x[exact]]))

    return x0


# Extract waveform features for all units that passed QC
# Extract waveform feature
wf_norm = []
wf_2D_norm = []

half_w = []
trough_w = []
pre_half = []
post_half = []
post_w = []

trough_post_ratio_1D = []
trough_pre_ratio_1D = []

post_slope = []
post_trough_slope = []
post_space = []
post_space_raw = []
pre_slope = []
pre_space = []
pre_space_raw = []

symmetry_inte = []
symmetry_half = []
symmetry_slope = []
symmetry_inte_div = []
symmetry_half_div = []
symmetry_slope_div = []
symmetry_space_raw_div = []
symmetry_trough_dis = []

trough_sum = []
slope_sum = []

# wf_norm = combined_tagged_units_filtered['wf']/np.abs(combined_tagged_units_filtered['peak'])
# wf_2D_norm = combined_tagged_units_filtered['wf_2d']/np.abs(combined_tagged_units_filtered['peak'])

wf_norm = combined_tagged_units_filtered[f'wf{waveform_version}']/np.abs(combined_tagged_units_filtered[f'peak{waveform_version}'])
# wf_2D_norm = combined_tagged_units_filtered['wf_2d']/np.abs(combined_tagged_units_filtered['peak'])
for rows in combined_tagged_units_filtered.iterrows():
    print(f'Processing unit {rows[1]["unit"]} of session {rows[1]["session"]}')
    # wf = rows[1]['wf']
    # peak = rows[1]['peak']
    # print(rows[1]['session'])
    # print(rows[1]['unit'])
    wf = rows[1][f'wf{waveform_version}'][:32*5]
    wf_bl = np.nanmean(wf[:5])
    peak = rows[1][f'peak{waveform_version}'] - wf_bl
    wf = wf - wf_bl
    peak_ind = np.argmin(wf)

    if np.abs(wf[0]-wf[-1])>50:
        half_w.append(np.nan)  # half width in samples
        trough_w.append(np.nan)  # trough width in samples
        post_w.append(np.nan)  # post trough width in samples
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
    curr_trough_loc_slope_post = ((peak - post_trough)/peak)/post_trough_ind
    curr_trough_slope = (post_trough/peak)/post_trough_ind
    curr_trough_loc_inte_post = ((post_trough)/peak)*post_trough_ind
    curr_trough_loc_slope_pre = ((peak - pre_trough)/peak)/pre_trough_ind # positive if same sign, negative if opposite sign
    curr_trough_loc_inte_pre = ((pre_trough)/peak)*pre_trough_ind # positive if same sign, negative if opposite sign
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
    # wf_half_crossings = np.where(np.diff(np.sign(wf - half_peak))!=0)[0]+1  # indices where waveform crosses half peak
    wf_half_crossings = zero_crossings_linear(np.arange(len(wf)), wf - half_peak)
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

    half_w.append(post_crossing + pre_crossing)  # half width in samples
    trough_w.append(post_trough_ind + pre_trough_ind)  # trough width in samples
    post_w.append(post_trough_ind)  # post trough width in samples
    pre_half.append(pre_crossing)
    post_half.append(post_crossing)

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
    symmetry_half.append(post_crossing - pre_crossing)
    symmetry_slope.append(curr_trough_loc_slope_post - curr_trough_loc_slope_pre)

    symmetry_inte_div.append(curr_trough_loc_inte_post/curr_trough_loc_inte_pre)
    symmetry_half_div.append(post_crossing/pre_crossing)
    symmetry_slope_div.append(curr_trough_loc_slope_post/curr_trough_loc_slope_pre)
    symmetry_space_raw_div.append(curr_post_space_raw/curr_pre_space_raw)
    symmetry_trough_dis.append(post_trough_ind - pre_trough_ind)

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

wf_features.to_csv(os.path.join(target_folder, f'combined_features.csv'), index=False)