"""
Step 11 of figure preparation pipeline: Generate fiber photometry features for analysis.

Prerequisites:
    NONE - This script is COMPLETELY INDEPENDENT.

    Data requirements:
    - hopkins_FP_session_assets.csv (photometry session metadata)
    - Per-session fiber photometry data files (if available)
    - Raw photometry signals (GCaMP/isosbestic channels)
    - Harp clock alignment for temporal synchronization
    - Trial event times for trial-aligned photometry
    - Session behavior tables for linking photometry to task events

Pipeline Position:
    Script #11 in sequence.txt (line 11) - FINAL step of the figure preparation pipeline.
    COMPLETELY INDEPENDENT - can run IN PARALLEL with ALL other scripts!
    Does NOT depend on any other figure_preparation script.

Purpose:
    Processes and extracts fiber photometry features for sessions with photometry recordings:
    - Baseline correction and normalization (ΔF/F calculation)
    - Motion artifact removal using isosbestic control channel
    - Temporal alignment with behavioral events using Harp clock
    - Trial-aligned photometry responses (stimulus, choice, outcome)
    - Peak detection and response latencies
    - Correlation with behavioral performance metrics
    - Exponential fitting to photometry decay kinetics

    Provides population-level DA/ACh/NE dynamics to complement single-unit recordings.

Input:
    - Per-session raw photometry data files
    - Behavioral trial tables with event timestamps
    - Session tables for trial-by-trial conditions
    - Harp clock data for temporal alignment

Output:
    - combined_photometry_tbl.pkl: DataFrame with photometry features per session
    - Includes: trial-aligned ΔF/F traces, peak amplitudes, response latencies,
      baseline-corrected signals, correlations with behavior
    - Per-session photometry plots and quality metrics

Usage:
    Run after all electrophysiology and behavioral pipelines complete. Only processes
    sessions with available photometry data. Can run in parallel with joblib for
    efficiency across multiple sessions.
"""
# %%
import numpy as np
import pandas as pd
import sys
from pathlib import Path
# Resolve code/beh_ephys_analysis (the folder containing `utils`) relative to this
# file's location, so imports work no matter where the repo is checked out.
import os
_beh_ephys_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _beh_ephys_root not in sys.path:
    sys.path.insert(0, _beh_ephys_root)
from utils.capsule_migration import CAPSULE_ROOT
from utils.ephys_functions import fitSpikeModelG
import platform
import shutil
from utils.beh_functions import session_dirs, get_session_tbl, makeSessionDF, parseSessionID
from utils.photometry_utils import get_FP_data
from utils.capsule_migration import capsule_directories
from matplotlib import pyplot as plt
from IPython.display import display
from scipy.signal import find_peaks
from harp.clock import align_timestamps_to_anchor_points
from scipy.signal import butter, filtfilt, medfilt, sosfiltfilt
from scipy.optimize import curve_fit
import json
from sklearn.linear_model import LinearRegression
from matplotlib.gridspec import GridSpec
import pickle
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time

# %matplotlib widget
import re
import random
from utils.photometry_combine import population_GLM, plot_tuning_curve, plot_psth, population_GLM_ani
from contextlib import redirect_stdout
capsule_dirs = capsule_directories()
# %%
session_csv = CAPSULE_ROOT + '/code/data_management/hopkins_FP_session_assets.csv'
session_tbl = pd.read_csv(session_csv)
session_list = session_tbl['session_id'].tolist()
target_folder = os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'photometry_regressions')
save_dir = target_folder

# %%
# Switch: session-wise GLM for SVS related variables
region_curr = 'PL'
channel_curr = 'G_tri-exp_mc' 
align_curr = 'goCue_start_time'
window_size_curr = 0.75
thresh_curr = 0.5
formula = 'spikes ~ 1+svs+outcome'
post = 2.5
pre = 3

params_dict_svs = {
    'region': region_curr,
    'channel': channel_curr,
    'align': align_curr,
    'window_size': window_size_curr,
    'formula': formula,
    'pre_time': pre,
    'post_time': post,
    'thresh': thresh_curr,
    'step_size': 0.1,
    'polar_regressors': ['svs', 'outcome']
}

print(
    f"Session-wise GLM with parameters: \n"
    f"Processing region: {params_dict_svs['region']}, "
    f"channel: {params_dict_svs['channel']}, "
    f"align: {params_dict_svs['align']}, "
    f"window size: {params_dict_svs['window_size']}, "
    f"threshold: {params_dict_svs['thresh']}, "
    f"formula: {params_dict_svs['formula']}"
)
session_list_test = sessions_current = [
    'behavior_699462_2024-01-13_15-11-33',
    'behavior_749472_2025-01-08_13-43-07',
    'behavior_699472_2024-01-02_13-53-59',
    'behavior_749472_2025-01-04_14-13-30',
    'behavior_749472_2025-01-01_16-22-33',
]
results = population_GLM(session_list_test, **params_dict_svs)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
formula_clean = (
    params_dict_svs["formula"]
    .replace(" ", "")
    .replace("+", "_")
    .replace("*", "x")
    .split("~")[1]
)
file_name = (
    f"population_GLM_session-wise_{params_dict_svs['region']}_{params_dict_svs['channel']}_"
    f"{params_dict_svs['align']}_win{params_dict_svs['window_size']}_"
    f"thresh{params_dict_svs['thresh']}_formula_{formula_clean}_test.pkl"
)
with open(os.path.join(save_dir, file_name), 'wb') as f:
    pickle.dump(params_dict_svs, f)
    pickle.dump(results, f)
print(f'Saved results to {os.path.join(save_dir, file_name)}')
results['fig'].savefig(os.path.join(save_dir, file_name.replace('.pkl', '.pdf')), dpi=300)
