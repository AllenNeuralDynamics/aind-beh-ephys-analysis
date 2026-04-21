# %%
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append('/root/capsule/code/beh_ephys_analysis')
from utils.ephys_functions import fitSpikeModelG
import platform
import os
from pathlib import Path
import shutil
from utils.beh_functions import session_dirs, get_session_tbl, makeSessionDF, parseSessionID
from utils.photometry_utils import get_FP_data
from utils.capsule_migration import capsule_directories
from matplotlib import pyplot as plt
from IPython.display import display
from scipy.signal import find_peaks
from harp.clock import align_timestamps_to_anchor_points
import numpy as np
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
from matplotlib.gridspec import GridSpec
from utils.photometry_combine import population_GLM, plot_tuning_curve, plot_psth, population_GLM_ani
from contextlib import redirect_stdout
capsule_dirs = capsule_directories()
# %%
session_csv = '/root/capsule/code/data_management/hopkins_FP_session_assets.csv'
session_tbl = pd.read_csv(session_csv)
session_list = session_tbl['session_id'].tolist()
target_folder = os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'photometry_regressions')
save_dir = target_folder
# %%
# session-wise GLM for reward-related variables
region_curr = 'PL'
channel_curr = 'G_tri-exp_mc'
align_curr = 'choice_time'
window_size_curr = 2
thresh_curr = 0.5
formula = 'spikes ~ 1+outcome*ipsi+Qchosen+iso'
post = 3
pre = 2

params_dict_reward = {
    'region': region_curr,
    'channel': channel_curr,
    'align': align_curr,
    'window_size': window_size_curr,
    'formula': formula,
    'pre_time': pre,
    'post_time': post,
    'thresh': thresh_curr,
    'step_size': 0.1,
    'polar_regressors': ['outcome', 'Qchosen']
}

print(
    f"Session-wise GLM with parameters: \n"
    f"Processing region: {params_dict_reward['region']}, "
    f"channel: {params_dict_reward['channel']}, "
    f"align: {params_dict_reward['align']}, "
    f"window size: {params_dict_reward['window_size']}, "
    f"threshold: {params_dict_reward['thresh']}, "
    f"formula: {params_dict_reward['formula']}"
)

results = population_GLM(session_list, **params_dict_reward)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
formula_clean = (
    params_dict_reward["formula"]
    .replace(" ", "")
    .replace("+", "_")
    .replace("*", "x")
    .split("~")[1]
)
file_name = (
    f"population_GLM_session-wise_{params_dict_reward['region']}_{params_dict_reward['channel']}_"
    f"{params_dict_reward['align']}_win{params_dict_reward['window_size']}_"
    f"thresh{params_dict_reward['thresh']}_formula_{formula_clean}.pkl"
)
with open(os.path.join(save_dir, file_name), 'wb') as f:
    pickle.dump(params_dict_reward, f)
    pickle.dump(results, f)
print(f'Saved results to {os.path.join(save_dir, file_name)}')
results['fig'].savefig(os.path.join(save_dir, file_name.replace('.pkl', '.pdf')), dpi=300)

# %% Animal-wise GLM for reward-related variables
print(
    f"Animal-wise GLM with parameters: \n"
    f"Processing region: {params_dict_reward['region']}, "
    f"channel: {params_dict_reward['channel']}, "
    f"align: {params_dict_reward['align']}, "
    f"window size: {params_dict_reward['window_size']}, "
    f"threshold: {params_dict_reward['thresh']}, "
    f"formula: {params_dict_reward['formula']}"
)

results = population_GLM_ani(session_list, **params_dict_reward)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
formula_clean = (
    params_dict_reward["formula"]
    .replace(" ", "")
    .replace("+", "_")
    .replace("*", "x")
    .split("~")[1]
)
file_name = (
    f"population_GLM_animal-wise_{params_dict_reward['region']}_{params_dict_reward['channel']}_"
    f"{params_dict_reward['align']}_win{params_dict_reward['window_size']}_"
    f"thresh{params_dict_reward['thresh']}_formula_{formula_clean}.pkl"
)
with open(os.path.join(save_dir, file_name), 'wb') as f:
    pickle.dump(params_dict_reward, f)
    pickle.dump(results, f)
print(f'Saved results to {os.path.join(save_dir, file_name)}')
results['fig'].savefig(os.path.join(save_dir, file_name.replace('.pkl', '.pdf')), dpi=300)

# %%
# session-wise GLM for engagement related variables
region_curr = 'PL'
channel_curr = 'G_tri-exp_mc'
align_curr = 'goCue_start_time'
window_size_curr = 2
thresh_curr = 0.5
formula = 'spikes ~ 1+hit+iso'
post = 2
pre = 3

params_dict_hit = {
    'region': region_curr,
    'channel': channel_curr,
    'align': align_curr,
    'window_size': window_size_curr,
    'formula': formula,
    'pre_time': pre,
    'post_time': post,
    'thresh': thresh_curr,
    'step_size': 0.1,
    'polar_regressors': ['hit', 'iso']
}

print(
    f"Session-wise GLM with parameters: \n"
    f"Processing region: {params_dict_hit['region']}, "
    f"channel: {params_dict_hit['channel']}, "
    f"align: {params_dict_hit['align']}, "
    f"window size: {params_dict_hit['window_size']}, "
    f"threshold: {params_dict_hit['thresh']}, "
    f"formula: {params_dict_hit['formula']}"
)

results = population_GLM(session_list, **params_dict_hit)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
formula_clean = (
    params_dict_hit["formula"]
    .replace(" ", "")
    .replace("+", "_")
    .replace("*", "x")
    .split("~")[1]
)
file_name = (
    f"population_GLM_session-wise_{params_dict_hit['region']}_{params_dict_hit['channel']}_"
    f"{params_dict_hit['align']}_win{params_dict_hit['window_size']}_"
    f"thresh{params_dict_hit['thresh']}_formula_{formula_clean}.pkl"
)
with open(os.path.join(save_dir, file_name), 'wb') as f:
    pickle.dump(params_dict_hit, f)
    pickle.dump(results, f)
print(f'Saved results to {os.path.join(save_dir, file_name)}')
results['fig'].savefig(os.path.join(save_dir, file_name.replace('.pkl', '.pdf')), dpi=300)

# %%
# animal-wise GLM for engagement related variables
print(
    f"Animal-wise GLM with parameters: \n"
    f"Processing region: {params_dict_hit['region']}, "
    f"channel: {params_dict_hit['channel']}, "
    f"align: {params_dict_hit['align']}, "
    f"window size: {params_dict_hit['window_size']}, "
    f"threshold: {params_dict_hit['thresh']}, "
    f"formula: {params_dict_hit['formula']}"
)

results = population_GLM_ani(session_list, **params_dict_hit)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
formula_clean = (
    params_dict_hit["formula"]
    .replace(" ", "")
    .replace("+", "_")
    .replace("*", "x")
    .split("~")[1]
)
file_name = (
    f"population_GLM_animal-wise_{params_dict_hit['region']}_{params_dict_hit['channel']}_"
    f"{params_dict_hit['align']}_win{params_dict_hit['window_size']}_"
    f"thresh{params_dict_hit['thresh']}_formula_{formula_clean}.pkl"
)
with open(os.path.join(save_dir, file_name), 'wb') as f:
    pickle.dump(params_dict_hit, f)
    pickle.dump(results, f)
print(f'Saved results to {os.path.join(save_dir, file_name)}')
results['fig'].savefig(os.path.join(save_dir, file_name.replace('.pkl', '.pdf')), dpi=300)

# %%
# Switch: session-wise GLM for SVS related variables
region_curr = 'PL'
channel_curr = 'G_tri-exp_mc' 
align_curr = 'goCue_start_time'
window_size_curr = 0.75
thresh_curr = 0.5
formula = 'spikes ~ 1+svs+iso'
post = 2
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
    'polar_regressors': ['hit', 'iso']
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

results = population_GLM(session_list, **params_dict_svs)

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
    f"thresh{params_dict_svs['thresh']}_formula_{formula_clean}.pkl"
)
with open(os.path.join(save_dir, file_name), 'wb') as f:
    pickle.dump(params_dict_svs, f)
    pickle.dump(results, f)
print(f'Saved results to {os.path.join(save_dir, file_name)}')
results['fig'].savefig(os.path.join(save_dir, file_name.replace('.pkl', '.pdf')), dpi=300)

# %%
# Switch: animal-wise GLM for SVS related variables

print(
    f"Animal-wise GLM with parameters: \n"
    f"Processing region: {params_dict_svs['region']}, "
    f"channel: {params_dict_svs['channel']}, "
    f"align: {params_dict_svs['align']}, "
    f"window size: {params_dict_svs['window_size']}, "
    f"threshold: {params_dict_svs['thresh']}, "
    f"formula: {params_dict_svs['formula']}"
)

results = population_GLM_ani(session_list, **params_dict_svs)

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
    f"population_GLM_animal-wise_{params_dict_svs['region']}_{params_dict_svs['channel']}_"
    f"{params_dict_svs['align']}_win{params_dict_svs['window_size']}_"
    f"thresh{params_dict_svs['thresh']}_formula_{formula_clean}.pkl"
)
with open(os.path.join(save_dir, file_name), 'wb') as f:
    pickle.dump(params_dict_svs, f)
    pickle.dump(results, f)
print(f'Saved results to {os.path.join(save_dir, file_name)}')
results['fig'].savefig(os.path.join(save_dir, file_name.replace('.pkl', '.pdf')), dpi=300)


