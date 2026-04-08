# Combined all antidromic information into a single dataframe and save

# %%
# Standard library
import os
import sys
import re
import json
import pickle
import glob
import ast
import warnings
from pathlib import Path
import importlib
sys.path.append('/root/capsule/code/beh_ephys_analysis')

# Scientific libraries
import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as signal
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# SpikeInterface
import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw

# Progress bar
from tqdm import tqdm
import seaborn as sns
from trimesh import load_mesh

# IPython display
from IPython.display import clear_output

# Open Ephys
from open_ephys.analysis import Session

# AIND analysis and utils
from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import load_nwb
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from aind_ephys_utils import align, sort

# Local utilities
import utils.analysis_funcs as af
import utils.plotting_funcs as pf
from utils.beh_functions import session_dirs
from utils.combine_tools import apply_qc
from utils.ccf_utils import pir_to_lps
from utils.plot_utils import combine_pdf_big
from utils.capsule_migration import capsule_directories

import k3d
from scipy.stats import pearsonr


# %% [markdown]
# # Pack data

# %%
criteria_name = 'basic_ephys_low'
capsure_dirs = capsule_directories()
# %%
# load constraints and data
with open(os.path.join(capsure_dirs["manuscript_fig_prep_dir"], 'combine_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
    combined_tagged_units = pickle.load(f)
    
with open(os.path.join('/root/capsule/code/beh_ephys_analysis/session_combine/metrics', f'{criteria_name}.json'), 'r') as f:
    constraints = json.load(f)
beh_folder = os.path.join(capsure_dirs['manuscript_fig_prep_dir'], 'antidromic_analysis')
if not os.path.exists(beh_folder):
    os.makedirs(beh_folder)
# start with a mask of all True
mask = pd.Series(True, index=combined_tagged_units.index)

# %%
combined_tagged_units_filtered, combined_tagged_units, fig, axes = apply_qc(combined_tagged_units, constraints)

# %% [markdown]
# # Load antidromic units

# %%
session_list = combined_tagged_units_filtered[combined_tagged_units_filtered['probe']=='2']['session'].unique().tolist()
file = os.path.join(beh_folder, 'combined_antidromic_results.pkl')
re_compute = True
# if os.path.exists(file):
#     print(f'Loading antidromic results from {file}')
#     with open(file, 'rb') as f:
#         concatenate_antidromic_results = pickle.load(f)
# else:
#     print('Collecting antidromic analysis for sessions:')
#     re_compute = True

# %%
if re_compute:
    concatenate_antidromic_results_all = []
    for session in session_list:
        session_dir = session_dirs(session)
        save_dir = os.path.join(session_dir['opto_dir_curated'], f'{session}_antidromic_results.pkl')
        if os.path.exists(save_dir):
            with open(save_dir, 'rb') as f:
                merged_df = pickle.load(f)
            merged_df['session'] = session
            concatenate_antidromic_results_all.append(merged_df)
    concatenate_antidromic_results = pd.concat(concatenate_antidromic_results_all, ignore_index=True)
    concatenate_antidromic_results.rename(columns={'unit_id': 'unit'}, inplace=True)
    # file = os.path.join(beh_folder, 'combined_antidromic_results.pkl')
    # with open(file, 'wb') as f:
    #     pickle.dump(concatenate_antidromic_results, f)

# %% [markdown]
# # Process target region

# %%
focuses = ['PrL', 'S1']
focus_save_dir = os.path.join(beh_folder, '_'.join(focuses))
os.makedirs(focus_save_dir, exist_ok=True)

all_focus_dfs = []

# --- Process each focus separately ---
for focus in focuses:
    cols_to_keep = [
        col for col in concatenate_antidromic_results.columns
        if col[1] in (f'surface_{focus}', '')  # keep both surface_PrL and no-site columns
    ]
    df_focus = concatenate_antidromic_results.loc[:, cols_to_keep]

    # Flatten MultiIndex if needed
    if isinstance(df_focus.columns, pd.MultiIndex):
        df_focus.columns = df_focus.columns.get_level_values(0)

    combined_df = df_focus.copy()
    combined_df['focus'] = focus

    # --- Derived metrics ---
    combined_df['t_collision'] = -combined_df['t_collision']
    combined_df['p_auto_inhi_log'] = -np.log10(combined_df['p_auto_inhi'] + 1e-20)
    combined_df['p_collision_log'] = -np.log10(combined_df['p_collision'] + 1e-20)
    combined_df['p_antidromic_log'] = -np.log10(combined_df['p_antidromic'] + 1e-20)

    # --- Tier logic ---
    combined_df['tier_1'] = (
        (combined_df['jitter'] < 0.01)
        & (combined_df['p_antidromic'] < 0.005)
        & (combined_df['t_antidromic'] > 0)
        & (combined_df['p_collision'] < 0.005)
        & (combined_df['t_collision'] > 0)
    ).astype(float)

    combined_df['tier_2'] = (
        (combined_df['jitter'] < 0.01)
        & (combined_df['p_antidromic'] < 0.005)
        & (combined_df['t_antidromic'] > 0)
    ).astype(float)

    combined_df['tier_1_long'] = (
        (combined_df['p_antidromic'] < 0.005)
        & (combined_df['t_antidromic'] > 0)
        & (combined_df['p_collision'] < 0.005)
        & (combined_df['t_collision'] > 0)
    ).astype(float)

    combined_df['tier_2_long'] = (
        (combined_df['p_antidromic'] < 0.005)
        & (combined_df['t_antidromic'] > 0)
    ).astype(float)

    combined_df['short'] = (
        (combined_df['jitter'] < 0.01)
        & (combined_df['antidromic_latency'] >= 0.025)
    ).astype(float)

    all_focus_dfs.append(combined_df)

# --- Combine all focuses ---
combined_all_focus_df = pd.concat(all_focus_dfs, ignore_index=True)

# --- Clean up tier columns ---
tier_cols = ['tier_1', 'tier_2', 'tier_1_long', 'tier_2_long', 'short']
# for c in tier_cols:
#     if c in combined_all_focus_df.columns:
#         combined_all_focus_df[c] = combined_all_focus_df[c].fillna(0).astype(float)

# --- Find the best (highest t_collision) per session/unit ---
combined_all_focus_df['_tcol'] = combined_all_focus_df['t_collision'].fillna(-np.inf)
idx_best = combined_all_focus_df.groupby(['session', 'unit'])['_tcol'].idxmax()
best_rows = combined_all_focus_df.loc[idx_best].drop(columns=['_tcol']).copy()

# --- Compute OR (max) of tiers across focuses ---
tier_max = (
    combined_all_focus_df.groupby(['session', 'unit'])[tier_cols]
    .max()
    .reset_index()
)

# %%
# --- Record which focuses passed each tier ---
focus_pass_cols = {}
for tier in tier_cols:
    if tier in combined_all_focus_df.columns:
        focus_pass = (
            combined_all_focus_df[combined_all_focus_df[tier] > 0]
            .groupby(['session', 'unit'])['focus']
            .apply(lambda x: ','.join(sorted(set(x))))
            .reset_index(name=f'{tier}_focus')
        )
        focus_pass_cols[tier] = focus_pass


# %%
# Merge all focus-passing columns into a single dataframe
tier_focus_df = tier_max.copy()
for tier, focus_df in focus_pass_cols.items():
    tier_focus_df = tier_focus_df.merge(focus_df, on=['session', 'unit'], how='left')


# %%
# --- Merge with best row (highest t_collision) ---
final_combined_df = (
    best_rows.drop(columns=tier_cols, errors='ignore')
    .merge(tier_focus_df, on=['session', 'unit'], how='left')
)

# --- Save result ---
combined_df = combined_tagged_units_filtered.merge(final_combined_df, on=['session', 'unit'], how='inner')
output_pkl = os.path.join(focus_save_dir, 'combined_antidromic_results.pkl')
final_combined_df.to_pickle(output_pkl)
print(f"Saved combined dataframe (one row per session+unit): {output_pkl}")

