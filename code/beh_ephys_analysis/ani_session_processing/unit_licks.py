import os, sys
# Resolve code/beh_ephys_analysis (the folder containing `utils`) relative to this
# file's location, so imports work no matter where the repo is checked out.
_anchor = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath(os.getcwd())
while _anchor != os.path.dirname(_anchor):
    _beh_ephys_root = os.path.join(_anchor, "code", "beh_ephys_analysis")
    if os.path.isdir(os.path.join(_beh_ephys_root, "utils")):
        if _beh_ephys_root in sys.path:
            sys.path.remove(_beh_ephys_root)
        sys.path.insert(0, _beh_ephys_root)
        break
    _anchor = os.path.dirname(_anchor)
from utils.capsule_migration import CAPSULE_ROOT
# %%
import sys
import os
# Add repoA to sys.path at the beginning
curr_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)
else:
    # move it to the front if it's already there
    sys.path.remove(curr_path)
    sys.path.insert(0, curr_path)
# %%
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
from utils.lick_utils import load_licks, load_licks_video
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
# %%

def plot_unit_licks(session, unit_ids = None, opto_only=True, data_type='curated', plot_licks=False, video = False):
    """
    Analyze and plot unit firing rates aligned to lick events.

    Creates raster plots and PSTHs showing neural responses to lick trains,
    separated by left vs right licks and early vs late in session.

    Parameters:
        session (str): Session identifier.
        unit_ids (list or None): Specific unit IDs to analyze. If None, analyze qualifying units.
        opto_only (bool): If True, only analyze opto-tagged units.
        data_type (str): Type of data to use ('curated' or 'raw').
        plot_licks (bool): If True, generate diagnostic lick detection plots.
        video (bool): If True, use licks detected from video; otherwise use behavior licks.

    Returns:
        None: Saves lick-aligned neural activity plots as PDF files in session directory.
    """
    session_tbl = get_session_tbl(session)
    if session_tbl is None or len(session_tbl) == 0:
        print(f'No session table found for session {session}.')
        return
    unit_tbl = get_unit_tbl(session, 'curated')
    if unit_tbl is None or len(unit_tbl) == 0:
        print(f'No units found for session {session} with data type {data_type}.')
        return
    if video:
        licks = load_licks_video(session, plot=plot_licks)
        if licks is None:
            print(f'No lick data found for session {session}. Loading behavior licks.')
            licks = load_licks(session, plot=plot_licks)
    else:
        licks = load_licks(session, plot=plot_licks)
    session_dir = session_dirs(session)
    qm_dir = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_dir, 'r') as f:
        qm = json.load(f)
    rec_start = qm['ephys_cut'][0]
    rec_end = qm['ephys_cut'][1]
    if opto_only:
        unit_tbl = unit_tbl[unit_tbl['opto_pass'] == True].reset_index(drop=True)
    if len(unit_tbl) == 0:
        print(f'No opto-responsive units found for session {session}.')
        return
    if unit_ids is None:
        unit_ids = unit_tbl['unit_id'].values
    for unit_id in unit_ids:
        spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]
        unit_drift = load_drift(session, unit_id)
        session_tbl_curr = session_tbl.copy()
        spike_times_curr = spike_times.copy()
        lick_starts_L = licks['lick_trains_L']['train_starts']
        lick_starts_R = licks['lick_trains_R']['train_starts']
        lick_starts_in_trial_L = licks['lick_trains_L']['in_trial']
        lick_starts_in_trial_R = licks['lick_trains_R']['in_trial']
        L_mask = np.ones_like(lick_starts_in_trial_L, dtype=bool)
        R_mask = np.ones_like(lick_starts_in_trial_R, dtype=bool)
        L_mask &= lick_starts_L >= rec_start
        R_mask &= lick_starts_R >= rec_start
        L_mask &= lick_starts_L <= rec_end
        R_mask &= lick_starts_R <= rec_end
        
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_tbl_curr = session_tbl_curr[session_tbl_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
                L_mask &= lick_starts_L >= unit_drift['ephys_cut'][0]
                R_mask &= lick_starts_R >= unit_drift['ephys_cut'][0]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_tbl_curr = session_tbl_curr[session_tbl_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
                L_mask &= lick_starts_L <= unit_drift['ephys_cut'][1]
                R_mask &= lick_starts_R <= unit_drift['ephys_cut'][1]

        lick_starts_in_trial_L = lick_starts_in_trial_L[L_mask]
        lick_starts_in_trial_R = lick_starts_in_trial_R[R_mask]
        lick_starts_L = lick_starts_L[L_mask]
        lick_starts_R = lick_starts_R[R_mask]

        lick_starts_all = np.concatenate([lick_starts_L, lick_starts_R])
        lick_starts_in_trial_all = np.concatenate([lick_starts_in_trial_L, lick_starts_in_trial_R])
        lick_sides = np.concatenate([np.zeros_like(lick_starts_L), np.ones_like(lick_starts_R)])  # 0 for L, 1 for R
        sort_idx = np.argsort(lick_starts_all)
        lick_starts_all = lick_starts_all[sort_idx]
        lick_starts_in_trial_all = lick_starts_in_trial_all[sort_idx]
        lick_sides = lick_sides[sort_idx]
        fig = plt.figure(figsize=(12, 8))
        time_bin = 0.5
        tb = -4
        tf = 4
        gs = gridspec.GridSpec(3, 4)
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'red'])
        fig, ax1, ax2 = plot_raster_rate(spike_times_curr, lick_starts_all[~lick_starts_in_trial_all], 
                        lick_sides[~lick_starts_in_trial_all], [-1, 0.5, 2], ['L', 'R'], custom_cmap, 
                        fig, gs[:, 0], time_bin = time_bin, kernel=True, tau_rise=0.01, tau_decay=0.2,
                        tb= tb, tf = tf)
        ax1.set_title('Spont licks')
        fig, ax1, ax2 = plot_raster_rate(spike_times_curr, lick_starts_L, 
                        lick_starts_in_trial_L, [-1, 0.5, 2], ['out', 'in'], custom_cmap, 
                        fig, gs[:, 1], time_bin = time_bin, kernel=True, tau_rise=0.01, tau_decay=0.2,
                        tb = tb, tf = tf)
        ax1.set_title('Licks L')
        fig, ax1, ax2 = plot_raster_rate(spike_times_curr, lick_starts_R,
                        lick_starts_in_trial_R, [-1, 0.5, 2], ['out', 'in'], custom_cmap, 
                        fig, gs[:, 2], time_bin = time_bin, kernel=True, tau_rise=0.01, tau_decay=0.2,
                        tb = tb, tf = tf)
        ax1.set_title('Licks R')
        fig, ax1, ax2 = plot_raster_rate(spike_times_curr, lick_starts_all,
                        lick_starts_in_trial_all, [-1, 0.5, 2], ['out', 'in'], custom_cmap, 
                        fig, gs[:, 3], time_bin = time_bin, kernel=True, tau_rise=0.01, tau_decay=0.2,
                        tb = tb, tf = tf)
        ax1.set_title('Licks All')

        plt.suptitle(f'Session: {session}, Unit ID: {unit_id}', fontsize=16)
        plt.tight_layout()

        save_path = session_dir['ephys_fig_dir_curated'] + '/lick_raster_rate/'
        if 'data' not in save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig(save_path + f'unit_{str(unit_id).split(".0")[0]}_lick_raster_{opto_only}.pdf')
            plt.close('all')
        else:
            return fig



# %%
if __name__ == "__main__":
    session = 'behavior_782394_2025-04-24_12-07-34'
    opto_only = True
    dfs = [pd.read_csv(CAPSULE_ROOT + '/code/data_management/session_assets.csv'),
            pd.read_csv(CAPSULE_ROOT + '/code/data_management/hopkins_session_assets.csv')]
    df = pd.concat(dfs)
    session_list = df['session_id'].values.tolist()
    # test
    plot_unit_licks(session, opto_only=opto_only, plot_licks=True)

    # # %%
    # from joblib import Parallel, delayed
    # def safe_process(session):
    #     try:
    #         plot_unit_licks(session, opto_only=opto_only, plot_licks=False)
    #     except Exception as e:
    #         print(f"Error processing session {session}: {e}")
    # Parallel(n_jobs=-4)(delayed(safe_process)(session) for session in session_list)



