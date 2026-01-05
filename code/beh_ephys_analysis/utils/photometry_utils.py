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
import re
from utils.beh_functions import session_dirs, get_session_tbl, makeSessionDF
from utils.ephys_functions import plot_rate

def bin_timeseries_around_align(
    ts,
    align_times,
    step_size,
    bin_size,
    t_start,
    t_stop,
):
    """
    Bin a timeseries around multiple alignment times.

    Parameters
    ----------
    ts : dict
        {'time': array-like, 'value': array-like}
    align_times : array-like
        Times to align to (n_align,)
    step_size : float
        Step between consecutive bins (same units as time)
    bin_size : float
        Width of each bin
    t_start : float
        Start time relative to align_time
    t_stop : float
        Stop time relative to align_time

    Returns
    -------
    out : ndarray
        Shape (n_align, n_steps)
    bin_centers : ndarray
        Shape (n_steps,)
    """

    time = np.asarray(ts["time"])
    value = np.asarray(ts["value"])
    align_times = np.asarray(align_times)

    # Bin starts relative to alignment
    bin_starts = np.arange(t_start, t_stop-bin_size, step_size)
    bin_starts = bin_starts[bin_starts + bin_size <= t_stop]
    n_steps = len(bin_starts)

    out = np.full((len(align_times), n_steps), np.nan)

    for i, t_align in enumerate(align_times):
        for j, bs in enumerate(bin_starts):
            t0 = t_align + bs
            t1 = t0 + bin_size

            mask = (time >= t0) & (time < t1)
            if np.any(mask):
                if np.mean(np.isnan(value[mask])) < 0.5:
                    out[i, j] = np.nanmean(value[mask])
                else:
                    out[i, j] = np.nan

    bin_centers = bin_starts + bin_size / 2

    return out, bin_centers