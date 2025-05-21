import numpy as np
from scipy import stats
import statsmodels.api as sm
import re
from PyPDF2 import PdfMerger
import pandas as pd
import os
import sys
from matplotlib.colors import LinearSegmentedColormap
sys.path.append('/root/capsule/code/beh_ephys_analysis/utils')
from beh_functions import session_dirs
from matplotlib import gridspec
from aind_ephys_utils import align 
import ast
import json


def filter_spikes(spk_units_cond, n_points = 300, tau_ker = .15):
    time = np.linspace(-t_min , t_max , n_points)
    dt = (t_max + t_min)/n_points
    filt_spk_cond= []
    s=0
    for spk_units in spk_units_cond:
        if s%100==0:
            print('Neuron', s)
        s+=1
        spikes_to_exp = []
        for spikes_trials in spk_units:
            convolve  = np.zeros(n_points)  
            for time_spike in spikes_trials:
                if time_spike < -t_min:
                    continue
                else:
                    t = (time - time_spike)/tau_ker
                    conv = np.exp(-t)
                    conv[t<0] = 0
                    conv = conv/sum(conv)
                    convolve = convolve + conv
            spikes_to_exp.append(convolve)
        filt_spk_cond.append(spikes_to_exp)
    filt_spk_cond = np.array(filt_spk_cond) * (1/dt)
    return filt_spk_cond, time

def build_time_window_domain(bin_edges, offsets, callback=None):
    callback = (lambda x: x) if callback is None else callback
    domain = np.tile(bin_edges[None, :], (len(offsets), 1))
    domain += offsets[:, None]
    return callback(domain)

def build_spike_histogram(time_domain,
                          spike_times,
                          dtype=None,
                          binarize=False):

    time_domain = np.array(time_domain)

    tiled_data = np.zeros(
        (time_domain.shape[0], time_domain.shape[1] - 1, len(spike_times)),
        dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
    )

    starts = time_domain[:, :-1]
    ends = time_domain[:, 1:]
        
    for ii in range(len(spike_times)):
        data = np.array(spike_times[ii])

        start_positions = np.searchsorted(data, starts.flat)
        end_positions = np.searchsorted(data, ends.flat, side="right")
        counts = (end_positions - start_positions)

        tiled_data[:, :, ii].flat = counts > 0 if binarize else counts
    
    time = 0.5*(starts + ends)
    return time, tiled_data
    
# Make xarray functions
def build_spike_histogram_overlap(time_domain,
                          binSize,
                          spike_times,
                          dtype=None,
                          binarize=False):

    time_domain = np.array(time_domain)

    tiled_data = np.zeros(
        (time_domain.shape[0], time_domain.shape[1] - 1, len(spike_times)),
        dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
    )

    starts = time_domain[:, :-1]
    ends = time_domain[:, :-1]  + binSize
        
    for ii in range(len(spike_times)):
        data = np.array(spike_times[ii])

        start_positions = np.searchsorted(data, starts.flat)
        end_positions = np.searchsorted(data, ends.flat, side="right")
        counts = (end_positions - start_positions)

        tiled_data[:, :, ii].flat = counts > 0 if binarize else counts

    time = starts + 0.5*binSize
    return time, tiled_data

def fitSpikeModelG(dfTrial, matSpikes, formula, matIso = None):
    TvCurrU = np.array([])
    PvCurrU = np.array([])
    EvCurrU = np.array([])
    for i in range(np.shape(matSpikes)[1]):
        currSpikes = np.squeeze(matSpikes[:,i])
        currData = dfTrial.copy()
        currData['spikes'] = np.squeeze(currSpikes)
        if matIso is not None:
            currData['iso'] = matIso[:,i]
        # Fit the GLM
        model = sm.GLM.from_formula(formula=formula, data=currData, family=sm.families.Gaussian()).fit()
        # t value
        regressors = [re.sub(r'\[.*?\]', '', x) for x in model.tvalues.index]
        tv = model.tvalues.values.reshape(1,-1)
        # p value
        pv = model.pvalues.values.reshape(1,-1)
        # t value
        ev = model.params.values.reshape(1,-1)
        # concatenate
        # initialize shape if not
        if np.shape(TvCurrU)[0] == 0:
            TvCurrU = np.empty((0, len(regressors)))
            PvCurrU = np.empty((0, len(regressors)))
            EvCurrU = np.empty((0, len(regressors)))

        TvCurrU = np.concatenate((TvCurrU, tv), axis = 0)
        PvCurrU = np.concatenate((PvCurrU, pv), axis = 0)
        EvCurrU = np.concatenate((EvCurrU, ev), axis = 0)

    return regressors, TvCurrU, PvCurrU, EvCurrU


def fitSpikeModelP(dfTrial, matSpikes, formula):
    TvCurrU = np.array([])
    PvCurrU = np.array([])
    EvCurrU = np.array([])
    for i in range(np.shape(matSpikes)[1]):
        currSpikes = np.squeeze(matSpikes[:,i])
        currData = dfTrial.copy()
        currData['spikes'] = currSpikes
        # Fit the GLM
        model = sm.GLM.from_formula(formula=formula, data=currData, family=sm.families.Poisson()).fit()
        # t value
        regressors = [re.sub(r'\[.*?\]', '', x) for x in model.tvalues.index]
        tv = model.tvalues.values.reshape(1,-1)
        # p value
        pv = model.pvalues.values.reshape(1,-1)
        # t value
        ev = model.params.values.reshape(1,-1)
        # concatenate
        # initialize shape if not
        if np.shape(TvCurrU)[0] == 0:
            TvCurrU = np.empty((0, len(regressors)))
            PvCurrU = np.empty((0, len(regressors)))
            EvCurrU = np.empty((0, len(regressors)))

        TvCurrU = np.concatenate((TvCurrU, tv), axis = 0)
        PvCurrU = np.concatenate((PvCurrU, pv), axis = 0)
        EvCurrU = np.concatenate((EvCurrU, ev), axis = 0)

    return regressors, TvCurrU, PvCurrU, EvCurrU

def shiftedColorMap(cmap, min_val, max_val, name):
    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val) # Edit #2
    midpoint = 1.0 - max_val/(max_val + abs(min_val))
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5) # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = LinearSegmentedColormap(name, cdict)

    # colormaps.register(cmap=newcmap, force=True)
    return newcmap

def template_reorder(template, right_left, all_channels_int, sample_to_keep = [-30, 60], y_neighbors_to_keep = 3, orginal_loc = False):
    peak_ind = np.argmin(np.min(template, axis=0))
    peak_channel = all_channels_int[peak_ind]
    peak_sample = np.argmin(template[:, peak_ind])  
    peak_group = np.arange(peak_channel - 2*y_neighbors_to_keep, peak_channel + 2*y_neighbors_to_keep + 1, 2)
    if right_left[peak_ind]: # peak is on the right
        sub_peak_channel = peak_channel - 1 
    else: 
        sub_peak_channel = peak_channel + 1 
    sub_group = np.arange(sub_peak_channel - 2*y_neighbors_to_keep, sub_peak_channel + 2*y_neighbors_to_keep + 1, 2)

    # get the reordered template: major column on left, minor column on right
    reordered_template = np.full((2*y_neighbors_to_keep + 1, 2*(sample_to_keep[1] - sample_to_keep[0])), np.nan)
    if peak_sample+sample_to_keep[1] > template.shape[0] or peak_sample+sample_to_keep[0] < 0:
        peak_sample = 89
    for channel_int, channel_curr in enumerate(peak_group):
        if channel_curr in all_channels_int:
            reordered_template[channel_int, :(sample_to_keep[1] - sample_to_keep[0])] = template[(peak_sample+sample_to_keep[0]):(peak_sample+sample_to_keep[1]), np.argwhere(all_channels_int == channel_curr)[0][0]].T

    for channel_int, channel_curr in enumerate(sub_group):
        if channel_curr in all_channels_int:
            reordered_template[channel_int, (sample_to_keep[1] - sample_to_keep[0]):2*(sample_to_keep[1] - sample_to_keep[0])] = template[(peak_sample+sample_to_keep[0]):(peak_sample+sample_to_keep[1]), np.argwhere(all_channels_int == channel_curr)[0][0]].T

    # whether switch back to original location
    if orginal_loc:
        if right_left[peak_ind]:
            reordered_template = reordered_template[:, list(range((sample_to_keep[1] - sample_to_keep[0]), 2*(sample_to_keep[1] - sample_to_keep[0]))) + list(range((sample_to_keep[1] - sample_to_keep[0])))]
    return reordered_template

def convert_values(value):
    if isinstance(value, str):
        try:
            # Convert NumPy-like array strings into proper lists
            if "nan" in value:
                list_array = json.loads(value.replace("nan", "null"))
                return np.array(list_array)
            elif "[" in value and "]" in value and "," in value:
                string_list = value.replace("nan", "np.nan")
                return np.array(ast.literal_eval(string_list))
            elif "[" in value and "]" in value and "," not in value:
                # return np.array(ast.literal_eval(value.replace(" ", ",")))
                return np.fromstring(value.strip("[]"), sep=" ")
            else:
                return ast.literal_eval(value)  # Convert regular lists
        except (ValueError, SyntaxError):
            return value  # If conversion fails, return original string
    return value  # Return value as-is if not a string
    
def load_drift(session, unit_id, data_type='curated'):
    session_dir = session_dirs(session)
    drift_file = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_drift_tbl.csv')
    if not os.path.exists(drift_file):
        return None
    else:
        opto_drift_tbl = pd.read_csv(drift_file)    
        opto_drift_tbl.reset_index(drop=True, inplace=True)
        if len(opto_drift_tbl) == 0:
            return None
        elif unit_id==None:
            return opto_drift_tbl
        elif unit_id in opto_drift_tbl['unit_id'].values:
            unit_drift = opto_drift_tbl[opto_drift_tbl['unit_id'] == unit_id].iloc[0].to_dict()
            # Apply conversion
            converted_data = {key: convert_values(val) for key, val in unit_drift.items()}
            unit_drift = converted_data
            return unit_drift
        else:
            return None

def get_spike_matrix(spike_times, align_time, pre_event, post_event, binSize, stepSize):
    bin_times = np.arange(pre_event, post_event, stepSize) - 0.5*stepSize
    spike_matrix = np.zeros((len(align_time), len(bin_times)))
    for i, t in enumerate(align_time):
        for j, b in enumerate(bin_times):
            spike_matrix[i, j] = np.sum((spike_times >= t + b - 0.5*binSize) & (spike_times < t + b + 0.5*binSize))
    spike_matrix = spike_matrix / binSize
    return spike_matrix, bin_times

def plot_filled_sem(time, y_mat, color, ax, label):
    ax.plot(time, np.nanmean(y_mat, 0), c = color, label = label)
    sem = np.std(y_mat, axis = 0)/np.sqrt(np.shape(y_mat)[0])
    ax.fill_between(time, np.nanmean(y_mat, 0) - sem, np.nanmean(y_mat, 0) + sem, color = color, alpha = 0.25, edgecolor = None)

def plot_raster_rate(
    spike_times,
    currArray,
    slide_times,
    align_events, # sorted by certain value
    map_value,
    bins,
    labels,
    colormap,
    fig,
    subplot_spec,
    tb=-2,
    tf=3,
):
    n_colors = len(bins)-1
    color_list = [colormap(i / (n_colors - 1)) for i in range(n_colors)]

    """Plot raster and rate aligned to events"""
    nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios= [3, 1], subplot_spec=subplot_spec)
    ax1 = fig.add_subplot(nested_gs[0, 0])
    ax2 = fig.add_subplot(nested_gs[1, 0])

    # order events by values
    sort_ind = np.argsort(map_value)
    align_events = align_events[sort_ind]

    df = align.to_events(spike_times, align_events, (tb, tf), return_df=True)
    
    # vertical line at time 0
    ax1.axvline(x=0, c="r", ls="--", lw=1, zorder=1)

    # raster plot
    ax1.scatter(df.time, df.event_index, c="k", marker="|", s=1)

    # horizontal line for each type if discrete
    if len(np.unique(map_value)) <= 4:
        discrete_types = np.sort(np.unique(map_value))
    else:
        discrete_types = bins
    
    for val in discrete_types:
        level = np.sum(map_value <= val)
        ax1.axhline(y=level, c="k", ls="--", lw=1)

    ax1.set_title(' '.join(labels))
    ax1.set_xlim(tb, tf)
    ax1.set_ylabel('__'.join(labels))

    # rate plot by binned values

    for bin_ind in range(len(bins)-1): 
        currList = np.where((np.array(map_value)>=bins[bin_ind]) & (np.array(map_value)<bins[bin_ind + 1]))[0]
        if len(currList) > 0:
            M = currArray[currList, :]
            plot_filled_sem(slide_times, M, color_list[bin_ind], ax2, labels[bin_ind])

    ax2.legend()

    ax2.set_title("spike rate")
    ax2.set_xlim(tb, tf)
    ax2.set_xlabel("Time from alignment (s)")

    return fig, ax1, ax2

def plot_rate(
    currArray,
    slide_times, 
    map_value,
    bins,
    labels,
    colormap,
    fig,
    subplot_spec,
    tb,
    tf,
):
    n_colors = len(bins)-1
    color_list = [colormap(i / (n_colors - 1)) for i in range(n_colors)]

    """Plot rate aligned to events"""

    # rate plot by binned values
    ax = fig.add_subplot(subplot_spec)
    for bin_ind in range(len(bins)-1): 
        currList = np.where((np.array(map_value)>=bins[bin_ind]) & (np.array(map_value)<bins[bin_ind + 1]))[0]
        if len(currList) > 0:
            M = currArray[currList, :]
            plot_filled_sem(slide_times, M, color_list[bin_ind], ax, labels[bin_ind])

    ax.legend()

    ax.set_title("spike rate")
    ax.set_xlim(tb, tf)
    ax.set_xlabel("Time from alignment (s)")

    return fig, ax

def regression_rwd(spike_counts, outcome, trials_back = [0, 2], sub_selection = None):
    outcome_matrix = np.zeros((len(outcome), trials_back[1] - trials_back[0] + 1))
    for i in range(len(outcome)):
        for j in range(trials_back[0], trials_back[1] + 1):
            if i-j >=0:
                outcome_matrix[i, j] = outcome[i-j]
            else:
                outcome_matrix[i, j] = np.nan

    outcome_matrix = sm.add_constant(outcome_matrix)
    if sub_selection is not None:
        outcome_matrix = outcome_matrix[sub_selection, :]
        spike_counts = spike_counts[sub_selection]
    lm = sm.OLS(spike_counts, outcome_matrix, missing='drop').fit()
    return lm.params[1:], lm.pvalues[1:], lm.tvalues[1:], lm.conf_int(alpha=0.05)[1:] 

class load_trial_drift:
    def __init__(self, session, data_type):
        """Initialize the object with a DataFrame."""
        session_dir = session_dirs(session)
        drift_tbl_dir = os.path.join(session_dir[f'ephys_dir_{data_type}'], f'{session}_drift_trial_table.csv')
        if not os.path.exists(drift_tbl_dir):
            drift_data = None
        else:
            drift_data = pd.read_csv(drift_tbl_dir)
        self.drift_data = drift_data

    def load_unit(self, unit_id, trial_range = None, cat = None):
        """Load the drift data for a specific unit."""
        """Load the drift data for a specific unit."""
        if self.drift_data is None:
            raise ValueError("Drift data is not available. Please generate drift data first.")
        unit_drift_data = self.drift_data[self.drift_data['unit_id'] == unit_id].sort_values(by='trial_ind')
        if trial_range is not None:
            if isinstance(trial_range, int):
                trial_range = [trial_range]
            elif isinstance(trial_range, list):
                trial_range = np.array(trial_range)
            else:
                raise ValueError("trial_range should be an integer or a list of integers.")
            unit_drift_data = unit_drift_data[unit_drift_data['trial_ind'].isin(trial_range)]
        if cat is not None:
            if cat not in unit_drift_data.columns:
                raise ValueError(f"Category '{cat}' not found in drift data.")
            unit_drift_data = unit_drift_data[cat]
        return unit_drift_data