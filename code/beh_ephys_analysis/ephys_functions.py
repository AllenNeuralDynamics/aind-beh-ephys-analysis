import numpy as np
from scipy import stats
import statsmodels.api as sm
import re
from PyPDF2 import PdfMerger
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap


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

def fitSpikeModelG(dfTrial, matSpikes, formula):
    TvCurrU = np.array([])
    PvCurrU = np.array([])
    EvCurrU = np.array([])
    for i in range(np.shape(matSpikes)[1]):
        currSpikes = np.squeeze(matSpikes[:,i])
        currData = dfTrial.copy()
        currData['spikes'] = currSpikes
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

def makeSessionDF(nwb, cut):
    tblTrials = nwb.trials.to_dataframe()
    tblTrials = tblTrials.iloc[cut[0]:cut[1]+1].copy()
    trialStarts = tblTrials.loc[tblTrials['animal_response']!=2, 'goCue_start_time'].values
    responseTimes = tblTrials[tblTrials['animal_response']!=2]
    responseTimes = responseTimes['reward_outcome_time'].values

    # responseInds
    responseInds = tblTrials['animal_response']!=2
    leftRewards = tblTrials.loc[responseInds, 'rewarded_historyL']

    # oucome 
    leftRewards = tblTrials.loc[tblTrials['animal_response']!=2, 'rewarded_historyL']
    rightRewards = tblTrials.loc[tblTrials['animal_response']!=2, 'rewarded_historyR']
    outcomes = leftRewards | rightRewards
    outcomePrev = np.concatenate((np.full((1), np.nan), outcomes[:-1]))

    # choices
    choices = tblTrials.loc[tblTrials['animal_response']!=2, 'animal_response'] == 1
    choicesPrev = np.concatenate((np.full((1), np.nan), choices[:-1]))
    
    # laser
    laserChoice = tblTrials.loc[tblTrials['animal_response']!=2, 'laser_on_trial'] == 1
    laser = tblTrials['laser_on_trial'] == 1
    laserPrev = np.concatenate((np.full((1), np.nan), laserChoice[:-1]))
    trialData = pd.DataFrame({
        'outcomes': outcomes.values.astype(float), 
        'choices': choices.values.astype(float),
        'laser': laserChoice.values.astype(float),
        'outcomePrev': outcomePrev,
        'laserPrev': laserPrev,
        'choicesPrev': choicesPrev,
        })
    return trialData

def plot_raster_rate_colormap(
    events,
    align_events, # sorted by certain value
    fig,
    subplot_spec,
    title,
    tb=-5,
    tf=10,
    bin_size=100 / 1000,
    step_size=50 / 1000,
):
    """Plot raster and rate aligned to events"""
    edges = np.arange(tb + 0.5 * bin_size, tf - 0.5 * bin_size, step_size)
    nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
    nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
    ax1 = fig.add_subplot(nested_gs[0, 0])
    ax2 = fig.add_subplot(nested_gs[1, 0])

    df = align.to_events(events, align_events, (tb, tf), return_df=True)
    ax1.scatter(df.time, df.event_index, c="k", marker="|", s=1, zorder=2)
    ax1.axvline(x=0, c="r", ls="--", lw=1, zorder=3)
    ax1.set_title(title)
    ax1.set_xlim(tb, tf)

    counts_pre = np.searchsorted(np.sort(df.time.values), edges - 0.5 * bin_size)
    counts_post = np.searchsorted(np.sort(df.time.values), edges + 0.5 * bin_size)
    counts_pre = np.searchsorted(np.sort(df.time.values), edges - 0.5 * bin_size)
    counts_post = np.searchsorted(np.sort(df.time.values), edges + 0.5 * bin_size)
    lick_rate = (counts_post - counts_pre) / (bin_size * len(align_events))
    ax2.plot(edges, lick_rate)
    ax2.set_title("lickRate")
    ax2.set_xlim(tb, tf)
    ax2.set_xlabel("Time from go cue (s)")

    return fig, ax1, ax2

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