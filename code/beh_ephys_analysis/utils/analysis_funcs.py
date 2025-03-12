import numpy as np
from scipy import stats


def event_locked_timestamps(spike_timestamps, event_timestamps, time_range, time_shift=0):
    """
    Converts spike timestamps over course of entire session to time stamps relative to an event.

    :param spike_timestamps: 1d array of floats, times a spike occurred over course of session
    :param event_timestamps: 1d array of floats, times an event occurred over course of session
    :param time_range: time range relative to event onset over which to collect spike times for each trial
    :param time_shift: some amount of time to shift these timestamps by (useful for specific pulse in train for instance)
    :return: event_locked_spike_times: list of arrays (1 per trial), array contains spike times relative to event onset
    for that trial
    """
    Ntrials = len(event_timestamps)

    event_locked_spike_times = []
    for ind_trial in np.arange(Ntrials):
        time_range_this_trial = event_timestamps[ind_trial] + time_range
        spike_inds_this_trial = np.searchsorted(spike_timestamps, time_range_this_trial)
        spike_times_this_trial = spike_timestamps[spike_inds_this_trial[0]:spike_inds_this_trial[1]]
        event_locked_spike_times.append(spike_times_this_trial - event_timestamps[ind_trial]-time_shift)

    return event_locked_spike_times


def timestamps_to_spike_counts(spike_timestamps, bins):
    """
    Converts timestamps at which spikes occurred to spike counts per time bin.

    :param spike_timestamps: should be a list of lists (one list per trial)
    :param bins: time ranges over which to calculate number of spikes
    :return: spike_counts: numpy array, dimensions num trials x len(bins)-1, number of spikes that happened in each time
    bin for each trial
    """
    Ntrials = len(spike_timestamps)
    spike_counts = np.empty((Ntrials, len(bins) - 1))
    for trial in range(Ntrials):
        this_spike_counts, bin_edges = np.histogram(spike_timestamps[trial], bins)
        spike_counts[trial, :] = this_spike_counts

    return spike_counts

def stim_evoked_response(cell_spike_times, event_timestamps, base_time_range, stim_time_range, alternative='two-sided'):
    """
    Computes p value and size of response to stimulus presentation. Uses a paired, non-parametric test
    ( signed-rank test ) comparing firing rate during the baseline time range (relative to trial start) and
    stimulus time range.

    Inputs:
    cell_spike_times: list of all timestamps when this unit spiked
    event_timestamps: list of all timestamps when the stimulus presentation began
    stim_time_range: range of times (relative to stimulus onset) to consider for calculating spike rate during stimulus
    base_time_range: range of times (relative to stimulus onset) to consider for calculating baseline spike rate
    alternative: hypothesis to test with signed-rank test (e.g. "greater", "less", "two-sided")

    Outputs:
    pVal: p-value of signed-rank test on spike data given
    statistic: test statistic of signed-rank test
    response: difference in firing rate (spikes/sec) between stimulus and baseline condition
    """
    stim_evoked_timestamps = event_locked_timestamps(cell_spike_times, event_timestamps, stim_time_range)
    baseline_timestamps = event_locked_timestamps(cell_spike_times, event_timestamps, base_time_range)

    stim_evoked_spike_counts = timestamps_to_spike_counts(stim_evoked_timestamps, stim_time_range)
    baseline_spike_counts = timestamps_to_spike_counts(baseline_timestamps, base_time_range)

    try:
        statistic, pVal = stats.wilcoxon(stim_evoked_spike_counts.flatten()/np.diff(stim_time_range),
                                         baseline_spike_counts.flatten()/np.diff(base_time_range),
                                         alternative=alternative)
    except(ValueError):  # wilcoxon test doesn't like it when there's no difference between passed values
        statistic = 0
        pVal = 1

    response = (np.mean(stim_evoked_spike_counts.flatten())/np.diff(stim_time_range)) - \
               (np.mean(baseline_spike_counts.flatten()) / np.diff(base_time_range))

    return pVal, statistic, response