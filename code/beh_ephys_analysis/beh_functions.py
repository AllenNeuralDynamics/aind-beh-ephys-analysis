import numpy as np
from scipy import stats
import statsmodels.api as sm
import re
from PyPDF2 import PdfMerger
import pandas as pd
import os

def test():
    print('test success!')

def get_barcode(harp_events, index):
    """
    Returns a subset of original DataFrame corresponding to a specific 
    barcode

    Parameter
    ---------
    index : int
        The index of the barcode being requested

    Returns
    -------
    sample_numbers : np.array
        Array of integer sample numbers for each barcode event
    states : np.array
        Array of states (1 or 0) for each barcode event
    
    """

    splits = np.where(np.diff(harp_events.timestamp) > 0.5)[0]

    barcode = harp_events.iloc[splits[index]+1:splits[index+1]+1]

    return barcode.sample_number.values, barcode.state.values


def convert_barcode_to_time(sample_numbers, 
        states, 
        baud_rate=1000.0, 
        sample_rate=30000.0):
    """
    Converts event sample numbers and states to
    a Harp timestamp in seconds.

    Harp timestamp is encoded as 32 bits, with
    the least significant bit coming first, and 2 bits 
    between each byte.
    """

    samples_per_bit = int(sample_rate / baud_rate)
    middle_sample = int(samples_per_bit / 2)
    
    intervals = np.diff(sample_numbers)

    barcode = np.concatenate([np.ones((count,)) * state 
                    for state, count in 
                    zip(states[:-1], intervals)]).astype("int")

    val = np.concatenate([
        np.arange(samples_per_bit + middle_sample + samples_per_bit * 10 * i, 
                  samples_per_bit * 10 * i - middle_sample + samples_per_bit * 10, 
                  samples_per_bit) 
                  for i in range(4)])
    s = np.flip(barcode[val])
    harp_time = s.dot(2**np.arange(s.size)[::-1])

    return harp_time


def rescale_times(times, t1_harp, t2_harp, t1_oe, t2_oe):
    new_times = np.copy(times)
    scaling = (t2_harp - t1_harp) / (t2_oe - t1_oe)
    new_times -= t1_oe
    new_times *= scaling
    new_times += t1_harp
    return new_times

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




def merge_pdfs(input_dir, output_filename='merged.pdf'):
    merger = PdfMerger()
    files = os.listdir(input_dir)
    files = sorted(files)
    # Iterate through all PDF files in the input directory
    for i, filename in enumerate(files):
        if filename.endswith('.pdf'):
            if i%50 == 0:
                print(f'Merging file {i} out of {len(os.listdir(input_dir))}')
            filepath = os.path.join(input_dir, filename)
            merger.append(filepath)

    # Write the merged PDF to the output file
    with open(output_filename, 'wb') as output_file:
        merger.write(output_file)

    print(f"PDF files in '{input_dir}' merged into '{output_filename}' successfully.")

    
def delete_files_without_name(folder_path, name):
    # Iterate through all files in the folder
    for i, filename in enumerate(os.listdir(folder_path)):
        # Check if the filename does not contain 'combined'
        if i%50 == 0:
            print(f'Deleting file {i} out of {len(os.listdir(folder_path))}')
        if name not in filename:
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Check if the path is a file
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)


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