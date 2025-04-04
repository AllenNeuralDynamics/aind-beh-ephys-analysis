# %%
# %%
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import json
from harp.clock import decode_harp_clock, align_timestamps_to_anchor_points
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from utils.beh_functions import parseSessionID, session_dirs
from utils.plot_utils import shiftedColorMap, template_reorder
from open_ephys.analysis import Session##
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import load_nwb
from aind_ephys_utils import align
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from spikeinterface.core.sorting_tools import random_spikes_selection
import pickle
import datetime
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
from tqdm import tqdm
import shutil

# %%
def load_and_preprocess_recording(session_folder, stream_name):
    ephys_path = os.path.dirname(session_folder)
    compressed_folder = os.path.join(ephys_path, 'ecephys_compressed')
    recording_zarr = [os.path.join(compressed_folder, f) for f in os.listdir(compressed_folder) if stream_name in f][0]
    recording = si.read_zarr(recording_zarr)
    # preprocess
    recording_processed = spre.phase_shift(recording)
    recording_processed = spre.highpass_filter(recording_processed)    
    recording_processed = spre.common_reference(recording_processed)
    return recording_processed

# %%
session = 'behavior_751004_2024-12-20_13-26-11'
data_type = 'curated'
target = 'soma'
load_sorting_analyzer = True
session_dir = session_dirs(session)
bin_size = 10*60*30000 # 10 minutes,in samples
waveform_zarr_folder = f'{session_dir[f"ephys_dir_{data_type}"]}/waveforms.zarr'
if load_sorting_analyzer:
    if not os.path.exists(waveform_zarr_folder):
        print("Analyzer doesn't exist, computing first.")
        load_sorting_analyzer = False
if not load_sorting_analyzer:
    # recording info
    sorting = si.load(session_dir[f'curated_dir_{data_type}'])
    sorting_analyzer = si.load(session_dir[f'postprocessed_dir_{data_type}'], load_extensions=False)
    amplitudes = sorting_analyzer.get_extension('spike_amplitudes').get_data(outputs="by_unit")[0]

    # %%
    spike_vector = sorting.to_spike_vector()
    unit_ids = sorting.unit_ids
    num_units = len(sorting.unit_ids)
    max_spikes_per_unit_spontaneous = 500
    # %%
    # Spike indices
    spike_indices = spike_vector["sample_index"]
    # unit_ids
    spike_unit_ids = spike_vector["unit_index"]
    # session_length
    session_length = spike_indices[-1]
    # devide session into 5 equal parts
    edges = np.linspace(0, session_length, 6)
    new_unit_ids = []
    new_vector = []
    for bin_ind in range(5):
        start = int(edges[bin_ind])
        end = int(edges[bin_ind+1])
        indices = np.where((spike_indices > start) & (spike_indices <= end))[0]
        curr_vector = spike_vector[indices].copy()
        curr_vector_copy = curr_vector.copy()
        curr_vector_copy['unit_index'] = curr_vector_copy['unit_index'] + bin_ind*num_units
        curr_unit_ids = [f'{unit_id}-{bin_ind}' for unit_id in unit_ids]
        new_unit_ids += curr_unit_ids
        new_vector.append(curr_vector_copy)
    spikes_binned = np.concatenate(new_vector)
    # %% make new sorting
    sorting_binned = si.NumpySorting(spikes_binned, 
                                unit_ids=new_unit_ids,
                                sampling_frequency=30000)


# %%
random_spike_indices = random_spikes_selection(sorting_binned, method="uniform",
                                            max_spikes_per_unit=200)
selected_spikes_binned = spikes_binned[random_spike_indices]
sorting_binned_selected = si.NumpySorting(selected_spikes_binned, 
                            unit_ids=new_unit_ids,
                            sampling_frequency=30000)

# %%
recording_processed = load_and_preprocess_recording(session_dir['session_dir'], 'ProbeA')
good_channel_ids = recording_processed.channel_ids[
     np.in1d(recording_processed.channel_ids, sorting_analyzer.channel_ids)
]

recording_processed_good = recording_processed.select_channels(good_channel_ids)
print(f"Num good channels: {recording_processed_good.get_num_channels()}")

# %%
sparsity_mask_all = np.tile(sorting_analyzer.sparsity.mask, (len(edges)-1, 1))
sparsity_all = si.ChannelSparsity(
    sparsity_mask_all,
    unit_ids=sorting_binned_selected.unit_ids,
    channel_ids=recording_processed_good.channel_ids
)
si.set_global_job_kwargs(n_jobs=-1, progress_bar=True)

# %%
analyzer_binned = si.create_sorting_analyzer(
    # sorting_all.select_units(ROI_unit_ids),
    sorting_binned_selected,
    recording_processed_good,
    sparsity=sparsity_all
)

# %%
min_spikes_per_unit = 5
keep_unit_ids = []
count_spikes = sorting_binned_selected.count_num_spikes_per_unit()
for unit_id, count in count_spikes.items():
    if count >= min_spikes_per_unit:
        keep_unit_ids.append(unit_id)
analyzer = analyzer_binned.select_units(keep_unit_ids)

# %%
_ = analyzer.compute("random_spikes", method="all", max_spikes_per_unit=200)

# %%
_ = analyzer.compute(["waveforms"])
analyzer.save_as(format='zarr', folder = waveform_zarr_folder)


