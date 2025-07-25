# %% [markdown]
# ### Check behavior
# put in session ID to process

# %%
import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.beh_functions import *
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session, plot_foraging_session_nwb
from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import plot_lick_analysis, cal_metrics, plot_met, load_data
from harp.clock import decode_harp_clock, align_timestamps_to_anchor_points
from open_ephys.analysis import Session
import datetime
from aind_ephys_rig_qc.temporal_alignment import search_harp_line
from matplotlib.gridspec import GridSpec
import json

# %%
def beh_and_time_alignment(session, ephys_cut = [0, 0]):
    session_dir = session_dirs(session)
    print(session)
    qm_dict = {'soundcard_sync': None, 'ephys_sync': None}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(session_dir['processed_dir'], f"{session}_process_record.txt")

    # Redirect stdout to the file
    if not os.path.exists(output_file):
        log_file = open(output_file, "w") 
    else: 
        log_file = open(output_file, "a")
    sys.stdout = log_file
    print(f"Session: {session} processed at {timestamp}")


    # %%
    print(session)
    session_dir = session_dirs(session)
    aniID, date_time, string = parseSessionID(session)
    session_json_dir = os.path.join(session_dir['raw_dir'], 'behavior')
    session_json_files = []
    for dir, _, files in os.walk(session_json_dir):
        print(dir)
        for file in files:
            print(file)
            if file.endswith('.json') and aniID in file and 'model' not in file:
                session_json_files.extend([os.path.join(dir, file)])
    print(f'{len(session_json_files)} session json files found.')
    nwb_file = os.path.join(session_dir['beh_fig_dir'], session + '.nwb')
    if len(session_json_files) == 1:
        session_json_file = session_json_files[0]
        if os.path.exists(os.path.join(session_dir['beh_fig_dir'], session + '.nwb')):
            print('NWB file already exists.')
        else:
            print('Processing NWB:')
            success, nwb = bonsai_to_nwb(session_json_file, os.path.join(session_dir['beh_fig_dir'], session + '.nwb'))

    # %%

    if not os.path.exists(nwb_file):
        print('NWB file does not exist.')
    else:
        print('Plotting session.')
        nwb = load_nwb_from_filename(nwb_file)
        fig = plot_session_in_time_all(nwb)
        fig.savefig(os.path.join(session_dir['beh_fig_dir'], session + '_choice_reward.pdf'))
        # display(fig)
        
        fig, _ = plot_lick_analysis(nwb)
        fig.savefig(os.path.join(session_dir['beh_fig_dir'], session + '_lick_analysis.pdf'))
        # display(fig)

        fig, _ = plot_session_glm(session, tMax=5)
        fig.savefig(os.path.join(session_dir['beh_fig_dir'], session + '_glm.pdf'))
        # display(fig)

        # plt.close('all')
    if os.path.exists(nwb_file):
        # %%
        left_licks = nwb.acquisition["left_lick_time"].timestamps[:]
        right_licks = nwb.acquisition["right_lick_time"].timestamps[:]
        all_licks = np.sort(np.concatenate((right_licks, left_licks)))
        df_trial = nwb.trials.to_dataframe()
        plt.figure()
        plt.hist(all_licks, alpha=0.5, label='licks', density=True)
        plt.hist(df_trial['goCue_start_time'], alpha=0.5, label='goCue', density=True)
        plt.legend()
        plt.savefig(os.path.join(session_dir['beh_fig_dir'], session + '_licks_vs_goCue.pdf'))
        if np.abs(np.mean(all_licks) - np.mean(df_trial['goCue_start_time'])) > 0.2*(df_trial['goCue_start_time'].max() - df_trial['goCue_start_time'].min()):
            print(f'{session} sound card is not synced.')
            qm_dict['soundcard_sync'] = False
        else:
            print(f'{session} sound card is synced.')
            qm_dict['soundcard_sync'] = True

        # %% [markdown]
        # ### Check ephys alignment

        # %%
        session_rec = Session(session_dir['session_dir'])  
        recording = session_rec.recordnodes[0].recordings[0]
        harp_line, nidaq_stream_name, source_node_id, figure = search_harp_line(recording, session_dir['session_dir'])
        figure.savefig(os.path.join(session_dir['alignment_dir'], 'harp_line.png'))
        print(F'Harp line {harp_line} found in {session}')

        # %%
        timestamps = recording.continuous[0].timestamps[::30000]
        # example neurons
        nwb = load_nwb_from_filename(session_dir['nwb_dir_raw'])
        unit_spikes = nwb.units[::10]['spike_times']
        mean_spike_times = [np.mean(unit_spike) for unit_spike in unit_spikes]
        mean_spike_times = np.mean(np.array(mean_spike_times))
        figure, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.hist(timestamps, bins=100, density=True, alpha=0.5, label='ephys')
        ax.hist(all_licks, bins=100, density=True, alpha=0.5, label='licks')
        ax.hist(df_trial['goCue_start_time'], bins=100, density=True, alpha=0.5, label='goCue')
        for i, unit_spike in enumerate(unit_spikes):
            ax.hist(unit_spike, bins=100, density=True, alpha=0.2, color='k')
        ax.legend()
        figure.savefig(os.path.join(session_dir['alignment_dir'], 'lick_goCue_ephys_time.pdf'))
        if np.abs(np.mean(all_licks) - np.mean(timestamps)) < 0.2*(timestamps[-1]-timestamps[0]) and np.abs(np.mean(timestamps) - mean_spike_times) < 0.2*(timestamps[-1]-timestamps[0]): 
            print(f'{session} ephys is synced.')
            qm_dict['ephys_sync'] = True
        else:
            print(f'{session} ephys is not synced.')
            qm_dict['ephys_sync'] = False
            events = recording.events
            harp_events = events[
                (events.stream_name == nidaq_stream_name)
                & (events.processor_id == source_node_id)
                & (events.line == harp_line[0])
            ]

            harp_states = harp_events.state.values
            harp_timestamps_local = harp_events.timestamp.values
            local_times, harp_times = decode_harp_clock(
                harp_timestamps_local, harp_states
            )
            np.save(os.path.join(session_dir['alignment_dir'], 'harp_times.npy'), harp_times)
            np.save(os.path.join(session_dir['alignment_dir'], 'local_times.npy'), local_times)

            print('Harp times saved to: {}'.format(os.path.join(session_dir['alignment_dir'], 'harp_times.npy')))
            print('Local times saved to: {}'.format(os.path.join(session_dir['alignment_dir'], 'local_times.npy')))
        # %% find a stable time period
        ephys_cut_new = [recording.continuous[0].timestamps[0]+ephys_cut[0], recording.continuous[0].timestamps[-1]-ephys_cut[1]]
        if not qm_dict['ephys_sync']:
            if '717121' not in session: # 717121 is misaligned only in events
                ephys_cut_new = align_timestamps_to_anchor_points(np.array(ephys_cut_new), local_times, harp_times)
                ephys_cut_new = list(ephys_cut_new)
        qm_dict['ephys_cut'] = ephys_cut_new
    else: 
        qm_dict['ephys_sync'] = True
        qm_dict['ephys_cut'] = ephys_cut
        qm_dict['soundcard_sync'] = True

    # %%
    qm_file = os.path.join(session_dir['processed_dir'], f"{session}_qm.json")
    with open(qm_file, 'w') as f:
        json.dump(qm_dict, f, indent=4)
    print(f"Output saved to {output_file}")
    sys.stdout = sys.__stdout__
    # Close the file
    log_file.close()

if __name__ == "__main__":  
    session = 'ecephys_763360_2025-04-16_13-29-55'
    ephys_cut = [0, 0]
    beh_and_time_alignment(session, ephys_cut=ephys_cut)




