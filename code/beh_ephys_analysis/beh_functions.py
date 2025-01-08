import numpy as np
from scipy import stats
import statsmodels.api as sm
import re
from PyPDF2 import PdfMerger
import pandas as pd
import os
from aind_dynamic_foraging_basic_analysis.lick_analysis import load_nwb
from scipy.stats import norm
from datetime import datetime
import ast
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session, plot_foraging_session_nwb

def parseSessionID(file_name):
    if len(re.split('[_.]', file_name)[0]) == 6 & re.split('[_.]', file_name)[0].isdigit():
        aniID = re.split('[_.]', file_name)[0]
        date = re.split('[_.]', file_name)[1]
        dateObj = datetime.strptime(date, "%Y-%m-%d")
        raw_id= session_id
    elif re.split('[_.]', file_name)[0] == 'behavior' or re.split('[_.]', file_name)[0] == 'ecephys':
        aniID = re.split('[_.]', file_name)[1]
        date = re.split('[_.]', file_name)[2]
        dateObj = datetime.strptime(date, "%Y-%m-%d")
        raw_id = '_'.join(re.split('[_.]', file_name)[1:])
    else:
        aniID = None
        dateObj = None
        raw_id = None
    
    return aniID, dateObj, raw_id

def session_dirs(session_id, model_name = None, data_dir = '/root/capsule/data', scratch_dir = '/root/capsule/scratch'):
    # parse session_id
    aniID, date_obj, raw_id = parseSessionID(session_id)
    # raw dirs
    raw_dir = os.path.join(data_dir, session_id+'_raw_data')
    session_dir = os.path.join(raw_dir, 'ecephys_clipped')
    if not os.path.exists(session_dir):
        session_dir = os.path.join(raw_dir, 'ecephys', 'ecephys_clipped')
    sorted_dir = os.path.join(data_dir, session_id+'_sorted_curated')
    nwb_dir_temp = os.path.join(sorted_dir, 'nwb')
    if not os.path.exists(nwb_dir_temp):
        nwb_dir_temp = os.path.join(sorted_dir)
        files_temp = os.listdir(nwb_dir_temp)
        nwb = [f for f in files_temp if f.endswith('.nwb')]
        nwb_dir = os.path.join(nwb_dir_temp, nwb[0])
    else:
        recordings = os.listdir(nwb_dir_temp)
        if len(recordings) == 1:
            nwb_dir = os.path.join(nwb_dir_temp, recordings[0])
            if not os.path.exists(nwb_dir):
                nwb_dir = os.path.join(sorted_dir, recordings[0])
        else:
            nwb_dir = None
            print('There are multiple recordings in the nwb directory. Please specify the recording you would like to use.')
        
    beh_nwb_dir = os.path.join('/root/capsule/data/all_behavior', raw_id+'.nwb')

    postprocessed_dir = os.path.join(sorted_dir, 'postprocessed')
    curated_dir = os.path.join(sorted_dir, 'curated')
    models_dir = os.path.join(data_dir, session_id+'_model_stan')

    # model dir
    
    if model_name is not None:
        model_dir = os.path.join(models_dir, model_name)
        model_file = os.path.join(model_dir, raw_id+'_session_model_dv.csv')
        session_curation_file = os.path.join(model_dir, 'ani_session_data.csv')
    else:
        model_dir = None
        model_file = None
        session_curation_file = os.path.join(models_dir, raw_id+'_session_data.csv')
    
    # processed dirs
    processed_dir = os.path.join(scratch_dir, session_id)
    alignment_dir = os.path.join(processed_dir, 'alignment')
    beh_fig_dir = os.path.join(processed_dir, 'behavior')
    ephys_fig_dir = os.path.join(processed_dir, 'ephys')
    opto_fig_dir = os.path.join(processed_dir, 'opto')

    dir_dict = {'aniID': aniID,
                'raw_id': raw_id,
                'datetime': date_obj,
                'raw_dir': raw_dir,
                'session_dir': session_dir,
                'processed_dir': processed_dir,
                'alignment_dir': alignment_dir,
                'beh_fig_dir': beh_fig_dir,
                'ephys_fig_dir': ephys_fig_dir,
                'opto_fig_dir': opto_fig_dir,
                'nwb_dir': nwb_dir,
                'postprocessed_dir': postprocessed_dir,
                'curated_dir': curated_dir,
                'models_dir': models_dir,
                'model_dir': model_dir,
                'model_file': model_file,
                'session_curation_file': session_curation_file,
                'beh_nwb_dir': beh_nwb_dir}

    # make directories
    makedirs(dir_dict)

    return dir_dict

def load_model_dv(session_id, model_name, data_dir = '/root/capsule/data', scratch_dir = '/root/capsule'):
    model_dirs = session_dirs(session_id, model_name, data_dir=data_dir, scratch_dir=scratch_dir)
    if model_dirs['model_file'] is not None:
        model_dv_temp = pd.read_csv(model_dirs['model_file'], index_col=0)
        nwb = load_nwb(model_dirs['nwb_dir'])
        trial_df = nwb.trials.to_dataframe()
        # model_dv = pd.DataFrame(np.nan, index=range(len(trial_df)), columns=model_dv_temp.columns)
        session_curation = pd.read_csv(model_dirs['session_curation_file'])
        session_cut = session_curation.loc[session_curation['session_id'] == model_dirs['raw_id'], 'session_cut'].values[0]
        session_cut = ast.literal_eval(session_cut)
        # model_dv[curr_cut[0]:curr_cut[1]] = model_dv_temp
        model_dv = model_dv_temp.copy()
    else:
        model_dv = None
        session_cut = None
    return model_dv, session_cut

def makedirs(directories):
    for directory_name, directory in directories.items():
        if 'dir' in directory_name and directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)

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

def makeSessionDF(nwb, cut = [0, np.nan]):
    tblTrials = nwb.trials.to_dataframe()
    if cut[1] == np.nan:
        tblTrials = tblTrials.iloc[cut[0]:].copy()
    else:
        tblTrials = tblTrials.iloc[cut[0]:cut[1]].copy()
    # tblTrials.reset_index(inplace=True)
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

    # go_cue_time
    go_cue_time = tblTrials.loc[tblTrials['animal_response']!=2, 'goCue_start_time']

    # choice_time
    choice_time = tblTrials.loc[tblTrials['animal_response']!=2, 'reward_outcome_time']

    # outcome_time
    outcome_time = tblTrials.loc[tblTrials['animal_response']!=2, 'reward_outcome_time'] + tblTrials.loc[tblTrials['animal_response']!=2, 'reward_delay']

    
    # laser
    laserChoice = tblTrials.loc[tblTrials['animal_response']!=2, 'laser_on_trial'] == 1
    laser = tblTrials['laser_on_trial'] == 1
    laserPrev = np.concatenate((np.full((1), np.nan), laserChoice[:-1]))
    # trial_id = tblTrials.loc[tblTrials['animal_response']!=2, 'id']
    trialData = pd.DataFrame({
        # 'trial_id': trial_id,
        'outcome': outcomes.values.astype(float), 
        'choice': choices.values.astype(float),
        'laser': laserChoice.values.astype(float),
        'outcome_prev': outcomePrev,
        'laser_prev': laserPrev,
        'choices_prev': choicesPrev,
        'go_cue_time': go_cue_time.values,
        'choice_time': choice_time.values,
        'outcome_time': outcome_time.values,
        })
    return trialData

def plot_session_in_time_all(nwb, bin_size = 10):
    fig = plt.Figure(figsize = (12, 6))
    gs = GridSpec(3, 1, figure = fig, height_ratios=[6,1,1], hspace = 0.5)
    choice_history, reward_history, p_reward, autowater_offered, random_number, trial_time = get_history_from_nwb(nwb)
    ax_choice_reward = fig.add_subplot(gs[0,0])
    plot_foraging_session(  # noqa: C901
                            choice_history,
                            reward_history,
                            p_reward = p_reward,
                            autowater_offered = autowater_offered,
                            trial_time = trial_time,
                            ax = ax_choice_reward,
                        )
    # plot licks
    data = load_data(nwb)    
    ax = fig.add_subplot(gs[1])
    bins = np.arange(np.min(data['all_licks']-data['tbl_trials']['goCue_start_time'][0]), np.max(data['all_licks']-data['tbl_trials']['goCue_start_time'][0]), bin_size)  
    ax.hist(data['left_licks']-data['tbl_trials']['goCue_start_time'][0], bins = bins, color = 'blue', alpha = 0.5, label = 'left licks', density = True)
    ax.set_xlim(0, -data['tbl_trials']['goCue_start_time'][0]+data['tbl_trials']['goCue_start_time'].values[-1])
    ax.legend(loc = 'upper right')
    ax.set_frame_on(False)
    ax = fig.add_subplot(gs[2])
    ax.hist(data['right_licks']-data['tbl_trials']['goCue_start_time'][0], bins = bins, color = 'red', alpha = 0.5, label = 'right licks', density = True)
    ax.set_xlim(0, -data['tbl_trials']['goCue_start_time'][0]+data['tbl_trials']['goCue_start_time'].values[-1])
    ax.legend(loc = 'upper right')
    ax.set_xlabel('Time in session (s)')
    ax.set_frame_on(False)
    plt.tight_layout()
    return fig
plot_session_in_time_all(nwb)