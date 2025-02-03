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
    sorted_curated_dir = os.path.join(data_dir, session_id+'_sorted_curated')
    sorted_raw_dir = os.path.join(data_dir, session_id+'_sorted')
    sorted_opto_dir = os.path.join(data_dir, session_id+'_sorted-opto')
    nwb_dir_temp = os.path.join(sorted_curated_dir, session_id+'_experiment1_recording1.nwb')
    nwb_dir = nwb_dir_temp

    if os.path.exists(sorted_curated_dir):
        sorted_dir = sorted_curated_dir
    elif os.path.exists(sorted_opto_dir):
        sorted_dir = sorted_opto_dir
    else:
        sorted_dir = sorted_raw_dir
    postprocessed_dir_temp = os.path.join(sorted_dir, 'postprocessed')
    postprocessed_sub_folders = os.listdir(postprocessed_dir_temp)
    postprocessed_sub_folder = [s for s in postprocessed_sub_folders if 'post' not in s]
    postprocessed_dir = os.path.join(postprocessed_dir_temp, postprocessed_sub_folder[0])
    qm_dir = os.path.join(postprocessed_dir, 'quality_metrics', 'metrics.csv')


    curated_dir_temp = os.path.join(sorted_dir, 'curated')
    curated_sub_folders = os.listdir(curated_dir_temp)
    curated_dir = os.path.join(curated_dir_temp, curated_sub_folders[0])

    opto_csv_dir = os.path.join(raw_dir, 'ecephys_clipped')

    if not os.path.exists(opto_csv_dir):
        opto_csv_dir = os.path.join(raw_dir, 'ecephys', 'ecephys_clipped')

    if os.path.exists(opto_csv_dir):
        temp_files = os.listdir(opto_csv_dir)
        opto_csv = [s for s in temp_files if '.opto.csv' in s]
        opto_csv = opto_csv[0]
        opto_csv_file = os.path.join(opto_csv_dir, opto_csv)

    # processed dirs
    processed_dir = os.path.join(scratch_dir, session_id)
    beh_fig_dir = os.path.join(processed_dir, 'behavior')
    ephys_fig_dir = os.path.join(processed_dir, 'ephys')
    opto_fig_dir = os.path.join(processed_dir, 'opto')
    opto_tag_dir = os.path.join(processed_dir, 'opto_tag')
    opto_tag_fig_dir = os.path.join(opto_tag_dir, 'figures')

    dir_dict = {'aniID': aniID,
                'raw_id': raw_id,
                'datetime': date_obj,
                'raw_dir': raw_dir,
                'session_dir': session_dir,
                'processed_dir': processed_dir,
                'opto_csv_file': opto_csv_file,
                'ephys_fig_dir': ephys_fig_dir,   
                'opto_fig_dir': opto_fig_dir,
                'postprocessed_dir': postprocessed_dir,
                'curated_dir': curated_dir,
                'nwb_dir': nwb_dir,
                'opto_tag_dir': opto_tag_dir,
                'opto_tag_fig_dir': opto_tag_fig_dir,
                'qm_dir': qm_dir}

    # make directories
    makedirs(dir_dict)

    return dir_dict

def makedirs(directories):
    for directory_name, directory in directories.items():
        if 'dir' in directory_name and 'scratch' in str(directory) and directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)
  
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
