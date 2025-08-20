import os
import re
import numpy as np
import pandas as pd
from scipy.stats import zscore
# from utils.basics.data_org import curr_computer, parse_session_string
import matplotlib.pyplot as plt
from scipy.io import loadmat
from itertools import chain
from scipy.stats import zscore
from scipy.stats import mode

def load_df_from_mat(file_path):
    # initialization
    beh_df = pd.DataFrame()
    licks_L = []
    licks_R = []
    # Load the .mat file
    mat_data = loadmat(file_path)

    # Access the 'beh' struct
    if 'behSessionData' in mat_data:
        beh = mat_data['behSessionData']
    elif 'sessionData' in mat_data:
        beh = mat_data['sessionData']
    else:
        print("No 'behSessionData' or 'sessionData' found in the .mat file.")

    # Convert the struct fields to a dictionary
    # Assuming 'beh' is a MATLAB struct with fields accessible via numpy record arrays
    if isinstance(beh, np.ndarray) and beh.dtype.names is not None:
        beh_dict = {field: beh[field].squeeze() for field in beh.dtype.names}

        # Create a DataFrame from the dictionary
        beh_df = pd.DataFrame(beh_dict)
        beh_df.head(10)
        # for column in beh_df.columns:
        #     if beh_df[column].dtype == np.object:
        #         beh_df[column] = beh_df[column].str[0]
        for column in beh_df.columns:
            if column in ['trialEnd', 'CSon', 'respondTime', 'rewardTime', 'rewardProbL', 'rewardProbR', 'laser', 'rewardL', 'rewardR']:
                curr_list = beh_df[column].tolist()
                curr_list = [np.float64(x[0][0]) if x.shape[0] > 0 and not np.isnan(x[0][0]) else np.nan for x in curr_list]
            elif column in ['licksL', 'licksR', 'trialType']:
                curr_list = beh_df[column].tolist()
                curr_list = [x[0] if len(x)>0 else x for x in curr_list] 
            beh_df[column] = curr_list
        # all licks
        licks_L = list(chain.from_iterable(beh_df['licksL'].tolist()))
        licks_R = list(chain.from_iterable(beh_df['licksR'].tolist()))

        list_to_drop = ['licksL', 'licksR', 'allLicks', 'allSpikes']
        unit_list = [x for x in beh_df.columns if 'TT' in x]
        list_to_drop.extend(unit_list)

        beh_df.drop(list_to_drop, axis=1, inplace=True, errors='ignore')

    else:
        print("'beh' is not a struct or has unexpected format.")
        
    return beh_df, licks_L, licks_R