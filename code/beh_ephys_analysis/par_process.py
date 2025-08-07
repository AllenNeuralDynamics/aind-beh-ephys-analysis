import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
from utils.beh_functions import parseSessionID, session_dirs, get_unit_tbl, get_session_tbl
from cross_auto_corr import cross_auto_corr, plot_cross_auto_corr
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

session_assets = pd.read_csv('/root/capsule/code/data_management/session_assets.csv')
session_list = session_assets['session_id']
probe_list = session_assets['probe']
probe_list = [probe for probe, session in zip(probe_list, session_list) if isinstance(session, str)]
session_list = [session for session in session_list if isinstance(session, str)]    

data_type = 'curated'
def process(session, data_type): 
    print(f'Starting {session}')
    session_dir = session_dirs(session)
    # if os.path.exists(os.path.join(session_dir['beh_fig_dir'], f'{session}.nwb')):
    print(session_dir[f'curated_dir_{data_type}'])
    if session_dir[f'curated_dir_{data_type}'] is not None:
        if os.path.exists(os.path.join(session_dir[f'ephys_processed_dir_{data_type}'], f'{session}_{data_type}_long_corr.png')):
            print(f'Cross correlation already computed for {session}. Skipping.')
            return None
        else:
        # try:
        # plot_ephys_probe(session, data_type=data_type, probe=probe) 
            cross_auto_corr(session, data_type)
            plot_cross_auto_corr(session, data_type)
            plt.close('all')
            print(f'Finished {session}')
    # except:
        # print(f'Error processing {session}')
        # plt.close('all')
    else: 
        print(f'No curated data found for {session}') 
    # elif session\_dir['curated_dir_raw'] is not None:
    #     data_type = 'raw' 
    #     opto_tagging_df_sess = opto_plotting_session(session, data_type, target, resp_thresh=resp_thresh, lat_thresh=lat_thresh, target_unit_ids= None, plot = True, save=True)
Parallel(n_jobs=10, backend='loky')(
    delayed(process)(session, data_type) 
    for session in session_list[65:80]
)
