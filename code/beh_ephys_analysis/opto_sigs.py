# %%
import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
from utils.beh_functions import parseSessionID, session_dirs, get_unit_tbl, get_session_tbl
from utils.plot_utils import shiftedColorMap, template_reorder, get_gradient_colors
from utils.opto_utils import opto_metrics, get_opto_tbl
from utils.ephys_functions import cross_corr_train, auto_corr_train, load_drift
import json
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from aind_ephys_utils import align
from scipy.stats import wilcoxon

# %%
def cal_opto_sigs(session, data_type):
    unit_tbl = get_unit_tbl(session, data_type)
    opto_tbl = get_opto_tbl(session, data_type, loc = 'soma')
    # loop through all conditions
    powers = opto_tbl['power'].unique().tolist()
    sites = opto_tbl['site'].unique().tolist()
    pre_posts = opto_tbl['pre_post'].unique().tolist()
    freqs = opto_tbl['freq'].unique().tolist()
    opto_sigs = pd.DataFrame()
    filter = unit_tbl['default_qc'] == 1
    unit_ids_focus = unit_tbl[filter]['unit_id'].unique().tolist()
    for unit in unit_ids_focus:
        # print(f"Processing unit {unit}...")
        spike_times = unit_tbl[unit_tbl['unit_id']== unit]['spike_times'].values[0]
        opto_tbl_curr = opto_tbl.copy()
        unit_drift = load_drift(session, unit)
        pulse_num = 5
        pre_win_ratio = 0.5
        post_win = 0.05
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times = spike_times[spike_times >= unit_drift['ephys_cut'][0]]
                opto_tbl_curr = opto_tbl_curr[opto_tbl_curr['time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times = spike_times[spike_times <= unit_drift['ephys_cut'][1]]
                opto_tbl_curr = opto_tbl_curr[opto_tbl_curr['time'] <= unit_drift['ephys_cut'][1]]
        for power_ind, power in enumerate(powers):
            for site_ind, site in enumerate(sites):
                for freq_ind, freq in enumerate(freqs):
                    for pre_post_ind, pre_post in enumerate(pre_posts):
                        # get the trials for this condition
                        trials = opto_tbl[(opto_tbl['power'] == power) & (opto_tbl['site'] == site) & (opto_tbl['pre_post'] == pre_post) & (opto_tbl['freq'] == freq)]
                        if len(trials) == 0:
                            # print(f"No trials for power {power}, site {site}, freq {freq}, pre_post {pre_post}")
                            continue
                        # get the pulse times
                        train_times = trials['time'].values
                        p_unit_condition = []
                        for pulse_ind in range(pulse_num):
                            pulse_times = train_times + (pulse_ind * 1/freq)  # assuming freq is in Hz and pulse_ind starts from 0
                            # get the opto signal
                            pre_win_counts = align.to_events(spike_times, pulse_times, [-1/freq * pre_win_ratio, 0], return_df=True)
                            post_win_counts = align.to_events(spike_times, pulse_times, [0, post_win], return_df=True)
                            pre_win_freq = [len(pre_win_counts[pre_win_counts['event_index'] == event_ind]) / (1/freq * pre_win_ratio) for event_ind in range(len(pulse_times))]
                            post_win_freq = [len(post_win_counts[post_win_counts['event_index'] == event_ind]) / (post_win) for event_ind in range(len(pulse_times))]
                            # paired non-parametric test
                            stat, p = wilcoxon(pre_win_freq, post_win_freq)
                            p_unit_condition.append(p)
                        # store the results
                        opto_sigs = pd.concat([opto_sigs, pd.DataFrame({
                                            'unit_id': [unit],
                                            'power': [power],
                                            'site': [site],
                                            'freq': [freq],
                                            'pre_post': [pre_post],
                                            'p_unit_condition': [p_unit_condition],  # wrap list of dicts in a list to keep in one cell
                                            'p_sig_count': [sum(p < 0.05 for p in p_unit_condition)],
                                        })], ignore_index=True)
    # save the results
    opto_sigs_file = os.path.join(session_dirs(session, data_type)[f'opto_dir_{data_type}'], f'{session}_opto_sigs.pkl')
    with open(opto_sigs_file, 'wb') as f:
        pickle.dump(opto_sigs, f)
    return opto_sigs

# %%
if __name__ == '__main__':
    session_assets = pd.read_csv('/root/capsule/code/data_management/session_assets.csv')
    session_list = session_assets['session_id']
    probe_list = session_assets['probe']
    probe_list = [probe for probe, session in zip(probe_list, session_list) if isinstance(session, str)]
    session_list = [session for session in session_list if isinstance(session, str)]    
    from joblib import Parallel, delayed
    data_type = 'curated'
    def process(session, data_type): 
        print(f'Starting {session}')
        session_dir = session_dirs(session)
        # if os.path.exists(os.path.join(session_dir['beh_fig_dir'], f'{session}.nwb')):
        if session_dir[f'curated_dir_{data_type}'] is not None:
            try:
                # plot_ephys_probe(session, data_type=data_type, probe=probe) 
                cal_opto_sigs(session, data_type)
                plt.close('all')
                print(f'Finished {session}')
            except:
                print(f'Error processing {session}')
                plt.close('all')
        else: 
            print(f'No curated data found for {session}') 
        # elif session\_dir['curated_dir_raw'] is not None:
        #     data_type = 'raw' 
        #     opto_tagging_df_sess = opto_plotting_session(session, data_type, target, resp_thresh=resp_thresh, lat_thresh=lat_thresh, target_unit_ids= None, plot = True, save=True)
    Parallel(n_jobs=5)(
        delayed(process)(session, data_type) 
        for session in session_list
    )


