# %%
import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
from utils.beh_functions import parseSessionID, session_dirs, get_unit_tbl, get_session_tbl
from utils.plot_utils import shiftedColorMap, template_reorder, get_gradient_colors
from utils.opto_utils import opto_metrics
from utils.ephys_functions import cross_corr_train, auto_corr_train, load_drift, load_auto_corr, load_cross_corr
import json
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import matplotlib.gridspec as gs


# %%
def cross_auto_corr(session, data_type):
    bin_long = 0.05
    win_long = 2
    bin_short = 0.002
    win_short = 0.08
    session_dir = session_dirs(session)
    unit_tbl = get_unit_tbl(session, data_type) 
    if unit_tbl['LC_range_top'].unique()[0] is None:
        print(f'LC range not inferred, probably no opto units in {session}. Exiting.')
        return None
    session_tbl = get_session_tbl(session)
    
    session_qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(session_qm_file, 'r') as f:
        session_qm = json.load(f)
    rec_start = session_qm['ephys_cut'][0]
    rec_end = session_qm['ephys_cut'][1]

    opto_file = os.path.join(session_dir['opto_dir_curated'], 
                                f'{session}_opto_session.csv')
    opto_tbl = pd.read_csv(opto_file)
    opto_times = opto_tbl['time'].values

    if len(opto_tbl['pre_post'].unique()) > 1:
        rec_start = opto_tbl[opto_tbl['pre_post'] == 'pre']['time'].max()
        rec_end = opto_tbl[opto_tbl['pre_post'] == 'post']['time'].min()
    elif len(opto_tbl['pre_post'].unique()) == 1:
        rec_end = opto_tbl['time'].min()

    filter = (
        (unit_tbl['decoder_label'] != 'artifact') &
        (unit_tbl['decoder_label'] != 'noise') &
        (unit_tbl['isi_violations_ratio'] < 0.1) &
        (unit_tbl['y_loc'] >= unit_tbl['LC_range_bottom'].unique()[0] - 0.1) &
        (unit_tbl['y_loc'] <= unit_tbl['LC_range_top'].unique()[0] + 0.1)
    )

    unit_ids_focus = unit_tbl[filter]['unit_id'].to_list()
    print(f'Number of units in LC range: {len(unit_ids_focus)}')

    cross_corr_df = pd.DataFrame(columns=['unit_1', 'unit_2', 'cross_corr_long', 'cross_corr_short', 'start', 'end', 'long_lags', 'short_lags'])
    auto_corr_df = pd.DataFrame(columns=['unit', 'auto_corr_long', 'auto_corr_short', 'start', 'end', 'long_lags', 'short_lags'])
    for unit_ind_1, unit_1 in enumerate(unit_ids_focus):
        print(f'Processing unit {unit_1}')
        spike_times_1 = unit_tbl[unit_tbl['unit_id'] == unit_1]['spike_times'].values[0]
        drift_1 = load_drift(session, unit_1, data_type)
        start_unit_1 = rec_start
        end_unit_1 = rec_end
        if drift_1 is not None:
            if drift_1['ephys_cut'][0] is not None:
                start_unit_1 = max(start_unit_1, drift_1['ephys_cut'][0])
            if drift_1['ephys_cut'][1] is not None:
                end_unit_1 = min(end_unit_1, drift_1['ephys_cut'][1])
        auto_corr_long, long_lags = auto_corr_train(spike_times_1, bin_long, win_long, start_unit_1, end_unit_1)
        auto_corr_short, short_lags = auto_corr_train(spike_times_1, bin_short, win_short, start_unit_1, end_unit_1)
        dist_out_auto = {'unit': unit_1,
                    'auto_corr_long': auto_corr_long,
                    'auto_corr_short': auto_corr_short,
                    'long_lags': long_lags,
                    'short_lags': short_lags,
                    'start': start_unit_1,
                    'end': end_unit_1}
        auto_corr_df = pd.concat([auto_corr_df, pd.DataFrame([dist_out_auto])], ignore_index=True)
        for unit_ind_2, unit_2 in enumerate(unit_ids_focus):
            cross_corr_long = None
            cross_corr_short = None
            if unit_ind_1 < unit_ind_2:
                # print(f'Processing units {unit_1} and {unit_2}')
                spike_times_2 = unit_tbl[unit_tbl['unit_id'] == unit_2]['spike_times'].values[0]
                drift_2 = load_drift(session, unit_2, data_type)
                start = start_unit_1
                end = end_unit_1
                if drift_2 is not None:
                    if drift_2['ephys_cut'][0] is not None:
                        start = max(start, drift_2['ephys_cut'][0])
                    if drift_2['ephys_cut'][1] is not None:
                        end = min(end, drift_2['ephys_cut'][1])
                
                if start >= end:
                    print(f'Skipping units {unit_1} and {unit_2} due to incompatible drift cuts')
                    continue
                cross_corr_long, lags_long = cross_corr_train(spike_times_1, spike_times_2, bin_long, win_long, start, end)
                cross_corr_short, lags_short = cross_corr_train(spike_times_1, spike_times_2, bin_short, win_short, start, end)

                dist_out = {'unit_1': unit_1,
                            'unit_2': unit_2,
                            'cross_corr_long': cross_corr_long,
                            'long_lags': lags_long,
                            'cross_corr_short': cross_corr_short,
                            'short_lags': lags_short,
                            'start': start,
                            'end': end}
                # Append the result to the cross_corr_df dataframe
                if cross_corr_long is not None or cross_corr_short is not None:
                    cross_corr_df = pd.concat([cross_corr_df, pd.DataFrame([dist_out])], ignore_index=True)
    # Save the results to pickle files
    cross_corr_file = os.path.join(session_dir[f'ephys_processed_dir_{data_type}'], f'{session}_{data_type}_cross_corr.pkl')
    auto_corr_file = os.path.join(session_dir[f'ephys_processed_dir_{data_type}'], f'{session}_{data_type}_auto_corr.pkl')
    with open(cross_corr_file, 'wb') as f:
        pickle.dump(cross_corr_df, f)
    with open(auto_corr_file, 'wb') as f:
        pickle.dump(auto_corr_df, f)

def plot_cross_auto_corr(session, data_type):
    unit_tbl = get_unit_tbl(session, data_type)
    filter = (
        (unit_tbl['decoder_label'] != 'artifact') &
        (unit_tbl['decoder_label'] != 'noise') &
        (unit_tbl['isi_violations_ratio'] < 0.1) &
        (unit_tbl['y_loc'] >= unit_tbl['LC_range_bottom'].unique()[0] - 0.1) &
        (unit_tbl['y_loc'] <= unit_tbl['LC_range_top'].unique()[0] + 0.1)
    )
    unit_ids_focus = unit_tbl[filter]['unit_id'].to_list()
    auto_corr = load_auto_corr(session, data_type)
    cross_corr = load_cross_corr(session, data_type)
    if auto_corr is None or cross_corr is None:
        print(f'No auto/cross correlation data found for session {session} and data type {data_type}. Exiting.')
        return None
    gs_crosscorr = gs.GridSpec(len(unit_ids_focus), len(unit_ids_focus))
    fig = plt.figure(figsize=(20, 20))
    for unit_1_ind, unit_1 in enumerate(unit_ids_focus):
        for unit_2_ind, unit_2 in enumerate(unit_ids_focus):
            ax = fig.add_subplot(gs_crosscorr[unit_1_ind, unit_2_ind])
            if unit_1 == unit_2:
                unit_auto_corr = auto_corr.load_unit(unit_1)
                ax.plot(unit_auto_corr['long_lags'][1:], unit_auto_corr['auto_corr_long'][1:], label='Long', color='k')
                # ax.plot(unit_auto_corr['short_lags'], unit_auto_corr['auto_corr_short'], label='Short', color='orange')
                ax.set_title(f'Unit {unit_1} Auto-Corr')
                # ax.legend()
            else:
                cross_corr_units = cross_corr.load_units(unit_1, unit_2)
                if cross_corr_units is not None:
                    ax.plot(cross_corr_units['long_lags'], cross_corr_units['cross_corr_long'], label='Long', color='blue')
                    # ax.plot(cross_corr['short_lags'], cross_corr['cross_corr_short'], label='Short', color='orange')
                    ax.set_title(f'{unit_1}')
                    ax.set_ylabel(f'{unit_2}')
                    # ax.legend()
                else:
                    ax.set_visible(False)
            if unit_tbl[unit_tbl['unit_id'] == unit_1]['opto_pass'].values[0]:
                ax.spines['bottom'].set_color('red')        # x-axis line
                ax.tick_params(axis='x', colors='red')      # tick labels and ticks
                ax.xaxis.label.set_color('red')
                ax.set_title(ax.get_title(), color='red')
            if unit_tbl[unit_tbl['unit_id'] == unit_2]['opto_pass'].values[0]:
                ax.spines['left'].set_color('red')         # y-axis line
                ax.tick_params(axis='y', colors='red')
                ax.yaxis.label.set_color('red')

    plt.tight_layout()
    fig.savefig(fname=os.path.join(session_dirs(session)[f'ephys_processed_dir_{data_type}'], f'{session}_{data_type}_cross_auto_corr.png'))

# %%
# cross_auto_corr('ecephys_713854_2024-03-08_14-54-25', 'curated')
if __name__ == '__main__':

# %%
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
        print(session_dir[f'curated_dir_{data_type}'])
        if session_dir[f'curated_dir_{data_type}'] is not None:
            # try:
            # plot_ephys_probe(session, data_type=data_type, probe=probe) 
            cross_auto_corr(session, data_type)
            plot_cross_auto_corr(session, data_type)
            plt.close('all')
            print(f'Finished {session}')
        # except:
            print(f'Error processing {session}')
            plt.close('all')
        else: 
            print(f'No curated data found for {session}') 
        # elif session\_dir['curated_dir_raw'] is not None:
        #     data_type = 'raw' 
        #     opto_tagging_df_sess = opto_plotting_session(session, data_type, target, resp_thresh=resp_thresh, lat_thresh=lat_thresh, target_unit_ids= None, plot = True, save=True)
    Parallel(n_jobs=-8)(
        delayed(process)(session, data_type) 
        for session in session_list
    )

    # process('ecephys_713854_2024-03-05_13-01-09', data_type)
    # for session, probe in zip(session_list, probe_list):
    #     process(session, data_type, probe)
    #     plt.close('all')



