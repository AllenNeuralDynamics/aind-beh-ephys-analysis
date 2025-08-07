import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
import ast
import json
import seaborn as sns
from PyPDF2 import PdfMerger
import re
from utils.beh_functions import session_dirs, parseSessionID, load_model_dv, makeSessionDF, get_session_tbl, get_unit_tbl, get_history_from_nwb
import pandas as pd
import pickle
import scipy.stats as stats
import shutil 
import seaborn as sns
import math  
import seaborn as sns

def apply_qc(combined_tagged_units, constraints):
    # start with a mask of all True
    mask = pd.Series(True, index=combined_tagged_units.index)
    mask_no_opto = pd.Series(True, index=combined_tagged_units.index)
    opto_list = ['p_max', 'eu', 'corr', 'tag_loc', 'lat_max_p', 'p_mean', 'sig_counts']
    multi_condition_list = ['p_max', 'eu', 'corr', 'lat_max_p', 'p_mean', 'sig_counts']
    for col, cfg in constraints.items():
        if col not in combined_tagged_units:
            print(f'Column {col} not found in combined_tagged_units, skipping...')
            continue
        if col in opto_list:
            continue  # skip opto conditions for now

        # Numeric range?
        if "bounds" in cfg:
            print(f'Applying bounds for {col}: {cfg["bounds"]}')
            lb, ub = np.array(cfg["bounds"], dtype=float)  # np.nan for null
            if not np.isnan(lb):
                mask &= combined_tagged_units[col] > lb
            if not np.isnan(ub):
                mask &= combined_tagged_units[col] < ub

        # Categorical list?
        elif "items" in cfg:
            print(f'Applying items for {col}: {cfg["items"]}')
            allowed = cfg["items"]
            mask &= combined_tagged_units[col].isin(allowed)
    mask_no_opto = mask.copy()  
    combined_tagged_units['selected_qc_only'] = mask
    # for each neuron, apply combined opto constraints
    opto_pass = np.full(len(combined_tagged_units), False)
    opto_cond_list = ['all_p_max', 'all_p_mean', 'all_lat_max_p', 'all_corr', 'all_eu', 'all_sig_counts']
    
    if len(set(opto_cond_list) & set(combined_tagged_units.columns)) > 0:
        print(f'Applying opto conditions: {opto_list}')

        for ind, row in combined_tagged_units.iterrows():
            curr_opto_tbl = pd.DataFrame(row[opto_cond_list].to_dict())
            opto_pass_curr = np.full(len(curr_opto_tbl), True)
            for col, cfg in constraints.items():
                if col not in opto_list or col == 'tag_loc':
                    continue
                else: 
                    if "bounds" in cfg:
                        lb, ub = np.array(cfg["bounds"], dtype=float)
                        if not np.isnan(lb):
                            opto_pass_curr &= curr_opto_tbl[f'all_{col}'] > lb
                        if not np.isnan(ub):
                            opto_pass_curr &= curr_opto_tbl[f'all_{col}'] < ub
                    elif "items" in cfg:
                        allowed = cfg["items"]
                        opto_pass_curr &= curr_opto_tbl[f'all_{col}'].isin(allowed)
            if opto_pass_curr.any():
                opto_pass[ind] = True
                ind_max = np.where(opto_pass_curr)[0][np.argmax(curr_opto_tbl[opto_pass_curr]['all_p_max'])]
                combined_tagged_units.at[ind, 'p_max'] = curr_opto_tbl['all_p_max'].values[ind_max].astype(float)
                combined_tagged_units.at[ind, 'p_mean'] = curr_opto_tbl['all_p_mean'].values[ind_max].astype(float)
                combined_tagged_units.at[ind, 'lat_max_p'] = curr_opto_tbl['all_lat_max_p'].values[ind_max].astype(float)
                combined_tagged_units.at[ind, 'corr'] = curr_opto_tbl['all_corr'].values[ind_max].astype(float)
                combined_tagged_units.at[ind, 'eu'] = curr_opto_tbl['all_eu'].values[ind_max].astype(float)
                combined_tagged_units.at[ind, 'count_max'] = curr_opto_tbl['all_sig_counts'].values[ind_max].astype(float)

        combined_tagged_units['selected_opto_only'] = opto_pass
        mask_all = mask & opto_pass
    else:
        mask_all = mask
    # apply and get filtered DataFrame
    combined_tagged_units_filtered = combined_tagged_units[mask_all].reset_index(drop=True)
    combined_tagged_units['selected'] = mask_all
    combined_tagged_units['selected_no_opto'] = mask_no_opto
    print(f'Number of opto units after filtering: {len(combined_tagged_units_filtered)}')
    print(f'Number of non-opto units after filtering: {len(combined_tagged_units[mask_no_opto])}')

    # plot
    valid_constraints = {col: cfg for col, cfg in constraints.items() if col in combined_tagged_units.columns}
    n = len(valid_constraints)

    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 4))
    axes = axes.flatten()

    for i, (col, cfg) in enumerate(valid_constraints.items()):
        ax = axes[i]
        full_data = combined_tagged_units[col].dropna()
        filtered_data = combined_tagged_units_filtered[col].dropna()
        half_filtered_data = combined_tagged_units[mask_no_opto][col].dropna()

        if "bounds" in cfg:
            lb, ub = np.array(cfg["bounds"], dtype=float)
            # bins = np.linspace(np.nanmin(full_data), np.nanmax(full_data), 50)
            # bins[0] = bins[0]-0.0001
            # bins[-1] = bins[-1]+0.0001
            # if col=='peak':
            #     bins = np.linspace(-500, 300, 50)
            bins = np.linspace(np.nanmin(full_data), np.nanmax(full_data), 50)
            match col:
                case 'isi_violations':
                    bins =  np.linspace(0, 1, 50)
                case 'eu':
                    bins =  np.linspace(0, 1, 50)
                case 'corr':
                    bins =  np.linspace(0.7, 1, 30)
                case 'p_max':
                    bins =  np.linspace(0, 1, 50)
                case 'lat_max_p':
                    bins =  np.linspace(0, 0.025, 50)
                case 'peak':
                    bins = np.linspace(-500, 300, 50)
                case 'snr':
                    bins = np.linspace(0, 50, 50)
                case 'peak_raw':
                    bins = np.linspace(-500, 300, 50)
            ax.hist(full_data, bins=bins, color='gray', edgecolor=None, alpha=0.5, label='All units', density=True)
            ax.hist(filtered_data, bins=bins, color='orange', edgecolor=None, alpha=0.5, label='Filtered units', density=True)
            ax.hist(half_filtered_data, bins=bins, color='green', edgecolor=None, alpha=0.5, label='Non-opto units', density=True)
            
                

            ax.set_title(f'{col}\nBounds: [{lb}, {ub}]')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')

            if not np.isnan(lb) and lb > full_data.min():
                ax.axvline(lb, color='red', linestyle='--', label='Lower bound')
            if not np.isnan(ub) and ub < full_data.max():
                ax.axvline(ub, color='green', linestyle='--', label='Upper bound')
            ax.legend()

        elif "items" in cfg:
            full_counts = combined_tagged_units[col].dropna().astype(str).value_counts()
            filtered_counts = combined_tagged_units_filtered[col].dropna().astype(str).value_counts()

            # Get all unique category labels from both datasets
            all_idx = full_counts.index.union(filtered_counts.index)

            # Reindex both with all categories, fill missing with 0
            full_counts = full_counts.reindex(all_idx, fill_value=0)
            filtered_counts = filtered_counts.reindex(all_idx, fill_value=0)

            # Plot
            full_counts.plot(kind='bar', ax=ax, color='skyblue', alpha=0.5, edgecolor='black', label='All units')
            filtered_counts.plot(kind='bar', ax=ax, color='orange', alpha=0.7, edgecolor='black', label='Filtered units')



            for item in cfg["items"]:
                if str(item) in full_counts.index:
                    ax.axvline(x=str(item), color='red', linestyle='--')

            ax.set_title(f'{col}\nItems: {cfg["items"]}') 
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.legend()

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Comparison of All Units vs Filtered {len(combined_tagged_units_filtered)}/{len(combined_tagged_units)} passed', fontsize=16)

    plt.tight_layout()
    # plt.savefig(os.path.join(wf_folder, f'Pass_histo_{criteria_name}.pdf'))
    plt.show()

    return combined_tagged_units_filtered, combined_tagged_units, fig