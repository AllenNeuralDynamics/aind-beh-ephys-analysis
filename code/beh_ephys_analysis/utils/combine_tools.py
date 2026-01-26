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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.model_selection import KFold
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ttest_ind

def apply_qc(combined_tagged_units, constraints, density = True, plot_all = True):
    # start with a mask of all True
    mask = pd.Series(True, index=combined_tagged_units.index)
    mask_no_opto = pd.Series(True, index=combined_tagged_units.index)
    opto_list = ['p_max', 'eu', 'corr', 'tag_loc', 'lat_max_p', 'p_mean', 'sig_counts']
    multi_condition_list = ['p_max', 'eu', 'corr', 'lat_max_p', 'p_mean', 'sig_counts']
    for col, cfg in constraints.items():
        curr_count_true = mask.sum()
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
                mask &= combined_tagged_units[col] >= lb
            if not np.isnan(ub):
                mask &= combined_tagged_units[col] <= ub


        # Categorical list?
        elif "items" in cfg:
            print(f'Applying items for {col}: {cfg["items"]}')
            allowed = cfg["items"]
            mask &= combined_tagged_units[col].isin(allowed)
        
        elif "conditional" in cfg:
            print(f'Applying conditional bounds for {col}')
            cond_mask = pd.Series(False, index=combined_tagged_units.index)
            for cond in cfg["conditional"]:
                if "if" in cond:
                    for dep_col, (op, val) in cond["if"].items():
                        if dep_col not in combined_tagged_units:
                            print(f"Dependent column {dep_col} missing, skipping condition.")
                            continue
                        if op == "<":
                            condition = combined_tagged_units[dep_col] < val
                        elif op == "<=":
                            condition = combined_tagged_units[dep_col] <= val
                        elif op == ">":
                            condition = combined_tagged_units[dep_col] > val
                        elif op == ">=":
                            condition = combined_tagged_units[dep_col] >= val
                        elif op == "==":
                            condition = combined_tagged_units[dep_col] == val
                        else:
                            raise ValueError(f"Unsupported operator: {op}")

                        lb, ub = np.array(cond["bounds"], dtype=float)
                        bounds_mask = pd.Series(True, index=combined_tagged_units.index)
                        if not np.isnan(lb):
                            bounds_mask &= combined_tagged_units[col] >= lb
                        if not np.isnan(ub):
                            bounds_mask &= combined_tagged_units[col] <= ub

                        cond_mask |= (condition & bounds_mask)

                elif "else" in cond:
                    lb, ub = np.array(cond["bounds"], dtype=float)
                    bounds_mask = pd.Series(True, index=combined_tagged_units.index)
                    if not np.isnan(lb):
                        bounds_mask &= combined_tagged_units[col] >= lb
                    if not np.isnan(ub):
                        bounds_mask &= combined_tagged_units[col] <= ub
                    cond_mask |= (~condition & bounds_mask)

            mask &= cond_mask
        new_count_true = mask.sum()
        print(f' - {col}: {curr_count_true} -> {new_count_true} units passed')
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
    print(f'Number of opto rows after filtering: {len(combined_tagged_units_filtered)}')
    print(f'Number of non-opto rows after filtering: {len(combined_tagged_units[mask_no_opto])}')

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

        if "bounds" in cfg or "conditional" in cfg:
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
            ax.hist(full_data, bins=bins, color='gray', edgecolor=None, alpha=0.5, label='All', density=density)
            ax.hist(filtered_data, bins=bins, color='orange', edgecolor=None, alpha=0.5, label='Opto-Filtered', density=density)
            if plot_all:
                ax.hist(half_filtered_data, bins=bins, color='green', edgecolor=None, alpha=0.5, label='Filtered', density=density)
            
                

            if "bounds" in cfg:
                ax.set_title(f'{col}\nBounds: [{lb}, {ub}]')
            ax.set_xlabel(col)
            if density:
                ax.set_ylabel('Probability Density')
            else:
                ax.set_ylabel('Count')

            if "bounds" in cfg:
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
    # plt.show()

    return combined_tagged_units_filtered, combined_tagged_units, fig

def to_str_intlike(x):
    """
    Convert any integer-like or numeric-like value to a clean string without '.0'.

    Examples:
        297       → '297'
        '297.0'   → '297'
        297.0     → '297'
        np.int64(297) → '297'
        'abc'     → 'abc'   (unchanged)
        np.nan    → np.nan   (kept as NaN)
    """
    if pd.isna(x):
        return np.nan  # or return '' if you prefer empty string for NaN

    # Try to convert to float first (handles both numeric and numeric-like strings)
    try:
        val = float(x)
        # If it's a whole number like 297.0 → cast to int and then str
        if val.is_integer():
            return str(int(val))
        else:
            # Keep decimals if not an integer-like number
            return str(val)
    except (ValueError, TypeError):
        # Non-numeric string: return as-is
        return str(x)


def merge_df_with_suffix(dfs, on_list, prefixes = None, suffixes = None, how = 'left'):
    """
    Merges multiple DataFrames on specified columns, renaming columns with prefixes and suffixes.
    """
    # check if dfs, prefixes, and suffixes are lists of same lengths
    
    for ind, df in enumerate(dfs):
        # loop through columns and rename them with suffix
        for col in df.columns:
            if col not in on_list:
                col_name = col
                if prefixes is not None:
                    col_name = f"{prefixes[ind]}_{col_name}"
                if suffixes is not None:
                    col_name = f"{col_name}_{suffixes[ind]}"
                # print(f"Renaming column '{col}' in DataFrame to '{col}_{suffixes[ind]}'")
                df.rename(columns={col: col_name}, inplace=True)
    
    merge_df = dfs[0]
    for df in dfs[1:]:
        merge_df = merge_df.merge(df, on=on_list, how=how)
    return merge_df


# def spatial_dependence_summary(
#     coords,
#     values,
#     *,
#     k_neighbors=15,
#     n_splits=5,
#     permutations=5000,
#     seed=0,
#     return_null=False,
# ):
#     """
#     Combines:
#       (1) Linear trend test: value ~ x + y + z (F-test vs intercept-only)
#       (2) 5-fold CV predictability with kNN (distance-weighted) + permutation p-value

#     Returns a dictionary of results.
#     """
#     rng = np.random.default_rng(seed)

#     X = np.asarray(coords, float)
#     y = np.asarray(values, float)

#     # drop NaNs / infs
#     ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
#     X = X[ok]
#     y = y[ok]
#     n = y.size
#     if n < 10:
#         raise ValueError(f"Too few valid points after filtering (n={n}).")

#     # -------------------------
#     # (1) Linear regression trend
#     # -------------------------
#     X_const = sm.add_constant(X)  # [1, x, y, z]
#     model_full = sm.OLS(y, X_const).fit()
#     model_null = sm.OLS(y, np.ones((n, 1))).fit()
#     f_stat, p_trend, _ = model_full.compare_f_test(model_null)

#     # -------------------------
#     # (2) CV predictability (kNN) + permutation test
#     # -------------------------
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

#     def cv_r2(y_target):
#         preds = np.zeros_like(y_target)
#         for tr, te in kf.split(X):
#             model = KNeighborsRegressor(
#                 n_neighbors=min(k_neighbors, len(tr)),
#                 weights="distance",
#             )
#             model.fit(X[tr], y_target[tr])
#             preds[te] = model.predict(X[te])
#         return r2_score(y_target, preds)

#     r2_cv_obs = cv_r2(y)

#     r2_cv_perm = np.empty(permutations, float)
#     for i in range(permutations):
#         r2_cv_perm[i] = cv_r2(rng.permutation(y))

#     p_cv = (np.sum(r2_cv_perm >= r2_cv_obs) + 1) / (permutations + 1)

#     out = {
#         "n_used": int(n),
#         "inputs": {
#             "k_neighbors": int(k_neighbors),
#             "n_splits": int(n_splits),
#             "permutations": int(permutations),
#             "seed": int(seed),
#         },
#         "linear_trend": {
#             "coef_const_x_y_z": model_full.params.tolist(),      # [const, bx, by, bz]
#             "p_value_F_test_vs_intercept_only": float(p_trend),
#             "F_stat": float(f_stat),
#             "r2": float(model_full.rsquared),
#         },
#         "cv_predictability_knn": {
#             "r2_cv": float(r2_cv_obs),
#             "p_value_permutation": float(p_cv),                 # one-sided: better than random
#             "null_mean": float(np.mean(r2_cv_perm)),
#             "null_std": float(np.std(r2_cv_perm, ddof=1)),
#         },
#     }

#     if return_null:
#         out["cv_predictability_knn"]["r2_null_distribution"] = r2_cv_perm.tolist()

#     return out

def spatial_dependence_summary(
    coords,
    values,
    *,
    k_neighbors=15,
    n_splits=5,
    permutations=5000,
    seed=0,
    return_null=False,
):
    """
    Combines:
      (1) Linear trend test: value ~ x + y + z (F-test vs intercept-only)
      (2) 5-fold CV predictability with kNN (distance-weighted) + permutation p-value

    Returns a dictionary of results.
    """
    import numpy as np
    import statsmodels.api as sm
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)

    X = np.asarray(coords, float)
    y = np.asarray(values, float)

    # drop NaNs / infs
    # reshape X to N by number of dimensions
    X = X.reshape(X.shape[0], -1)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[ok]
    y = y[ok]
    n = y.size
    if n < 10:
        raise ValueError(f"Too few valid points after filtering (n={n}).")

    # -------------------------
    # (1) Linear regression trend
    # -------------------------
    X_const = sm.add_constant(X)  # [1, x, y, z]
    model_full = sm.OLS(y, X_const).fit()
    model_null = sm.OLS(y, np.ones((n, 1))).fit()
    f_stat, p_trend, _ = model_full.compare_f_test(model_null)

    # -------------------------
    # (2) CV predictability (kNN) + permutation test (parallel)
    # -------------------------
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def cv_r2(y_target):
        preds = np.zeros_like(y_target)
        for tr, te in kf.split(X):
            model = KNeighborsRegressor(
                n_neighbors=min(k_neighbors, len(tr)),
                weights="distance",
            )
            model.fit(X[tr], y_target[tr])
            preds[te] = model.predict(X[te])
        return r2_score(y_target, preds)

    r2_cv_obs = cv_r2(y)

    # --- parallel permutation null ---
    # Make deterministic per-permutation seeds
    perm_seeds = np.random.SeedSequence(seed).spawn(permutations)
    perm_seeds_uint = np.array([s.generate_state(1, dtype=np.uint32)[0] for s in perm_seeds], dtype=np.uint32)

    def _one_perm_r2(s_uint):
        r = np.random.default_rng(int(s_uint))
        return cv_r2(r.permutation(y))

    r2_cv_perm = None
    try:
        from joblib import Parallel, delayed

        r2_cv_perm = np.asarray(
            Parallel(n_jobs=-1, prefer="processes")(
                delayed(_one_perm_r2)(s_uint) for s_uint in perm_seeds_uint
            ),
            dtype=float,
        )
    except Exception:
        # Fallback to sequential if joblib isn't available or parallel fails
        r2_cv_perm = np.empty(permutations, float)
        for i, s_uint in enumerate(perm_seeds_uint):
            r2_cv_perm[i] = _one_perm_r2(s_uint)

    p_cv = (np.sum(r2_cv_perm >= r2_cv_obs) + 1) / (permutations + 1)

    out = {
        "n_used": int(n),
        "inputs": {
            "k_neighbors": int(k_neighbors),
            "n_splits": int(n_splits),
            "permutations": int(permutations),
            "seed": int(seed),
        },
        "linear_trend": {
            "coef_const_x_y_z": model_full.params.tolist(),  # [const, bx, by, bz]
            "p_value_F_test_vs_intercept_only": float(p_trend),
            "F_stat": float(f_stat),
            "r2": float(model_full.rsquared),
        },
        "cv_predictability_knn": {
            "r2_cv": float(r2_cv_obs),
            "p_value_permutation": float(p_cv),  # one-sided: better than random
            "null_mean": float(np.mean(r2_cv_perm)),
            "null_std": float(np.std(r2_cv_perm, ddof=1)),
        },
    }

    if return_null:
        out["cv_predictability_knn"]["r2_null_distribution"] = r2_cv_perm.tolist()

    return out



def welch_shift_P_vs_U(
    x_all,
    is_known_P,
    *,
    alternative="two-sided",   # "two-sided", "greater", "less"
    permutations=20000,
    seed=0,
    n_boot=5000,
):
    """
    More powerful distribution-shift test for continuous, roughly normal data:
      Known P vs unlabeled U using Welch's t-test (unequal variances),
      plus a label-permutation p-value and effect size.

    Returns a dict with stats and uncertainty.
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x_all, float)
    mP = np.asarray(is_known_P, dtype=bool)

    # filter NaNs
    ok = np.isfinite(x) & np.isfinite(mP.astype(float))
    x = x[ok]
    mP = mP[ok]

    xP = x[mP]
    xU = x[~mP]
    nP, nU = len(xP), len(xU)

    if nP < 2 or nU < 2:
        raise ValueError(f"Need at least 2 samples in each group. Got nP={nP}, nU={nU}.")

    # --- observed Welch t-test ---
    t_obs, p_asym = ttest_ind(xP, xU, equal_var=False, alternative=alternative)

    # effect size: Cohen's d (using pooled SD; for unequal variances, this is still a useful standardized diff)
    sP = np.var(xP, ddof=1)
    sU = np.var(xU, ddof=1)
    s_pooled = np.sqrt(((nP - 1) * sP + (nU - 1) * sU) / (nP + nU - 2))
    cohen_d = (np.mean(xP) - np.mean(xU)) / (s_pooled + 1e-12)

    # --- permutation p-value (label shuffle; keeps group sizes fixed) ---
    x_all_clean = np.concatenate([xP, xU])
    n = len(x_all_clean)
    idx = np.arange(n)

    t_perm = np.empty(permutations, float)
    for i in range(permutations):
        rng.shuffle(idx)
        p_idx = idx[:nP]
        u_idx = idx[nP:]
        # Welch t statistic on permuted labels
        t_perm[i] = ttest_ind(
            x_all_clean[p_idx],
            x_all_clean[u_idx],
            equal_var=False,
            alternative="two-sided",   # compute symmetric t; handle alternative below
        ).statistic

    # convert perm stats to p-value matching the requested alternative
    if alternative == "two-sided":
        p_perm = (np.sum(np.abs(t_perm) >= np.abs(t_obs)) + 1) / (permutations + 1)
    elif alternative == "greater":
        # t is larger when mean(P) > mean(U)
        p_perm = (np.sum(t_perm >= t_obs) + 1) / (permutations + 1)
    elif alternative == "less":
        p_perm = (np.sum(t_perm <= t_obs) + 1) / (permutations + 1)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")

    # --- bootstrap CI for mean difference (P - U) ---
    # (Useful and interpretable for firing rate)
    boot = np.empty(n_boot, float)
    for i in range(n_boot):
        bp = rng.choice(xP, size=nP, replace=True)
        bu = rng.choice(xU, size=nU, replace=True)
        boot[i] = np.mean(bp) - np.mean(bu)
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    return {
        "n_total": int(n),
        "n_P": int(nP),
        "n_U": int(nU),
        "test": "Welch t-test (P vs unlabeled U)",
        "alternative": alternative,
        "mean_P": float(np.mean(xP)),
        "mean_U": float(np.mean(xU)),
        "mean_diff_P_minus_U": float(np.mean(xP) - np.mean(xU)),
        "mean_diff_bootstrap_CI95": [float(ci_low), float(ci_high)],
        "t_stat": float(t_obs),
        "p_value_asymptotic": float(p_asym),
        "p_value_permutation": float(p_perm),
        "cohens_d": float(cohen_d),
    }


def binary_shift_P_vs_U(
    x_all,
    is_known_P,
    *,
    alternative="two-sided",  # "two-sided", "larger", "smaller" (Fisher wording)
    permutations=20000,
    seed=0,
):
    """
    Test whether binary values differ between known P and unlabeled U (mixture).

    Returns:
      - event rates
      - odds ratio + Fisher exact p-value
      - two-proportion z-test p-value
      - permutation p-value (label shuffle)
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x_all)
    mP = np.asarray(is_known_P, dtype=bool)

    # Keep only finite / valid entries; ensure binary
    ok = np.isfinite(x.astype(float)) & np.isfinite(mP.astype(float))
    x = x[ok].astype(int)
    mP = mP[ok]
    if not np.isin(x, [0, 1]).all():
        raise ValueError("x_all must be binary (0/1).")

    xP = x[mP]
    xU = x[~mP]

    nP, nU = len(xP), len(xU)
    if nP < 1 or nU < 1:
        raise ValueError(f"Need at least 1 sample in each group. Got nP={nP}, nU={nU}.")

    # counts
    a = int(xP.sum())           # P successes
    b = int(nP - a)             # P failures
    c = int(xU.sum())           # U successes
    d = int(nU - c)             # U failures

    rate_P = a / nP
    rate_U = c / nU
    risk_diff = rate_P - rate_U
    risk_ratio = (rate_P / rate_U) if rate_U > 0 else np.inf

    # 2x2 table for Fisher
    table = np.array([[a, b],
                      [c, d]], dtype=int)

    # Fisher exact: alternative uses "larger"/"smaller"/"two-sided"
    OR, p_fisher = fisher_exact(table, alternative=alternative)

    # Two-proportion z-test (more powerful when counts are moderate)
    # statsmodels uses alternative: 'two-sided', 'larger', 'smaller'
    z_stat, p_z = proportions_ztest(
        count=[a, c],
        nobs=[nP, nU],
        alternative=alternative
    )

    # Permutation test on risk difference (rate_P - rate_U)
    x_all_clean = np.concatenate([xP, xU])
    n = len(x_all_clean)
    idx = np.arange(n)

    perm_stats = np.empty(permutations, float)
    for i in range(permutations):
        rng.shuffle(idx)
        p_idx = idx[:nP]
        u_idx = idx[nP:]
        perm_stats[i] = x_all_clean[p_idx].mean() - x_all_clean[u_idx].mean()

    if alternative == "two-sided":
        p_perm = (np.sum(np.abs(perm_stats) >= abs(risk_diff)) + 1) / (permutations + 1)
    elif alternative == "larger":
        p_perm = (np.sum(perm_stats >= risk_diff) + 1) / (permutations + 1)
    else:  # "smaller"
        p_perm = (np.sum(perm_stats <= risk_diff) + 1) / (permutations + 1)

    return {
        "n_P": int(nP),
        "n_U": int(nU),
        "count_P_1": int(a),
        "count_U_1": int(c),
        "rate_P": float(rate_P),
        "rate_U": float(rate_U),
        "risk_diff_P_minus_U": float(risk_diff),
        "risk_ratio_P_over_U": float(risk_ratio),
        "odds_ratio_fisher": float(OR),
        "p_value_fisher": float(p_fisher),
        "z_stat": float(z_stat),
        "p_value_ztest": float(p_z),
        "p_value_permutation": float(p_perm),
    }

# Example:
# res = welch_shift_P_vs_U(x_all, is_known_P, alternative="two-sided", permutations=20000, seed=0)
# print(res)
