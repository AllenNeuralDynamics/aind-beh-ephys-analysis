def get_unit_tiers(session):
    """
    For a given session, load data and return unit_tiers DataFrame
    with tier assignment (tier 3 if p_val < 0.05 except surface_LC).
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd

    data_folder = rf'D:\Ephys\OptoTaggingAnalysis\antidromic\{session}'
    with open(os.path.join(data_folder, f'{session}_curated_soma_opto_tagging_summary.pkl'), 'rb') as f:
        unit_tbl = pickle.load(f)
    opto_units = unit_tbl.query("opto_pass == True & default_qc == True")["unit_id"].to_list()

    with open(os.path.join(data_folder, 'spiketimes.pkl'), 'rb') as f:
        spiketimes = pickle.load(f)
    opto_event_file = os.path.join(data_folder, f'{session}_opto_session.csv')
    event_ids = pd.read_csv(opto_event_file)

    antidromic_results = []
    for unit_id in opto_units:
        unit_spike_times = spiketimes[unit_id]
        for site in event_ids['site'].unique():
            if site == 'surface_LC':
                base_window = (-0.02, 0)
                roi_window = (0, 0.02)
            else:
                base_window = (-0.02, 0)
                roi_window = (0.03, 0.05)

            tag_trials = event_ids.query('site == @site and pre_post == "post"')
            if tag_trials.empty:
                continue
            max_power = tag_trials.power.max()
            tag_trials = tag_trials.query('power == @max_power')
            if tag_trials.empty:
                continue

            duration = tag_trials.duration.iloc[0]
            num_pulses = tag_trials.num_pulses.iloc[0]
            pulse_interval = tag_trials.pulse_interval.iloc[0]
            time_range_raster = np.array([-100 / 1000, 70 / 1000])
            this_event_timestamps = tag_trials.time.tolist()

            int_event_locked_timestamps = []
            for pulse_num in range(num_pulses):
                time_shift = pulse_num * (duration + pulse_interval) / 1000
                this_time_range = time_range_raster + time_shift
                this_locked = af.event_locked_timestamps(
                    unit_spike_times, this_event_timestamps, this_time_range, time_shift=time_shift
                )
                int_event_locked_timestamps.extend(this_locked)

            p_val = opto_tagging_response(int_event_locked_timestamps, base_window, roi_window)
            antidromic_results.append({
                'unit_id': unit_id,
                'site': site,
                'p_val': p_val                
            })
            if p_val < 0.05:
                first_spikes_after_light = np.array([arr[arr > 0][0] for arr in int_event_locked_timestamps if np.any(arr > 0)])
                median_first_spikes_after_light = np.median(first_spikes_after_light)
                jitteriness = np.std(first_spikes_after_light)
                antidromic_results[-1]['median_first_spike_latency'] = median_first_spikes_after_light
                antidromic_results[-1]['jitter'] = jitteriness
                antidromic_latency = median_first_spikes_after_light

                if site != 'surface_LC':
                    collision_results = collision_test(int_event_locked_timestamps, antidromic_latency, bin_size=10, antidromic_jitter=0.005, plot=False)
                    antidromic_results[-1]['collision_pvalue'] = collision_results['p_value']
                    antidromic_results[-1]['collision_pbinom'] = collision_results['p_binom']
            if site == 'surface_S1':
                break

    antidromic_df = pd.DataFrame(antidromic_results)
    antidromic_pivot = antidromic_df.pivot(index='unit_id', columns='site').reset_index()

    def categorize_units(antidromic_pivot):
        # Flatten MultiIndex columns if needed
        if isinstance(antidromic_pivot.columns, pd.MultiIndex):
            antidromic_pivot.columns = ['_'.join([str(i) for i in col if i]) for col in antidromic_pivot.columns.values]
        categories = []
        for idx, row in antidromic_pivot.iterrows():
            unit_id = row['unit_id']
            tier = None
            for col in antidromic_pivot.columns:
                if col.startswith('p_val_') and not col.endswith('surface_LC'):
                    site = col.replace('p_val_', '')
                    p_val = row[col]
                    jitter_col = f'jitter_{site}'
                    jitter = row.get(jitter_col, None)
                    if pd.notnull(p_val) and p_val < 0.05:
                        tier = 3
                        if pd.notnull(jitter) and jitter < 0.005:
                            collision_pvalue_col = f'collision_pvalue_{site}'
                            collision_pbinom_col = f'collision_pbinom_{site}'
                            collision_pvalue = row.get(collision_pvalue_col, None)
                            collision_pbinom = row.get(collision_pbinom_col, None)
                            if pd.notnull(collision_pvalue) and collision_pvalue < 0.05 and collision_pbinom == 1:
                                tier = 1
                            else:
                                tier = 2
            if tier is None:
                tier = 0
            categories.append({'unit_id': unit_id, 'tier': tier, 'site': site if tier > 0 else None})
        return pd.DataFrame(categories)

    unit_tiers = categorize_units(antidromic_pivot)
    return unit_tiers