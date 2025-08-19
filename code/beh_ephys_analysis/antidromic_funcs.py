from scipy.stats import wilcoxon
# Scientific libraries
import numpy as np
import pandas as pd
import utils.analysis_funcs as af
import utils.plotting_funcs as pf

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches


def remove_spikes_during_laser_pulse(int_event_locked_timestamps, duration):    
    for i, arr in enumerate(int_event_locked_timestamps):
        # Remove spikes during the laser pulse        
        int_event_locked_timestamps[i] = arr[(arr < 0) | (arr > (duration / 1000))]        
    return int_event_locked_timestamps

def opto_tagging_response(int_event_locked_timestamps, base_window, roi_window):
    base_counts = []
    roi_counts = []
    for spike_times in int_event_locked_timestamps:
        if spike_times.size > 0:
            base_count = np.sum((spike_times >= base_window[0]) & (spike_times < base_window[1]))
            roi_count = np.sum((spike_times >= roi_window[0]) & (spike_times < roi_window[1]))
            base_counts.append(base_count)
            roi_counts.append(roi_count)

    base_counts = np.array(base_counts)
    roi_counts = np.array(roi_counts)

    # ✅ Only run test if we have at least 2 samples
    if len(base_counts) >= 2 and len(roi_counts) >= 2:
        try:
            stat, p_value = wilcoxon(base_counts, roi_counts)
            return p_value
        except ValueError:
            return np.nan
    else:
        return np.nan

def antidromic_latency_jitter(int_event_locked_timestamps):
    from scipy.ndimage import gaussian_filter1d
    pulse_duration = 0.005 # ms    
    first_spikes_after_light = np.array([arr[arr > pulse_duration][0] for arr in int_event_locked_timestamps if np.any(arr > pulse_duration)])
    histogram, bins = np.histogram(first_spikes_after_light, bins=np.arange(pulse_duration, 0.1, 0.0005))
    # Convolve the histogram with a Gaussian kernel
    sigma = 0.5 / 1000 / (bins[1] - bins[0])  # Convert sigma to bin units    
    smoothed_psth = gaussian_filter1d(histogram, sigma)
    
    # Find the peak of the smoothed PSTH
    peak_index = np.argmax(smoothed_psth)
    peak_time = bins[peak_index]
    peak_value = smoothed_psth[peak_index]

    # Calculate FWHM (Full Width at Half Maximum)
    half_max = peak_value / 2
    above_half_max = np.where(smoothed_psth >= half_max)[0]
    fwhm = bins[above_half_max[-1]] - bins[above_half_max[0]]

    antidromic_latency = peak_time
    antidromic_jitter = fwhm
    return antidromic_latency, antidromic_jitter


def plot_opto_responses(unit_tbl, event_ids, spiketimes, session_id):
    """
    Find antidromic units based on opto stimulation data.
    """
    # Filter for opto tagged units
    opto_criteria = (unit_tbl['opto_pass'] == True) & (unit_tbl['default_qc'] == True)
    opto_units = unit_tbl[opto_criteria]["unit_id"].to_list()
    print(f"Number of opto units: {len(opto_units)}")

    
    # Unique values
    sites = list(np.unique(event_ids.emission_location))
    powers = list(np.unique(event_ids.power))
    trial_types = np.unique(event_ids.type)

    
    # Settings
    prepost = 'post'
    num_sites = len(sites)
    num_units = len(opto_units)

    # Create one figure for all units × sites (3 rows per unit: raster + PSTH + antidromic raster)
    fig_height_per_unit = 6
    fig = plt.figure(figsize=(num_sites * 4, num_units * fig_height_per_unit))
    fig.suptitle(f'{session_id} # units: {num_units}')    
    gs = gridspec.GridSpec(4 * num_units, num_sites, height_ratios=[3, 6, 2, 0.5] * num_units, hspace=0.8)

    # Loop through units
    for u_idx, unit_id in enumerate(opto_units):
        # if unit_id == 244:
        unit_spike_times = spiketimes[unit_id]

        for i, site in enumerate(sites):
            if site == 'surface_LC':
                # Define windows
                base_window = (-0.02, 0)
                roi_window = (0, 0.02)    
            else:
                # Define windows
                base_window = (-0.02, 0)
                roi_window = (0.03, 0.05)    


            # Filter trials
            tag_trials = event_ids.query('site == @site and pre_post == @prepost')
            max_power = tag_trials.power.max()
            tag_trials = tag_trials.query('power == @max_power')
            if tag_trials.empty:
                continue

            # Stimulation parameters
            duration = np.unique(tag_trials.duration)[0]
            num_pulses = np.unique(tag_trials.num_pulses)[0]
            pulse_interval = np.unique(tag_trials.pulse_interval)[0]

            # Time window
            time_range_raster = np.array([-100 / 1000, 70 / 1000])
            this_event_timestamps = tag_trials.time.tolist()

            int_event_locked_timestamps = []
            pulse_nums = []

            for pulse_num in range(num_pulses):
                time_shift = pulse_num * (duration + pulse_interval) / 1000
                this_time_range = time_range_raster + time_shift

                this_locked = af.event_locked_timestamps(
                    unit_spike_times, this_event_timestamps, this_time_range, time_shift=time_shift
                )
                # Remove spikes during the laser pulse
                int_event_locked_timestamps.extend(this_locked)
                pulse_nums.extend([pulse_num + 1] * len(this_locked))            

            int_event_locked_timestamps = remove_spikes_during_laser_pulse(int_event_locked_timestamps, duration)
            # Raster plot
            ax_raster = fig.add_subplot(gs[4 * u_idx, i])
            pf.raster_plot(int_event_locked_timestamps, time_range_raster, cond_each_trial=pulse_nums, ms=100, ax=ax_raster)
            p_val = opto_tagging_response(int_event_locked_timestamps, base_window, roi_window)


            # Add laser pulse patch
            yLims = np.array(ax_raster.get_ylim())
            laser_color = 'tomato'
            rect = patches.Rectangle((0, yLims[0]), duration / 1000, yLims[1] - yLims[0],
                                    linewidth=1, edgecolor=laser_color, facecolor=laser_color,
                                    alpha=0.2, clip_on=True)
            ax_raster.add_patch(rect)

            # Raster axis settings
            if i == 0:
                ax_raster.set_ylabel(f'Unit {unit_id}\nPulse #')
            else:
                ax_raster.set_yticklabels([])

            ax_raster.set_xlim(time_range_raster)
            if u_idx == num_units - 1:
                ax_raster.set_xlabel('Time (s)')

            if p_val < 0.05:
                text_color = 'red'
                ax_raster.set_title(f'{site}, {max_power} mW, p:{p_val:.3f}', color=text_color)
            else:
                ax_raster.set_title(f'{site}, {max_power} mW, p:{p_val:.3f}', color='black')

            # Add laser pulse aligned but sorted by spike times
            # Antidromic raster plot
            ax_antidromic = fig.add_subplot(gs[4 * u_idx + 1, i], sharex=ax_raster)
            sorted_data = sorted(int_event_locked_timestamps, key=lambda x: (len(x) == 0, x[0] if len(x) > 0 else np.inf))
            pf.raster_plot(sorted_data, time_range_raster)
            # pf.raster_plot(sorted_data, time_range_raster, ax=ax_antidromic)

            yLims = np.array(ax_antidromic.get_ylim())
            rect = patches.Rectangle((0, yLims[0]), duration / 1000, yLims[1] - yLims[0],
                                    linewidth=1, edgecolor='tomato', facecolor='tomato',
                                    alpha=0.2, clip_on=False)
            ax_antidromic.add_patch(rect)

            ax_antidromic.set_xlim(time_range_raster)
            if u_idx == num_units - 1:
                ax_antidromic.set_xlabel('Time (s)')

            if i == 0:
                ax_antidromic.set_ylabel('Sorted Trials')
            else:
                ax_antidromic.set_yticklabels([])

            # PSTH plot
            ax_psth = fig.add_subplot(gs[4 * u_idx + 2, i], sharex=ax_raster)
            psth, _, bins = pf.psth(int_event_locked_timestamps, time_range_raster, bin_size=0.003, smooth_window_size=3)
            antidromic_latency, antidromic_jitter = antidromic_latency_jitter(int_event_locked_timestamps)
            ax_psth.plot(bins, psth, color='k')
            if i == 0:
                ax_psth.set_ylabel('PSTH')
            else:
                ax_psth.set_yticklabels([])
            ax_psth.set_xlim(time_range_raster)
            if u_idx == num_units - 1:
                ax_psth.set_xlabel('Time (s)')
            # Add antidromic latency line
            # if antidromic_latency is not None:
                # ax_psth.axvline(antidromic_latency, color='red', linestyle='--', label='Antidromic latency')
                # ax_psth.axvline(antidromic_latency + antidromic_jitter, color='orange', linestyle='--', label='Antidromic jitter')
                # ax_psth.axvline(antidromic_latency - antidromic_jitter, color='orange', linestyle='--')
                # ax_psth.legend()
            # Add title to PSTH
            # ax_psth.set_title(f'PSTH for {site}, {max_power} mW\nAntidromic latency: {antidromic_latency:.3f} s, Jitter: {antidromic_jitter:.3f} s')




    # Final layout adjustments
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])    
    # plt.show()

    return fig

def Avg_y_over_x(x, y, bin_size):
    """
    Compute the average and standard error of y values binned by x.

    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values.
    bin_size (float): The size of the bins for grouping x values.

    Returns:
    pd.DataFrame: A DataFrame with columns for bin centers (x), average y values (y), and standard error of the mean (sem).
    """
    # Calculate bin edges based on the bin size    
    edges = np.arange(np.min(x), np.max(x) + bin_size, bin_size)
    
    # Digitize x into bins
    x_index = np.digitize(x, edges) - 1  # Bin indices for each x value
    
    # Calculate bin centers
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    # Create a DataFrame to store results
    Avg = pd.DataFrame({'x': bin_centers, 'y': np.nan, 'sem': np.nan})

    # Compute mean and standard error for each bin
    for i in range(len(bin_centers)):
        # Mask y values corresponding to the current bin
        bin_y_values = y[x_index == i]
        if bin_y_values.size > 0:
            Avg.at[i, 'y'] = np.nanmean(bin_y_values)
            Avg.at[i, 'sem'] = np.nanstd(bin_y_values) / np.sqrt(bin_y_values.size)

    return Avg, edges

def analyze_antidromic_responses(opto_units, spiketimes, event_ids, plot=False):
    """
    Analyze antidromic responses for a given set of opto-tagged units.

    Parameters:
        opto_units (list): List of unit IDs.
        spiketimes (dict): Dictionary mapping unit_id -> np.array of spike times.
        event_ids (pd.DataFrame): DataFrame containing stimulation event metadata.
        plot (bool): Whether to plot collision raster for each unit and site.

    Returns:
        pd.DataFrame: DataFrame containing antidromic response metrics and tier categorization.
    """
    
    

    if not opto_units:
        print("No opto units found.")
        return pd.DataFrame()

    antidromic_results = []
    time_range_raster = np.array([-0.1, 0.07])  # in seconds

    for unit_id in opto_units:
        unit_spike_times = spiketimes[unit_id]
        
        for site in event_ids['site'].unique():
            if site == 'surface':
                site = 'surface_LC'  # Treat 'surface' as 'surface_LC' for consistency
            
            # Set analysis windows
            if site != 'surface_LC':
                base_window = (-0.02, 0)
                roi_window = (0, 0.02)
            else:
                base_window = (-0.02, 0)
                roi_window = (0.02, 0.06)
            
            tag_trials = event_ids.query('site == @site and pre_post == "post"')
            if tag_trials.empty:
                continue

            max_power = tag_trials['power'].max()
            tag_trials = tag_trials.query('power == @max_power')
            
            duration = tag_trials.duration.iloc[0]
            num_pulses = tag_trials.num_pulses.iloc[0]
            pulse_interval = tag_trials.pulse_interval.iloc[0]
            this_event_timestamps = tag_trials.time.tolist()

            # Collect event-locked spike times
            int_event_locked_timestamps = []
            for pulse_num in range(num_pulses):
                time_shift = pulse_num * (duration + pulse_interval) / 1000
                this_time_range = time_range_raster + time_shift
                this_locked = af.event_locked_timestamps(
                    unit_spike_times, this_event_timestamps, this_time_range, time_shift=time_shift
                )
                int_event_locked_timestamps.extend(this_locked)
            # Remove spikes during the laser pulse
            int_event_locked_timestamps = remove_spikes_during_laser_pulse(int_event_locked_timestamps, duration)

            # Optional: plot collision raster
            if plot:
                try:
                    plot_collision_raster(
                        int_event_locked_timestamps=int_event_locked_timestamps,
                        time_range_raster=time_range_raster
                    )
                except Exception as e:
                    print(f"Plotting failed for unit {unit_id}, site {site}: {e}")


            # Statistical analysis
            opto_p_val = opto_tagging_response(int_event_locked_timestamps, base_window, roi_window)
            # Latency and jitter
            antidromic_latency, antidromic_jitter = antidromic_latency_jitter(int_event_locked_timestamps)
            first_spikes_after_light = np.array([arr[arr > 0][0] for arr in int_event_locked_timestamps if np.any(arr > 0)])
            median_first_spikes_after_light = np.median(first_spikes_after_light)

            result = {                
                'unit_id': unit_id,
                'site': site,
                'opto_p_val': opto_p_val,
                'int_event_locked_timestamps': int_event_locked_timestamps,
                'median_first_spike_latency': median_first_spikes_after_light,
                'antidromic_latency': antidromic_latency,
                'jitter': antidromic_jitter
            }  

            # Collision test for non-surface_LC sites
            if site != 'surface_LC':
                collision_results = collision_test(
                    int_event_locked_timestamps, antidromic_latency, bin_size=10, antidromic_jitter=0.005, plot=False
                )
                result.update({
                    'collision_pvalue': collision_results['p_value'],
                    'collision_pbinom': collision_results['p_binom'],
                    'pre_boundary_prob': collision_results['pre_boundary_prob'],
                    'post_boundary_prob': collision_results['post_boundary_prob'],
                    'table': collision_results['table'],
                    'oddsratio': collision_results['oddsratio'],
                    'collision_boundary': collision_results['collision_boundary'],
                    'Avg': collision_results['Avg'],
                    'edges': collision_results['edges'],
                    'last_orthodromic_spike_time': collision_results['last_orthodromic_spike_time']
                })

            antidromic_results.append(result)

    antidromic_df = pd.DataFrame(antidromic_results)
    if antidromic_df.empty:
        print("No antidromic results found.")    

    antidromic_pivot = antidromic_df.pivot(index='unit_id', columns='site').reset_index()
    unit_tiers = antidromic_tier_categorization(antidromic_pivot)
    print(unit_tiers)
    unit_tiers_pivot = unit_tiers.pivot(index='unit_id', columns='site_for_tier', values='tier').reset_index()
    merged_df = pd.merge(antidromic_pivot, unit_tiers_pivot, on='unit_id', how='outer')
    return merged_df


def antidromic_tier_categorization(antidromic_pivot):
    if isinstance(antidromic_pivot.columns, pd.MultiIndex):
        antidromic_pivot.columns = ['_'.join([str(i) for i in col if i]) for col in antidromic_pivot.columns.values]

    categories = []
    for idx, row in antidromic_pivot.iterrows():
        unit_id = row['unit_id']
        tier = 0
        site_for_tier = None
        for col in antidromic_pivot.columns:
            if col.startswith('opto_p_val_') and not col.endswith('surface_LC'):
                site = col.replace('opto_p_val_', '')
                opto_p_val = row.get(col, None)
                opto_p_val_col = f'opto_p_val_{site}'
                antidromic_latency_col = f'antidromic_latency_{site}'
                antidromic_latency = row.get(antidromic_latency_col, None)
                jitter_col = f'jitter_{site}'
                jitter = row.get(jitter_col, None)
                collision_pvalue = None
                collision_pbinom = None
                if pd.notnull(opto_p_val) and opto_p_val < 0.05:
                    tier = 3                    
                if pd.notnull(jitter) and jitter < 0.007:
                    tier = 2
                    collision_pvalue_col = f'collision_pvalue_{site}'
                    collision_pbinom_col = f'collision_pbinom_{site}'
                    collision_pvalue = row.get(collision_pvalue_col, None)
                    collision_pbinom = row.get(collision_pbinom_col, None)
                    print(f"Unit {unit_id} is categorized as tier {tier} for site {site} with antidromic latency: {antidromic_latency} and jitter: {jitter} and collision_pvalue: {collision_pvalue} and collision_pbinom: {collision_pbinom}")
                    if pd.notnull(collision_pvalue) and collision_pvalue < 0.05 and pd.notnull(collision_pbinom) and collision_pbinom > 0.05:
                    # if pd.notnull(collision_pbinom) and collision_pbinom > 0.05:
                        tier = 1                    
                        print(f"Unit {unit_id} is categorized as tier {tier} for site {site} with antidromic latency: {antidromic_latency} and jitter: {jitter} and collision_pvalue: {collision_pvalue} and collision_pbinom: {collision_pbinom}")
                site_for_tier = site
                categories.append({'unit_id': unit_id, 'tier': tier, 'site_for_tier': f'{site_for_tier}_antidromic_tier'})

                # print(f"Unit {unit_id} is tier {tier} for site {site}, latency: {antidromic_latency}, jitter: {jitter}, collision_pvalue: {collision_pvalue} and collision_pbinom: {collision_pbinom}")
    return pd.DataFrame(categories)
    
def collision_test(int_event_locked_timestamps, antidromic_latency, bin_size=10, antidromic_jitter=0.005, plot=True):
    """
    Perform collision test for antidromic and orthodromic spikes.

    Parameters:
        int_event_locked_timestamps (list of np.ndarray): Event-locked spike times per trial.
        antidromic_latency (float): Median first spike latency after light onset.
        bin_size (float): Bin size in ms for averaging.
        antidromic_jitter (float): Jitter window in seconds.
        plot (bool): Whether to plot the results.

    Returns:
        dict: Results including DataFrame, p-value, and odds ratio.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import fisher_exact
    from scipy.stats import binomtest as binom_test
    import matplotlib.pyplot as plt

    data = []
    trial_num = 0
    collision = 1  # Default to collision

    for a, spike_times in enumerate(int_event_locked_timestamps):
        if spike_times.size > 0:
            # Find orthodromic and antidromic spike times
            orthodromic_spike_times = spike_times[spike_times <= antidromic_latency - antidromic_jitter]
            antidromic_spike_times = spike_times[
                (antidromic_latency - antidromic_jitter < spike_times) &
                (spike_times < antidromic_latency + antidromic_jitter)
            ]
            # Collision determination
            if orthodromic_spike_times.size > 0:
                last_orthodromic_spike_time = orthodromic_spike_times[-1]
                collision = 0 if antidromic_spike_times.size > 0 else 1
            else:
                last_orthodromic_spike_time = None
        else:
            orthodromic_spike_times = np.array([])
            antidromic_spike_times = np.array([])
            last_orthodromic_spike_time = None

        data.append({
            'trial_num': trial_num,
            'spike_times': spike_times.tolist(),
            'ortho_spike_times': orthodromic_spike_times.tolist(),
            'last_orthodromic_spike_time': last_orthodromic_spike_time,
            'anti_spike_times': antidromic_spike_times.tolist(),
            'collision': collision
        })
        trial_num += 1

    antidromic_df = pd.DataFrame(data)
    roi_df = antidromic_df.query('last_orthodromic_spike_time.notnull()')
    x = (roi_df['last_orthodromic_spike_time'] - antidromic_latency) * 1000
    y = 1 - roi_df['collision']

    # Bin and average    
    Avg, edges = Avg_y_over_x(x, y, bin_size=bin_size)

    # Statistical test between y in x range (-100, 70) and x range (-30, 0)
    pulse_duration = 0.005
    collision_boundary_unit = (antidromic_latency-pulse_duration) * 1000
    # print('collision boundary', collision_boundary)
    # mask1= (x >= -4*collision_boundary_unit) & (x <= -2*collision_boundary_unit)    
    mask1= (x >= -100) & (x <= -2*collision_boundary_unit)    
    mask2= (x >= -2*collision_boundary_unit) & (x <= 0)    
    y1 = y[mask1]
    y2 = y[mask2]
    table = [
        [np.sum(y1 == 1), np.sum(y1 == 0)],
        [np.sum(y2 == 1), np.sum(y2 == 0)]
    ]
    # print(f"Table for Fisher's exact test:\n{table}")
    oddsratio, p_value = fisher_exact(table)

    # Test if y in x range (-30, 0) is different from 0
    mask =  (x >= -1*collision_boundary_unit) & (x <= 0)
    y_in_range = y[mask]
    successes = np.sum(y_in_range == 1)
    n = y_in_range.size
    if n > 0:
        p_binom = binom_test(successes, n, p=0.0, alternative='greater').pvalue       
        # print(f"Binomial test p-value (y > 0 in -30 < x < 0 ms): {p_binom:.4g}")
    else:
        p_binom = np.nan
    pre_prob = np.mean(y[mask1])
    post_prob = np.mean(y[mask2])

    if plot:
        plt.plot(x, y, 'o', color=[0.8, 0.8, 0.8])
        plt.errorbar(Avg['x'], Avg['y'], yerr=Avg['sem'], fmt='o-', label='Average y with SEM')
        plt.axvline(antidromic_latency-2*collision_boundary_unit, color='red', linestyle='--', label='Collision boundary')
        plt.xlabel('Time from antidromic latency (ms)')
        plt.ylabel('P(antidromic spike|orthodromic spike)')        
        plt.xlim([-100, 0])
        plt.ylim([-0.05, 1.05])
        plt.legend()
        # Show pre- and post-boundary probabilities on the plot
        
        plt.text(0.02, 0.92, f'Pre-boundary prob: {pre_prob:.2f}', transform=plt.gca().transAxes, fontsize=10, color='blue')
        plt.text(0.02, 0.85, f'Post-boundary prob: {post_prob:.2f}', transform=plt.gca().transAxes, fontsize=10, color='green')
        # plt.text(0.02, 0.78, f'binomial p-value: {p_binom:.4g}', transform=plt.gca().transAxes, fontsize=10, color='purple')
        # plt.show()
        plt.title(f"Collision Test Results\nFisher's p-value: {p_value:.4g}, Binomial p-value: {p_binom:.4g}")
        print(f"Fisher's exact test p-value: {p_value:.4g}")
        if p_value < 0.05:
            print("Collision detected between antidromic and orthodromic spikes")

    return {
        'antidromic_df': antidromic_df,
        'roi_df': roi_df,
        'Avg': Avg,
        'edges': edges,
        'table': table,
        'oddsratio': oddsratio,
        'p_value': p_value,
        'p_binom': p_binom,
        'collision_boundary': antidromic_latency - 2 * collision_boundary_unit,
        'pre_boundary_prob': pre_prob,  # Probability of antidromic spike before collision boundary
        'post_boundary_prob': post_prob,  # Probability of antidromic spike after collision boundary
        'last_orthodromic_spike_time':antidromic_df['last_orthodromic_spike_time'],
    }
