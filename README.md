# aind-beh-physiology-analysis

### Table of Contents
- aind-beh-physiology-analysis
    - Table of Contents
    - Overview
- Code
    - Data Preparation Pipeline
    - Analysis Notebooks
        - Behavior
        - Behavior and Electrophysiology
        - Electrophysiology
        - Waveform and Spatial Organization
        - Behavior and Photometry
- Instructions for running locally



### Overview

This capsule contains analysis code for a study of physiology of LC NE neurons and behavior in a dynamic foraging task, focusing on the distribution of neuron properties across space. 

- **Manuscript**: https://www.biorxiv.org/content/10.64898/2026.04.10.717727v1
- **Github Repository**: https://github.com/AllenNeuralDynamics/aind-beh-ephys-analysis
- **Code Ocean Capsule**: NEED TO ADD

The code is organized into two sections, a data preparation pipeline, and then dedicated notebooks for each analysis. The data preparation pipeline aggregates and preprocesses data and generates combined tables and metrics. The analysis notebooks in `code/beh_ephys_analysis/session_combine/manuscript_figures/` use those aggregated results to produce the figures in the manuscript. 

# Code

## Data Preparation Pipeline

Before running any analysis notebook, the **figure preparation scripts** in [`code/beh_ephys_analysis/session_combine/figure_preparation`](code/beh_ephys_analysis/session_combine/figure_preparation) must be executed first. These scripts aggregate and preprocess data across all sessions and animals, generating combined tables and derived metrics that the analysis notebooks depend on. 

> [!CAUTION] 
> The scripts must be run **in the exact order** specified in the [`sequence`](code/beh_ephys_analysis/session_combine/figure_preparation/sequence) file, as later steps depend on outputs from earlier ones. These scripts must be run before any analysis notebook. 


**Estimated run time for each script:**
1. [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py) - 10 min
2. [`antidromic_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/antidromic_generation.py) - < 1 min
3. [`waveform_generation_np.py`](code/beh_ephys_analysis/session_combine/figure_preparation/waveform_generation_np.py) - < 1 min
4. [`waveform_generation_tt.py`](code/beh_ephys_analysis/session_combine/figure_preparation/waveform_generation_tt.py) - < 1 min
5. [`basic_ephys_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/basic_ephys_generation.py) - 8 min
6. [`behavior_metrics_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/behavior_metrics_generation.py) - 4 min
7. [`acg_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/acg_generation.py) - 2 min
8. [`response_tstats_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/response_tstats_generation.py) - 3 min
9. [`outcome_window_generation_parallel.py`](code/beh_ephys_analysis/session_combine/figure_preparation/outcome_window_generation_parallel.py) - 4 min
10. [`beh_combined_outcome_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/beh_combined_outcome_generation.py) - 12 min
11. [`photometry_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/photometry_generation.py) - 10 min

**Total estimated time:** ~55 min



## Behavior Analysis

**Notebook:** [`F_behavior.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_behavior.ipynb)

Characterizes animal behavior across all recording sessions. Fits a multi-lag logistic regression GLM (up to 10 reward-history lags) predicting choice switching using. Summarizes session-level metrics including reward rate, hit rate, model prediction accuracy, and post-outcome lick counts.

**Prerequisites:**
- `combined_beh_sessions.pkl` (from [`behavior_metrics_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/behavior_metrics_generation.py))

**Run time:** ~5 min

**Manuscript figure panels**
- Panel(s): Fig4 b-c, Fig5 a,e, Fig6 b-c, FigA14 a-e, k-m

**Notebook:** [`F_hit_miss.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_hit_miss.ipynb)

Analyzes behavioral and neural factors underlying hit vs. miss responses to the go cue. Fits a logistic regression GLM predicting upcoming hit/miss from reward history across multiple lags, both per-session and per-animal.

**Prerequisites:**
- `combined_beh_sessions.pkl` (from [`behavior_metrics_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/behavior_metrics_generation.py))

**Run time:** ~4 min

**Manuscript figure panels**
- Panel(s): Fig5

## Behavior and Electrophysiology Analysis

**Notebook:** [`F_ephys_behavior_action&outcome.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_ephys_behavior_action&outcome.ipynb)

The primary neural coding notebook. Analyzes how single neurons encode action (stay vs. switch) and outcome (reward vs. no reward) across the population. Computes population PSTHs split by behavioral condition, extracts per-unit T-statistics for outcome and action coding dimensions, and maps neural selectivity in CCF brain atlas space.

**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- `combined_outcome_window_responses.pkl` (from [`outcome_window_generation_parallel.py`](code/beh_ephys_analysis/session_combine/figure_preparation/outcome_window_generation_parallel.py))

**Run time:** ~22 min

**Manuscript figure panels**
- Panel(s): Fig4 d-g, Fig5 b-d, f-h, k, Fig6 a,d-h, FigA15 a-f, FigA15 t, FigA17 i

**Notebook:** [`F_ephys_behavior_examples.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_ephys_behavior_examples.ipynb)

Generates single-unit example raster + PSTH figures for a curated set of representative neurons (13 examples from both silicon probe and tetrode recordings), plotted for stay-vs-switch and hit-vs-miss behavioral splits.

**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))

**Run time:** ~3 min

**Manuscript figure panels**
- Panel(s): Fig6 d

**Notebook:** [`F_auc_psth.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_auc_psth.ipynb)

Quantifies neural discriminability of behavioral variables (reward outcome, hit/miss, stay-vs-switch) over time using a sliding-window ROC-AUC analysis per neuron. Produces population-level AUC heatmaps (sorted by peak discriminability) and histograms.

**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))

**Manuscript figure panels**
- Panel(s): FigA15 a-c


## Electrophysiology Analysis

**Notebook:** [`F_basic_ephys.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_basic_ephys.ipynb)

Comprehensive characterization of electrophysiological unit properties across all recorded neurons. Analyzes baseline and response firing rates, burst properties (ACG fit parameters), waveform features, and opto-tagging quality. Fits OLS models examining how intrinsic properties predict the degree of outcome vs. action coding.

**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- `combined_basic_ephys.pkl` (from [`basic_ephys_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/basic_ephys_generation.py))
- `combined_acg.pkl` (from [`acg_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/acg_generation.py))

**Run time:** ~11 min

**Manuscript figure panels**
- Panel(s): FigA15 d-f

**Notebook:** [`F_cross_correlation.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_cross_correlation.ipynb)

Analyzes spike train temporal structure using auto-correlations and cross-correlations. Computes pairwise cross-correlations between neurons (including across PrL and S1) to assess functional connectivity, and visualizes correlation structure mapped to CCF coordinates.

**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- Per-session spike data

**Run time:** ~7 min

**Manuscript figure panels**
- Panel(s): FigA15 r-t

**Notebook:** [`F_antidromic_combined.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_antidromic_combined.ipynb)

Identifies and characterizes antidromically-activated projection neurons across sessions. Applies a tiered classification system (tier 1: jitter, collision test, and antidromic response criteria; tier 2: looser thresholds) to classify PrL → subcortical projection neurons.

**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- `combined_antidromic_tbl.pkl` (from [`antidromic_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/antidromic_generation.py))

**Run time:** ~1 min

**Manuscript figure panels**
- Panel(s): FigA12


## Waveform and Spatial Organization

**Notebook:** [`F_waveform_space.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_waveform_space.ipynb)

Characterizes action potential waveform morphology across the unit population (silicon probe recordings). Extracts waveform shape features, reduces via PCA, maps onto CCF coordinates with brain mesh overlays, and tests spatial dependence statistics. Opto-tagged units are overlaid to reveal waveform-type identity.

**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- `combined_waveform_NP.pkl` (from [`waveform_generation_np.py`](code/beh_ephys_analysis/session_combine/figure_preparation/waveform_generation_np.py))

**Run time:** ~8 min

**Manuscript figure panels**
- Panel(s): FigA13 a-f

**Notebook:** [`F_waveform_space_tetrode.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_waveform_space_tetrode.ipynb)

Identical waveform analysis applied exclusively to tetrode-recorded units, producing the same spatial feature maps for the tetrode recording subset.

**Prerequisites:**
- `combined_unit_tbl.pkl` (from [`make_combined_unit_tbl.py`](code/beh_ephys_analysis/session_combine/figure_preparation/make_combined_unit_tbl.py))
- `combined_waveform_TT.pkl` (from [`waveform_generation_tt.py`](code/beh_ephys_analysis/session_combine/figure_preparation/waveform_generation_tt.py))

**Run time:** ~1 min

**Manuscript figure panels**
- Panel(s): FigA13 g-i

**Notebook:** [`F_spatial-axis-comparison.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_spatial-axis-comparison.ipynb)

Integrates three datasets to compare cellular organization axes: electrophysiology waveform features, MERFISH spatial transcriptomics (~2,200 cells), and retrograde tracing from 18 brains. Fits a linear spatial axis to each dataset and compares principal spatial gradients across data modalities.

**Prerequisites:**
- `combined_waveform_NP.pkl` (from [`waveform_generation_np.py`](code/beh_ephys_analysis/session_combine/figure_preparation/waveform_generation_np.py))

**Run time:** ~11 min

**Manuscript figure panels**
- Panel(s): Fig3e, FigA18


## Behavior and Photometry Analysis


**Notebook:** [`F_photometry_tuning_psth.ipynb`](code/beh_ephys_analysis/session_combine/manuscript_figures/F_photometry_tuning_psth.ipynb)

Computes tuning curves and PSTHs for fiber photometry signals by binning the signal into 6 prediction error (PE) levels aligned to choice time. Reveals how the PrL photometry signal scales as a function of reward prediction error.

**Prerequisites:**
- Photometry GLM results (from [`photometry_generation.py`](code/beh_ephys_analysis/session_combine/figure_preparation/photometry_generation.py))
- Per-session photometry data

**Run time:** ~5 min

**Manuscript figure panels**
- Panel(s): Fig5i-k, Fig6i-l, FigA17

# Data Organization
## Experiments and Derived Data Used

### Electrophysiology Recordings with and without behavior
- `all 'raw' data`: raw electrophysiology data with and without behavior，include high speed video for bottom and side of the face and whole body. 
- `all 'sorted' data`: Kilosort data, containing single neuron activity, cluster quality and probe drift estimation.
- `all 'sorted_curated' data`: Kilosort data after manual curation. Single neuron activity here is used for analysis.
- `scratch_data`: Animal/session-structured derived data generated by the session processing pipeline in [`beh_ephys_processing.ipynb`](code/beh_ephys_analysis/ani_session_processing/beh_ephys_processing.ipynb). Organized as `{animal_id}/{session_id}/` with the following outputs:
  - **Alignment**: `qm.json` files with behavior/sound/ephys stream alignment metrics and session cut points
  - **Behavior analysis plots**: Choice/reward patterns, GLM fits, lick analysis (rasters, feature space, video-based lick detection), lick trains by side
  - **Ephys processing** (raw and curated): Spike times, waveform analyzers, drift trial tables, and quality control figures
  - **Opto-tagging analysis** (raw and curated): Opto response dataframes (`opto_session.csv`), waveform similarity metrics, correlation matrices (laser-triggered auto/cross-correlograms), antidromic response statistics, and per-unit tagging figures
  - **Behavior + ephys analysis plots**: Neural activity aligned to behavioral events (go-cue, response), unit PSTHs split by choice/outcome, burst analysis, LFP power spectra, waveform quality checks with behavioral alignment
  - **Animal-level summaries**: GLM choice-history models, lick statistics across sessions

### Anatomical Registration and Spatial Mapping
- `alignment_fix`: Single neuron location in CCF space inferred using IBL gui.
- `dorsal_edges`: Dorsal edge of LC, labeled by hand and converted to CCF locations.
- `LC_percentile_meshs`: LC mesh generated by Drew Friedmann.

### Behavioral Video Tracking
- `all_tongue_movements`: Tongue movement data derived from pose tracking of high speed video. Generated by Matt Becker.

### MERFISH Spatial Transcriptomics
- `merfish_data`: Merfish data used to compare spatial distribution of gene expression with other modalities. Generated by Shuonan Chen.

### Retrograde Tracing
- `LC_retro`: Retrograde labeling data with location in CCF space. Generated by Polina Kosillo.

---

# Running This Code Locally

This codebase is designed to run both on Code Ocean and on local machines. The import structure ensures that all functions can be properly imported regardless of where the repository root is located.

### Getting Started

To run this analysis pipeline on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AllenNeuralDynamics/aind-beh-ephys-analysis.git
   cd aind-beh-ephys-analysis
   ```

2. **Set up your Python environment:**
   - Python 3.8+ recommended
   - Install required dependencies (see `environment/` folder for environment files)

3. **Obtain the data:**
   - **If running in Code Ocean reproducible runs:** Run [`/root/capsule/code/data_management/attach_all_data_capsule.py`](code/data_management/attach_all_data_capsule.py)
   - **If running in a VSCode session:** Run [`/root/capsule/code/data_management/attach_all_data.py`](code/data_management/attach_all_data.py)
   - **If running locally (outside Code Ocean):** To be updated

4. **Run the preparation scripts (required before running notebooks):**
   ```bash
   cd code/beh_ephys_analysis/session_combine/figure_preparation
   # Run scripts in order listed in the 'sequence' file
   python make_combined_unit_tbl.py
   python antidromic_generation.py
   # ... continue with remaining scripts
   ```
   See the [preparation scripts section](#generated-files-used-across-the-manuscript-notebooks-and-figure-preparation-dependency) for the complete list and estimated run times (~55 min total).

5. **Run the analysis notebooks:**
   - Open any notebook in [`code/beh_ephys_analysis/session_combine/manuscript_figures/`](code/beh_ephys_analysis/session_combine/manuscript_figures/)
   - The automatic import system will resolve paths correctly
   - Notebooks can be run in any order after preparation scripts complete

**Note:** The code automatically detects the repository root, so you can run notebooks from any working directory within the repository.

### Import System Design

The code uses a robust import resolution system that works across different environments:

1. **Notebooks use automatic root detection**: Each notebook includes an automatic root-finding snippet at the top that walks up the directory tree to locate `code/beh_ephys_analysis/`. This ensures imports work whether you're running from Jupyter, VS Code, or any other environment.

2. **Centralized path management**: The [`code/beh_ephys_analysis/utils/capsule_migration.py`](code/beh_ephys_analysis/utils/capsule_migration.py) module provides the `capsule_root()` function that resolves the repository root in the following order:
   - `$CAPSULE_ROOT` environment variable (if set)
   - `/root/capsule` (Code Ocean default)
   - Repository root derived from the file's location (for local checkouts)

3. **Standard directory structure**: The `capsule_directories()` function in `capsule_migration.py` returns standard paths for outputs, figures, and data directories, creating them if they don't exist.

### Customizing Directory Paths

If you need to change where data is stored or loaded from, modify [`code/beh_ephys_analysis/utils/capsule_migration.py`](code/beh_ephys_analysis/utils/capsule_migration.py):

**To change the repository root location:**
- Set the `CAPSULE_ROOT` environment variable before running notebooks:
  ```bash
  export CAPSULE_ROOT=/path/to/your/repo
  ```
- Or modify the `capsule_root()` function to add your custom path to the resolution order

**To change output/data directories:**
- Edit the `capsule_directories()` function (lines 29-58) to customize these paths:
  - `output_dir`: Main results output directory (default: `<root>/scratch/results`)
  - `manuscript_fig_dir`: Manuscript figures (default: `<root>/scratch/results/manuscript/figures`)
  - `manuscript_fig_prep_dir`: Manuscript prep files (default: `<root>/scratch/results/manuscript/prep`)
  - `derived_dir`: Derived/processed data (default: `<root>/data/scratch_data`)
  - `data_dir`: Raw data directory (default: `<root>/data`)

**Example modification:**
```python
def capsule_directories():
    root = capsule_root()
    output_dir = root / 'my_custom_output'  # Change output location
    dirs = {
        'output_dir': output_dir,
        'manuscript_fig_dir': output_dir / 'figures',
        'manuscript_fig_prep_dir': output_dir / 'prep',
        'derived_dir': root / 'my_data' / 'processed',  # Change data location
        'data_dir': root / 'my_data' / 'raw',
    }
    # ... rest of function
```

All notebooks and scripts that use `capsule_directories()` will automatically pick up these changes.

