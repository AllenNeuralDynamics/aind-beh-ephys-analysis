# aind-beh-phys-analysis

This capsule contains analysis code for a study of physiology of LC NE neurons and behavior in a dynamic foraging task, focusing on the distribution of neuron properties across space.

Analysis is organized into the following topics, each covered by a dedicated notebook in `code/beh_ephys_analysis/session_combine/manuscript_figures/`:

---

## 1. Behavior Analysis

**Notebook:** `F_behavior.ipynb`

Characterizes animal behavior across all recording sessions. Fits a multi-lag logistic regression GLM (up to 10 reward-history lags) predicting choice switching using. Summarizes session-level metrics including reward rate, hit rate, model prediction accuracy, and post-outcome lick counts.

**Manuscript figure panels**
- Panel(s): Fig4 b-c, Fig5 a,e, Fig6 b-c, FigA14 a-e, k-m

**Notebook:** `F_hit_miss.ipynb`

Analyzes behavioral and neural factors underlying hit vs. miss responses to the go cue. Fits a logistic regression GLM predicting upcoming hit/miss from reward history across multiple lags, both per-session and per-animal.

**Manuscript figure panels**
- Panel(s): Fig5
---

## 2. Behavior and Electrophysiology Analysis

**Notebook:** `F_ephys_behavior_action&outcome.ipynb`

The primary neural coding notebook. Analyzes how single neurons encode action (stay vs. switch) and outcome (reward vs. no reward) across the population. Computes population PSTHs split by behavioral condition, extracts per-unit T-statistics for outcome and action coding dimensions, and maps neural selectivity in CCF brain atlas space.

**Manuscript figure panels**
- Panel(s): Fig4 d-g, Fig5 b-d, f-h, k, Fig6 a,d-h, FigA15 a-f, FigA15 t, FigA17 i

**Notebook:** `F_ephys_behavior_examples.ipynb`

Generates single-unit example raster + PSTH figures for a curated set of representative neurons (13 examples from both silicon probe and tetrode recordings), plotted for stay-vs-switch and hit-vs-miss behavioral splits.

**Manuscript figure panels**
- Panel(s): Fig6 d

**Notebook:** `F_auc_psth.ipynb`

Quantifies neural discriminability of behavioral variables (reward outcome, hit/miss, stay-vs-switch) over time using a sliding-window ROC-AUC analysis per neuron. Produces population-level AUC heatmaps (sorted by peak discriminability) and histograms.

**Manuscript figure panels**
- Panel(s): FigA15 a-c
---

## 3. Electrophysiology Analysis

**Notebook:** `F_basic_ephys.ipynb`

Comprehensive characterization of electrophysiological unit properties across all recorded neurons. Analyzes baseline and response firing rates, burst properties (ACG fit parameters), waveform features, and opto-tagging quality. Fits OLS models examining how intrinsic properties predict the degree of outcome vs. action coding.

**Manuscript figure panels**
- Panel(s): FigA15 d-f

**Notebook:** `F_cross_correlation.ipynb`

Analyzes spike train temporal structure using auto-correlations and cross-correlations. Computes pairwise cross-correlations between neurons (including across PrL and S1) to assess functional connectivity, and visualizes correlation structure mapped to CCF coordinates.

**Manuscript figure panels**
- Panel(s): FigA15 r-t

**Notebook:** `F_antidromic_combined.ipynb`

Identifies and characterizes antidromically-activated projection neurons across sessions. Applies a tiered classification system (tier 1: jitter, collision test, and antidromic response criteria; tier 2: looser thresholds) to classify PrL → subcortical projection neurons.

**Manuscript figure panels**
- Panel(s): FigA12

---

## 4. Waveform and Spatial Organization

**Notebook:** `F_waveform_space.ipynb`

Characterizes action potential waveform morphology across the unit population (silicon probe recordings). Extracts waveform shape features, reduces via PCA, maps onto CCF coordinates with brain mesh overlays, and tests spatial dependence statistics. Opto-tagged units are overlaid to reveal waveform-type identity.

**Manuscript figure panels**
- Panel(s): FigA13 a-f

**Notebook:** `F_waveform_space_tetrode.ipynb`

Identical waveform analysis applied exclusively to tetrode-recorded units, producing the same spatial feature maps for the tetrode recording subset.

**Manuscript figure panels**
- Panel(s): FigA13 g-i

**Notebook:** `F_spatial-axis-comparison.ipynb`

Integrates three datasets to compare cellular organization axes: electrophysiology waveform features, MERFISH spatial transcriptomics (~2,200 cells), and retrograde tracing from 18 brains. Fits a linear spatial axis to each dataset and compares principal spatial gradients across data modalities.

**Manuscript figure panels**
- Panel(s): Fig3e, FigA18

---

## 5. Behavior and Photometry Analysis


**Notebook:** `F_photometry_tuning_psth.ipynb`

Computes tuning curves and PSTHs for fiber photometry signals by binning the signal into 6 prediction error (PE) levels aligned to choice time. Reveals how the PrL photometry signal scales as a function of reward prediction error.

**Manuscript figure panels**
- Panel(s): Fig5i-k, Fig6i-l, FigA17

---


### Generated files used across the manuscript notebooks and figure preparation dependency

The notebooks in `code/beh_ephys_analysis/session_combine/manuscript_figures/` call shared and pre-generated input files in data_preparation folder.

These shared inputs are generated from the preparation scripts in `/root/capsule/code/beh_ephys_analysis/session_combine/figure_preparation`, following the order specified in the `sequence` file in that folder.

> **Important:** Before running any notebook in `code/beh_ephys_analysis/session_combine/manuscript_figures/`, make sure the generation files in `figure_preparation` have been run first, in the exact order listed in `sequence`.
