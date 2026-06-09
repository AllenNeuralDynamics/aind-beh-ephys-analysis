# Session-Level Opto-Ephys Processing

This folder contains scripts and notebooks for processing individual behavioral and electrophysiology recording sessions with optogenetic tagging.

## Purpose

This pipeline processes raw and curated session data through multiple stages:
1. **Alignment and preprocessing** - Synchronize behavior, sound, and ephys streams
2. **Opto-tagging analysis** - Identify optogenetically tagged units
3. **Antidromic analysis** - Analyze antidromic responses and compute statistics
4. **Quality control** - Check waveform quality, drift, and alignment artifacts
5. **Optional behavior/pupil analysis** - Link neural activity to behavioral events and pupil dynamics

## Getting Started

See [beh_ephys_processing.ipynb](beh_ephys_processing.ipynb) for:
- Complete workflow documentation
- Step-by-step processing instructions
- Configuration options for raw vs. curated data
- Batch processing examples

## Key Scripts

**behavior_and_time_alignment.py**  
Before curation - Check behavior, sound card and ephys alignment, creates a .json file about alignment metric. Only needs to be ran once at the beginning. Update .json to include beginning or end of session.

**session_preprocessing.py**  
Before and after curation (raw and curated version) - Create opto_tagging df, given alignment info from qm.json; create spiketimes, laser responses based on alignment; recompute crosscorr during session and laser

**opto_waveforms_preprocessing.py**  
Before and after curation - Create sorting analyzer based on stimulation conditions, compute waveforms in each condition and compare similarity to spont waveforms

**opto_tagging.py**  
Plot and decide if units are opto tagged on session level or single unit level.

**ephys_behavior_curation.py**  
Plot units that are potentially opto tagged with behavior and set cut off time

