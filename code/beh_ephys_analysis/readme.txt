behavior_and_time_alignment.py
Before curation
Check behavior, sound card and ephys alignment, creates a .json file about alignment metric


session_preprocessing.py
Before and after curation (raw and curated version)
Create opto_tagging df, given alignment info from qm.json; create spiketimes, laser responses based on alignment


opto_waveforms_preprocessing.py
Before and after curation 
Create sorting analyzer based on stimulation conditions, compute waveforms in each condition and compare similarity to spont waveforms

