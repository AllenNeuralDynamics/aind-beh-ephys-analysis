behavior_and_time_alignment.py
Before curation
Check behavior, sound card and ephys alignment, creates a .json file about alignment metric. Only needs to be ran once at the beginning. Update .json to include beginning or end of session. 


session_preprocessing.py
Before and after curation (raw and curated version)
Create opto_tagging df, given alignment info from qm.json; create spiketimes, laser responses based on alignment; recompute crosscorr during session and laser


opto_waveforms_preprocessing.py
Before and after curation 
Create sorting analyzer based on stimulation conditions, compute waveforms in each condition and compare similarity to spont waveforms

opto_tagging.py
Plot and decide if units are opto tagged on session level or single unit level.

ephys_behavior_curation:
plot units that are potentialy opto tagged with behavior and set cut off time


