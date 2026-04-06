from pathlib import Path

def capsule_directories():
    return {
        'preparation_dir': Path('/root/capsule/scratch/combined'),
        'manuscript_fig_dir': Path('/root/capsule/scratch/manuscript'),
        'scratch_dir': Path('/root/capsule/scratch'),
        'data_dir': Path('/root/capsule/data'),
    }