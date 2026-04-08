from pathlib import Path

def capsule_directories():
    output_dir = Path('/root/capsule/results/')
    dirs = {
        'output_dir': output_dir,
        'manuscript_fig_dir': output_dir / 'manuscript' / 'figures',
        'manuscript_fig_prep_dir': output_dir / 'manuscript' / 'prep',
        'derived_dir': Path('/root/capsule/scratch'),
        'data_dir': Path('/root/capsule/data'),
    }
    for dir in dirs.values():
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)
    return dirs