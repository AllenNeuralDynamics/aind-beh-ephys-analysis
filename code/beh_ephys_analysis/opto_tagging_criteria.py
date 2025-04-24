import json
import os
import sys

name = "waveform_good"
constraints = {
    'isi_violations': { "bounds": [0,        0.1 ]},
    'p_max':          { "bounds": [0.5,      1]},
    'lat_max_p':      { "bounds": [0.005,    0.02  ]},
    'eu':             { "bounds": [0,        0.25  ]},
    'corr':           { "bounds": [0.95,     1     ]},
    'qc_pass':        { "items" : [True]        },
    'probe':         { "items" : ['2.0'] },
}

response = 'Y'

if os.path.exists(f'/root/capsule/scratch/combined/metrics/{name}.json'):
    print(f"Warning: {name}.json already exists, delete it before overwriting.")
    print("Are you sure you want to overwrite it? (Y/N)")
    response = input().strip().upper()
if response == 'Y':
    with open(f'/root/capsule/scratch/combined/metrics/{name}.json', 'w') as f:
        # Note: numpy types (like np.inf) aren’t JSON serializable by default,
        # so convert them to Python floats first:Y
        json_safe = {}
        for k, v in constraints.items():
            if 'bounds' in v:
                lb, ub = v['bounds']
                json_safe[k] = {
                    'bounds': [
                        float(lb) if lb is not None else None,
                        float(ub) if ub is not None else None,
                    ]
                }
            else:
                json_safe[k] = v

        json.dump(json_safe, f, indent=2)
    print(f"{name}.json has been created.")
else:
    print("Operation cancelled. No file was overwritten.")
    sys.exit(0)