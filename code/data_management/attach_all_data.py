import subprocess
import sys
from pathlib import Path

# Get current directory (where this script lives)
base_dir = Path(__file__).parent

scripts = [
    "mount_data_multi_hopkins.py",
    "mount_data_multi.py",
]

for script in scripts:
    script_path = base_dir / script
    print(f"\n=== Running {script} ===")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,   # set True if you want to capture output
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Error in {script}, stopping execution.")
        break