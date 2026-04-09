"""Top-level runner for figure-preparation scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
FIG_PREP_DIR = CODE_DIR / "beh_ephys_analysis" / "session_combine" / "figure_preparation"
SEQUENCE_FILE = FIG_PREP_DIR / "sequence.txt"


def load_sequence(sequence_file: Path = SEQUENCE_FILE) -> list[str]:
    """Load the ordered script list from `sequence.txt`."""
    if not sequence_file.exists():
        raise FileNotFoundError(f"Sequence file not found: {sequence_file}")

    scripts = []
    for line in sequence_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scripts.append(line)

    if not scripts:
        raise ValueError(f"No scripts found in: {sequence_file}")

    return scripts


def run_script(script_name: str, check_only: bool = False) -> None:
    """Run a single script, or just validate that it exists."""
    script_path = FIG_PREP_DIR / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Script listed in sequence does not exist: {script_path}")

    if check_only:
        print(f"[CHECK] {script_path}", flush=True)
        return

    print(f"\n=== Running {script_name} ===", flush=True)
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(WORKSPACE_DIR),
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)


def run(check_only: bool = False) -> int:
    """Run the figure-preparation scripts in the sequence-file order."""
    scripts = load_sequence()
    print(f"Loaded {len(scripts)} scripts from {SEQUENCE_FILE}", flush=True)

    for idx, script_name in enumerate(scripts, start=1):
        print(f"[{idx}/{len(scripts)}] {script_name}", flush=True)
        run_script(script_name, check_only=check_only)

    if check_only:
        print("Sequence validation completed successfully.", flush=True)
    else:
        print("All scripts completed successfully.", flush=True)
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run figure-preparation scripts in the order stored in sequence.txt."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the sequence and file existence without executing the scripts.",
    )
    args = parser.parse_args()

    try:
        return run(check_only=args.check_only)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    except subprocess.CalledProcessError as exc:
        failed_script = exc.cmd[-1] if isinstance(exc.cmd, (list, tuple)) and exc.cmd else "<unknown>"
        print(
            f"ERROR: script failed with exit code {exc.returncode}: {failed_script}",
            file=sys.stderr,
            flush=True,
        )
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())