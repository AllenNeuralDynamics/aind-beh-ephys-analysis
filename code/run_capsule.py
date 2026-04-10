"""Top-level runner for figure-preparation scripts and manuscript notebooks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
DATA_ATTACH_SCRIPT = CODE_DIR / "data_management" / "attach_all_data_capsule.py"
FIG_PREP_DIR = CODE_DIR / "beh_ephys_analysis" / "session_combine" / "figure_preparation"
FIG_PREP_SEQUENCE_FILE = FIG_PREP_DIR / "sequence.txt"
MANUSCRIPT_FIG_DIR = CODE_DIR / "beh_ephys_analysis" / "session_combine" / "manuscript_figures"
FIG_NOTEBOOK_LIST = MANUSCRIPT_FIG_DIR / "fig_notebook_list.txt"


def load_sequence(sequence_file: Path) -> list[str]:
    """Load an ordered list of files from a plain-text sequence file."""
    if not sequence_file.exists():
        raise FileNotFoundError(f"Sequence file not found: {sequence_file}")

    entries = []
    for line in sequence_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)

    if not entries:
        raise ValueError(f"No entries found in: {sequence_file}")

    return entries


def run_script(script_name: str, check_only: bool = False) -> None:
    """Run one figure-preparation Python script, or just validate that it exists."""
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


def run_notebook(notebook_name: str, check_only: bool = False) -> None:
    """Execute one manuscript-figure notebook in place, or just validate it exists."""
    notebook_path = MANUSCRIPT_FIG_DIR / notebook_name
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Notebook listed in figure order does not exist: {notebook_path}")

    if check_only:
        print(f"[CHECK] {notebook_path}", flush=True)
        return

    print(f"\n=== Executing {notebook_name} ===", flush=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=-1",
            str(notebook_path),
        ],
        cwd=str(WORKSPACE_DIR),
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)


def run_data_attachment(check_only: bool = False) -> None:
    """Run data attachment helper before executing the main sequence."""
    if not DATA_ATTACH_SCRIPT.is_file():
        raise FileNotFoundError(f"Data-attachment script not found: {DATA_ATTACH_SCRIPT}")

    if check_only:
        print(f"[CHECK] {DATA_ATTACH_SCRIPT}", flush=True)
        return

    print("\n=== Attaching data ===", flush=True)
    completed = subprocess.run(
        [sys.executable, str(DATA_ATTACH_SCRIPT)],
        cwd=str(WORKSPACE_DIR),
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)


def run(check_only: bool = False) -> int:
    """Run figure-preparation scripts, then manuscript notebooks, in their listed order."""
    scripts = load_sequence(FIG_PREP_SEQUENCE_FILE)
    notebooks = load_sequence(FIG_NOTEBOOK_LIST)
    print(f"Loaded {len(scripts)} scripts from {FIG_PREP_SEQUENCE_FILE}", flush=True)
    print(f"Loaded {len(notebooks)} notebooks from {FIG_NOTEBOOK_LIST}", flush=True)

    run_data_attachment(check_only=check_only)

    for idx, script_name in enumerate(scripts, start=1):
        print(f"[prep {idx}/{len(scripts)}] {script_name}", flush=True)
        run_script(script_name, check_only=check_only)

    for idx, notebook_name in enumerate(notebooks, start=1):
        print(f"[figure {idx}/{len(notebooks)}] {notebook_name}", flush=True)
        run_notebook(notebook_name, check_only=check_only)

    if check_only:
        print("Sequence validation completed successfully for scripts and notebooks.", flush=True)
    else:
        print("All scripts and notebooks completed successfully.", flush=True)
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run figure-preparation scripts and manuscript notebooks in their listed order."
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