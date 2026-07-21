"""
Combine per-figure CSVs into one xlsx file per figure.

For each Figure* subfolder under capsule_dirs["manuscript_fig_dir"] every CSV
becomes a sheet in a single xlsx. Supplementary figures (FigureS*) are renamed
to FigureED* in the output filename.
"""

import argparse
import os
import re
import sys

import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, '..', 'beh_ephys_analysis'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.capsule_migration import capsule_directories

INVALID_SHEET_CHARS = re.compile(r'[\[\]\*\?/\\:]')


def to_output_name(fig_name):
    return re.sub(r'^FigureS', 'FigureED', fig_name)


def make_sheet_name(base, used):
    name = INVALID_SHEET_CHARS.sub('_', base)[:31] or 'sheet'
    if name not in used:
        used.add(name)
        return name
    stem = name
    for i in range(2, 1000):
        suffix = f'_{i}'
        trimmed = stem[: 31 - len(suffix)] + suffix
        if trimmed not in used:
            used.add(trimmed)
            return trimmed
    raise RuntimeError(f'Could not disambiguate sheet name for {base}')


def build_xlsx_for_figure(fig_dir, out_path):
    csv_files = sorted(
        os.path.join(fig_dir, f)
        for f in os.listdir(fig_dir)
        if f.lower().endswith('.csv') and os.path.isfile(os.path.join(fig_dir, f))
    )
    if not csv_files:
        return 0

    used = set()
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for csv_path in csv_files:
            stem = os.path.splitext(os.path.basename(csv_path))[0]
            sheet = make_sheet_name(stem, used)
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            df.to_excel(writer, sheet_name=sheet, index=False)
    return len(csv_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--figure', '-f', action='append', default=None,
        help='Figure name to build (e.g. Figure4, FigureS12). '
             'Repeat to pass multiple. If omitted, builds all figures.'
    )
    args = parser.parse_args()

    capsule_dirs = capsule_directories()
    fig_dir = capsule_dirs['manuscript_fig_dir']

    fig_pattern = re.compile(r'^Figure(?:S)?\d+$')
    all_figure_dirs = sorted(
        d for d in os.listdir(fig_dir)
        if os.path.isdir(os.path.join(fig_dir, d)) and fig_pattern.match(d)
    )

    if args.figure:
        requested = set(args.figure)
        missing = requested - set(all_figure_dirs)
        if missing:
            sys.exit(f'Figure folder(s) not found: {sorted(missing)}\n'
                     f'Available: {all_figure_dirs}')
        figure_dirs = [d for d in all_figure_dirs if d in requested]
    else:
        figure_dirs = all_figure_dirs

    for fig_name in figure_dirs:
        src = os.path.join(fig_dir, fig_name)
        out_name = to_output_name(fig_name) + '.xlsx'
        out_path = os.path.join(src, out_name)
        n = build_xlsx_for_figure(src, out_path)
        print(f'{fig_name} -> {out_path}: {n} csv(s)')


if __name__ == '__main__':
    main()
