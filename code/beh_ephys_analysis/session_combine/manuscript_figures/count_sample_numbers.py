"""Count reporting sample numbers for every manuscript figure panel.

This script walks the organized figure folders (``Figure4`` ... ``FigureS18``),
matches each panel to its exported CSV(s), and fills the ``sample_numbers`` field
of every panel in ``manuscript_figure_panel_map.json``.

Counting rules
--------------
0. **Example plot** -> ``animals_number = session_number = unit_number = 1``.
   Detected from the audit (``category == 2_example_regeneration`` or a filled
   ``session_if_example``), an "example" keyword in the title, or a session id
   embedded in the file name.

1. **Population scatter** -> non-nan point counts, in two versions:
     - ``n_points_xy``            : rows where x AND y are non-nan
     - ``n_points_xy_colorcode``  : rows where x, y AND colorcode are non-nan
   If a session-id column exists, ``session_number`` / ``animals_number`` are
   counted over the plotted (x&y non-nan) rows (animal parsed like
   ``beh_functions.parseSessionID``). Scatters that also carry a marginal
   histogram additionally report per-axis ``n_x`` / ``n_y``.

2. **Population PSTH** -> ``session_number`` read from the ``n_sessions`` column.

3. **Histogram / distribution** -> ``per_column_sample_number`` = non-nan count
   for each measurement column. Identifier columns (``unit_id``) are dropped and
   reported instead as ``unit_number = number of rows`` (session id + unit id
   together identify a unit, so one row == one unit). A session-id column, if
   present, is pulled out into ``session_number`` / ``animals_number``.

4. **tier / tier_1 columns** are 0/1 (or True/False) filters: their count is the
   SUM (number flagged), not the non-nan count -- applied both as a histogram
   per-column entry and as a scatter colorcode.

5. **Anything else** -> falls back to explicit count columns in the CSV
   (``sessions_included`` / ``animals_included`` / ``n_sessions`` ...). If none
   exist it is left as a null placeholder with an explanatory ``note``.

Per-panel column choices (which column is x / y / colorcode / session, which
file to read for multi-file panels) live in the SCATTER / PSTH_NSESS /
HISTOGRAM / DISTRIB / NULL_NOTE tables below -- edit those if the exports change.

Usage
-----
    python count_sample_numbers.py                      # use default paths
    python count_sample_numbers.py --figures-dir <dir>  # override figures folder
    python count_sample_numbers.py --dry-run            # print, do not write json
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import OrderedDict, defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "manuscript_figure_panel_map.json")
DEFAULT_AUDIT = os.path.join(HERE, "panel_coverage_audit.csv")


# Local OneDrive export used during the 2026-07 review (fallback default).
ONEDRIVE_FIGURES = (r"C:\Users\zhixi\OneDrive - Allen Institute\LCpaper\submission"
                    r"\review_07052026\figures\0720\manuscript\figures")


def _has_figure_folders(path):
    return bool(path) and os.path.isdir(path) and any(
        os.path.isdir(os.path.join(path, d)) for d in ("Figure4", "Figure5", "Figure6"))


def default_figures_dir():
    """Best-effort default for the organized figures folder.

    Prefers ``capsule_directories()['manuscript_fig_dir']`` when it actually
    contains the figure folders, otherwise falls back to the local OneDrive
    export. Pass ``--figures-dir`` to override.
    """
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "utils")))
        from capsule_migration import capsule_directories  # type: ignore
        cap = str(capsule_directories()["manuscript_fig_dir"])
        if _has_figure_folders(cap):
            return cap
    except Exception:
        pass
    return ONEDRIVE_FIGURES


# --------------------------------------------------------------------------- #
# Panel -> file matching
# --------------------------------------------------------------------------- #
# Positional sub-panel suffixes that are part of a panel label (longest first).
SUFFIX = ['bottom_left', 'bottom_right', 'top_left', 'top_right',
          'bottomleft', 'bottomright', 'topleft', 'topright',
          'bottom', 'top', 'left', 'right', 'mid']


def panel_label(fname):
    """Extract the panel label (e.g. ``FigureS12c_bottom``) from a file name."""
    m = re.match(r'(Figure(?:S)?\d+[a-z]?)(.*)', fname)
    if not m:
        return None
    label, rest = m.group(1), m.group(2)
    changed = True
    while changed:
        changed = False
        for s in SUFFIX:
            if rest.startswith('_' + s + '_') or rest == '_' + s or rest.startswith('_' + s + '.'):
                label += '_' + s
                rest = rest[len('_' + s):]
                changed = True
                break
    return label


# --------------------------------------------------------------------------- #
# Session / animal parsing (mirrors beh_functions.parseSessionID intent)
# --------------------------------------------------------------------------- #
def animal_of(session_str):
    """Return the animal id from a session id string, or None."""
    toks = re.split('[_.]', str(session_str))
    if not toks:
        return None
    if toks[0] in ('behavior', 'ecephys') and len(toks) > 1:
        return toks[1]
    if len(toks[0]) == 6 and toks[0].isdigit():
        return toks[0]
    if re.match(r'^[A-Za-z]{2}\d+$', toks[0]):  # ZS-style, e.g. ZS061
        return toks[0]
    return None


def sess_animal_counts(series):
    vals = series.dropna().astype(str)
    sessions = set(vals)
    animals = set(a for a in (animal_of(v) for v in vals) if a)
    return len(sessions), (len(animals) if animals else None)


def is_session_series(s):
    """True if >=80% of values look like session ids (animal + date)."""
    vals = s.dropna().astype(str)
    if len(vals) == 0:
        return False
    hits = sum(1 for v in vals if animal_of(v) is not None and re.search(r'\d{4}-\d{2}-\d{2}', v))
    return hits >= 0.8 * len(vals)


def is_tier_filter(colname, s):
    """tier_1 / tier1 columns are 0/1 (or True/False) filters -> count = sum."""
    if 'tier' not in str(colname).lower():
        return False
    vals = set(s.dropna().unique())
    return len(vals) > 0 and vals.issubset({0, 1, 0.0, 1.0, True, False})


def tier_sum(s):
    return int(s.dropna().astype(float).sum())


SESS_COUNT_RE = re.compile(r'(sessions?_includ|n_sessions?|num_sessions?)', re.I)
ANIM_COUNT_RE = re.compile(r'(animals?_includ|n_animals?|num_animals?)', re.I)


# --------------------------------------------------------------------------- #
# Per-panel configuration
# --------------------------------------------------------------------------- #
# Scatter panels: (file_substring, x, y, colorcode, session_col)
SCATTER = {
    "Figure4/g": ("response_fr_vs_bl_mean.csv", "bl_mean", "response_fr", None, None),
    "Figure4/h": (".csv", "bl_response_corr_short", "response_rate_mean", None, None),
    "Figure5/b": (".csv", "respond_baseline", "ignore_baseline", None, "session"),
    "Figure5/c": (".csv", "x_ccf_mm", "y_ccf_mm", "T_baseline_hit_all", None),
    "Figure5/d": (".csv", "proj_gene_axis", "T_baseline_hit_all", None, None),
    "Figure5/f": (".csv", "switch_response", "stay_response", None, "session"),
    "Figure5/g": (".csv", "x_ccf_mm", "y_ccf_mm", "T_response_svs_hit", None),
    "Figure5/h": (".csv", "proj_gene_axis", "T_response_svs_hit", None, None),
    "Figure6/a": (".csv", "x_ccf_mm", "y_ccf_mm", "T_outcome_com_mc", None),
    "Figure6/e": ("_scatter.csv", "T_outcome", "T_qchosen", "project_to_PL", None),
    "Figure6/j": ("_scatter.csv", "T_outcome", "T_q", None, None),
    "FigureS12/h": (".csv", "rl_ratio_in_reward_choices", "rl_ratio_out", "ratio_diff_rwd", None),
    "FigureS12/i": (".csv", "ratio_diff_rwd", "in_out_ratio", None, None),
    "FigureS13/e": (".csv", "p_max", "sig_counts", "opto_tagged", None),
    "FigureS13/k": ("_response.csv", "t_antidromic_log", "t_collision_log", "tier_1", None),
    "FigureS13/m": (".csv", "x_ccf_mm", "y_ccf_mm", "tier_1", None),
    "FigureS14/b": ("_scatter.csv", "x_ccf_mm", "y_ccf_mm", "colorcode", None),
    "FigureS14/c": ("_scatter.csv", "x_ccf_mm", "y_ccf_mm", "colorcode", None),
    "FigureS14/d": ("_scatter.csv", "x_ccf_mm", "y_ccf_mm", "colorcode_pc1_rank", None),
    "FigureS14/f": (".csv", "symmetry_slope_div_log", "post_w", "colorcode_pc1_rank", None),
    "FigureS14/g": (".csv", "PC1", "y_loc_ranked", None, None),
    "FigureS14/i": (".csv", "y_loc_ranked", "amp", None, None),
    "FigureS15/f_top": (".csv", "bl_response_corr_short", "response_diff", "colorcode_bl_response_corr_short", None),
    "FigureS15/f_bottom": (".csv", "bl_response_corr_short", "T_baseline_hit_all", "colorcode_bl_response_corr_short", None),
    "FigureS15/q": (".csv", "peak_val_in", "peak_val_out", None, None),
    "FigureS15/t": ("_beh_spatial_axis", "ranked_proj_T_outcome_com_mc", "value_PC_1", None, None),
    "FigureS15/t_topleft": ("_wf_spatial_axis", "ranked_proj_wf_pc_1", "value_PC_1", None, None),
    "FigureS17/b": (".csv", "ml", "dv", None, None),
    "FigureS17/g": ("_mean.csv", "response_G_mean", "response_Iso_mean", "go_cue_G_response", "session"),
    "FigureS18/d": (".csv", "proj_gene_axis", "T_outcome_com_mc", None, None),
}

# Scatter panels that also have a marginal histogram: also report per-axis n_x / n_y.
SCATTER_AXIS_COUNTS = {"FigureS13/e", "FigureS13/k", "FigureS14/b", "FigureS14/c", "FigureS15/q"}

# Population PSTH panels: value = file substring; session count read from n_sessions column.
PSTH_NSESS = {
    "Figure5/j": "_stay_1_switch.csv",
    "Figure6/i": "_numbins6.csv",
    "Figure6/l": ".csv",
    "FigureS17/h_right": "_hit=1_miss=0.csv",
}

# Histogram / distribution panels: value = file substring to pick the histogram csv.
HISTOGRAM = {
    "Figure5/k_left": ".csv",
    "Figure5/k_right": ".csv",
    "Figure6/f": "_polar_histogram.csv",
    "Figure6/k": "_polar_histogram.csv",
    "FigureS12/a": ".csv",
    "FigureS12/j": ".csv",
    "FigureS12/l": ".csv",
    "FigureS13/f_topleft": ".csv",
    "FigureS13/f_topright": ".csv",
    "FigureS13/f_bottomleft": ".csv",
    "FigureS13/f_bottomright": ".csv",
    "FigureS13/k": "_response.csv",
    "FigureS13/l": "_jitter.csv",
    "FigureS15/e": ".csv",
    "FigureS16/d": ".csv",
    "FigureS16/f": ".csv",
    "FigureS16/h": ".csv",
    "FigureS16/j": ".csv",
    "FigureS16/l": ".csv",
    "FigureS16/n": ".csv",
    "FigureS17/i_left": ".csv",
    "FigureS17/i_right": ".csv",
}

# Distribution / heatmap panels counted by row/column shape: ("kind", file_substr, params).
DISTRIB = {
    "FigureS15/a": ("rows_unit", ".csv", None),   # heatmap: rows == units
    "FigureS15/b": ("rows_unit", ".csv", None),
    "FigureS15/c": ("rows_unit", ".csv", None),
    "FigureS12/d": ("session_rows", "_scatter.csv", None),  # per-session scatter
}

# Panels with no directly parseable sample count -> null placeholder + note.
NULL_NOTE = {
    "Figure4/c": "errorbar of choice-history GLM coefficients pooled across sessions; per-sample n not in csv",
    "Figure5/a": "errorbar of GLM coefficients pooled across sessions; per-sample n not in csv",
    "Figure5/e": "errorbar of GLM coefficients pooled across sessions; per-sample n not in csv",
    "Figure6/c": "errorbar over RPE bins; per-sample n not in csv",
    "Figure6/g": "errorbar tuning curve over RPE bins; per-sample n not in csv",
    "Figure6/h": "errorbar tuning curve over bins; per-sample n not in csv",
    "FigureS12/m": "errorbar over RPE bins; per-sample n not in csv",
    "FigureS14/e": "filled psth of waveforms binned by PC1 (per-bin unit n encoded in column names)",
    "FigureS14/h": "filled psth of tetrode waveforms binned by PC1",
    "FigureS18/a": "three primary-axis unit vectors (not a sample count)",
    "FigureS18/b": "bootstrap distribution (2000 iterations), not biological n",
    "FigureS18/c": "bootstrap distribution (10000 iterations), not biological n",
}


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
class PanelCounter:
    def __init__(self, figures_dir, audit):
        self.base = figures_dir
        self.audit = audit

    def files_for(self, fig, key):
        out = defaultdict(list)
        for c in glob.glob(os.path.join(self.base, fig, '*.csv')):
            out[panel_label(os.path.basename(c))].append(os.path.basename(c))
        return sorted(out.get(fig + key, []))

    def load(self, fig, fname):
        return pd.read_csv(os.path.join(self.base, fig, fname))

    @staticmethod
    def pick(files, substr):
        for f in files:
            if substr in f:
                return f
        return None

    @staticmethod
    def nn(df, col):
        return int(df[col].notna().sum())

    def explicit_counts(self, fig, files):
        """Read sessions_included / animals_included / n_sessions from a panel's csv(s)."""
        def first_val(df, rx):
            for c in df.columns:
                if rx.search(str(c)):
                    v = df[c].dropna()
                    if len(v):
                        return int(round(float(v.iloc[0])))
            return None
        for fn in files:
            try:
                df = self.load(fig, fn)
            except Exception:
                continue
            s = first_val(df, SESS_COUNT_RE)
            a = first_val(df, ANIM_COUNT_RE)
            if s is not None or a is not None:
                return s, a
        return None, None

    def is_example(self, fig, key, title, files):
        a = self.audit.get(fig + key, {})
        if (a.get('category') or '').strip() == '2_example_regeneration':
            return True
        if (a.get('session_if_example') or '').strip():
            return True
        if 'example' in title.lower():
            return True
        for f in files:
            if re.search(r'(behavior|ecephys)_[A-Za-z0-9]+_\d{4}-\d{2}-\d{2}', f):
                return True
            if re.search(r'_ZS\d+_\d{4}-\d{2}-\d{2}', f):
                return True
        return False

    def _scatter(self, fig, pk, files):
        sub, x, y, color, scol = SCATTER[pk]
        fn = self.pick(files, sub) or files[0]
        df = self.load(fig, fn)
        sn = OrderedDict()
        if pk in SCATTER_AXIS_COUNTS:
            sn["n_x"] = int(df[x].notna().sum())
            sn["n_y"] = int(df[y].notna().sum())
        valid_xy = df[x].notna() & df[y].notna()
        sn["n_points_xy"] = int(valid_xy.sum())
        if color and color in df.columns:
            if is_tier_filter(color, df[color]):
                sn["n_points_xy_colorcode"] = tier_sum(df[valid_xy][color])
            else:
                sn["n_points_xy_colorcode"] = int((valid_xy & df[color].notna()).sum())
        else:
            sn["n_points_xy_colorcode"] = None
        if scol and scol in df.columns:
            ns, na = sess_animal_counts(df[valid_xy][scol])
            sn["session_number"], sn["animals_number"] = ns, na
        else:
            sn["session_number"] = sn["animals_number"] = None
        return sn

    def _histogram(self, fig, pk, files):
        fn = self.pick(files, HISTOGRAM[pk]) or files[0]
        df = self.load(fig, fn)
        per_col, sess_col, has_unit_id = OrderedDict(), None, False
        for c in df.columns:
            cl = str(c).lower()
            if str(c).startswith("Unnamed"):
                continue
            # session id + unit id together identify a unit -> row == unit
            if cl in ("unit_id", "unit", "unit_ind", "unit_index"):
                has_unit_id = True
                continue
            if is_session_series(df[c]):
                sess_col = c
                continue
            per_col[str(c)] = tier_sum(df[c]) if is_tier_filter(c, df[c]) else self.nn(df, c)
        sn = OrderedDict([("per_column_sample_number", per_col)])
        if has_unit_id:
            sn["unit_number"] = int(len(df))
        if sess_col is not None:
            ns, na = sess_animal_counts(df[sess_col])
            sn["session_number"], sn["animals_number"] = ns, na
        else:
            sn["session_number"] = sn["animals_number"] = None
        return sn

    def _distrib(self, fig, pk, files):
        kind, sub, param = DISTRIB[pk]
        df = self.load(fig, self.pick(files, sub) or files[0])
        if kind == "rows_unit":
            return OrderedDict([("unit_number", int(len(df))), ("session_number", None), ("animals_number", None)])
        if kind == "session_rows":
            return OrderedDict([("session_number", int(len(df))), ("animals_number", None)])
        if kind == "unit":
            return OrderedDict([("unit_number", self.nn(df, param)), ("session_number", None), ("animals_number", None)])
        if kind == "two_group":
            c1, c2 = param
            return OrderedDict([("unit_number", self.nn(df, c1) + self.nn(df, c2)),
                                ("session_number", None), ("animals_number", None)])
        raise ValueError(kind)

    def sample_numbers(self, fig, key, title):
        pk = f"{fig}/{key}"
        files = self.files_for(fig, key)
        note = None

        if not files:
            sn = OrderedDict([("session_number", None), ("animals_number", None)])
            note = "cartoon / no csv"
        elif pk in SCATTER:
            sn = self._scatter(fig, pk, files)
        elif self.is_example(fig, key, title, files):
            sn = OrderedDict([("animals_number", 1), ("session_number", 1), ("unit_number", 1)])
        elif pk in PSTH_NSESS:
            df = self.load(fig, self.pick(files, PSTH_NSESS[pk]) or files[0])
            ns = int(df['n_sessions'].dropna().mode().iloc[0])
            sn = OrderedDict([("session_number", ns), ("animals_number", None)])
        elif pk in HISTOGRAM:
            sn = self._histogram(fig, pk, files)
        elif pk in DISTRIB:
            sn = self._distrib(fig, pk, files)
        else:
            sc, ac = self.explicit_counts(fig, files)  # sessions_included / animals_included / ...
            if sc is not None or ac is not None:
                sn = OrderedDict([("session_number", sc), ("animals_number", ac)])
            else:
                sn = OrderedDict([("session_number", None), ("animals_number", None)])
                if pk in NULL_NOTE:
                    note = NULL_NOTE[pk]

        # special case: population spike-pupil cross-corr, one column per unit
        if pk == "FigureS16/g_right":
            df = self.load(fig, files[0])
            n_units = sum(1 for c in df.columns if re.match(r'unit_\d+$', c))
            sn = OrderedDict([("unit_number", n_units), ("session_number", None), ("animals_number", None)])

        if note:
            sn["note"] = note
        return sn


def load_audit(audit_path):
    audit = {}
    with open(audit_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            p = r['panel'].strip()
            if p:
                audit[p] = r
    return audit


def main():
    ap = argparse.ArgumentParser(description="Fill sample_numbers for manuscript figure panels.")
    ap.add_argument("--figures-dir", default=default_figures_dir(),
                    help="Organized figure folders (Figure4 ... FigureS18).")
    ap.add_argument("--json", default=DEFAULT_JSON, help="Panel map json to update in place.")
    ap.add_argument("--audit", default=DEFAULT_AUDIT, help="panel_coverage_audit.csv.")
    ap.add_argument("--dry-run", action="store_true", help="Print results without writing json.")
    args = ap.parse_args()

    with open(args.json, encoding='utf-8') as f:
        jmap = json.load(f, object_pairs_hook=OrderedDict)

    counter = PanelCounter(args.figures_dir, load_audit(args.audit))

    for fig, panels in jmap.items():
        for key, meta in panels.items():
            meta['sample_numbers'] = counter.sample_numbers(fig, key, meta['title'])
            print(f"{fig}/{key:14s} {dict(meta['sample_numbers'])}")

    if args.dry_run:
        print("\n[dry-run] json not written.")
    else:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(jmap, f, indent=2, ensure_ascii=False)
        print(f"\nWROTE {args.json}")


if __name__ == "__main__":
    main()
