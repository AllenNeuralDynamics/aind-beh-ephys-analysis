"""Utilities for saving manuscript figure panels.

Figure-generating cells in ``session_combine/manuscript_figures`` are tagged with
one or more panel labels of the form ``##FigureS16g_right##`` right after the plot
is generated. ``save_panels`` writes the figure once per panel, prefixing the panel
label to the file name so the saved files map one-to-one onto manuscript panels.
"""

import os

import numpy as np
import pandas as pd


def columns_to_df(**cols):
    """Build a DataFrame from named 1D columns of possibly unequal length.

    Shorter columns are NaN-padded. Use for scatter (x/y/colorcode), histogram
    raw values (one column per series), or shared pre-average data.
    """
    arrs = {k: np.atleast_1d(np.asarray(v, dtype=object)) for k, v in cols.items()}
    n = max((len(v) for v in arrs.values()), default=0)
    for k, v in arrs.items():
        if len(v) < n:
            arrs[k] = np.concatenate([v, np.full(n - len(v), np.nan, dtype=object)])
    return pd.DataFrame(arrs)


def heatmap_to_df(matrix, x, y):
    """2D matrix -> DataFrame with ``y`` as the row index and ``x`` as columns.

    Saving with ``index=True`` preserves the x and y extents as the axis labels.
    """
    return pd.DataFrame(np.asarray(matrix), index=np.asarray(y), columns=np.asarray(x))


def save_panel_csv(df, folder, name, panels, plot_type=None, index=False):
    """Write ``df`` to ``{panel}_{name}[_{plot_type}].csv`` for each panel.

    Mirrors :func:`save_panels` naming so each panel's CSV sits beside its figure.
    ``plot_type`` (rule 7) appends a suffix when a cell produces more than one
    plot type. Pass ``index=True`` for heatmaps (to keep the y-axis labels).
    """
    if isinstance(panels, str):
        panels = [panels]
    os.makedirs(folder, exist_ok=True)
    suffix = f"_{plot_type.replace(' ', '_')}" if plot_type else ""
    paths = []
    for panel in panels:
        path = os.path.join(folder, f"{panel}_{name}{suffix}.csv")
        df.to_csv(path, index=index)
        paths.append(path)
    return paths


def save_panels(fig, folder, name, panels, exts=("pdf", "svg"), **savefig_kwargs):
    """Save ``fig`` once per manuscript panel, prefixing the panel label.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    folder : str
        Output directory (created if missing).
    name : str
        Base file name without extension, e.g. ``f'crosscorrelation_all_units_{criteria_name}'``.
    panels : str or sequence of str
        Panel label(s) such as ``'FigureS16g_right'``. One copy of the figure is
        written per panel (rule 4: multiple labels on one plot -> multiple copies).
    exts : sequence of str
        File extensions to write for each panel.
    **savefig_kwargs
        Passed through to ``fig.savefig`` (e.g. ``dpi=300``, ``bbox_inches='tight'``).

    Returns
    -------
    list of str
        Paths written.
    """
    if isinstance(panels, str):
        panels = [panels]
    os.makedirs(folder, exist_ok=True)
    paths = []
    for panel in panels:
        for ext in exts:
            path = os.path.join(folder, f"{panel}_{name}.{ext}")
            fig.savefig(path, **savefig_kwargs)
            paths.append(path)
    return paths
