"""
Utilities for fitting and comparing spatial organization axes across datasets.

Provides:
  - Linear regression, CCA, and LDA axis fitting from 3D CCF coordinates
  - Bootstrap resampling for confidence intervals on spatial axes
  - Statistical comparison of two 3D directional axes (tangent-plane / Wald test)
  - Visualization helpers: azimuth/elevation scatter, confidence cone arrows,
    directional difference plots, regression CI bands
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unit(v, eps=1e-15):
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def pd_notnull_1d(arr):
    """Pandas-free not-null check for 1-D label arrays (strings, objects, None, NaN)."""
    arr = np.asarray(arr, dtype=object)
    out = np.ones(len(arr), dtype=bool)
    for i, v in enumerate(arr):
        if v is None:
            out[i] = False
        else:
            try:
                if isinstance(v, float) and np.isnan(v):
                    out[i] = False
            except Exception:
                pass
    return out


def _orthonormal_basis_perp(v0, v1):
    """Return (e1, e2, u0) where u0=normalized v0 and e1 points toward v1 in the plane perp to u0."""
    u0 = _unit(v0)
    diff = v1 - np.dot(v1, u0) * u0
    if np.linalg.norm(diff) < 1e-12:
        a = np.array([1.0, 0.0, 0.0]) if abs(u0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        diff = a - np.dot(a, u0) * u0
    e1 = _unit(diff)
    e2 = _unit(np.cross(u0, e1))
    return e1, e2, u0


# ---------------------------------------------------------------------------
# Spatial bin helpers (used by bootstrap_spatial_axis_cca)
# ---------------------------------------------------------------------------

def _make_spatial_bin_ids(coords, spatial_bin_edges=None, spatial_bin_counts=(4, 4, 4)):
    X = np.asarray(coords, float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3).")

    if spatial_bin_edges is None:
        spatial_bin_counts = [int(v) for v in spatial_bin_counts]
        spatial_bin_edges = []
        for d in range(3):
            x = X[:, d]
            mn, mx = np.nanmin(x), np.nanmax(x)
            if not np.isfinite(mn) or not np.isfinite(mx):
                raise ValueError("Non-finite coordinate range.")
            edges = np.array([mn - 0.5, mx + 0.5]) if mx <= mn else np.linspace(mn, mx, spatial_bin_counts[d] + 1)
            if mx > mn:
                edges[-1] = np.nextafter(edges[-1], np.inf)
            spatial_bin_edges.append(edges)
    else:
        spatial_bin_edges = [np.asarray(e, float) for e in spatial_bin_edges]

    bin_idx = []
    valid = np.ones(len(X), dtype=bool)
    for d in range(3):
        idx = np.digitize(X[:, d], spatial_bin_edges[d][1:-1], right=False)
        valid &= (X[:, d] >= spatial_bin_edges[d][0]) & (X[:, d] < spatial_bin_edges[d][-1])
        bin_idx.append(idx)

    n0 = len(spatial_bin_edges[0]) - 1
    n1 = len(spatial_bin_edges[1]) - 1
    multipliers = np.array([1, n0, n0 * n1], dtype=int)
    bin_ids = (np.column_stack(bin_idx) * multipliers[None, :]).sum(axis=1)
    return bin_ids, valid, spatial_bin_edges


def _bootstrap_indices_within_bins(bin_ids, rng):
    """Sample with replacement within each spatial bin, preserving per-bin counts."""
    bin_ids = np.asarray(bin_ids)
    sampled = []
    for b in np.unique(bin_ids):
        idx = np.flatnonzero(bin_ids == b)
        if len(idx):
            sampled.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(sampled) if sampled else np.array([], dtype=int)


# ---------------------------------------------------------------------------
# Linear regression axis
# ---------------------------------------------------------------------------

def fit_spatial_axis_linear(feature, coords, *, add_intercept=True,
                             center_coords=False, center_feature=False,
                             return_model_details=False):
    """Fit feature ~ coords via OLS and return the unit-normalized spatial gradient."""
    y = np.asarray(feature, dtype=float).reshape(-1)
    X = np.asarray(coords, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3).")
    if len(y) != len(X):
        raise ValueError("feature and coords must have the same length.")

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    if len(y) < 4:
        raise ValueError("Need at least 4 valid samples.")

    coords_mean = X.mean(axis=0) if center_coords else np.zeros(3)
    feature_mean = y.mean() if center_feature else 0.0
    X_used = X - coords_mean if center_coords else X.copy()
    y_used = y - feature_mean if center_feature else y.copy()

    if add_intercept:
        X_design = np.column_stack([np.ones(len(X_used)), X_used])
        coef = np.linalg.lstsq(X_design, y_used, rcond=None)[0]
        intercept, beta = float(coef[0]), coef[1:]
        y_hat = X_design @ coef
    else:
        beta = np.linalg.lstsq(X_used, y_used, rcond=None)[0]
        intercept = 0.0
        y_hat = X_used @ beta

    beta = np.asarray(beta, dtype=float).reshape(3)
    result = {"axis_unit": _unit(beta), "beta": beta, "intercept": intercept}

    if return_model_details:
        residuals = y_used - y_hat
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_used - y_used.mean()) ** 2)
        result.update({"coords_mean": coords_mean, "feature_mean": feature_mean,
                        "fitted": y_hat, "residuals": residuals,
                        "r2": np.nan if ss_tot == 0 else 1 - ss_res / ss_tot,
                        "n_valid": len(y_used)})
    return result


def bootstrap_spatial_axis_linear(feature, coords, *, n_boot=5000, seed=0,
                                   add_intercept=True, center_coords=False,
                                   center_feature=False, align_to_observed=True,
                                   min_norm=1e-12, return_full=False):
    """Bootstrap the linear spatial axis to produce confidence distributions."""
    y = np.asarray(feature, dtype=float).reshape(-1)
    X = np.asarray(coords, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    if len(y) < 4:
        raise ValueError("Need at least 4 valid samples.")

    obs = fit_spatial_axis_linear(y, X, add_intercept=add_intercept,
                                   center_coords=center_coords, center_feature=center_feature)
    axis_obs = obs["axis_unit"]
    rng = np.random.default_rng(seed)
    axis_boot, beta_boot, intercept_boot, failed = [], [], [], 0

    for _ in range(n_boot):
        ind = rng.integers(0, len(y), size=len(y))
        try:
            res_b = fit_spatial_axis_linear(y[ind], X[ind], add_intercept=add_intercept,
                                             center_coords=center_coords, center_feature=center_feature)
            beta_b = np.asarray(res_b["beta"], dtype=float).reshape(3)
            if np.linalg.norm(beta_b) < min_norm:
                failed += 1; continue
            axis_b = np.asarray(res_b["axis_unit"], dtype=float).reshape(3)
            if align_to_observed and np.dot(axis_b, axis_obs) < 0:
                axis_b, beta_b = -axis_b, -beta_b
            axis_boot.append(axis_b); beta_boot.append(beta_b)
            intercept_boot.append(res_b["intercept"])
        except Exception:
            failed += 1

    axis_boot = np.asarray(axis_boot, dtype=float)
    result = {"axis_unit": axis_obs, "beta": obs["beta"], "intercept": obs["intercept"],
              "axis_boot": axis_boot, "n_boot_valid": len(axis_boot), "n_boot_failed": failed}
    if return_full:
        result["beta_boot"] = np.asarray(beta_boot, dtype=float)
        result["intercept_boot"] = np.asarray(intercept_boot, dtype=float)
    return result


# ---------------------------------------------------------------------------
# LDA axis
# ---------------------------------------------------------------------------

def fit_spatial_axis_LDA(labels, coords, *, max_components=1):
    """Fit LDA to find the direction that maximally separates categorical labels in 3D space."""
    y = np.asarray(labels)
    X = np.asarray(coords, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3).")
    ok = np.all(np.isfinite(X), axis=1) & pd_notnull_1d(y)
    y, X = y[ok], X[ok]
    if len(y) < 3:
        raise ValueError("Need at least 3 valid samples.")

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Need at least 2 label classes for LDA.")
    if np.any(counts < 2):
        raise ValueError("Each class needs at least 2 samples for stable LDA.")

    n_components = min(max_components, 1, X.shape[1], len(classes) - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X, y)

    if hasattr(lda, "scalings_") and lda.scalings_ is not None and lda.scalings_.shape[1] >= 1:
        beta = np.asarray(lda.scalings_[:, 0], dtype=float).reshape(3)
    else:
        beta = np.asarray(lda.coef_[0], dtype=float).reshape(3)

    beta_norm = np.linalg.norm(beta)
    if not np.isfinite(beta_norm) or beta_norm < 1e-12:
        raise ValueError("Estimated LDA axis has near-zero norm.")
    return {"axis_unit": beta / beta_norm, "beta": beta, "intercept": 0.0}


def bootstrap_spatial_axis_LDA(labels, coords, *, n_boot=5000, seed=0,
                                align_to_observed=True, min_norm=1e-12, return_full=False):
    """Bootstrap the LDA spatial axis."""
    y = np.asarray(labels)
    X = np.asarray(coords, dtype=float)
    ok = np.all(np.isfinite(X), axis=1) & pd_notnull_1d(y)
    y, X = y[ok], X[ok]
    if len(y) < 3:
        raise ValueError("Need at least 3 valid samples.")

    obs = fit_spatial_axis_LDA(y, X)
    axis_obs = obs["axis_unit"]
    rng = np.random.default_rng(seed)
    axis_boot, beta_boot, intercept_boot, failed = [], [], [], 0

    for _ in range(n_boot):
        ind = rng.integers(0, len(y), size=len(y))
        try:
            yb, Xb = y[ind], X[ind]
            classes_b, counts_b = np.unique(yb, return_counts=True)
            if len(classes_b) < 2 or np.any(counts_b < 2):
                failed += 1; continue
            res_b = fit_spatial_axis_LDA(yb, Xb)
            beta_b = np.asarray(res_b["beta"], dtype=float).reshape(3)
            if np.linalg.norm(beta_b) < min_norm:
                failed += 1; continue
            axis_b = np.asarray(res_b["axis_unit"], dtype=float).reshape(3)
            if align_to_observed and np.dot(axis_b, axis_obs) < 0:
                axis_b, beta_b = -axis_b, -beta_b
            axis_boot.append(axis_b); beta_boot.append(beta_b)
            intercept_boot.append(res_b["intercept"])
        except Exception:
            failed += 1

    axis_boot = np.asarray(axis_boot, dtype=float)
    result = {"axis_unit": axis_obs, "beta": obs["beta"], "intercept": obs["intercept"],
              "axis_boot": axis_boot, "n_boot_valid": len(axis_boot), "n_boot_failed": failed}
    if return_full:
        result["beta_boot"] = np.asarray(beta_boot, dtype=float)
        result["intercept_boot"] = np.asarray(intercept_boot, dtype=float)
    return result


# ---------------------------------------------------------------------------
# CCA axis
# ---------------------------------------------------------------------------

def fit_spatial_axis_cca(features, coords, *, n_components=1,
                          standardize_features=True, standardize_coords=False):
    """Find the spatial direction most correlated with a multivariate feature set via CCA."""
    S = np.asarray(features, dtype=float)
    X = np.asarray(coords, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3).")
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if len(S) != len(X):
        raise ValueError("features and coords must have the same length.")

    ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(S), axis=1)
    X, S = X[ok], S[ok]
    if len(X) < 4:
        raise ValueError("Need at least 4 valid samples.")

    keep_feat = np.nanstd(S, axis=0) > 0
    keep_coord = np.nanstd(X, axis=0) > 0
    S = S[:, keep_feat]
    X_used = X[:, keep_coord]

    if standardize_coords:
        X_used = StandardScaler().fit_transform(X_used)
    S_used = StandardScaler().fit_transform(S) if standardize_features else S.copy()

    max_comp = min(n_components, X_used.shape[1], S_used.shape[1], len(X_used) - 1)
    if max_comp < 1:
        raise ValueError("Not enough data/features to fit at least 1 CCA component.")

    cca = CCA(n_components=max_comp)
    cca.fit(X_used, S_used)
    X_c, S_c = cca.transform(X_used, S_used)
    canonical_corr = float(np.corrcoef(X_c[:, 0], S_c[:, 0])[0, 1])

    beta_reduced = np.asarray(cca.x_weights_[:, 0], dtype=float).reshape(-1)
    beta = np.zeros(3, dtype=float)
    beta[keep_coord] = beta_reduced
    beta_norm = np.linalg.norm(beta)
    if not np.isfinite(beta_norm) or beta_norm < 1e-12:
        raise ValueError("Estimated CCA axis has near-zero norm.")

    return {"axis_unit": beta / beta_norm, "beta": beta,
            "intercept": 0.0, "canonical_corr": canonical_corr}


def bootstrap_spatial_axis_cca(features, coords, *, n_boot=5000, seed=0, n_components=1,
                                standardize_features=True, standardize_coords=False,
                                align_to_observed=True, min_norm=1e-12, return_full=False,
                                bootstrap_within_spatial_bins=False, spatial_bin_edges=None,
                                spatial_bin_counts=(2, 3, 3)):
    """Bootstrap the CCA spatial axis, optionally resampling within spatial bins."""
    S = np.asarray(features, dtype=float)
    X = np.asarray(coords, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(S), axis=1)
    X, S = X[ok], S[ok]
    if len(X) < 4:
        raise ValueError("Need at least 4 valid samples.")

    bin_ids = None
    used_bin_edges = None
    if bootstrap_within_spatial_bins:
        bin_ids, valid_bins, used_bin_edges = _make_spatial_bin_ids(
            X, spatial_bin_edges=spatial_bin_edges, spatial_bin_counts=spatial_bin_counts)
        X, S, bin_ids = X[valid_bins], S[valid_bins], bin_ids[valid_bins]

    obs = fit_spatial_axis_cca(S, X, n_components=n_components,
                                standardize_features=standardize_features,
                                standardize_coords=standardize_coords)
    axis_obs = obs["axis_unit"]
    rng = np.random.default_rng(seed)
    axis_boot, beta_boot, intercept_boot, corr_boot, failed = [], [], [], [], 0

    for _ in range(n_boot):
        ind = (_bootstrap_indices_within_bins(bin_ids, rng)
               if bootstrap_within_spatial_bins else rng.integers(0, len(X), size=len(X)))
        try:
            res_b = fit_spatial_axis_cca(S[ind], X[ind], n_components=n_components,
                                          standardize_features=standardize_features,
                                          standardize_coords=standardize_coords)
            beta_b = np.asarray(res_b["beta"], dtype=float).reshape(3)
            if np.linalg.norm(beta_b) < min_norm:
                failed += 1; continue
            axis_b = np.asarray(res_b["axis_unit"], dtype=float).reshape(3)
            if align_to_observed and np.dot(axis_b, axis_obs) < 0:
                axis_b, beta_b = -axis_b, -beta_b
            axis_boot.append(axis_b); beta_boot.append(beta_b)
            intercept_boot.append(res_b["intercept"]); corr_boot.append(float(res_b["canonical_corr"]))
        except Exception:
            failed += 1

    axis_boot = np.asarray(axis_boot, dtype=float)
    result = {"axis_unit": axis_obs, "beta": obs["beta"], "intercept": obs["intercept"],
              "axis_boot": axis_boot, "canonical_corr": obs["canonical_corr"],
              "n_boot_valid": len(axis_boot), "n_boot_failed": failed}
    if bootstrap_within_spatial_bins:
        result["spatial_bin_edges"] = used_bin_edges
    if return_full:
        result["beta_boot"] = np.asarray(beta_boot, dtype=float)
        result["intercept_boot"] = np.asarray(intercept_boot, dtype=float)
        result["canonical_corr_boot"] = np.asarray(corr_boot, dtype=float)
    return result


# ---------------------------------------------------------------------------
# Direction comparison
# ---------------------------------------------------------------------------

def compare_bootstrap_directions(b_x, b_y, b_x_boot, b_y_boot, *,
                                  assume_paired=True, flip_to_observed=True):
    """Compare two 3D unit-vector directions using bootstrap tangent-plane Wald test."""
    b_x, b_y = _unit(b_x), _unit(b_y)
    b_x_boot = np.asarray(b_x_boot, float)
    b_y_boot = np.asarray(b_y_boot, float)

    valid = (np.all(np.isfinite(b_x_boot), axis=1) & (np.linalg.norm(b_x_boot, axis=1) > 0) &
             np.all(np.isfinite(b_y_boot), axis=1) & (np.linalg.norm(b_y_boot, axis=1) > 0))
    b_x_boot, b_y_boot = b_x_boot[valid], b_y_boot[valid]
    b_x_boot /= np.linalg.norm(b_x_boot, axis=1, keepdims=True)
    b_y_boot /= np.linalg.norm(b_y_boot, axis=1, keepdims=True)

    if flip_to_observed:
        b_x_boot[np.sum(b_x_boot * b_x, axis=1) < 0] *= -1
        b_y_boot[np.sum(b_y_boot * b_y, axis=1) < 0] *= -1

    e1, e2, _ = _orthonormal_basis_perp(b_x, b_y)
    A = np.vstack([e1, e2])
    d_obs = A @ b_y
    Px = (A @ b_x_boot.T).T
    Py = (A @ b_y_boot.T).T

    if assume_paired:
        if len(Px) != len(Py):
            raise ValueError("Paired bootstrap requires the same number of bootstrap samples.")
        d_boot = Py - Px
    else:
        rng = np.random.default_rng(0)
        n_pair = min(10000, len(Px) * len(Py))
        ix = rng.integers(0, len(Px), size=n_pair)
        iy = rng.integers(0, len(Py), size=n_pair)
        d_boot = Py[iy] - Px[ix]

    mean_boot = d_boot.mean(axis=0)
    cov_2d = np.cov(d_boot.T, ddof=1) + np.eye(2) * 1e-12
    cov_inv = np.linalg.inv(cov_2d)

    W_obs = float(d_obs.T @ cov_inv @ d_obs)
    p_chi2 = chi2.sf(W_obs, df=2)

    d_boot_null = d_boot - mean_boot
    W_boot_null = np.einsum("ni,ij,nj->n", d_boot_null, cov_inv, d_boot_null)
    p_boot = (np.sum(W_boot_null >= W_obs) + 1) / (len(W_boot_null) + 1)

    thr95 = chi2.ppf(0.95, df=2)
    delta0 = -mean_boot
    maha2_zero = float(delta0.T @ cov_inv @ delta0)

    ci95_e1 = np.percentile(d_boot[:, 0], [2.5, 97.5])
    ci95_e2 = np.percentile(d_boot[:, 1], [2.5, 97.5])

    return {
        "d_obs": d_obs, "d_boot": d_boot, "d_boot_null": d_boot_null,
        "mean_boot": mean_boot, "cov_2d": cov_2d, "W_obs": W_obs,
        "W_boot_null": W_boot_null, "p_chi2": p_chi2, "p_boot": p_boot,
        "angle_deg": float(np.degrees(np.arccos(np.clip(np.dot(b_x, b_y), -1, 1)))),
        "basis_e1": e1, "basis_e2": e2, "reference_axis": b_x,
        "n_boot_valid": len(d_boot), "ci95_e1": ci95_e1, "ci95_e2": ci95_e2,
        "maha2_zero": maha2_zero, "zero_inside_95ci": maha2_zero <= thr95,
    }


# ---------------------------------------------------------------------------
# Azimuth / elevation helpers
# ---------------------------------------------------------------------------

def vectors_to_az_el(vectors, degrees=True):
    """Convert 3D unit vectors to (azimuth, elevation)."""
    v = np.asarray(vectors)
    az = np.arctan2(v[:, 1], v[:, 0])
    el = np.arcsin(np.clip(v[:, 2], -1, 1))
    if degrees:
        az, el = np.degrees(az), np.degrees(el)
    return az, el


# ---------------------------------------------------------------------------
# 3D cone visualization helpers
# ---------------------------------------------------------------------------

def _angle_deg_between(v, u):
    return np.degrees(np.arccos(np.clip(np.dot(_unit(v), _unit(u)), -1, 1)))


def _orthonormal_basis_perp_axis(axis):
    axis = _unit(axis)
    a = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(a - np.dot(a, axis) * axis)
    return e1, _unit(np.cross(axis, e1))


def cone_half_angle_from_boot(axis, axis_boot, q=95):
    """Return the q-th percentile half-angle of bootstrap axes around the observed axis."""
    ang = np.array([_angle_deg_between(b, axis) for b in axis_boot])
    return np.percentile(ang, q), ang


def cone_boundary_3d(axis, half_angle_deg, n=240):
    """Return points on the boundary circle of a cone on the unit sphere."""
    axis = _unit(axis)
    e1, e2 = _orthonormal_basis_perp_axis(axis)
    theta = np.linspace(0, 2 * np.pi, n)
    alpha = np.radians(half_angle_deg)
    return (np.cos(alpha) * axis[None, :]
            + np.sin(alpha) * (np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2))


def plot_projected_arrow_with_cone(ax, origin, axis, axis_boot, dims, *,
                                   color="red", scale=1.0, head_width=0.1, head_length=0.2,
                                   cone_q=95, cone_alpha=0.18, cone_n=240, label=None):
    """Plot a projected 3D axis arrow with a shaded 95% confidence cone band."""
    axis = _unit(axis)
    v2 = axis[list(dims)] * scale
    half_angle_deg, ang = cone_half_angle_from_boot(axis, axis_boot, q=cone_q)
    cone_pts_3d = cone_boundary_3d(axis, half_angle_deg, n=cone_n)
    cone_pts_2d = cone_pts_3d[:, list(dims)] * scale + origin[None, :]

    poly = np.vstack([origin[None, :], cone_pts_2d, origin[None, :]])
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=cone_alpha, linewidth=0)
    ax.plot(cone_pts_2d[:, 0], cone_pts_2d[:, 1], color=color, alpha=0.6, linewidth=1)
    ax.arrow(origin[0], origin[1], v2[0], v2[1],
             head_width=head_width, head_length=head_length,
             fc=color, ec=color, linewidth=2, length_includes_head=True, label=label)
    return {"half_angle_deg": half_angle_deg, "angles_deg": ang}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_vector_distribution_az_el_2d(axis_boot, axis_obs, color, ax=None):
    """Scatter bootstrap axis directions in azimuth-elevation space."""
    az, el = vectors_to_az_el(axis_boot)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(az, el, s=2, alpha=1, color=color, edgecolor='w', linewidth=0.5)
    if axis_obs is not None:
        az_obs, el_obs = vectors_to_az_el(axis_obs.reshape(1, 3))
        ax.scatter(az_obs, el_obs, s=120, marker='x')
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_aspect('equal')
    return ax


def plot_direction_difference(res, ci_level=0.95, ax=None, print_stats=True):
    """Plot bootstrap distribution of the 2D tangent-plane directional difference."""
    d_boot = np.asarray(res["d_boot"])
    d_obs = np.asarray(res["d_obs"])
    cov = np.asarray(res["cov_2d"])
    mean_boot = np.asarray(res.get("mean_boot", d_boot.mean(axis=0)))

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(d_boot[:, 0], d_boot[:, 1], s=5, alpha=0.25, color="k",
               edgecolor='none', label="Bootstrap difference")
    ax.scatter(d_obs[0], d_obs[1], s=120, marker="x", linewidths=2.5,
               label="Observed y-from-x deviation", zorder=5)
    ax.scatter(mean_boot[0], mean_boot[1], s=70, marker="o", label="Bootstrap mean", zorder=4)
    ax.scatter(0, 0, s=60, marker="+", linewidths=2, label="No difference (0,0)", zorder=4)
    ax.axhline(0, linewidth=1); ax.axvline(0, linewidth=1)

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    radii = np.sqrt(vals * chi2.ppf(ci_level, df=2))
    theta = np.linspace(0, 2 * np.pi, 300)
    ellipse = vecs @ np.diag(radii) @ np.vstack([np.cos(theta), np.sin(theta)]) + mean_boot[:, None]
    ax.fill_between(ellipse[0], ellipse[1], color="gray", alpha=0.2, zorder=2, edgecolor='none')

    ax.set_xlabel("Deviation along e1"); ax.set_ylabel("Deviation along e2")
    ax.set_aspect("equal")
    ax.set_title(f"2D directional difference: y vs x\n"
                 f"p_boot={res['p_boot']:.4g}, p_chi2={res['p_chi2']:.4g}, "
                 f"angle={res['angle_deg']:.2f}°")
    ax.legend(frameon=False)

    if print_stats:
        print("=== Direction comparison stats ===")
        print(f"Observed angle (deg): {res['angle_deg']:.6f}")
        print(f"d_obs: [{d_obs[0]:.6f}, {d_obs[1]:.6f}]")
        print(f"Bootstrap mean: [{mean_boot[0]:.6f}, {mean_boot[1]:.6f}]")
        print(f"W_obs: {res['W_obs']:.6f},  p_boot: {res['p_boot']:.6g},  p_chi2: {res['p_chi2']:.6g}")
        print(f"n_boot_valid: {res['n_boot_valid']}")
        cov_inv = np.linalg.inv(cov)
        maha2 = float((d_obs - mean_boot).T @ cov_inv @ (d_obs - mean_boot))
        print(f"Observed point inside {int(ci_level*100)}% CI ellipse: {maha2 <= chi2.ppf(ci_level, df=2)}")
    return ax


def get_regression_CI(x, y):
    """Return regression line and analytic 95% CI bands."""
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1)
    reg = LinearRegression().fit(x, y)
    x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    y_fit = reg.predict(x_fit.reshape(-1, 1))

    n = len(x)
    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean) ** 2)
    residuals = y - reg.predict(x)
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    t_val = stats.t.ppf(0.975, df=n - 2)
    conf = t_val * s_err * np.sqrt(1 / n + (x_fit - x_mean.squeeze()) ** 2 / Sxx)
    return y_fit, x_fit, y_fit - conf, y_fit + conf
