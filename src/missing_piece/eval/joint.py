"""Metrics for the joint distribution, where a generative model should earn its keep.

Per-gene AUROC only scores marginals, and a discriminative classifier optimises
those directly. The reason to prefer a generative model is that it can produce
*coherent whole profiles*: right co-occurrence structure, right mutual
exclusivity, right burden distribution. These metrics measure that, by
comparing statistics of sampled profiles against the held-out truth.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

__all__ = [
    "cooccurrence_matrix",
    "cooccurrence_fidelity",
    "burden_distribution_distance",
    "evaluate_joint",
]


def cooccurrence_matrix(y: np.ndarray, min_prevalence: float = 0.0) -> np.ndarray:
    """Phi (mean-centred) correlation between genes, NaN-safe for constant genes."""
    y = np.asarray(y, dtype=float)
    keep = y.mean(axis=0) > min_prevalence
    y = y[:, keep]
    if y.shape[1] < 2:
        return np.full((0, 0), np.nan)
    sd = y.std(axis=0)
    sd[sd < 1e-12] = np.nan
    centred = (y - y.mean(axis=0)) / sd
    corr = (centred.T @ centred) / y.shape[0]
    return corr


def cooccurrence_fidelity(
    y_true: np.ndarray, y_sampled: np.ndarray, min_prevalence: float = 0.005
) -> dict:
    """How well sampled profiles reproduce observed gene-gene co-alteration.

    ``y_sampled`` is (n_samples, n_patients, n_genes); statistics are pooled
    over draws. Reported as the correlation between the off-diagonal entries of
    the true and sampled phi matrices, plus their mean absolute difference.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_sampled = np.asarray(y_sampled, dtype=float)
    if y_sampled.ndim == 2:
        y_sampled = y_sampled[None]
    flat = y_sampled.reshape(-1, y_sampled.shape[-1])

    keep = y_true.mean(axis=0) > min_prevalence
    if keep.sum() < 2:
        return {"n_genes": int(keep.sum()), "phi_corr": np.nan, "phi_mae": np.nan}

    a = cooccurrence_matrix(y_true[:, keep])
    b = cooccurrence_matrix(flat[:, keep])
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return {"n_genes": int(keep.sum()), "phi_corr": np.nan, "phi_mae": np.nan}

    iu = np.triu_indices_from(a, k=1)
    va, vb = a[iu], b[iu]
    ok = np.isfinite(va) & np.isfinite(vb)
    if ok.sum() < 3:
        return {"n_genes": int(keep.sum()), "phi_corr": np.nan, "phi_mae": np.nan}
    return {
        "n_genes": int(keep.sum()),
        "phi_corr": float(np.corrcoef(va[ok], vb[ok])[0, 1]),
        "phi_mae": float(np.mean(np.abs(va[ok] - vb[ok]))),
    }


def burden_distribution_distance(y_true: np.ndarray, y_sampled: np.ndarray) -> dict:
    """KS distance between true and sampled per-patient target-alteration counts."""
    y_true = np.asarray(y_true, dtype=float)
    y_sampled = np.asarray(y_sampled, dtype=float)
    if y_sampled.ndim == 2:
        y_sampled = y_sampled[None]
    true_counts = y_true.sum(axis=1)
    sampled_counts = y_sampled.sum(axis=-1).ravel()
    ks = ks_2samp(true_counts, sampled_counts)
    return {
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "true_mean_burden": float(true_counts.mean()),
        "sampled_mean_burden": float(sampled_counts.mean()),
    }


def evaluate_joint(
    y_true: np.ndarray, y_sampled: np.ndarray, min_prevalence: float = 0.005
) -> dict:
    """Bundle the joint-structure metrics."""
    out = {}
    out.update(cooccurrence_fidelity(y_true, y_sampled, min_prevalence))
    out.update(burden_distribution_distance(y_true, y_sampled))
    return out
