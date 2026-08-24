"""Metrics for panel completion.

Why there is more than one AUROC here
-------------------------------------
With 164 genes at ~1.2% prevalence, *how* you average AUROC decides what the
number means:

``pooled``
    One ranking over all patient-gene pairs. Gene prevalence alone ranks pairs
    well -- a model that ignores the patient completely and emits each gene's
    training prevalence scores far above 0.5. A pooled AUROC is therefore not
    evidence of patient-specific prediction, and must always be read against
    the prevalence-only baseline computed the same way.

``macro`` (per-gene)
    AUROC computed within each gene, then averaged. Constant-per-gene scores
    give exactly 0.5 by construction, so any lift is patient-specific signal.
    This is the metric that tests the hypothesis.

``within_patient``
    Rank the target genes within each patient. Answers the clinical question
    "which unassayed genes should I suspect in *this* tumour?".

Report all three. They routinely disagree, and only the last two survive the
prevalence confound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.stats import rankdata

__all__ = [
    "auroc_columnwise",
    "auroc_rowwise",
    "pooled_auroc",
    "macro_auroc",
    "within_patient_auroc",
    "average_precision_columnwise",
    "macro_average_precision",
    "brier_score",
    "expected_calibration_error",
    "EvaluationResult",
    "evaluate_predictions",
    "bootstrap_metric",
    "permutation_test",
]

_UNDEFINED = np.nan


def _as_float_matrix(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {arr.shape}")
    return arr.astype(float, copy=False)


def auroc_columnwise(
    y_true, y_score, min_positives: int = 1, min_negatives: int = 1
) -> np.ndarray:
    """AUROC within each column. Returns NaN for columns without both classes.

    Uses the rank-sum identity, which handles ties by mid-rank exactly as the
    trapezoidal ROC does, and is fast enough to bootstrap.
    """
    y = _as_float_matrix(y_true)
    s = _as_float_matrix(y_score)
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {s.shape}")

    n_pos = y.sum(axis=0)
    n_neg = y.shape[0] - n_pos
    ranks = rankdata(s, axis=0)
    rank_sum_pos = (ranks * y).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    undefined = (n_pos < min_positives) | (n_neg < min_negatives)
    auc[undefined] = _UNDEFINED
    return auc


def auroc_rowwise(y_true, y_score, min_positives: int = 1) -> np.ndarray:
    """AUROC within each row (e.g. ranking genes inside one patient)."""
    return auroc_columnwise(
        _as_float_matrix(y_true).T, _as_float_matrix(y_score).T, min_positives
    )


def pooled_auroc(y_true, y_score) -> float:
    """AUROC over every patient-gene pair pooled into a single ranking."""
    y = _as_float_matrix(y_true).ravel()[:, None]
    s = _as_float_matrix(y_score).ravel()[:, None]
    return float(auroc_columnwise(y, s)[0])


def macro_auroc(y_true, y_score, min_positives: int = 5) -> float:
    """Mean per-gene AUROC over genes with enough positives to be estimable."""
    per_gene = auroc_columnwise(y_true, y_score, min_positives=min_positives)
    return float(np.nanmean(per_gene)) if np.isfinite(per_gene).any() else _UNDEFINED


def within_patient_auroc(y_true, y_score, min_positives: int = 1) -> float:
    """Mean per-patient AUROC over patients with at least one target alteration."""
    per_patient = auroc_rowwise(y_true, y_score, min_positives=min_positives)
    return float(np.nanmean(per_patient)) if np.isfinite(per_patient).any() else _UNDEFINED


def average_precision_columnwise(y_true, y_score, min_positives: int = 1) -> np.ndarray:
    """Average precision within each column, vectorised over columns."""
    y = _as_float_matrix(y_true)
    s = _as_float_matrix(y_score)
    n_rows, n_cols = y.shape

    order = np.argsort(-s, axis=0, kind="stable")
    y_sorted = np.take_along_axis(y, order, axis=0)
    tp = np.cumsum(y_sorted, axis=0)
    k = np.arange(1, n_rows + 1)[:, None]
    precision = tp / k
    n_pos = y.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = (precision * y_sorted).sum(axis=0) / n_pos
    ap[n_pos < min_positives] = _UNDEFINED
    return ap


def macro_average_precision(y_true, y_score, min_positives: int = 5) -> float:
    ap = average_precision_columnwise(y_true, y_score, min_positives=min_positives)
    return float(np.nanmean(ap)) if np.isfinite(ap).any() else _UNDEFINED


def brier_score(y_true, y_prob) -> float:
    y = _as_float_matrix(y_true)
    p = np.clip(_as_float_matrix(y_prob), 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins: int = 20) -> float:
    """ECE with equal-count bins, which behave better than equal-width at 1% prevalence."""
    y = _as_float_matrix(y_true).ravel()
    p = np.clip(_as_float_matrix(y_prob).ravel(), 0.0, 1.0)
    if y.size == 0:
        return _UNDEFINED
    order = np.argsort(p)
    y, p = y[order], p[order]
    bins = np.array_split(np.arange(y.size), min(n_bins, max(1, y.size)))
    ece = 0.0
    for b in bins:
        if b.size == 0:
            continue
        ece += (b.size / y.size) * abs(p[b].mean() - y[b].mean())
    return float(ece)


@dataclass
class EvaluationResult:
    """All headline metrics for one model on one split."""

    macro_auroc: float
    pooled_auroc: float
    within_patient_auroc: float
    macro_average_precision: float
    brier: float
    ece: float
    n_patients: int
    n_genes_scored: int
    n_positive_pairs: int
    per_gene_auroc: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    per_gene_ap: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    gene_names: tuple[str, ...] = field(repr=False, default=())
    extra: dict = field(default_factory=dict)

    def to_dict(self, include_per_gene: bool = False) -> dict:
        out = {
            "macro_auroc": self.macro_auroc,
            "pooled_auroc": self.pooled_auroc,
            "within_patient_auroc": self.within_patient_auroc,
            "macro_average_precision": self.macro_average_precision,
            "brier": self.brier,
            "ece": self.ece,
            "n_patients": self.n_patients,
            "n_genes_scored": self.n_genes_scored,
            "n_positive_pairs": self.n_positive_pairs,
            **self.extra,
        }
        if include_per_gene:
            out["per_gene_auroc"] = {
                g: (None if not np.isfinite(v) else float(v))
                for g, v in zip(self.gene_names, self.per_gene_auroc)
            }
        return out


def evaluate_predictions(
    y_true,
    y_prob,
    gene_names: Sequence[str] | None = None,
    min_positives: int = 5,
) -> EvaluationResult:
    """Compute the full metric panel for one set of predictions."""
    y = _as_float_matrix(y_true)
    p = _as_float_matrix(y_prob)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: labels {y.shape}, predictions {p.shape}")

    per_gene = auroc_columnwise(y, p, min_positives=min_positives)
    per_gene_ap = average_precision_columnwise(y, p, min_positives=min_positives)
    return EvaluationResult(
        macro_auroc=float(np.nanmean(per_gene)) if np.isfinite(per_gene).any() else _UNDEFINED,
        pooled_auroc=pooled_auroc(y, p),
        within_patient_auroc=within_patient_auroc(y, p),
        macro_average_precision=(
            float(np.nanmean(per_gene_ap)) if np.isfinite(per_gene_ap).any() else _UNDEFINED
        ),
        brier=brier_score(y, p),
        ece=expected_calibration_error(y, p),
        n_patients=int(y.shape[0]),
        n_genes_scored=int(np.isfinite(per_gene).sum()),
        n_positive_pairs=int(y.sum()),
        per_gene_auroc=per_gene,
        per_gene_ap=per_gene_ap,
        gene_names=tuple(gene_names) if gene_names is not None else (),
    )


def bootstrap_metric(
    y_true,
    y_prob,
    metric: Callable[[np.ndarray, np.ndarray], float] = macro_auroc,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Patient-level bootstrap CI.

    Resampling patients (not pairs) is the right unit: the 164 predictions for
    one patient are highly dependent, so pair-level resampling would understate
    the interval badly.
    """
    y = _as_float_matrix(y_true)
    p = _as_float_matrix(y_prob)
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = metric(y[idx], p[idx])
    finite = stats[np.isfinite(stats)]
    if finite.size == 0:
        return {"point": _UNDEFINED, "lo": _UNDEFINED, "hi": _UNDEFINED, "n_boot": 0}
    return {
        "point": float(metric(y, p)),
        "lo": float(np.quantile(finite, alpha / 2)),
        "hi": float(np.quantile(finite, 1 - alpha / 2)),
        "n_boot": int(finite.size),
    }


def permutation_test(
    y_true,
    y_prob,
    metric: Callable[[np.ndarray, np.ndarray], float] = macro_auroc,
    n_permutations: int = 500,
    seed: int = 0,
) -> dict:
    """Test whether predictions carry *patient-specific* information.

    Shuffling the rows of the prediction matrix destroys the patient-prediction
    correspondence while leaving each gene's predicted score distribution
    untouched. A model that only encodes gene prevalence is therefore invariant
    under this permutation, and its observed metric sits inside the null. A
    model that genuinely reads the patient's observed panel does not.
    """
    y = _as_float_matrix(y_true)
    p = _as_float_matrix(y_prob)
    rng = np.random.default_rng(seed)
    observed = metric(y, p)
    null = np.empty(n_permutations)
    n = y.shape[0]
    for i in range(n_permutations):
        null[i] = metric(y, p[rng.permutation(n)])
    finite = null[np.isfinite(null)]
    # Add-one correction keeps the p-value strictly positive and unbiased.
    p_value = float((1 + (finite >= observed).sum()) / (1 + finite.size))
    return {
        "observed": float(observed),
        "null_mean": float(finite.mean()) if finite.size else _UNDEFINED,
        "null_sd": float(finite.std(ddof=1)) if finite.size > 1 else _UNDEFINED,
        "p_value": p_value,
        "n_permutations": int(finite.size),
    }
