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
    Rank the target genes within each patient. This is the clinical question
    ("which unassayed genes should I suspect in *this* tumour?") but it is
    **also** prevalence-confounded: ranking genes inside a patient by predicted
    probability is, for a patient-blind model, just ranking them by prevalence,
    and the positives are preferentially the common genes. In the simulator's
    ``independent`` regime -- where the target block is statistically
    independent of the patient -- pooled AUROC reaches ~0.80 and within-patient
    AUROC ~0.81, while macro AUROC correctly reads 0.500.

``*_prevalence_adjusted``
    Pooled and within-patient AUROC recomputed after removing each gene's own
    mean predicted score (:func:`residualize_by_gene`). This strips the
    prevalence channel and leaves only patient-specific ranking, so these read
    0.5 under independence. Because AUROC within a gene is invariant to
    per-gene monotone shifts, macro AUROC is unchanged by the adjustment.

Report macro AUROC as the headline. Report pooled and within-patient only
alongside their prevalence-adjusted counterparts and the prevalence-only
baseline, or they will be read as evidence they cannot supply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.stats import rankdata

__all__ = [
    "benjamini_hochberg",
    "per_gene_permutation_test",
    "quantize_by_column",
    "residualize_by_gene",
    "auroc_columnwise",
    "auroc_rowwise",
    "pooled_auroc",
    "macro_auroc",
    "within_patient_auroc",
    "pooled_auroc_prevalence_adjusted",
    "within_patient_auroc_prevalence_adjusted",
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


def quantize_by_column(values, tol: float = 1e-9, scale=None) -> np.ndarray:
    """Collapse per-column differences smaller than ``tol`` x the column scale.

    Rank-based metrics have no floor on what counts as a difference: two scores
    separated by one ULP are ranked as confidently as two separated by 0.5. Any
    arithmetic that *should* produce identical scores but produces
    floating-point noise instead -- centring a constant column, aligning fold
    means -- therefore leaks a spurious ordering into AUROC. Snapping to a
    relative grid makes "numerically indistinguishable" mean "tied", which is
    what the statistics assume.
    """
    v = _as_float_matrix(values)
    if scale is None:
        scale = np.abs(v).max(axis=0, keepdims=True)
    # The grid must be set by the scale of the *inputs*, not of the output. For
    # a residual that is pure rounding noise the output scale is ~1e-16, and a
    # grid derived from it would preserve exactly the noise being removed.
    step = tol * np.maximum(np.asarray(scale, dtype=float), 1e-12)
    return np.round(v / step) * step


def residualize_by_gene(
    y_score, eps: float = 1e-6, tol: float = 1e-9
) -> np.ndarray:
    """Remove each gene's own mean score, on the logit scale.

    Turns "how likely is this gene in general" into "how much more likely is it
    in *this* patient than the model expects on average". Pooled and
    within-patient AUROC computed on the residuals measure patient-specific
    ranking only. Per-gene (macro) AUROC is unaffected, since subtracting a
    per-gene constant is monotone within each gene.

    ``tol`` snaps numerically-negligible residuals to exactly zero. This is not
    cosmetic: for a patient-blind model every column is constant, the column
    mean can differ from that constant by one ULP, and the surviving per-column
    offsets would then be *ranked* by pooled AUROC -- silently reinstating the
    prevalence ordering this function exists to remove.
    """
    s = _as_float_matrix(y_score)
    # Probabilities get a logit first so the centring is on an additive scale;
    # scores that are not probabilities are centred as-is.
    if s.size and np.all((s >= 0) & (s <= 1)):
        clipped = np.clip(s, eps, 1 - eps)
        s = np.log(clipped / (1 - clipped))
    return quantize_by_column(
        s - s.mean(axis=0, keepdims=True),
        tol,
        scale=np.maximum(np.abs(s).max(axis=0, keepdims=True), 1.0),
    )


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


def pooled_auroc_prevalence_adjusted(y_true, y_score) -> float:
    """Pooled AUROC after removing the per-gene prevalence channel."""
    return pooled_auroc(y_true, residualize_by_gene(y_score))


def within_patient_auroc_prevalence_adjusted(
    y_true, y_score, min_positives: int = 1
) -> float:
    """Within-patient AUROC after removing the per-gene prevalence channel."""
    return within_patient_auroc(y_true, residualize_by_gene(y_score), min_positives)


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
    pooled_auroc_adjusted: float
    within_patient_auroc_adjusted: float
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
            "pooled_auroc_adjusted": self.pooled_auroc_adjusted,
            "within_patient_auroc_adjusted": self.within_patient_auroc_adjusted,
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
        pooled_auroc_adjusted=pooled_auroc_prevalence_adjusted(y, p),
        within_patient_auroc_adjusted=within_patient_auroc_prevalence_adjusted(y, p),
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


def benjamini_hochberg(p_values, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR control over the per-gene tests.

    164 genes means 164 hypotheses. At alpha = 0.05 roughly eight genes clear
    the bar by chance alone, which is enough to populate a "genes we can
    predict" list entirely with noise. BH controls the expected proportion of
    false discoveries among the genes reported, which is the right error rate
    when the output is a shortlist rather than a single decision.

    NaN p-values (genes with too few positives to test) are carried through as
    NaN and excluded from the correction, so they cannot inflate the ranking.
    """
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    q = np.full(p.shape, np.nan)

    tested = p[finite]
    n = tested.size
    if n == 0:
        return {"q_values": q, "n_tested": 0, "n_significant": 0, "alpha": alpha}

    order = np.argsort(tested)
    ranked = tested[order]
    # q_(i) = min over j >= i of  n/j * p_(j)   -- enforced by a reverse cumulative min
    scaled = ranked * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_tested = np.empty(n)
    q_tested[order] = q_sorted
    q[finite] = q_tested

    return {
        "q_values": q,
        "n_tested": int(n),
        "n_significant": int((q_tested <= alpha).sum()),
        "alpha": alpha,
    }


def per_gene_permutation_test(
    y_true, y_score, n_permutations: int = 500, seed: int = 0, min_positives: int = 5
) -> dict:
    """Permutation test per gene, with BH correction across genes.

    The cohort-level permutation test says *some* gene is predictable. Naming
    *which* genes needs a test per gene, and therefore a correction.
    """
    y = _as_float_matrix(y_true)
    s = _as_float_matrix(y_score)
    rng = np.random.default_rng(seed)

    observed = auroc_columnwise(y, s, min_positives=min_positives)
    n_rows = y.shape[0]
    ge = np.zeros(y.shape[1])
    valid = np.zeros(y.shape[1])
    for _ in range(n_permutations):
        permuted = auroc_columnwise(y, s[rng.permutation(n_rows)], min_positives=min_positives)
        ok = np.isfinite(permuted) & np.isfinite(observed)
        ge[ok] += permuted[ok] >= observed[ok]
        valid[ok] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        p = (1.0 + ge) / (1.0 + valid)
    p[~np.isfinite(observed)] = np.nan
    bh = benjamini_hochberg(p)
    return {
        "auroc": observed,
        "p_values": p,
        "q_values": bh["q_values"],
        "n_tested": bh["n_tested"],
        "n_significant": bh["n_significant"],
    }


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
