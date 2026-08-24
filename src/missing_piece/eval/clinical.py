"""Decision metrics, for when the claim becomes clinical rather than statistical.

AUROC is threshold-free and prevalence-blind. Neither property survives contact
with a clinic: a clinician acts at *one* threshold, on a population with a
*specific* base rate. At ~1.2% prevalence a gene with AUROC 0.80 can still have
a positive predictive value in the low single digits at any sensitivity worth
using, which is the difference between an interesting model and a useless test.

These functions answer the questions that actually gate deployment:

- If we flag the top N% of patients for reflex broad sequencing, what fraction
  of flagged patients truly carry the alteration (PPV), and what fraction of
  true carriers do we catch (sensitivity)?
- Is acting on the model better than the two trivial strategies -- sequence
  everybody, sequence nobody? That is a net-benefit question, not an AUROC one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "operating_points",
    "net_benefit",
    "decision_curve",
    "clinical_summary",
]


def _as_1d(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    return y, p


def operating_points(
    y_true, y_prob, flag_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.50)
) -> pd.DataFrame:
    """PPV / sensitivity / lift when flagging the top ``q`` fraction by score.

    Expressed as a budget ("we can afford to reflex-sequence 10% of patients")
    rather than as a probability cutoff, because that is the form the decision
    actually takes and it is comparable across models regardless of calibration.
    """
    y, p = _as_1d(y_true, y_prob)
    n = y.size
    prevalence = y.mean()
    order = np.argsort(-p, kind="stable")
    y_sorted = y[order]

    rows = []
    for q in flag_fractions:
        k = max(1, int(round(q * n)))
        flagged = y_sorted[:k]
        tp = float(flagged.sum())
        ppv = tp / k
        rows.append(
            {
                "flag_fraction": q,
                "n_flagged": k,
                "ppv": ppv,
                "sensitivity": tp / y.sum() if y.sum() else np.nan,
                "lift_over_prevalence": ppv / prevalence if prevalence > 0 else np.nan,
                "prevalence": prevalence,
            }
        )
    return pd.DataFrame(rows)


def net_benefit(y_true, y_prob, threshold: float) -> float:
    """Net benefit at a threshold probability (Vickers & Elkin decision curve).

    ``threshold`` encodes the clinical trade-off: it is the probability at which
    a decision-maker is indifferent between acting and not acting, so a
    threshold of 0.05 means one true positive is worth 19 false positives.

        NB = TP/n - FP/n * (t / (1 - t))
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    y, p = _as_1d(y_true, y_prob)
    n = y.size
    flag = p >= threshold
    tp = float((flag & (y == 1)).sum())
    fp = float((flag & (y == 0)).sum())
    return tp / n - (fp / n) * (threshold / (1 - threshold))


def decision_curve(
    y_true, y_prob, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """Net benefit of the model against 'act on everyone' and 'act on no one'.

    A model is only worth deploying where its curve sits above *both*
    references. At low prevalence that window is often narrow or empty even for
    a model with a respectable AUROC -- which is precisely the thing AUROC
    cannot tell you.
    """
    y, p = _as_1d(y_true, y_prob)
    if thresholds is None:
        thresholds = np.linspace(0.005, 0.30, 60)
    prevalence = y.mean()

    rows = []
    for t in thresholds:
        treat_all = prevalence - (1 - prevalence) * (t / (1 - t))
        model = net_benefit(y, p, t)
        rows.append(
            {
                "threshold": float(t),
                "net_benefit_model": model,
                "net_benefit_treat_all": float(treat_all),
                "net_benefit_treat_none": 0.0,
                "model_is_best": bool(model > max(treat_all, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def clinical_summary(
    y_true,
    y_prob,
    gene_names: list[str] | None = None,
    flag_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
    min_positives: int = 5,
) -> pd.DataFrame:
    """Per-gene operating points, for genes with enough positives to be meaningful.

    Report this restricted to actionable genes (e.g. OncoKB Level 1/2 among the
    164). A mean over 164 genes of mixed clinical relevance does not tell an
    oncologist anything they can act on.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    names = list(gene_names) if gene_names is not None else [
        f"gene_{i}" for i in range(y.shape[1])
    ]

    frames = []
    for j, name in enumerate(names):
        if y[:, j].sum() < min_positives:
            continue
        table = operating_points(y[:, j], p[:, j], flag_fractions)
        table.insert(0, "gene", name)
        table.insert(1, "n_positives", int(y[:, j].sum()))
        frames.append(table)
    if not frames:
        return pd.DataFrame(
            columns=["gene", "n_positives", "flag_fraction", "n_flagged", "ppv",
                     "sensitivity", "lift_over_prevalence", "prevalence"]
        )
    return pd.concat(frames, ignore_index=True)
