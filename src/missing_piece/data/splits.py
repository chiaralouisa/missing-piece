"""Patient-level train/validation/test splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cohort import Cohort

__all__ = ["SplitSpec", "Splits", "make_splits"]


@dataclass(frozen=True)
class SplitSpec:
    """Split fractions. They must sum to 1.

    (The submitted abstract quotes 75% train / 15% validation / 15% test, which
    sums to 105%; the intended design was presumably 70/15/15. Whatever is used,
    it has to be stated exactly, so this class refuses to guess.)
    """

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 0
    stratify: bool = True
    n_strata: int = 5

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"split fractions must sum to 1, got {total:.3f} "
                f"({self.train}/{self.val}/{self.test})"
            )
        if min(self.train, self.val, self.test) <= 0:
            raise ValueError("every split fraction must be positive")


@dataclass
class Splits:
    train: Cohort
    val: Cohort
    test: Cohort
    assignment: pd.Series

    def summary(self) -> str:
        return (
            f"train {self.train.n_patients} | "
            f"val {self.val.n_patients} | "
            f"test {self.test.n_patients}\n"
            f"target sparsity  train {self.train.target_sparsity():.4%} | "
            f"val {self.val.target_sparsity():.4%} | "
            f"test {self.test.target_sparsity():.4%}"
        )


def _strata(cohort: Cohort, n_strata: int) -> np.ndarray:
    """Bin patients by observed-panel burden.

    Stratifying on the observed block (never the target block) keeps the splits
    comparable in the covariate that drives most of the predictable signal,
    without leaking the labels being predicted.
    """
    burden = cohort.observed.sum(axis=1).to_numpy(dtype=float)
    quantiles = np.quantile(burden, np.linspace(0, 1, n_strata + 1)[1:-1])
    return np.digitize(burden, quantiles)


def make_splits(cohort: Cohort, spec: SplitSpec | None = None) -> Splits:
    """Split patients into train/val/test, optionally stratified by burden."""
    spec = spec or SplitSpec()
    rng = np.random.default_rng(spec.seed)
    n = cohort.n_patients
    ids = np.asarray(cohort.patient_ids)
    assignment = np.empty(n, dtype=object)

    groups = _strata(cohort, spec.n_strata) if spec.stratify else np.zeros(n, dtype=int)

    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        rng.shuffle(idx)
        n_g = idx.size
        n_train = int(round(spec.train * n_g))
        n_val = int(round(spec.val * n_g))
        # Guarantee every split is non-empty within a stratum big enough to fill it.
        n_train = min(n_train, max(0, n_g - 2))
        n_val = min(n_val, max(0, n_g - n_train - 1))
        assignment[idx[:n_train]] = "train"
        assignment[idx[n_train : n_train + n_val]] = "val"
        assignment[idx[n_train + n_val :]] = "test"

    series = pd.Series(assignment, index=cohort.patient_ids, name="split")
    parts = {
        name: cohort.subset(list(ids[series.to_numpy() == name]))
        for name in ("train", "val", "test")
    }
    for name, part in parts.items():
        if part.n_patients == 0:
            raise ValueError(f"split '{name}' is empty; cohort too small for {spec}")
    return Splits(train=parts["train"], val=parts["val"], test=parts["test"], assignment=series)
