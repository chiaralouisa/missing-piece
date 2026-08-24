"""Common interface for panel-completion models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.cohort import Cohort

__all__ = ["FeatureSpec", "ModelInputs", "PanelCompletionModel"]


@dataclass
class FeatureSpec:
    """How a cohort is turned into a model input matrix.

    Only information available from the *small* panel may enter the features.
    Covariates come from the clinical record and from burden computed over the
    observed genes -- never from the target block.
    """

    use_observed_genes: bool = True
    use_covariates: bool = True
    covariate_columns: tuple[str, ...] | None = None
    standardize_covariates: bool = True

    _cov_mean: np.ndarray | None = None
    _cov_scale: np.ndarray | None = None
    _cov_names: tuple[str, ...] = ()

    def fit(self, cohort: Cohort) -> "FeatureSpec":
        if self.use_covariates:
            cov = self._select_covariates(cohort)
            self._cov_names = tuple(cov.columns)
            values = cov.to_numpy(dtype=float)
            self._cov_mean = values.mean(axis=0)
            scale = values.std(axis=0)
            scale[scale < 1e-8] = 1.0
            self._cov_scale = scale
        return self

    def _select_covariates(self, cohort: Cohort) -> pd.DataFrame:
        cov = cohort.covariates
        if self.covariate_columns is not None:
            missing = [c for c in self.covariate_columns if c not in cov.columns]
            if missing:
                raise KeyError(f"covariates {missing} not present in cohort")
            cov = cov[list(self.covariate_columns)]
        return cov

    def transform(self, cohort: Cohort) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.use_observed_genes:
            blocks.append(cohort.observed.to_numpy(dtype=float))
        if self.use_covariates:
            cov = self._select_covariates(cohort)
            if self._cov_names and tuple(cov.columns) != self._cov_names:
                cov = cov[list(self._cov_names)]
            values = cov.to_numpy(dtype=float)
            if self.standardize_covariates and self._cov_mean is not None:
                values = (values - self._cov_mean) / self._cov_scale
            blocks.append(values)
        if not blocks:
            raise ValueError("FeatureSpec produces no features")
        return np.concatenate(blocks, axis=1)

    def feature_names(self, cohort: Cohort) -> list[str]:
        names: list[str] = []
        if self.use_observed_genes:
            names += [f"obs:{g}" for g in cohort.observed.columns]
        if self.use_covariates:
            names += [f"cov:{c}" for c in self._select_covariates(cohort).columns]
        return names


@dataclass
class ModelInputs:
    """Arrays handed to a model."""

    X: np.ndarray           # (n, d) features from the observed panel
    Y: np.ndarray | None    # (n, k) binary target block, None at inference
    gene_names: tuple[str, ...] = ()

    @classmethod
    def from_cohort(cls, cohort: Cohort, spec: FeatureSpec, with_labels: bool = True):
        return cls(
            X=spec.transform(cohort),
            Y=cohort.target.to_numpy(dtype=float) if with_labels else None,
            gene_names=tuple(cohort.target.columns),
        )


class PanelCompletionModel(ABC):
    """Predict alterations in unassayed genes from an observed panel."""

    name: str = "model"
    #: True when the model defines a joint distribution over the target block
    #: and can draw coherent multi-gene profiles, not just per-gene marginals.
    is_generative: bool = False

    @abstractmethod
    def fit(
        self, train: Cohort, val: Cohort | None = None
    ) -> "PanelCompletionModel":
        """Fit on the training cohort, optionally using ``val`` for early stopping."""

    @abstractmethod
    def predict_proba(self, cohort: Cohort) -> np.ndarray:
        """Marginal P(altered) per patient per target gene, shape (n, k)."""

    def sample(self, cohort: Cohort, n_samples: int = 1, seed: int = 0) -> np.ndarray:
        """Draw joint binary profiles, shape (n_samples, n, k).

        The default draws independently from the predicted marginals, which is
        the correct behaviour for a discriminative model and the right null for
        judging whether a generative model captures joint structure.
        """
        probs = self.predict_proba(cohort)
        rng = np.random.default_rng(seed)
        return (rng.random((n_samples, *probs.shape)) < probs).astype(float)

    def describe(self) -> dict:
        return {"name": self.name, "is_generative": self.is_generative}
