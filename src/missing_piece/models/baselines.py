"""Baselines that a panel-completion result has to beat to mean anything.

The ladder is deliberate. Each rung removes one explanation for a good score:

``PrevalenceBaseline``
    Ignores the patient. Its pooled AUROC is the floor any pooled number must
    be read against; its macro AUROC is 0.5 by construction.
``BurdenBaseline``
    Sees only *how many* observed genes are altered, never *which*. If a model
    barely beats this, the panel is not carrying gene-specific information --
    it is carrying mutational burden, and the finding reduces to "hypermutated
    tumours have more alterations everywhere".
``LogisticBaseline``
    Sees the full observed panel, one independent L2 logistic regression per
    target gene. A well-specified linear read of co-alteration structure.
``MLPBaseline``
    Shared-trunk multi-label network -- the discriminative ceiling. A
    generative model that does not reach it is not being helped by being
    generative on per-gene AUROC, and should be justified on joint metrics.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..data.cohort import Cohort
from .base import FeatureSpec, PanelCompletionModel

log = logging.getLogger(__name__)

__all__ = [
    "PrevalenceBaseline",
    "BurdenBaseline",
    "LogisticBaseline",
    "MLPBaseline",
]

_EPS = 1e-6


class PrevalenceBaseline(PanelCompletionModel):
    """Predict each gene's training prevalence for every patient."""

    name = "prevalence"

    def __init__(self, smoothing: float = 0.5):
        #: Laplace-style smoothing keeps genes with zero training positives from
        #: producing exact ties at 0, which AUROC would resolve arbitrarily.
        self.smoothing = smoothing
        self.prevalence_: np.ndarray | None = None

    def fit(self, train: Cohort, val: Cohort | None = None) -> "PrevalenceBaseline":
        y = train.target.to_numpy(dtype=float)
        n = y.shape[0]
        self.prevalence_ = (y.sum(axis=0) + self.smoothing) / (n + 2 * self.smoothing)
        return self

    def predict_proba(self, cohort: Cohort) -> np.ndarray:
        if self.prevalence_ is None:
            raise RuntimeError("call fit() first")
        return np.tile(self.prevalence_, (cohort.n_patients, 1))


class BurdenBaseline(PanelCompletionModel):
    """One logistic regression per target gene on observed-panel burden only.

    The only feature is the number of altered genes on the small panel (plus
    optional clinical burden proxies such as TMB). This is the ablation that
    separates "the panel predicts the genome" from "burden predicts burden".
    """

    name = "burden"

    def __init__(
        self,
        C: float = 1.0,
        extra_covariates: tuple[str, ...] = (),
        max_iter: int = 500,
    ):
        self.C = C
        self.extra_covariates = extra_covariates
        self.max_iter = max_iter
        self.models_: list[LogisticRegression | float] = []
        self._columns: tuple[str, ...] = ()

    def _features(self, cohort: Cohort) -> np.ndarray:
        cols = ("log1p_n_altered_observed", *self.extra_covariates)
        missing = [c for c in cols if c not in cohort.covariates.columns]
        if missing:
            raise KeyError(f"burden covariates {missing} missing from cohort")
        self._columns = cols
        return cohort.covariates[list(cols)].to_numpy(dtype=float)

    def fit(self, train: Cohort, val: Cohort | None = None) -> "BurdenBaseline":
        X = self._features(train)
        Y = train.target.to_numpy(dtype=float)
        self.models_ = _fit_per_gene_logistic(X, Y, C=self.C, max_iter=self.max_iter)
        return self

    def predict_proba(self, cohort: Cohort) -> np.ndarray:
        return _predict_per_gene_logistic(self.models_, self._features(cohort))


class LogisticBaseline(PanelCompletionModel):
    """Independent L2 logistic regression per target gene on the full panel."""

    name = "logistic"

    def __init__(
        self,
        C: float = 0.05,
        feature_spec: FeatureSpec | None = None,
        max_iter: int = 300,
    ):
        self.C = C
        self.feature_spec = feature_spec or FeatureSpec()
        self.max_iter = max_iter
        self.models_: list[LogisticRegression | float] = []

    def fit(self, train: Cohort, val: Cohort | None = None) -> "LogisticBaseline":
        self.feature_spec.fit(train)
        X = self.feature_spec.transform(train)
        Y = train.target.to_numpy(dtype=float)
        self.models_ = _fit_per_gene_logistic(X, Y, C=self.C, max_iter=self.max_iter)
        return self

    def predict_proba(self, cohort: Cohort) -> np.ndarray:
        return _predict_per_gene_logistic(
            self.models_, self.feature_spec.transform(cohort)
        )


def _fit_per_gene_logistic(
    X: np.ndarray, Y: np.ndarray, C: float, max_iter: int
) -> list:
    """Fit one logistic model per column of Y, falling back to prevalence."""
    models: list = []
    n = Y.shape[0]
    for j in range(Y.shape[1]):
        y = Y[:, j]
        n_pos = int(y.sum())
        # Below two positives the fit is degenerate; a smoothed constant is the
        # honest prediction and keeps the model's output well defined.
        if n_pos < 2 or n_pos > n - 2:
            models.append(float((n_pos + 0.5) / (n + 1.0)))
            continue
        clf = LogisticRegression(
            C=C, max_iter=max_iter, solver="lbfgs", class_weight=None
        )
        clf.fit(X, y)
        models.append(clf)
    return models


def _predict_per_gene_logistic(models: list, X: np.ndarray) -> np.ndarray:
    if not models:
        raise RuntimeError("call fit() first")
    out = np.empty((X.shape[0], len(models)))
    for j, model in enumerate(models):
        if isinstance(model, float):
            out[:, j] = model
        else:
            out[:, j] = model.predict_proba(X)[:, 1]
    return np.clip(out, _EPS, 1 - _EPS)


class MLPBaseline(PanelCompletionModel):
    """Shared-trunk multi-label MLP trained with binary cross-entropy.

    The discriminative reference point: it optimises exactly the per-gene
    marginals that AUROC scores, so it marks the ceiling for that metric.
    """

    name = "mlp"

    def __init__(
        self,
        hidden: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        batch_size: int = 128,
        patience: int = 20,
        feature_spec: FeatureSpec | None = None,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.feature_spec = feature_spec or FeatureSpec()
        self.seed = seed
        self.device = device
        self.net_ = None

    def fit(self, train: Cohort, val: Cohort | None = None) -> "MLPBaseline":
        import torch
        from torch import nn

        from .nets import ResidualMLP, train_supervised

        torch.manual_seed(self.seed)
        self.feature_spec.fit(train)
        X = self.feature_spec.transform(train)
        Y = train.target.to_numpy(dtype=float)

        # Initialise output biases at the training log-odds so the network
        # starts calibrated; at ~1% prevalence a zero-init head spends many
        # epochs just learning the base rate.
        prior = (Y.sum(axis=0) + 0.5) / (Y.shape[0] + 1.0)
        self.net_ = ResidualMLP(
            in_dim=X.shape[1],
            out_dim=Y.shape[1],
            hidden=self.hidden,
            dropout=self.dropout,
            output_bias_init=np.log(prior / (1 - prior)),
        ).to(self.device)

        val_arrays = None
        if val is not None:
            val_arrays = (
                self.feature_spec.transform(val),
                val.target.to_numpy(dtype=float),
            )
        train_supervised(
            self.net_,
            X,
            Y,
            val=val_arrays,
            lr=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            batch_size=self.batch_size,
            patience=self.patience,
            device=self.device,
        )
        return self

    def predict_proba(self, cohort: Cohort) -> np.ndarray:
        import torch

        if self.net_ is None:
            raise RuntimeError("call fit() first")
        X = torch.as_tensor(
            self.feature_spec.transform(cohort), dtype=torch.float32, device=self.device
        )
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(X)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.clip(probs, _EPS, 1 - _EPS)
