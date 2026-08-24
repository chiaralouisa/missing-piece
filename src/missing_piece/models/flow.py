"""Conditional generative models for panel completion, based on flow matching.

Two formulations, because the choice interacts strongly with how the result is
scored:

:class:`GaussianFlowMatching`
    Classic conditional flow matching. Binary calls are lifted to +-1, a
    straight-line probability path is built from Gaussian noise to the data,
    and a velocity field is regressed with MSE. Marginal probabilities have to
    be *estimated* by integrating many trajectories, so the probability
    returned is a Monte-Carlo estimate whose resolution is bounded by the
    number of samples. That quantisation creates ties, and ties cost AUROC --
    a measurement artefact, not a modelling failure. Use ``readout="expected"``
    to avoid it where possible.

:class:`DiscreteFlowMatching`
    Flow matching directly on the discrete state space {0, 1, MASK}. The
    interpolant unmasks each gene independently at a rate set by the flow time,
    and the network is trained to denoise masked positions. This has a property
    the Gaussian version lacks: at t = 0 every gene is masked, so a single
    forward pass returns the exact conditional marginals P(gene | observed
    panel) with no sampling noise -- while ancestral unmasking still yields
    coherent joint profiles. For sparse binary genomics it is both the better
    estimator and the cheaper one.

Both support classifier-free guidance: the conditioning vector is dropped for a
fraction of training steps, letting sampling interpolate between the
conditional and unconditional fields.
"""

from __future__ import annotations

import copy
import logging
from typing import Literal

import numpy as np
import torch
from torch import nn

from ..data.cohort import Cohort
from .base import FeatureSpec, PanelCompletionModel
from .nets import TimeConditionedMLP

log = logging.getLogger(__name__)

__all__ = ["GaussianFlowMatching", "DiscreteFlowMatching"]

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class _FlowBase(PanelCompletionModel):
    """Shared plumbing: features, conditioning dropout, training loop scaffolding."""

    is_generative = True

    def __init__(
        self,
        width: int = 512,
        depth: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 300,
        batch_size: int = 128,
        patience: int = 30,
        cond_dropout: float = 0.1,
        guidance_scale: float = 1.0,
        feature_spec: FeatureSpec | None = None,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.cond_dropout = cond_dropout
        self.guidance_scale = guidance_scale
        self.feature_spec = feature_spec or FeatureSpec()
        self.seed = seed
        self.device = device
        self.net_: nn.Module | None = None
        self.k_: int = 0
        self.prior_: np.ndarray | None = None

    # -- conditioning ----------------------------------------------------
    def _cond(self, cohort: Cohort) -> torch.Tensor:
        X = self.feature_spec.transform(cohort)
        return torch.as_tensor(X, dtype=torch.float32, device=self.device)

    def _maybe_drop_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Randomly blank whole conditioning vectors (classifier-free guidance)."""
        if self.cond_dropout <= 0:
            return cond
        keep = (
            torch.rand(cond.shape[0], 1, device=cond.device) >= self.cond_dropout
        ).float()
        return cond * keep

    def _loss(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _fit_loop(self, train: Cohort, val: Cohort | None) -> None:
        torch.manual_seed(self.seed)
        y = torch.as_tensor(
            train.target.to_numpy(dtype=float), dtype=torch.float32, device=self.device
        )
        cond = self._cond(train)
        if val is not None:
            yv = torch.as_tensor(
                val.target.to_numpy(dtype=float), dtype=torch.float32, device=self.device
            )
            condv = self._cond(val)

        opt = torch.optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        best = float("inf")
        best_state = copy.deepcopy(self.net_.state_dict())
        bad = 0
        n = y.shape[0]

        for epoch in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = self._loss(y[idx], self._maybe_drop_cond(cond[idx]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()

            if val is None:
                continue
            self.net_.eval()
            with torch.no_grad():
                # Average several noise draws: the flow objective is stochastic
                # in t (and in the mask), so a single pass is too noisy to early
                # stop on reliably.
                v = float(np.mean([float(self._loss(yv, condv)) for _ in range(8)]))
            if v < best - 1e-6:
                best, best_state, bad = v, copy.deepcopy(self.net_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if val is not None:
            self.net_.load_state_dict(best_state)


class GaussianFlowMatching(_FlowBase):
    """Conditional flow matching in a continuous relaxation of the binary block."""

    name = "flow_gaussian"

    def __init__(
        self,
        *,
        n_sampling_steps: int = 50,
        n_samples_for_marginals: int = 128,
        readout: Literal["expected", "mc"] = "expected",
        dequantization_noise: float = 0.0,
        sigma_min: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_sampling_steps = n_sampling_steps
        self.n_samples_for_marginals = n_samples_for_marginals
        self.readout = readout
        self.dequantization_noise = dequantization_noise
        self.sigma_min = sigma_min

    def fit(self, train: Cohort, val: Cohort | None = None) -> "GaussianFlowMatching":
        self.feature_spec.fit(train)
        cond_dim = self.feature_spec.transform(train).shape[1]
        self.k_ = train.target.shape[1]
        self.net_ = TimeConditionedMLP(
            state_dim=self.k_,
            cond_dim=cond_dim,
            width=self.width,
            depth=self.depth,
            dropout=self.dropout,
        ).to(self.device)
        self._fit_loop(train, val)
        return self

    def _loss(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # {0,1} -> {-1,+1}; optional dequantisation softens the atoms.
        x1 = 2.0 * y - 1.0
        if self.dequantization_noise > 0:
            x1 = x1 + self.dequantization_noise * torch.randn_like(x1)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1, device=x1.device)
        # Straight-line (optimal-transport) conditional path.
        x_t = (1 - (1 - self.sigma_min) * t) * x0 + t * x1
        target_v = x1 - (1 - self.sigma_min) * x0
        return nn.functional.mse_loss(self.net_(x_t, t.squeeze(-1), cond), target_v)

    def _velocity(self, x_t, t, cond) -> torch.Tensor:
        v = self.net_(x_t, t, cond)
        if self.guidance_scale != 1.0:
            v_uncond = self.net_(x_t, t, torch.zeros_like(cond))
            v = v_uncond + self.guidance_scale * (v - v_uncond)
        return v

    @torch.no_grad()
    def _integrate(self, cond: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Heun-integrate the ODE from noise at t=0 to data at t=1."""
        n = cond.shape[0]
        x = torch.randn(n, self.k_, device=self.device, generator=generator)
        ts = torch.linspace(0, 1, self.n_sampling_steps + 1, device=self.device)
        for i in range(self.n_sampling_steps):
            t0, t1 = ts[i], ts[i + 1]
            dt = t1 - t0
            v0 = self._velocity(x, t0.repeat(n), cond)
            x_euler = x + dt * v0
            if i == self.n_sampling_steps - 1:
                x = x_euler
            else:
                v1 = self._velocity(x_euler, t1.repeat(n), cond)
                x = x + dt * 0.5 * (v0 + v1)
        return x

    @torch.no_grad()
    def predict_proba(self, cohort: Cohort) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("call fit() first")
        self.net_.eval()
        cond = self._cond(cohort)
        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        acc = torch.zeros(cohort.n_patients, self.k_, device=self.device)
        for _ in range(self.n_samples_for_marginals):
            x1 = self._integrate(cond, gen)
            if self.readout == "mc":
                acc += (x1 > 0).float()
            else:
                # Expectation readout: average the continuous endpoint, which
                # is an unbiased estimate of E[x1] = 2 P(alt) - 1 and does not
                # quantise the probability to multiples of 1/n_samples.
                acc += x1.clamp(-1.0, 1.0)

        mean = acc / self.n_samples_for_marginals
        probs = mean if self.readout == "mc" else 0.5 * (mean + 1.0)
        return np.clip(probs.cpu().numpy(), _EPS, 1 - _EPS)

    @torch.no_grad()
    def sample(self, cohort: Cohort, n_samples: int = 1, seed: int = 0) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("call fit() first")
        self.net_.eval()
        cond = self._cond(cohort)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        out = [
            (self._integrate(cond, gen) > 0).float().cpu().numpy()
            for _ in range(n_samples)
        ]
        return np.stack(out)


class DiscreteFlowMatching(_FlowBase):
    """Masked discrete flow matching over {0, 1, MASK} per target gene.

    The interpolant reveals each gene independently with probability t, so the
    network learns to denoise arbitrary partial profiles. At t = 0 nothing is
    revealed, which makes a single forward pass an exact, noise-free estimate
    of the conditional marginals.
    """

    name = "flow_discrete"

    def __init__(
        self,
        *,
        n_sampling_steps: int = 64,
        loss_weighting: Literal["elbo", "uniform"] = "uniform",
        t_min: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_sampling_steps = n_sampling_steps
        self.loss_weighting = loss_weighting
        self.t_min = t_min

    def fit(self, train: Cohort, val: Cohort | None = None) -> "DiscreteFlowMatching":
        self.feature_spec.fit(train)
        cond_dim = self.feature_spec.transform(train).shape[1]
        Y = train.target.to_numpy(dtype=float)
        self.k_ = Y.shape[1]
        self.prior_ = (Y.sum(axis=0) + 0.5) / (Y.shape[0] + 1.0)
        self.net_ = TimeConditionedMLP(
            # State encoding is 2 channels per gene: revealed value, mask flag.
            state_dim=2 * self.k_,
            cond_dim=cond_dim,
            width=self.width,
            depth=self.depth,
            dropout=self.dropout,
            out_dim=self.k_,
            output_bias_init=_logit(self.prior_),
        ).to(self.device)
        self._fit_loop(train, val)
        return self

    @staticmethod
    def _encode(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """mask == 1 means 'still masked' (value hidden)."""
        revealed = y * (1 - mask)
        return torch.cat([revealed, mask], dim=-1)

    def _loss(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        n = y.shape[0]
        t = torch.rand(n, 1, device=y.device) * (1 - self.t_min) + self.t_min
        mask = (torch.rand_like(y) >= t).float()  # revealed w.p. t
        logits = self.net_(self._encode(y, mask), t.squeeze(-1), cond)
        per_elem = nn.functional.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        )
        if self.loss_weighting == "elbo":
            # Masked-diffusion ELBO weight; clipped because 1/(1-t) diverges.
            per_elem = per_elem * (1.0 / (1.0 - t).clamp(min=0.05))
        masked = per_elem * mask
        return masked.sum() / mask.sum().clamp(min=1.0)

    def _logits(self, state, t, cond) -> torch.Tensor:
        logits = self.net_(state, t, cond)
        if self.guidance_scale != 1.0:
            logits_uncond = self.net_(state, t, torch.zeros_like(cond))
            logits = logits_uncond + self.guidance_scale * (logits - logits_uncond)
        return logits

    @torch.no_grad()
    def predict_proba(self, cohort: Cohort) -> np.ndarray:
        """Exact conditional marginals: one forward pass at t = 0, fully masked."""
        if self.net_ is None:
            raise RuntimeError("call fit() first")
        self.net_.eval()
        cond = self._cond(cohort)
        n = cond.shape[0]
        mask = torch.ones(n, self.k_, device=self.device)
        y0 = torch.zeros(n, self.k_, device=self.device)
        logits = self._logits(
            self._encode(y0, mask), torch.zeros(n, device=self.device), cond
        )
        return np.clip(torch.sigmoid(logits).cpu().numpy(), _EPS, 1 - _EPS)

    @torch.no_grad()
    def sample(self, cohort: Cohort, n_samples: int = 1, seed: int = 0) -> np.ndarray:
        """Ancestral unmasking, which couples the genes and gives joint profiles."""
        if self.net_ is None:
            raise RuntimeError("call fit() first")
        self.net_.eval()
        cond = self._cond(cohort)
        n = cond.shape[0]
        gen = torch.Generator(device=self.device).manual_seed(seed)
        ts = torch.linspace(0, 1, self.n_sampling_steps + 1, device=self.device)

        draws = []
        for _ in range(n_samples):
            y = torch.zeros(n, self.k_, device=self.device)
            mask = torch.ones(n, self.k_, device=self.device)
            for i in range(self.n_sampling_steps):
                t0, t1 = ts[i], ts[i + 1]
                logits = self._logits(self._encode(y, mask), t0.repeat(n), cond)
                probs = torch.sigmoid(logits)
                # Probability that a still-masked coordinate is revealed in
                # this step, under the independent-unmasking interpolant.
                reveal_p = ((t1 - t0) / (1 - t0).clamp(min=1e-6)).clamp(0.0, 1.0)
                reveal = (
                    torch.rand(y.shape, device=self.device, generator=gen) < reveal_p
                ).float() * mask
                value = (
                    torch.rand(y.shape, device=self.device, generator=gen) < probs
                ).float()
                y = y * (1 - reveal) + value * reveal
                mask = mask * (1 - reveal)
            if mask.any():  # reveal any stragglers at t = 1
                logits = self._logits(
                    self._encode(y, mask), torch.ones(n, device=self.device), cond
                )
                value = (
                    torch.rand(y.shape, device=self.device, generator=gen)
                    < torch.sigmoid(logits)
                ).float()
                y = y * (1 - mask) + value * mask
            draws.append(y.cpu().numpy())
        return np.stack(draws)
