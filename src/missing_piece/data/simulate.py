"""Synthetic cohorts for validating the panel-completion pipeline.

The simulator exists to answer a question the real data cannot: *would our
evaluation report success on a cohort where the hypothesis is false?* It
generates cohorts under three regimes:

``full``
    Observed and target genes share latent biology (subtype, co-mutation
    factors) on top of mutational burden. The hypothesis is TRUE: the small
    panel carries gene-specific information about the unassayed genes.
``burden_only``
    Observed and target genes are coupled *only* through total mutational
    burden. Every target gene is conditionally independent of *which* observed
    genes are altered. Panel completion should reduce to "hypermutated tumours
    have more of everything" -- an honest evaluation must show near-zero
    gene-specific lift over a burden-only baseline.
``independent``
    Target genes are independent of the patient entirely. Per-gene AUROC must
    come out at 0.5. Anything above that is a bug or a leak.

``burden_only`` and ``independent`` are negative controls. A metric that scores
well on them is measuring the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from ..panels import Panel, PanelPair, make_panel
from .cohort import Cohort

__all__ = ["SimulationConfig", "simulate_cohort", "simulate_panel_pair"]

Regime = Literal["full", "burden_only", "independent"]


@dataclass
class SimulationConfig:
    """Parameters of the synthetic cohort generator."""

    n_patients: int = 2226
    n_observed_genes: int = 341
    n_target_genes: int = 164
    regime: Regime = "full"

    #: Mean alteration rate among target genes (the abstract reports 1.21%).
    target_sparsity: float = 0.0121
    #: Mean alteration rate among observed-panel genes. Small panels are
    #: enriched for recurrently altered genes, so this is higher.
    observed_sparsity: float = 0.035

    #: Number of shared latent co-alteration factors (subtype-independent).
    n_factors: int = 6
    #: Strength of the shared factors -- the genuine cross-panel signal.
    factor_scale: float = 1.1
    #: Fraction of target genes that load on the shared factors at all.
    factor_active_fraction: float = 0.55

    #: Molecular subtypes and their cohort frequencies.
    subtype_names: tuple[str, ...] = ("EGFR_like", "KRAS_smoking", "squamous", "other")
    subtype_weights: tuple[float, ...] = (0.26, 0.38, 0.22, 0.14)
    subtype_scale: float = 1.3

    #: Spread of per-patient mutational burden (log-normal sigma).
    burden_sigma: float = 0.55
    #: Per-gene sensitivity to burden.
    burden_coef_mean: float = 0.85
    burden_coef_sd: float = 0.35

    #: Prevalence heterogeneity across genes (log-normal on the odds scale).
    prevalence_sigma: float = 1.15

    #: Chromosome arms. Copy-number events are arm-scale, so genes on the same
    #: arm are co-altered for reasons of position rather than biology. This is a
    #: real confound in panel completion: an arm-level event seen on the small
    #: panel predicts genes on the same arm off it, without any co-mutation.
    n_arms: int = 39                      # autosomal p/q arms
    arm_effect_scale: float = 0.9
    arm_active_fraction: float = 0.45     # genes whose alteration is arm-driven

    #: Mutually exclusive driver groups (EGFR / KRAS / ALK style). Within a
    #: group at most one gene is altered per patient, which is the single most
    #: characteristic structure in an NSCLC alteration matrix.
    n_exclusivity_groups: int = 8
    exclusivity_group_size: int = 4
    exclusivity_strength: float = 0.9     # P(constraint enforced) per patient

    seed: int = 0
    gene_prefix: str = "SIMG"

    def __post_init__(self) -> None:
        if len(self.subtype_names) != len(self.subtype_weights):
            raise ValueError("subtype_names and subtype_weights must align")
        if not np.isclose(sum(self.subtype_weights), 1.0):
            raise ValueError("subtype_weights must sum to 1")
        if not 0 < self.target_sparsity < 1:
            raise ValueError("target_sparsity must be in (0, 1)")


def _draw_prevalences(
    rng: np.random.Generator, n_genes: int, mean_prevalence: float, sigma: float
) -> np.ndarray:
    """Heavy-tailed per-gene prevalences with a controlled mean.

    Panel genes are not uniformly rare: a handful (TP53, KRAS, ...) sit in the
    tens of percent while the tail sits well under 1%. A log-normal on the odds
    scale reproduces that shape; we then rescale to hit the requested mean.
    """
    raw = rng.lognormal(mean=0.0, sigma=sigma, size=n_genes)
    raw = raw / raw.mean() * mean_prevalence
    return np.clip(raw, 1.0 / 5000.0, 0.6)


def _calibrate_intercepts(
    logits_without_intercept: np.ndarray, target_prevalence: np.ndarray
) -> np.ndarray:
    """Solve per gene for the intercept giving the requested marginal rate.

    ``mean_i sigmoid(a_g + eta_ig) = p_g`` is monotone in ``a_g``, so bisection
    converges quickly and exactly.
    """
    n_genes = logits_without_intercept.shape[1]
    lo = np.full(n_genes, -30.0)
    hi = np.full(n_genes, 30.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        rate = _sigmoid(logits_without_intercept + mid).mean(axis=0)
        too_high = rate > target_prevalence
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return 0.5 * (lo + hi)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def simulate_panel_pair(cfg: SimulationConfig) -> PanelPair:
    """Synthetic nested panel pair matching the configured sizes."""
    width = max(4, len(str(cfg.n_observed_genes + cfg.n_target_genes)))
    all_genes = [
        f"{cfg.gene_prefix}{i:0{width}d}"
        for i in range(1, cfg.n_observed_genes + cfg.n_target_genes + 1)
    ]
    observed = make_panel(
        "SIM_SMALL",
        all_genes[: cfg.n_observed_genes],
        description=f"Simulated small panel ({cfg.n_observed_genes} genes)",
    )
    target = make_panel(
        "SIM_LARGE",
        all_genes,
        description=f"Simulated large panel ({len(all_genes)} genes)",
    )
    return PanelPair(observed=observed, target=target)


def _apply_mutual_exclusivity(
    draws: np.ndarray,
    probs: np.ndarray,
    cfg: SimulationConfig,
    rng: np.random.Generator,
    n_obs: int,
) -> list[list[int]]:
    """Force at most one altered gene per driver group, in place.

    Mutual exclusivity is the signature of a driver landscape: a tumour driven
    by EGFR is not also driven by KRAS. It shows up as *negative* co-occurrence,
    which a model can exploit -- observing an activated driver on the small
    panel argues against the drivers it does not cover.

    Under the negative-control regimes the groups are confined to the observed
    block, so the exclusivity cannot leak cross-panel information.
    """
    n_all = draws.shape[1]
    span_target = cfg.regime == "full"
    groups: list[list[int]] = []
    if cfg.n_exclusivity_groups <= 0 or cfg.exclusivity_group_size < 2:
        return groups

    # Prefer the commonest genes as drivers, which is where exclusivity lives.
    ranked_obs = np.argsort(-probs[:, :n_obs].mean(axis=0))
    ranked_tgt = np.argsort(-probs[:, n_obs:].mean(axis=0)) + n_obs
    obs_pool, tgt_pool = list(ranked_obs), list(ranked_tgt)

    # The observed members are drawn identically in every regime, and the
    # target member is only ever *appended*. Substituting another observed gene
    # in the control regimes would consume the observed pool at a different
    # rate, making the observed block differ between regimes -- and the controls
    # are only interpretable if the model inputs are held fixed.
    n_observed_members = max(2, cfg.exclusivity_group_size - 1)
    for _ in range(cfg.n_exclusivity_groups):
        group: list[int] = []
        for _ in range(n_observed_members):
            if not obs_pool:
                break
            group.append(int(obs_pool.pop(0)))
        if span_target and tgt_pool:
            group.append(int(tgt_pool.pop(0)))
        if len(group) >= 2:
            groups.append(group)

    # Pass 1 -- exclusivity *within the observed panel*. Identical in every
    # regime, because the observed membership and the draw order both are.
    for group in groups:
        idx = np.array([g for g in group if g < n_obs])
        if idx.size < 2:
            continue
        _enforce_one_of(draws, probs, idx, cfg.exclusivity_strength, rng)

    # Pass 2 -- cross-panel exclusivity, applied in one direction only: an
    # observed driver suppresses its target-panel partner, never the reverse.
    # Clearing the observed gene instead would make the model's *input* depend
    # on the regime, and the negative controls are only interpretable while the
    # inputs are held fixed.
    for group in groups:
        tgt = [g for g in group if g >= n_obs]
        obs = [g for g in group if g < n_obs]
        if not tgt or not obs:
            continue
        driver_present = draws[:, obs].any(axis=1)
        suppress = driver_present & (rng.random(draws.shape[0]) < cfg.exclusivity_strength)
        for t in tgt:
            draws[suppress, t] = False

    return groups


def _enforce_one_of(
    draws: np.ndarray,
    probs: np.ndarray,
    idx: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> None:
    """Keep at most one altered gene among ``idx``, weighted by its probability."""
    block = draws[:, idx]
    multi = block.sum(axis=1) > 1
    enforce = multi & (rng.random(draws.shape[0]) < strength)
    rows = np.flatnonzero(enforce)
    if rows.size == 0:
        return
    weights = block[rows] * probs[np.ix_(rows, idx)]
    totals = weights.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    cumulative = np.cumsum(weights / totals, axis=1)
    picks = np.clip((cumulative < rng.random((rows.size, 1))).sum(axis=1), 0, idx.size - 1)
    cleared = np.zeros_like(block[rows])
    cleared[np.arange(rows.size), picks] = True
    draws[np.ix_(rows, idx)] = cleared


def simulate_cohort(
    cfg: SimulationConfig | None = None,
    panel_pair: PanelPair | None = None,
    **overrides,
) -> Cohort:
    """Generate a synthetic panel-completion cohort.

    When ``panel_pair`` is supplied (e.g. the real IMPACT341/IMPACT505 pair) the
    simulated genes take the real gene symbols and panel sizes, which makes the
    simulated study structurally identical to the real one.
    """
    cfg = cfg or SimulationConfig()
    if overrides:
        cfg = SimulationConfig(**{**cfg.__dict__, **overrides})

    if panel_pair is not None:
        observed_genes = list(panel_pair.observed_genes)
        target_genes = list(panel_pair.target_genes)
        cfg = SimulationConfig(
            **{
                **cfg.__dict__,
                "n_observed_genes": len(observed_genes),
                "n_target_genes": len(target_genes),
            }
        )
    else:
        pair = simulate_panel_pair(cfg)
        observed_genes = list(pair.observed_genes)
        target_genes = list(pair.target_genes)

    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_patients
    n_obs = len(observed_genes)
    n_tgt = len(target_genes)
    n_all = n_obs + n_tgt

    # ---- patient-level latents -------------------------------------------
    log_burden = rng.normal(0.0, cfg.burden_sigma, size=n)
    subtype = rng.choice(len(cfg.subtype_names), size=n, p=list(cfg.subtype_weights))
    factors = rng.normal(0.0, 1.0, size=(n, cfg.n_factors))
    # One arm-level state per patient per arm; genes inherit their arm's state.
    arm_state = rng.normal(0.0, 1.0, size=(n, cfg.n_arms))

    # ---- gene-level parameters -------------------------------------------
    burden_coef = np.abs(
        rng.normal(cfg.burden_coef_mean, cfg.burden_coef_sd, size=n_all)
    )

    loadings = rng.normal(0.0, cfg.factor_scale, size=(cfg.n_factors, n_all))
    active = rng.random(n_all) < cfg.factor_active_fraction
    loadings[:, ~active] = 0.0

    subtype_effect = rng.normal(
        0.0, cfg.subtype_scale, size=(len(cfg.subtype_names), n_all)
    )

    # Genes are laid out along the genome, so consecutive indices share an arm.
    gene_arm = np.floor(np.arange(n_all) / n_all * cfg.n_arms).astype(int)
    gene_arm = np.clip(gene_arm, 0, cfg.n_arms - 1)
    arm_loading = np.abs(rng.normal(0.0, cfg.arm_effect_scale, size=n_all))
    arm_loading[rng.random(n_all) >= cfg.arm_active_fraction] = 0.0

    # Negative controls sever the gene-specific channels for TARGET genes only:
    # the observed panel keeps its structure, so the input distribution is
    # unchanged and only the hypothesis under test is switched off.
    if cfg.regime in ("burden_only", "independent"):
        loadings[:, n_obs:] = 0.0
        subtype_effect[:, n_obs:] = 0.0
        arm_loading[n_obs:] = 0.0
    if cfg.regime == "independent":
        burden_coef[n_obs:] = 0.0

    eta = (
        burden_coef[None, :] * log_burden[:, None]
        + factors @ loadings
        + subtype_effect[subtype]
        + arm_loading[None, :] * arm_state[:, gene_arm]
    )

    prevalence = np.concatenate(
        [
            _draw_prevalences(rng, n_obs, cfg.observed_sparsity, cfg.prevalence_sigma),
            _draw_prevalences(rng, n_tgt, cfg.target_sparsity, cfg.prevalence_sigma),
        ]
    )
    intercepts = _calibrate_intercepts(eta, prevalence)
    probs = _sigmoid(eta + intercepts)
    draws = rng.random((n, n_all)) < probs

    exclusivity_groups = _apply_mutual_exclusivity(draws, probs, cfg, rng, n_obs)

    patients = pd.Index([f"SIM-P{i:05d}" for i in range(n)], name="SAMPLE_ID")
    observed = pd.DataFrame(draws[:, :n_obs], index=patients, columns=observed_genes)
    target = pd.DataFrame(draws[:, n_obs:], index=patients, columns=target_genes)

    clinical = pd.DataFrame(
        {
            "PATIENT_ID": [f"SIM-{i:05d}" for i in range(n)],
            "ONCOTREE_CODE": "LUAD",
            "CANCER_TYPE": "Non-Small Cell Lung Cancer",
            "SUBTYPE_TRUE": [cfg.subtype_names[s] for s in subtype],
            "LOG_BURDEN_TRUE": log_burden,
            "SAMPLE_TYPE": "Primary",
        },
        index=patients,
    )

    covariates = pd.DataFrame(index=patients)
    covariates["n_altered_observed"] = observed.sum(axis=1).astype(float)
    covariates["log1p_n_altered_observed"] = np.log1p(covariates["n_altered_observed"])

    return Cohort(
        observed=observed,
        target=target,
        covariates=covariates,
        clinical=clinical,
        provenance={
            "source": "simulation",
            "regime": cfg.regime,
            "config": {k: v for k, v in cfg.__dict__.items()},
            "realised_target_sparsity": float(target.to_numpy().mean()),
            "realised_observed_sparsity": float(observed.to_numpy().mean()),
            "n_exclusivity_groups": len(exclusivity_groups),
            "exclusivity_groups": [list(map(int, g)) for g in exclusivity_groups],
        },
    )
