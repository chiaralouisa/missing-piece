"""Run a panel-completion experiment: build cohort, fit models, evaluate, report."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.cohort import NSCLC_ONCOTREE_CODES, Cohort, build_cohort
from .data.simulate import SimulationConfig, simulate_cohort
from .data.splits import SplitSpec, Splits, make_splits
from .eval.joint import evaluate_joint
from .eval.metrics import (
    EvaluationResult,
    bootstrap_metric,
    evaluate_predictions,
    macro_auroc,
    permutation_test,
    pooled_auroc,
    within_patient_auroc,
)
from .models import build_model
from .panels import PanelPair

log = logging.getLogger(__name__)

__all__ = ["ExperimentConfig", "ExperimentResult", "run_experiment"]


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce one experiment."""

    name: str = "panel_completion"

    # --- data source: exactly one of `study_dir` or `simulation` -----------
    study_dir: str | None = None
    panel_dir: str | None = None
    observed_panel: str = "IMPACT341"
    target_panel: str = "IMPACT505"
    simulation: dict[str, Any] = field(default_factory=dict)
    use_real_panels_in_simulation: bool = True

    # --- cohort -----------------------------------------------------------
    restrict_to_nsclc: bool = True
    include_cna: bool = True
    deep_cna_only: bool = True

    # --- protocol ---------------------------------------------------------
    split: dict[str, Any] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "prevalence": {},
            "burden": {},
            "logistic": {},
            "mlp": {},
            "flow_discrete": {},
        }
    )

    # --- evaluation -------------------------------------------------------
    min_positives_for_gene_auroc: int = 5
    n_bootstrap: int = 400
    n_permutations: int = 200
    n_joint_samples: int = 20
    reference_model: str = "prevalence"
    burden_reference_model: str = "burden"
    seed: int = 0
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    cohort_summary: str
    split_summary: str
    metrics: dict[str, dict]
    per_gene: pd.DataFrame
    provenance: dict

    def table(self) -> pd.DataFrame:
        """Headline metrics, one row per model, ordered by macro AUROC."""
        rows = []
        for name, m in self.metrics.items():
            rows.append(
                {
                    "model": name,
                    "macro_auroc": m.get("macro_auroc"),
                    "macro_auroc_lo": m.get("macro_auroc_ci", {}).get("lo"),
                    "macro_auroc_hi": m.get("macro_auroc_ci", {}).get("hi"),
                    "pooled_auroc": m.get("pooled_auroc"),
                    "within_patient_auroc": m.get("within_patient_auroc"),
                    "macro_ap": m.get("macro_average_precision"),
                    "brier": m.get("brier"),
                    "ece": m.get("ece"),
                    "d_macro_vs_prevalence": m.get("delta_macro_vs_reference"),
                    "d_macro_vs_burden": m.get("delta_macro_vs_burden"),
                    "perm_p": m.get("permutation_test", {}).get("p_value"),
                    "fit_seconds": m.get("fit_seconds"),
                }
            )
        table = pd.DataFrame(rows)
        return table.sort_values("macro_auroc", ascending=False, na_position="last")

    def save(self, directory: str | Path | None = None) -> Path:
        directory = Path(directory or self.config.output_dir) / self.config.name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(asdict(self.config), indent=2, default=str))
        (directory / "metrics.json").write_text(json.dumps(self.metrics, indent=2, default=str))
        (directory / "provenance.json").write_text(json.dumps(self.provenance, indent=2, default=str))
        self.table().to_csv(directory / "summary.csv", index=False)
        self.per_gene.to_csv(directory / "per_gene_auroc.csv", index=False)
        (directory / "summary.txt").write_text(self.render())
        return directory

    def render(self) -> str:
        lines = [
            f"# {self.config.name}",
            "",
            "## Cohort",
            self.cohort_summary,
            "",
            "## Splits",
            self.split_summary,
            "",
            "## Results",
            self.table().to_string(index=False, float_format=lambda v: f"{v:.4f}"),
            "",
            "macro_auroc          per-gene AUROC averaged over genes (prevalence-free; 0.5 = no signal)",
            "pooled_auroc         all patient-gene pairs in one ranking (inflated by gene prevalence)",
            "within_patient_auroc ranking target genes inside each patient",
            "d_macro_vs_burden    macro AUROC minus the burden-only baseline; this is the",
            "                     gene-specific signal that panel completion actually adds",
            "perm_p               permutation test that predictions carry patient-specific information",
        ]
        return "\n".join(lines)


def _load_cohort(cfg: ExperimentConfig) -> tuple[Cohort, dict]:
    """Build the cohort from a real study directory or from the simulator."""
    provenance: dict = {}

    if cfg.study_dir:
        from .data.cbioportal import StudyFiles, build_alteration_matrix
        from .panels import load_panels

        pair = PanelPair.from_directory(
            cfg.panel_dir or cfg.study_dir, cfg.observed_panel, cfg.target_panel
        )
        warnings = pair.validate()
        for w in warnings:
            log.warning("panel pair: %s", w)
        provenance["panel_warnings"] = warnings
        provenance["panel_pair"] = pair.describe()

        files = StudyFiles.discover(cfg.study_dir)
        panels = load_panels(cfg.panel_dir or cfg.study_dir)
        matrix = build_alteration_matrix(
            files,
            genes=list(pair.target.genes) + list(pair.observed_only_genes),
            panels=panels,
            include_cna=cfg.include_cna,
            deep_cna_only=cfg.deep_cna_only,
        )
        provenance["matrix_summary"] = matrix.summary()
        cohort = build_cohort(
            matrix,
            observed_genes=pair.observed_genes,
            target_genes=pair.target_genes,
            oncotree_codes=NSCLC_ONCOTREE_CODES if cfg.restrict_to_nsclc else None,
        )
        return cohort, provenance

    sim_cfg = SimulationConfig(**{"seed": cfg.seed, **cfg.simulation})
    pair = None
    if cfg.panel_dir and cfg.use_real_panels_in_simulation:
        try:
            pair = PanelPair.from_directory(
                cfg.panel_dir, cfg.observed_panel, cfg.target_panel
            )
            provenance["panel_pair"] = pair.describe()
        except (FileNotFoundError, KeyError, ValueError) as exc:
            log.warning("falling back to synthetic panels: %s", exc)
    cohort = simulate_cohort(sim_cfg, panel_pair=pair)
    provenance["simulation"] = cohort.provenance
    return cohort, provenance


def _evaluate_one(
    name: str,
    spec: dict,
    splits: Splits,
    cfg: ExperimentConfig,
) -> tuple[dict, np.ndarray, EvaluationResult]:
    model = build_model(name, **spec)
    t0 = time.time()
    model.fit(splits.train, splits.val)
    fit_seconds = time.time() - t0

    probs = model.predict_proba(splits.test)
    y_test = splits.test.target.to_numpy(dtype=float)
    result = evaluate_predictions(
        y_test,
        probs,
        gene_names=splits.test.target.columns,
        min_positives=cfg.min_positives_for_gene_auroc,
    )

    metrics = result.to_dict()
    metrics["fit_seconds"] = fit_seconds
    metrics["is_generative"] = model.is_generative
    metrics["params"] = spec

    if cfg.n_bootstrap:
        metrics["macro_auroc_ci"] = bootstrap_metric(
            y_test,
            probs,
            metric=lambda a, b: macro_auroc(a, b, cfg.min_positives_for_gene_auroc),
            n_boot=cfg.n_bootstrap,
            seed=cfg.seed,
        )
        metrics["pooled_auroc_ci"] = bootstrap_metric(
            y_test, probs, metric=pooled_auroc, n_boot=cfg.n_bootstrap, seed=cfg.seed
        )
    if cfg.n_permutations:
        metrics["permutation_test"] = permutation_test(
            y_test,
            probs,
            metric=lambda a, b: macro_auroc(a, b, cfg.min_positives_for_gene_auroc),
            n_permutations=cfg.n_permutations,
            seed=cfg.seed,
        )
    if cfg.n_joint_samples and model.is_generative:
        samples = model.sample(splits.test, n_samples=cfg.n_joint_samples, seed=cfg.seed)
        metrics["joint"] = evaluate_joint(y_test, samples)
    elif cfg.n_joint_samples:
        samples = model.sample(splits.test, n_samples=cfg.n_joint_samples, seed=cfg.seed)
        metrics["joint"] = evaluate_joint(y_test, samples)
        metrics["joint"]["note"] = "independent draws from predicted marginals"

    return metrics, probs, result


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Fit and evaluate every configured model on one cohort."""
    cohort, provenance = _load_cohort(cfg)
    splits = make_splits(cohort, SplitSpec(**{"seed": cfg.seed, **cfg.split}))

    log.info("cohort: %s", cohort.summary())
    log.info("splits: %s", splits.summary())

    metrics: dict[str, dict] = {}
    per_gene_frames: list[pd.DataFrame] = []

    for name, spec in cfg.models.items():
        log.info("fitting %s", name)
        m, _probs, result = _evaluate_one(name, spec, splits, cfg)
        metrics[name] = m
        per_gene_frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "gene": result.gene_names,
                    "auroc": result.per_gene_auroc,
                    "average_precision": result.per_gene_ap,
                    "test_prevalence": splits.test.target.mean(axis=0).to_numpy(),
                    "train_prevalence": splits.train.target.mean(axis=0).to_numpy(),
                }
            )
        )

    # Deltas against the two reference points that decide interpretation.
    ref = metrics.get(cfg.reference_model, {}).get("macro_auroc")
    burden_ref = metrics.get(cfg.burden_reference_model, {}).get("macro_auroc")
    ref_pooled = metrics.get(cfg.reference_model, {}).get("pooled_auroc")
    for m in metrics.values():
        if ref is not None and m.get("macro_auroc") is not None:
            m["delta_macro_vs_reference"] = m["macro_auroc"] - ref
        if burden_ref is not None and m.get("macro_auroc") is not None:
            m["delta_macro_vs_burden"] = m["macro_auroc"] - burden_ref
        if ref_pooled is not None and m.get("pooled_auroc") is not None:
            m["delta_pooled_vs_reference"] = m["pooled_auroc"] - ref_pooled

    provenance["cohort_provenance"] = cohort.provenance
    provenance["split_assignment_counts"] = splits.assignment.value_counts().to_dict()

    return ExperimentResult(
        config=cfg,
        cohort_summary=cohort.summary(),
        split_summary=splits.summary(),
        metrics=metrics,
        per_gene=pd.concat(per_gene_frames, ignore_index=True),
        provenance=provenance,
    )
