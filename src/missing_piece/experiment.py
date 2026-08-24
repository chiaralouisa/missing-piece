"""Run a panel-completion experiment: build cohort, fit models, evaluate, report.

Two evaluation protocols:

``holdout``
    A single train/validation/test split, as in the submitted abstract. Fast,
    but at ~1.2% prevalence a 15% test split leaves most target genes with too
    few positives to estimate a per-gene AUROC at all -- typically only a third
    of the 164 genes are scoreable, and the macro average is then taken over a
    biased, high-prevalence subset.

``cv``
    Repeated K-fold cross-validation with out-of-fold predictions pooled into a
    single matrix covering every patient. Every gene gets its positives from
    the whole cohort, so far more genes become estimable and the macro average
    stops being a statement about the common genes only. This is the protocol
    to report.
"""

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
    bootstrap_metric,
    evaluate_predictions,
    macro_auroc,
    permutation_test,
    pooled_auroc,
)
from .models import build_model
from .panels import PanelPair

log = logging.getLogger(__name__)

__all__ = ["ExperimentConfig", "ExperimentResult", "run_experiment"]


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce one experiment."""

    name: str = "panel_completion"

    # --- data source: `study_dir` for real data, else the simulator --------
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
    protocol: str = "holdout"          # "holdout" | "cv"
    n_folds: int = 5
    n_repeats: int = 1
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

    def __post_init__(self) -> None:
        if self.protocol not in ("holdout", "cv"):
            raise ValueError(f"protocol must be 'holdout' or 'cv', got {self.protocol!r}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
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
        rows = []
        for name, m in self.metrics.items():
            ci = m.get("macro_auroc_ci", {})
            rows.append(
                {
                    "model": name,
                    "macro_auroc": m.get("macro_auroc"),
                    "ci_lo": ci.get("lo"),
                    "ci_hi": ci.get("hi"),
                    "d_vs_burden": m.get("delta_macro_vs_burden"),
                    "pooled_auroc": m.get("pooled_auroc"),
                    "pooled_adj": m.get("pooled_auroc_adjusted"),
                    "within_pt": m.get("within_patient_auroc"),
                    "within_pt_adj": m.get("within_patient_auroc_adjusted"),
                    "macro_ap": m.get("macro_average_precision"),
                    "brier": m.get("brier"),
                    "ece": m.get("ece"),
                    "genes": m.get("n_genes_scored"),
                    "perm_p": m.get("permutation_test", {}).get("p_value"),
                    "fit_s": m.get("fit_seconds"),
                }
            )
        return pd.DataFrame(rows).sort_values(
            "macro_auroc", ascending=False, na_position="last"
        )

    def save(self, directory: str | Path | None = None) -> Path:
        directory = Path(directory or self.config.output_dir) / self.config.name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2, default=str)
        )
        (directory / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2, default=str)
        )
        (directory / "provenance.json").write_text(
            json.dumps(self.provenance, indent=2, default=str)
        )
        self.table().to_csv(directory / "summary.csv", index=False)
        self.per_gene.to_csv(directory / "per_gene_auroc.csv", index=False)
        (directory / "summary.txt").write_text(self.render())
        return directory

    def render(self) -> str:
        return "\n".join(
            [
                f"# {self.config.name}",
                f"protocol: {self.config.protocol}"
                + (
                    f" ({self.config.n_folds}-fold x {self.config.n_repeats})"
                    if self.config.protocol == "cv"
                    else ""
                ),
                "",
                "## Cohort",
                self.cohort_summary,
                "",
                "## Evaluation set",
                self.split_summary,
                "",
                "## Results",
                self.table().to_string(index=False, float_format=lambda v: f"{v:.4f}"),
                "",
                "## Reading the columns",
                "macro_auroc    per-gene AUROC averaged over scoreable genes. Prevalence-free:",
                "               a patient-blind model scores exactly 0.5. This is the headline.",
                "d_vs_burden    macro AUROC minus the burden-only baseline. This is the",
                "               gene-specific information the panel adds beyond 'how mutated",
                "               is this tumour' -- the number that supports or sinks the claim.",
                "pooled_auroc   all patient-gene pairs in one ranking. Prevalence alone scores",
                "               ~0.79 here with zero patient-specific signal; never report alone.",
                "pooled_adj     pooled AUROC after removing each gene's mean score (0.5 = no",
                "               patient-specific signal).",
                "within_pt      ranking genes inside a patient; also prevalence-confounded.",
                "within_pt_adj  the same, prevalence-adjusted.",
                "genes          number of genes with enough positives to score at all.",
                "perm_p         permutation test for patient-specific information.",
            ]
        )


def _load_cohort(cfg: ExperimentConfig) -> tuple[Cohort, dict]:
    provenance: dict = {}

    if cfg.study_dir:
        from .data.cbioportal import StudyFiles, build_alteration_matrix
        from .panels import load_panels

        panel_dir = cfg.panel_dir or cfg.study_dir
        pair = PanelPair.from_directory(panel_dir, cfg.observed_panel, cfg.target_panel)
        warnings = pair.validate()
        for w in warnings:
            log.warning("panel pair: %s", w)
        provenance["panel_warnings"] = warnings
        provenance["panel_pair"] = pair.describe()

        files = StudyFiles.discover(cfg.study_dir)
        matrix = build_alteration_matrix(
            files,
            genes=list(pair.target.genes) + list(pair.observed_only_genes),
            panels=load_panels(panel_dir),
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


def _fit_predict(
    name: str, spec: dict, train: Cohort, val: Cohort, test: Cohort
) -> tuple[np.ndarray, float, Any]:
    model = build_model(name, **spec)
    t0 = time.time()
    model.fit(train, val)
    seconds = time.time() - t0
    return model.predict_proba(test), seconds, model


def _predictions_holdout(
    cohort: Cohort, cfg: ExperimentConfig
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, Any], str]:
    splits = make_splits(cohort, SplitSpec(**{"seed": cfg.seed, **cfg.split}))
    y = splits.test.target.to_numpy(dtype=float)
    probs, seconds, models = {}, {}, {}
    for name, spec in cfg.models.items():
        log.info("[holdout] fitting %s", name)
        p, s, model = _fit_predict(name, spec, splits.train, splits.val, splits.test)
        probs[name], seconds[name], models[name] = p, s, model
    return y, probs, seconds, {"test": splits.test, "models": models}, splits.summary()


def _predictions_cv(
    cohort: Cohort, cfg: ExperimentConfig
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, Any], str]:
    """Repeated K-fold with out-of-fold predictions pooled over the whole cohort."""
    n = cohort.n_patients
    ids = np.asarray(cohort.patient_ids)
    y = cohort.target.to_numpy(dtype=float)
    k = cfg.n_folds

    acc = {name: np.zeros((n, y.shape[1])) for name in cfg.models}
    seconds = {name: 0.0 for name in cfg.models}

    # Stratify folds on observed-panel burden, mirroring the holdout splitter.
    burden = cohort.observed.sum(axis=1).to_numpy(dtype=float)
    order_by_burden = np.argsort(burden, kind="stable")

    for repeat in range(cfg.n_repeats):
        rng = np.random.default_rng(cfg.seed + repeat)
        fold_of = np.empty(n, dtype=int)
        # Assign consecutive burden-ranked blocks round-robin to folds, with the
        # block order shuffled, so every fold matches the burden distribution.
        for start in range(0, n, k):
            block = order_by_burden[start : start + k]
            fold_of[block] = rng.permutation(len(block))

        for fold in range(k):
            test_mask = fold_of == fold
            val_mask = fold_of == ((fold + 1) % k)
            train_mask = ~(test_mask | val_mask)
            train = cohort.subset(list(ids[train_mask]))
            val = cohort.subset(list(ids[val_mask]))
            test = cohort.subset(list(ids[test_mask]))
            for name, spec in cfg.models.items():
                log.info("[cv r%d f%d] fitting %s", repeat, fold, name)
                p, s, _ = _fit_predict(name, spec, train, val, test)
                acc[name][test_mask] += p
                seconds[name] += s

    probs = {name: a / cfg.n_repeats for name, a in acc.items()}
    summary = (
        f"{n} patients, {cfg.n_folds}-fold CV x {cfg.n_repeats} repeat(s); "
        f"metrics computed on pooled out-of-fold predictions for every patient\n"
        f"target sparsity {cohort.target_sparsity():.4%}"
    )
    return y, probs, seconds, {"test": cohort, "models": {}}, summary


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Fit and evaluate every configured model on one cohort."""
    cohort, provenance = _load_cohort(cfg)
    log.info("cohort: %s", cohort.summary())

    runner = _predictions_cv if cfg.protocol == "cv" else _predictions_holdout
    y, probs, seconds, context, split_summary = runner(cohort, cfg)
    test_cohort: Cohort = context["test"]
    gene_names = test_cohort.target.columns

    metrics: dict[str, dict] = {}
    per_gene_frames: list[pd.DataFrame] = []

    for name, p in probs.items():
        result = evaluate_predictions(
            y, p, gene_names=gene_names, min_positives=cfg.min_positives_for_gene_auroc
        )
        m = result.to_dict()
        m["fit_seconds"] = seconds[name]
        m["params"] = cfg.models[name]

        if cfg.n_bootstrap:
            m["macro_auroc_ci"] = bootstrap_metric(
                y,
                p,
                metric=lambda a, b: macro_auroc(a, b, cfg.min_positives_for_gene_auroc),
                n_boot=cfg.n_bootstrap,
                seed=cfg.seed,
            )
            m["pooled_auroc_ci"] = bootstrap_metric(
                y, p, metric=pooled_auroc, n_boot=cfg.n_bootstrap, seed=cfg.seed
            )
        if cfg.n_permutations:
            m["permutation_test"] = permutation_test(
                y,
                p,
                metric=lambda a, b: macro_auroc(a, b, cfg.min_positives_for_gene_auroc),
                n_permutations=cfg.n_permutations,
                seed=cfg.seed,
            )

        model = context["models"].get(name)
        if cfg.n_joint_samples and model is not None:
            samples = model.sample(
                test_cohort, n_samples=cfg.n_joint_samples, seed=cfg.seed
            )
            m["joint"] = evaluate_joint(y, samples)
            m["is_generative"] = model.is_generative
            if not model.is_generative:
                m["joint"]["note"] = "independent draws from predicted marginals"

        metrics[name] = m
        per_gene_frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "gene": result.gene_names,
                    "auroc": result.per_gene_auroc,
                    "average_precision": result.per_gene_ap,
                    "n_positives": y.sum(axis=0).astype(int),
                    "prevalence": y.mean(axis=0),
                }
            )
        )

    ref = metrics.get(cfg.reference_model, {}).get("macro_auroc")
    burden_ref = metrics.get(cfg.burden_reference_model, {}).get("macro_auroc")
    ref_pooled = metrics.get(cfg.reference_model, {}).get("pooled_auroc")
    for m in metrics.values():
        if ref is not None:
            m["delta_macro_vs_reference"] = m["macro_auroc"] - ref
        if burden_ref is not None:
            m["delta_macro_vs_burden"] = m["macro_auroc"] - burden_ref
        if ref_pooled is not None:
            m["delta_pooled_vs_reference"] = m["pooled_auroc"] - ref_pooled

    provenance["cohort_provenance"] = cohort.provenance
    provenance["n_target_genes_total"] = int(y.shape[1])
    provenance["n_target_genes_scoreable"] = int(
        (y.sum(axis=0) >= cfg.min_positives_for_gene_auroc).sum()
    )

    return ExperimentResult(
        config=cfg,
        cohort_summary=cohort.summary(),
        split_summary=split_summary,
        metrics=metrics,
        per_gene=pd.concat(per_gene_frames, ignore_index=True),
        provenance=provenance,
    )
