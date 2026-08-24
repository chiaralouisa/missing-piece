"""Negative controls: check that the evaluation cannot manufacture a result.

Before trusting a headline AUROC we need to know what the pipeline reports when
the hypothesis is *false*. Three regimes are run through the identical
train/evaluate path:

===================  =====================================================
regime               what a correct pipeline must report
===================  =====================================================
``independent``      macro AUROC ~ 0.50 for every model. Target genes carry
                     no patient information at all, so anything above 0.5 is
                     leakage or a metric bug.
``burden_only``      macro AUROC above 0.5 (burden is real, shared signal)
                     but *no gain over the burden-only baseline*. This is the
                     regime that separates "the panel predicts the genome"
                     from "burden predicts burden".
``full``             macro AUROC clearly above the burden baseline. The
                     hypothesis is true here, so the pipeline must detect it
                     -- a control against being so conservative that real
                     signal is missed.
===================  =====================================================

The ``pooled`` AUROC column is printed alongside to show how much of it
survives in the ``independent`` regime, where by construction *none* of it is
patient-specific information.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .data.simulate import SimulationConfig, simulate_cohort
from .data.splits import SplitSpec, make_splits
from .eval.metrics import evaluate_predictions, macro_auroc, permutation_test
from .models import build_model

log = logging.getLogger(__name__)

__all__ = ["run_negative_controls", "control_table"]

REGIMES = ("independent", "burden_only", "full")

#: Small, fast defaults -- the controls are about correctness, not tuning.
_FAST_OVERRIDES = {
    "mlp": {"epochs": 60, "patience": 12},
    "flow_discrete": {"epochs": 80, "patience": 15, "width": 256, "depth": 3},
    "flow_gaussian": {
        "epochs": 80,
        "patience": 15,
        "width": 256,
        "depth": 3,
        "n_samples_for_marginals": 32,
        "n_sampling_steps": 20,
    },
}


def control_table(
    n_patients: int = 2226,
    seed: int = 0,
    models: list[str] | None = None,
    regimes: tuple[str, ...] = REGIMES,
    n_permutations: int = 200,
    min_positives: int = 5,
) -> pd.DataFrame:
    """Fit every model under every regime and collect the diagnostic metrics."""
    models = models or ["prevalence", "burden", "logistic", "flow_discrete"]
    rows: list[dict] = []

    for regime in regimes:
        cohort = simulate_cohort(
            SimulationConfig(n_patients=n_patients, regime=regime, seed=seed)
        )
        splits = make_splits(cohort, SplitSpec(seed=seed))
        y_test = splits.test.target.to_numpy(dtype=float)

        burden_macro = None
        results: dict[str, dict] = {}
        for name in models:
            model = build_model(name, **_FAST_OVERRIDES.get(name, {}))
            model.fit(splits.train, splits.val)
            probs = model.predict_proba(splits.test)
            ev = evaluate_predictions(
                y_test, probs, splits.test.target.columns, min_positives=min_positives
            )
            perm = permutation_test(
                y_test,
                probs,
                metric=lambda a, b: macro_auroc(a, b, min_positives),
                n_permutations=n_permutations,
                seed=seed,
            )
            results[name] = {"ev": ev, "perm": perm}
            if name == "burden":
                burden_macro = ev.macro_auroc

        for name, r in results.items():
            ev, perm = r["ev"], r["perm"]
            rows.append(
                {
                    "regime": regime,
                    "model": name,
                    "macro_auroc": ev.macro_auroc,
                    "pooled_auroc": ev.pooled_auroc,
                    "within_patient_auroc": ev.within_patient_auroc,
                    "vs_burden": (
                        None if burden_macro is None else ev.macro_auroc - burden_macro
                    ),
                    "perm_p": perm["p_value"],
                    "n_genes_scored": ev.n_genes_scored,
                }
            )
    return pd.DataFrame(rows)


def _verdicts(table: pd.DataFrame, tol: float = 0.02) -> list[str]:
    """Turn the control table into pass/fail statements."""
    out: list[str] = []

    ind = table[table["regime"] == "independent"]
    worst = (ind["macro_auroc"] - 0.5).abs().max() if not ind.empty else float("nan")
    ok = worst <= tol * 2
    out.append(
        f"[{'PASS' if ok else 'FAIL'}] independent regime: macro AUROC stays at 0.5 "
        f"(max deviation {worst:.4f}); no model invents patient-specific signal."
    )
    if not ind.empty:
        pooled_max = ind["pooled_auroc"].max()
        out.append(
            f"[INFO] independent regime: pooled AUROC still reaches {pooled_max:.4f} "
            "with zero patient-specific signal -- pooled AUROC is a prevalence "
            "statistic and must never be reported on its own."
        )

    bo = table[table["regime"] == "burden_only"]
    if not bo.empty:
        gain = bo[bo["model"] != "burden"]["vs_burden"].max()
        ok = gain <= tol * 2
        out.append(
            f"[{'PASS' if ok else 'FAIL'}] burden_only regime: best gain over the "
            f"burden baseline is {gain:+.4f}; nothing beats burden when burden is "
            "the only channel."
        )

    full = table[table["regime"] == "full"]
    if not full.empty:
        gain = full[full["model"] != "burden"]["vs_burden"].max()
        ok = gain > tol
        out.append(
            f"[{'PASS' if ok else 'FAIL'}] full regime: best gain over the burden "
            f"baseline is {gain:+.4f}; real cross-panel signal is detected."
        )
    return out


def run_negative_controls(
    n_patients: int = 2226,
    seed: int = 0,
    models: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> str:
    """Run the battery and return a human-readable report."""
    table = control_table(n_patients=n_patients, seed=seed, models=models)
    lines = [
        "# Negative-control battery",
        "",
        table.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        "## Verdicts",
        *_verdicts(table),
    ]
    report = "\n".join(lines)
    if output_dir:
        directory = Path(output_dir) / "negative_controls"
        directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(directory / "controls.csv", index=False)
        (directory / "report.txt").write_text(report)
    return report
