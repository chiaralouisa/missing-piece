#!/usr/bin/env python3
"""Render figures from a saved experiment directory.

Usage:
    python scripts/make_figures.py results/simulation_full

Produces, next to the results:
  auroc_comparison.png   macro AUROC with CIs, against the two baselines that
                         decide interpretation
  confound.png           pooled and within-patient AUROC beside their
                         prevalence-adjusted twins
  per_gene.png           per-gene AUROC against gene prevalence
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BASELINE_COLOUR = "#9aa0a6"
MODEL_COLOUR = "#1f77b4"
GENERATIVE_COLOUR = "#d62728"


def _load(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary = pd.read_csv(directory / "summary.csv")
    per_gene = pd.read_csv(directory / "per_gene_auroc.csv")
    metrics = json.loads((directory / "metrics.json").read_text())
    return summary, per_gene, metrics


def _colour(model: str) -> str:
    if model in ("prevalence", "burden"):
        return BASELINE_COLOUR
    return GENERATIVE_COLOUR if model.startswith("flow") else MODEL_COLOUR


def figure_auroc(summary: pd.DataFrame, out: Path) -> None:
    data = summary.dropna(subset=["macro_auroc"]).sort_values("macro_auroc")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    err = None
    if {"ci_lo", "ci_hi"} <= set(data.columns) and data["ci_lo"].notna().any():
        err = [
            (data["macro_auroc"] - data["ci_lo"]).to_numpy(),
            (data["ci_hi"] - data["macro_auroc"]).to_numpy(),
        ]
    ax.barh(
        data["model"],
        data["macro_auroc"],
        xerr=err,
        color=[_colour(m) for m in data["model"]],
        capsize=3,
    )
    ax.axvline(0.5, color="black", lw=1, ls="--")
    ax.text(0.5, -0.7, "chance", ha="center", va="top", fontsize=8)

    burden = summary.loc[summary["model"] == "burden", "macro_auroc"]
    if not burden.empty:
        ax.axvline(float(burden.iloc[0]), color="#e07b39", lw=1.2, ls=":")
        ax.text(
            float(burden.iloc[0]),
            len(data) - 0.3,
            " burden baseline",
            color="#e07b39",
            fontsize=8,
            va="top",
        )
    ax.set_xlabel("macro (per-gene) AUROC")
    ax.set_xlim(0.4, max(0.85, data["macro_auroc"].max() + 0.08))
    ax.set_title("Panel completion: prevalence-free performance")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def figure_confound(summary: pd.DataFrame, out: Path) -> None:
    cols = ["pooled_auroc", "pooled_adj", "within_pt", "within_pt_adj"]
    have = [c for c in cols if c in summary.columns]
    data = summary.set_index("model")[have]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    data.plot.bar(ax=ax, width=0.78)
    ax.axhline(0.5, color="black", lw=1, ls="--")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("")
    ax.set_title(
        "Prevalence confound: raw vs adjusted\n"
        "(a patient-blind model reaches ~0.8 raw, 0.5 adjusted)",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def figure_per_gene(per_gene: pd.DataFrame, out: Path) -> None:
    models = [m for m in per_gene["model"].unique() if m != "prevalence"]
    fig, axes = plt.subplots(
        1, len(models), figsize=(3.1 * len(models), 3.4), sharey=True, squeeze=False
    )
    for ax, model in zip(axes[0], models):
        sub = per_gene[(per_gene["model"] == model) & per_gene["auroc"].notna()]
        ax.scatter(
            sub["prevalence"], sub["auroc"], s=12, alpha=0.6, color=_colour(model)
        )
        ax.axhline(0.5, color="black", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("gene prevalence")
    axes[0][0].set_ylabel("per-gene AUROC")
    fig.suptitle("Per-gene performance vs how rare the gene is", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    directory = Path(argv[1])
    summary, per_gene, _ = _load(directory)
    figure_auroc(summary, directory / "auroc_comparison.png")
    figure_confound(summary, directory / "confound.png")
    figure_per_gene(per_gene, directory / "per_gene.png")
    print(f"figures written to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
