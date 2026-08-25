"""Diagnose a cBioPortal study directory before running an experiment.

Loading a real study is where a panel-completion pipeline silently goes wrong:
a missing gene-panel matrix turns unassayed genes into wild-type calls, a
renamed clinical column drops a covariate without complaining, and an LFS
pointer looks like a file until pandas reads it. This module answers "will my
data load, and what will the cohort actually be?" without fitting anything.

Run it first, on real data, every time.
"""

from __future__ import annotations

from pathlib import Path


from .data.cbioportal import (
    StudyFiles,
    build_alteration_matrix,
    read_clinical,
    read_gene_panel_matrix,
)
from .data.cohort import NSCLC_ONCOTREE_CODES, build_cohort
from .panels import PanelPair, load_panels

__all__ = ["inspect_study"]

#: Clinical columns the covariate builder will use if they are present.
_WANTED_CLINICAL = [
    "PATIENT_ID",
    "ONCOTREE_CODE",
    "CANCER_TYPE",
    "SAMPLE_TYPE",
    "SEX",
    "SMOKING_STATUS",
    "TMB_NONSYNONYMOUS",
    "MSI_SCORE",
    "FRACTION_GENOME_ALTERED",
    "AGE_AT_SEQUENCING",
    "CURRENT_AGE_DEID",
]


def _line(label: str, value: object) -> str:
    return f"  {label:<34} {value}"


def inspect_study(
    study_dir: str | Path,
    panel_dir: str | Path | None = None,
    observed_panel: str = "IMPACT341",
    target_panel: str = "IMPACT505",
    restrict_to_nsclc: bool = True,
) -> str:
    """Return a human-readable report on what the study contains."""
    study_dir = Path(study_dir)
    out: list[str] = [f"# Study inspection: {study_dir}", ""]

    # ---- files ---------------------------------------------------------
    out.append("## Files")
    try:
        files = StudyFiles.discover(study_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        return "\n".join(out + [f"  FATAL: {exc}"])

    for label, path in [
        ("data_clinical_sample.txt", files.clinical_sample),
        ("data_clinical_patient.txt", files.clinical_patient),
        ("data_mutations.txt", files.mutations),
        ("data_cna.txt", files.cna),
        ("data_gene_panel_matrix.txt", files.gene_panel_matrix),
    ]:
        if path is None:
            note = "MISSING"
            if label == "data_gene_panel_matrix.txt":
                note += "  <-- without this, unassayed genes look wild-type"
        else:
            size = path.stat().st_size
            head = path.read_text(errors="replace")[:40]
            note = (
                "LFS POINTER - run `git lfs pull`"
                if head.startswith("version https://git-lfs")
                else f"{size / 1e6:.1f} MB"
            )
        out.append(_line(label, note))

    # ---- panels --------------------------------------------------------
    out.append("")
    out.append("## Gene panels")
    panel_root = Path(panel_dir) if panel_dir else None
    panels = None
    for candidate in [panel_root, study_dir, study_dir / "gene_panels"]:
        if candidate is None:
            continue
        try:
            panels = load_panels(candidate)
            out.append(_line("found in", candidate))
            break
        except (FileNotFoundError, NotADirectoryError):
            continue
    if panels is None:
        out.append("  FATAL: no readable gene-panel files found.")
        out.append("  Panels define the task; without them there is nothing to predict.")
        return "\n".join(out)

    out.append(_line("panels available", ", ".join(sorted(panels))))
    missing = [p for p in (observed_panel, target_panel) if p.upper() not in panels]
    if missing:
        out.append(f"  FATAL: requested panel(s) {missing} not present.")
        return "\n".join(out)

    pair = PanelPair(panels[observed_panel.upper()], panels[target_panel.upper()])
    for row in pair.describe().splitlines():
        out.append(f"  {row}")
    for w in pair.validate():
        out.append(f"  WARNING: {w}")

    # ---- clinical ------------------------------------------------------
    out.append("")
    out.append("## Clinical columns")
    clinical = read_clinical(files)
    out.append(_line("samples in clinical table", len(clinical)))
    present = [c for c in _WANTED_CLINICAL if c in clinical.columns]
    absent = [c for c in _WANTED_CLINICAL if c not in clinical.columns]
    out.append(_line("present", ", ".join(present) or "none"))
    out.append(_line("absent (covariates skipped)", ", ".join(absent) or "none"))
    if "PATIENT_ID" in clinical.columns:
        out.append(_line("distinct patients", clinical["PATIENT_ID"].nunique()))
    if "ONCOTREE_CODE" in clinical.columns:
        codes = clinical["ONCOTREE_CODE"].fillna("").str.upper()
        hits = codes.isin(NSCLC_ONCOTREE_CODES)
        out.append(_line("samples matching NSCLC codes", int(hits.sum())))
        top = codes[hits].value_counts().head(6)
        out.append(_line("  top codes", ", ".join(f"{k}={v}" for k, v in top.items())))

    # ---- panel matrix --------------------------------------------------
    out.append("")
    out.append("## Panel assignment")
    gpm = read_gene_panel_matrix(files)
    if gpm is None:
        out.append("  WARNING: no gene-panel matrix; every gene assumed assayed.")
        out.append("  On a mixed-panel cohort this fabricates true negatives.")
    else:
        out.append(_line("profile columns", ", ".join(gpm.columns)))
        col = "mutations" if "mutations" in gpm.columns else gpm.columns[0]
        counts = gpm[col].value_counts().head(8)
        out.append(_line(f"panels in '{col}'", ", ".join(f"{k}={v}" for k, v in counts.items())))
        unknown = sorted(set(gpm[col].dropna().unique()) - set(panels))
        if unknown:
            out.append(_line("no gene list for", ", ".join(unknown[:8])))

    # ---- cohort --------------------------------------------------------
    out.append("")
    out.append("## Cohort after filtering")
    try:
        matrix = build_alteration_matrix(
            files,
            genes=list(pair.target.genes) + list(pair.observed_only_genes),
            panels=panels,
        )
    except (ValueError, KeyError) as exc:
        # Reporting the failure *is* the output here; re-raising would defeat
        # the point of a tool whose job is to diagnose unloadable studies.
        out.append(f"  FATAL: could not build the alteration matrix: {exc}")
        return "\n".join(out)
    out.append(_line("matrix", matrix.summary()))
    out.append(_line("mutation events kept", matrix.provenance["n_mutation_events"]))
    out.append(_line("CNA events kept", matrix.provenance["n_cna_events"]))

    try:
        cohort = build_cohort(
            matrix,
            observed_genes=pair.observed_genes,
            target_genes=pair.target_genes,
            oncotree_codes=NSCLC_ONCOTREE_CODES if restrict_to_nsclc else None,
        )
    except ValueError as exc:
        out.append(f"  FATAL: {exc}")
        return "\n".join(out)

    for step in cohort.provenance["cohort_steps"]:
        out.append(_line(f"after {step['step']}", f"{step['n_samples']} samples"))
    out.append("")
    for row in cohort.summary().splitlines():
        out.append(f"  {row}")

    prev = cohort.target_prevalence()
    n_scoreable = int((cohort.target.sum(axis=0) >= 5).sum())
    out.append("")
    out.append("## Feasibility")
    out.append(_line("target genes with >=5 positives", f"{n_scoreable} / {len(prev)}"))
    out.append(
        _line("expected positives per gene in a 15% holdout",
              f"{0.15 * cohort.n_patients * cohort.target_sparsity():.1f}")
    )
    if n_scoreable < len(prev) * 0.5:
        out.append("  NOTE: fewer than half the target genes are individually")
        out.append("  estimable even on the full cohort. Use protocol: cv, and")
        out.append("  report how many genes entered the macro average.")
    panels_used = cohort.provenance.get("panels_present", {})
    if panels_used:
        out.append(_line("panels in final cohort", panels_used))
    return "\n".join(out)
