"""Export the derived cohort as one small, portable file.

The MSK-CHORD download is large and carries a CC BY-NC licence and patient-level
clinical detail. The *modelling* cohort is neither: it is a binary matrix of
2-3k patients x ~500 genes, which compresses to a couple of megabytes and
contains no free text, no dates, and no identifiers beyond the study's own
de-identified sample ids.

Exporting that lets the analysis run somewhere the raw study cannot go -- a
sandboxed session, a collaborator's machine, a CI job -- without moving the
study itself.

**Check the study's data-use terms before sharing an export.** It still contains
patient-level genomic data. Use ``anonymize=True`` to replace sample ids with
opaque indices when the destination does not need to join back to the study.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data.cbioportal import StudyFiles, build_alteration_matrix
from .data.cohort import NSCLC_ONCOTREE_CODES, Cohort, build_cohort
from .panels import PanelPair, load_panels

__all__ = ["export_cohort", "load_exported_cohort"]

_FORMAT_VERSION = 1


def export_cohort(
    study_dir: str | Path,
    out_path: str | Path = "cohort_nsclc.npz",
    panel_dir: str | Path | None = None,
    observed_panel: str = "IMPACT341",
    target_panel: str = "IMPACT505",
    restrict_to_nsclc: bool = True,
    include_cna: bool = True,
    deep_cna_only: bool = True,
    anonymize: bool = False,
) -> Path:
    """Build the cohort from a study directory and write it to a single .npz."""
    study_dir = Path(study_dir)
    panel_root = Path(panel_dir) if panel_dir else study_dir
    panels = load_panels(panel_root)
    pair = PanelPair(panels[observed_panel.upper()], panels[target_panel.upper()])

    files = StudyFiles.discover(study_dir)
    matrix = build_alteration_matrix(
        files,
        genes=list(pair.target.genes) + list(pair.observed_only_genes),
        panels=panels,
        include_cna=include_cna,
        deep_cna_only=deep_cna_only,
    )
    cohort = build_cohort(
        matrix,
        observed_genes=pair.observed_genes,
        target_genes=pair.target_genes,
        oncotree_codes=NSCLC_ONCOTREE_CODES if restrict_to_nsclc else None,
    )

    ids = (
        np.array([f"P{i:06d}" for i in range(cohort.n_patients)])
        if anonymize
        else np.asarray(cohort.patient_ids, dtype=str)
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        format_version=_FORMAT_VERSION,
        observed=cohort.observed.to_numpy(dtype=bool),
        target=cohort.target.to_numpy(dtype=bool),
        observed_genes=np.asarray(cohort.observed.columns, dtype=str),
        target_genes=np.asarray(cohort.target.columns, dtype=str),
        patient_ids=ids,
        covariates=cohort.covariates.to_numpy(dtype=float),
        covariate_names=np.asarray(cohort.covariates.columns, dtype=str),
        observed_panel=observed_panel,
        target_panel=target_panel,
        provenance=str(cohort.provenance),
        anonymized=anonymize,
    )
    return out_path


def load_exported_cohort(path: str | Path) -> Cohort:
    """Read back a cohort written by :func:`export_cohort`."""
    with np.load(path, allow_pickle=False) as z:
        version = int(z["format_version"])
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"{path} is export format v{version}; this build reads v{_FORMAT_VERSION}"
            )
        ids = pd.Index(z["patient_ids"], name="SAMPLE_ID")
        observed = pd.DataFrame(z["observed"], index=ids, columns=list(z["observed_genes"]))
        target = pd.DataFrame(z["target"], index=ids, columns=list(z["target_genes"]))
        covariates = pd.DataFrame(
            z["covariates"], index=ids, columns=list(z["covariate_names"])
        )
        provenance = {
            "source": "exported_cohort",
            "path": str(path),
            "observed_panel": str(z["observed_panel"]),
            "target_panel": str(z["target_panel"]),
            "original_provenance": str(z["provenance"]),
        }
    clinical = pd.DataFrame(index=ids)
    clinical["PATIENT_ID"] = ids
    return Cohort(
        observed=observed,
        target=target,
        covariates=covariates,
        clinical=clinical,
        provenance=provenance,
    )
