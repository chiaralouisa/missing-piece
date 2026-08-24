"""Turn a loaded study into the modelling cohort for panel completion.

Two decisions here carry most of the statistical risk:

1. **Who is eligible.** Ground truth for the target genes exists only for
   samples sequenced on a panel that actually covers them. A patient sequenced
   with IMPACT341 has no IMPACT505-only calls at all -- including them would
   supply 164 guaranteed zeros per patient and fabricate an easy majority class.
2. **One sample per patient.** MSK-IMPACT cohorts contain repeat samples from
   the same patient. Two samples from one tumour are near-duplicates; letting
   them straddle a train/test split leaks the answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .cbioportal import AlterationMatrix

log = logging.getLogger(__name__)

__all__ = [
    "NSCLC_ONCOTREE_CODES",
    "Cohort",
    "select_cancer_type",
    "one_sample_per_patient",
    "eligible_for_target_panel",
    "build_cohort",
]

#: OncoTree codes rolled up as non-small cell lung cancer.
NSCLC_ONCOTREE_CODES: frozenset[str] = frozenset(
    {
        "NSCLC",  # non-small cell lung cancer, NOS
        "LUAD",   # lung adenocarcinoma
        "LUSC",   # lung squamous cell carcinoma
        "LUAS",   # lung adenosquamous carcinoma
        "LCLC",   # large cell lung carcinoma
        "NSCLCPD",  # poorly differentiated NSCLC
        "LUPC",   # pleomorphic carcinoma of the lung
        "SPCC",   # spindle cell carcinoma of the lung
        "GCLC",   # giant cell carcinoma of the lung
        "GCTLC",  # giant cell tumour of the lung
        "GRCT",   # granular cell tumour
        "GCA",    # giant cell adenocarcinoma
        "BASQ",   # basaloid large cell carcinoma
        "LUACC",  # lung adenoid cystic carcinoma
        "LUMEC",  # lung mucoepidermoid carcinoma
    }
)


@dataclass
class Cohort:
    """Modelling-ready cohort: aligned observed/target blocks plus covariates."""

    observed: pd.DataFrame        # patients x observed genes, bool
    target: pd.DataFrame          # patients x target genes, bool
    covariates: pd.DataFrame      # patients x covariates, numeric
    clinical: pd.DataFrame        # patients x raw clinical columns
    provenance: dict

    def __post_init__(self) -> None:
        if not self.observed.index.equals(self.target.index):
            raise ValueError("observed and target must share a patient index")
        if not self.observed.index.equals(self.covariates.index):
            raise ValueError("covariates must share the patient index")

    @property
    def n_patients(self) -> int:
        return self.observed.shape[0]

    @property
    def patient_ids(self) -> pd.Index:
        return self.observed.index

    def target_prevalence(self) -> pd.Series:
        return self.target.mean(axis=0)

    def target_sparsity(self) -> float:
        """Fraction of evaluated gene-patient pairs that carry an alteration."""
        return float(self.target.to_numpy().mean())

    def subset(self, patients: Sequence[str]) -> "Cohort":
        keep = list(patients)
        return Cohort(
            observed=self.observed.loc[keep],
            target=self.target.loc[keep],
            covariates=self.covariates.loc[keep],
            clinical=self.clinical.reindex(keep),
            provenance=dict(self.provenance),
        )

    def summary(self) -> str:
        prev = self.target_prevalence()
        return (
            f"{self.n_patients} patients | "
            f"{self.observed.shape[1]} observed genes -> "
            f"{self.target.shape[1]} target genes\n"
            f"target sparsity {self.target_sparsity():.4%} "
            f"({int(self.target.to_numpy().sum())} positive pairs)\n"
            f"observed alteration rate {self.observed.to_numpy().mean():.4%}\n"
            f"target genes with >=1 positive: {int((prev > 0).sum())}/{len(prev)}; "
            f"max prevalence {prev.max():.2%} ({prev.idxmax() if len(prev) else 'n/a'})"
        )


def select_cancer_type(
    clinical: pd.DataFrame,
    oncotree_codes: frozenset[str] | set[str] = NSCLC_ONCOTREE_CODES,
    column: str = "ONCOTREE_CODE",
    fallback_column: str = "CANCER_TYPE",
    fallback_match: str = "Non-Small Cell Lung Cancer",
) -> pd.Index:
    """Sample ids belonging to the requested cancer type."""
    if column in clinical.columns:
        codes = clinical[column].fillna("").str.strip().str.upper()
        hit = codes.isin({c.upper() for c in oncotree_codes})
        if hit.any():
            return clinical.index[hit]
        log.warning("no samples matched %s codes; falling back to %s", column, fallback_column)
    if fallback_column in clinical.columns:
        hit = clinical[fallback_column].fillna("").str.contains(fallback_match, case=False)
        return clinical.index[hit]
    raise KeyError(f"neither {column} nor {fallback_column} present in clinical data")


def eligible_for_target_panel(
    matrix: AlterationMatrix, target_genes: Sequence[str], min_coverage: float = 1.0
) -> pd.Index:
    """Samples whose panel covers (at least ``min_coverage`` of) the target genes.

    These are the only samples with usable ground truth. ``min_coverage`` below
    1.0 admits partially covering panels; genes off-panel for such a sample stay
    masked and must be excluded from that sample's evaluation.
    """
    present = [g for g in target_genes if g in matrix.assayed.columns]
    if not present:
        raise ValueError("none of the target genes appear in the alteration matrix")
    coverage = matrix.assayed[present].mean(axis=1)
    return matrix.assayed.index[coverage >= min_coverage]


def one_sample_per_patient(
    clinical: pd.DataFrame,
    samples: pd.Index,
    patient_column: str = "PATIENT_ID",
    prefer_primary: bool = True,
    tie_breaker: pd.Series | None = None,
) -> pd.Index:
    """Pick one sample per patient.

    Preference order: primary over metastatic (when ``prefer_primary``), then
    the highest ``tie_breaker`` value (e.g. number of alterations detected, as a
    crude proxy for assay success), then the lexicographically first sample id
    so the choice is deterministic.
    """
    if patient_column not in clinical.columns:
        log.warning("no %s column; treating every sample as its own patient", patient_column)
        return pd.Index(sorted(samples))

    frame = pd.DataFrame(index=pd.Index(samples, name="SAMPLE_ID"))
    frame["PATIENT"] = clinical.reindex(frame.index)[patient_column].values

    rank_primary = np.zeros(len(frame))
    if prefer_primary and "SAMPLE_TYPE" in clinical.columns:
        stype = clinical.reindex(frame.index)["SAMPLE_TYPE"].fillna("").str.lower()
        rank_primary = (~stype.str.startswith("primary")).astype(int).to_numpy()
    frame["R_PRIMARY"] = rank_primary
    frame["R_TIE"] = (
        -tie_breaker.reindex(frame.index).fillna(-np.inf).to_numpy()
        if tie_breaker is not None
        else 0.0
    )
    frame["R_ID"] = frame.index

    frame = frame.sort_values(["PATIENT", "R_PRIMARY", "R_TIE", "R_ID"])
    chosen = frame.groupby("PATIENT", sort=True).head(1)
    return pd.Index(chosen.index, name="SAMPLE_ID")


def _build_covariates(clinical: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Numeric covariates available at prediction time from the small panel."""
    cov = pd.DataFrame(index=observed.index)

    # Burden measured on the observed panel only -- computable without the
    # large panel, so it is a legitimate input (and a critical ablation).
    cov["n_altered_observed"] = observed.sum(axis=1).astype(float)
    cov["log1p_n_altered_observed"] = np.log1p(cov["n_altered_observed"])

    clin = clinical.reindex(observed.index)
    for col, name in (
        ("TMB_NONSYNONYMOUS", "tmb"),
        ("MSI_SCORE", "msi"),
        ("FRACTION_GENOME_ALTERED", "fga"),
        ("AGE_AT_SEQUENCING", "age"),
        ("CURRENT_AGE_DEID", "age_deid"),
    ):
        if col in clin.columns:
            vals = pd.to_numeric(clin[col], errors="coerce")
            cov[name] = vals.fillna(vals.median())
            if name == "tmb":
                cov["log1p_tmb"] = np.log1p(cov["tmb"].clip(lower=0))

    if "SEX" in clin.columns:
        cov["sex_female"] = (
            clin["SEX"].fillna("").str.strip().str.lower().eq("female").astype(float)
        )
    if "SMOKING_STATUS" in clin.columns:
        smoke = clin["SMOKING_STATUS"].fillna("").str.lower()
        cov["smoker"] = smoke.str.contains("smok").astype(float)
        cov["never_smoker"] = smoke.str.contains("never").astype(float)
    if "SAMPLE_TYPE" in clin.columns:
        cov["is_metastasis"] = (
            clin["SAMPLE_TYPE"].fillna("").str.lower().str.startswith("metast").astype(float)
        )
    if "ONCOTREE_CODE" in clin.columns:
        codes = clin["ONCOTREE_CODE"].fillna("").str.upper()
        for code in ("LUAD", "LUSC"):
            cov[f"histology_{code.lower()}"] = codes.eq(code).astype(float)

    return cov.astype(float)


def build_cohort(
    matrix: AlterationMatrix,
    observed_genes: Sequence[str],
    target_genes: Sequence[str],
    oncotree_codes: frozenset[str] | set[str] | None = NSCLC_ONCOTREE_CODES,
    min_target_coverage: float = 1.0,
    require_observed_coverage: bool = True,
    dedupe_patients: bool = True,
) -> Cohort:
    """Apply cohort filters and split the matrix into observed/target blocks."""
    steps: list[dict] = []
    samples = matrix.values.index
    steps.append({"step": "loaded", "n_samples": len(samples)})

    if oncotree_codes is not None:
        samples = samples.intersection(
            select_cancer_type(matrix.clinical, oncotree_codes)
        )
        steps.append({"step": "cancer_type_filter", "n_samples": len(samples)})

    eligible = eligible_for_target_panel(matrix, target_genes, min_target_coverage)
    samples = samples.intersection(eligible)
    steps.append({"step": "target_panel_coverage", "n_samples": len(samples)})

    if require_observed_coverage:
        present_obs = [g for g in observed_genes if g in matrix.assayed.columns]
        obs_cov = matrix.assayed.loc[samples, present_obs].mean(axis=1)
        samples = samples[obs_cov >= 1.0]
        steps.append({"step": "observed_panel_coverage", "n_samples": len(samples)})

    if dedupe_patients:
        tie = matrix.values.loc[samples].sum(axis=1)
        samples = one_sample_per_patient(matrix.clinical, samples, tie_breaker=tie)
        steps.append({"step": "one_sample_per_patient", "n_samples": len(samples)})

    if len(samples) == 0:
        raise ValueError(
            "cohort is empty after filtering; check cancer-type codes and that "
            "some samples were sequenced on a panel covering the target genes"
        )

    obs_cols = [g for g in observed_genes if g in matrix.values.columns]
    tgt_cols = [g for g in target_genes if g in matrix.values.columns]
    observed = matrix.values.loc[samples, obs_cols].astype(bool)
    target = matrix.values.loc[samples, tgt_cols].astype(bool)
    clinical = matrix.clinical.reindex(samples)
    covariates = _build_covariates(clinical, observed)

    provenance = dict(matrix.provenance)
    provenance["cohort_steps"] = steps
    provenance["n_observed_genes"] = len(obs_cols)
    provenance["n_target_genes"] = len(tgt_cols)
    provenance["panels_present"] = (
        matrix.panel_of_sample.reindex(samples).value_counts().to_dict()
    )
    return Cohort(
        observed=observed,
        target=target,
        covariates=covariates,
        clinical=clinical,
        provenance=provenance,
    )
