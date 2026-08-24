import numpy as np
import pytest

from missing_piece.data.cbioportal import (
    StudyFiles,
    build_alteration_matrix,
    read_cna,
    read_mutations,
)
from missing_piece.data.cohort import build_cohort, eligible_for_target_panel
from missing_piece.panels import PanelPair, load_panels


@pytest.fixture
def files(small_study):
    return StudyFiles.discover(small_study)


def test_silent_mutations_are_dropped(files):
    muts = read_mutations(files)
    assert not ((muts.SAMPLE_ID == "S2") & (muts.GENE == "GENEB")).any()
    assert ((muts.SAMPLE_ID == "S1") & (muts.GENE == "GENEA")).any()


def test_only_deep_cna_counts_by_default(files):
    deep = read_cna(files)
    pairs = set(map(tuple, deep[["SAMPLE_ID", "GENE"]].to_numpy()))
    assert ("S2", "GENEB") in pairs   # -2, deep deletion
    assert ("S1", "GENEF") in pairs   # +2, amplification
    assert ("S1", "GENEC") not in pairs  # +1, shallow gain

    shallow = read_cna(files, deep_only=False)
    assert ("S1", "GENEC") in set(map(tuple, shallow[["SAMPLE_ID", "GENE"]].to_numpy()))


def test_unassayed_genes_are_masked_not_zeroed(small_study, files):
    panels = load_panels(small_study)
    matrix = build_alteration_matrix(
        files, genes=list(panels["BIG"].genes), panels=panels
    )
    # S3/S4 were sequenced on SMALL, so GENED/E/F were never interrogated.
    assert not matrix.assayed.loc["S3", ["GENED", "GENEE", "GENEF"]].any()
    assert matrix.assayed.loc["S1", ["GENED", "GENEE", "GENEF"]].all()
    # And a call must never survive where the assay did not cover the gene.
    assert not (matrix.values & ~matrix.assayed).to_numpy().any()


def test_eligibility_excludes_small_panel_samples(small_study, files):
    panels = load_panels(small_study)
    pair = PanelPair(observed=panels["SMALL"], target=panels["BIG"])
    matrix = build_alteration_matrix(
        files, genes=list(panels["BIG"].genes), panels=panels
    )
    eligible = eligible_for_target_panel(matrix, pair.target_genes)
    assert set(eligible) == {"S1", "S2", "S5"}
    assert "S3" not in eligible and "S4" not in eligible


def test_build_cohort_dedupes_patients_and_prefers_primary(small_study, files):
    panels = load_panels(small_study)
    pair = PanelPair(observed=panels["SMALL"], target=panels["BIG"])
    matrix = build_alteration_matrix(
        files, genes=list(panels["BIG"].genes), panels=panels
    )
    cohort = build_cohort(
        matrix,
        observed_genes=pair.observed_genes,
        target_genes=pair.target_genes,
        oncotree_codes=None,
    )
    # S1 and S5 are both patient P1; exactly one survives, and it is the primary.
    assert len(cohort.patient_ids) == len(set(cohort.clinical["PATIENT_ID"]))
    assert "S1" in cohort.patient_ids and "S5" not in cohort.patient_ids
    assert list(cohort.target.columns) == ["GENED", "GENEE", "GENEF"]


def test_cohort_without_panel_info_assumes_full_coverage(files):
    matrix = build_alteration_matrix(files, genes=["GENEA", "GENED"], panels=None)
    assert matrix.assayed.to_numpy().all()
