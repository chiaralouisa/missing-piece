import numpy as np
import pytest

from missing_piece.experiment import ExperimentConfig, run_experiment
from missing_piece.export import export_cohort, load_exported_cohort
from missing_piece.inspect_study import inspect_study


def test_inspect_reports_panels_and_cohort(small_study):
    report = inspect_study(
        small_study, observed_panel="SMALL", target_panel="BIG", restrict_to_nsclc=False
    )
    assert "FATAL" not in report
    assert "SMALL" in report and "BIG" in report
    assert "after target_panel_coverage" in report
    assert "3 genes on BIG only" in report


def test_inspect_flags_missing_panels(tmp_path, small_study):
    report = inspect_study(small_study, observed_panel="SMALL", target_panel="NOPE")
    assert "FATAL" in report


def test_inspect_flags_lfs_pointers(tmp_path, small_study):
    (small_study / "data_mutations.txt").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:x\nsize 1\n"
    )
    report = inspect_study(
        small_study, observed_panel="SMALL", target_panel="BIG", restrict_to_nsclc=False
    )
    assert "LFS POINTER" in report


def test_inspect_warns_without_panel_matrix(small_study):
    (small_study / "data_gene_panel_matrix.txt").unlink()
    report = inspect_study(
        small_study, observed_panel="SMALL", target_panel="BIG", restrict_to_nsclc=False
    )
    assert "unassayed genes look wild-type" in report


def test_export_roundtrip_preserves_the_cohort(small_study, tmp_path):
    path = export_cohort(
        small_study,
        out_path=tmp_path / "c.npz",
        observed_panel="SMALL",
        target_panel="BIG",
        restrict_to_nsclc=False,
    )
    cohort = load_exported_cohort(path)
    assert list(cohort.target.columns) == ["GENED", "GENEE", "GENEF"]
    assert list(cohort.observed.columns) == ["GENEA", "GENEB", "GENEC"]
    # Eligible on BIG: S1, S2, S5. S1 and S5 are both patient P1, so dedup
    # leaves S1 and S2.
    assert cohort.n_patients == 2
    assert set(cohort.patient_ids) == {"S1", "S2"}
    assert "n_altered_observed" in cohort.covariates.columns


def test_export_can_anonymize(small_study, tmp_path):
    path = export_cohort(
        small_study,
        out_path=tmp_path / "a.npz",
        observed_panel="SMALL",
        target_panel="BIG",
        restrict_to_nsclc=False,
        anonymize=True,
    )
    cohort = load_exported_cohort(path)
    assert all(pid.startswith("P0") for pid in cohort.patient_ids)


def test_experiment_runs_from_an_exported_cohort(tmp_path):
    """The handoff path: export once, run the full protocol from the file."""
    from missing_piece.data.simulate import SimulationConfig, simulate_cohort

    sim = simulate_cohort(
        SimulationConfig(n_patients=300, n_observed_genes=30, n_target_genes=15,
                         target_sparsity=0.08, seed=0)
    )
    path = tmp_path / "sim.npz"
    np.savez_compressed(
        path,
        format_version=1,
        observed=sim.observed.to_numpy(dtype=bool),
        target=sim.target.to_numpy(dtype=bool),
        observed_genes=np.asarray(sim.observed.columns, dtype=str),
        target_genes=np.asarray(sim.target.columns, dtype=str),
        patient_ids=np.asarray(sim.patient_ids, dtype=str),
        covariates=sim.covariates.to_numpy(dtype=float),
        covariate_names=np.asarray(sim.covariates.columns, dtype=str),
        observed_panel="SIM_SMALL",
        target_panel="SIM_LARGE",
        provenance="test",
        anonymized=False,
    )
    result = run_experiment(
        ExperimentConfig(
            name="from_file",
            cohort_file=str(path),
            protocol="holdout",
            models={"prevalence": {}, "burden": {}},
            n_bootstrap=10,
            n_permutations=10,
            n_joint_samples=0,
            min_positives_for_gene_auroc=3,
        )
    )
    assert result.metrics["prevalence"]["macro_auroc"] == pytest.approx(0.5, abs=1e-6)


def test_export_format_version_mismatch_is_caught(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, format_version=99)
    with pytest.raises(ValueError, match="format v99"):
        load_exported_cohort(path)
