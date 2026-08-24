import numpy as np
import pytest

from missing_piece.experiment import (
    ExperimentConfig,
    _align_fold_levels,
    run_experiment,
)

FAST_MODELS = {
    "prevalence": {},
    "burden": {},
    "logistic": {},
    "flow_discrete": {"epochs": 10, "patience": 4, "width": 64, "depth": 2},
}

SMALL_SIM = {
    "n_patients": 300,
    "n_observed_genes": 40,
    "n_target_genes": 20,
    "target_sparsity": 0.06,
}


def _cfg(**kwargs):
    base = dict(
        name="test",
        simulation=SMALL_SIM,
        models=FAST_MODELS,
        n_bootstrap=20,
        n_permutations=20,
        n_joint_samples=0,
        min_positives_for_gene_auroc=3,
    )
    base.update(kwargs)
    return ExperimentConfig(**base)


def test_align_fold_levels_collapses_per_fold_constants():
    """The bias fix: fold-varying constants must become one value per gene."""
    fold_of = np.repeat(np.arange(5), 40)
    rng = np.random.default_rng(0)
    base = rng.uniform(0.01, 0.2, size=6)
    probs = np.empty((200, 6))
    for f in range(5):
        probs[fold_of == f] = base + rng.normal(0, 0.01, 6)
    aligned = _align_fold_levels(probs, fold_of)
    for j in range(6):
        assert len(np.unique(np.round(aligned[:, j], 12))) == 1


def test_align_fold_levels_preserves_within_fold_ranking():
    rng = np.random.default_rng(1)
    fold_of = np.repeat(np.arange(4), 25)
    probs = rng.random((100, 5)) * 0.4
    aligned = _align_fold_levels(probs, fold_of)
    for f in range(4):
        m = fold_of == f
        for j in range(5):
            np.testing.assert_array_equal(
                np.argsort(probs[m, j]), np.argsort(aligned[m, j])
            )


def test_cv_prevalence_baseline_scores_chance():
    """Without fold alignment this lands well below 0.5, biasing every model."""
    result = run_experiment(_cfg(protocol="cv", n_folds=4, align_folds=True))
    assert result.metrics["prevalence"]["macro_auroc"] == pytest.approx(0.5, abs=1e-6)


def test_cv_without_alignment_exposes_the_bias():
    result = run_experiment(_cfg(protocol="cv", n_folds=4, align_folds=False))
    assert result.metrics["prevalence"]["macro_auroc"] < 0.5


def test_holdout_prevalence_baseline_scores_chance():
    result = run_experiment(_cfg(protocol="holdout"))
    assert result.metrics["prevalence"]["macro_auroc"] == pytest.approx(0.5, abs=1e-6)


def test_cv_scores_more_genes_than_holdout():
    """The power argument for cross-validation, made concrete."""
    holdout = run_experiment(_cfg(protocol="holdout"))
    cv = run_experiment(_cfg(protocol="cv", n_folds=4))
    assert (
        cv.metrics["prevalence"]["n_genes_scored"]
        >= holdout.metrics["prevalence"]["n_genes_scored"]
    )


def test_deltas_against_burden_are_reported():
    result = run_experiment(_cfg(protocol="holdout"))
    for name, m in result.metrics.items():
        assert "delta_macro_vs_burden" in m
        assert "delta_macro_vs_reference" in m
    assert result.metrics["burden"]["delta_macro_vs_burden"] == pytest.approx(0.0)


def test_result_saves_expected_artifacts(tmp_path):
    result = run_experiment(_cfg(protocol="holdout", output_dir=str(tmp_path)))
    out = result.save()
    for f in (
        "config.json",
        "metrics.json",
        "provenance.json",
        "summary.csv",
        "per_gene_auroc.csv",
        "summary.txt",
    ):
        assert (out / f).exists(), f
    assert "macro_auroc" in (out / "summary.txt").read_text()


def test_invalid_protocol_rejected():
    with pytest.raises(ValueError, match="protocol"):
        ExperimentConfig(protocol="bogus")


def test_yaml_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nnot_a_field: 1\n")
    with pytest.raises(KeyError, match="unknown config keys"):
        ExperimentConfig.from_yaml(path)
