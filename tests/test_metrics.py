import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from missing_piece.eval.metrics import (
    auroc_columnwise,
    average_precision_columnwise,
    bootstrap_metric,
    evaluate_predictions,
    macro_auroc,
    permutation_test,
    pooled_auroc,
    pooled_auroc_prevalence_adjusted,
    residualize_by_gene,
    within_patient_auroc,
    within_patient_auroc_prevalence_adjusted,
)


@pytest.fixture
def signal_data():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(600, 1))
    loading = rng.normal(size=(1, 25))
    base = rng.uniform(0.02, 0.25, size=25)
    logit = np.log(base / (1 - base)) + latent @ loading
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random((600, 25)) < p).astype(float)
    return y, p


@pytest.fixture
def prevalence_only_data():
    """Target block statistically independent of the patient."""
    rng = np.random.default_rng(1)
    prev = rng.uniform(0.01, 0.3, size=40)
    y = (rng.random((800, 40)) < prev).astype(float)
    scores = np.tile(prev, (800, 1))
    return y, scores


def test_auroc_matches_sklearn(signal_data):
    y, p = signal_data
    mine = auroc_columnwise(y, p)
    ref = np.array([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])])
    np.testing.assert_allclose(mine, ref, atol=1e-12)


def test_average_precision_matches_sklearn(signal_data):
    y, p = signal_data
    mine = average_precision_columnwise(y, p)
    ref = np.array([average_precision_score(y[:, j], p[:, j]) for j in range(y.shape[1])])
    np.testing.assert_allclose(mine, ref, atol=1e-12)


def test_pooled_matches_sklearn(signal_data):
    y, p = signal_data
    assert pooled_auroc(y, p) == pytest.approx(roc_auc_score(y.ravel(), p.ravel()))


def test_auroc_is_nan_without_both_classes():
    y = np.zeros((10, 2))
    y[:, 1] = 1
    scores = np.random.default_rng(0).random((10, 2))
    assert np.isnan(auroc_columnwise(y, scores)).all()


def test_macro_auroc_is_half_for_patient_blind_scores(prevalence_only_data):
    """The central property: prevalence alone cannot score above chance."""
    y, scores = prevalence_only_data
    assert macro_auroc(y, scores, min_positives=5) == pytest.approx(0.5, abs=1e-9)


def test_pooled_auroc_is_inflated_by_prevalence(prevalence_only_data):
    """The confound the abstract's headline number is exposed to."""
    y, scores = prevalence_only_data
    assert pooled_auroc(y, scores) > 0.65
    assert within_patient_auroc(y, scores) > 0.65


def test_prevalence_adjustment_removes_the_confound(prevalence_only_data):
    y, scores = prevalence_only_data
    assert pooled_auroc_prevalence_adjusted(y, scores) == pytest.approx(0.5, abs=0.02)
    assert within_patient_auroc_prevalence_adjusted(y, scores) == pytest.approx(0.5, abs=0.02)


def test_prevalence_adjustment_preserves_real_signal(signal_data):
    y, p = signal_data
    assert pooled_auroc_prevalence_adjusted(y, p) > 0.6


def test_macro_auroc_invariant_under_residualization(signal_data):
    y, p = signal_data
    assert macro_auroc(y, p) == pytest.approx(macro_auroc(y, residualize_by_gene(p)))


def test_permutation_test_detects_no_signal(prevalence_only_data):
    y, scores = prevalence_only_data
    out = permutation_test(y, scores, n_permutations=100, seed=0)
    assert out["p_value"] > 0.05


def test_permutation_test_detects_signal(signal_data):
    y, p = signal_data
    out = permutation_test(y, p, n_permutations=100, seed=0)
    assert out["p_value"] <= 0.02


def test_bootstrap_interval_brackets_point_estimate(signal_data):
    y, p = signal_data
    out = bootstrap_metric(y, p, n_boot=100, seed=0)
    assert out["lo"] <= out["point"] <= out["hi"]


def test_evaluate_predictions_reports_gene_coverage(signal_data):
    y, p = signal_data
    res = evaluate_predictions(y, p, gene_names=[f"G{i}" for i in range(25)])
    assert res.n_genes_scored <= 25
    assert res.n_positive_pairs == int(y.sum())
    assert set(res.to_dict(include_per_gene=True)["per_gene_auroc"]) == {
        f"G{i}" for i in range(25)
    }


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="shape"):
        evaluate_predictions(np.zeros((5, 3)), np.zeros((5, 4)))


def test_residualization_of_constant_columns_is_exactly_zero():
    """Guards the ULP trap: constant columns must not leave rankable offsets."""
    scores = np.tile(np.array([0.01, 0.13, 0.4, 0.29]), (500, 1))
    assert np.count_nonzero(residualize_by_gene(scores)) == 0


def test_prevalence_baseline_scores_exactly_half_when_adjusted():
    rng = np.random.default_rng(7)
    prev = rng.uniform(0.01, 0.35, size=30)
    y = (rng.random((900, 30)) < prev).astype(float)
    scores = np.tile(prev, (900, 1))
    assert pooled_auroc_prevalence_adjusted(y, scores) == pytest.approx(0.5, abs=1e-12)
    assert within_patient_auroc_prevalence_adjusted(y, scores) == pytest.approx(0.5, abs=1e-12)
