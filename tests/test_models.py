import numpy as np
import pytest

from missing_piece.data.simulate import SimulationConfig, simulate_cohort
from missing_piece.data.splits import SplitSpec, make_splits
from missing_piece.eval.metrics import macro_auroc
from missing_piece.models import MODEL_REGISTRY, build_model

FAST = {
    "mlp": {"epochs": 15, "patience": 5, "hidden": (64,)},
    "flow_discrete": {"epochs": 15, "patience": 5, "width": 64, "depth": 2},
    "flow_gaussian": {
        "epochs": 15,
        "patience": 5,
        "width": 64,
        "depth": 2,
        "n_samples_for_marginals": 8,
        "n_sampling_steps": 5,
    },
}


@pytest.fixture(scope="module")
def splits():
    cohort = simulate_cohort(
        SimulationConfig(n_patients=400, n_observed_genes=40, n_target_genes=20, seed=3)
    )
    return make_splits(cohort, SplitSpec(seed=3))


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_model_fits_and_predicts_valid_probabilities(name, splits):
    model = build_model(name, **FAST.get(name, {}))
    model.fit(splits.train, splits.val)
    probs = model.predict_proba(splits.test)
    assert probs.shape == splits.test.target.shape
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_model_samples_are_binary_and_shaped(name, splits):
    model = build_model(name, **FAST.get(name, {}))
    model.fit(splits.train, splits.val)
    draws = model.sample(splits.test, n_samples=3, seed=0)
    assert draws.shape == (3, *splits.test.target.shape)
    assert set(np.unique(draws)).issubset({0.0, 1.0})


def test_prevalence_baseline_is_constant_across_patients(splits):
    model = build_model("prevalence").fit(splits.train)
    probs = model.predict_proba(splits.test)
    assert np.allclose(probs, probs[0])
    # And therefore scores exactly chance on the prevalence-free metric.
    y = splits.test.target.to_numpy(dtype=float)
    assert macro_auroc(y, probs, min_positives=1) == pytest.approx(0.5, abs=1e-9)


def test_prevalence_baseline_tracks_training_rates(splits):
    model = build_model("prevalence", smoothing=0.0).fit(splits.train)
    np.testing.assert_allclose(
        model.predict_proba(splits.test)[0], splits.train.target.mean(axis=0).to_numpy()
    )


def test_burden_baseline_only_sees_burden(splits):
    """Two patients with equal observed burden must get identical predictions."""
    model = build_model("burden").fit(splits.train)
    test = splits.test
    burden = test.observed.sum(axis=1)
    counts = burden.value_counts()
    repeated = counts[counts >= 2].index
    if len(repeated) == 0:
        pytest.skip("no two test patients share an observed burden")
    probs = model.predict_proba(test)
    idx = np.flatnonzero(burden.to_numpy() == repeated[0])[:2]
    np.testing.assert_allclose(probs[idx[0]], probs[idx[1]])


def test_discrete_flow_marginals_agree_with_its_own_samples(splits):
    """predict_proba must be the marginal of the joint the sampler draws from."""
    model = build_model("flow_discrete", **FAST["flow_discrete"]).fit(
        splits.train, splits.val
    )
    probs = model.predict_proba(splits.test)
    draws = model.sample(splits.test, n_samples=200, seed=0)
    empirical = draws.mean(axis=0)
    # Per-gene cohort means: sampling noise averages out over patients.
    assert np.abs(empirical.mean(axis=0) - probs.mean(axis=0)).max() < 0.05


def test_unknown_model_name_is_rejected():
    with pytest.raises(KeyError, match="unknown model"):
        build_model("does_not_exist")


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        build_model("flow_discrete").predict_proba(
            simulate_cohort(SimulationConfig(n_patients=20, n_observed_genes=5, n_target_genes=3))
        )
