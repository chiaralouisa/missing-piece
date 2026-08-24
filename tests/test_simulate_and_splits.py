import numpy as np
import pytest

from missing_piece.data.simulate import SimulationConfig, simulate_cohort
from missing_piece.data.splits import SplitSpec, make_splits


def test_simulator_hits_requested_sparsity():
    cohort = simulate_cohort(SimulationConfig(target_sparsity=0.0121, seed=0))
    assert cohort.target_sparsity() == pytest.approx(0.0121, abs=0.002)


def test_simulator_reproduces_abstract_shape():
    cohort = simulate_cohort(SimulationConfig(seed=0))
    assert cohort.n_patients == 2226
    assert cohort.observed.shape[1] == 341
    assert cohort.target.shape[1] == 164


def test_regimes_share_the_observed_block():
    """Controls must differ only in the hypothesis, not in the model inputs."""
    a = simulate_cohort(SimulationConfig(regime="full", seed=5))
    b = simulate_cohort(SimulationConfig(regime="independent", seed=5))
    np.testing.assert_array_equal(a.observed.to_numpy(), b.observed.to_numpy())


def test_independent_regime_has_no_patient_specific_target_signal():
    cohort = simulate_cohort(SimulationConfig(regime="independent", seed=2))
    burden = cohort.observed.sum(axis=1).to_numpy()
    target_burden = cohort.target.sum(axis=1).to_numpy()
    r = np.corrcoef(burden, target_burden)[0, 1]
    assert abs(r) < 0.08


def test_full_regime_couples_observed_and_target():
    cohort = simulate_cohort(SimulationConfig(regime="full", seed=2))
    burden = cohort.observed.sum(axis=1).to_numpy()
    target_burden = cohort.target.sum(axis=1).to_numpy()
    assert np.corrcoef(burden, target_burden)[0, 1] > 0.2


def test_simulator_is_deterministic():
    a = simulate_cohort(SimulationConfig(seed=11))
    b = simulate_cohort(SimulationConfig(seed=11))
    np.testing.assert_array_equal(a.target.to_numpy(), b.target.to_numpy())


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        SplitSpec(train=0.75, val=0.15, test=0.15)


def test_splits_are_disjoint_and_exhaustive():
    cohort = simulate_cohort(SimulationConfig(n_patients=500, seed=0))
    s = make_splits(cohort, SplitSpec(seed=0))
    ids = [set(p.patient_ids) for p in (s.train, s.val, s.test)]
    assert set.union(*ids) == set(cohort.patient_ids)
    assert sum(len(i) for i in ids) == cohort.n_patients


def test_stratification_balances_burden_across_splits():
    cohort = simulate_cohort(SimulationConfig(n_patients=1500, seed=0))
    s = make_splits(cohort, SplitSpec(seed=0, stratify=True))
    means = [p.observed.sum(axis=1).mean() for p in (s.train, s.val, s.test)]
    assert max(means) - min(means) < 0.6
