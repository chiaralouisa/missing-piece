import numpy as np
import pytest

from missing_piece.eval.clinical import (
    clinical_summary,
    decision_curve,
    net_benefit,
    operating_points,
)


@pytest.fixture
def rare_outcome():
    rng = np.random.default_rng(0)
    n = 4000
    latent = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-4.4 + 1.2 * latent)))
    y = (rng.random(n) < p).astype(float)
    return y, p


def test_operating_points_ppv_and_sensitivity_are_consistent(rare_outcome):
    y, p = rare_outcome
    table = operating_points(y, p)
    assert (table["ppv"].between(0, 1)).all()
    assert (table["sensitivity"].between(0, 1)).all()
    # Flagging a larger fraction can only increase sensitivity.
    assert table["sensitivity"].is_monotonic_increasing


def test_ppv_beats_prevalence_when_the_model_ranks(rare_outcome):
    y, p = rare_outcome
    table = operating_points(y, p, flag_fractions=(0.05,))
    assert table.loc[0, "lift_over_prevalence"] > 1.5


def test_random_scores_give_no_lift_on_average(rare_outcome):
    """One draw is far too noisy to test: flagging 10% of a 1%-prevalence
    cohort puts only ~4 true positives in the flagged set, so a single random
    ranking swings the lift between 0 and 2. The expectation is what is 1.0."""
    y, _ = rare_outcome
    rng = np.random.default_rng(1)
    lifts = [
        operating_points(y, rng.random(y.size), flag_fractions=(0.1,)).loc[
            0, "lift_over_prevalence"
        ]
        for _ in range(200)
    ]
    assert np.mean(lifts) == pytest.approx(1.0, abs=0.1)


def test_net_benefit_of_flagging_nobody_is_zero(rare_outcome):
    y, _ = rare_outcome
    assert net_benefit(y, np.zeros_like(y), 0.05) == 0.0


def test_net_benefit_rejects_invalid_threshold(rare_outcome):
    y, p = rare_outcome
    with pytest.raises(ValueError, match="threshold"):
        net_benefit(y, p, 0.0)


def test_decision_curve_has_reference_strategies(rare_outcome):
    y, p = rare_outcome
    curve = decision_curve(y, p)
    assert {"net_benefit_model", "net_benefit_treat_all", "net_benefit_treat_none"} <= set(curve.columns)
    # Treat-all becomes harmful once the threshold exceeds prevalence.
    high = curve[curve["threshold"] > 3 * y.mean()]
    assert (high["net_benefit_treat_all"] < 0).all()


def test_clinical_summary_skips_genes_without_enough_positives():
    rng = np.random.default_rng(3)
    y = np.zeros((500, 3))
    y[:20, 0] = 1
    y[:2, 1] = 1
    p = rng.random((500, 3))
    out = clinical_summary(y, p, gene_names=["A", "B", "C"], min_positives=5)
    assert set(out["gene"]) == {"A"}
