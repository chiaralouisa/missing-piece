import numpy as np
import pytest

from missing_piece.eval.clinical import (
    actionable_subset_report,
    load_actionable_genes,
)
from missing_piece.eval.metrics import benjamini_hochberg, per_gene_permutation_test


def test_load_actionable_genes_ignores_comments_and_blanks(tmp_path):
    path = tmp_path / "genes.txt"
    path.write_text("# a comment\n\nEGFR\n  met  \n\nALK # trailing\n")
    assert load_actionable_genes(path) == {"EGFR", "MET", "ALK"}


def test_load_actionable_genes_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("# only comments\n\n")
    with pytest.raises(ValueError, match="no gene symbols"):
        load_actionable_genes(path)


def test_benjamini_hochberg_matches_scipy():
    from scipy.stats import false_discovery_control

    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.01, 12), rng.uniform(0, 1, 150)])
    np.testing.assert_allclose(
        benjamini_hochberg(p)["q_values"], false_discovery_control(p, method="bh"),
        atol=1e-12,
    )


def test_benjamini_hochberg_carries_nan_through():
    out = benjamini_hochberg(np.array([0.001, np.nan, 0.5, 0.02]))
    assert np.isnan(out["q_values"][1])
    assert out["n_tested"] == 3


def test_benjamini_hochberg_is_never_more_lenient_than_raw_p():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200)
    assert (benjamini_hochberg(p)["q_values"] >= p - 1e-12).all()


def test_per_gene_permutation_finds_nothing_in_noise():
    rng = np.random.default_rng(2)
    y = (rng.random((400, 20)) < 0.15).astype(float)
    scores = rng.random((400, 20))
    out = per_gene_permutation_test(y, scores, n_permutations=80, seed=0)
    # With no signal, essentially nothing should survive FDR control.
    assert out["n_significant"] <= 1


def test_per_gene_permutation_finds_real_signal():
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(700, 1))
    loading = rng.normal(size=(1, 15))
    base = rng.uniform(0.08, 0.3, size=15)
    prob = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + latent @ loading)))
    y = (rng.random((700, 15)) < prob).astype(float)
    out = per_gene_permutation_test(y, prob, n_permutations=80, seed=0)
    assert out["n_significant"] >= 8


def test_actionable_subset_restricts_to_the_named_genes():
    rng = np.random.default_rng(4)
    genes = ["EGFR", "MET", "FOO1", "FOO2", "FOO3"]
    y = (rng.random((300, 5)) < 0.2).astype(float)
    p = rng.random((300, 5))
    out = actionable_subset_report(y, p, genes, {"EGFR", "MET", "ALK"}, min_positives=3)
    assert out["n_actionable_in_panel"] == 2
    assert out["matched_genes"] == ["EGFR", "MET"]
    assert "ALK" in out["unmatched_actionable_genes"]


def test_actionable_subset_handles_no_overlap():
    y = np.zeros((10, 2)); y[:5, 0] = 1
    p = np.random.default_rng(0).random((10, 2))
    out = actionable_subset_report(y, p, ["FOO", "BAR"], {"EGFR"})
    assert out["n_actionable_in_panel"] == 0
    assert "note" in out
