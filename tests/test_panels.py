import pytest

from missing_piece.panels import (
    Panel,
    PanelPair,
    load_panel_file,
    load_panels,
    make_panel,
    write_panel_file,
)


def test_load_panel_file(small_study):
    panel = load_panel_file(small_study / "data_gene_panel_BIG.txt")
    assert panel.stable_id == "BIG"
    assert len(panel) == 6
    assert "GENEA" in panel


def test_load_panels_indexes_by_stable_id(small_study):
    panels = load_panels(small_study)
    assert set(panels) == {"SMALL", "BIG"}


def test_lfs_pointer_is_rejected_not_silently_parsed(tmp_path):
    path = tmp_path / "data_gene_panel_X.txt"
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\nsize 3122\n"
    )
    with pytest.raises(ValueError, match="LFS"):
        load_panel_file(path)


def test_panel_rejects_duplicate_genes():
    with pytest.raises(ValueError, match="duplicate"):
        Panel(stable_id="X", genes=("A", "B", "A"))


def test_panel_pair_splits_observed_and_target(small_study):
    pair = PanelPair.from_directory(small_study, "SMALL", "BIG")
    assert pair.observed_genes == ("GENEA", "GENEB", "GENEC")
    assert pair.target_genes == ("GENED", "GENEE", "GENEF")
    assert pair.validate() == []


def test_non_nested_panels_are_flagged():
    pair = PanelPair(
        observed=make_panel("S", ["A", "Z"]), target=make_panel("T", ["A", "B"])
    )
    warnings = pair.validate()
    assert any("not nested" in w for w in warnings)
    # Z has no ground truth in T, so it must not appear among the targets.
    assert "Z" not in pair.target_genes


def test_panel_roundtrip(tmp_path):
    panel = make_panel("P", ["G1", "G2"], description="d")
    reloaded = load_panel_file(write_panel_file(panel, tmp_path / "data_gene_panel_P.txt"))
    assert reloaded.genes == panel.genes
    assert reloaded.stable_id == panel.stable_id
