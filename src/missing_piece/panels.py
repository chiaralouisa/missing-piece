"""Gene panel definitions and the panel-completion task.

Panels are read from cBioPortal ``data_gene_panel_*.txt`` files, which look like::

    stable_id: IMPACT505
    description: Targeted sequencing of 505 cancer-associated genes.
    gene_list: ABL1	ACVR1	AKT1	...

We never hard-code gene lists: the MSK-IMPACT panel contents are data, and a
stale or mis-remembered list would silently redefine the prediction task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "Panel",
    "PanelPair",
    "load_panel_file",
    "load_panels",
    "make_panel",
    "write_panel_file",
]


@dataclass(frozen=True)
class Panel:
    """An assay panel: a named set of genes that the assay interrogates."""

    stable_id: str
    genes: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if len(set(self.genes)) != len(self.genes):
            dupes = sorted({g for g in self.genes if self.genes.count(g) > 1})
            raise ValueError(f"{self.stable_id}: duplicate genes {dupes[:10]}")
        if not self.genes:
            raise ValueError(f"{self.stable_id}: empty gene list")

    def __len__(self) -> int:
        return len(self.genes)

    def __contains__(self, gene: object) -> bool:
        return gene in set(self.genes)

    @property
    def gene_set(self) -> frozenset[str]:
        return frozenset(self.genes)


def _parse_gene_list(raw: str) -> tuple[str, ...]:
    """Split a gene_list field. cBioPortal uses tabs; some exports use commas."""
    tokens = re.split(r"[\t,;\s]+", raw.strip())
    return tuple(t.strip().upper() for t in tokens if t.strip())


def load_panel_file(path: str | Path) -> Panel:
    """Parse one cBioPortal gene-panel file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.lstrip().startswith("version https://git-lfs"):
        raise ValueError(
            f"{path} is a Git LFS pointer, not the panel itself. Run "
            "`git lfs pull` (or download the study from cBioPortal) first."
        )

    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip()

    if "gene_list" not in fields:
        raise ValueError(f"{path}: no 'gene_list:' field found")

    stable_id = fields.get("stable_id") or path.stem.replace("data_gene_panel_", "")
    return Panel(
        stable_id=stable_id.upper(),
        genes=_parse_gene_list(fields["gene_list"]),
        description=fields.get("description", ""),
    )


def load_panels(directory: str | Path) -> dict[str, Panel]:
    """Load every ``data_gene_panel_*.txt`` in a directory, keyed by stable id."""
    directory = Path(directory)
    panels: dict[str, Panel] = {}
    for path in sorted(directory.glob("data_gene_panel_*.txt")):
        try:
            panel = load_panel_file(path)
        except ValueError:
            continue
        panels[panel.stable_id] = panel
    if not panels:
        raise FileNotFoundError(f"no readable gene-panel files in {directory}")
    return panels


@dataclass(frozen=True)
class PanelPair:
    """The completion task: predict ``target``-only genes from ``observed``.

    ``observed_genes`` are the model inputs (assayed by the small panel) and
    ``target_genes`` are the prediction outputs -- genes on the large panel that
    the small panel does not assay.
    """

    observed: Panel
    target: Panel

    @property
    def observed_genes(self) -> tuple[str, ...]:
        """Genes assayed by both panels, in target-panel order."""
        obs = self.observed.gene_set
        return tuple(g for g in self.target.genes if g in obs)

    @property
    def target_genes(self) -> tuple[str, ...]:
        """Genes on the large panel only -- the prediction targets."""
        obs = self.observed.gene_set
        return tuple(g for g in self.target.genes if g not in obs)

    @property
    def observed_only_genes(self) -> tuple[str, ...]:
        """Genes on the small panel that the large panel dropped."""
        tgt = self.target.gene_set
        return tuple(g for g in self.observed.genes if g not in tgt)

    def describe(self) -> str:
        lines = [
            f"observed panel : {self.observed.stable_id} ({len(self.observed)} genes)",
            f"target panel   : {self.target.stable_id} ({len(self.target)} genes)",
            f"model inputs   : {len(self.observed_genes)} genes on both panels",
            f"model outputs  : {len(self.target_genes)} genes on "
            f"{self.target.stable_id} only",
        ]
        dropped = self.observed_only_genes
        if dropped:
            lines.append(
                f"NOTE: {len(dropped)} gene(s) on {self.observed.stable_id} are absent "
                f"from {self.target.stable_id}; they are usable as inputs but have no "
                f"ground truth: {', '.join(dropped[:8])}"
                + (" ..." if len(dropped) > 8 else "")
            )
        return "\n".join(lines)

    def validate(self, strict: bool = False) -> list[str]:
        """Check the nesting assumption. Returns a list of human-readable warnings."""
        warnings: list[str] = []
        dropped = self.observed_only_genes
        if dropped:
            warnings.append(
                f"{len(dropped)} gene(s) on {self.observed.stable_id} are not on "
                f"{self.target.stable_id}: the panels are not nested."
            )
        if not self.target_genes:
            warnings.append(
                f"{self.target.stable_id} adds no genes over "
                f"{self.observed.stable_id}: nothing to predict."
            )
        if strict and warnings:
            raise ValueError("; ".join(warnings))
        return warnings

    @classmethod
    def from_directory(
        cls, directory: str | Path, observed: str, target: str
    ) -> "PanelPair":
        panels = load_panels(directory)
        missing = [p for p in (observed, target) if p.upper() not in panels]
        if missing:
            raise KeyError(
                f"panel(s) {missing} not found in {directory}; "
                f"available: {sorted(panels)}"
            )
        return cls(observed=panels[observed.upper()], target=panels[target.upper()])


def make_panel(stable_id: str, genes: Iterable[str], description: str = "") -> Panel:
    """Build a Panel from an in-memory gene list (used by the simulator)."""
    return Panel(stable_id=stable_id, genes=tuple(genes), description=description)


def write_panel_file(panel: Panel, path: str | Path) -> Path:
    """Write a Panel back out in cBioPortal gene-panel format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"stable_id: {panel.stable_id}\n"
        f"description: {panel.description}\n"
        f"gene_list: {chr(9).join(panel.genes)}\n",
        encoding="utf-8",
    )
    return path
