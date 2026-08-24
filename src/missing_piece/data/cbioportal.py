"""Read an MSK-IMPACT study in cBioPortal's flat-file format.

The important subtlety this module exists to handle: a gene that is *not on the
panel a sample was sequenced with* was never interrogated. It is missing data,
not a wild-type call. Encoding it as 0 manufactures false negatives, which is
exactly the failure mode that would inflate a panel-completion result. Every
matrix produced here therefore carries a companion ``assayed`` mask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "NONSYNONYMOUS_VARIANT_CLASSIFICATIONS",
    "StudyFiles",
    "AlterationMatrix",
    "read_clinical",
    "read_mutations",
    "read_cna",
    "read_gene_panel_matrix",
    "build_alteration_matrix",
    "load_study",
]

#: MAF ``Variant_Classification`` values counted as non-synonymous coding events.
#: Matches the cBioPortal/OncoKB convention used for IMPACT mutation calls.
NONSYNONYMOUS_VARIANT_CLASSIFICATIONS: frozenset[str] = frozenset(
    {
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Nonstop_Mutation",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "In_Frame_Del",
        "In_Frame_Ins",
        "Splice_Site",
        "Splice_Region",
        "Translation_Start_Site",
        "Targeted_Region",
    }
)

_SILENT = frozenset({"Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "IGR", "RNA"})


def _read_cbio_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a cBioPortal TSV, skipping the leading ``#`` metadata rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(256)
    if head.lstrip().startswith("version https://git-lfs"):
        raise ValueError(
            f"{path} is a Git LFS pointer, not data. Fetch the real file "
            "(see scripts/fetch_msk_chord.sh) before loading the study."
        )
    n_comment = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                n_comment += 1
            else:
                break
    return pd.read_csv(
        path, sep="\t", skiprows=n_comment, low_memory=False, dtype=str, **kwargs
    )


@dataclass
class StudyFiles:
    """Locations of the flat files that make up a cBioPortal study."""

    root: Path
    clinical_sample: Path
    clinical_patient: Path | None
    mutations: Path | None
    cna: Path | None
    gene_panel_matrix: Path | None

    @classmethod
    def discover(cls, root: str | Path) -> "StudyFiles":
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(root)

        def opt(name: str) -> Path | None:
            p = root / name
            return p if p.exists() else None

        sample = opt("data_clinical_sample.txt")
        if sample is None:
            raise FileNotFoundError(f"{root}/data_clinical_sample.txt is required")
        return cls(
            root=root,
            clinical_sample=sample,
            clinical_patient=opt("data_clinical_patient.txt"),
            mutations=opt("data_mutations.txt") or opt("data_mutations_extended.txt"),
            cna=opt("data_cna.txt"),
            gene_panel_matrix=opt("data_gene_panel_matrix.txt"),
        )


def read_clinical(files: StudyFiles) -> pd.DataFrame:
    """Sample-level clinical table, joined to patient-level attributes."""
    sample = _read_cbio_table(files.clinical_sample)
    if "SAMPLE_ID" not in sample.columns:
        raise ValueError("data_clinical_sample.txt has no SAMPLE_ID column")
    sample = sample.set_index("SAMPLE_ID")

    if files.clinical_patient is not None:
        patient = _read_cbio_table(files.clinical_patient)
        if "PATIENT_ID" in patient.columns and "PATIENT_ID" in sample.columns:
            patient = patient.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")
            overlap = [c for c in patient.columns if c in sample.columns]
            patient = patient.drop(columns=overlap)
            sample = sample.join(patient, on="PATIENT_ID")
    return sample


def read_mutations(
    files: StudyFiles,
    genes: set[str] | None = None,
    nonsynonymous_only: bool = True,
) -> pd.DataFrame:
    """Long-form (sample, gene) table of mutation events."""
    if files.mutations is None:
        return pd.DataFrame(columns=["SAMPLE_ID", "GENE"])
    maf = _read_cbio_table(files.mutations)

    col_sample = "Tumor_Sample_Barcode"
    col_gene = "Hugo_Symbol"
    for col in (col_sample, col_gene):
        if col not in maf.columns:
            raise ValueError(f"{files.mutations} has no {col} column")

    if nonsynonymous_only:
        if "Variant_Classification" not in maf.columns:
            log.warning("no Variant_Classification column; keeping all mutation rows")
        else:
            vc = maf["Variant_Classification"].fillna("")
            keep = vc.isin(NONSYNONYMOUS_VARIANT_CLASSIFICATIONS)
            unknown = sorted(set(vc[~keep & ~vc.isin(_SILENT)].unique()))
            if unknown:
                log.info("dropping mutation classes: %s", unknown[:12])
            maf = maf[keep]

    out = pd.DataFrame(
        {
            "SAMPLE_ID": maf[col_sample].str.strip(),
            "GENE": maf[col_gene].str.strip().str.upper(),
        }
    ).dropna()
    if genes is not None:
        out = out[out["GENE"].isin(genes)]
    return out.drop_duplicates()


def read_cna(
    files: StudyFiles,
    genes: set[str] | None = None,
    deep_only: bool = True,
) -> pd.DataFrame:
    """Long-form (sample, gene) table of copy-number alterations.

    cBioPortal discrete CNA uses GISTIC-style codes: -2 deep deletion,
    -1 shallow deletion, 0 diploid, 1 gain, 2 high-level amplification. With
    ``deep_only`` (the default, and what MSK-IMPACT reporting uses) only the
    high-confidence |value| == 2 calls count as alterations.
    """
    if files.cna is None:
        return pd.DataFrame(columns=["SAMPLE_ID", "GENE"])
    cna = _read_cbio_table(files.cna)

    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in cna.columns else cna.columns[0]
    drop = [c for c in ("Entrez_Gene_Id", "Cytoband") if c in cna.columns]
    cna = cna.drop(columns=drop)
    cna[gene_col] = cna[gene_col].str.strip().str.upper()
    if genes is not None:
        cna = cna[cna[gene_col].isin(genes)]
    cna = cna.set_index(gene_col)

    values = cna.apply(pd.to_numeric, errors="coerce")
    threshold = 2 if deep_only else 1
    hits = values.abs() >= threshold
    stacked = hits.stack()
    stacked = stacked[stacked]
    if stacked.empty:
        return pd.DataFrame(columns=["SAMPLE_ID", "GENE"])
    idx = stacked.index
    return pd.DataFrame(
        {"SAMPLE_ID": idx.get_level_values(1), "GENE": idx.get_level_values(0)}
    ).drop_duplicates()


def read_gene_panel_matrix(files: StudyFiles) -> pd.DataFrame | None:
    """Per-sample panel assignment, one column per molecular profile."""
    if files.gene_panel_matrix is None:
        return None
    gpm = _read_cbio_table(files.gene_panel_matrix)
    if "SAMPLE_ID" not in gpm.columns:
        return None
    gpm = gpm.set_index("SAMPLE_ID")
    return gpm.apply(lambda s: s.str.strip().str.upper())


@dataclass
class AlterationMatrix:
    """Binary alteration calls plus the mask of what was actually assayed.

    ``values`` and ``assayed`` are aligned (samples x genes). ``values`` is only
    meaningful where ``assayed`` is True.
    """

    values: pd.DataFrame
    assayed: pd.DataFrame
    clinical: pd.DataFrame
    panel_of_sample: pd.Series
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.shape != self.assayed.shape:
            raise ValueError("values and assayed must have the same shape")
        if not self.values.index.equals(self.assayed.index):
            raise ValueError("values and assayed must share a sample index")
        if not self.values.columns.equals(self.assayed.columns):
            raise ValueError("values and assayed must share a gene index")

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]

    @property
    def n_genes(self) -> int:
        return self.values.shape[1]

    def subset_genes(self, genes: list[str]) -> "AlterationMatrix":
        present = [g for g in genes if g in self.values.columns]
        return AlterationMatrix(
            values=self.values[present].copy(),
            assayed=self.assayed[present].copy(),
            clinical=self.clinical,
            panel_of_sample=self.panel_of_sample,
            provenance=dict(self.provenance),
        )

    def subset_samples(self, samples: list[str]) -> "AlterationMatrix":
        keep = [s for s in samples if s in self.values.index]
        return AlterationMatrix(
            values=self.values.loc[keep].copy(),
            assayed=self.assayed.loc[keep].copy(),
            clinical=self.clinical.reindex(keep),
            panel_of_sample=self.panel_of_sample.reindex(keep),
            provenance=dict(self.provenance),
        )

    def alteration_rate(self) -> float:
        """Fraction of *assayed* gene-sample pairs carrying an alteration."""
        n = int(self.assayed.to_numpy().sum())
        if n == 0:
            return float("nan")
        return float((self.values.to_numpy() & self.assayed.to_numpy()).sum()) / n

    def summary(self) -> str:
        return (
            f"{self.n_samples} samples x {self.n_genes} genes; "
            f"assayed {self.assayed.to_numpy().mean():.1%} of pairs; "
            f"alteration rate {self.alteration_rate():.4%}"
        )


def build_alteration_matrix(
    files: StudyFiles,
    genes: list[str],
    panels: dict[str, "object"] | None = None,
    nonsynonymous_only: bool = True,
    deep_cna_only: bool = True,
    include_cna: bool = True,
    mutation_profile: str = "mutations",
    cna_profile: str = "cna",
) -> AlterationMatrix:
    """Assemble the binary alteration matrix and the assayed mask.

    ``panels`` maps panel stable id -> object with a ``gene_set`` attribute
    (a :class:`~missing_piece.panels.Panel`). When supplied together with the
    study's gene-panel matrix, ``assayed`` reflects true per-sample panel
    coverage. Without it every gene is assumed assayed for every sample, which
    is only safe on a cohort already restricted to one panel.
    """
    genes = [g.upper() for g in genes]
    clinical = read_clinical(files)
    samples = list(clinical.index)
    gene_set = set(genes)

    sample_index = pd.Index(samples)
    gene_index = pd.Index(genes)
    arr = np.zeros((len(samples), len(genes)), dtype=bool)

    events = []
    muts = read_mutations(files, genes=gene_set, nonsynonymous_only=nonsynonymous_only)
    events.append(muts)
    n_cna = 0
    if include_cna:
        cnas = read_cna(files, genes=gene_set, deep_only=deep_cna_only)
        n_cna = len(cnas)
        events.append(cnas)
    all_events = pd.concat(events, ignore_index=True).drop_duplicates()

    if not all_events.empty:
        row = sample_index.get_indexer(all_events["SAMPLE_ID"])
        col = gene_index.get_indexer(all_events["GENE"])
        ok = (row >= 0) & (col >= 0)
        arr[row[ok], col[ok]] = True
    values = pd.DataFrame(arr, index=sample_index, columns=gene_index)

    gpm = read_gene_panel_matrix(files)
    panel_of_sample = pd.Series("UNKNOWN", index=samples, name="PANEL", dtype=object)
    if gpm is not None:
        col = mutation_profile if mutation_profile in gpm.columns else gpm.columns[0]
        panel_of_sample = (
            gpm[col].reindex(samples).fillna("UNKNOWN").astype(object).rename("PANEL")
        )
    elif "GENE_PANEL" in clinical.columns:
        panel_of_sample = (
            clinical["GENE_PANEL"].str.upper().fillna("UNKNOWN").rename("PANEL")
        )

    assayed = pd.DataFrame(True, index=samples, columns=genes)
    if panels:
        coverage = {
            pid: np.array([g in p.gene_set for g in genes], dtype=bool)
            for pid, p in panels.items()
        }
        unknown_panels = sorted(set(panel_of_sample.unique()) - set(coverage))
        if unknown_panels:
            log.warning(
                "no gene list for panel(s) %s; treating their genes as assayed",
                unknown_panels[:8],
            )
        mask = np.stack(
            [
                coverage.get(pid, np.ones(len(genes), dtype=bool))
                for pid in panel_of_sample
            ]
        )
        assayed = pd.DataFrame(mask, index=samples, columns=genes)
        # A call cannot exist for a gene the assay never covered.
        values = values & assayed

    return AlterationMatrix(
        values=values,
        assayed=assayed,
        clinical=clinical,
        panel_of_sample=panel_of_sample,
        provenance={
            "study_root": str(files.root),
            "n_mutation_events": int(len(muts)),
            "n_cna_events": int(n_cna),
            "nonsynonymous_only": nonsynonymous_only,
            "deep_cna_only": deep_cna_only,
            "include_cna": include_cna,
            "panel_aware_mask": bool(panels),
        },
    )


def load_study(
    root: str | Path,
    genes: list[str],
    panel_dir: str | Path | None = None,
    **kwargs,
) -> AlterationMatrix:
    """Convenience wrapper: discover files, load panels, build the matrix."""
    from ..panels import load_panels

    files = StudyFiles.discover(root)
    panels = None
    for candidate in (panel_dir, root, Path(root) / "gene_panels"):
        if candidate is None:
            continue
        try:
            panels = load_panels(candidate)
            break
        except (FileNotFoundError, NotADirectoryError):
            continue
    if panels is None:
        log.warning("no gene-panel files found; assayed mask will be all-True")
    return build_alteration_matrix(files, genes=genes, panels=panels, **kwargs)
