# Importing MSK-CHORD

Three ways to get real data into this pipeline, in order of preference.

---

## Why this document exists

The environment this code was built in cannot reach cBioPortal. Its egress
policy returns **403 at the proxy** for `www.cbioportal.org` and
`cbioportal-datahub.s3.amazonaws.com`, and the GitHub mirror of the datahub
serves only Git LFS *pointers* over the anonymous lane — so even the gene panel
definitions come back as 130-byte stubs rather than gene lists.

That is a property of the sandbox, not of the data. MSK-CHORD is public
(CC BY-NC), and any ordinary machine can fetch it.

---

## Route 1 — run it yourself, locally (simplest)

```bash
git clone https://github.com/chiaralouisa/missing-piece
cd missing-piece
git checkout claude/genomic-profile-prediction-p8jum2
pip install -e ".[dev]"

scripts/fetch_msk_chord.sh data/msk_chord_2024
python -m missing_piece inspect data/msk_chord_2024      # ALWAYS do this first
python -m missing_piece run -c configs/msk_chord_nsclc.yaml
```

`fetch_msk_chord.sh` needs `curl`, `git`, and **`git-lfs`** (the gene panel
files are LFS-tracked; without it you get pointer stubs and the loader will say
so rather than guessing).

### What `inspect` tells you, and why to run it first

It loads the study without fitting anything and reports what it found:

- which files are present, their size, and whether any is an **LFS pointer**
- which panels exist, and how many genes are in the observed/target split
- which clinical columns are present, and which covariates will be **silently
  skipped** because their column is absent
- the **panel assignment** per sample — and a warning if the gene-panel matrix
  is missing, because without it unassayed genes look wild-type
- the cohort size **after each filter**, so you can see exactly where patients go
- whether enough genes have enough positives to estimate per-gene AUROC at all

Example on a mixed-panel cohort:

```
## Cohort after filtering
  after loaded                       1200 samples
  after cancer_type_filter            986 samples
  after target_panel_coverage         562 samples     <- IMPACT341 patients drop out here
  after one_sample_per_patient        561 samples

## Feasibility
  target genes with >=5 positives    92 / 164
  expected positives per gene in a 15% holdout 0.8
```

That `target_panel_coverage` line is the one to check against your 2,226. It is
where patients sequenced on the smaller panel are removed, because they have no
ground truth for the genes being predicted.

---

## Route 2 — export the cohort and send me the file

If you would rather I ran the analysis, you do not need to move the 
study. The *modelling* cohort is a binary matrix — a few megabytes, no free 
text, no dates, no identifiers beyond the study's own de-identified sample ids:

```bash
python -m missing_piece export data/msk_chord_2024 -o cohort_nsclc.npz
# add --anonymize to replace sample ids with opaque indices
```

Then commit it to the repo (it is small enough) or attach it to a session:

```bash
git add cohort_nsclc.npz && git commit -m "Add derived NSCLC cohort" && git push
python -m missing_piece run -c configs/exported_cohort.yaml
```

The `.npz` contains: the observed and target binary matrices, gene names, the
covariate block, and provenance. Everything downstream is byte-identical to
running against the study directory — `configs/exported_cohort.yaml` just points
`cohort_file` at it.

> **Check the study's data-use terms before sharing an export.** It is still
> patient-level genomic data under CC BY-NC. `--anonymize` removes the sample
> ids; it does not change the licence, and genomic profiles are not
> anonymous in the legal sense.

---

## Route 3 — a session that can reach cBioPortal

If a Claude Code environment is configured with an egress policy allowing
`cbioportal-datahub.s3.amazonaws.com` and `www.cbioportal.org`, route 1 works
unchanged inside it. That is an environment setting, not something this code can
work around — see the network-policy section of the Claude Code on the web docs.

---

## What to check once it loads

Run `inspect` and compare against the abstract:

| check | what to look for |
|---|---|
| cohort size | does `after one_sample_per_patient` land near 2,226? |
| panel filter | are only IMPACT505-covered patients left? |
| sparsity | does target sparsity come out near 1.21%? |
| scoreable genes | how many of 164 clear 5 positives on the full cohort? |

If the cohort size matches 2,226 only *before* the panel-coverage filter, the
original analysis probably included IMPACT341-sequenced patients — see finding
03 in the methods review, because that would mean 164 guaranteed negatives per
patient.

---

## Column names this loader looks for

Real cBioPortal studies vary. The loader requires the first group and treats the
rest as optional, skipping absent covariates silently — which is why `inspect`
prints exactly which ones it found.

**Required**

| file | columns |
|---|---|
| `data_clinical_sample.txt` | `SAMPLE_ID`, `PATIENT_ID` |
| `data_mutations.txt` | `Hugo_Symbol`, `Tumor_Sample_Barcode`, `Variant_Classification` |
| `data_cna.txt` | `Hugo_Symbol` + one column per sample (GISTIC −2…2) |
| `data_gene_panel_matrix.txt` | `SAMPLE_ID` + one column per profile |

**Optional, used if present**

`ONCOTREE_CODE`, `CANCER_TYPE`, `SAMPLE_TYPE`, `SEX`, `SMOKING_STATUS`,
`TMB_NONSYNONYMOUS`, `MSI_SCORE`, `FRACTION_GENOME_ALTERED`,
`AGE_AT_SEQUENCING`, `CURRENT_AGE_DEID`.

If MSK-CHORD names something differently — likely for the NLP-derived clinical
fields — `inspect` will list it under "absent", and adding it is a one-line
change in `data/cohort.py:_build_covariates`.
