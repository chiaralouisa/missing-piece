# Running the real MSK-CHORD analysis

Start to finish, on your own machine. Roughly 30 minutes of waiting, ~15 minutes
of your attention.

You have to do this locally: Claude Code web sessions sit behind an egress policy
that returns 403 for `cbioportal.org` and the datahub S3 bucket, so the data
cannot be fetched from there.

---

## 0. Prerequisites

| Need | Check | If missing |
|---|---|---|
| Python ≥ 3.10 | `python3 --version` | python.org or `brew install python` |
| git-lfs | `git lfs version` | https://git-lfs.com, then `git lfs install` |
| Disk | `df -h .` | ~10 GB free for the download and extraction |

`git-lfs` is the one people forget. Without it the gene-panel files arrive as
130-byte pointer stubs. The loader detects that and refuses them rather than
guessing, but you will have wasted the download.

---

## 1. Get the code

```bash
git clone https://github.com/chiaralouisa/missing-piece
cd missing-piece
git checkout claude/genomic-profile-prediction-p8jum2
pip install -e ".[dev]"
pytest -q            # ~1 min; expect 98 passed
```

If `pytest` fails, stop — everything downstream assumes it passes.

---

## 2. Download the data

```bash
scripts/fetch_msk_chord.sh
```

Several GB, so expect a wait. The script resumes a partial download, checks your
prerequisites before starting, and verifies the result. To re-verify later
without re-downloading:

```bash
scripts/fetch_msk_chord.sh --check data/msk_chord_2024
```

If your network blocks the datahub, download `msk_chord_2024.tar.gz` by hand from
<https://www.cbioportal.org/datasets>, put it in `data/`, and re-run the script —
it will reuse the tarball instead of fetching it.

---

## 3. Inspect before you model

**This is the step that matters.** It loads the study without fitting anything
and tells you what your cohort actually is.

```bash
python -m missing_piece inspect data/msk_chord_2024
```

Read the cohort cascade:

```
after loaded                       25040 samples
after cancer_type_filter            ???? samples
after target_panel_coverage         ???? samples   <- IMPACT341 patients drop here
after one_sample_per_patient        ???? samples
```

### What to check, and what it means

**Does `after one_sample_per_patient` land near 2,226?**
If yes, your original cohort was IMPACT505-sequenced patients and finding 03 in
the methods review is already handled — say so explicitly in the paper.

If instead your 2,226 matches a number *before* `target_panel_coverage`, then the
original run probably included IMPACT341-sequenced patients. Those patients have
no IMPACT505-only calls at all, so every one of the 164 target genes would have
been scored as a guaranteed negative — 164 false negatives per patient. That
would inflate every metric and needs re-running.

**Does target sparsity come out near 1.21%?**
That is the abstract's figure. A large discrepancy means the alteration encoding
differs — check whether CNAs are included and whether shallow (±1) calls count.

**How many of 164 genes clear 5 positives?**
The report says. If it is under half, the `NOTE` in the output will tell you to
use cross-validation, and it is right.

**Which clinical columns are listed as "absent"?**
Those covariates are silently skipped. MSK-CHORD's NLP-derived fields may be
named differently than expected; adding one is a one-line change in
`data/cohort.py:_build_covariates`.

---

## 4. Run the analysis

```bash
python -m missing_piece run -c configs/msk_chord_nsclc.yaml
```

5-fold cross-validation over six models. Expect 30–60 minutes on a laptop CPU.

Before running, consider adding to the config:

```yaml
per_gene_permutations: 300                                   # per-gene FDR
actionable_genes_file: panels/actionable_nsclc_example.txt   # replace with yours
```

The example actionable list is a placeholder, not a curated one — export Level
1/2 NSCLC genes from OncoKB or use your tumour board's list.

---

## 5. Read the output

`results/msk_chord_nsclc/summary.csv` is the table. Read it in this order:

1. **`prevalence` row, `macro_auroc` column — must be exactly 0.500.**
   If it is not, something is wrong with the pipeline, not with your data.
2. **`d_vs_burden` for your best model.** This is the headline. It is the
   gene-specific information the panel adds beyond "how mutated is this tumour".
   If it is near zero, the finding reduces to burden predicting burden.
3. **`genes`** — how many of 164 entered the macro average. Report this number.
4. **`macro_auroc` + CI** — the prevalence-free performance figure.
5. **`pooled_auroc` vs `pooled_adj`** — the gap between them is the size of the
   prevalence confound in your data.

Then:

```bash
python scripts/make_figures.py results/msk_chord_nsclc
```

---

## 6. Put it on the poster

Open `docs/poster/poster.html`, fill in the `POSTER` block at the top from
`summary.csv`, then:

```bash
python scripts/poster_to_pdf.py
```

Red dashed boxes mark anything still unfilled, so a placeholder cannot print as a
result. See `docs/poster/README.md` for the field-by-field mapping.

---

## If something looks wrong

Send me the output of `inspect` — it is a few dozen lines and says more about
what went wrong than any error message will. Or export the derived cohort, which
is a few MB with no free text or dates, and I can run the analysis directly:

```bash
python -m missing_piece export data/msk_chord_2024 -o cohort_nsclc.npz
```

(Check the CC BY-NC terms before sharing it; `--anonymize` strips sample ids.)
