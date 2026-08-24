# missing-piece

Reconstructing unassayed tumour genomic profiles from targeted sequencing
panels — predicting the 164 genes on MSK-IMPACT **IMPACT505** that
**IMPACT341** does not assay, from the 341 genes it does.

Research code for the ESMO abstract *Predicting Whole Tumor Genomic Profiles
from Targeted Panels with Generative Models* (Hempel & Capuano). It implements
the modelling pipeline, a set of baselines the result has to beat to mean
anything, and an evaluation designed so that a null result looks like one.

> **Read [`docs/methods-review.md`](docs/methods-review.md) first.** It documents
> what this pipeline measured about the abstract's methodology — including the
> finding that pooled AUROC reaches **0.795 on data containing zero recoverable
> signal**, which overlaps the abstract's reported 0.77–0.79 range.

---

## Install

```bash
pip install -e ".[dev]"
pytest                       # 60+ tests, ~5 minutes
```

Python ≥3.10. CPU is fine; the full simulated experiment runs in ~10 minutes.

## Quickstart — no data required

Everything runs immediately on a simulated cohort matched to the abstract's
dimensions (2,226 patients, 341 → 164 genes, ~1.21% sparsity).

```bash
# Is the evaluation trustworthy? Fit every model under three regimes,
# two of which contain no signal to find.
python -m missing_piece controls

# Full experiment: all models, 5-fold CV, bootstrap CIs, permutation tests.
python -m missing_piece run -c configs/simulation_full.yaml

# The abstract's original protocol, for comparison.
python -m missing_piece run -c configs/holdout_as_submitted.yaml
```

## Running on real MSK-CHORD data

```bash
scripts/fetch_msk_chord.sh data/msk_chord_2024
python -m missing_piece panels data/msk_chord_2024/gene_panels \
    --observed IMPACT341 --target IMPACT505
python -m missing_piece run -c configs/msk_chord_nsclc.yaml
```

The fetch script needs outbound access to `cbioportal-datahub.s3.amazonaws.com`
and `github.com`, plus `git-lfs`. Restrictive corporate or sandboxed networks
often block the cBioPortal hosts — in that case run it somewhere with access and
copy the directory over.

Switching `study_dir` on in a config is the only change needed: the same models,
splits and metrics then run against real data.

---

## What the pipeline is careful about

**Genes a sample was never assayed for are missing, not wild-type.** A patient
sequenced with IMPACT341 has no IMPACT505-only calls at all. Scoring those as
"no alteration" adds 164 guaranteed negatives per patient. The loader carries an
`assayed` mask from the study's gene-panel matrix, and only samples whose panel
actually covers the targets are eligible for evaluation.

**One sample per patient.** Repeat samples from one patient are near-duplicates;
letting them straddle a split scores the model on memorisation.

**AUROC is reported three ways, because they disagree.** Per-gene (macro) AUROC
is prevalence-free — a patient-blind model scores exactly 0.5. Pooled and
within-patient AUROC are not, and reach ~0.80 on data with no signal at all.
Both are reported beside prevalence-adjusted twins.

**Burden is a baseline, not a result.** A hypermutated tumour has more
alterations everywhere, so counting altered genes on the small panel predicts
the large panel with no gene-specific information. `d_vs_burden` — macro AUROC
minus the burden-only baseline — is the number that supports or sinks the claim.

**Negative controls run in the same code path as the experiment.** The simulator
generates cohorts where the hypothesis is true (`full`), where only burden
couples the panels (`burden_only`), and where the target block is independent of
the patient entirely (`independent`). The last two must come back null.

---

## Models

| name | what it is | role |
|---|---|---|
| `prevalence` | each gene's training rate, ignoring the patient | the floor; 0.5 macro AUROC by construction |
| `burden` | logistic per gene on observed-panel burden only | separates "panel predicts genome" from "burden predicts burden" |
| `logistic` | L2 logistic per target gene on the full panel | linear read of co-alteration structure |
| `mlp` | shared-trunk multi-label network, BCE | the discriminative ceiling for per-gene AUROC |
| `flow_gaussian` | conditional flow matching in a ±1 relaxation | generative; marginals estimated by integrating trajectories |
| `flow_discrete` | flow matching on {0, 1, MASK} | generative; exact marginals in one forward pass at *t*=0 |

`flow_discrete` is the recommended generative variant: because every gene is
masked at *t* = 0, a single forward pass gives noise-free conditional marginals,
while ancestral unmasking still produces coherent joint profiles. The Gaussian
variant must Monte-Carlo its marginals, which quantises the probability and
costs AUROC for reasons that have nothing to do with model quality.

## Metrics

Beyond the three AUROCs: macro average precision, Brier, ECE, patient-level
bootstrap CIs, and a permutation test that shuffles patients to check the
predictions carry patient-specific information at all.

Because per-gene AUROC only scores marginals — exactly what a discriminative
model optimises — `eval/joint.py` adds the metrics a generative model should win
on: fidelity of sampled gene-gene co-occurrence structure, and the distance
between sampled and true per-patient burden distributions.

---

## Layout

```
src/missing_piece/
  panels.py            panel files, and the observed→target task definition
  data/
    cbioportal.py      study loader; the assayed-vs-altered distinction
    cohort.py          cancer-type filter, eligibility, patient dedup, covariates
    simulate.py        synthetic cohorts: full / burden_only / independent
    splits.py          patient-level stratified splits
  models/
    baselines.py       prevalence, burden, logistic, MLP
    flow.py            Gaussian and discrete conditional flow matching
    nets.py            FiLM-conditioned residual trunks, training loops
  eval/
    metrics.py         the AUROC family, calibration, bootstrap, permutation
    joint.py           co-occurrence fidelity, burden distribution distance
  experiment.py        holdout and cross-validated protocols, reporting
  validation.py        the negative-control battery
configs/               experiment definitions (simulated and real)
docs/methods-review.md review of the abstract's methodology
```

## Status

The methodology is validated end-to-end against known-signal and known-null
cohorts. **No result here is about MSK-CHORD** — the cBioPortal hosts were
unreachable from the environment this was built in, so every number comes from
simulation. Point `study_dir` at a real study download to change that.
