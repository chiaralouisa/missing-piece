# Methods review: predicting whole tumour genomic profiles from targeted panels

A review of the submitted ESMO abstract against what the pipeline in this
repository measures. Findings are ordered by how much they could change the
conclusion. Nothing here says the result is wrong — several findings say the
number as reported cannot distinguish a real result from a null one, which is a
different and fixable problem.

---

## 1. The reported AUROC may not measure what the conclusion claims

**The claim.** "The AUROC score ... are in the 0.77-0.79 range ... our method's
performance exceeded a prevalence-only baseline, indicating that the smaller
targeted panel does indeed contain information about alterations outside its
assayed gene set."

**The problem.** With 164 genes at 1.21% prevalence, the value of AUROC depends
entirely on how it is averaged, and the abstract does not say.

*Pooled* AUROC — every patient-gene pair in one ranking — is dominated by
between-gene prevalence differences. A gene altered in 12% of tumours supplies
mostly-positive pairs; a gene altered in 0.3% supplies mostly-negative ones. A
model that never looks at the patient and emits each gene's training prevalence
therefore ranks pairs well.

**Measured, not asserted.** In this repository's `independent` simulation
regime the target genes are constructed to be *statistically independent of the
patient* — there is, by construction, exactly zero recoverable signal. The
pipeline reports:

| metric | value under zero signal |
|---|---|
| pooled AUROC | **0.795** |
| within-patient AUROC | **0.809** |
| macro (per-gene) AUROC | **0.500** |

Pooled AUROC of 0.795 with nothing to find sits inside the abstract's reported
0.77–0.79 range. If the reported figure is pooled, it is consistent with a real
effect *and equally consistent with none*.

**What to do.** Report **macro (per-gene) AUROC** as the headline: it averages
AUROC computed within each gene, so a patient-blind model scores exactly 0.5 and
any lift is necessarily patient-specific. Keep pooled AUROC if you like, but
always beside (a) the prevalence-only baseline computed identically and (b) the
prevalence-adjusted version (`pooled_adj`), which removes each gene's mean score
and reads 0.5 under the null.

Note also that under macro AUROC the sentence "exceeded a prevalence-only
baseline" becomes nearly vacuous — that baseline is 0.5 by construction, so any
signal at all exceeds it. The informative comparison is the next finding.

*Implementation:* `eval/metrics.py`, `validation.py`. Reproduce with
`python -m missing_piece controls`.

---

## 2. The decisive baseline is mutational burden, and it is missing

Two very different mechanisms produce cross-panel predictability:

1. **Gene-specific co-alteration** — *EGFR*-mutant tumours look different from
   *KRAS*-mutant ones, so which genes are hit on the small panel tells you which
   are likely hit off it. This is the interesting claim.
2. **Mutational burden** — a hypermutated tumour has more alterations
   *everywhere*. Counting altered genes on the 341-gene panel predicts the count
   on the other 164 with no gene-specific information whatsoever.

Mechanism 2 alone will beat a prevalence-only baseline comfortably. So the
abstract's stated comparison cannot separate "the panel predicts the genome"
from "burden predicts burden" — and only the first supports the conclusion.

**What to do.** Add a burden-only baseline: one logistic regression per target
gene whose sole input is the number of altered genes on the observed panel (plus
TMB if you want to be generous to it). Report **macro AUROC minus burden
baseline** as the primary effect size. This repository does that as
`d_vs_burden` and runs a `burden_only` simulation regime confirming that when
burden is the only channel, nothing beats the burden baseline (best gain
+0.007).

Chromosomal proximity is a smaller version of the same issue: arm-level copy
number events co-occur because genes sit on the same arm. If CNAs drive much of
the signal, a cytoband-adjacency baseline is worth adding too.

*Implementation:* `models/baselines.py:BurdenBaseline`.

---

## 3. Cohort eligibility: a trap that manufactures true negatives

Patients sequenced with IMPACT341 were **never assayed** for the 164
IMPACT505-only genes. Their calls for those genes are missing data, not
wild-type. Encoding them as 0 adds 164 guaranteed negatives per patient and
inflates every metric.

The pipeline here refuses to do this: `data/cbioportal.py` carries an `assayed`
mask derived from the study's `data_gene_panel_matrix.txt`, and
`eligible_for_target_panel()` keeps only samples whose panel actually covers the
targets.

**Worth confirming in your run:** MSK-CHORD is ~25,000 tumours and its NSCLC
subset is far larger than 2,226. If your 2,226 came from restricting to
IMPACT505-sequenced patients, this is already handled — say so explicitly in the
methods, because a reviewer will ask. If it came from another filter, check this
first.

**A framing consequence.** Restricting to IMPACT505 patients means the
"IMPACT341 input" is a *retrospective mask* on IMPACT505 data, not real
IMPACT341 assay output. That is the right way to build the benchmark, but it is
not the deployment setting: genuinely 341-sequenced patients differ by era,
assay chemistry, and referral pattern. The abstract's claim of "broadening
access" rests on transfer across that gap, which this design does not test.
State it as a limitation, and if possible validate on true IMPACT341-era
patients for the genes both panels share.

---

## 4. Repeat samples per patient will leak across the split

MSK-IMPACT cohorts contain multiple samples from the same patient (MSK-CHORD:
~25,040 tumours from ~24,950 patients). Two samples from one patient share most
of their alterations. If one lands in train and the other in test, the model is
scored partly on memorisation.

**What to do.** Deduplicate to one sample per patient, or split by patient with
grouping. `data/cohort.py:one_sample_per_patient()` does the former
deterministically (primary over metastatic, then a stated tie-break). The
abstract says "held-out patients", which suggests this was handled — make it
explicit.

---

## 5. The split percentages sum to 105%

"trained on 75% of the available data, using 15% for further validation ...
assessed in the last 15% of held-out patients" — 75 + 15 + 15 = 105.

Presumably 70/15/15. Trivial to fix, but it is the kind of thing a reviewer
notices and it costs credibility on everything else. `SplitSpec` in this
repository raises on fractions that do not sum to 1, so the mistake cannot
survive into a run.

---

## 6. A 15% test split cannot support a 164-gene macro average

334 test patients x 1.21% prevalence ≈ **4 expected positives per gene**. Most
of the 164 genes will have too few positives for a per-gene AUROC to mean
anything, and a macro average then silently describes only the handful of common
genes — a biased, high-prevalence subset.

Measured here: with a single 15% holdout, **~50 of 164 genes** were scoreable at
a ≥5-positive threshold. Under 5-fold cross-validation with pooled out-of-fold
predictions, **138 of 164** were.

**What to do.** Use repeated K-fold CV and report metrics on pooled out-of-fold
predictions (`protocol: cv`). Always report how many genes entered the macro
average. If you keep the holdout for comparability with the abstract, report the
gene count alongside it.

---

## 7. Pooling out-of-fold predictions introduces a bias of its own

Found while building this pipeline, and worth knowing before you switch to CV.

A gene's training prevalence is estimated *excluding* the fold it is applied to.
A fold holding unusually many positives was trained on a cohort with
correspondingly fewer, so it receives systematically *lower* scores. Pool the
folds and the ranking runs backwards.

Concretely, the prevalence baseline — which must score exactly 0.5 — came out at
**0.424**, and every model inherited the same downward bias, distorting all
between-model comparisons.

The fix (`experiment.py:_align_fold_levels`) shifts each fold's logits so all
folds share a gene's mean, leaving within-fold ranking untouched. After it, the
prevalence baseline scores 0.500 exactly.

A related trap bit twice: **floating-point noise gets ranked**. Centring a
constant column leaves per-gene residues of ~1e-16, and AUROC ranks a
one-ULP difference as confidently as a difference of 0.5 — silently restoring
the prevalence ordering the centring existed to remove.
`quantize_by_column()` makes numerically indistinguishable scores tie. If you
implement prevalence adjustment or fold alignment yourself, check that your
prevalence baseline returns exactly 0.5; if it does not, this is why.

---

## 8. Nothing in the evaluation yet rewards the model for being generative

Per-gene AUROC scores **marginals**: P(gene g altered | observed panel), one
gene at a time. A discriminative multi-label classifier optimises exactly that
objective, so on this metric it is the ceiling, not the comparison. Measured on
the `full` regime under 5-fold CV, *both* discriminative baselines beat *both*
flow-matching variants on macro AUROC (logistic 0.732, MLP 0.727, discrete flow
0.690, Gaussian flow 0.655).

That is not an argument against the generative framing — it is an argument that
the framing needs a metric that reflects it. What a generative model uniquely
provides is a **joint** distribution: coherent whole profiles preserving
co-occurrence and mutual exclusivity, calibrated uncertainty, and the ability to
condition on partial information.

**What to do.** Report joint-structure metrics alongside AUROC
(`eval/joint.py`): correlation between sampled and observed gene-gene
co-alteration structure, and the distance between sampled and true
per-patient burden distributions. Independent draws from a discriminative
model's marginals are the right null — they cannot reproduce co-occurrence by
construction. If the flow model wins there while trailing slightly on AUROC,
that is a coherent and defensible story. If it wins on neither, the generative
machinery is not earning its place.

---

## 9. How you read probabilities out of a flow model changes the AUROC

Gaussian flow matching produces samples, not probabilities. Estimating a
marginal means integrating many trajectories and averaging, so the probability's
resolution is bounded by the sample count — with 64 samples nothing finer than
1/64 exists, ties proliferate, and AUROC pays for it. That is a **measurement
artifact of the readout**, not a property of the model, and it is easy to
mistake for the method underperforming.

Two mitigations, both implemented:

- **Expectation readout** (`readout="expected"`): average the continuous
  endpoint rather than its binarisation. Unbiased for `2P(alt) - 1` and not
  quantised.
- **Discrete flow matching** (`models/flow.py:DiscreteFlowMatching`): flow
  matching directly on {0, 1, MASK}. At *t* = 0 every gene is masked, so **one
  forward pass returns the exact conditional marginals** with no sampling noise,
  while ancestral unmasking still yields joint samples. For sparse binary
  genomics this is both the better estimator and roughly 2x cheaper.

If the abstract's "variants of neural network architecture and training
paradigm" spanned a 0.77–0.79 range, some of that spread may be readout
variance rather than genuine architectural difference. Worth checking before
attributing it to modelling choices.

---

## 10. AUROC is not a clinical claim

The conclusion says "clinically relevant genomic information". AUROC is
threshold-free and prevalence-blind; neither property survives contact with a
clinic. At 1.2% prevalence, a gene with AUROC 0.80 still has poor positive
predictive value at any sensitivity worth using.

**What to do before the clinical framing is defensible:**

- Report **PPV and sensitivity at usable operating points**, per gene, not just
  AUROC. Add decision-curve / net-benefit analysis.
- Report **calibration** (Brier, ECE) — a probability that will inform a
  decision has to be right, not merely well-ordered. Already reported here.
- Restrict a secondary analysis to **actionable genes** (OncoKB Level 1/2 among
  the 164). Mean AUROC over 164 genes of mixed relevance does not tell an
  oncologist anything actionable. Genes like *MET*, *RET*, *ERBB2*, *KRAS G12C*
  context genes and *STK11*/*KEAP1* carry the clinical weight.
- Correct for **multiple testing** across 164 genes when claiming per-gene
  significance.
- Frame the output as **triage / hypothesis generation** — "this tumour warrants
  reflex broad sequencing" — not as a substitute for assaying the gene. A
  predicted alteration is not a biomarker and must not gate therapy.

---

## Reference results (simulated `full` regime, 5-fold CV, 2,226 patients)

Produced by `configs/simulation_full.yaml`. These validate the **pipeline**, not
any claim about MSK-CHORD.

| model | macro AUROC [95% CI] | Δ vs burden | pooled | pooled adj. | genes scored |
|---|---|---|---|---|---|
| logistic | 0.732 [0.718, 0.746] | +0.107 | 0.855 | 0.764 | 138 |
| mlp | 0.727 [0.716, 0.747] | +0.102 | 0.867 | 0.763 | 138 |
| flow_discrete | 0.690 [0.681, 0.711] | +0.065 | 0.847 | 0.723 | 138 |
| flow_gaussian | 0.655 [0.642, 0.671] | +0.030 | 0.812 | 0.695 | 138 |
| burden | 0.625 [0.612, 0.639] | 0.000 | 0.824 | 0.629 | 138 |
| prevalence | **0.500** [0.500, 0.500] | -0.125 | **0.803** | 0.500 | 138 |

Read the last row first. A model that never looks at the patient scores **0.803
pooled** — inside the abstract's reported range — and **0.500** on every metric
that has had the prevalence channel removed. That contrast is the whole point of
this table.

---

## Suggested minimum reporting set

For each model, on held-out patients:

| quantity | why |
|---|---|
| macro (per-gene) AUROC + bootstrap CI | prevalence-free headline |
| Δ macro AUROC vs **burden-only** baseline | the effect size that supports the claim |
| Δ macro AUROC vs prevalence baseline | comparability with the abstract |
| number of genes entering the macro average | guards against a biased subset |
| pooled AUROC **and** its prevalence-adjusted twin | transparency about the confound |
| permutation-test p-value | evidence of patient-specific information |
| Brier / ECE | calibration, for any clinical framing |
| co-occurrence fidelity, burden KS | what the generative model is actually for |

All of these are produced by `python -m missing_piece run`.

---

## What is validated, and what is not

**Validated in this repository.** The pipeline detects real cross-panel signal
when it exists (`full` regime), reports no gene-specific gain when only burden
couples the panels (`burden_only`), and reports exactly chance when the target
block is independent of the patient (`independent`). Two bias sources found
during construction — pooled-CV fold offsets and ULP-level rank noise — are
fixed and covered by regression tests.

**Not validated.** Any claim about MSK-CHORD itself. The cBioPortal hosts are
unreachable from the environment this was built in, so every number here comes
from simulated cohorts matched to the abstract's dimensions (2,226 patients,
341 → 164 genes, ~1.21% sparsity). The simulator was built to reproduce the
abstract's *shape*, which means it can validate the **methodology** and cannot
say anything about the **finding**. Run `configs/msk_chord_nsclc.yaml` on the
real study to get that.
