# ESMO poster — 145 × 115 cm

`poster.html` is authored 1:1 in millimetres at 1450 × 1150 mm.

## Getting a PDF

`poster.pdf` in this folder is ready to send to the printer: one page, exactly
1450 × 1150 mm. **Send the PDF, never the HTML.**

After editing your numbers, regenerate it:

```bash
python scripts/poster_to_pdf.py            # writes docs/poster/poster.pdf
python scripts/poster_to_pdf.py --check    # verify size and page count only
```

This drives the export headlessly, so the page size is passed explicitly and the
result is verified before you get it. That matters: browsers are inconsistent
about honouring a custom `@page` size, and "Fit to page" silently rescales a
1.45 m board down to A4.

If you would rather print from the browser: Chrome → **Print** → *Save as PDF*,
paper size custom 1450 × 1150 mm, **Scale 100%** (not "Fit to page"), margins
None, **Background graphics ON** — otherwise every panel header prints
white-on-white.

## Filling in your numbers

Open `poster.html` in any text editor. Near the top there is a single block
marked **FILL IN YOUR NUMBERS HERE** — that is the only part you edit:

```js
const POSTER = {
  esmoId:        "1234P",
  results: [
    { model: "Flow matching",  macro: 0.731, ci: "0.716–0.747" },
    ...
  ],
  deltaVsBurden: "+0.106",
  nGenesScored:  138,
  ...
};
```

Anything left as `null` keeps its red dashed placeholder, so a value you forgot
can never print as if it were a result. The on-screen bar tells you how many are
still outstanding. Bar widths in the results table are computed from the macro
AUROC you enter, so the chance hairline lines up by construction.

What each field wants:

| Field | Where it comes from |
|---|---|
| `esmoId` | your abstract number |
| `split` / `splitConfirmed` | the split actually used — the abstract's 75/15/15 sums to 105%. Set `splitConfirmed: true` to remove the warning |
| `results[].macro` / `.ci` | `macro_auroc`, `ci_lo`–`ci_hi` in `results/<run>/summary.csv` |
| `deltaVsBurden` | `d_vs_burden` for your best model |
| `nGenesScored` | the `genes` column — how many of 164 were scoreable |
| `topGenes` | highest `auroc` rows in `per_gene_auroc.csv` |
| `permP` | `perm_p` |
| `perGeneFigure` | `"per_gene.png"`, copied next to `poster.html` |
| `ref1`–`ref3` | MSK-CHORD, MSK-IMPACT, flow matching |
| `disclosures` | conflicts and funding, or `"None declared."` |

## Generating the numbers

```bash
python -m missing_piece inspect data/msk_chord_2024
python -m missing_piece run -c configs/msk_chord_nsclc.yaml
python scripts/make_figures.py results/msk_chord_nsclc
```

`summary.csv` maps directly onto the Results and Effect size panels.

For the clinical framing, add an actionable-gene list and per-gene FDR to the
config before running:

```yaml
actionable_genes_file: panels/actionable_nsclc_example.txt   # replace with yours
per_gene_permutations: 300
```

## Layout

Four columns under a full-width key-message strip, reading left to right:
problem → method → results → how we know the result is real. The evaluation
design panel is deliberately on the board: it is the strongest defence against
the first question a statistician will ask.
