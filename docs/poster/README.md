# ESMO poster — 145 × 115 cm

`poster.html` is authored 1:1 in millimetres at 1450 × 1150 mm.

## Printing

Open in Chrome → **Print** → *Save as PDF*, with:

- **Paper size:** custom 1450 × 1150 mm (or "Manage custom sizes")
- **Scale:** 100% — **not** "Fit to page", which silently shrinks everything
- **Margins:** None
- **Background graphics:** ON, or every panel header prints white-on-white

The `@page` rule already declares the size, so most print dialogs pick it up.
Send the resulting PDF to the printer — do not send the HTML.

## Before you print

Every value I could not verify is marked with a **red dashed box**. They are
deliberately loud so a placeholder can never be mistaken for a result. Search the
file for `class="slot"` to find all of them:

| Slot | What to enter |
|---|---|
| ESMO ID | your abstract number |
| confirm split | the split actually used (the abstract's 75/15/15 sums to 105%) |
| Macro AUROC ×3 + CIs | from `results/<run>/summary.csv` |
| Δ vs burden-only | `d_vs_burden` for your best model |
| n genes | `genes` column — how many of 164 were scoreable |
| top genes | best per-gene AUROC from `per_gene_auroc.csv` |
| p-value | `perm_p` |
| per_gene.png | drop the figure from `scripts/make_figures.py` |
| references ×3 | MSK-CHORD, MSK-IMPACT, flow matching |
| disclosures | conflicts and funding, or "none declared" |

Also replace the bar widths in the Results table (`style="width:78%"` etc.) with
your measured macro AUROC values — they are illustrative until then.

## Generating the numbers

```bash
python -m missing_piece inspect data/msk_chord_2024
python -m missing_piece run -c configs/msk_chord_nsclc.yaml
python scripts/make_figures.py results/msk_chord_nsclc
```

`summary.csv` maps directly onto the Results and Effect size panels.

## Layout

Four columns under a full-width key-message strip, reading left to right:
problem → method → results → how we know the result is real. The evaluation
design panel is deliberately on the board: it is the strongest defence against
the first question a statistician will ask.
