# The Capacity of In-Context Binding: a Recipe Split, a Scaling Law, and the Price of Memory

Code, committed data, and paper source for the paper. Manas Venkata Sai Ravulapalli and
Samrath Chadha, Efficient Computation Inc., 2026. This revision merges the earlier
"Binding Capacity Tracks the Pretraining Recipe, Not Scale Alone" with the capacity-law
companion into one paper; the pre-merge version is retained in this repository's history.

Every number in the paper is a LaTeX macro generated from a committed results file. No number
in the prose is typed by hand. This repository contains the full chain: the released capacity
table and robustness curves (`results/figdata.json`), the statistics
(`results/capacity_stats.json`, written by `analysis/capacity_stats.py`), the continuous
re-measurement (`results/capacity_trn1_summary.json`, the `K50` scaling-law fit in
`results/capacity_alpha_fit.json` with its family-clustered and leave-one-model-out
diagnostics from `analysis/capacity_alpha_fit.py`), the regime boundary
(`results/cross_campaign_capacity.json`), the cost-of-load fits
(`results/extra_analyses.json`, `results/devaxis_analyses.json`), the token-denominated
conversion (`results/token_costs.json`, derived by `analysis/token_costs.py` from the raw
run records of the two training codebases), the number pipeline (`analysis/make_numbers.py`
writes `numbers.tex`), the figure generators (`figs/`, `analysis/make_p456_figs.py`), and
the paper source (`p3_capacity.tex`).

## Reproduce the paper from the committed data

Requirements: Python 3.10+, `numpy`, `scipy`, `matplotlib`, and a LaTeX distribution with
`pdflatex`.

```bash
# 1. Regenerate every numeric macro from the committed results
python analysis/make_numbers.py          # rewrites numbers.tex byte-identically

# 2. Regenerate the figures (the generators audit themselves for text/data collisions)
cd figs && python gen_capacity_figs.py && cd ..
python analysis/make_p456_figs.py        # the capacity-law and cost-of-load figures

# 3. Compile the paper
pdflatex p3_capacity.tex && pdflatex p3_capacity.tex
```

The three steps are independent: the committed `numbers.tex`, `figs/*.pdf`, and
`figs_p456/*.pdf` already match the committed results, so step 3 alone rebuilds the paper as
released.

`analysis/token_costs.py` documents how `results/token_costs.json` was derived from the raw
run records (batch size and sequence length per pool, with the sequence rule asserted against
every record); the raw records themselves live in the two training codebases and are
available on request. Per-claim epistemic status (pre-declared vs post-hoc) is recorded in
`results/claim_ledger.json`.

## Analyses for the LP4FM 2026 camera-ready

The camera-ready version (NeurIPS 2026 workshop on Linguistic Principles for Foundation Models)
adds analyses of saved results. None of them runs a model. Each script writes one results file,
and its `--emit DIR` option writes the TeX macros and tables that the paper uses.

| Script | Reads | Writes |
|---|---|---|
| `analysis/p3_revision_checks.py` | `results/capacity_trn1_summary.json`, `results/capacity_x/`, `results/capacity_trn1/`, `results/figdata.json` | `results/p3_revision_checks.json`: the guess among the listed values, tokenizer replay, censoring-aware and non-embedding size fits, the held-out test of the single-binding account of distractors, first-round disclosure |
| `analysis/p3_model_intervals.py` | `results/capacity_x/`, `results/capacity_trn1_summary.json`, `results/p3_revision_checks.json` | `results/p3_model_intervals.json`: per-model K50 intervals from resampling seeds, and the size fit with seed and family resampling |
| `analysis/p3_development.py` | `results/qgfix*/step*/qgeom.json`, `results/dynamics/em_step*/`, `results/legibility_x/`, `results/legibility_trn1/leg-p14b-K4D0.json` | `results/p3_development.json`: probe accuracy across Pythia pretraining, probes and steering on wrong answers, final-checkpoint probes without distractors |
| `analysis/make_p3_revision_figs.py` | the files above | `figs_p456/fig_p3_capacity.pdf`, `fig_p3_interference.pdf`, `fig_p3_development.pdf` |

The generators of the saved runs are included for reading: `analysis/neuron_capacity.py`
(capacity curves), `analysis/qgeom.py` and `analysis/qgeom_8ffd7c06ecee.py` (probe accuracy
across checkpoints; each run records the hash of the version that produced it, and the two
versions compute the probe accuracy the same way), `analysis/steer.py` (probes and steering on
wrong answers), and `analysis/neuron_legibility.py` with `analysis/trilogy_tasks.py`
(four-binding runs without distractors).

```bash
python analysis/p3_revision_checks.py --emit DIR
python analysis/p3_model_intervals.py --emit DIR
python analysis/p3_development.py --emit DIR
python analysis/make_p3_revision_figs.py
```
