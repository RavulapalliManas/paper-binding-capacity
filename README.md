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
