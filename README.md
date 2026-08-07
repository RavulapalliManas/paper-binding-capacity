# Binding Capacity Tracks the Pretraining Recipe, Not Scale Alone

Code, committed data, and paper source for the paper. Manas Venkata Sai Ravulapalli and
Samrath Chadha, Efficient Computation Inc., 2026.

Every number in the paper is a LaTeX macro generated from a committed results file. No number
in the prose is typed by hand. This repository contains the full chain: the released capacity
table and robustness curves (`results/figdata.json`), the statistics (`results/capacity_stats.json`,
written by `analysis/capacity_stats.py`), the number pipeline (`analysis/make_numbers.py` writes
`numbers.tex`), the figure generators (`figs/`), and the paper source (`p3_capacity.tex`).

## Reproduce the paper from the committed data

Requirements: Python 3.10+, `numpy`, `scipy`, `matplotlib`, and a LaTeX distribution with
`pdflatex`.

```bash
# 1. Regenerate every numeric macro from the committed results
python analysis/make_numbers.py          # rewrites numbers.tex (421 macros, byte-identical)

# 2. Regenerate the figures (the generator audits itself for text/data collisions)
cd figs && python gen_capacity_figs.py && cd ..

# 3. Compile the paper
pdflatex p3_capacity.tex && pdflatex p3_capacity.tex
```

The three steps are independent: the committed `numbers.tex` and `figs/*.pdf` already match the
committed results, so step 3 alone rebuilds the paper as released.

## What produced the measurements

| Paper section | Source |
|---|---|
| The capacity table (k* per model; Figs. 1 and 2) | committed at `results/figdata.json["geom"]`; the original sweep predates this repository. `analysis/ksweep.py` is the re-measurement runner and `analysis/compare_ksweep.py` the agreement check against the frozen table (the open release item stated in the paper, Section 2) |
| Capacity statistics (medians, family-clustered CIs) | `analysis/capacity_stats.py` -> `results/capacity_stats.json` |
| Robustness D-sweep (Fig. 3) | committed at `results/figdata.json["robust"]`; `analysis/robust_dsweep.py` is the committed re-runner |
| Query-site control (Section 7) | `analysis/capacity_control.py` -> `results/capacity_control.json` |
| Declaration-site geometry diagnostics (Section 7) | committed at `results/dynamics/geo_*/geom.json`; the d_decl measurement script for the capacity table was not preserved, as disclosed in the paper (Section 7) |

Known data notes, all disclosed in the paper or in `figs/gen_capacity_figs.py`: the frozen
`figdata.json["robust"]` entries mis-tag three models' recipes (the figure generator joins
recipe and k* from the capacity table on the `model` field), and the robustness sweep's 0.5B
Qwen entry is the Instruct variant of the base model the capacity table measured.

`results/` is the full committed-results closure of the shared macro pipeline: `numbers.tex`
also carries the macros of the companion paper ("Outgrowing the Readout", same pipeline), so
regenerating it byte-identically requires the companion aggregates that ship here too. Each
results JSON records its own provenance where available (script `argv`, MD5 of the model's
embedding weights, MD5 of the producing script). Absolute machine paths inside recorded `argv`
strings are redacted to relative paths; nothing else in any results file is altered.

## Layout

```
p3_capacity.tex         paper source (bibliography embedded; compiles standalone)
numbers.tex             generated numeric macros -- do not edit by hand
analysis/               number pipeline + capacity runners
figs/                   figure generators (pubstyle.py = shared style + collision audit)
results/                committed aggregates the paper's numbers derive from
```

## License

Code (`analysis/`, `figs/*.py`) is released under the MIT License (see `LICENSE`).
The paper text and figures are (c) the authors.
