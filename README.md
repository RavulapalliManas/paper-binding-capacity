# In-Context Binding Capacity in Language Models

**What the Query and Readout Change**

Manas Venkata Sai Ravulapalli and Samrath Chadha, Efficient Computation Inc.
Camera-ready revision for the NeurIPS 2026 Workshop on Linguistic Principles for Foundation Models (LP4FM).

This study compares retrieval load, matched query changes, and supervised readouts in small open language models. A low score under one question does not identify an intrinsic storage limit. The registered comparison tests hidden-state probes against native output and trained candidate-logit decoders. The query pilots remain exploratory, including the conditions that reduce accuracy.

[Paper PDF](https://github.com/RavulapalliManas/paper-binding-capacity/releases/download/lp4fm-2026-camera-ready/p3_capacity.pdf) · [Paper source](p3_capacity.tex) · [Source archive](https://github.com/RavulapalliManas/paper-binding-capacity/releases/download/lp4fm-2026-camera-ready/P3_capacity_arxiv.zip) · [arXiv](https://arxiv.org/abs/2609.30634)

## Reproduce and verify the camera-ready paper

Use Python 3.10 or later and a LaTeX installation with `pdflatex`. Install the plotting and statistics dependencies:

```bash
python3 -m pip install -r requirements.txt
make
```

`make` regenerates the current figures and TeX tables from committed results, checks recorded predictions, paired intervals and figure data/labels, compiles the paper three times, rejects LaTeX warnings, and creates the self-contained source archive. Outputs stay under ignored `build/`; the checkout stays clean.

- `build/p3_capacity.pdf`: compiled paper.
- `build/P3_capacity_arxiv.zip`: exact TeX, style, table and figure dependencies for the paper.
- `build/assets/`: regenerated tables, numerical macros and figures.

Run `make verify` to check the data and generated assets without LaTeX. To use a different interpreter, run `make PYTHON=/path/to/python`. The recorded results reproduce without downloading model weights, fitting a probe or running inference. The build directory is fixed at `build/` so compilation and packaging use the same verified figures.

## Code and evidence map

| Owner | Purpose |
|---|---|
| `p3_capacity.tex`, `numbers_measured.tex`, `numbers_development.tex`, `numbers_revision.tex`, `table_*.tex` | Current manuscript and generated numerical inputs. |
| `analysis/make_p3_revision_figs.py` | Current figures, tables and numerical macros; use `--measurement-root results/measurement`. The supporting development figure uses `--development-only`. |
| `analysis/p3_development.py` | Earlier checkpoint, error-group and steering summaries. `--from-summary` emits saved results; the original path reanalyzes the preserved historical inputs. |
| `analysis/p3_paired_ci.py` | Paired prompt bootstrap and compact release-data export. |
| `analysis/p3_probe_capacity.py` | Separately trained out-of-fold probes, score decoders and controls. |
| `analysis/gpu_capacity_dev.py` | Frozen-model execution and exact prompt reconstruction; despite its name, it also supports CPU execution. |
| `analysis/p3_error_anatomy.py` | Classification of native wrong answers as listed or unlisted values. |
| `analysis/verify_release.py` | Input hashes, recorded native cells, pilot/checkpoint probe rows, paired comparisons and source-archive closure. |
| `results/measurement/` | Corrected execution records, query pilots, checkpoint summaries and compact predictions. |
| `results/measurement/release_manifest.json` | File hashes and the original run record associated with each compact prediction file. |
| `results/measurement/PREREG_p3_heldout_probe.md` | Unchanged registration for the held-out-model comparison. |
| `results/p3_development.json` and its historical inputs | Earlier protocols, reported separately in the appendix. |

The compact `.npz` records contain gold values, trial identifiers, listed assignments, native predictions and available saved probe/decoder predictions. They exclude large hidden-state tensors and full vocabulary scores. The release verifier recomputes native per-cell recall and the recorded paired intervals from these predictions. Foil and shuffled-label controls are retained as recorded summaries; reproducing their fitting requires model execution and probe training.

The held-out test concerns a fixed analysis recipe with a separate classifier for each model. It does not test transfer of a trained probe between models. Prompt intervals condition on fitted predictions and do not include probe-refitting or training-run variation. Query changes alter internal states as well as behavior.

## Rerun model execution or probe fitting

Model execution is separate from rebuilding the paper and can require substantial memory and time. Install `torch`, `transformers==4.57.1`, `scikit-learn`, and `joblib` in an appropriate compute environment. Actual execution versions, padding and position settings are recorded in each model JSON.

```bash
# Check exact agreement with the original prompt generator, without a model.
python analysis/gpu_capacity_dev.py --selftest

# Example configuration used for the corrected measurements.
python analysis/gpu_capacity_dev.py \
  --model EleutherAI/pythia-410m --label pythia-410m \
  --device cpu --dtype float32 --pad dynamic --batch 16 --out generated_runs

# Fit supervised probes to the saved states in that new run.
python analysis/p3_probe_capacity.py \
  --inp generated_runs --out generated_runs/probes.json --jobs 2
```

The corrected measurements change arithmetic precision, padding and GPT-2 position handling together. They do not isolate precision as the cause. Execution uses float32 while saved sampled hidden states use float16. Checkpoint probes use a subset of the main load grid; the metadata records every evaluated cell.

The adjacency-reference calculation in `p3_error_anatomy.py` now counts neighbors as integers. The archived error summaries preserve the original NumPy boolean-addition result in `adj_exp`; that field is superseded and does not support any claim in the current paper. Listed-error counts and their uniform-pool references are unaffected. Run `python analysis/p3_error_anatomy.py --selftest` for the regression and negative-control checks.

The working probe runner includes a later `--ctl-conds` option. Its default preserves the earlier control conditions, but its bytes differ from the hash recorded in the frozen registration. The release does not claim byte-identical reproduction of that frozen source.

## Historical results

Earlier capacity sweeps, scaling fits, task-training comparisons, and their generators remain in this repository for provenance. They do not define the current paper's headline claims. The corrected cohort replaces the older execution configuration and model exclusions. Use Git history to inspect earlier manuscript versions. Do not use the earlier README's reproduction recipe or treat older result files as current capacity estimates.

The synthetic task does not establish downstream tool-use gains, joint retrieval reliability, or a causal binding mechanism. Those remain separate experiments.

## License

See [LICENSE](LICENSE).
