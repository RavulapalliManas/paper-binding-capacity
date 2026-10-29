# In-Context Binding Capacity in Language Models: What the Query and Readout Change

Manas Venkata Sai Ravulapalli and Samrath Chadha

Efficient Computation Inc. · LP4FM, NeurIPS 2026

Code and data for the paper. We study how language models retrieve name–value associations as more pairs appear in context. Changing the question can improve recall without changing the model's weights. We also compare predictions from hidden states with predictions from the model's output scores.

[Paper](https://github.com/RavulapalliManas/paper-binding-capacity/releases/latest/download/p3_capacity.pdf) · [arXiv](https://arxiv.org/abs/2609.30634) · [LaTeX source](p3_capacity.tex)

## Build the paper

Requires Python 3.10 or later and `pdflatex`.

```bash
python3 -m pip install -r requirements.txt
make
```

This produces `build/p3_capacity.pdf`, the figures and tables in `build/assets/`, and a LaTeX source archive. It uses the saved results; it does not run the models or train probes. Use `make verify` to check the results without compiling the paper.

## Run experiments

For model evaluation and probe training, also install `torch`, `transformers==4.57.1`, `scikit-learn`, and `joblib`.

```bash
python analysis/gpu_capacity_dev.py \
  --model EleutherAI/pythia-410m --label pythia-410m \
  --device cpu --dtype float32 --pad dynamic --batch 16 --out generated_runs

python analysis/p3_probe_capacity.py \
  --inp generated_runs --out generated_runs/probes.json --jobs 2
```

## Code and data

- `analysis/gpu_capacity_dev.py`: prompts and model evaluation.
- `analysis/p3_probe_capacity.py`: probes, output-score decoders and controls.
- `analysis/p3_paired_ci.py`: paired confidence intervals.
- `analysis/make_p3_revision_figs.py`: figures and tables.
- `analysis/p3_development.py`: analysis of earlier checkpoint and steering experiments.
- `results/measurement/`: results for the current experiments. Per-trial assignments and predictions are in `predictions/`. Hidden activations are not included.

The [preregistration](results/measurement/PREREG_p3_heldout_probe.md) describes the held-out-model test. Each model has its own trained probe. The released probe script adds a `--ctl-conds` option after preregistration; it retains the original default conditions.

Earlier experiments remain in `results/` and use different protocols. The current measurements use float32 execution, dynamic padding and corrected GPT-2 position IDs. These changes were made together, so their individual effects are unresolved. Sampled hidden states are stored in float16.

The archived `adj_exp` error baseline is incorrect and unused in the paper. The script now counts neighboring entities correctly; listed-error counts are unaffected.

## License

See [LICENSE](LICENSE).
