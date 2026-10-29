# P3 held-out test: does the three-quarter-depth probe beat the output on unseen models?

Frozen 2026-10-24 17:27 UTC, before any probe was fit on the models below. Owner: p3-binding-revise.

## Why

On 13 models (exploratory: found in the data, not predicted), a linear probe at layer 3L/4 decoded the
queried value more often than the model's top logit, and more often than a regression on the model's
own 67 value logits. Three reviewers asked for a confirmation that the layer, cell and rule were not
chosen after seeing the result. These three models were still running when this file was written.

## Held-out models

Qwen2.5-1.5B, Pythia-1.4B, Pythia-2.8B: final checkpoints, float32 re-measurement on the published
prompts (`cpu-angad:/mnt/data/p3_gpu_20261024/capacity_fp32/<label>/main.{json,npz}`). At freeze time
their logs held only the pooled output recall of the no-distractor condition (0.647 and 0.556 for the
first two); no hidden state had been saved and no probe fit.

## Frozen analysis

Scripts (md5 on cpu-angad at freeze): `analysis/p3_probe_capacity.py` e4fe1b1f3332611acb1ec73df6bc1e49,
`analysis/p3_paired_ci.py` 09bcb4193ed92b0571754c73f0b555e5, `analysis/capacity_curve.py`
0ca95e59759df8d8d1fb3fcd5eeb5ad1.

1. `p3_probe_capacity.py --inp <held-out dir> --save-oof <oof dir>` (value probes, four layers).
2. `p3_probe_capacity.py --inp <held-out dir> --readout-baselines --save-oof <oof dir>`.
3. `p3_probe_capacity.py --inp <held-out dir> --controls-at 2` (foil, shuffled, entity at 3L/4).
4. `p3_paired_ci.py --layer-index 2 --K 16 --conds D0,D256` (paired bootstrap over prompts, 10,000
   resamples).

Layer: `layers_saved[2]` (3L/4). Load: K = 16. Primary condition: no distractor (D0); D256 is reported
next to it but is not part of the decision.

## Predictions

- P1. probe - output >= 0.10, with the lower end of the 95% interval above 0, in at least 2 of 3 models.
- P2. probe - logit decoder has the lower end of its 95% interval above 0 in at least 2 of 3 models.
- P3. The foil probe among listed values lies in [0.03, 0.13] in all 3 models (guess 0.076).

## Decision

If P1 and P2 both hold, the paper states the probe result as confirmed on held-out models. If either
fails, the paper reports the probe result as exploratory on 13 models and states the held-out failure.
If P3 fails, the probe result is not reported as reading the binding in that model.
