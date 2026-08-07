"""Re-run the K=8 interference D-sweep so a model can (re)enter figdata.json["robust"] with real backing.

WHY THIS EXISTS. p3_capacity.tex claimed "gpt-oss-20B holds ~0.87 under interference" with no entry in
results/figdata.json["robust"] (8 models: dsc, mist, olmo7, olmoe, p14, p69, q05, q7) -- a single unlogged
run. The claim was removed (paper/README.md audit item 2; CLAIM_LEDGER.md sec. 7). This script is the
restoration path: it measures the same quantity fig7 plots (behavioural recall vs distractor length D at
fixed K) with committed code, so the curve it emits is reproducible.

It reuses probe_battery_v2.Battery2 verbatim for the task, model loading, and the rung-1 behavioural
readout (argmax over obligation-token logits), so the protocol matches the battery the papers use.

PROVENANCE CAVEAT, disclosed in the output: the committed robust curves have no committed generator (the
K-sweep launcher died with its box), so the distractor CONDITION they used is not recorded. This script
defaults to --cond random and records the condition in the output; when comparing a new curve against the
committed eight, say so in the caption if the condition may differ.

Run (H100 box; needs transformers/torch + this repo):

    cd paper/analysis
    python robust_dsweep.py --model openai/gpt-oss-20b --label gptoss20b --recipe modern --dequantize
    # optional back-fill for any committed model, e.g.:
    python robust_dsweep.py --model EleutherAI/pythia-1.4b --label p14 --recipe old

Output: paper/results/robust_dsweep/<label>.json with a figdata.json["robust"]-compatible entry under
"entry" plus full provenance. To use it in fig7, extend gen_capacity_figs.py to read
results/robust_dsweep/*.json alongside DATA["robust"] -- do NOT hand-edit figdata.json, which stays frozen
as the (disclosed) unregenerable table.
"""
from __future__ import annotations
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_battery_v2 import Battery2, Cfg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "results", "robust_dsweep")

DVALS = (0, 32, 64, 128, 256)   # the grid fig7 plots
K = 8                           # fixed load, matching the committed curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id, e.g. openai/gpt-oss-20b")
    ap.add_argument("--label", required=True, help="figdata-style label, e.g. gptoss20b")
    ap.add_argument("--recipe", required=True, choices=["old", "modern"])
    ap.add_argument("--cond", default="random", choices=["random", "prose", "code", "code_confusable"])
    ap.add_argument("--n-trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dequantize", action="store_true", help="needed for gpt-oss (MXFP4)")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    cfg = Cfg(model=args.model, out=OUTDIR, n_trials=args.n_trials, revision=args.revision,
              dequantize=args.dequantize, trust_remote_code=args.trust_remote_code, K=K)
    bat = Battery2(cfg)

    curve = []
    for D in DVALS:
        rng = random.Random(args.seed * 10_000 + D)
        _feats, Y, _foil, LG = bat.collect(rng, K, args.cond, D)
        recall = bat.rung1(LG, Y)
        curve.append([D, round(recall, 4)])
        print(f"[dsweep] {args.label} K={K} D={D:>3} cond={args.cond} recall={recall:.3f}", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    out = {
        "_provenance": (f"analysis/robust_dsweep.py --model {args.model} --label {args.label} "
                        f"--recipe {args.recipe} --cond {args.cond} --n-trials {args.n_trials} "
                        f"--seed {args.seed}" + (f" --revision {args.revision}" if args.revision else "")),
        "protocol": {"K": K, "dvals": list(DVALS), "cond": args.cond, "n_trials": args.n_trials,
                     "seed": args.seed, "readout": "rung1 argmax over obligation-token logits",
                     "task": "probe_battery_v2.Battery2.build"},
        "condition_caveat": ("the committed figdata robust curves have no recorded condition; "
                             "comparisons across that difference must be disclosed"),
        "entry": {"label": args.label, "recipe": args.recipe, "curve": curve},
    }
    path = os.path.join(OUTDIR, f"{args.label}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[dsweep] wrote {path}")


if __name__ == "__main__":
    main()
