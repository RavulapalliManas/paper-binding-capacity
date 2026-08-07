"""Re-run the capacity K-sweep with committed code, so the capacity table stops being unregenerable.

WHY THIS EXISTS. The 30-model capacity table (results/figdata.json["geom"], and capacity_stats.json
derived from it) has NO committed generator -- its sweep launcher died with its box. The unified paper
ships that table under a written verify.py exception and discloses it in the Reproducibility Statement,
which names "re-running the sweep with committed code" as the open item. This script is that item.

PROTOCOL (recorded here and in every output file, because the original's was not):
  - Task/model loading/behavioural readout reuse probe_battery_v2.Battery2 verbatim: K entity:obligation
    bindings, distractor block, single-entity recall query; recall = rung-1 (argmax over the obligation
    single-token logits).
  - Curve: recall vs K over K_GRID, at a FIXED distractor condition (--cond, default random) and length
    (--dlen, default 0: pure capacity, no interference -- robustness-vs-D is the separate axis measured
    by robust_dsweep.py).
  - k* = smallest K in the grid at which mean recall falls below the midpoint of the model's own K=1
    ceiling and the obligation-pool chance (the definition the paper states). Right-censored at the top
    of the grid; a model whose K=1 recall is at chance gets kstar=None (unusable, the Pythia-70M case).
  - Seeds averaged per K (--seeds, default 2); every cell logged.

COMPARABILITY CAVEAT, disclosed in the output: the frozen table's distractor condition, trial count, and
seed policy are unrecorded, so this re-run is a NEW measurement of the same quantity, not a byte-level
replication. Agreement is evidence the law is real; disagreement is a finding about the frozen table.

Run (H100 box; needs transformers/torch + this repo):
    cd paper/analysis
    python ksweep.py --model Qwen/Qwen2.5-7B --label qwen2.5-7b --recipe modern
    python ksweep.py --model openai/gpt-oss-20b --label gptoss20b --recipe modern --dequantize

Output: paper/results/ksweep_rerun/<label>.json with the recall-vs-K curve, kstar, and full provenance.
Do NOT hand-edit figdata.json, which stays frozen as the (disclosed) unregenerable table; comparison and
any figure use read results/ksweep_rerun/*.json alongside it.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_battery_v2 import Battery2, Cfg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "results", "ksweep_rerun")

K_GRID = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24)   # matches the frozen table's observed k* values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="short label, becomes <label>.json")
    ap.add_argument("--recipe", required=True, choices=["old", "modern"])
    ap.add_argument("--cond", default="random", choices=["random", "prose", "code", "code_confusable"])
    ap.add_argument("--dlen", type=int, default=0, help="distractor length (0 = pure capacity)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n-trials", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--dequantize", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    out_path = os.path.join(OUTDIR, f"{a.label}.json")
    if os.path.exists(out_path) and json.load(open(out_path)).get("done"):
        print(f"[ksweep] {a.label} already done -> {out_path}"); return

    bat = Battery2(Cfg(model=a.model, out=OUTDIR, seeds=a.seeds, n_trials=a.n_trials, batch=a.batch,
                       dtype=a.dtype, dequantize=a.dequantize, trust_remote_code=a.trust_remote_code))
    max_k = min(len(bat.ent), len(bat.obl))
    grid = [k for k in K_GRID if k <= max_k]
    curve = []
    for K in grid:
        accs = []
        for seed in range(a.seeds):
            rng = random.Random((7000 + seed) * 100003 + K)
            _F, Y, _FO, LG = bat.collect(rng, K, a.cond, a.dlen)
            accs.append(bat.rung1(LG, Y))
        m = sum(accs) / len(accs)
        curve.append({"K": K, "recall_mean": m, "recall_per_seed": accs})
        print(f"[ksweep] {a.label} K={K:3d} recall={m:.3f} ({', '.join(f'{x:.3f}' for x in accs)})", flush=True)

    ceil_k1 = curve[0]["recall_mean"]
    midpoint = (ceil_k1 + bat.chance) / 2.0
    kstar, censored, usable = None, False, ceil_k1 > 2 * bat.chance
    if usable:
        below = [c["K"] for c in curve if c["recall_mean"] < midpoint]
        if below:
            kstar = below[0]
        else:
            kstar, censored = grid[-1], True  # never crossed: right-censored at the grid top

    json.dump({
        "label": a.label, "model": a.model, "recipe": a.recipe, "done": True,
        "kstar": kstar, "right_censored": censored, "usable": usable,
        "ceiling_k1": ceil_k1, "chance": bat.chance, "midpoint": midpoint,
        "curve": curve, "k_grid": grid, "max_k_pool_limited": max_k,
        "protocol": {"cond": a.cond, "dlen": a.dlen, "seeds": a.seeds, "n_trials": a.n_trials,
                     "readout": "rung1 argmax over obligation single-token logits",
                     "kstar_def": "smallest grid K with mean recall < (K=1 ceiling + chance)/2",
                     "comparability": "NEW measurement; frozen table's condition/trials/seeds unrecorded"},
        "dequantize": a.dequantize, "dtype": a.dtype,
        "n_entities_single_tok": len(bat.ent), "n_obligations_single_tok": len(bat.obl),
        "ts": time.time(),
    }, open(out_path, "w"), indent=2)
    print(f"[ksweep] DONE {a.label}: kstar={kstar} censored={censored} usable={usable} -> {out_path}")


if __name__ == "__main__":
    main()
