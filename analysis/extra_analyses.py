"""Four analyses the saved data already supports and nobody has run.

Each is self-contained, reports its own denominator, and states what it does NOT establish.

  A  READ-LAYER DEPTH.  Across models the probe's read layer was selected independently on each
     model's selection fold.  If those selections land at a consistent RELATIVE depth, the binding
     lives at a reproducible place in the stack rather than at a model-specific accident.

  B  EMERGENCE TIME vs BINDING LOAD.  The clocks campaign trained hundreds of models from scratch
     at K in {2..8}.  How does time-to-emergence scale with load?  A law here pairs with the
     capacity law: one says how many bindings a model can hold, the other how long it takes to
     learn to hold them.

  C  THE NOISE FLOOR.  The clocks cells carry 11-160 replicate runs.  Binomial spread across
     replicates is the run-to-run noise floor -- the thing the trilogy campaign, at 1-6 seeds,
     cannot see.  It says how large an effect has to be before it means anything.

  D  LEARNABILITY THRESHOLD.  1.31M-parameter models emerged 0/23; 4.98M models reach K=3-4.
     Somewhere between them the task stops being learnable at all, which is a second scale
     threshold independent of the capacity law.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
CLOCKS = os.path.join(HERE, "..", "..", "..", "emergence-clocks-2026-08", "data")
DST = os.path.join(RES, "extra_analyses.json")
OUT = {}


def hr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ------------------------------------------------------------------ A: read-layer depth
def analysis_A():
    hr("A. Where in the stack does the binding live?")
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "trilogy_trainium", "legibility*", "*.json"))):
        d = json.load(open(p))
        nl = d.get("n_layers")
        if not nl:
            continue
        for fam, v in d.get("families", {}).items():
            L = v.get("read_layer")
            if L is None:
                continue
            rows.append((d["label"], fam, L, nl, L / nl))
    if not rows:
        print("no legibility runs carry n_layers")
        return
    print(f"{'model':<24}{'family':<13}{'read layer':>11}{'n_layers':>10}{'relative depth':>16}")
    print("-" * 74)
    for r in sorted(rows, key=lambda x: x[4]):
        print(f"{r[0][:23]:<24}{r[1]:<13}{r[2]:>11}{r[3]:>10}{r[4]:>16.3f}")
    rel = np.array([r[4] for r in rows])
    dec = [r for r in rows if r[1] == "declarative"]
    reld = np.array([r[4] for r in dec])
    print()
    print(f"all families  n={len(rel):<3} median relative depth {np.median(rel):.3f}  "
          f"IQR [{np.percentile(rel,25):.3f}, {np.percentile(rel,75):.3f}]  "
          f"range [{rel.min():.3f}, {rel.max():.3f}]")
    if len(reld):
        print(f"declarative   n={len(reld):<3} median {np.median(reld):.3f}  "
              f"IQR [{np.percentile(reld,25):.3f}, {np.percentile(reld,75):.3f}]")
    print()
    print("The read layer was chosen independently per model, on that model's own selection fold.")
    print("A tight relative-depth band means the binding sits at a reproducible place in the stack.")
    print("NOT established: that it is the SAME computation in each model — only the same depth.")
    OUT["A_read_layer_depth"] = {
        "n": len(rows), "median_relative_depth": round(float(np.median(rel)), 4),
        "iqr": [round(float(np.percentile(rel, 25)), 4), round(float(np.percentile(rel, 75)), 4)],
        "range": [round(float(rel.min()), 4), round(float(rel.max()), 4)],
        "rows": [{"model": r[0], "family": r[1], "read_layer": r[2], "n_layers": r[3],
                  "relative_depth": round(r[4], 4)} for r in rows]}


# ------------------------------------------------------------------ clocks loader
def load_clocks():
    runs = []
    for f in glob.glob(os.path.join(CLOCKS, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "K" not in d or "nparam" not in d:
            continue
        if d.get("p") != 1.0 or "p_schedule" in d or "--p-after" in str(d.get("argv", "")):
            continue
        ev = d.get("evals") or []
        ch = d.get("chance_query_ctx")
        if not ev or ch is None:
            continue
        crit = ch + 0.5 * (1 - ch)
        t = None
        for e in ev:
            if e.get("query_ctx", 0) >= crit:
                t = e["step"]
                break
        runs.append({"K": d["K"], "nparam": d["nparam"], "seed": d.get("seed"),
                     "emerged": t is not None, "t_emerge": t,
                     "last_step": ev[-1]["step"], "lr": d.get("lr")})
    return runs


# ------------------------------------------------------------------ B: emergence time vs K
def analysis_B(runs):
    hr("B. How does time-to-emergence scale with binding load?")
    big = [r for r in runs if abs(r["nparam"] - 4976128) < 20000]
    byK = defaultdict(list)
    for r in big:
        if r["emerged"]:
            byK[r["K"]].append(r["t_emerge"])
    if len(byK) < 3:
        print("too few loads with emergence")
        return
    print(f"{'K':>4}{'n emerged':>11}{'median step':>14}{'IQR':>26}")
    print("-" * 56)
    Ks, med = [], []
    for k in sorted(byK):
        v = np.array(byK[k], float)
        print(f"{k:>4}{len(v):>11}{np.median(v):>14.0f}"
              f"{'['+format(np.percentile(v,25),'.0f')+', '+format(np.percentile(v,75),'.0f')+']':>26}")
        Ks.append(k); med.append(float(np.median(v)))
    Ks, med = np.array(Ks, float), np.array(med, float)
    if len(Ks) >= 3:
        A = np.column_stack([np.ones(len(Ks)), np.log(Ks)])
        b, *_ = np.linalg.lstsq(A, np.log(med), rcond=None)
        pred = A @ b
        r2 = 1 - ((np.log(med) - pred) ** 2).sum() / ((np.log(med) - np.log(med).mean()) ** 2).sum()
        print()
        print(f"power-law fit  t_emerge = {math.exp(b[0]):.1f} * K^{b[1]:.3f}   R^2 = {r2:.4f}")
        print(f"  -> each doubling of binding load costs {2**b[1]:.2f}x the training steps")
        print()
        print("Emerged runs only. Runs that never emerged are RIGHT-CENSORED and excluded, which")
        print("biases the high-K medians DOWNWARD (the slow ones are the ones that never made it).")
        print("The true exponent is therefore a LOWER BOUND.")
        OUT["B_emergence_time_vs_K"] = {
            "exponent": round(float(b[1]), 4), "coef": round(float(math.exp(b[0])), 3),
            "r2": round(float(r2), 4), "cost_per_doubling": round(float(2 ** b[1]), 3),
            "cells": {int(k): {"n_emerged": len(byK[k]), "median_step": float(np.median(byK[k]))}
                      for k in sorted(byK)},
            "caveat": "emerged runs only; censoring biases the exponent downward"}


# ------------------------------------------------------------------ C: the noise floor
def analysis_C(runs):
    hr("C. The run-to-run noise floor")
    cells = defaultdict(lambda: [0, 0])
    for r in runs:
        cells[(round(r["nparam"] / 1e6, 2), r["K"])][0] += r["emerged"]
        cells[(round(r["nparam"] / 1e6, 2), r["K"])][1] += 1
    print(f"{'params (M)':>11}{'K':>4}{'emerged':>10}{'rate':>8}{'95% CI (Wilson)':>22}")
    print("-" * 57)
    rows = []
    for (s, k), (ok, n) in sorted(cells.items()):
        if n < 8:
            continue
        p = ok / n
        z = 1.96
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        lo, hi = max(0, c - half), min(1, c + half)
        print(f"{s:>11.2f}{k:>4}{str(ok)+'/'+str(n):>10}{p:>8.3f}{'['+format(lo,'.3f')+', '+format(hi,'.3f')+']':>22}")
        rows.append({"params_M": s, "K": k, "emerged": ok, "n": n, "rate": round(p, 4),
                     "ci95": [round(lo, 4), round(hi, 4)], "ci_width": round(hi - lo, 4)})
    if rows:
        mid = [r for r in rows if 0.15 < r["rate"] < 0.85]
        if mid:
            w = np.mean([r["ci_width"] for r in mid])
            print()
            print(f"Mean 95% CI width in the informative band (rate 0.15-0.85): {w:.3f}")
            print(f"-> at these replicate counts, a difference smaller than about {w:.2f} in")
            print("   emergence rate is indistinguishable from run-to-run noise.")
            print()
            print("This is the noise floor the TRILOGY campaign cannot see at 1-6 seeds. It is why")
            print("the pythia-1.4b > pythia-2.8b inversion at 2 seeds was noise and disappeared at 6.")
        OUT["C_noise_floor"] = {"cells": rows,
                                "mean_ci_width_informative_band":
                                    round(float(np.mean([r["ci_width"] for r in mid])), 4) if mid else None}


# ------------------------------------------------------------------ D: learnability threshold
def analysis_D(runs):
    hr("D. The learnability threshold — a second scale effect")
    bysize = defaultdict(lambda: [0, 0])
    for r in runs:
        bysize[round(r["nparam"] / 1e6, 2)][0] += r["emerged"]
        bysize[round(r["nparam"] / 1e6, 2)][1] += 1
    print(f"{'params (M)':>11}{'emerged':>12}{'rate':>8}   K values present")
    print("-" * 60)
    for s in sorted(bysize):
        ok, n = bysize[s]
        Ks = sorted({r["K"] for r in runs if abs(r["nparam"] / 1e6 - s) < 0.01})
        print(f"{s:>11.2f}{str(ok)+'/'+str(n):>12}{ok/n:>8.3f}   {Ks}")
    sizes = sorted(bysize)
    if len(sizes) >= 2:
        lo, hi = sizes[0], sizes[-1]
        okl, nl = bysize[lo]; okh, nh = bysize[hi]
        print()
        print(f"{lo}M: {okl}/{nl} emerged.  {hi}M: {okh}/{nh} emerged.")
        if okl == 0 and okh > 0:
            print(f"-> the task is NOT LEARNABLE at {lo}M and IS at {hi}M, a {hi/lo:.1f}x range.")
            print("   This is a threshold in whether the capability can form at all, which is a")
            print("   different quantity from how many bindings a trained model then holds.")
            print()
            print("CONFOUND, stated: the two sizes were not necessarily run at matched K, step")
            print("budget, or learning rate. Comparing them as a clean scale contrast requires")
            print("checking those match; on this data the smaller size was run only at K=4 and 6.")
        OUT["D_learnability_threshold"] = {
            "by_size": {str(s): {"emerged": bysize[s][0], "n": bysize[s][1]} for s in sizes}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    analysis_A()
    runs = load_clocks()
    print(f"\n[clocks] {len(runs)} clean p=1.0 runs loaded")
    if runs:
        analysis_B(runs)
        analysis_C(runs)
        analysis_D(runs)
    if a.write:
        OUT["_provenance"] = ("analysis/extra_analyses.py over results/trilogy_trainium/ and "
                              "research/emergence-clocks-2026-08/data. Clocks runs filtered to "
                              "p=1.0 with no supply manipulation; emergence uses each run's own "
                              "chance + 0.5*(1-chance).")
        json.dump(OUT, open(DST, "w"), indent=2)
        print(f"\nwrote {DST}")


if __name__ == "__main__":
    main()
