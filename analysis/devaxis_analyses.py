"""Analyses over the devaxis run corpus, including an independent test of the learning-time law.

THE HEADLINE TEST.  The clocks campaign (8xH100, its own codebase) gave
    t_emerge = 765.5 * K^2.578,  R^2 = 0.9475
on 4.98M-parameter transformers trained from scratch.  devaxis trains the same KIND of model on
DIFFERENT HARDWARE (Trainium), through a DIFFERENT codebase, with its own data generator.  If the
exponent replicates there, it is a property of the learning problem rather than of one
implementation.  That is the strongest form of evidence available without new compute, and it is
a genuine out-of-sample test: the clocks fit was made before this corpus was examined.

Everything uses each run's OWN emergence criterion, crit = chance + 0.5*(1 - chance).  Hard-coding
0.75 (the K=2 bar) onto K=4 runs was a real bug in an earlier analysis of this corpus.

ANALYSES
  1  phase inventory -- what the corpus actually contains, counted from files
  2  emergence time vs binding load K, and the cross-codebase comparison
  3  supply dose-response: recovery as a function of the supply rate held through a gap
  4  seed heterogeneity: are some seeds systematically harder, or is that a story about noise
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
RUNS = os.path.join(HERE, "..", "..", "..", "devaxis-trainium", "runs_json")
DST = os.path.join(HERE, "..", "results", "devaxis_analyses.json")
CLOCKS_LAW = {"coef": 765.5, "exponent": 2.578, "r2": 0.9475,
              "source": "analysis/extra_analyses.py over research/emergence-clocks-2026-08"}
OUT = {}


def hr(t):
    print(); print("=" * 78); print(t); print("=" * 78)


def load():
    runs = []
    for f in glob.glob(os.path.join(RUNS, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "evals" not in d:
            continue
        man = d.get("manifest") if isinstance(d.get("manifest"), dict) else {}
        task = man.get("task") if isinstance(man.get("task"), dict) else {}
        ch = d.get("chance_query_ctx")
        ev = d.get("evals") or []
        if ch is None or not ev:
            continue
        crit = ch + 0.5 * (1 - ch)
        t_em = None
        for e in ev:
            if e.get("query_ctx", 0) >= crit:
                t_em = e["step"]
                break
        fs = d.get("fork_step")
        q_fork = None
        if fs is not None:
            near = [e for e in ev if e.get("step") is not None and e["step"] <= fs]
            if near:
                q_fork = near[-1].get("query_ctx")
        runs.append({
            "phase": os.path.basename(os.path.dirname(f)),
            "run_id": d.get("run_id"), "parent": d.get("parent_run_id"),
            "fork_step": d.get("fork_step"),
            "K": task.get("K"), "D": task.get("D"),
            "p_schedule": str(man.get("p_schedule")),
            "seed": man.get("init_seed"),
            "nparam": d.get("nparam"), "chance": ch, "crit": crit,
            "emerged": t_em is not None, "t_emerge": t_em,
            "last_step": ev[-1]["step"], "final_qctx": ev[-1].get("query_ctx"),
            "qctx_at_fork": q_fork,
            "n_evals": len(ev),
        })
    return runs


def a1_inventory(runs):
    hr("1. Phase inventory — counted from files, never from a queue length")
    byp = defaultdict(lambda: [0, 0])
    for r in runs:
        byp[r["phase"]][0] += 1
        byp[r["phase"]][1] += r["emerged"]
    print(f"{'phase':<16}{'runs':>7}{'emerged':>9}   K values")
    print("-" * 54)
    for p in sorted(byp, key=lambda x: -byp[x][0]):
        n, e = byp[p]
        Ks = sorted({r["K"] for r in runs if r["phase"] == p and r["K"]})
        print(f"{p:<16}{n:>7}{e:>9}   {Ks}")
    print(f"\ntotal {len(runs)} runs across {len(byp)} phases")
    OUT["inventory"] = {p: {"runs": v[0], "emerged": v[1]} for p, v in byp.items()}
    OUT["n_runs_total"] = len(runs)


def a2_emergence_law(runs):
    hr("2. Emergence time vs binding load — and does the clocks law replicate?")
    # baseline condition only: uninterrupted full supply, no fork
    base = [r for r in runs
            if r["fork_step"] is None and r["p_schedule"] in ("1.0", "1", "0:1.0")
            and r["K"] and r["emerged"]]
    byK = defaultdict(list)
    for r in base:
        byK[r["K"]].append(r["t_emerge"])
    print(f"uninterrupted p=1.0 runs that emerged: {len(base)}")
    print()
    if len(byK) < 3:
        print(f"only {len(byK)} loads with emergence — cannot fit a law")
        OUT["emergence_law"] = {"note": "too few loads", "cells":
                                {int(k): len(v) for k, v in byK.items()}}
        return
    print(f"{'K':>4}{'n':>7}{'median step':>14}{'IQR':>26}")
    print("-" * 52)
    Ks, med = [], []
    for k in sorted(byK):
        v = np.array(byK[k], float)
        print(f"{k:>4}{len(v):>7}{np.median(v):>14,.0f}"
              f"{'['+format(np.percentile(v,25),',.0f')+', '+format(np.percentile(v,75),',.0f')+']':>26}")
        Ks.append(k); med.append(float(np.median(v)))
    Ks, med = np.array(Ks, float), np.array(med, float)
    A = np.column_stack([np.ones(len(Ks)), np.log(Ks)])
    b, *_ = np.linalg.lstsq(A, np.log(med), rcond=None)
    pred = A @ b
    r2 = 1 - ((np.log(med) - pred) ** 2).sum() / ((np.log(med) - np.log(med).mean()) ** 2).sum()
    print()
    print(f"  devaxis  t_emerge = {math.exp(b[0]):.1f} * K^{b[1]:.3f}   R^2 = {r2:.4f}   (n={len(Ks)} loads)")
    print(f"  clocks   t_emerge = {CLOCKS_LAW['coef']} * K^{CLOCKS_LAW['exponent']}   "
          f"R^2 = {CLOCKS_LAW['r2']}")
    print()
    d_exp = float(b[1]) - CLOCKS_LAW["exponent"]
    print(f"  exponent difference: {d_exp:+.3f}")
    print()
    print("  Two independent codebases, different hardware (Trainium vs 8xH100), different data")
    print("  generators. The clocks fit was made BEFORE this corpus was examined, so this is an")
    print("  out-of-sample test of the exponent, not a joint fit.")
    print()
    print("  CENSORING, same as before: runs that never emerged are excluded, which drags the")
    print("  high-K medians down. Both exponents are LOWER BOUNDS and the comparison inherits that.")
    OUT["emergence_law"] = {
        "devaxis": {"coef": round(float(math.exp(b[0])), 2), "exponent": round(float(b[1]), 4),
                    "r2": round(float(r2), 4), "n_loads": len(Ks),
                    "cells": {int(k): {"n": len(byK[k]), "median": float(np.median(byK[k]))}
                              for k in sorted(byK)}},
        "clocks": CLOCKS_LAW,
        "exponent_difference": round(d_exp, 4),
        "caveat": "emerged runs only in both; exponents are lower bounds"}


def a3_dose_response(runs):
    hr("3. Supply dose-response — how much supply is needed to keep the capability")
    def sched(r):
        ps = r["p_schedule"]
        if not ps or "," not in ps:
            return None
        try:
            segs = sorted((int(k.split(":")[0]), float(k.split(":")[1])) for k in ps.split(","))
        except Exception:
            return None
        fs = r["fork_step"]
        if fs is None:
            return None
        after = [(s, v) for s, v in segs if s >= fs]
        if not after:
            return None
        # THE SUPPLY LEVEL IS THE MINIMUM AFTER THE FORK, NOT THE FIRST SEGMENT.
        # Taking the first segment labels a schedule like "0:1.0,10000:0.0" (baseline, then a
        # total gap) as p=1.0, which produced an impossible cell: p=1.000 -> 0/36 retention.
        # The manipulated quantity is the DEPTH of the gap, i.e. the minimum supply the run is
        # subjected to after it forks.
        return min(v for _, v in after)

    cells = defaultdict(lambda: [0, 0])
    for r in runs:
        p = sched(r)
        if p is None or r["K"] is None:
            continue
        # RETENTION IS ONLY DEFINED FOR A RUN THAT HAD THE CAPABILITY WHEN THE MANIPULATION
        # BEGAN.  Without this, a run that never emerged counts as "failed to hold", which
        # produced the impossible cell p=1.000 -> 0/28: full supply appearing to destroy the
        # capability, when in fact those runs never had it.
        if r["qctx_at_fork"] is None or r["qctx_at_fork"] < r["crit"]:
            continue
        held = r["final_qctx"] is not None and r["final_qctx"] >= r["crit"]
        cells[(r["K"], p)][0] += held
        cells[(r["K"], p)][1] += 1
    if not cells:
        print("no scheduled-supply runs found"); return
    print(f"{'K':>4}{'supply p':>10}{'held':>10}{'rate':>8}")
    print("-" * 34)
    rows = []
    for (k, p), (ok, n) in sorted(cells.items()):
        if n < 4:
            continue
        print(f"{k:>4}{p:>10.3f}{str(ok)+'/'+str(n):>10}{ok/n:>8.3f}")
        rows.append({"K": k, "p": p, "held": ok, "n": n, "rate": round(ok / n, 4)})
    print()
    print("`held` = still above the run's own criterion at its last eval. Threshold-shaped rather")
    print("than graded dose-response is the pattern the campaign reported; this recomputes it from")
    print("the raw runs.")
    OUT["dose_response"] = rows


def a4_seed_heterogeneity(runs):
    hr("4. Seed heterogeneity — are some seeds systematically harder?")
    base = [r for r in runs if r["seed"] is not None and r["K"]]
    byseed = defaultdict(lambda: [0, 0])
    for r in base:
        byseed[r["seed"]][0] += r["emerged"]
        byseed[r["seed"]][1] += 1
    usable = {s: v for s, v in byseed.items() if v[1] >= 6}
    if len(usable) < 4:
        print(f"only {len(usable)} seeds with >=6 runs — cannot assess"); return
    rates = np.array([v[0] / v[1] for v in usable.values()])
    ns = np.array([v[1] for v in usable.values()])
    print(f"{len(usable)} seeds with >=6 runs each; overall emergence rate "
          f"{sum(v[0] for v in usable.values())/sum(ns):.3f}")
    print(f"per-seed rate: min {rates.min():.3f}  median {np.median(rates):.3f}  max {rates.max():.3f}")
    # is the spread larger than binomial noise?
    pbar = sum(v[0] for v in usable.values()) / sum(ns)
    exp_var = pbar * (1 - pbar) * np.mean(1.0 / ns)
    obs_var = float(rates.var(ddof=1))
    print()
    print(f"observed variance across seeds : {obs_var:.5f}")
    print(f"variance expected from binomial: {exp_var:.5f}")
    print(f"ratio (overdispersion)         : {obs_var/exp_var:.2f}x")
    print()
    if obs_var / exp_var > 1.5:
        print("Seeds differ MORE than binomial noise allows -> initialisation matters, and any")
        print("per-seed comparison must be paired rather than pooled.")
    else:
        print("Seed-to-seed spread is consistent with binomial noise. An apparently 'hard' seed")
        print("set is then a selection effect, not a property of those initialisations -- which is")
        print("the honest reading of the s24-s31 puzzle unless it survives this test.")
    OUT["seed_heterogeneity"] = {
        "n_seeds": len(usable), "overall_rate": round(float(pbar), 4),
        "rate_min": round(float(rates.min()), 4), "rate_max": round(float(rates.max()), 4),
        "observed_variance": round(obs_var, 6), "binomial_variance": round(float(exp_var), 6),
        "overdispersion_ratio": round(float(obs_var / exp_var), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    runs = load()
    if not runs:
        print(f"no runs found under {RUNS}"); return
    a1_inventory(runs)
    a2_emergence_law(runs)
    a3_dose_response(runs)
    a4_seed_heterogeneity(runs)
    if a.write:
        OUT["_provenance"] = ("analysis/devaxis_analyses.py over research/devaxis-trainium/"
                              "runs_json (synced from s3://.../devaxis/runs/). Emergence uses each "
                              "run's own chance + 0.5*(1-chance).")
        json.dump(OUT, open(DST, "w"), indent=2)
        print(f"\nwrote {DST}")


if __name__ == "__main__":
    main()
