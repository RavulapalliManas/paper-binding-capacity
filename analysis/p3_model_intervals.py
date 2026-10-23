"""Per-model K50 intervals for the capacity paper, from the saved per-seed cells; no model is run.

For each of the 12 continuous-protocol models, the 6 seeds of every no-distractor cell are resampled
with replacement, jointly across loads (a seed fixes the items of every cell), the seed-mean curve is
refit with capacity_curve.fit_k50, and the 2.5th and 97.5th percentiles of the refit K50 give a 95%
interval. A refit whose normalized recall stays above one half through K=64 is right-censored; when
more than 2.5% of refits are censored, the upper bound is reported as '>64'.

The same refits then enter the size fit: each draw resamples families (as the family-clustered
bootstrap in p3_revision_checks.py does), takes one seed-bootstrap refit per selected model, drops
censored refits as the point fit drops the censored model, and fits log K50 on log N by least squares.

Inputs:  results/capacity_trn1_summary.json, results/capacity_x/*_w0.json
Outputs: results/p3_model_intervals.json; with --emit DIR: DIR/numbers_intervals.tex, DIR/table_models.tex
Memory:  12 models x 15 loads x 6 seeds; the refit arrays are 12 x N_SEED_BOOT floats (< 1 MB).
Run:     python p3_model_intervals.py [--emit DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)
from capacity_curve import fit_k50  # noqa: E402
from p3_revision_checks import EMBED_M, PARAMS_TOTAL_M  # noqa: E402

N_SEED_BOOT = 2000
N_TWO_LEVEL = 10000
SEED = 0
PRETTY = {
    "olmo2-1b-x": "OLMo-2-1B", "opt-1.3b-x": "OPT-1.3B", "opt-125m-x": "OPT-125M",
    "opt-350m-x": "OPT-350M", "pythia-1.4b-x": "Pythia-1.4B", "pythia-160m-x": "Pythia-160M",
    "pythia-1b-x": "Pythia-1B", "pythia-2.8b-x": "Pythia-2.8B", "pythia-410m-x": "Pythia-410M",
    "qwen2.5-0.5b-x": "Qwen2.5-0.5B", "qwen2.5-1.5b-x": "Qwen2.5-1.5B",
    "smollm2-360m-x": "SmolLM2-360M",
}
FAMNAME = {"opt": "OPT", "pythia": "Pythia", "qwen2.5": "Qwen2.5", "smollm2": "SmolLM2", "olmo2": "OLMo-2"}


def jl(rel):
    with open(os.path.join(RES, rel)) as f:
        return json.load(f)


def cells_by_seed(path):
    """Recall per (K, seed) at D=0, as a dict K -> array over seeds sorted by seed."""
    d = json.load(open(path))
    out, n = {}, {}
    for c in d["cells"]:
        if c["D"] != 0:
            continue
        out.setdefault(c["K"], {})[c["seed"]] = c["recall"]
        n.setdefault(c["K"], {})[c["seed"]] = c["n"]
    Ks = sorted(out)
    seeds = sorted(out[Ks[0]])
    assert all(sorted(out[K]) == seeds for K in Ks), "every load needs the same seeds"
    R = np.array([[out[K][s] for s in seeds] for K in Ks])          # loads x seeds
    N = np.array([[n[K][s] for s in seeds] for K in Ks])
    return np.array(Ks, float), R, N, d["chance"]


def fit(Ks, R, N, chance):
    rec = R.mean(1)
    return fit_k50(Ks, rec, chance, n_trials=N.sum(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", default=None)
    a = ap.parse_args()
    S = jl("capacity_trn1_summary.json")["models"]
    S = S if isinstance(S, dict) else {m["label"]: m for m in S}
    rng = np.random.default_rng(SEED)
    res, refits = {}, {}
    for lab in sorted(S, key=lambda k: PARAMS_TOTAL_M[k]):
        path = os.path.join(RES, "capacity_x", f"{lab}_w0.json")
        Ks, R, N, chance = cells_by_seed(path)
        point = fit(Ks, R, N, chance)
        summ = S[lab]["by_D"]["0"]
        # the saved summary must be reproduced from the per-seed cells before any interval is built
        assert abs(point["K50"] - summ["K50"]) <= 1e-6 * max(1.0, summ["K50"]), (lab, point["K50"], summ["K50"])
        assert bool(point["censored"]) == bool(summ["censored"]), lab
        ns = R.shape[1]
        k50s, cens = [], []
        for _ in range(N_SEED_BOOT):
            pick = rng.integers(0, ns, ns)
            f = fit(Ks, R[:, pick], N[:, pick], chance)
            k50s.append(f["K50"])
            cens.append(bool(f["censored"]))
        k50s, cens = np.array(k50s), np.array(cens)
        refits[lab] = (k50s, cens)
        unc = k50s[~cens]
        if cens.all():                       # every refit right-censored: no finite bound exists
            lo = hi_raw = float("inf")
        else:
            lo = float(np.percentile(np.where(cens, np.inf, k50s), 2.5))
            hi_raw = float(np.percentile(np.where(cens, np.inf, k50s), 97.5))
        res[lab] = dict(model=PRETTY[lab], family=FAMNAME[S[lab]["family"]],
                        params_total_m=PARAMS_TOTAL_M[lab],
                        params_nonembedding_m=round(PARAMS_TOTAL_M[lab] - EMBED_M[lab], 1),
                        k1_recall=round(float(R[0].mean()), 4) if Ks[0] == 1 else None,
                        K50=round(point["K50"], 3), censored=bool(point["censored"]),
                        ci95=[round(lo, 3) if np.isfinite(lo) else None,
                              round(hi_raw, 3) if np.isfinite(hi_raw) else None],
                        censored_share=round(float(cens.mean()), 4), n_seeds=int(ns),
                        n_unc_refits=int(len(unc)))
        print(f"[k50] {PRETTY[lab]:13s} K50={point['K50']:.2f} cens={point['censored']} "
              f"ci=[{lo:.2f}, {hi_raw:.2f}] censored_share={cens.mean():.3f}", flush=True)

    # size fit with seed uncertainty and family resampling
    labs = sorted(res)
    fam = sorted({res[l]["family"] for l in labs})
    by_fam = {f: [l for l in labs if res[l]["family"] == f] for f in fam}
    alphas = []
    for _ in range(N_TWO_LEVEL):
        pick = rng.choice(fam, size=len(fam), replace=True)
        xs, ys = [], []
        for f in pick:
            for l in by_fam[f]:
                k50s, cens = refits[l]
                j = rng.integers(0, len(k50s))
                if cens[j]:
                    continue
                xs.append(math.log(PARAMS_TOTAL_M[l]))
                ys.append(math.log(k50s[j]))
        if len(set(xs)) < 3:
            continue
        alphas.append(np.polyfit(xs, ys, 1)[0])
    alphas = np.array(alphas)
    point_alpha = np.polyfit([math.log(PARAMS_TOTAL_M[l]) for l in labs if not res[l]["censored"]],
                             [math.log(res[l]["K50"]) for l in labs if not res[l]["censored"]], 1)[0]
    size = dict(alpha_point=round(float(point_alpha), 4),
                ci95_family_and_seed=[round(float(np.percentile(alphas, 2.5)), 3),
                                      round(float(np.percentile(alphas, 97.5)), 3)],
                n_draws_valid=int(len(alphas)), n_draws=N_TWO_LEVEL)
    print("[alpha] point", size["alpha_point"], "family+seed CI", size["ci95_family_and_seed"], flush=True)
    out = dict(models=res, size_fit=size, n_seed_boot=N_SEED_BOOT, seed=SEED,
               _provenance="analysis/p3_model_intervals.py from results/capacity_x/*_w0.json")
    with open(os.path.join(RES, "p3_model_intervals.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote results/p3_model_intervals.json")
    if a.emit:
        emit(out, a.emit)


def emit(out, tex_dir):
    lo, hi = out["size_fit"]["ci95_family_and_seed"]
    rel = [(r["ci95"][1] - r["ci95"][0]) / r["K50"] for r in out["models"].values()
           if not r["censored"] and r["ci95"][1] is not None]
    M = {"IvAlphaCILo": f"{lo:.2f}", "IvAlphaCIHi": f"{hi:.2f}",
         "IvSeedBoot": f"{out['n_seed_boot']:,}".replace(",", "{,}"),
         "IvMaxRelWidth": f"{max(rel):.2f}"}
    with open(os.path.join(tex_dir, "numbers_intervals.tex"), "w") as f:
        f.write("% Generated by analysis/p3_model_intervals.py from results/p3_model_intervals.json. Do not edit.\n")
        f.write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
    rows = sorted(out["models"].values(), key=lambda r: r["params_total_m"])
    t = ["% Generated by analysis/p3_model_intervals.py. Do not edit.",
         "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
         "model & family & $N$ (M) & non-embedding (M) & recall at $K=1$ & $K_{50}$ & 95\\% interval \\\\",
         "\\midrule"]
    for r in rows:
        k50 = ">64" if r["censored"] else f"{r['K50']:.1f}"
        lo, hi = r["ci95"]
        if lo is None:
            iv = "all $>$64"
        else:
            iv = f"{lo:.1f}--" + ("$>$64" if hi is None else f"{hi:.1f}")
        t.append(f"{r['model']} & {r['family']} & {r['params_total_m']:,} & {r['params_nonembedding_m']:,.0f} & "
                 f"{r['k1_recall']:.2f} & {k50.replace('>64', '$>$64')} & {iv} \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(tex_dir, "table_models.tex"), "w") as f:
        f.write("\n".join(t).replace(",", "{,}").replace("{,}{,}", "{,}") + "\n")
    # The 7 models shared by the threshold sweep and the continuous protocol, from
    # results/p3_revision_checks.json (instrument_agreement), beside this file's K50 intervals.
    ia = jl("p3_revision_checks.json")["instrument_agreement"]
    t = ["% Generated by analysis/p3_model_intervals.py from results/p3_revision_checks.json. Do not edit.",
         "\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
         "model & $\\kstar$, sweep & $K_{50}$, continuous & 95\\% interval \\\\", "\\midrule"]
    for p in sorted(ia["pairs"], key=lambda p: PARAMS_TOTAL_M[p["label"]]):
        r = out["models"][p["label"]]
        assert abs(r["K50"] - p["K50"]) < 1e-3, p["label"]
        lo, hi = r["ci95"]
        t.append(f"{r['model']} & {p['kstar']} & {p['K50']:.1f} & {lo:.1f}--{hi:.1f} \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(tex_dir, "table_instruments.tex"), "w") as f:
        f.write("\n".join(t) + "\n")
    print("wrote", os.path.join(tex_dir, "numbers_intervals.tex"), "table_models.tex, table_instruments.tex")


if __name__ == "__main__":
    main()
