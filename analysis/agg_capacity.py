"""Turn the per-worker capacity cells into paper-ready capacity statistics.

INPUT   results/capacity_trn1/<label>_w<i>.json, written by neuron_capacity.py (one file per
        worker; workers never share a path).
OUTPUT  results/capacity_trn1_summary.json -- per model: the full recall-vs-K curve, K50 with a
        censoring flag, threshold-free capacity AUC, and the recipe/family labels; plus the
        recipe comparison computed the way capacity_stats.py insists on.

THREE RULES INHERITED FROM capacity_stats.py, because they were established by finding errors:

  1. EXCLUDE MODELS THAT CANNOT HOLD ONE BINDING.  r(K) divides by (a(1) - chance); if the K=1
     ceiling is at chance, that denominator is noise and every derived number is noise
     amplification.  capacity_curve.normalised raises rather than returning a value.

  2. THIRTY MODELS ARE TEN FAMILIES.  Every comparison is clustered on family, plus a
     leave-one-family-out sweep, because one prolific family can carry an apparent effect.

  3. CENSORED MODELS ARE BOUNDS, NOT VALUES.  A model still above r=0.5 at the top of the grid
     has K50 > Kmax.  Recording it as Kmax drags the high tail down, and the censored models are
     concentrated in one recipe group -- so clipping would bias the headline comparison in a
     specific direction.  Censored rows are reported and are excluded from the median test with
     their count stated, never silently converted.

THE DENOMINATOR IS ALWAYS REPORTED.  Some architectures may fail to compile on Neuron; the
summary states how many models were attempted, how many produced curves, and which failed.  A
statistic over an unstated subset is not a measurement.

Run:  python agg_capacity.py [--src results/capacity_trn1] [--write]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from capacity_curve import fit_k50

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "..", "results", "capacity_trn1")
DST = os.path.join(HERE, "..", "results", "capacity_trn1_summary.json")

RECIPE = {
    "pythia": "old", "opt": "old", "gpt2": "old",
    "qwen2.5": "modern", "smollm2": "modern", "olmo2": "modern", "phi": "modern",
}

# Total parameters, in millions, from each model card.  Needed because the recipe groups are NOT
# size-matched: the old-recipe set reaches 2.8B while the modern set is skewed small.  A raw median
# difference between those groups is a scale comparison wearing a recipe label, so the headline
# test MUST condition on scale.
PARAMS_M = {
    "pythia-70m": 70, "pythia-160m": 160, "pythia-410m": 410, "pythia-1b": 1000,
    "pythia-1.4b": 1400, "pythia-2.8b": 2800,
    "opt-125m": 125, "opt-350m": 350, "opt-1.3b": 1300, "opt-2.7b": 2700,
    "gpt2": 124, "gpt2-medium": 355, "gpt2-large": 774,
    "qwen2.5-0.5b": 500, "qwen2.5-1.5b": 1500, "qwen2.5-3b": 3000,
    "smollm2-135m": 135, "smollm2-360m": 360, "smollm2-1.7b": 1700,
    "olmo2-1b": 1000, "phi-1.5": 1400,
}


def base_label(lab):
    """Strip campaign suffixes so parameter lookup works across runs.

    The strict re-run labels models `pythia-160m-x`; without this the PARAMS_M lookup misses every
    row and the scale-controlled test silently reports 'too few models' instead of an answer.
    """
    return lab[:-2] if lab.endswith("-x") else lab


def scale_controlled(rows, n_boot=20000, seed=0):
    """log K50 ~ recipe + log(params), bootstrapped over FAMILIES.

    This is the test the claim actually needs.  The raw median difference confounds recipe with
    scale whenever the two groups are not size-matched, which they are not here.  Censored rows are
    EXCLUDED and counted, never clipped to the grid maximum -- clipping drags the top of the
    distribution down, and the censored models are not evenly spread across recipes.
    """
    import numpy as _np
    use = [r for r in rows if not r["censored"] and base_label(r["label"]) in PARAMS_M]
    n_cens = sum(1 for r in rows if r["censored"])
    if len({r["recipe"] for r in use}) < 2 or len(use) < 5:
        return {"note": "too few uncensored models with known parameter counts",
                "n_used": len(use), "n_censored_excluded": n_cens}

    def fit(sample):
        y = _np.log([r["K50"] for r in sample])
        X = _np.column_stack([_np.ones(len(sample)),
                              [1.0 if r["recipe"] == "modern" else 0.0 for r in sample],
                              _np.log([PARAMS_M[base_label(r["label"])] for r in sample])])
        try:
            beta, *_ = _np.linalg.lstsq(X, y, rcond=None)
            return float(beta[1]), float(beta[2])
        except Exception:
            return None

    obs = fit(use)
    fams = sorted({r["family"] for r in use})
    byfam = {f: [r for r in use if r["family"] == f] for f in fams}
    rng = _np.random.default_rng(seed)
    b_rec, b_scale = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(fams), len(fams), replace=True)
        sample = [r for i in pick for r in byfam[fams[i]]]
        if len({r["recipe"] for r in sample}) < 2 or len(sample) < 4:
            continue
        f = fit(sample)
        if f:
            b_rec.append(f[0]); b_scale.append(f[1])
    b_rec, b_scale = _np.array(b_rec), _np.array(b_scale)
    return {
        "recipe_coef_logK50": round(obs[0], 4),
        "recipe_ci95": [round(float(_np.percentile(b_rec, 2.5)), 3),
                        round(float(_np.percentile(b_rec, 97.5)), 3)],
        "recipe_frac_boot_le_zero": round(float((b_rec <= 0).mean()), 4),
        "log_params_coef": round(obs[1], 4),
        "log_params_ci95": [round(float(_np.percentile(b_scale, 2.5)), 3),
                            round(float(_np.percentile(b_scale, 97.5)), 3)],
        "n_used": len(use), "n_censored_excluded": n_cens, "n_families": len(fams),
        "n_boot_valid": int(len(b_rec)),
    }


def family(label):
    label = base_label(label)
    for f in ("pythia", "opt", "gpt2", "qwen2.5", "smollm2", "olmo2", "phi"):
        if label.startswith(f):
            return f
    return label.split("-")[0]


def collect(src):
    """Merge every worker's cells per label.  Duplicate (K, D, seed) across workers would mean the
    work split was wrong, so it is an error rather than something to average away."""
    per = defaultdict(lambda: {"cells": {}, "meta": None})
    for p in sorted(glob.glob(os.path.join(src, "*_w*.json"))):
        d = json.load(open(p))
        lab = d["label"]
        per[lab]["meta"] = {k: v for k, v in d.items() if k != "cells"}
        for c in d["cells"]:
            key = (c["K"], c["D"], c["seed"])
            if key in per[lab]["cells"]:
                raise ValueError(f"{lab}: duplicate cell {key} — the worker split overlapped")
            per[lab]["cells"][key] = c
    return per


def missing_cells(cells, expect_seeds):
    """Which (K, D, seed) cells never came back.

    A worker that dies takes its cells with it -- gpt2-medium returned 31 files instead of 32.
    Silently averaging the survivors would report a seed mean over n=1 while the metadata still
    says seeds=2.  Every gap is listed, and cells with fewer seeds than expected are flagged.
    """
    Ks = sorted({k for (k, _d, _s) in cells})
    Ds = sorted({d for (_k, d, _s) in cells})
    miss = [(K, D, s) for K in Ks for D in Ds for s in range(expect_seeds)
            if (K, D, s) not in cells]
    return miss


def curves(cells):
    """Average seeds within each (K, D).  Returns {D: (Ks, recalls, ns)}; ns is the seed count
    actually present, never the count that was requested."""
    byD = defaultdict(lambda: defaultdict(list))
    for (K, D, _s), c in cells.items():
        byD[D][K].append(c["recall"])
    out = {}
    for D, kk in byD.items():
        Ks = sorted(kk)
        out[D] = (Ks, [float(np.mean(kk[k])) for k in Ks], [len(kk[k]) for k in Ks])
    return out


def clustered_median_diff(rows, n_boot=20000, seed=0):
    """Difference in median K50 between recipes, bootstrapped over FAMILIES not models."""
    rng = np.random.default_rng(seed)
    fams = sorted({r["family"] for r in rows})
    byfam = {f: [r for r in rows if r["family"] == f] for f in fams}
    obs = None
    def stat(sample):
        old = [r["K50"] for r in sample if r["recipe"] == "old"]
        mod = [r["K50"] for r in sample if r["recipe"] == "modern"]
        if not old or not mod:
            return None
        return float(np.median(mod) - np.median(old))
    obs = stat(rows)
    if obs is None:
        return {"observed": None, "note": "only one recipe class present"}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(fams), len(fams), replace=True)
        sample = [r for i in pick for r in byfam[fams[i]]]
        s = stat(sample)
        if s is not None:
            boots.append(s)
    boots = np.array(boots)
    return {"observed_median_diff": round(obs, 3),
            "ci95": [round(float(np.percentile(boots, 2.5)), 3),
                     round(float(np.percentile(boots, 97.5)), 3)],
            "frac_boot_le_zero": round(float((boots <= 0).mean()), 4),
            "n_families": len(fams), "n_boot_valid": int(len(boots))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--primary-d", type=int, default=0, help="D used for the headline K50")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    per = collect(a.src)
    models, excluded = [], []
    for lab, blob in sorted(per.items()):
        meta = blob["meta"]
        cur = curves(blob["cells"])
        miss = missing_cells(blob["cells"], 2)
        entry = {"label": lab, "family": family(lab), "recipe": RECIPE.get(family(lab), "unknown"),
                 "model": meta.get("model"), "chance": meta.get("chance"),
                 "n_obligations": meta.get("n_obligations"), "dtype": meta.get("dtype"),
                 "n_trials_per_cell": meta.get("n_trials"),
                 "n_cells_present": len(blob["cells"]),
                 "n_cells_missing": len(miss),
                 "missing_cells": [list(m) for m in miss[:20]], "by_D": {}}
        if miss:
            print(f"  WARNING {lab}: {len(miss)} missing cell(s), e.g. {miss[:3]} "
                  f"— affected K values are averaged over fewer seeds")
        ok = False
        for D, (Ks, rec, ns) in sorted(cur.items()):
            try:
                fit = fit_k50(Ks, rec, meta["chance"], n_trials=[meta["n_trials"] * n for n in ns])
                entry["by_D"][str(D)] = fit
                ok = True
            except ValueError as e:                      # K=1 ceiling at chance -> not a measurement
                entry["by_D"][str(D)] = {"excluded": str(e), "K_grid": Ks, "recall": rec}
        (models if ok else excluded).append(entry)

    prim = [m for m in models if str(a.primary_d) in m["by_D"]
            and "K50" in m["by_D"][str(a.primary_d)]]
    rows = [{"label": m["label"], "family": m["family"], "recipe": m["recipe"],
             "K50": m["by_D"][str(a.primary_d)]["K50"],
             "censored": m["by_D"][str(a.primary_d)]["censored"],
             "auc": m["by_D"][str(a.primary_d)]["capacity_auc"]} for m in prim]

    uncens = [r for r in rows if not r["censored"]]
    summary = {
        "_provenance": ("Measured on trn1.32xlarge by analysis/neuron_capacity.py, aggregated by "
                        "analysis/agg_capacity.py, K50/AUC by analysis/capacity_curve.py. "
                        "bfloat16 behavioural readout (argmax over obligation logits)."),
        "n_models_attempted": len(per), "n_models_with_curves": len(models),
        "n_models_excluded_at_chance": len(excluded),
        "excluded_labels": [m["label"] for m in excluded],
        "primary_D": a.primary_d,
        "n_censored": sum(1 for r in rows if r["censored"]),
        "K50_range_uncensored": ([round(min(r["K50"] for r in uncens), 2),
                                  round(max(r["K50"] for r in uncens), 2)] if uncens else None),
        "recipe_test_clustered_on_family": clustered_median_diff(rows),
        "recipe_test_controlling_for_scale": scale_controlled(rows),
        "leave_one_family_out": {},
        "models": models,
    }
    for f in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] != f]
        if len({r["recipe"] for r in sub}) == 2:
            summary["leave_one_family_out"][f"drop_{f}"] = clustered_median_diff(sub)

    print(f"attempted {summary['n_models_attempted']}  with-curves {summary['n_models_with_curves']}"
          f"  excluded-at-chance {summary['n_models_excluded_at_chance']}  "
          f"censored {summary['n_censored']}")
    for r in sorted(rows, key=lambda x: x["K50"]):
        flag = " (censored, K50 is a LOWER BOUND)" if r["censored"] else ""
        print(f"  {r['label']:<16} {r['recipe']:<7} K50={r['K50']:>7.2f}  AUC={r['auc']:.3f}{flag}")
    print("\nrecipe (family-clustered):", json.dumps(summary["recipe_test_clustered_on_family"]))

    if a.write:
        json.dump(summary, open(DST, "w"), indent=2)
        print("wrote", DST)


if __name__ == "__main__":
    main()
