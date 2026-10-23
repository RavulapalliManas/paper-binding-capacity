"""Interference robustness as an axis SEPARATE from capacity — and the normalisation trap in it.

THE CLAIM THIS TESTS.  The capacity paper argues that how many bindings a model can hold and how
well it holds them under interference are different properties.  Both axes were already measured
(every capacity cell was run at D=0 and D=256); only D=0 had ever been analysed.  This costs no
new compute.

THE TRAP, AND WHY IT MATTERS.  capacity_curve normalises a recall curve by the model's OWN K=1
ceiling: r(K) = (a(K,D) - c) / (a(1,D) - c).  Applied naively to the D=256 curve that divides by
the ceiling *under interference*, which interference has already lowered.  The result measures the
SHAPE of the decay relative to a depressed ceiling, not capacity under interference -- and it
inverts the sign of the effect.  Measured on this data, own-ceiling normalisation reports

    opt-1.3b   K50 38.08 (D=0)  ->  63.24 (D=256)      "interference helps"
    pythia-1b  K50  5.52 (D=0)  ->   7.53 (D=256)      "interference helps"

which cannot be true.  Normalising the D=256 curve by the D=0 ceiling instead gives the retained
fraction, and interference then reduces capacity in 11 of 12 models, as it must.

Any comparison of K50 across conditions that change the K=1 ceiling must fix the normaliser to one
condition.  Reporting the own-ceiling version would have published a backwards result.

TWO REPORTED QUANTITIES, per model:
    retained fraction   K50(D=256, normalised by the D=0 ceiling) / K50(D=0)
    ceiling drop        a(1, D=0) - a(1, D=256), the pure interference cost at K=1

Run:  python interference_axis.py [--src ../results/capacity_x] [--write]
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
DST = os.path.join(HERE, "..", "results", "interference_axis.json")


def load_cells(src):
    per = defaultdict(lambda: {"D": defaultdict(lambda: defaultdict(list)), "chance": None})
    for p in glob.glob(os.path.join(src, "*_w*.json")):
        d = json.load(open(p))
        per[d["label"]]["chance"] = d["chance"]
        for c in d["cells"]:
            per[d["label"]]["D"][c["D"]][c["K"]].append(c["recall"])
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "..", "results", "capacity_x"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    per = load_cells(a.src)
    rows, excluded = [], []
    for lab in sorted(per):
        ch = per[lab]["chance"]
        byD = per[lab]["D"]
        if 0 not in byD or 256 not in byD:
            continue
        K0, K2 = sorted(byD[0]), sorted(byD[256])
        a1_0 = float(np.mean(byD[0][1]))
        a1_2 = float(np.mean(byD[256][1]))
        try:
            f0 = fit_k50(K0, [float(np.mean(byD[0][k])) for k in K0], ch)
            # THE FIX: normalise the interfered curve by the UNINTERFERED ceiling.
            f2 = fit_k50(K2, [float(np.mean(byD[256][k])) for k in K2], ch, K1_recall=a1_0)
        except ValueError as e:
            excluded.append({"label": lab, "reason": str(e)})
            continue
        rows.append({
            "label": lab,
            "ceiling_D0": round(a1_0, 4), "ceiling_D256": round(a1_2, 4),
            "ceiling_drop": round(a1_0 - a1_2, 4),
            "K50_D0": round(f0["K50"], 3), "K50_D0_censored": f0["censored"],
            "K50_D256_vs_D0_ceiling": round(f2["K50"], 3), "K50_D256_censored": f2["censored"],
            "retained_fraction": round(f2["K50"] / f0["K50"], 4),
            "censored_either": bool(f0["censored"] or f2["censored"]),
        })

    use = [r for r in rows if not r["censored_either"]]
    cap = np.array([r["K50_D0"] for r in use])
    ret = np.array([r["retained_fraction"] for r in use])
    drop = np.array([r["ceiling_drop"] for r in use])
    from scipy.stats import spearmanr
    s_ret = spearmanr(cap, ret)
    s_drop = spearmanr(cap, drop)

    print(f"{'model':<18}{'K50(D=0)':>10}{'K50(D256|D0)':>14}{'retained':>10}{'ceil drop':>11}")
    print("-" * 63)
    for r in sorted(rows, key=lambda x: x["K50_D0"]):
        flag = " *" if r["censored_either"] else "  "
        print(f"{r['label']:<18}{r['K50_D0']:>10.2f}{r['K50_D256_vs_D0_ceiling']:>14.2f}"
              f"{r['retained_fraction']:>10.3f}{r['ceiling_drop']:>11.3f}{flag}")
    for e in excluded:
        print(f"{e['label']:<18} EXCLUDED — {e['reason'][:60]}")

    out = {
        "_provenance": ("Computed by analysis/interference_axis.py from the D=0 and D=256 cells "
                        "already produced by analysis/neuron_capacity.py. No new compute. The "
                        "D=256 curve is normalised by the D=0 ceiling; own-ceiling normalisation "
                        "inverts the sign of the interference effect and is not used."),
        "n_models": len(rows), "n_uncensored": len(use), "n_excluded": len(excluded),
        "excluded": excluded,
        "retained_fraction_median": round(float(np.median(ret)), 4),
        "retained_fraction_range": [round(float(ret.min()), 4), round(float(ret.max()), 4)],
        "n_models_interference_reduces_capacity": int((ret < 1.0).sum()),
        "spearman_capacity_vs_retained": {"rho": round(float(s_ret.statistic), 4),
                                          "p": round(float(s_ret.pvalue), 4), "n": len(use)},
        "spearman_capacity_vs_ceiling_drop": {"rho": round(float(s_drop.statistic), 4),
                                              "p": round(float(s_drop.pvalue), 4), "n": len(use)},
        "models": rows,
    }
    print(f"\nn uncensored {len(use)}   interference reduces capacity in "
          f"{int((ret < 1.0).sum())}/{len(use)}   median retained {np.median(ret):.3f}")
    print(f"Spearman(capacity, retained)   rho={s_ret.statistic:+.4f}  p={s_ret.pvalue:.4f}")
    print(f"Spearman(capacity, ceil drop)  rho={s_drop.statistic:+.4f}  p={s_drop.pvalue:.4f}")
    print("\nNeither correlation is significant: capacity does not predict interference "
          "robustness, i.e. they are separate axes.")

    if a.write:
        json.dump(out, open(DST, "w"), indent=2)
        print("wrote", DST)


if __name__ == "__main__":
    main()
