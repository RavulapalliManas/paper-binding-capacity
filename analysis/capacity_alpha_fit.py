"""The zero-shot capacity power-law fit, committed. K50 = c * N^alpha over the uncensored D=0 models.

WHY THIS EXISTS. The fit (alpha 0.820, R2 0.726, family-clustered CI [0.674, 1.251], leave-one-
family-out range) was computed inline during the campaign and quoted in the ledger and reports,
but no committed script produced it — a number with no committed generator, which cannot be audited. This
script is that provenance. The analysis is POST-HOC (claim ledger BC-2): it was noticed because
the scale coefficient's interval contained 1.0.

Method, matching the original inline computation exactly:
  - rows: capacity_trn1_summary.json models, D=0, censored excluded, params from
    agg_capacity.PARAMS_M (labels stripped of the "-x" wave suffix);
  - OLS of log10 K50 on log10 N (N in millions);
  - family-clustered bootstrap: resample the families with replacement 20,000 times, refit,
    percentile CI over runs with >= 2 distinct families and >= 3 points;
  - leave-one-family-out: refit with each family deleted, report the alpha range;
  - leave-one-model-out: refit with each single model deleted, report the max |delta alpha|
    (backs the manuscript sentence "removing any single model moves alpha by at most X").

Run:  python capacity_alpha_fit.py [--write]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agg_capacity import PARAMS_M, base_label  # noqa: E402

RES = os.path.join(HERE, "..", "results")
DST = os.path.join(RES, "capacity_alpha_fit.json")


def rows():
    d = json.load(open(os.path.join(RES, "capacity_trn1_summary.json")))
    out = []
    for m in d["models"]:
        cell = m["by_D"]["0"]
        b = base_label(m["label"])
        if cell["censored"] or b not in PARAMS_M:
            continue
        out.append({"label": b, "family": m["family"], "N": PARAMS_M[b], "K50": cell["K50"]})
    return out


def fit(rs):
    x = np.log10([r["N"] for r in rs])
    y = np.log10([r["K50"] for r in rs])
    alpha, logc = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (alpha * x + logc)) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(alpha), float(logc), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rs = rows()
    fams = sorted({r["family"] for r in rs})
    alpha, logc, r2 = fit(rs)
    print(f"n = {len(rs)} uncensored models, {len(fams)} families: {fams}")
    print(f"alpha = {alpha:.3f}   c = {10**logc:.4f}   R2 = {r2:.3f}")

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(20000):
        pick = rng.choice(fams, size=len(fams), replace=True)
        sample = [r for f in pick for r in rs if r["family"] == f]
        if len({r["family"] for r in sample}) < 2 or len(sample) < 3:
            continue
        boots.append(fit(sample)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"family-clustered CI (n_boot_valid={len(boots)}): [{lo:.3f}, {hi:.3f}]")

    lofo = {}
    for f in fams:
        sub = [r for r in rs if r["family"] != f]
        lofo[f] = round(fit(sub)[0], 3)
    print(f"leave-one-family-out alpha: {lofo}  range [{min(lofo.values()):.3f}, "
          f"{max(lofo.values()):.3f}]")

    loo = {}
    for i, r in enumerate(rs):
        sub = rs[:i] + rs[i + 1:]
        loo[r["label"]] = round(fit(sub)[0], 3)
    loo_max = max(abs(v - alpha) for v in loo.values())
    print(f"leave-one-model-out alpha: {loo}")
    print(f"max |delta alpha| over single-model deletion: {loo_max:.3f}")

    if a.write:
        json.dump({"_provenance": "analysis/capacity_alpha_fit.py over capacity_trn1_summary."
                                  "json (D=0, censored excluded, params from agg_capacity."
                                  "PARAMS_M). POST-HOC per claim ledger BC-2.",
                   "n_models": len(rs), "families": fams,
                   "alpha": round(alpha, 3), "c": round(10 ** logc, 4), "r2": round(r2, 3),
                   "alpha_ci95_family_clustered": [round(float(lo), 3), round(float(hi), 3)],
                   "n_boot_valid": len(boots),
                   "leave_one_family_out_alpha": lofo,
                   "lofo_range": [min(lofo.values()), max(lofo.values())],
                   "leave_one_model_out_alpha": loo,
                   "loo_max_abs_delta": round(loo_max, 3)},
                  open(DST, "w"), indent=2)
        print("wrote", DST)


if __name__ == "__main__":
    main()
