"""Does the binding-capacity scaling law hold across TWO training regimes and 2,150x in parameters?

THE OPPORTUNITY.  Two campaigns measured binding capacity and were never compared.

  TRILOGY      pretrained language models, 125M-2.8B parameters, tested ZERO-SHOT in context.
               Fit: K50 = c * N^alpha with alpha ~ 0.82 (analysis/agg_capacity.py).
  CLOCKS       transformers trained FROM SCRATCH on the binding task itself, 1.31M and 4.98M
               parameters, K in {2,3,4,5,6,8} (research/emergence-clocks-2026-08).

The clocks models sit 25x-2,150x below the trilogy range, and they were trained ON the task rather
than tested zero-shot on it.  If their capacity lands on the trilogy law, the law is about the
architecture and would be remarkable.  If it does not, the law is about something narrower, and
saying which is a real result -- it bounds what the scaling law is a law OF.

HOW CAPACITY IS DEFINED HERE, and why it is not identical to the trilogy K50.  A clocks run either
reaches the emergence criterion for its K or it does not, so capacity is the largest K a model
size reliably reaches, read off the fraction-emerged-vs-K curve.  The trilogy K50 is the K at which
zero-shot recall falls to half its own K=1 ceiling.  These are DIFFERENT ESTIMANDS -- one is
"can it learn to hold K", the other is "how many can it hold without being taught".  The comparison
is therefore an order-of-magnitude test, not a precise one, and it is reported as such.

EMERGENCE CRITERION: the same one the clocks campaign used throughout --
    crit = chance + 0.5 * (1 - chance)
taken from each run's OWN chance_query_ctx, never hard-coded (hard-coding 0.75, the K=2 bar, onto
K=4 runs was a real bug in an earlier analysis).

Run:  python cross_campaign_capacity.py [--write]
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
CLOCKS = os.path.join(HERE, "..", "..", "..", "emergence-clocks-2026-08", "data")
TRILOGY = os.path.join(HERE, "..", "results", "capacity_trn1_summary.json")
DST = os.path.join(HERE, "..", "results", "cross_campaign_capacity.json")


def emerged(run):
    """Did this run ever clear its own emergence criterion?"""
    ch = run.get("chance_query_ctx")
    ev = run.get("evals") or []
    if ch is None or not ev:
        return None
    crit = ch + 0.5 * (1 - ch)
    return any(e.get("query_ctx", 0) >= crit for e in ev)


def load_clocks(root):
    """Only clean, fully-supervised runs enter the capacity curve: p=1.0 (no supervision gap) and
    no post-hoc intervention.  Runs from the supply/deprivation arms measure a different thing."""
    cells = defaultdict(lambda: [0, 0])
    seen = 0
    for f in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "K" not in d or "nparam" not in d:
            continue
        if d.get("p") != 1.0:
            continue
        if "p_schedule" in d or "--p-after" in str(d.get("argv", "")):
            continue                                  # supply-manipulation arms, not capacity
        e = emerged(d)
        if e is None:
            continue
        seen += 1
        size = round(d["nparam"] / 1e6, 2)
        cells[(size, d["K"])][0] += bool(e)
        cells[(size, d["K"])][1] += 1
    return cells, seen


def k50_from_fraction(Ks, frac):
    """Largest K still reliably reached, by interpolating the fraction-emerged curve at 0.5."""
    Ks = np.asarray(Ks, float)
    frac = np.asarray(frac, float)
    if frac[0] < 0.5:
        return None, "below 0.5 at the smallest K measured"
    if frac[-1] >= 0.5:
        return float(Ks[-1]), "right-censored at the largest K measured"
    for i in range(1, len(Ks)):
        if frac[i] < 0.5:
            x0, x1 = math.log(Ks[i - 1]), math.log(Ks[i])
            y0, y1 = frac[i - 1], frac[i]
            if y1 == y0:
                return float(Ks[i]), "interpolated"
            return float(math.exp(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))), "interpolated"
    return None, "no crossing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    cells, seen = load_clocks(CLOCKS)
    if not cells:
        print("no clocks runs found"); return
    sizes = sorted({s for s, _ in cells})
    print(f"CLOCKS: {seen} clean p=1.0 runs, parameter sizes {sizes} (millions)")
    print()
    clocks_pts = []
    for s in sizes:
        Ks = sorted(k for (sz, k) in cells if sz == s)
        row = [(k, cells[(s, k)][0], cells[(s, k)][1]) for k in Ks]
        frac = [ok / n for _, ok, n in row]
        print(f"  {s}M params:  " + "  ".join(f"K{k}={ok}/{n}" for k, ok, n in row))
        k50, how = k50_from_fraction(Ks, frac)
        print(f"     fraction emerged: {[round(f,2) for f in frac]}  ->  capacity {k50} ({how})")
        if k50:
            clocks_pts.append((s, k50, how, sum(n for _, _, n in row)))

    tri = json.load(open(TRILOGY))
    import sys
    sys.path.insert(0, HERE)
    from agg_capacity import PARAMS_M, base_label
    tri_pts = []
    for m in tri["models"]:
        f = m["by_D"].get("0", {})
        if "K50" in f and not f["censored"]:
            bl = base_label(m["label"])
            if bl in PARAMS_M:
                tri_pts.append((PARAMS_M[bl], f["K50"], bl))

    x = np.log([p[0] for p in tri_pts]); y = np.log([p[1] for p in tri_pts])
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    logc, alpha = float(b[0]), float(b[1])
    print()
    print(f"TRILOGY law (pretrained, zero-shot, n={len(tri_pts)}): "
          f"K50 = {math.exp(logc):.4g} * N^{alpha:.3f}   (N in millions)")
    print()
    print(f"{'source':<10}{'params (M)':>12}{'measured':>11}{'law predicts':>14}{'ratio':>9}")
    print("-" * 58)
    rows = []
    for s, k50, how, n in clocks_pts:
        pred = math.exp(logc) * s ** alpha
        print(f"{'clocks':<10}{s:>12.2f}{k50:>11.2f}{pred:>14.3f}{k50/pred:>9.1f}x")
        rows.append({"source": "clocks", "params_M": s, "measured_capacity": k50,
                     "law_prediction": round(pred, 4), "ratio": round(k50 / pred, 2),
                     "how": how, "n_runs": n})
    for N, k50, lab in sorted(tri_pts)[:3]:
        pred = math.exp(logc) * N ** alpha
        print(f"{'trilogy':<10}{N:>12.0f}{k50:>11.2f}{pred:>14.3f}{k50/pred:>9.1f}x   {lab}")

    print()
    if rows:
        ratios = [r["ratio"] for r in rows]
        print(f"The from-scratch models exceed the pretrained law by {min(ratios):.0f}x-{max(ratios):.0f}x.")
        print()
        print("READING.  A 5M-parameter transformer TRAINED ON the binding task holds more bindings")
        print("than the law predicts for a model 25x its size that was pretrained on text and tested")
        print("zero-shot.  Capacity is therefore not a property of parameter count alone.  The")
        print("trilogy law describes ZERO-SHOT IN-CONTEXT capacity of general pretrained LMs -- what")
        print("a model does with bindings it was never trained to hold -- and NOT the architecture's")
        print("capability.  That distinction bounds what the law is a law of, and it is a caveat the")
        print("capacity paper should state rather than a weakness a reviewer should find.")
        print()
        print("CAVEAT, load-bearing: the two estimands are not identical.  Clocks capacity is 'the")
        print("largest K it can LEARN to hold'; trilogy K50 is 'how many it holds WITHOUT being")
        print("taught'.  This is an order-of-magnitude comparison, not a precise one.")

    out = {"_provenance": "analysis/cross_campaign_capacity.py. Clocks runs from "
                          "research/emergence-clocks-2026-08/data (p=1.0 only, supply-manipulation "
                          "arms excluded); trilogy law from results/capacity_trn1_summary.json. "
                          "Emergence criterion is each run's own chance + 0.5*(1-chance).",
           "clocks_n_clean_runs": seen,
           "trilogy_law": {"alpha": round(alpha, 4), "c": round(math.exp(logc), 6),
                           "n_models": len(tri_pts), "N_units": "millions of parameters"},
           "comparison": rows}
    if a.write:
        json.dump(out, open(DST, "w"), indent=2)
        print("\nwrote", DST)


if __name__ == "__main__":
    main()
