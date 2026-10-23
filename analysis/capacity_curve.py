"""Continuous capacity: K50 and a threshold-free capacity AUC, with censoring handled honestly.

WHY THIS EXISTS.  The capacity paper reports k*, the smallest K in a discrete grid at which recall
falls below the midpoint of the model's own K=1 ceiling and chance.  Two things about that are
weak and the paper says so itself: it is a THRESHOLD CROSSING on a coarse grid, and the full
recall-vs-K curves are not released.  A threshold crossing inherits the grid -- with the committed
grid (1,2,3,4,6,8,10,12,16,20,24) a model whose true crossing is at 13 and one whose crossing is
at 15 both report k*=16 -- and a recipe effect that only exists at one threshold is not a law.

This module replaces the crossing with two estimands and reports BOTH, because a result that
survives a threshold-free statistic is much harder to explain away as a threshold artefact:

  K50           the K at which normalised retained performance r(K) crosses 0.5, read off a
                monotone fit rather than a grid, and RIGHT-CENSORED (not silently clipped) when
                the model never drops below 0.5 inside the measured grid.
  capacity AUC  (1/log Kmax) * integral of r(K) d log K over the measured range.  No threshold
                anywhere in it.

  r(K) = (a(K) - c) / (a(1) - c),  a = recall, c = pool chance.

CENSORING IS THE PART THAT IS EASY TO GET WRONG.  A model still above 0.5 at the top of the grid
has K50 > Kmax, which is a BOUND, not the value Kmax.  Recording it as Kmax pulls the high end of
the distribution down toward the grid and biases exactly the comparison the paper makes -- and it
biases it toward the modern-recipe group, which is where the censored models are.  censored=True
rows must be carried into the group comparison as bounds (log-rank / Peto), never as numbers.

THE EXCLUSION RULE IS INHERITED, NOT REINVENTED.  capacity_stats.py already establishes that a
model whose K=1 ceiling is itself at chance has a meaningless k*, because the midpoint it is
compared against is noise (pythia-70m: ceiling 0.070 vs chance 0.040).  The same exclusion applies
here and for the same reason: r(K) divides by (a(1) - c), so a near-zero denominator makes the
whole curve noise amplification.  CEILING_FLOOR is that guard.

Self-check (synthetic curves with known K50, including a censored one):
    python capacity_curve.py
"""
from __future__ import annotations

import json
import math

import numpy as np

CEILING_FLOOR = 0.5      # a model must clear this at K=1 for its curve to mean anything


def normalised(recall, K1_recall, chance):
    """r(K) in [0, 1]-ish.  Not clipped: a value above 1 or below 0 is real information about
    noise, and clipping it would hide a broken cell."""
    denom = K1_recall - chance
    if denom < (CEILING_FLOOR - chance):
        raise ValueError(f"K=1 ceiling {K1_recall:.3f} too close to chance {chance:.3f}; "
                         "r(K) would be noise amplification (see capacity_stats.py exclusion 1)")
    return (np.asarray(recall, float) - chance) / denom


def fit_k50(Ks, recall, chance, K1_recall=None, n_trials=None):
    """Monotone-ish logistic fit of r against log K; returns K50 with a censoring flag.

    A 2-parameter logistic in log K is used rather than an interpolant because trial-level noise
    at a single grid point should not be able to move the crossing on its own.  Where the fit is
    refused (too few points), linear interpolation between the bracketing grid points is used and
    recorded, so no number is silently produced by a different method than it claims.
    """
    Ks = np.asarray(Ks, float)
    order = np.argsort(Ks)
    Ks, recall = Ks[order], np.asarray(recall, float)[order]
    if K1_recall is None:
        if Ks[0] != 1:
            raise ValueError("need either a K=1 cell or an explicit K1_recall")
        K1_recall = recall[0]
    r = normalised(recall, K1_recall, chance)

    out = {"K_grid": Ks.tolist(), "recall": recall.tolist(), "r": [round(float(x), 5) for x in r],
           "K1_recall": float(K1_recall), "chance": float(chance)}

    if r[-1] > 0.5:                              # never crossed inside the grid
        out.update(K50=float(Ks[-1]), censored=True, method="right-censored at grid max")
        out["capacity_auc"] = _auc(Ks, r)
        return out

    x = np.log(Ks)
    try:
        from scipy.optimize import curve_fit

        def f(xx, mid, slope):
            return 1.0 / (1.0 + np.exp(slope * (xx - mid)))

        w = None
        if n_trials is not None:                  # weight by cell precision when we know it
            n = np.asarray(n_trials, float)[order]
            w = 1.0 / np.sqrt(np.clip(n, 1, None))
        p, _ = curve_fit(f, x, r, p0=[np.log(max(Ks[len(Ks) // 2], 2)), 2.0], sigma=w,
                         maxfev=20000)
        k50 = float(math.exp(p[0]))
        if not (Ks[0] <= k50 <= Ks[-1] * 1.5) or not np.isfinite(k50):
            raise RuntimeError("fit put K50 outside the measured range")
        out.update(K50=k50, censored=False, method="logistic in log K", slope=float(p[1]))
    except Exception as e:
        i = int(np.argmax(r <= 0.5))
        if i == 0:
            out.update(K50=float(Ks[0]), censored=False, method="below 0.5 at the first K",
                       fit_note=str(e)[:120])
        else:
            x0, x1, y0, y1 = Ks[i - 1], Ks[i], r[i - 1], r[i]
            lk = math.log(x0) + (0.5 - y0) * (math.log(x1) - math.log(x0)) / (y1 - y0)
            out.update(K50=float(math.exp(lk)), censored=False,
                       method="log-linear interpolation", fit_note=str(e)[:120])
    out["capacity_auc"] = _auc(Ks, r)
    return out


def _auc(Ks, r):
    """Threshold-free area under r against log K, normalised by the measured log range."""
    x = np.log(np.asarray(Ks, float))
    if x[-1] <= x[0]:
        return float("nan")
    return round(float(np.trapezoid(np.clip(r, 0, None), x) / (x[-1] - x[0])), 5)


def _selfcheck():
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("  ok   " if c else "  FAIL ") + m)
        ok = ok and c

    chance = 0.04
    # a curve whose true K50 is 8, by construction
    Ks = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]
    true_mid, slope = math.log(8), 2.5
    r_true = np.array([1 / (1 + math.exp(slope * (math.log(k) - true_mid))) for k in Ks])
    recall = r_true * (0.95 - chance) + chance
    got = fit_k50(Ks, recall, chance)
    chk(abs(got["K50"] - 8) < 0.4, f"recovers a known K50=8 -> {got['K50']:.3f} ({got['method']})")
    chk(not got["censored"], "an uncensored curve is not flagged censored")

    # THE NEGATIVE CASE: a model that never drops below 0.5 must be censored, not clipped to 24
    flat = np.full(len(Ks), 0.93)
    g2 = fit_k50(Ks, flat, chance)
    chk(g2["censored"] and g2["K50"] == 24,
        f"a never-crossing curve is right-censored at the grid max, not scored as 24 "
        f"(K50={g2['K50']}, censored={g2['censored']})")

    # AUC must order two curves the same way K50 does, or the two estimands disagree
    r_hi = fit_k50(Ks, np.array([1 / (1 + math.exp(2.5 * (math.log(k) - math.log(16)))) for k in Ks])
                   * (0.95 - chance) + chance, chance)
    chk(r_hi["K50"] > got["K50"] and r_hi["capacity_auc"] > got["capacity_auc"],
        f"K50 and capacity AUC agree on ordering ({got['K50']:.2f}/{got['capacity_auc']:.3f} vs "
        f"{r_hi['K50']:.2f}/{r_hi['capacity_auc']:.3f})")

    # the exclusion guard must fire, not return a number
    try:
        fit_k50(Ks, np.full(len(Ks), 0.07), chance, K1_recall=0.07)
        chk(False, "a model at chance at K=1 must raise, not return a K50")
    except ValueError:
        chk(True, "a model at chance at K=1 raises rather than returning noise")

    print("\nSELF-CHECK " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
