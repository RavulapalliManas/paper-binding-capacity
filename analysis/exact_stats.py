"""Exact small-sample statistics, because our n is small and the asymptotic ones lie.

`scipy.stats.spearmanr` computes its p-value from a t-approximation. For a perfectly monotone relationship
it returns EXACTLY 0.0, and we reported that as "p < 1e-4".

That is not merely optimistic. It is impossible. With n points there are n! orderings, and exactly 2 of them
achieve |rho| = 1. So the smallest attainable two-sided p-value is 2/n!:

    n = 5   ->  2/120   = 1.7e-2
    n = 6   ->  2/720   = 2.8e-3
    n = 7   ->  2/5040  = 4.0e-4      <- our checkpoint studies
    n = 8   ->  2/40320 = 5.0e-5

So "p < 1e-4" cannot be true of a 7-point correlation, no matter how clean. A reviewer who knows this reads
it as evidence that we did not check our own estimator.

The same asymptotic bias hits our other n = 7 correlation: scipy gives p = 0.0068 for rho = -0.893, while the
exact permutation p is 0.0123 -- nearly double.

Use `spearman_exact` for n <= 9 (9! = 362,880 permutations, still fast). Use `sign_test` for the "13/16
models" style counts, which we have so far reported with no test at all.
"""
from __future__ import annotations
from itertools import permutations
from math import comb, factorial

import numpy as np
from scipy.stats import spearmanr

EXACT_MAX_N = 9


def spearman_exact(x, y):
    """Spearman rho with an EXACT two-sided permutation p-value.

    Returns (rho, p_exact, n, p_asymptotic).  Falls back to the asymptotic p for n > EXACT_MAX_N and says so
    by returning p_exact = None.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    rho, p_asym = spearmanr(x, y)
    if n > EXACT_MAX_N:
        return float(rho), None, n, float(p_asym)

    base = np.arange(n)
    obs = abs(float(rho))
    hits = sum(1 for perm in permutations(base)
               if abs(spearmanr(base, np.asarray(perm)).correlation) >= obs - 1e-12)
    return float(rho), hits / factorial(n), n, float(p_asym)


def min_attainable_p(n):
    """The smallest two-sided Spearman p-value reachable with n points."""
    return 2 / factorial(n)


def sign_test(k, n):
    """Two-sided exact binomial test of k successes in n trials against p = 1/2.

    This is what "positive on 13/16 models" needs and does not have.
    """
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _fmt(p):
    return "< 1e-16" if p < 1e-16 else f"{p:.2e}"


if __name__ == "__main__":
    print("Every small-sample statistic in the papers, exact vs as reported.\n")

    cases = [
        ("sep(t) vs step  (capacity_control)",
         [1000, 8000, 16000, 32000, 64000, 96000, 143000],
         [-0.052, 0.000, 0.047, 0.076, 0.226, 0.288, 0.317], "p < 1e-4"),
        ("frozen probe vs RL step",
         [0, 400, 1100, 1700, 2300, 2500, 2900],
         [0.864, 0.836, 0.825, 0.764, 0.731, 0.717, 0.714], "p = 0.000"),
        ("refit probe vs RL step",
         [0, 400, 1100, 1700, 2300, 2500, 2900],
         [0.864, 0.833, 0.831, 0.808, 0.789, 0.794, 0.792], "p = 0.007"),
    ]
    for name, xs, ys, reported in cases:
        rho, pe, n, pa = spearman_exact(xs, ys)
        print(f"{name}\n  rho = {rho:+.3f}   n = {n}")
        print(f"  reported : {reported}")
        print(f"  scipy    : {_fmt(pa)}   (asymptotic; wrong at this n)")
        print(f"  EXACT    : {pe:.2e}      floor at n={n} is {min_attainable_p(n):.2e}\n")

    print("Model-count claims, which currently carry no test at all:")
    for k, n, what in [(13, 16, "certificate beats self-confidence"),
                       (14, 16, "pAcc|wrong clears the present-token baseline"),
                       (8, 8, "steering lifts accuracy"),
                       (5, 6, "prompt repair lifts accuracy"),
                       (7, 8, "certificate beats predictive entropy")]:
        print(f"  {k}/{n:<3} {what:<48} two-sided sign test p = {sign_test(k, n):.4f}")
