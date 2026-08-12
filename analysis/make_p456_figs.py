"""Paper figures for the merged capacity paper (p3_capacity.tex) and training_history.tex,
from committed result JSONs only.

Every curve here is computed from a file under paper/results/ at build time; no literal data
values appear in this script. Outputs land in paper/figs_p456/ as vector PDFs.

Style: figs/pubstyle.py house style (7 pt body, 6.5 pt ticks, Helvetica, Type-42), the same
system as the flagship figures. Figures are drawn AT FINAL SIZE and included at exactly the
drawn width:
  * training_history.tex is article/10pt with 1.1 in margins -> \\textwidth = 6.30 in.
    p4_retention, p5_harm at 4.20 in = 0.667\\linewidth; p4_gaplaw (hero) at 4.60 in = 0.730.
  * p3_capacity.tex (neurips preprint) has \\textwidth = 5.50 in.
    p6_capacitylaw (hero) at 4.40 in = 0.8\\linewidth; p6_timelaw at 3.85 in = 0.7.
Uncertainty: proportion panels carry 95% Wilson intervals computed from the same committed
counts they plot. Every figure must pass the pubstyle pixel-space collision audit.

Run:  python make_p456_figs.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "figs_p456")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(HERE, "..", "figs"))
import pubstyle as ps  # noqa: E402
from pubstyle import TEAL, CLAY, GREY, BODY, SMALL, MATH  # noqa: E402

ps.house()
import matplotlib.pyplot as plt  # noqa: E402


def jl(name):
    return json.load(open(os.path.join(RES, name)))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def yerr(counts):
    """[[down], [up]] errorbar arrays from (k, n) pairs."""
    lo, hi = zip(*[wilson(k, n) for k, n in counts])
    y = [k / n for k, n in counts]
    return y, [[max(a - b, 0.0) for a, b in zip(y, lo)],
               [max(b - a, 0.0) for a, b in zip(hi, y)]]


FAILS = 0


def save(fig, name):
    global FAILS
    fig.tight_layout()
    n = ps.audit(fig, name)
    if n:
        FAILS += n
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


# ---------------- retention dose-response (training_history fig) ---------------------------
dvx = jl("devaxis_analyses.json")
fig, ax = plt.subplots(figsize=(4.20, 2.70))
ps.ygrid(ax)
for K, col, mk, ls in ((2, TEAL, "o", "-"), (4, CLAY, "s", "--")):
    pts = sorted((r["p"], r["held"], r["n"]) for r in dvx["dose_response"]
                 if r["K"] == K and r["p"] <= 0.06)
    y, err = yerr([(h, n) for _, h, n in pts])
    ax.errorbar([p for p, _, _ in pts], y, yerr=err, marker=mk, ls=ls, color=col,
                ms=4, lw=1.1, elinewidth=0.6, capsize=0, label=f"$K={K}$")
ax.set_xlabel("supervision rate through the deprivation window")
ax.set_ylabel("fraction retaining the capability")
ax.set_ylim(-0.04, 1.06)
ax.legend(loc="lower right")
save(fig, "p4_retention.pdf")

# ---------------- the gap law (training_history HERO) --------------------------------------
gl = jl("x14_gap_law.json")
xs = sorted(int(k) for k in gl["gap_axis"])
counts = [tuple(int(v) for v in gl["gap_axis"][str(g)].split("/")) for g in xs]
y, err = yerr(counts)
fig, ax = plt.subplots(figsize=(4.60, 2.90))
ps.ygrid(ax)
ax.errorbar(xs, y, yerr=err, marker="o", ls="-", color=TEAL, ms=4.5, lw=1.2,
            elinewidth=0.6, capsize=0)
ax.set_xscale("log")
ax.tick_params(axis="x", labelsize=MATH)
# per-point counts: the denominators are the evidence; offsets keep clear of markers and CIs
offs = {0: (7, -3), 1: (8, 2), 2: (8, 2), 3: (2, 9)}
for i, (g, (a, b)) in enumerate(zip(xs, counts)):
    ax.annotate(f"{a}/{b}", (g, y[i]), textcoords="offset points",
                xytext=offs.get(i, (7, 4)), fontsize=SMALL, color="#444444")
g50 = gl["fit"]["gap50"]
ax.axvline(g50, color=GREY, lw=0.8, ls=":")
ax.annotate(f"50% at gap $\\approx${g50:g}", (g50, 0.07), textcoords="offset points",
            xytext=(5, 0), fontsize=SMALL, color=GREY)
ax.set_xlabel("maximum unsupervised gap (steps); mean rate and totals matched")
ax.set_ylabel("fraction re-forming")
ax.set_ylim(-0.06, 1.13)
ax.set_xlim(8, 3.2e4)
save(fig, "p4_gaplaw.pdf")

# ---------------- the harm dissociation (training_history fig) -----------------------------
q5 = [r for r in jl("h100_reanalysis.json")["Q5_harm"] if r["warmup"] == 500]
fig, ax = plt.subplots(figsize=(4.20, 2.70))
ps.ygrid(ax)
for key, col, mk, ls, dodge, lab in (
        ("chess", TEAL, "o", "-", 0.97, "structured (chess)"),
        ("shuffle", CLAY, "s", "--", 1.03, "structure destroyed, same statistics")):
    ds = [r["dose"] * dodge for r in q5]  # small horizontal dodge: identical points overlap
    cnt = [tuple(int(v) for v in r[key].split("/")) for r in q5]
    y, err = yerr(cnt)
    ax.errorbar(ds, y, yerr=err, marker=mk, ls=ls, color=col, ms=4, lw=1.1,
                elinewidth=0.6, capsize=0, label=lab)
ax.legend(loc="lower left")
ax.set_xscale("log")
ax.tick_params(axis="x", labelsize=MATH)
ax.set_xlabel("priming dose (steps)")
ax.set_ylabel("fraction learning the target")
ax.set_ylim(-0.06, 1.13)
save(fig, "p5_harm.pdf")

# ---------------- the capacity law and its boundary (p3_capacity HERO) ---------------------
from agg_capacity import PARAMS_M, base_label  # noqa: E402

summ = jl("capacity_trn1_summary.json")
fit = jl("capacity_alpha_fit.json")
cross = jl("cross_campaign_capacity.json")
pts = []
for m in summ["models"]:
    cell = m["by_D"]["0"]
    b = base_label(m["label"])
    if cell["censored"] or b not in PARAMS_M:
        continue
    pts.append((PARAMS_M[b], cell["K50"]))
cb = cross["comparison"][0]
fig, ax = plt.subplots(figsize=(4.40, 3.05))
ax.set_xscale("log")
ax.set_yscale("log")
ax.tick_params(axis="both", labelsize=MATH)
xf = np.logspace(np.log10(2.5), np.log10(4000), 50)
ax.plot(xf, fit["c"] * xf ** fit["alpha"], "-", color=TEAL, lw=1, alpha=0.55)
ax.plot([p for p, _ in pts], [k for _, k in pts], "o", color=TEAL, ms=4.5)
# the boundary is the argument: dotted riser from the law's prediction up to the measured star
law_at = fit["c"] * cb["params_M"] ** fit["alpha"]
ax.plot([cb["params_M"], cb["params_M"]], [law_at, cb["measured_capacity"]],
        ls=":", color=CLAY, lw=0.9)
ax.plot([cb["params_M"]], [cb["measured_capacity"]], "*", color=CLAY, ms=12)
ax.annotate(f"task-trained: {cb['ratio']:.0f}$\\times$ the law",
            (cb["params_M"], cb["measured_capacity"]), textcoords="offset points",
            xytext=(2, 8), ha="left", fontsize=SMALL, color=CLAY)
ax.annotate("pretrained, zero-shot $K_{50}$", (14, 9), ha="left",
            fontsize=MATH, color=TEAL)  # carries a subscript: parent >= 7.2pt keeps it >= 5pt
ax.text(0.03, 0.97, f"$K_{{50}} = c\\,N^{{{fit['alpha']:.2f}}}$", transform=ax.transAxes,
        ha="left", va="top", fontsize=MATH, color=TEAL)
ax.set_xlabel("parameters (millions)")
ax.set_ylabel("binding capacity")
save(fig, "p6_capacitylaw.pdf")

# ---------------- the cost-of-load law, two codebases (p3_capacity fig) --------------------
eB = jl("extra_analyses.json")["B_emergence_time_vs_K"]["cells"]
eD = jl("devaxis_analyses.json")["emergence_law"]["devaxis"]["cells"]
law = jl("extra_analyses.json")["B_emergence_time_vs_K"]
lawD = jl("devaxis_analyses.json")["emergence_law"]["devaxis"]
h100 = sorted((int(k), v["median_step"]) for k, v in eB.items())
dvx2 = sorted((int(k), v["median"]) for k, v in eD.items())
fig, ax = plt.subplots(figsize=(3.85, 2.70))
ax.set_xscale("log")
ax.set_yscale("log")
ax.tick_params(axis="both", labelsize=MATH)
kf = np.logspace(np.log10(1.9), np.log10(8.6), 40)
ax.plot(kf, law["coef"] * kf ** law["exponent"], "-", color=TEAL, lw=0.8, alpha=0.35)
ax.plot(kf, lawD["coef"] * kf ** lawD["exponent"], "--", color=CLAY, lw=0.8, alpha=0.35)
ax.plot([k for k, _ in h100], [t for _, t in h100], marker="o", ls="none", color=TEAL, ms=4)
ax.plot([k for k, _ in dvx2], [t for _, t in dvx2], marker="s", ls="none", color=CLAY, ms=4)
ax.annotate("codebase A\n(8$\\times$H100)", (h100[-2][0], h100[-2][1]),
            textcoords="offset points", xytext=(-8, 6), ha="right",
            fontsize=SMALL, color=TEAL)
ax.annotate("codebase B\n(Trainium)", (dvx2[-1][0], dvx2[-1][1]),
            textcoords="offset points", xytext=(6, -3), ha="left", va="top",
            fontsize=SMALL, color=CLAY)
ax.set_xlim(1.75, 12.5)
kticks = sorted({k for k, _ in h100} | {k for k, _ in dvx2})
ax.set_xticks(kticks)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.minorticks_off()
ax.tick_params(axis="x", labelsize=SMALL)
ax.set_xlabel("binding load $K$")
ax.set_ylabel("median steps to formation")
save(fig, "p6_timelaw.pdf")

if FAILS:
    print(f"AUDIT FAILED: {FAILS} collision(s)")
    sys.exit(1)
print("all figures written to", OUT)
