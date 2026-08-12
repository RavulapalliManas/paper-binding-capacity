"""Regenerate the capacity/geometry figures (5, 7, 8) from committed data.

These figures previously existed as PDFs with no generator and no data, which meant no number in them
could be checked and no confidence interval could be produced. The source data is committed at
paper/results/figdata.json (30 models: k*, declaration-site subspace dimension, interference, packing
efficiency, per-model robustness curves).

Corrections baked in (each documented in CLAIM_LEDGER):

1. NAMING. The quantity called d_bind is measured at the ENTITY token of each declaration, which causally
   precedes its obligation, so it cannot carry the binding. It is the effective dimension of the
   declaration-site ENTITY code. The comparative results are unaffected; the label was wrong.

2. STATISTICS. 30 models are not 30 independent samples (10 families; Pythia and OLMo contribute 6 each).
   The paper reports the family-clustered statistics beside the model-level ones; this generator prints
   both to stdout as a cross-check against capacity_stats.json but draws NO statistics inside any panel —
   captions carry them via numbers.tex macros (single source).

3. POPULATION. Every panel uses the n=29 population (kstar >= 3): Pythia-70M's K=1 accuracy (0.070) is at
   chance (0.040), so its threshold crossing is not a measurement, and it is the sole source of the
   retracted "12x" range (CLAIM_LEDGER sec. 9). The pre-2026-08-07 fig8 drew all 30 — its scatter did not
   match the quoted n=29 correlations.

4. RECIPE TAGS (fig 7). The frozen robust entries tag mist/q05/q7 "old", contradicting figdata["geom"]
   and the paper's model list (Mistral-7B and Qwen2.5 are modern recipes). Recipe and k* are therefore
   joined from figdata["geom"] on the "model" field; figdata.json itself stays frozen (CLAIM_LEDGER
   sec. 8.9). The q05 sweep used Qwen2.5-0.5B-Instruct; k*=3 is the base 0.5B (disclosed in the paper).

2026-08-08 rebuild on pubstyle (the P1 house style): final-size design at the measured NeurIPS preprint
\\textwidth (397.485 pt = 5.50 in, identical to ICLR's), TEAL = modern recipe / CLAY = pre-2023 /
OCHRE = re-run overlay / GREY = censoring & baselines, direct labels over legend boxes, and a
pixel-space collision audit asserted at zero for every figure.

Peak memory: 30 models x ~6 floats — KB-scale; nothing is loaded but the committed JSON.

Usage: python gen_capacity_figs.py     (writes fig5_capacity.pdf, fig7_robustness.pdf, fig8_recipe.pdf)
"""
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

import pubstyle as ps
from pubstyle import CLAY, GREY, INK, OCHRE, TEAL, BODY, MATH, SMALL

ps.house()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "..", "results", "figdata.json")))
G_ALL = DATA["geom"]
G = [x for x in G_ALL if x["kstar"] >= 3]                      # n=29 measurement population
OLD_S = [x for x in G if x["recipe"] == "old"]
MOD_S = [x for x in G if x["recipe"] == "modern"]
FAMS = sorted({x["family"] for x in G})
CEIL = 24

W = 5.50  # measured \textwidth of neurips_2024 [preprint] (397.485 pt), same as ICLR


def cluster_boot(key, n=10000, seed=0):
    """Resample FAMILIES with replacement; the model is not the unit of independence."""
    rng = np.random.RandomState(seed)
    byfam = {f: [x for x in G if x["family"] == f] for f in FAMS}
    d = []
    for _ in range(n):
        s = rng.choice(FAMS, len(FAMS), replace=True)
        o = [x[key] for f in s for x in byfam[f] if x["recipe"] == "old"]
        m = [x[key] for f in s for x in byfam[f] if x["recipe"] == "modern"]
        if len(o) >= 2 and len(m) >= 2:
            d.append(np.median(m) - np.median(o))
    d = np.array(d)
    return float(np.median(d)), tuple(np.percentile(d, [2.5, 97.5]))


def fam_test(key):
    byfam = {f: [x for x in G if x["family"] == f] for f in FAMS}
    rec = {f: byfam[f][0]["recipe"] for f in FAMS}
    o = [np.median([x[key] for x in byfam[f]]) for f in FAMS if rec[f] == "old"]
    m = [np.median([x[key] for x in byfam[f]]) for f in FAMS if rec[f] == "modern"]
    return mannwhitneyu(o, m, alternative="two-sided")[1]


# ---------------------------------------------------------------- fig 5: capacity by recipe
def fig5():
    order = sorted(G, key=lambda x: (x["recipe"] != "old", x["kstar"]))
    xs = np.arange(len(order))
    ks = [x["kstar"] for x in order]
    cols = [CLAY if x["recipe"] == "old" else TEAL for x in order]

    fig = plt.figure(figsize=(W, 2.85))
    ax = fig.add_axes([0.085, 0.285, 0.905, 0.60])
    ps.ygrid(ax)
    ax.bar(xs, ks, color=cols, width=0.72, linewidth=0, zorder=2)
    for i, x in enumerate(order):                       # right-censored at the entity-pool ceiling
        if x["kstar"] >= CEIL:
            t = ax.text(i, ks[i] + 0.45, "$\\uparrow$", ha="center", va="bottom",
                        fontsize=SMALL, color=GREY)
            t.set_gid("deliberate")                     # intentionally adjacent to its own bar
    ax.set_xticks(xs)
    ax.set_xticklabels([x["label"] for x in order], rotation=90, fontsize=SMALL)
    ax.set_ylabel("$k^{\\ast}$ (bindings held)", fontsize=MATH)
    ax.set_yticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0, 29.5)
    ax.set_title("Capacity clusters by pretraining recipe, not scale", fontsize=BODY, loc="left", pad=4)
    n_old, n_mod = len(OLD_S), len(MOD_S)
    ax.text(0.015, 0.96, f"pre-2023 recipe (n={n_old})", transform=ax.transAxes,
            ha="left", va="top", fontsize=SMALL, color=CLAY)
    ax.text(0.015, 0.86, f"modern recipe (n={n_mod})", transform=ax.transAxes,
            ha="left", va="top", fontsize=SMALL, color=TEAL)
    ax.text(0.985, 0.96, "$\\uparrow$ right-censored at the pool ceiling (24)",
            transform=ax.transAxes, ha="right", va="top", fontsize=SMALL, color=GREY)

    kmin, kmax = min(ks), max(ks)
    assert ps.audit(fig, "fig5_capacity") == 0
    fig.savefig(os.path.join(HERE, "fig5_capacity.pdf"))
    plt.close(fig)
    print(f"fig5: k* {kmin}..{kmax} = {kmax // kmin}x range, {len(order)} models "
          f"(pythia-70m excluded), {len(FAMS)} families")


# ---------------------------------------------------------------- fig 7: robustness curves
def fig7():
    # Recipe and k* joined from the geom table on the "model" field (correction 4 above).
    kj = {g["model"]: g for g in G_ALL}
    kj["Qwen/Qwen2.5-0.5B-Instruct"] = kj["Qwen/Qwen2.5-0.5B"]
    NAME = {"dsc": "DeepSeek-C-6.7B", "mist": "Mistral-7B", "olmo7": "OLMo-2-7B",
            "olmoe": "OLMoE", "p14": "Pythia-1.4B", "p69": "Pythia-6.9B",
            "q05": "Qwen2.5-0.5B-It", "q7": "Qwen2.5-7B"}
    R = list(DATA["robust"])
    rerun = []                                          # committed re-runs drawn distinctly (OCHRE)
    dsweep_dir = os.path.join(HERE, "..", "results", "robust_dsweep")
    if os.path.isdir(dsweep_dir):
        for f in sorted(os.listdir(dsweep_dir)):
            if f.endswith(".json"):
                d = json.load(open(os.path.join(dsweep_dir, f)))
                if d.get("entry") and d["entry"].get("label") not in {x.get("label") for x in R}:
                    rerun.append(d["entry"])

    fig = plt.figure(figsize=(4.40, 2.60))
    ax = fig.add_axes([0.115, 0.165, 0.615, 0.72])
    ps.ygrid(ax)
    ends = []
    for x in R:
        gm = kj[x["model"]]
        col = TEAL if gm["recipe"] == "modern" else CLAY
        dash = (0, (3.2, 1.6)) if gm["kstar"] < x["fixed_K"] else "solid"
        ds = [p[0] for p in x["curve"]]
        rec = [p[1] for p in x["curve"]]
        ax.plot(ds, rec, ls=dash, marker="o", color=col, lw=1.1, ms=2.2, zorder=3)
        ends.append((rec[-1], NAME[x["label"]], col))
    for x in rerun:
        ds = [p[0] for p in x["curve"]]
        rec = [p[1] for p in x["curve"]]
        ax.plot(ds, rec, "-s", color=OCHRE, lw=1.4, ms=2.6, zorder=4)
        ends.append((rec[-1], x["label"] + " (re-run)", OCHRE))

    ax.axhline(0.04, color=GREY, lw=0.7, ls=(0, (1.6, 1.6)), zorder=1)
    ps.line_label(ax, 64, 0.04, "chance (0.04)", ha="left", side="above")
    ax.set_xlabel("distractor tokens $D$", fontsize=MATH)
    ax.set_ylabel("recall at $K{=}8$", fontsize=MATH)
    ax.set_xticks([0, 32, 64, 128, 256])
    ax.set_xlim(-6, 262)
    ax.set_ylim(0, 1.0)
    ax.set_title("Collapse rates do not follow the capacity ranking",
                 fontsize=BODY, loc="left", pad=4)

    # right-margin label column: slots separated in axes fraction, connectors to the curve ends
    ends.sort(key=lambda e: e[0])
    sep = 0.062
    ys = [ax.transLimits.transform((0, e[0]))[1] for e in ends]
    slots = list(ys)
    for i in range(1, len(slots)):
        slots[i] = max(slots[i], slots[i - 1] + sep)
    over = slots[-1] - 0.99
    if over > 0:
        slots = [s - over for s in slots]
        for i in range(len(slots) - 2, -1, -1):
            slots[i] = min(slots[i], slots[i + 1] - sep)
    for (endy, name, col), sy in zip(ends, slots):
        ax.annotate(name, xy=(256, endy), xycoords="data",
                    xytext=(1.05, sy), textcoords=ax.transAxes,
                    ha="left", va="center", fontsize=SMALL, color=col,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY,
                                    shrinkA=1, shrinkB=2))

    assert ps.audit(fig, "fig7_robustness") == 0
    fig.savefig(os.path.join(HERE, "fig7_robustness.pdf"))
    plt.close(fig)
    print(f"fig7: {len(R)} frozen + {len(rerun)} re-run models; recipe/k* joined from geom "
          f"(3 frozen robust recipe tags overridden: mist/q05/q7 are modern)")


# ---------------------------------------------------------------- fig 8: the geometry, honestly
def fig8():
    fig = plt.figure(figsize=(W, 2.30))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.55], left=0.075, right=0.94,
                          top=0.865, bottom=0.185, wspace=0.52)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax, key, letter, finding, ylab in [
            (axs[0], "pack", "a", "packing separates recipes", "$k^{\\ast}/d_{\\mathrm{decl}}$"),
            (axs[1], "offdiag", "b", "interference: model level only", "mean off-diag $\\cos^{2}$")]:
        o = [x[key] for x in OLD_S]
        m = [x[key] for x in MOD_S]
        ps.ygrid(ax)
        parts = ax.violinplot([o, m], positions=[0, 1], widths=0.7, showmedians=True,
                              showextrema=False)
        for j, b in enumerate(parts["bodies"]):
            b.set_facecolor(CLAY if j == 0 else TEAL)
            b.set_alpha(0.28)
            b.set_edgecolor("none")
        parts["cmedians"].set_color(INK)
        parts["cmedians"].set_linewidth(1.4)
        rng = np.random.RandomState(1)
        for j, s in enumerate([OLD_S, MOD_S]):
            xj = j + rng.uniform(-0.09, 0.09, len(s))
            ax.scatter(xj, [x[key] for x in s], s=11, color=CLAY if j == 0 else TEAL,
                       edgecolor="white", linewidths=0.4, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"pre-2023\n(n={len(o)})", f"modern\n(n={len(m)})"], fontsize=SMALL)
        ax.set_ylabel(ylab, fontsize=MATH)
        ps.panel_title(ax, letter, finding)

    ks = np.array([x["kstar"] for x in G], float)
    db = np.array([x["d_bind"] for x in G], float)
    od = np.array([x["offdiag"] for x in G], float)
    ax = axs[2]
    ps.ygrid(ax)
    sc = None
    for mk, sel in [("^", [i for i, x in enumerate(G) if x["recipe"] == "old"]),
                    ("o", [i for i, x in enumerate(G) if x["recipe"] == "modern"])]:
        sc = ax.scatter(db[sel], ks[sel], c=od[sel], cmap="cividis_r",
                        vmin=od.min(), vmax=od.max(), s=26, marker=mk,
                        edgecolor="#555555", linewidths=0.5, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("interference", fontsize=SMALL)
    cb.ax.tick_params(labelsize=SMALL)
    cb.solids.set_rasterized(False)                     # keep the whole figure vector
    ax.set_xlabel("$d_{\\mathrm{decl}}$ (entity subspace)", fontsize=MATH)
    ax.set_ylabel("$k^{\\ast}$", fontsize=MATH)
    ax.set_yticks([0, 8, 16, 24])
    ps.panel_title(ax, "c", "capacity vs entity-subspace size")
    ax.legend(handles=[Line2D([], [], marker="^", ls="", color=INK, ms=3.6, label="pre-2023"),
                       Line2D([], [], marker="o", ls="", color=INK, ms=3.6, label="modern")],
              loc="lower right", fontsize=SMALL, handletextpad=0.3, borderaxespad=0.2)

    assert ps.audit(fig, "fig8_recipe") == 0
    fig.savefig(os.path.join(HERE, "fig8_recipe.pdf"))
    plt.close(fig)

    # stdout cross-check against capacity_stats.json (the macros' source); nothing drawn from these
    r1, _ = pearsonr(db, ks)
    s1, _ = spearmanr(db, ks)
    r2, _ = pearsonr(od, ks)
    pk_d, pk_ci = cluster_boot("pack")
    od_d, od_ci = cluster_boot("offdiag")
    cs = json.load(open(os.path.join(HERE, "..", "results", "capacity_stats.json")))["stats"]
    print(f"fig8 (n={len(G)}): d_decl r={r1:+.2f} rho={s1:+.2f}; interference r={r2:+.2f}")
    print(f"      pack   family Δ={pk_d:+.3f} CI[{pk_ci[0]:+.3f},{pk_ci[1]:+.3f}] fam-p={fam_test('pack'):.4f}"
          f"  (capacity_stats: {cs['pack']['family_ci']})")
    print(f"      offdiag family Δ={od_d:+.3f} CI[{od_ci[0]:+.3f},{od_ci[1]:+.3f}] fam-p={fam_test('offdiag'):.4f}"
          f"  (capacity_stats: {cs['offdiag']['family_ci']})  <- CI touches zero")


if __name__ == "__main__":
    fig5()
    fig7()
    fig8()
    print("\nwrote fig5_capacity.pdf, fig7_robustness.pdf, fig8_recipe.pdf from results/figdata.json")
