"""Figures for the LP4FM camera-ready of the capacity paper, from committed result JSONs only.

  fig_p3_capacity.pdf      (a) recall against load for the 12 continuous-protocol models,
                           (b) K50 against parameter count with the fitted size law.
  fig_p3_interference.pdf  K50 with 256 distractor tokens against K50 without, under the two
                           normalisations of Section 4: (a) the uninterfered single-binding
                           recall, (b) each curve's own single-binding recall.
  fig_p3_development.pdf   Pythia checkpoints: (a) probe accuracy by model size, (b) Pythia-1.4B
                           recall and probe accuracy on its wrong answers, (c) wrong answers
                           repaired by pushes toward the gold value, the probe's answer, and a
                           random direction.

Inputs: results/capacity_trn1_summary.json, results/interference_axis.json,
        results/p3_revision_checks.json (parameter counts and the config-total size fit),
        results/p3_development.json (analysis/p3_development.py).
No data value is typed into this script. Drawn at final size for the NeurIPS 2026 template
(\\textwidth = 5.5 in) and checked with the pubstyle collision audit.

Run:  python make_p3_revision_figs.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
sys.path.insert(0, os.path.join(HERE, "..", "figs"))
sys.path.insert(0, HERE)
import pubstyle as ps  # noqa: E402
from pubstyle import TEAL, CLAY, GREY, INK, SMALL, MATH  # noqa: E402
from p3_revision_checks import PARAMS_TOTAL_M, MODELS  # noqa: E402

ps.house()
import matplotlib.pyplot as plt  # noqa: E402

PRETTY = {
    "olmo2-1b-x": "OLMo-2-1B", "opt-1.3b-x": "OPT-1.3B", "opt-125m-x": "OPT-125M",
    "opt-350m-x": "OPT-350M", "pythia-1.4b-x": "Pythia-1.4B", "pythia-160m-x": "Pythia-160M",
    "pythia-1b-x": "Pythia-1B", "pythia-2.8b-x": "Pythia-2.8B", "pythia-410m-x": "Pythia-410M",
    "qwen2.5-0.5b-x": "Qwen2.5-0.5B", "qwen2.5-1.5b-x": "Qwen2.5-1.5B",
    "smollm2-360m-x": "SmolLM2-360M",
}
MARK = {"opt": "o", "pythia": "s", "qwen2.5": "^", "smollm2": "D", "olmo2": "v"}
FAMNAME = {"opt": "OPT", "pythia": "Pythia", "qwen2.5": "Qwen2.5", "smollm2": "SmolLM2",
           "olmo2": "OLMo-2"}
FAILS = 0


def jl(name):
    return json.load(open(os.path.join(RES, name)))


def save(fig, out, name):
    global FAILS
    n = ps.audit(fig, name)
    FAILS += n
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print("wrote", name, "collisions:", n)


def models(summary):
    m = summary["models"]
    return m if isinstance(m, dict) else {x["label"]: x for x in m}


def fig_capacity(out):
    S = models(jl("capacity_trn1_summary.json"))
    chk = jl("p3_revision_checks.json")
    fit = chk["size_exponent"]["config_total"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.6),
                                   gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.42})

    # (a) raw recall against load, D = 0
    pools = []
    for lab in sorted(S, key=lambda k: PARAMS_TOTAL_M[k]):
        d0 = S[lab]["by_D"]["0"]
        K = np.array(d0["K_grid"], float)
        small = PARAMS_TOTAL_M[lab] < 1000
        ax1.plot(K, d0["recall"], "-", color=CLAY if small else TEAL, lw=0.9,
                 alpha=0.85, marker=MARK[S[lab]["family"]], ms=2.2, mew=0)
        pools.append(round(1 / S[lab]["chance"]))
    Kc = np.geomspace(1, 64, 200)
    V = float(np.median(pools))
    ax1.plot(Kc, 1 / Kc + (Kc - 1) / (Kc * V), "--", color=GREY, lw=0.8)
    ax1.axhline(1 / V, color=GREY, lw=0.6, ls=":")
    ps.logx(ax1)
    ax1.set_xlim(1, 64)
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax1.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax1.set_xlabel("bindings in the prompt, $K$")
    ax1.set_ylabel("recall")
    leg = [plt.Line2D([], [], color=CLAY, lw=0.9, label="below 1B"),
           plt.Line2D([], [], color=TEAL, lw=0.9, label="1B to 2.8B"),
           plt.Line2D([], [], color=GREY, lw=0.8, ls="--", label="guess among listed values"),
           plt.Line2D([], [], color=GREY, lw=0.6, ls=":", label="chance")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               fontsize=SMALL, handlelength=1.8, handletextpad=0.4, columnspacing=1.4)
    ps.panel_title(ax1, "a", "Recall against load")

    # (b) K50 against parameter count
    xs, ys = [], []
    for lab, m in S.items():
        d0 = m["by_D"]["0"]
        N = PARAMS_TOTAL_M[lab] * 1e6
        fam = m["family"]
        col = CLAY if PARAMS_TOTAL_M[lab] < 1000 else TEAL
        if d0["censored"]:
            ax2.plot([N], [d0["K50"]], MARK[fam], color=col, ms=4.2, mfc="white", mew=0.9)
            ax2.annotate("", xy=(N, d0["K50"] * 1.9), xytext=(N, d0["K50"] * 1.12),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8,
                                         mutation_scale=6))
        else:
            ax2.plot([N], [d0["K50"]], MARK[fam], color=col, ms=4.2, mew=0)
            xs.append(N)
            ys.append(d0["K50"])
    xf = np.geomspace(1.0e8, 3.2e9, 50)
    a = fit["alpha"]
    lx, ly = np.log(np.array(xs) / 1e6), np.log(ys)
    b = float(np.mean(ly - a * lx))
    ax2.plot(xf, np.exp(b) * (xf / 1e6) ** a, "-", color=INK, lw=0.7, alpha=0.5)
    ax2.axvspan(1.0e8, 1.0e9, color=CLAY, alpha=0.06, lw=0)
    floor = float(np.mean([v["K50"] for v in chk["floors_and_tests"]["entity_blind_floor"]["by_pool_size"].values()]))
    ax2.axhline(floor, color=GREY, lw=0.8, ls="--")
    ax2.text(3.6e8, floor / 1.06, "guess among listed values", color=GREY, fontsize=SMALL, va="top")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1.0e8, 3.2e9)
    ax2.set_ylim(1.6, 160)
    ax2.set_yticks([2, 4, 8, 16, 32, 64, 128])
    ax2.set_yticklabels(["2", "4", "8", "16", "32", "64", "128"])
    ax2.tick_params(axis="x", labelsize=MATH)
    ax2.set_xlabel("parameters, $N$")
    ax2.set_ylabel("capacity, $K_{50}$")
    ax2.text(1.15e8, 100, f"$K_{{50}}\\propto N^{{{a:.2f}}}$", fontsize=MATH, color=INK)
    handles = [plt.Line2D([], [], ls="none", marker=MARK[f], color=GREY, ms=3.6, mew=0,
                          label=FAMNAME[f]) for f in ("opt", "pythia", "qwen2.5", "smollm2", "olmo2")]
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=SMALL,
               handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
    ps.panel_title(ax2, "b", "Capacity against size")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.27, top=0.89)
    save(fig, out, "fig_p3_capacity.pdf")


def fig_interference(out):
    S = models(jl("capacity_trn1_summary.json"))
    I = {m["label"]: m for m in jl("interference_axis.json")["models"]}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), gridspec_kw={"wspace": 0.38})
    for ax, key, letter, title in (
            (axes[0], "fixed", "a", "Against recall without distractors"),
            (axes[1], "own", "b", "Against its own single-binding recall")):
        for lab, m in I.items():
            x = m["K50_D0"]
            if key == "fixed":
                y, ycens = m["K50_D256_vs_D0_ceiling"], m["K50_D256_censored"]
            else:
                d = S[lab]["by_D"]["256"]
                if "K50" not in d:     # own-reference curve undefined: recall at K=1 near chance
                    continue
                y, ycens = d["K50"], d["censored"]
            col = CLAY if PARAMS_TOTAL_M[lab] < 1000 else TEAL
            fam = S[lab]["family"]
            floor = key == "fixed" and y <= 1.0     # below the midpoint already at K = 1
            open_ = m["K50_D0_censored"] or ycens or floor
            ax.plot([x], [y], MARK[fam], color=col, ms=4.0, mfc="white" if open_ else col,
                    mew=0.9 if open_ else 0)
            if floor:
                ax.annotate("", xy=(x, y / 1.55), xytext=(x, y / 1.1),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8,
                                            mutation_scale=6))
            if m["K50_D0_censored"]:
                ax.annotate("", xy=(x * 1.8, y), xytext=(x * 1.12, y),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8,
                                            mutation_scale=6))
        g = np.geomspace(1.5, 130, 20)
        ax.plot(g, g, "-", color=GREY, lw=0.7)
        ax.plot(g, 0.5 * g, ":", color=GREY, lw=0.7)
        for a_ in (ax,):
            a_.set_xscale("log")
            a_.set_yscale("log")
            a_.set_xlim(1.5, 130)
            a_.set_ylim(0.55, 130)
            ticks = [2, 4, 8, 16, 32, 64]
            a_.set_xticks(ticks)
            a_.set_xticklabels([str(t) for t in ticks])
            a_.set_yticks(ticks)
            a_.set_yticklabels([str(t) for t in ticks])
        ax.set_xlabel("$K_{50}$, no distractors")
        ax.set_ylabel("$K_{50}$, with distractor block")
        ax.text(4.5, 11, "equal", color=GREY, fontsize=SMALL)
        ax.text(60, 17, "half", color=GREY, fontsize=SMALL)
        ps.panel_title(ax, letter, title)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.2, top=0.87)
    save(fig, out, "fig_p3_interference.pdf")


def fig_development(out):
    """Pythia checkpoints: (a) probe accuracy by size; (b) Pythia-1.4B recall and probe accuracy on its
    wrong answers; (c) Pythia-1.4B wrong answers repaired by pushes toward three targets.
    Input: results/p3_development.json (analysis/p3_development.py)."""
    dev = jl("p3_development.json")
    L, S = dev["ladder"], dev["steer"]["1.4B"]["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05), gridspec_kw={"wspace": 0.42})
    ticks = [1e3, 1e4, 1e5]
    tlab = ["$10^3$", "$10^4$", "$10^5$"]

    ax = axes[0]
    style = {"160M": (CLAY, "o"), "410M": (CLAY, "s"), "1B": (TEAL, "^"), "1.4B": (TEAL, "D"),
             "6.9B": (TEAL, "v")}
    for lab, (col, mk) in style.items():
        e = L[lab]
        ax.plot(e["steps"], e["acc"], "-", color=col, lw=0.8, alpha=0.9, marker=mk, ms=2.4, mew=0,
                label=f"Pythia-{lab}")
    base = 1 / L["1B"]["K"]
    ax.axhline(base, color=GREY, lw=0.8, ls="--")
    ax.axhline(L["1B"]["pool_chance"], color=GREY, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(8e2, 1.8e5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlab)
    ax.set_xlabel("pretraining step")
    ax.set_ylabel("probe accuracy")
    ax.legend(loc="upper left", fontsize=SMALL - 1, handlelength=1.2, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.15, frameon=False)
    ps.panel_title(ax, "a", "Probe, by model size")

    steps = [r["step"] for r in S]
    ax = axes[1]
    ax.plot(steps, [r["probe_wrong"] for r in S], "-", color=TEAL, lw=0.9, marker="D", ms=2.4, mew=0)
    ax.plot(steps, [r["acc"] for r in S], "-", color=INK, lw=0.9, marker="o", ms=2.4, mew=0)
    ax.axhline(base, color=GREY, lw=0.8, ls="--")
    ax.text(1.1e3, 0.80, "probe, on\nwrong answers", color=TEAL, fontsize=SMALL, va="bottom")
    ax.text(1.2e3, 0.27, "model", color=INK, fontsize=SMALL, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(8e2, 1.8e5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlab)
    ax.set_xlabel("pretraining step")
    ax.set_ylabel("accuracy")
    ps.panel_title(ax, "b", "Pythia-1.4B")

    ax = axes[2]
    ax.plot(steps, [r["gold_w_0.5"] for r in S], "-", color=INK, lw=0.9, marker="o", ms=2.4, mew=0)
    ax.plot(steps, [r["probe_w_0.5"] for r in S], "-", color=TEAL, lw=0.9, marker="D", ms=2.4, mew=0)
    ax.plot(steps, [r["random_0.5"] for r in S], "-", color=GREY, lw=0.8, marker="s", ms=2.2, mew=0)
    ax.text(1.2e3, 0.93, "gold", color=INK, fontsize=SMALL, va="bottom")
    ax.text(6.5e4, 0.68, "probe", color=TEAL, fontsize=SMALL, va="bottom")
    ax.text(2.0e4, 0.10, "random", color=GREY, fontsize=SMALL, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(8e2, 1.8e5)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlab)
    ax.set_xlabel("pretraining step")
    ax.set_ylabel("wrong answers repaired")
    ps.panel_title(ax, "c", "Steering the hidden state")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.21, top=0.87)
    save(fig, out, "fig_p3_development.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figs_p456"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_capacity(a.out)
    fig_interference(a.out)
    fig_development(a.out)
    if FAILS:
        raise SystemExit(f"{FAILS} collisions; fix label positions before using the figures")


if __name__ == "__main__":
    main()
