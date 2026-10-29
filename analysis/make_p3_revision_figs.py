"""Figures for the LP4FM camera-ready of the capacity paper, from committed result JSONs only.

  fig_p3_capacity.pdf      (a) recall against load for the 12 continuous-protocol models,
                           (b) K50 against parameter count with the fitted size law.
  fig_p3_interference.pdf  K50 with 256 distractor tokens against K50 without, under the two
                           normalisations of Section 4: (a) the uninterfered single-binding
                           recall, (b) each curve's own single-binding recall.
  fig_p3_development.pdf   Pythia checkpoints: (a) probe accuracy by model size, (b) Pythia-1.4B
                           accuracy, and probe accuracy on its wrong answers at the chosen middle
                           layer and at the final layer, (c) wrong answers
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


def save_with_data_hash(fig, path):
    """Fingerprint plotted series and labels, independently of PDF fonts/dates."""
    import hashlib
    panels = []
    for ax in fig.axes:
        panel = {"title": ax.get_title(loc="left") or ax.get_title(),
                 "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
                 "scales": [ax.get_xscale(), ax.get_yscale()],
                 "limits": [list(ax.get_xlim()), list(ax.get_ylim())],
                 "lines": [], "collections": []}
        for line in ax.lines:
            panel["lines"].append({"x": np.asarray(line.get_xdata(), dtype=float).tolist(),
                                   "y": np.asarray(line.get_ydata(), dtype=float).tolist(),
                                   "label": line.get_label() if not line.get_label().startswith("_") else ""})
        for collection in ax.collections:
            item = {}
            if hasattr(collection, "get_segments"):
                item["segments"] = [np.asarray(segment, dtype=float).tolist()
                                    for segment in collection.get_segments()]
            if hasattr(collection, "get_offsets"):
                item["offsets"] = np.asarray(collection.get_offsets(), dtype=float).tolist()
            panel["collections"].append(item)
        panels.append(panel)
    payload = json.dumps(panels, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    fig.savefig(path, metadata={"Subject": "p3-series-sha256:"+digest})


def save(fig, out, name):
    global FAILS
    n = ps.audit(fig, name)
    FAILS += n
    save_with_data_hash(fig, os.path.join(out, name))
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
    ax.plot(steps, [r["probe_wrong"] for r in S], "-", color=TEAL, lw=0.9, marker="D", ms=2.4, mew=0,
            label="probe, selected layer")
    ax.plot(steps, [r["probe_wrong_final"] for r in S], "--", color=TEAL, lw=0.9, marker="D", ms=2.2,
            mew=0, mfc="white", label="probe, final layer")
    ax.plot(steps, [r["acc"] for r in S], "-", color=INK, lw=0.9, marker="o", ms=2.4, mew=0,
            label="model")
    ax.axhline(base, color=GREY, lw=0.8, ls="--")
    ax.legend(loc="upper left", fontsize=SMALL - 1, handlelength=1.6, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.15, frameon=False)
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


def measured_revision(root, out):
    """Render the float32 revision from recorded results; run on the analysis host.

    No inference or probe fitting. Peak input memory is below 10 MB of JSON.
    The output summary records input hashes and the generator hash.
    """
    import glob
    import hashlib
    from capacity_curve import fit_k50

    inputs = {}

    def read(rel):
        path = os.path.join(root, rel)
        with open(path, "rb") as stream:
            raw = stream.read()
        inputs[rel] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def nice(lab):
        return {"gpt2": "GPT-2", "gpt2-medium": "GPT-2-medium",
                "gpt2-large": "GPT-2-large", "pythia-70m": "Pythia-70M"}.get(
                    lab, PRETTY.get(lab + "-x", lab))

    def curve(rec, cond):
        cells = [c for c in rec["cells"] if c["cond"] == cond]
        ks = sorted({c["K"] for c in cells})
        return ks, np.array([np.mean([c["recall"] for c in cells if c["K"] == k])
                             for k in ks])

    def at(rec, cond, k):
        cells = [c for c in rec["cells"] if c["cond"] == cond and c["K"] == k]
        assert cells and len({c["n"] for c in cells}) == 1
        return float(np.mean([c["recall"] for c in cells]))

    def interval(p, n):
        # Wilson score interval; descriptive, no model-level inference.
        z = 1.959963984540054
        mid = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1-p) / n + z*z / (4*n*n)) / (1 + z*z/n)
        return mid-half, mid+half

    caps = {}
    for path in sorted(glob.glob(os.path.join(root, "capacity_fp32", "*", "main.json"))):
        rel = os.path.relpath(path, root)
        rec = read(rel)
        assert rec["dtype"] == "float32", rel
        keys = {(c["cond"], c["K"], c["seed"]) for c in rec["cells"]}
        assert len(keys) == len(rec["cells"]) == 180 and all(c["n"] == 96 for c in rec["cells"])
        caps[rec["label"]] = rec
    assert len(caps) == 16
    frames = {lab: read("frame3/" + lab + "/main.json") for lab in
              ["pythia-160m", "pythia-410m", "smollm2-360m", "qwen2.5-0.5b", "pythia-1b"]}
    paired = read("probe_cap/heldout_paired_K16.json")["results"]
    ctl = read("probe_cap/heldout_ctl3.json")["checkpoints"]
    ckpt = read("probe_cap/p3_probe_ckpt.json")["checkpoints"]
    exploratory = read("probe_cap/p3_paired_ci_13_K16.json")["results"]
    overview = {"models": {}, "wording": {}, "heldout": paired,
                "exploratory_paired": exploratory, "checkpoint_K16": {}}
    small = [x for x in caps if x not in ["olmo2-1b", "opt-1.3b", "pythia-1b",
                                          "pythia-1.4b", "pythia-2.8b", "qwen2.5-1.5b"]]
    macros = {}

    def macro(key, value):
        macros[key] = str(value)

    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Model & $p_0(1)$ & $p_0(2)$ & $p_0(16)$ & $p_D(16)$ & $K_{50}$ & $K_{50}^{D}$ \\", r"\midrule"]
    for lab, rec in caps.items():
        ks, p = curve(rec, "D0")
        kd, pd = curve(rec, "D256")
        b = 1 / rec["n_obligations"]
        fit = fit_k50(ks, p, b)
        fd = fit_k50(kd, pd, b, K1_recall=p[0])
        overview["models"][lab] = {"D0": fit, "D256": fd, "source": "capacity_fp32/"+lab+"/main.json"}
        def fm(f):
            return "$>64$" if f["censored"] else (r"$\leq 1$" if f["method"] == "below 0.5 at the first K" else f"{f['K50']:.1f}")
        lines.append(f"{nice(lab)} & {p[0]:.3f} & {at(rec,'D0',2):.3f} & {at(rec,'D0',16):.3f} & {at(rec,'D256',16):.3f} & {fm(fit)} & {fm(fd)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_measured.tex"), "w").write("\n".join(lines)+"\n")
    macro("NewNModels", len(caps))
    macro("NewNSmall", len(small))
    macro("NewNPrompts", f"{sum(sum(c['n'] for c in r['cells']) for r in caps.values()):,}".replace(",", r"{,}"))
    macro("NewSingleLo", f"{min(at(r,'D0',1) for r in caps.values()):.3f}")
    macro("NewSingleHi", f"{max(at(r,'D0',1) for r in caps.values()):.3f}")
    macro("NewTwoHi", f"{max(at(r,'D0',2) for r in caps.values()):.3f}")
    macro("NewKLo", f"{min(x['D0']['K50'] for x in overview['models'].values()):.1f}")
    macro("NewSmallSixteenLo", f"{min(at(caps[l],'D0',16) for l in small):.3f}")
    macro("NewSmallSixteenHi", f"{max(at(caps[l],'D0',16) for l in small):.3f}")
    ratios = [v["D256"]["K50"]/v["D0"]["K50"] for v in overview["models"].values()
              if not v["D0"]["censored"] and not v["D256"]["censored"]]
    macro("NewRetained", f"{np.median(ratios):.2f}")
    macro("NewRetainedN", len(ratios))
    # Figure: original query across all models, separated for readability.
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.65), sharey=True, layout="constrained")
    for ax, group, title in zip(axs, [small, [x for x in caps if x not in small]],
                               ["Below 1B parameters", "1B to 2.8B parameters"]):
        for i, lab in enumerate(group):
            ks, p = curve(caps[lab], "D0")
            ax.plot(ks, p, label=nice(lab), lw=0.8, ls=["-", "--", ":", "-."][i % 4])
        ks = np.array(ks)
        ax.plot(ks, 1/ks+(ks-1)/(ks*67), color=GREY, lw=1, ls="--")
        ax.set(xscale="log", xlabel="Number of bindings", ylim=(0,1.03), title=title)
        ax.set_xticks([1,4,16,64], ["1","4","16","64"])
        ax.legend(fontsize=5.8, ncol=2, loc="upper right", columnspacing=0.5, handlelength=1.5)
    axs[0].set_ylabel("Recall")
    save_with_data_hash(fig, os.path.join(out,"fig_p3_measured.pdf")); plt.close(fig)

    flines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
              r"Model & Original & Drop \texttt{is} & $e_j$\texttt{ is:} & Matched list \\", r"\midrule"]
    for lab, rec in frames.items():
        v = {c: at(rec,c,16) for c in rec["conds"]}
        overview["wording"][lab] = v
        flines.append(nice(lab)+" & "+" & ".join(f"{v[c]:.3f}" for c in ["D0","taskcolon","xis","matched"])+r" \\")
    flines += [r"\bottomrule",r"\end{tabular}"]
    open(os.path.join(out,"table_wording.tex"),"w").write("\n".join(flines)+"\n")
    for prefix, lab in [("FourTen","pythia-410m"),("QwenSmall","qwen2.5-0.5b"),("OneSixty","pythia-160m")]:
        for suffix, c in [("Original","D0"),("Colon","taskcolon")]:
            macro("New"+prefix+suffix,f"{at(frames[lab],c,16):.3f}")
    fig, axs = plt.subplots(1,2,figsize=(5.5,2.5),layout="constrained")
    ax=axs[0]
    for i,(c,label,col) in enumerate([("D0","Original query",CLAY),("taskcolon","Drop is",TEAL),
                                      ("xis","Entity is:",GREY),("matched","Matched list","#B8860B")]):
        ks,p=curve(frames["pythia-410m"],c)
        ax.plot(ks,p,marker=["o","s","^","D"][i],ms=2.5,lw=1,color=col,label=label)
    ax.set(xscale="log",xlabel="Number of bindings",ylabel="Recall",ylim=(0,1.02),title="Pythia-410M: same bindings")
    ax.set_xticks([2,8,32,64],["2","8","32","64"]);ax.legend(fontsize=6)
    ax=axs[1]
    labs=list(frames)
    for i,lab in enumerate(labs):
        for off,c,col,mark in [(-.1,"D0",CLAY,"o"),(.1,"taskcolon",TEAL,"s")]:
            p=at(frames[lab],c,16);lo,hi=interval(p,288)
            ax.errorbar(p,i+off,xerr=[[p-lo],[hi-p]],fmt=mark,ms=3,color=col,lw=.7)
    ax.set_yticks(range(len(labs)),[nice(l) for l in labs],fontsize=6.5)
    ax.set(xlabel="Recall at 16 bindings",xlim=(0,1),title="Original (circles), drop is (squares)")
    ax.invert_yaxis()
    save_with_data_hash(fig, os.path.join(out,"fig_p3_wording.pdf"));plt.close(fig)

    hlines=[r"\begin{tabular}{@{}llrrrr@{}}",r"\toprule",
            r"Model & Text & Output & Logit decoder & Probe & Probe $-$ logits (95\% CI) \\",r"\midrule"]
    foil=[]
    for key,r in paired.items():
        d=r["probe_minus_logit"]
        # The paired JSON rounds means to four decimals. Format the native output
        # from the original cells to avoid double rounding between the tables.
        native = at(caps[r["label"]], r["cond"], r["K"])
        assert abs(native-r["output"]) <= 0.000051
        hlines.append(f"{nice(r['label'])} & {'None' if r['cond']=='D0' else 'Distractor'} & {native:.3f} & {r['logit_decoder']:.3f} & {r['probe']:.3f} & {d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] "+r"\\")
        runs=ctl[r["label"]+"|main"]["runs"]
        row=[x for x in runs if x["control"]=="foil" and x["cond"]==r["cond"]][0]
        foil.append([x for x in row["rows"] if x["K"]==16][0]["probe_listed"])
    hlines += [r"\bottomrule",r"\end{tabular}"]
    open(os.path.join(out,"table_heldout.tex"),"w").write("\n".join(hlines)+"\n")
    primary=[r for r in paired.values() if r["cond"]=="D0"]
    macro("NewHeldPone",sum(r["probe_minus_output"]["mean"]>=.10 and r["probe_minus_output"]["lo"]>0 for r in primary))
    macro("NewHeldPtwo",sum(r["probe_minus_logit"]["lo"]>0 for r in primary))
    macro("NewFoilLo",f"{min(foil):.3f}");macro("NewFoilHi",f"{max(foil):.3f}")
    macro("NewGuessSixteen",f"{1/16+15/(16*67):.3f}")
    for prefix,lab in [("HeldOneFour","pythia-1.4b"),("HeldTwoEight","pythia-2.8b"),("HeldQwen","qwen2.5-1.5b")]:
        r=paired[lab+"|D0"]
        for suffix,key in [("Output","output"),("Probe","probe"),("Logits","logit_decoder")]:
            macro("New"+prefix+suffix,f"{r[key]:.3f}")
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.15),layout="constrained",sharex=True,sharey=True)
    for ax,cond,title in zip(axs,["D0","D256"],["No distractor","Distractor text"]):
        labs=["pythia-1.4b","pythia-2.8b","qwen2.5-1.5b"]
        for i,lab in enumerate(labs):
            r=paired[lab+"|"+cond]
            for off,key,col,mark in [(-.10,"probe_minus_output",CLAY,"o"),(.10,"probe_minus_logit",TEAL,"s")]:
                d=r[key];ax.errorbar(d["mean"],i+off,xerr=[[d["mean"]-d["lo"]],[d["hi"]-d["mean"]]],fmt=mark,color=col,ms=3,lw=.8)
        ax.axvline(0,color=GREY,ls="--",lw=.6)
        ax.set_yticks(range(3),[nice(l) for l in labs]);ax.set(title=title,xlabel="Probe advantage (recall)",xlim=(-.02,.52))
        ax.invert_yaxis()
    save_with_data_hash(fig, os.path.join(out,"fig_p3_heldout.pdf"));plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(5.5,3.7),layout="constrained",sharex=True,sharey=True)
    for row,lab in enumerate(["pythia-160m","pythia-410m"]):
        keys=sorted([k for k in ckpt if k.startswith(lab+"|")],key=lambda k:int(k.split("step")[1]))
        for col,cond in enumerate(["D0","D256"]):
            ax=axs[row,col];ss=[];vals=[]
            for key in keys:
                rs=[r for r in ckpt[key]["runs"] if r["cond"]==cond and r["control"] is None]
                rr={r["layer"]:[x for x in r["rows"] if x["K"]==16][0] for r in rs}
                assert all(x["n"]==576 for x in rr.values())
                ss.append(int(key.split("step")[1]));vals.append(rr)
                overview["checkpoint_K16"][key+"|"+cond]=rr
            layers=sorted(vals[0]);ax.plot(ss,[v[layers[0]]["output"] for v in vals],color=CLAY,lw=1.3,label="Output")
            for i,l in enumerate(layers):
                ax.plot(ss,[v[l]["probe"] for v in vals],color=TEAL,lw=.85,ls=[":","--","-","-."][i],label=f"Probe L{l}")
            ax.axhline(1/16+15/(16*67),color=GREY,lw=.6,ls="--")
            ax.set(xscale="log",ylim=(0,.4),title=nice(lab)+(": no distractor" if cond=="D0" else ": distractor"))
            if row==1:ax.set_xlabel("Pretraining step")
            if col==0:ax.set_ylabel("Recall at 16 bindings")
            if col==0:ax.legend(fontsize=5.7,ncol=2)
    save_with_data_hash(fig, os.path.join(out,"fig_p3_checkpoint.pdf"));plt.close(fig)
    last=overview["checkpoint_K16"]["pythia-410m|step96000|D256"]
    macro("NewCkptProbe",f"{last[18]['probe']:.3f}")
    macro("NewCkptMiddle",f"{last[12]['probe']:.3f}")
    macro("NewCkptOutput",f"{last[18]['output']:.3f}")
    macro("NewCkptSmallMax",f"{max(x['probe'] for k,rs in overview['checkpoint_K16'].items() if k.startswith('pythia-160m|') for x in rs.values()):.3f}")
    # Restore the recorded pilot controls discussed before the wording factorial.
    # These are descriptive summaries of existing predictions, not new fits.
    pilots = {}
    for path in sorted(glob.glob(os.path.join(root, "frame_test", "*", "main.json"))):
        rec = read(os.path.relpath(path, root))
        pilots[rec["label"]] = rec
    assert len(pilots) == 7
    pilot_train = {sum(c["n"] for c in rec["cells"] if c["cond"] == "D0" and c["seed"] in (0, 1))
                   for rec in pilots.values()}
    main_train = {sum(c["n"] for c in rec["cells"] if c["cond"] == "D0" and c["seed"] in (0, 1, 2, 3))
                  for rec in caps.values()}
    assert len(pilot_train) == len(main_train) == 1
    macro("NewPilotTrain", f"{next(iter(pilot_train)):,}".replace(",", r"{,}"))
    macro("NewMainTrain", f"{next(iter(main_train)):,}".replace(",", r"{,}"))
    plines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
              r"Model & Original & Exact cue & Two examples \\", r"\midrule"]
    overview["wording_pilot"] = {}
    for lab, rec in pilots.items():
        vals = {c: at(rec, c, 16) for c in ["D0", "verbatim", "shots2"]}
        assert all(sum(x["n"] for x in rec["cells"] if x["cond"] == c and x["K"] == 16) == 288
                   for c in vals)
        overview["wording_pilot"][lab] = vals
        plines.append(nice(lab)+" & "+" & ".join(f"{vals[c]:.3f}" for c in vals)+r" \\")
    plines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_wording_pilot.tex"), "w").write("\n".join(plines)+"\n")
    for prefix, lab in [("PilotSmall", "pythia-160m"), ("PilotFourTen", "pythia-410m")]:
        for suffix, c in [("Original", "D0"), ("Exact", "verbatim"), ("Examples", "shots2")]:
            macro("New"+prefix+suffix, f"{at(pilots[lab], c, 16):.3f}")
    probe_pilot = read("probe_cap/p3_probe_frame.json")["checkpoints"]
    logit_pilot = read("probe_cap/p3_readout_frame.json")["checkpoints"]
    ctl_pilot = read("probe_cap/p3_probe_frame_ctl_L2.json")["checkpoints"]
    qlines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
              r"Model & Query & Output & Probe & Logit dec. & Foil & Shuffled \\", r"\midrule"]
    overview["wording_pilot_probes"] = {}
    for key, rec in probe_pilot.items():
        layer = sorted({r["layer"] for r in rec["runs"] if r["control"] is None})[2]
        for c in ["verbatim", "shots2"]:
            def result(runs, control, chosen_layer=None):
                rs = [r for r in runs if r["cond"] == c and r["control"] == control
                      and (chosen_layer is None or r["layer"] == chosen_layer)]
                assert len(rs) == 1, (key, c, control, chosen_layer)
                row = next(x for x in rs[0]["rows"] if x["K"] == 16)
                assert row["n"] == 288
                return row
            pr = result(rec["runs"], None, layer)
            lg = result(logit_pilot[key]["runs"], "logits")
            foil = result(ctl_pilot[key]["runs"], "foil", layer)
            shuf = result(ctl_pilot[key]["runs"], "shuffled", layer)
            native = at(pilots[rec["label"]], c, 16)
            assert abs(native-pr["output"]) < 1e-6
            record = dict(layer=layer, n=288, output=native, probe=pr["probe"],
                          logit_decoder=lg["probe"], foil_listed=foil["probe_listed"],
                          shuffled_pool=shuf["probe"])
            overview["wording_pilot_probes"][key+"|"+c] = record
            qlines.append(nice(rec["label"])+" & "+("Exact cue" if c=="verbatim" else "Examples")
                          +" & "+" & ".join(f"{record[q]:.3f}" for q in
                          ["output", "probe", "logit_decoder", "foil_listed", "shuffled_pool"])+r" \\")
    qlines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_wording_readouts.tex"), "w").write("\n".join(qlines)+"\n")
    pilot = overview["wording_pilot_probes"]["pythia-160m|main|verbatim"]
    macro("NewPilotSmallProbe", f"{pilot['probe']:.3f}")
    macro("NewPilotSmallLogits", f"{pilot['logit_decoder']:.3f}")
    anatomy = read("anatomy/p3_error_anatomy_13.json")["models"]
    error_rows = []
    for rec in anatomy.values():
        if rec["cond"] != "D0":
            continue
        row = next(r for r in rec["rows"] if r["K"] == 16)
        assert row["n"] == 576 and row["n_wrong"] == row["listed"]+row["unlisted"]
        error_rows.append(dict(label=rec["label"], n=row["n"], n_wrong=row["n_wrong"],
                               listed=row["listed"], share=row["listed"]/row["n_wrong"],
                               uniform_wrong_reference=row["listed_exp"]))
    assert len(error_rows) == 13
    overview["listed_error_K16_D0"] = error_rows
    macro("NewListedErrorLo", f"{min(r['share'] for r in error_rows):.3f}")
    macro("NewListedErrorHi", f"{max(r['share'] for r in error_rows):.3f}")
    macro("NewListedNullLo", f"{min(r['uniform_wrong_reference'] for r in error_rows):.3f}")
    macro("NewListedNullHi", f"{max(r['uniform_wrong_reference'] for r in error_rows):.3f}")
    overview["input_sha256"]=inputs
    overview["generator_sha256"]=hashlib.sha256(open(__file__,"rb").read()).hexdigest()
    overview["_provenance"]="analysis/make_p3_revision_figs.py --measurement-root; no refitted probes"
    open(os.path.join(out,"numbers_measured.tex"),"w").write("% Generated from saved float32 runs by make_p3_revision_figs.py\n"+"".join("\\newcommand{\\"+k+"}{"+v+"}\n" for k,v in macros.items()))
    json.dump(overview,open(os.path.join(out,"p3_measured_revision.json"),"w"),indent=2)
    print("Rendered float32 revision:",len(caps),"models; held-out P1/P2",macros["NewHeldPone"],macros["NewHeldPtwo"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figs_p456"))
    ap.add_argument("--measurement-root", help="saved float32 campaign root; render the new revision only")
    ap.add_argument("--development-only", action="store_true", help="render the supporting development figure only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.development_only:
        fig_development(a.out)
        if FAILS:
            raise SystemExit(f"{FAILS} collisions in the development figure")
        return
    if a.measurement_root:
        measured_revision(a.measurement_root, a.out)
        return
    fig_capacity(a.out)
    fig_interference(a.out)
    fig_development(a.out)
    if FAILS:
        raise SystemExit(f"{FAILS} collisions; fix label positions before using the figures")


if __name__ == "__main__":
    main()
