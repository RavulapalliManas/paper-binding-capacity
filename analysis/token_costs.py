"""Token-denominated versions of the step-denominated headline results, from raw run configs.

WHY. The papers report costs in optimizer steps. Tokens-per-step = batch_size * sequence_length
is recorded in (or derivable from) every raw run record, so the headline economics can be stated
in tokens without guessing anything. This script derives the constants from the raw corpora,
asserts the sequence-length rule against every record that stores ctx, converts the committed
headline numbers, and refits the time law in token units.

SOURCES (all local, read-only):
  devaxis corpus   ../../devaxis-trainium/runs_json/**/*.json   (manifest.task.{K,D}, manifest.bs, ctx)
  farm maint pool  ../../emergence-clocks-2026-08/farm_raw/maint/**/*.json  (K, D, bs, ctx)
  farm dsp pool    ../../emergence-clocks-2026-08/farm_raw/dsp/**/*.json    (argv.{bs,target}, ctx)
  headline numbers paper/results/{extra_analyses,devaxis_analyses,h100_reanalysis,x14_gap_law,d250_verdict}.json

SEQUENCE RULE (identical in both codebases; asserted per record below):
  T = 1 + 2K + D + 1 + 2K + 4   (BOS | K decls | D distractors | RECAP | K decls | query tail)

MEMORY: streams one small JSON at a time; peak well under 100 MB for ~6k files.

Run:  python token_costs.py [--write]     (writes ../results/token_costs.json)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DEVAXIS = os.path.join(HERE, "..", "..", "..", "devaxis-trainium")
FARM = os.path.join(HERE, "..", "..", "..", "emergence-clocks-2026-08", "farm_raw")


def seq_len(K, D):
    return 1 + 2 * K + D + 1 + 2 * K + 4


def load(name):
    return json.load(open(os.path.join(RES, name)))


def scan_devaxis():
    """(K, D, bs) -> count over the SCIENCE records (d_model=256, D=32); smoke/bench runs with
    tiny models or toy tasks are excluded and counted. Every stored ctx asserted against the rule."""
    combos, ctx_checked, excluded = Counter(), 0, 0
    for f in glob.glob(os.path.join(DEVAXIS, "runs_json", "**", "*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        m = d.get("manifest") or {}
        t = m.get("task") or {}
        K, D, bs = t.get("K"), t.get("D"), m.get("bs")
        if K is None or bs is None:
            continue  # fork/branch records without a task manifest inherit the parent config
        if (m.get("model") or {}).get("d") != 256 or D != 32:
            excluded += 1
            continue
        if d.get("ctx") is not None:
            assert d["ctx"] == seq_len(K, D), f"{f}: ctx {d['ctx']} != rule {seq_len(K, D)}"
            ctx_checked += 1
        combos[(K, D, bs)] += 1
    return combos, ctx_checked, excluded


def scan_farm(pool):
    """(K, D, bs, ctx, target) -> count; ctx asserted against the rule where the task is binding."""
    combos, ctx_checked = Counter(), 0
    for f in glob.glob(os.path.join(FARM, pool, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        argv = d.get("argv")
        if not isinstance(argv, dict):
            argv = {}  # some records store argv as the raw command-line string
        bs = d.get("bs", argv.get("bs"))
        target = argv.get("target", "binding")
        K, D, ctx = d.get("K"), d.get("D"), d.get("ctx")
        if bs is None or ctx is None:
            continue
        if target == "binding" and K is not None and D is not None:
            assert ctx == seq_len(K, D), f"{f}: ctx {ctx} != rule {seq_len(K, D)}"
            ctx_checked += 1
        combos[(K, D, bs, ctx, target)] += 1
    return combos, ctx_checked


def fit_loglog(xs, ys):
    a, b = np.polyfit(np.log10(xs), np.log10(ys), 1)
    pred = a * np.log10(xs) + b
    r2 = 1 - np.sum((np.log10(ys) - pred) ** 2) / np.sum((np.log10(ys) - np.mean(np.log10(ys))) ** 2)
    return float(a), float(10 ** b), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    out = {"_provenance": "analysis/token_costs.py over the devaxis and farm raw run records "
                          "(bs, ctx per pool; sequence rule T=1+2K+D+1+2K+4 asserted per record) "
                          "and the committed headline JSONs. Tokens/step = bs * ctx."}

    # ---- constants, derived ----------------------------------------------------------------
    dv, dv_ctx_n, dv_excl = scan_devaxis()
    assert dv, f"no devaxis run records found under {os.path.abspath(DEVAXIS)}"
    out["devaxis_combos"] = [{"K": k, "D": d, "bs": b, "n_runs": n} for (k, d, b), n in sorted(dv.items())]
    out["devaxis_ctx_records_asserted"] = dv_ctx_n
    out["devaxis_smoke_runs_excluded"] = dv_excl
    # dominant bs per K by run count; a lone bench run at another bs must stay a <1% minority
    by_k = {}
    for (k, d, b), n in dv.items():
        by_k.setdefault(k, Counter())[b] += n
    for k, c in by_k.items():
        dom, n_dom = c.most_common(1)[0]
        assert dom == 128, f"devaxis K={k}: dominant bs is {dom}, expected 128"
        assert n_dom / sum(c.values()) > 0.99, f"devaxis K={k}: bs mixture too large: {dict(c)}"

    maint, maint_ctx_n = scan_farm("maint")
    assert maint, f"no maint run records found under {os.path.abspath(FARM)}"
    out["maint_combos"] = [{"K": k, "D": d, "bs": b, "ctx": c, "n_runs": n}
                           for (k, d, b, c, t), n in sorted(maint.items())]
    out["maint_ctx_records_asserted"] = maint_ctx_n
    assert all(b == 64 and d == 128 for (k, d, b, c, t) in maint), "maint bs/D not uniform"

    dsp, dsp_ctx_n = scan_farm("dsp")
    out["dsp_combos"] = [{"target": t, "ctx": c, "bs": b, "n_runs": n}
                         for (k, d, b, c, t), n in sorted(dsp.items(), key=lambda x: -x[1])]
    dsp_primary = [(c, b, n) for (k, d, b, c, t), n in dsp.items() if t == "stateupd"]
    assert len(dsp_primary) == 1, f"dsp primary target not single-config: {dsp_primary}"
    dsp_ctx, dsp_bs, dsp_n = dsp_primary[0]

    def dv_tps(K):   # devaxis tokens per step
        return 128 * seq_len(K, 32)

    def h100_tps(K):  # H100 binding pools tokens per step
        return 64 * seq_len(K, 128)

    dsp_tps = dsp_bs * dsp_ctx
    out["tokens_per_step"] = {
        "devaxis_binding_by_K": {str(k): dv_tps(k) for k in (2, 3, 4, 8)},
        "h100_binding_by_K": {str(k): h100_tps(k) for k in (2, 3, 4, 5, 6)},
        "dsp_primary_target": {"target": "stateupd", "bs": dsp_bs, "ctx": dsp_ctx,
                               "tokens_per_step": dsp_tps, "n_runs": dsp_n,
                               "note": "ctx is the run-level recorded sequence length; the priming "
                                       "phase of each run is recorded at the same ctx"},
    }

    # ---- 1. the time law in tokens ---------------------------------------------------------
    eb = load("extra_analyses.json")["B_emergence_time_vs_K"]["cells"]
    dl = load("devaxis_analyses.json")["emergence_law"]["devaxis"]["cells"]
    h_cells = {int(k): v["median_step"] * h100_tps(int(k)) for k, v in eb.items()}
    d_cells = {int(k): v["median"] * dv_tps(int(k)) for k, v in dl.items()}
    ha, hc, hr2 = fit_loglog(sorted(h_cells), [h_cells[k] for k in sorted(h_cells)])
    da, dc, dr2 = fit_loglog(sorted(d_cells), [d_cells[k] for k in sorted(d_cells)])
    d34 = {k: v for k, v in d_cells.items() if k != 8}
    da_nok8, _, _ = fit_loglog(sorted(d34), [d34[k] for k in sorted(d34)])
    out["time_law_tokens"] = {
        "h100_cells_tokens": {str(k): round(v) for k, v in sorted(h_cells.items())},
        "devaxis_cells_tokens": {str(k): round(v) for k, v in sorted(d_cells.items())},
        "h100_fit": {"exponent": round(ha, 3), "coef_tokens": round(hc), "r2": round(hr2, 3)},
        "devaxis_fit": {"exponent": round(da, 3), "coef_tokens": round(dc), "r2": round(dr2, 3)},
        "devaxis_exponent_no_K8": round(da_nok8, 3),
        "K4_formation_tokens": {"h100": round(h_cells[4]), "devaxis": round(d_cells[4])},
        "caveat": "cross-platform token totals are not directly comparable: the corpora differ in "
                  "distractor length (D=128 vs 32) and batch size (64 vs 128); medians over emerged "
                  "runs only, so all totals are lower bounds",
    }

    # ---- 2. maintenance economics in tokens (devaxis K=4 paired fork) ----------------------
    window_steps = 10_000
    out["maintenance_tokens"] = {
        "devaxis_K4_window_tokens": window_steps * dv_tps(4),
        "trickle_supervised_step_equivalents": 50,
        "trickle_tokens": 50 * dv_tps(4),
        "trickle_fraction_of_window": round(50 / window_steps, 4),
        "note": "the paired-fork deciding signal: ~50 supervised step-equivalents (p=0.005 over a "
                "10,000-step window) = ~0.35M tokens of supervised throughput inside a ~69M-token window",
    }

    # ---- 3. the gap law in tokens (X14: K=2, maint pool) ------------------------------------
    gl = load("x14_gap_law.json")
    out["gap_law_tokens"] = {
        "tokens_per_step_K2": h100_tps(2),
        "gap50_steps": gl["fit"]["gap50"],
        "gap50_tokens": round(gl["fit"]["gap50"] * h100_tps(2)),
        "gap_axis_tokens": {g: round(int(g) * h100_tps(2)) for g in gl["gap_axis"]},
        "note": "the maximum unsupervised gap converted to unsupervised token throughput between "
                "supervised contacts; X14 runs are K=2, D=128, bs=64 (asserted from farm_raw/maint)",
    }

    # ---- 4. priming economics in tokens (DSP primary target) --------------------------------
    hr = load("h100_reanalysis.json")
    dz = load("d250_verdict.json")
    q1 = hr["Q1_warmup"]["strata"]
    out["priming_tokens"] = {
        "dose_tokens": {str(d): round(d * dsp_tps) for d in (250, 2000, 5000, 10000, 20000)},
        "baseline_requirement_tokens": {
            "warmup_free_rederived": round(q1["1"]["median"] * dsp_tps),
            "warmup_500": round(q1["500"]["median"] * dsp_tps),
            "campaign_equalised_5875": round(5875 * dsp_tps),
            "replication_pooled": round(dz["baseline"]["median"] * dsp_tps),
        },
        "warmup_artifact_extra_tokens": round((q1["500"]["median"] - q1["1"]["median"]) * dsp_tps),
        "harm_dose20k_tokens": round(20000 * dsp_tps),
        "dose2000_fraction_of_baseline": round(2000 / 5875, 3),
        "note": "DSP primary target (stateupd): 64 x 37 = 2,368 tokens/step on both phases",
    }

    print(json.dumps(out, indent=1)[:4000])
    if args.write:
        dst = os.path.join(RES, "token_costs.json")
        json.dump(out, open(dst, "w"), indent=1)
        print("\nwrote", dst)


if __name__ == "__main__":
    main()
