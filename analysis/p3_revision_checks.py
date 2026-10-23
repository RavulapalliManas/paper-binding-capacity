"""Measurement checks requested by the 2026-10-01 P3 audit, run before the LP4FM camera-ready.

QUESTIONS (one section each; every number written to results/p3_revision_checks.json):
  R  Does the query survive tokenisation?  The continuous evaluator (neuron_capacity.py) pads to
     --seq and truncates from the right, so an over-long prompt loses its query.  Replays every
     item of every cell with each model's own tokenizer and counts prompts longer than seq.
     Also records how many tokens the nominal D=256 filler block really is, per tokenizer.
  B  How much recall does a strategy get that ignores the queried entity?  Two baselines on the
     exact items: pick a uniformly random value present in the prompt, and pick the most
     frequent present value (ties split).  Compared with each model's measured recall.
  T  Does the recipe coefficient change when the right-censored OLMo-2-1B enters the fit?
     Tobit (right-censored normal) regression of log K50 on log N and a modern-recipe indicator,
     validated first on synthetic data with a planted effect and a planted null.
  N  How does the size exponent change with non-embedding parameter counts?

INPUTS   results/capacity_trn1_summary.json (per-model curves, K50, censoring flags)
         results/capacity_x/<label>_w0.json  (seq and n_trials actually used)
         HuggingFace tokenizers for the 12 models (section R and B only).
OUTPUT   results/p3_revision_checks.json, and with --tex, numbers_revision.tex next to the paper.

PEAK MEMORY  12 tokenizers loaded one at a time; at most 15 K x 2 D x 6 seeds x 96 prompts
             = 17,280 short strings per model.  Under 1 GB.

Run:  python p3_revision_checks.py [--res ../results] [--tex PATH] [--skip-tokenizers]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter

import numpy as np
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Task definition: imported from the evaluator (lazily, since it imports torch) so the replayed
# items cannot drift from what was measured.
TASK_SOURCE = "imported from analysis/neuron_capacity.py"


def task():
    import neuron_capacity as nc
    return nc

# Hugging Face model ids, as recorded in capacity_trn1_summary.json.
MODELS = {
    "olmo2-1b-x": "allenai/OLMo-2-0425-1B", "opt-1.3b-x": "facebook/opt-1.3b",
    "opt-125m-x": "facebook/opt-125m", "opt-350m-x": "facebook/opt-350m",
    "pythia-1.4b-x": "EleutherAI/pythia-1.4b", "pythia-160m-x": "EleutherAI/pythia-160m",
    "pythia-1b-x": "EleutherAI/pythia-1b", "pythia-2.8b-x": "EleutherAI/pythia-2.8b",
    "pythia-410m-x": "EleutherAI/pythia-410m", "qwen2.5-0.5b-x": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b-x": "Qwen/Qwen2.5-1.5B", "smollm2-360m-x": "HuggingFaceTB/SmolLM2-360M",
}

# Total parameters (millions), from each model's HuggingFace config; identical to the map in
# analysis/paper_addenda.py, which the AUC and interference analyses already use.
PARAMS_TOTAL_M = {
    "olmo2-1b-x": 1480, "opt-1.3b-x": 1316, "opt-125m-x": 125, "opt-350m-x": 331,
    "pythia-1.4b-x": 1415, "pythia-160m-x": 162, "pythia-1b-x": 1012,
    "pythia-2.8b-x": 2775, "pythia-410m-x": 405, "qwen2.5-0.5b-x": 494,
    "qwen2.5-1.5b-x": 1544, "smollm2-360m-x": 362,
}
# Nominal sizes from analysis/agg_capacity.py, which produced the published alpha = 0.820.
PARAMS_NOMINAL_M = {
    "olmo2-1b-x": 1000, "opt-1.3b-x": 1300, "opt-125m-x": 125, "opt-350m-x": 350,
    "pythia-1.4b-x": 1400, "pythia-160m-x": 160, "pythia-1b-x": 1000,
    "pythia-2.8b-x": 2800, "pythia-410m-x": 410, "qwen2.5-0.5b-x": 500,
    "qwen2.5-1.5b-x": 1500, "smollm2-360m-x": 360,
}
# Embedding parameters from config.json (vocab_size, hidden_size or word_embed_proj_dim,
# tie_word_embeddings, max_position_embeddings), read 2026-10-23 from
# https://huggingface.co/<model id>/resolve/main/config.json.
#   untied (Pythia, OLMo-2): input + output embedding = 2 * vocab * hidden
#   tied (Qwen2.5, SmolLM2): vocab * hidden
#   OPT (tied, learned positions offset by 2): vocab * proj_dim + (max_pos + 2) * hidden
EMBED_M = {
    "olmo2-1b-x": 2 * 100352 * 2048 / 1e6,
    "opt-1.3b-x": (50272 * 2048 + 2050 * 2048) / 1e6,
    "opt-125m-x": (50272 * 768 + 2050 * 768) / 1e6,
    "opt-350m-x": (50272 * 512 + 2050 * 1024) / 1e6,
    "pythia-1.4b-x": 2 * 50304 * 2048 / 1e6, "pythia-160m-x": 2 * 50304 * 768 / 1e6,
    "pythia-1b-x": 2 * 50304 * 2048 / 1e6, "pythia-2.8b-x": 2 * 50304 * 2560 / 1e6,
    "pythia-410m-x": 2 * 50304 * 1024 / 1e6,
    "qwen2.5-0.5b-x": 151936 * 896 / 1e6, "qwen2.5-1.5b-x": 151936 * 1536 / 1e6,
    "smollm2-360m-x": 49152 * 960 / 1e6,
}
MAX_POSITIONS = {  # config.json max_position_embeddings, same source and date
    "olmo2-1b-x": 4096, "opt-1.3b-x": 2048, "opt-125m-x": 2048, "opt-350m-x": 2048,
    "pythia-1.4b-x": 2048, "pythia-160m-x": 2048, "pythia-1b-x": 2048, "pythia-2.8b-x": 2048,
    "pythia-410m-x": 2048, "qwen2.5-0.5b-x": 32768, "qwen2.5-1.5b-x": 131072,
    "smollm2-360m-x": 8192,
}

N_BOOT = 20000          # OLS bootstraps
N_BOOT_TOBIT = 5000     # each draw is two Nelder-Mead fits
SEED = 0


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# ------------------------------------------------------------------ items, exactly as measured
def items(ents, obls, K, D, s, n_trials):
    rng = random.Random((7000 + s) * 100003 + K * 31 + D)   # neuron_capacity.py seeding
    return [task().make_prompt(rng, ents, obls, K, D) for _ in range(n_trials)]


def values_in(prompt, K):
    decl = prompt.split("\n")[:K]
    return [line.split(": ", 1)[1] for line in decl]


# ------------------------------------------------------------------ R + B
def replay_and_baselines(summary, xdir, use_tokenizers):
    from transformers import AutoTokenizer
    nc = task()
    K_GRID, D_GRID = nc.K_GRID, nc.D_GRID
    out_r, out_b = {}, {}
    for label, mid in MODELS.items():
        w0 = json.load(open(os.path.join(xdir, f"{label}_w0.json")))
        seq, n_trials, seeds = w0["seq"], w0["n_trials"], sorted({c["seed"] for c in w0["cells"]})
        tok = AutoTokenizer.from_pretrained(mid)
        ents = [w for w, _ in nc.single_token(tok, nc.ENTITY_POOL)]
        obls = [w for w, _ in nc.single_token(tok, nc.OBLIGATION_POOL)]
        assert len(obls) == w0["n_obligations"], (label, len(obls), w0["n_obligations"])
        lens, over, filler = [], [], []
        base = {}
        len_d0 = {}
        for K in K_GRID:
            if K > min(len(ents), len(obls)):
                continue
            for D in D_GRID:
                rand_hits, modal_hits, n = 0.0, 0.0, 0
                for s in seeds:
                    its = items(ents, obls, K, D, s, n_trials)
                    for prompt, gold in its:
                        ids = tok(prompt, add_special_tokens=True)["input_ids"]
                        lens.append(len(ids))
                        if D == 0:
                            len_d0.setdefault(K, []).append(len(ids))
                        if len(ids) > seq:
                            over.append((K, D, s))
                        if D > 0 and K == 1 and s == seeds[0]:
                            body, query = prompt.rsplit("\n", 1)
                            decl, fill = body.split("\n", 1)
                            filler.append(len(tok("\n" + fill, add_special_tokens=False)["input_ids"]))
                        vals = values_in(prompt, K)
                        c = Counter(vals)
                        rand_hits += c[gold] / K
                        top = max(c.values())
                        modal = [v for v, k in c.items() if k == top]
                        modal_hits += (1.0 / len(modal)) if gold in modal else 0.0
                        n += 1
                base[(K, D)] = (rand_hits / n, modal_hits / n, n)
        kk = sorted(len_d0)
        tpb = float(np.polyfit(kk, [np.mean(len_d0[k]) for k in kk], 1)[0])
        out_r[label] = {"seq": seq, "n_prompts": len(lens), "max_len": int(max(lens)),
                        "tokens_per_binding_D0": round(tpb, 2),
                        "n_over_seq": len(over), "cells_over_seq": sorted(set(over)),
                        "filler_tokens_D256": {"min": int(min(filler)), "max": int(max(filler)),
                                               "mean": round(float(np.mean(filler)), 1)}}
        meas = summary_models(summary)[label]["by_D"]
        rows = {}
        for D in D_GRID:
            m = meas[str(D)]
            rec = dict(zip([int(k) for k in m["K_grid"]], m["recall"]))
            rows[str(D)] = [{"K": K, "recall": round(rec[K], 4),
                             "present_random": round(base[(K, D)][0], 4),
                             "present_modal": round(base[(K, D)][1], 4)}
                            for K in K_GRID if (K, D) in base and K in rec]
        out_b[label] = rows
        print(f"[R/B] {label}: max_len {max(lens)} / seq {seq}, over {len(over)}, "
              f"filler {out_r[label]['filler_tokens_D256']}", flush=True)
        del tok
    return out_r, out_b


def summary_models(summary):
    m = summary["models"]
    return m if isinstance(m, dict) else {x["label"]: x for x in m}


# ------------------------------------------------------------------ T: Tobit
def tobit_nll(theta, X, y, cens):
    beta, log_s = theta[:-1], theta[-1]
    s = math.exp(log_s)
    mu = X @ beta
    z = (y - mu) / s
    ll_unc = stats.norm.logpdf(z[~cens]) - log_s
    ll_cen = stats.norm.logsf(z[cens])          # P(latent > censoring point)
    return -(ll_unc.sum() + ll_cen.sum())


def tobit_fit(X, y, cens):
    unc = ~cens
    b0, *_ = np.linalg.lstsq(X[unc], y[unc], rcond=None)
    r = y[unc] - X[unc] @ b0
    s0 = max(float(np.sqrt(np.mean(r ** 2))), 1e-2)
    th0 = np.concatenate([b0, [math.log(s0)]])
    res = optimize.minimize(tobit_nll, th0, args=(X, y, cens), method="Nelder-Mead",
                            options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
    return res.x, res.fun, res.success


def design(rows, params, with_recipe=True):
    x = np.array([math.log(params[r["label"]]) for r in rows])
    cols = [np.ones(len(rows)), x]
    if with_recipe:
        cols.append(np.array([1.0 if r["recipe"] == "modern" else 0.0 for r in rows]))
    return np.column_stack(cols)


def selftest_tobit():
    """Planted effect and planted null, with censoring at the top, recovered by the fitter."""
    rng = np.random.default_rng(1)
    out = {}
    for name, c_true in (("planted_effect", 1.0), ("planted_null", 0.0)):
        est = []
        for _ in range(200):
            n = 60
            x = rng.uniform(np.log(100), np.log(3000), n)
            mod = rng.integers(0, 2, n).astype(float)
            y = -3.0 + 0.8 * x + c_true * mod + rng.normal(0, 0.3, n)
            cap = math.log(64)
            cens = y > cap
            y = np.where(cens, cap, y)
            X = np.column_stack([np.ones(n), x, mod])
            th, _, ok = tobit_fit(X, y, cens)
            est.append(th[2])
        out[name] = {"true": c_true, "mean_est": round(float(np.mean(est)), 3),
                     "sd_est": round(float(np.std(est)), 3)}
    passed = (abs(out["planted_effect"]["mean_est"] - 1.0) < 0.1
              and abs(out["planted_null"]["mean_est"]) < 0.1)
    out["passed"] = bool(passed)
    return out


def regression(summary):
    M = summary_models(summary)
    rows = []
    for lab, m in M.items():
        d0 = m["by_D"]["0"]
        rows.append({"label": lab, "family": m["family"], "recipe": m["recipe"],
                     "K50": d0["K50"], "censored": bool(d0["censored"])})
    y = np.array([math.log(r["K50"]) for r in rows])
    cens = np.array([r["censored"] for r in rows])
    out = {"n_models": len(rows), "n_censored": int(cens.sum()),
           "censored_labels": [r["label"] for r in rows if r["censored"]],
           "censoring_point_log": "log K50 recorded at the grid maximum (64)"}

    # Cross-check: OLS without the censored model reproduces capacity_trn1_summary.json.
    unc = ~cens
    for tag, P in (("nominal", PARAMS_NOMINAL_M), ("config_total", PARAMS_TOTAL_M)):
        X = design(rows, P)
        b, *_ = np.linalg.lstsq(X[unc], y[unc], rcond=None)
        out[f"ols_uncensored_{tag}"] = {"log_params_coef": round(float(b[1]), 4),
                                        "recipe_coef": round(float(b[2]), 4), "n_used": int(unc.sum())}

    def fit_all(P, idx, with_recipe=True):
        X = design([rows[i] for i in idx], P, with_recipe)
        th, nll, ok = tobit_fit(X, y[idx], cens[idx])
        return th, nll, ok

    idx_all = np.arange(len(rows))
    for tag, P in (("config_total", PARAMS_TOTAL_M),):
        th_full, nll_full, ok1 = fit_all(P, idx_all, True)
        th_red, nll_red, ok2 = fit_all(P, idx_all, False)
        lr = 2 * (nll_red - nll_full)
        fam = sorted({r["family"] for r in rows})
        by_fam = {f: [i for i, r in enumerate(rows) if r["family"] == f] for f in fam}
        rng = np.random.default_rng(SEED)
        boot_alpha, boot_recipe, boot_alpha_red = [], [], []
        n_unidentified = 0
        for _ in range(N_BOOT_TOBIT):
            pick = rng.choice(fam, size=len(fam), replace=True)
            idx = np.array([i for f in pick for i in by_fam[f]])
            if len({PARAMS_TOTAL_M[rows[i]["label"]] for i in idx}) < 3 or (~cens[idx]).sum() < 4:
                continue
            thr, _, okr = fit_all(P, idx, False)
            if okr:
                boot_alpha_red.append(thr[1])
            # The recipe coefficient is identified only if each recipe has an uncensored model;
            # a draw whose only modern model is censored sends it to +infinity.  Count, skip.
            unc_rec = {rows[i]["recipe"] for i in idx if not cens[i]}
            if unc_rec != {"old", "modern"}:
                n_unidentified += 1
                continue
            th, _, ok = fit_all(P, idx, True)
            if ok:
                boot_alpha.append(th[1]); boot_recipe.append(th[2])
        lofo = {}
        for f in fam:
            idx = np.array([i for i in idx_all if rows[i]["family"] != f])
            th, _, ok = fit_all(P, idx, True)
            lofo[f"drop_{f}"] = {"log_params_coef": round(float(th[1]), 3),
                                 "recipe_coef": round(float(th[2]), 3)}
        q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
        out[f"tobit_{tag}"] = {
            "log_params_coef": round(float(th_full[1]), 4),
            "recipe_coef": round(float(th_full[2]), 4),
            "recipe_ratio_exp": round(float(math.exp(th_full[2])), 3),
            "sigma": round(float(math.exp(th_full[3])), 4),
            "converged": bool(ok1 and ok2),
            "lr_test_recipe": {"stat": round(float(lr), 3),
                               "p_chi2_1df": round(float(stats.chi2.sf(lr, 1)), 4)},
            "boot_ci95_family_clustered": {"log_params": q(boot_alpha), "recipe": q(boot_recipe)},
            "n_boot_valid": len(boot_recipe), "n_boot_draws": N_BOOT_TOBIT,
            "n_boot_recipe_unidentified": n_unidentified, "n_families": len(fam),
            "leave_one_family_out": lofo,
            "scale_only": {"log_params_coef": round(float(th_red[1]), 4),
                           "sigma": round(float(math.exp(th_red[2])), 4),
                           "boot_ci95_family_clustered": q(boot_alpha_red),
                           "n_boot_valid": len(boot_alpha_red)},
        }
    return out, rows


# ------------------------------------------------------------------ N: non-embedding exponent
def nonembedding(rows):
    y = np.array([math.log(r["K50"]) for r in rows])
    cens = np.array([r["censored"] for r in rows])
    unc = ~cens
    out = {}
    for tag, P in (("config_total", PARAMS_TOTAL_M),
                   ("non_embedding", {k: PARAMS_TOTAL_M[k] - EMBED_M[k] for k in PARAMS_TOTAL_M})):
        x = np.array([math.log(P[r["label"]]) for r in rows])
        a, b = np.polyfit(x[unc], y[unc], 1)
        pred = a * x[unc] + b
        r2 = 1 - np.sum((y[unc] - pred) ** 2) / np.sum((y[unc] - y[unc].mean()) ** 2)
        fam = sorted({r["family"] for i, r in enumerate(rows) if unc[i]})
        by_fam = {f: [i for i, r in enumerate(rows) if unc[i] and r["family"] == f] for f in fam}
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(fam, size=len(fam), replace=True)
            idx = [i for f in pick for i in by_fam[f]]
            if len({P[rows[i]["label"]] for i in idx}) < 3:
                continue
            boots.append(np.polyfit(x[idx], y[idx], 1)[0])
        ui = [i for i in range(len(rows)) if unc[i]]
        lofo = {f: round(float(np.polyfit(x[[i for i in ui if rows[i]["family"] != f]],
                                          y[[i for i in ui if rows[i]["family"] != f]], 1)[0]), 3)
                for f in fam}
        loo = {rows[j]["label"]: round(float(np.polyfit(x[[i for i in ui if i != j]],
                                                        y[[i for i in ui if i != j]], 1)[0]), 3)
               for j in ui}
        out[tag] = {"alpha": round(float(a), 4), "intercept": round(float(b), 4),
                    "r2": round(float(r2), 4), "n": int(unc.sum()),
                    "ci95_family_clustered": [round(float(np.percentile(boots, 2.5)), 3),
                                              round(float(np.percentile(boots, 97.5)), 3)],
                    "n_boot_valid": len(boots),
                    "leave_one_family_out": lofo,
                    "leave_one_model_out_max_abs_delta": round(float(max(abs(v - a) for v in loo.values())), 3)}
    out["embedding_share"] = {k: round(EMBED_M[k] / PARAMS_TOTAL_M[k], 3) for k in PARAMS_TOTAL_M}
    return out


# ------------------------------------------------------------------ S: instrument agreement
def instrument_agreement(summary, figdata_path):
    """Threshold k* (legacy 30-model table) against K50 (continuous curves) on shared models."""
    geom = json.load(open(figdata_path))["geom"]
    kstar = {g["model"]: g["kstar"] for g in geom}
    M = summary_models(summary)
    pairs = []
    for lab, mid in MODELS.items():
        if mid in kstar:
            d0 = M[lab]["by_D"]["0"]
            pairs.append({"label": lab, "kstar": kstar[mid], "K50": round(d0["K50"], 3),
                          "K50_censored": bool(d0["censored"])})
    rho, p = stats.spearmanr([q["kstar"] for q in pairs], [q["K50"] for q in pairs])
    return {"n_shared": len(pairs), "pairs": pairs, "spearman_rho": round(float(rho), 3),
            "spearman_p": round(float(p), 4),
            "note": "k* legacy table: generator unrecovered, 25-value pool; K50: 67-71 value pools"}


# ------------------------------------------------------------------ Q: stricter requirements
def usable_loads(summary, thresholds=(0.9, 0.8)):
    """Largest grid load K such that measured raw recall >= pi at every load from 1 to K.

    Direct grid reading, no curve fit: the feasible set of a task that needs recall >= pi.
    Also ranks models by the retained-performance criterion q in {0.5, 0.9} using the first
    grid crossing of r(K) < q, to show whether the midpoint ranking survives a stricter q.
    """
    M = summary_models(summary)
    out = {}
    for lab in MODELS:
        row = {}
        for D in ("0", "256"):
            m = M[lab]["by_D"][D]
            Ks = [int(k) for k in m["K_grid"]]
            rec = m["recall"]
            for pi in thresholds:
                kmax = 0
                for K, p in zip(Ks, rec):
                    if p >= pi:
                        kmax = K
                    else:
                        break
                row[f"D{D}_raw_ge_{pi}"] = kmax
            # Curves whose single-binding recall is too close to chance carry no r (agg_capacity
            # marks them excluded); the own-reference crossing is undefined for them.
            if "r" not in m:
                for q in (0.5, 0.9):
                    row[f"D{D}_first_K_r_below_{q}"] = "excluded"
                continue
            for q in (0.5, 0.9):
                first_below = next((K for K, x in zip(Ks, m["r"]) if x < q), None)
                row[f"D{D}_first_K_r_below_{q}"] = first_below if first_below is not None else f">{Ks[-1]}"
        out[lab] = row
    return out


# ------------------------------------------------------------------ F: floors, size models, ceiling-only test
def floors_and_tests(summary, xdir):
    """Three checks raised by a blind review (2026-10-23).

    1. The entity-blind floor of K50.  The in-prompt guess has recall 1/K + (K-1)/(K|V|) and
       single-binding recall 1, so its normalised curve is exactly r(K) = 1/K and its exact
       midpoint crossing is K = 2.  Here: its K50 under the paper's own estimator (fit_k50).
    2. Size models.  The power law over the 11 uncensored models, against fits within the two
       size groups and a two-level step (group means of log K50), and against capacity above
       the entity-blind floor.
    3. The ceiling-only account of interference, tested on held-out seeds.  Seeds 0-2 estimate
       the clean profile r_0(K) and the distractor single-binding recall C_D; the prediction
       p_D(K) = b + (C_D - b) r_0(K) is scored on seeds 3-5 (and the reverse).  Noise reference:
       the split-half difference of the measured distractor curve itself.
    """
    from capacity_curve import fit_k50
    M = summary_models(summary)
    out = {}
    # 1 -- floor
    grid = [int(k) for k in M["pythia-1b-x"]["by_D"]["0"]["K_grid"]]
    fl = {}
    for V in sorted({round(1 / M[k]["chance"]) for k in MODELS}):
        rec = [1 / K + (K - 1) / (K * V) for K in grid]
        f = fit_k50(grid, rec, 1 / V, K1_recall=1.0)
        fl[str(V)] = {"K50": round(float(f["K50"]), 3), "censored": bool(f["censored"])}
    out["entity_blind_floor"] = {"by_pool_size": fl, "exact_midpoint_crossing": 2.0,
                                 "note": "normalised in-prompt guess is exactly 1/K"}
    floor = float(np.mean([v["K50"] for v in fl.values()]))
    # 2 -- size models
    rows = [(k, PARAMS_TOTAL_M[k], M[k]["by_D"]["0"]["K50"]) for k in MODELS
            if not M[k]["by_D"]["0"]["censored"]]

    def ols(pts):
        x = np.log([p[1] for p in pts]); y = np.log([p[2] for p in pts])
        a, b = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (a * x + b)) ** 2) / np.sum((y - y.mean()) ** 2)
        return {"slope": round(float(a), 3), "r2": round(float(r2), 3), "n": len(pts)}
    small = [p for p in rows if p[1] < 1000]
    big = [p for p in rows if p[1] >= 1000]
    y = np.log([p[2] for p in rows])
    grp = np.array([p[1] >= 1000 for p in rows])
    pred = np.where(grp, y[grp].mean(), y[~grp].mean())
    step_r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    above = [(k, n, K50 - floor) for k, n, K50 in rows if K50 - floor > 0]
    out["size_models"] = {"power_law_all_uncensored": ols(rows), "within_below_1B": ols(small),
                          "within_1B_and_up": ols(big),
                          "two_level_step": {"r2": round(float(step_r2), 3), "n": len(rows),
                                             "split": "N < 1000M vs N >= 1000M"},
                          "power_law_above_floor": {**ols(above), "floor_K50": round(floor, 3),
                                                    "n_dropped_at_or_below_floor": len(rows) - len(above)}}
    # 3 -- ceiling-only account, held-out seeds
    res = {}
    for k in MODELS:
        w0 = json.load(open(os.path.join(xdir, f"{k}_w0.json")))
        b = w0["chance"]
        cell = {}
        for c in w0["cells"]:
            cell.setdefault((c["K"], c["D"]), {})[c["seed"]] = c["recall"]
        Ks = sorted({K for K, _ in cell})
        seeds = sorted({s for v in cell.values() for s in v})
        half = (seeds[: len(seeds) // 2], seeds[len(seeds) // 2:])

        def mean(K, D, ss):
            return float(np.mean([cell[(K, D)][s] for s in ss]))
        resid, noise, signed = [], [], []
        for est, ev in (half, half[::-1]):
            C0, CD = mean(1, 0, est), mean(1, 256, est)
            if C0 - b <= 0:
                continue
            for K in Ks:
                if K < 2:
                    continue
                r0 = (mean(K, 0, est) - b) / (C0 - b)
                pred_ = b + (CD - b) * r0
                meas = mean(K, 256, ev)
                resid.append(abs(meas - pred_)); signed.append(meas - pred_)
                noise.append(abs(mean(K, 256, est) - meas))
        res[k] = {"mean_abs_residual": round(float(np.mean(resid)), 4),
                  "mean_signed_residual": round(float(np.mean(signed)), 4),
                  "mean_abs_split_half": round(float(np.mean(noise)), 4),
                  "ratio": round(float(np.mean(resid) / np.mean(noise)), 2)}
    ratios = [v["ratio"] for v in res.values()]
    out["ceiling_only_test"] = {"per_model": res, "median_ratio": round(float(np.median(ratios)), 2),
                                "n_models": len(res), "n_ratio_above_2": int(sum(r > 2 for r in ratios)),
                                "n_signed_positive": int(sum(v["mean_signed_residual"] > 0 for v in res.values())),
                                "status": "EXPLORATORY: tolerance (2x split-half noise) set after the "
                                          "summary statistics of Section 4 were seen"}
    return out


# ------------------------------------------------------------------ G: calibration and disclosure (number audit, 2026-10-23)
FIRST_ROUND_ONLY = ["opt-2.7b", "phi-1.5", "qwen2.5-3b", "smollm2-135m", "smollm2-1.7b"]
FIRST_ROUND_EXCLUDED = ["gpt2", "gpt2-medium", "gpt2-large", "pythia-70m"]


def calibration(summary, xdir, firstdir, n_sim=2000):
    """Null calibration for the two tests in Sections 3-4, plus the disclosures they need.

    max_margin_null   A model that returns the in-prompt guess, scored on 576 trials per cell
                      like the real ones: the distribution of its largest margin over the guess.
    ceiling_null      For each model, data simulated under the ceiling-only account (binomial
                      per seed-cell, 96 trials) and scored with the same split-seed statistic;
                      the observed ratio's upper-tail p-value against that null.
    feasible_sets     {K : p(K) >= pi} at D = 0 and D = 256, read as a set (no prefix rule).
    recipe_ols_ci     The recipe coefficient with the censored model dropped, config-total N,
                      family-clustered bootstrap.
    first_round       Recall at K = 1, 2, 8, 24 for every model of the first round.
    """
    rng = np.random.default_rng(SEED)
    M = summary_models(summary)
    out = {}
    # max-margin null: guesser on |V| values, values drawn with replacement, 576 trials per cell
    grid = [int(k) for k in M["pythia-1b-x"]["by_D"]["0"]["K_grid"] if int(k) >= 2]
    V = 69
    sims = []
    for _ in range(n_sim):
        best = -1.0
        for K in grid:
            n = 576
            vals = rng.integers(0, V, size=(n, K))
            q = rng.integers(0, K, size=n)            # queried entity
            g = rng.integers(0, K, size=n)            # guessed listed entity
            hit = (vals[np.arange(n), q] == vals[np.arange(n), g]).mean()
            best = max(best, hit - (1 / K + (K - 1) / (K * V)))
        sims.append(best)
    out["max_margin_null"] = {"n_sim": n_sim, "pool": V, "p95": round(float(np.percentile(sims, 95)), 3),
                              "p99": round(float(np.percentile(sims, 99)), 3)}
    # ceiling-only null, per model
    cal = {}
    for k in MODELS:
        w0 = json.load(open(os.path.join(xdir, f"{k}_w0.json")))
        b, ntr = w0["chance"], w0["n_trials"]
        cell = {}
        for c in w0["cells"]:
            cell.setdefault((c["K"], c["D"]), {})[c["seed"]] = c["recall"]
        Ks = sorted({K for K, _ in cell})
        seeds = sorted({s for v in cell.values() for s in v})

        def stat(get):
            half = (seeds[: len(seeds) // 2], seeds[len(seeds) // 2:])
            res, noi = [], []
            for est, ev in (half, half[::-1]):
                m_ = lambda K, D, ss: float(np.mean([get(K, D, s) for s in ss]))
                C0, CD = m_(1, 0, est), m_(1, 256, est)
                if C0 - b <= 0:
                    continue
                for K in Ks:
                    if K < 2:
                        continue
                    pr = b + (CD - b) * (m_(K, 0, est) - b) / (C0 - b)
                    me = m_(K, 256, ev)
                    res.append(abs(me - pr)); noi.append(abs(m_(K, 256, est) - me))
            return float(np.mean(res) / np.mean(noi))
        obs = stat(lambda K, D, s: cell[(K, D)][s])
        p0 = {K: float(np.mean(list(cell[(K, 0)].values()))) for K in Ks}
        C0, CD = p0[1], float(np.mean(list(cell[(1, 256)].values())))
        pD = {K: (CD if K == 1 else b + (CD - b) * (p0[K] - b) / (C0 - b)) for K in Ks}
        null = []
        for _ in range(n_sim // 4):
            sim = {(K, D, s): rng.binomial(ntr, min(max((p0 if D == 0 else pD)[K], 0.0), 1.0)) / ntr
                   for K in Ks for D in (0, 256) for s in seeds}
            null.append(stat(lambda K, D, s: sim[(K, D, s)]))
        cal[k] = {"observed_ratio": round(obs, 2), "null_p95": round(float(np.percentile(null, 95)), 2),
                  "p_upper": round(float((np.sum(np.array(null) >= obs) + 1) / (len(null) + 1)), 4)}
    out["ceiling_null"] = {"per_model": cal, "n_sim_per_model": n_sim // 4,
                           "n_reject_05": int(sum(v["p_upper"] < 0.05 for v in cal.values()))}
    # feasible sets
    fs = {}
    for k in MODELS:
        row = {}
        for D in ("0", "256"):
            m = M[k]["by_D"][D]
            for pi in (0.9, 0.8):
                row[f"D{D}_{pi}"] = [int(K) for K, p in zip(m["K_grid"], m["recall"]) if p >= pi]
        fs[k] = row
    out["feasible_sets"] = fs
    # recipe coefficient, censored model dropped, config-total N, family-clustered CI
    rows = [(k, M[k]["family"], M[k]["recipe"], PARAMS_TOTAL_M[k], M[k]["by_D"]["0"]["K50"])
            for k in MODELS if not M[k]["by_D"]["0"]["censored"]]
    def fit(rs):
        X = np.column_stack([np.ones(len(rs)), np.log([r[3] for r in rs]),
                             [1.0 if r[2] == "modern" else 0.0 for r in rs]])
        return np.linalg.lstsq(X, np.log([r[4] for r in rs]), rcond=None)[0]
    coef = fit(rows)
    fam = sorted({r[1] for r in rows}); byf = {f: [r for r in rows if r[1] == f] for f in fam}
    boots = []
    for _ in range(N_BOOT):
        rs = [r for f in rng.choice(fam, size=len(fam)) for r in byf[f]]
        if len({r[2] for r in rs}) < 2 or len({r[3] for r in rs}) < 3:
            continue
        boots.append(fit(rs)[2])
    out["recipe_ols_ci"] = {"coef": round(float(coef[2]), 3), "n": len(rows),
                            "ci95": [round(float(np.percentile(boots, 2.5)), 3),
                                     round(float(np.percentile(boots, 97.5)), 3)], "n_boot_valid": len(boots)}
    # first round
    fr = {}
    import glob as _g
    for f in sorted(_g.glob(os.path.join(firstdir, "*_w*.json"))):
        d = json.load(open(f))
        e = fr.setdefault(d["label"], {"cells": [], "meta": {k_: d.get(k_) for k_ in
                                                              ("model", "seq", "n_trials", "chance", "n_obligations")}})
        e["cells"] += d["cells"]
    first = {}
    for lab, e in fr.items():
        c = e["cells"]
        rec = {K: float(np.mean([x["recall"] for x in c if x["K"] == K and x["D"] == 0]))
               for K in sorted({x["K"] for x in c})}
        first[lab] = {**e["meta"], "n_seeds": len({x["seed"] for x in c}),
                      "recall_D0": {str(K): round(rec[K], 3) for K in (1, 2, 8, 24) if K in rec}}
    out["first_round"] = {"models": first, "n_models": len(first),
                          "excluded_below_0.5": FIRST_ROUND_EXCLUDED,
                          "not_remeasured": FIRST_ROUND_ONLY}
    return out


# ------------------------------------------------------------------ TeX emitters
PRETTY = {
    "olmo2-1b-x": "OLMo-2-1B", "opt-1.3b-x": "OPT-1.3B", "opt-125m-x": "OPT-125M",
    "opt-350m-x": "OPT-350M", "pythia-1.4b-x": "Pythia-1.4B", "pythia-160m-x": "Pythia-160M",
    "pythia-1b-x": "Pythia-1B", "pythia-2.8b-x": "Pythia-2.8B", "pythia-410m-x": "Pythia-410M",
    "qwen2.5-0.5b-x": "Qwen2.5-0.5B", "qwen2.5-1.5b-x": "Qwen2.5-1.5B",
    "smollm2-360m-x": "SmolLM2-360M",
}


def f2(x):
    s = f"{x:.2f}"
    return "0.00" if s == "-0.00" else s


def up2(x):
    """Upper bound printed at two decimals: round up, so 'at most' stays true."""
    return f"{math.ceil(round(x * 100, 9)) / 100:.2f}"


def fp(x):
    """A p-value with one significant digit below 0.01."""
    return f"{x:.2f}" if x >= 0.01 else f"{x:.1g}"


def emit_tex(res_dir, tex_dir):
    """numbers_revision.tex and table_capacity.tex from results/p3_revision_checks.json only."""
    R = json.load(open(os.path.join(res_dir, "p3_revision_checks.json")))
    S = summary_models(json.load(open(os.path.join(res_dir, "capacity_trn1_summary.json"))))
    se, rg = R["size_exponent"], R["regression"]
    ct, ne, tb = se["config_total"], se["non_embedding"], rg["tobit_config_total"]
    lofo = list(ct["leave_one_family_out"].values())
    share = list(se["embedding_share"].values())
    sub = sorted(S[k]["by_D"]["0"]["K50"] for k in MODELS if PARAMS_TOTAL_M[k] < 1000)
    mid = [k for k in MODELS if 1300 <= PARAMS_TOTAL_M[k] <= 1600]
    mid_unc = sorted(S[k]["by_D"]["0"]["K50"] for k in mid if not S[k]["by_D"]["0"]["censored"])
    rep = R.get("replay", {})
    m = [
        ("RevAlpha", f2(ct["alpha"]), "size_exponent.config_total -- 11 uncensored, config-total N"),
        ("RevAlphaRsq", f2(ct["r2"]), "size_exponent.config_total"),
        ("RevAlphaCILo", f2(ct["ci95_family_clustered"][0]), "family-clustered bootstrap"),
        ("RevAlphaCIHi", f2(ct["ci95_family_clustered"][1]), "family-clustered bootstrap"),
        ("RevAlphaLofoLo", f2(min(lofo)), "leave one family out, min"),
        ("RevAlphaLofoHi", f2(max(lofo)), "leave one family out, max"),
        ("RevAlphaLooMax", f2(ct["leave_one_model_out_max_abs_delta"]), "leave one model out, max |change|"),
        ("RevAlphaNonEmb", f2(ne["alpha"]), "size_exponent.non_embedding"),
        ("RevAlphaNonEmbCILo", f2(ne["ci95_family_clustered"][0]), "family-clustered"),
        ("RevAlphaNonEmbCIHi", f2(ne["ci95_family_clustered"][1]), "family-clustered"),
        ("RevEmbShareLo", f"{100 * min(share):.0f}", "embedding share of parameters, min, percent"),
        ("RevEmbShareHi", f"{100 * max(share):.0f}", "embedding share of parameters, max, percent"),
        ("RevTobitAlpha", f2(tb["scale_only"]["log_params_coef"]), "Tobit, OLMo-2-1B right-censored, size only"),
        ("RevTobitAlphaCILo", f2(tb["scale_only"]["boot_ci95_family_clustered"][0]), "family-clustered"),
        ("RevTobitAlphaCIHi", f2(tb["scale_only"]["boot_ci95_family_clustered"][1]), "family-clustered"),
        ("RevOlsRecipe", f2(rg["ols_uncensored_config_total"]["recipe_coef"]), "OLS, censored model dropped"),
        ("RevTobitRecipe", f2(tb["recipe_coef"]), "Tobit, censored model kept"),
        ("RevTobitRecipeRatio", f2(tb["recipe_ratio_exp"]), "exp(recipe coef)"),
        ("RevTobitRecipeCILo", f2(tb["boot_ci95_family_clustered"]["recipe"][0]), "family-clustered"),
        ("RevTobitRecipeCIHi", f2(tb["boot_ci95_family_clustered"]["recipe"][1]), "family-clustered"),
        ("RevTobitLRp", f2(tb["lr_test_recipe"]["p_chi2_1df"]), "likelihood ratio, chi2 1 df"),
        ("RevTobitNBoot", str(tb["n_boot_valid"]), "identified bootstrap draws"),
        ("RevTobitNBootDraws", str(tb["n_boot_draws"]), "bootstrap draws attempted"),
        ("RevNFamilies", str(tb["n_families"]), "families in the continuous protocol"),
        ("RevSubOneBN", str(len(sub)), "models below 1B parameters"),
        ("RevSubOneBKLo", f"{sub[0]:.1f}", "lowest K50 below 1B"),
        ("RevSubOneBKHi", f"{sub[-1]:.1f}", "highest K50 below 1B"),
        ("RevMidN", str(len(mid)), "models of 1.3B to 1.6B parameters"),
        ("RevMidKLo", f"{mid_unc[0]:.1f}", "lowest K50 in that band"),
        ("RevBigKLo", f"{min(S[k]['by_D']['0']['K50'] for k in MODELS if PARAMS_TOTAL_M[k] >= 1000):.1f}",
         "lowest K50 among models >= 1B"),
    ]
    ia = R.get("instrument_agreement")
    if ia:
        m += [("RevAgreeRho", f2(ia["spearman_rho"]), "Spearman k* vs K50, shared models"),
              ("RevAgreeP", fp(ia["spearman_p"]), "Spearman p"),
              ("RevAgreeN", str(ia["n_shared"]), "shared models")]
    if rep:
        fl_lo = min(v["filler_tokens_D256"]["min"] for v in rep.values())
        fl_hi = max(v["filler_tokens_D256"]["max"] for v in rep.values())
        m += [("RevSeq", str(next(iter(rep.values()))["seq"]), "padded sequence length"),
              ("RevMaxLen", str(max(v["max_len"] for v in rep.values())), "longest prompt, tokens"),
              ("RevNOver", str(sum(v["n_over_seq"] for v in rep.values())), "prompts longer than seq"),
              ("RevNPrompts", f"{sum(v['n_prompts'] for v in rep.values()):,}".replace(",", "{,}"),
               "prompts replayed"),
              ("RevFillerLo", str(fl_lo), "D=256 block, tokens, shortest prompt over all tokenizers"),
              ("RevFillerHi", str(fl_hi), "D=256 block, tokens, longest prompt over all tokenizers"),
              ("RevTokPerBindLo", f"{min(v['tokens_per_binding_D0'] for v in rep.values()):.1f}",
               "tokens added per binding, D=0, lowest tokenizer"),
              ("RevTokPerBindHi", f"{max(v['tokens_per_binding_D0'] for v in rep.values()):.1f}",
               "tokens added per binding, D=0, highest tokenizer")]
    # Recall above the in-prompt guess (a guesser that knows the listed values but ignores the
    # queried entity), largest over all loads K >= 2 of the grid; and recall at K = 2.
    B = R["baselines"]
    exc, p2 = {}, {}
    for k in MODELS:
        rows0 = [x for x in B[k]["0"] if x["K"] >= 2]
        best = max(rows0, key=lambda x: x["recall"] - x["present_random"])
        exc[k] = (best["recall"] - best["present_random"], best["K"])
        p2[k] = next(x["recall"] for x in B[k]["0"] if x["K"] == 2)
    small = [k for k in MODELS if PARAMS_TOTAL_M[k] < 1000]
    big = [k for k in MODELS if PARAMS_TOTAL_M[k] >= 1000]
    hiC = [k for k in MODELS if S[k]["by_D"]["0"]["K1_recall"] >= 0.9]
    U = R["usable_loads"]
    m += [("RevExcessSubLo", f2(min(exc[k][0] for k in small)), "max_K (recall - in-prompt guess), D=0, <1B, min"),
          ("RevExcessSubHi", up2(max(exc[k][0] for k in small)), "same, <1B, max, rounded up (an upper bound)"),
          ("RevExcessBigLo", f2(min(exc[k][0] for k in big)), "same, >=1B, min"),
          ("RevExcessBigHi", f2(max(exc[k][0] for k in big)), "same, >=1B, max"),
          ("RevRecallTwoLo", f2(min(p2[k] for k in hiC)), "recall at K=2, D=0, models with C>=0.9, min"),
          ("RevRecallTwoHi", f2(max(p2[k] for k in hiC)), "same, max"),
          ("RevNHiC", str(len(hiC)), "models with single-binding recall >= 0.9"),
          ("RevNinetyMaxK", str(max(U[k]["D0_raw_ge_0.9"] for k in MODELS)), "largest K with recall>=0.9 at every load up to K, D=0"),
          ("RevEightyMaxK", str(max(U[k]["D0_raw_ge_0.8"] for k in MODELS)), "same at recall>=0.8"),
          ("RevEightyNModels", str(sum(U[k]["D0_raw_ge_0.8"] > 1 for k in MODELS)), "models exceeding K=1 at recall>=0.8"),
          ("RevModalGapMax", f2(max(x["present_modal"] - x["present_random"]
                                    for k in MODELS for D in ("0", "256") for x in B[k][D])),
           "most-frequent-value guess minus in-prompt guess, max over models, D, K")]
    F = R.get("floors_and_tests")
    if F:
        sm, ct_ = F["size_models"], F["ceiling_only_test"]["per_model"]
        small_r = [ct_[k]["ratio"] for k in MODELS if PARAMS_TOTAL_M[k] < 1000]
        big_r = [ct_[k]["ratio"] for k in MODELS if PARAMS_TOTAL_M[k] >= 1000]
        tpb = [v["tokens_per_binding_D0"] for v in rep.values()]
        m += [("RevFloorKfifty", f2(np.mean([v["K50"] for v in F["entity_blind_floor"]["by_pool_size"].values()])),
               "K50 of the in-prompt guess under fit_k50"),
              ("RevStepRsq", f2(sm["two_level_step"]["r2"]), "two-level step (<1B vs >=1B), R^2, 11 models"),
              ("RevWithinBigRsq", f2(sm["within_1B_and_up"]["r2"]), "power law within the uncensored >=1B models"),
              ("RevWithinBigN", str(sm["within_1B_and_up"]["n"]), "uncensored models >= 1B"),
              ("RevWithinSmallRsq", f2(sm["within_below_1B"]["r2"]), "power law within the <1B models"),
              ("RevCeilSmallMin", f2(min(small_r)), "ceiling-only test, residual/noise, <1B, min"),
              ("RevCeilSmallMax", f2(max(small_r)), "ceiling-only test, residual/noise, <1B, max"),
              ("RevCeilBigLo", f2(min(big_r)), "same, >=1B, min"),
              ("RevCeilBigHi", f2(max(big_r)), "same, >=1B, max"),
              ("RevCeilBigNAbove", str(sum(r > 2 for r in big_r)), ">=1B models with ratio above 2"),
              ("RevTokSixtyFourLo", f"{63 * min(tpb):.0f}", "tokens added by bindings 2..64, lowest tokenizer"),
              ("RevTokSixtyFourHi", f"{63 * max(tpb):.0f}", "same, highest tokenizer")]
    G = R.get("calibration")
    if G:
        cn = G["ceiling_null"]["per_model"]
        small_k = [k for k in MODELS if PARAMS_TOTAL_M[k] < 1000]
        big_k = [k for k in MODELS if PARAMS_TOTAL_M[k] >= 1000]
        fs = G["feasible_sets"]
        olmo80 = [K for K in fs["olmo2-1b-x"]["D0_0.8"] if K > 1]
        qwen80 = [K for K in fs["qwen2.5-1.5b-x"]["D0_0.8"] if K > 1]
        nullp95 = G["max_margin_null"]["p95"]
        fr = G["first_round"]["models"]
        m += [("RevNullMarginP", up2(nullp95), "95th pct of the largest margin for a pure in-prompt guesser, rounded up"),
              ("RevExcessSubNAbove", str(sum(exc[k][0] > nullp95 for k in small_k)), "<1B models above that"),
              ("RevCeilNReject", str(G["ceiling_null"]["n_reject_05"]), "ceiling-only account rejected, p<0.05"),
              ("RevCeilNRejectBig", str(sum(cn[k]["p_upper"] < 0.05 for k in big_k)), "same, >=1B"),
              ("RevCeilNRejectSmall", str(sum(cn[k]["p_upper"] < 0.05 for k in small_k)), "same, <1B"),
              ("RevCeilNBonf", str(sum(cn[k]["p_upper"] < 0.05 / len(MODELS) for k in MODELS)), "same, Bonferroni over 12"),
              ("RevCeilNullHi", f2(max(v["null_p95"] for v in cn.values())), "largest null 95th pct of the ratio"),
              ("RevOlmoEightyLo", str(min(olmo80)), "OLMo-2-1B, smallest K>1 with recall>=0.8"),
              ("RevOlmoEightyHi", str(max(olmo80)), "OLMo-2-1B, largest K with recall>=0.8"),
              ("RevQwenEightyHi", str(max(qwen80)), "Qwen2.5-1.5B, largest K with recall>=0.8"),
              ("RevOlsRecipeCILo", f2(G["recipe_ols_ci"]["ci95"][0]), "OLS recipe coef, censored dropped, family CI"),
              ("RevOlsRecipeCIHi", f2(G["recipe_ols_ci"]["ci95"][1]), "same"),
              ("RevFirstN", str(G["first_round"]["n_models"]), "models in the first round"),
              ("RevFirstQwenThreeTwo", f2(fr["qwen2.5-3b"]["recall_D0"]["2"]), "Qwen2.5-3B first round, recall K=2"),
              ("RevFirstQwenThreeEight", f2(fr["qwen2.5-3b"]["recall_D0"]["8"]), "Qwen2.5-3B first round, recall K=8")]
        FR_NAME = {"gpt2": ("GPT-2", 124), "gpt2-medium": ("GPT-2-medium", 355), "gpt2-large": ("GPT-2-large", 774),
                   "pythia-70m": ("Pythia-70M", 70), "pythia-160m": ("Pythia-160M", 162), "pythia-410m": ("Pythia-410M", 405),
                   "pythia-1b": ("Pythia-1B", 1012), "pythia-1.4b": ("Pythia-1.4B", 1415), "pythia-2.8b": ("Pythia-2.8B", 2775),
                   "opt-125m": ("OPT-125M", 125), "opt-350m": ("OPT-350M", 331), "opt-1.3b": ("OPT-1.3B", 1316),
                   "opt-2.7b": ("OPT-2.7B", 2700), "phi-1.5": ("Phi-1.5", 1400), "qwen2.5-0.5b": ("Qwen2.5-0.5B", 494),
                   "qwen2.5-1.5b": ("Qwen2.5-1.5B", 1544), "qwen2.5-3b": ("Qwen2.5-3B", 3000),
                   "smollm2-135m": ("SmolLM2-135M", 135), "smollm2-360m": ("SmolLM2-360M", 362),
                   "smollm2-1.7b": ("SmolLM2-1.7B", 1700), "olmo2-1b": ("OLMo-2-1B", 1480)}
        rows_fr = []
        for lab in sorted(fr, key=lambda l: FR_NAME[l][1]):
            r_ = fr[lab]["recall_D0"]
            note = ("excluded" if lab in FIRST_ROUND_EXCLUDED else
                    "not re-measured" if lab in FIRST_ROUND_ONLY else "re-measured")
            rows_fr.append(f"{FR_NAME[lab][0]} & {r_['1']:.2f} & {r_['2']:.2f} & {r_['8']:.2f} & {r_['24']:.2f} & {note} \\\\")
        tab_fr = ["% Generated by analysis/p3_revision_checks.py --emit. Regenerate, never hand-edit.",
                  "\\begin{tabular}{@{}lrrrrl@{}}", "\\toprule",
                  "model & $K{=}1$ & 2 & 8 & 24 & final round \\\\", "\\midrule"] + rows_fr + ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(tex_dir, "table_firstround.tex"), "w").write("\n".join(tab_fr) + "\n")
    lines = ["% Generated by analysis/p3_revision_checks.py --emit from results/p3_revision_checks.json.",
             "% Regenerate, never hand-edit. EXPLORATORY re-analysis of saved curves (2026-10-23).", ""]
    for name, val, note in m:
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}" + " " * max(1, 34 - len(name) - len(val)) + f"% {note}")
    open(os.path.join(tex_dir, "numbers_revision.tex"), "w").write("\n".join(lines) + "\n")

    # Table 1: one row per model, sorted by parameter count.
    I = {x["label"]: x for x in json.load(open(os.path.join(res_dir, "interference_axis.json")))["models"]}
    rows = []
    for k in sorted(MODELS, key=lambda k: PARAMS_TOTAL_M[k]):
        d0 = S[k]["by_D"]["0"]
        rec = dict(zip([int(x) for x in d0["K_grid"]], d0["recall"]))
        k50 = f"$>${d0['K50']:.0f}" if d0["censored"] else f"{d0['K50']:.1f}"
        it = I[k]
        if it.get("K50_D256_vs_D0_ceiling") is None:
            k50d = "--"
        elif it["K50_D256_vs_D0_ceiling"] <= 1.0:   # fixed-reference recall below midpoint at K=1
            k50d = "$\\leq$1"
        else:
            k50d = (f"$>${it['K50_D256_vs_D0_ceiling']:.0f}" if it.get("K50_D256_censored")
                    else f"{it['K50_D256_vs_D0_ceiling']:.1f}")
        rows.append(f"{PRETTY[k]} & {PARAMS_TOTAL_M[k]:,} & {MAX_POSITIONS[k]:,}".replace(",", "{,}")
                    + f" & {rec[1]:.2f} & {rec[2]:.2f} & {rec[8]:.2f} & {rec[32]:.2f}"
                    + f" & {f2(exc[k][0])} & {k50} & {k50d} \\\\")
    tab = ["% Generated by analysis/p3_revision_checks.py --emit. Regenerate, never hand-edit.",
           "\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
           " & & & \\multicolumn{4}{c}{recall at load $K$} & & & \\\\",
           "\\cmidrule(lr){4-7}",
           "model & $N$ (M) & window & 1 & 2 & 8 & 32 & above guess & $K_{50}$ & $K_{50}^{\\,D}$ \\\\",
           "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tex_dir, "table_capacity.tex"), "w").write("\n".join(tab) + "\n")
    print("wrote numbers_revision.tex and table_capacity.tex to", tex_dir)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--tex", default=None, help="write macros to this numbers_revision.tex path")
    ap.add_argument("--skip-tokenizers", action="store_true")
    ap.add_argument("--emit", default=None, help="only write the TeX files into this directory")
    ap.add_argument("--update-calibration", action="store_true",
                    help="add section G to an existing results file without rerunning the rest")
    ap.add_argument("--update-floors", action="store_true",
                    help="add section F to an existing results file without rerunning the rest")
    a = ap.parse_args()
    if a.emit:
        emit_tex(a.res, a.emit)
        return
    if a.update_calibration:
        dst = os.path.join(a.res, "p3_revision_checks.json")
        out = json.load(open(dst))
        summary = json.load(open(os.path.join(a.res, "capacity_trn1_summary.json")))
        out["calibration"] = calibration(summary, os.path.join(a.res, "capacity_x"),
                                         os.path.join(a.res, "capacity_trn1"))
        json.dump(out, open(dst, "w"), indent=1)
        c = out["calibration"]
        print(json.dumps({k: v for k, v in c.items() if k not in ("first_round", "feasible_sets")}, indent=1))
        print(json.dumps(c["feasible_sets"]))
        print(json.dumps({k: v["recall_D0"] for k, v in c["first_round"]["models"].items()}))
        return
    if a.update_floors:
        dst = os.path.join(a.res, "p3_revision_checks.json")
        out = json.load(open(dst))
        summary = json.load(open(os.path.join(a.res, "capacity_trn1_summary.json")))
        out["floors_and_tests"] = floors_and_tests(summary, os.path.join(a.res, "capacity_x"))
        json.dump(out, open(dst, "w"), indent=1)
        print(json.dumps(out["floors_and_tests"], indent=1)[:3000])
        return

    spath = os.path.join(a.res, "capacity_trn1_summary.json")
    xdir = os.path.join(a.res, "capacity_x")
    summary = json.load(open(spath))

    st = selftest_tobit()
    print("[selftest]", st, flush=True)
    if not st["passed"]:
        raise SystemExit("Tobit self-test failed; refusing to report a censoring-aware fit.")

    # Cheap sections first, so a failure surfaces before the bootstrap starts.
    fast = {"instrument_agreement": instrument_agreement(summary, os.path.join(a.res, "figdata.json")),
            "usable_loads": usable_loads(summary)}
    if not a.skip_tokenizers:
        fast["replay"], fast["baselines"] = replay_and_baselines(summary, xdir, True)
    fast["floors_and_tests"] = floors_and_tests(summary, xdir)
    fast["calibration"] = calibration(summary, xdir, os.path.join(a.res, "capacity_trn1"))
    reg, rows = regression(summary)
    ne = nonembedding(rows)
    out = {"_provenance": {"script": "analysis/p3_revision_checks.py", "task": TASK_SOURCE,
                           "inputs_sha256_16": {"capacity_trn1_summary.json": sha256(spath),
                                                "figdata.json": sha256(os.path.join(a.res, "figdata.json"))},
                           "status": "EXPLORATORY re-analysis of saved curves; no new model runs"},
           "selftest_tobit": st, "regression": reg, "size_exponent": ne,
           "max_positions": MAX_POSITIONS, **fast}
    dst = os.path.join(a.res, "p3_revision_checks.json")
    json.dump(out, open(dst, "w"), indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    print("wrote", dst)


if __name__ == "__main__":
    main()
