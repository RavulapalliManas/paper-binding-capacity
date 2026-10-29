"""Pretraining development and hidden-state readout of in-context bindings, for the capacity paper.

Re-analyses three sets of saved runs. No model is loaded and no data value is typed in here.

  ladder  qgeom.py at public Pythia checkpoints: results/qgfix_p160, qgfix_p410, qgfix_p1b,
          qgfix (1.4B), qgfix_p69, qgfix_p28 /step*/qgeom.json. Each file records its argv:
          K=6 bindings with distinct values, a D=256 code-like distractor block drawn per trial,
          1,500 trials, the residual stream at layer n_layers//2 at the query's last token,
          standardised on a random 70% split, a multinomial logistic probe whose L2 strength is
          chosen by 3-fold cross-validation on that split, scored on the other 30% (450 trials).
          Runs whose probe scored below 3x pool chance stopped after recording that score
          ('aborted'); the score is the same estimator and is kept. A checkpoint whose embedding
          hash repeats an earlier one is a duplicate download and is dropped; a model with fewer
          than 4 distinct checkpoints is excluded (this drops Pythia-2.8B).
  steer   steer.py at Pythia-1.4B and 6.9B checkpoints: results/dynamics/em_step{S}{,_p69}/
          {steer,per_layer}.json. K=6, D=256, the trials split into thirds: a probe per layer is
          fit on the first third, the steering layer is the one whose probe is most accurate on
          the second third's wrong answers, and everything is reported on the last third. The
          model's answer is its highest logit among the six listed values. A push adds
          alpha * mean||h|| along a unit direction at the steering layer, at the query position.
  post    dyn.py posterior batteries at four Pythia-1.4B checkpoints: results/dynamics/
          dev_step{1000,8000,32000,143000}/posterior.json. K=6, D=256 code-like distractor tokens,
          1,200 trials split into thirds (probe fit on the first, layer chosen on the second,
          probe refit on both and reported on the last, 400 trials). Each saved row is one test trial:
          whether the model's answer is wrong, whether the probe's top value is right, and the
          model's probability for the right value among the six listed values.
  k4      neuron_legibility.py at the final checkpoints of eight models, K=4, no distractors,
          the 'declarative' task family: results/legibility_x/leg-*-K4D0.json and
          results/legibility_trn1/leg-p14b-K4D0.json. Its interventions replace the state at the
          read layer with a value's class mean.

Run:     python p3_development.py [--emit DIR] [--selftest]
Output:  results/p3_development.json; with --emit, DIR/numbers_development.tex
Memory:  the inputs are JSON files under 1 MB in total; peak use is a few MB.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import re
import sys

import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

LADDER = [("160M", "qgfix_p160"), ("410M", "qgfix_p410"), ("1B", "qgfix_p1b"),
          ("1.4B", "qgfix"), ("2.8B", "qgfix_p28"), ("6.9B", "qgfix_p69")]
STEER = [("1.4B", "em_step{}"), ("6.9B", "em_step{}_p69")]
K4_FILES = ["legibility_x/leg-p410m-K4D0.json", "legibility_x/leg-q05b-K4D0.json",
            "legibility_x/leg-p1b-K4D0.json", "legibility_x/leg-olmo1b-K4D0.json",
            "legibility_trn1/leg-p14b-K4D0.json", "legibility_x/leg-q15b-K4D0.json",
            "legibility_x/leg-sm17b-K4D0.json", "legibility_x/leg-p28b-K4D0.json"]
PRETTY = {"EleutherAI/pythia-410m": "Pythia-410M", "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
          "EleutherAI/pythia-1b": "Pythia-1B", "allenai/OLMo-2-0425-1B": "OLMo-2-1B",
          "EleutherAI/pythia-1.4b": "Pythia-1.4B", "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
          "HuggingFaceTB/SmolLM2-1.7B": "SmolLM2-1.7B", "EleutherAI/pythia-2.8b": "Pythia-2.8B"}
SMALL = ("160M", "410M")              # the two Pythia models below 1B parameters
N_PERM = 200_000


def jl(rel):
    with open(os.path.join(RES, rel)) as f:
        return json.load(f)


def wilson(p, n, z=1.96):
    """95% Wilson interval for a proportion p measured on n trials."""
    if n <= 0:
        return (float("nan"), float("nan"))
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def spearman_perm(x, y, n_perm=N_PERM, seed=0):
    """Spearman rho with a two-sided permutation p: exact enumeration up to n=8, seeded Monte Carlo above."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearmanr(x, y).correlation)
    n = len(x)
    rx, ry = rankdata(x), rankdata(y)                  # average ranks, as spearmanr uses
    rxc = rx - rx.mean()

    def rhos(perms):                                   # Pearson correlation of rx with each permuted ry
        pc = perms - perms.mean(1, keepdims=True)
        return (pc @ rxc) / np.sqrt((rxc ** 2).sum() * (pc ** 2).sum(1))

    if n <= 8:
        perms = ry[np.array(list(itertools.permutations(range(n))))]
        return rho, float(np.mean(np.abs(rhos(perms)) >= abs(rho) - 1e-12)), "exact"
    rng = np.random.default_rng(seed)
    perms = ry[np.argsort(rng.random((n_perm, n)), axis=1)]
    hits = int((np.abs(rhos(perms)) >= abs(rho) - 1e-12).sum())
    return rho, (hits + 1) / (n_perm + 1), "monte_carlo"


def selftest():
    """The permutation test and the interval must recover known answers before they are used."""
    rho, p, kind = spearman_perm(range(7), range(7))
    assert kind == "exact" and abs(rho - 1) < 1e-12 and abs(p - 2 / math.factorial(7)) < 1e-12, (rho, p)
    rho, p, kind = spearman_perm(range(12), range(12), n_perm=20_000)
    assert kind == "monte_carlo" and p < 1e-3, p
    rng = np.random.default_rng(3)
    ps = [spearman_perm(range(7), rng.permutation(7))[1] for _ in range(200)]
    assert 0.02 < float(np.mean(np.array(ps) < 0.05)) < 0.09, np.mean(np.array(ps) < 0.05)
    lo, hi = wilson(0.45, 100)
    assert abs(lo - 0.3561) < 1e-3 and abs(hi - 0.5476) < 1e-3, (lo, hi)
    return {"spearman_exact_monotone": True, "spearman_mc_monotone": True,
            "null_rejection_rate_n7": round(float(np.mean(np.array(ps) < 0.05)), 3), "wilson": True}


def ladder():
    out = {}
    for lab, d in LADDER:
        rows, seen, dropped = [], set(), 0
        for p in sorted(glob.glob(os.path.join(RES, d, "step*", "qgeom.json")),
                        key=lambda q: int(re.search(r"step(\d+)", q).group(1))):
            r = json.load(open(p))
            if r.get("grad_frame", "mul") != "mul":     # the 1.4B folder also holds one 'div' rerun of step 1000
                continue
            if r["embed_md5"] in seen:
                dropped += 1
                continue
            seen.add(r["embed_md5"])
            acc = r["decode_obl_full"] if "decode_obl_full" in r else r["decode_obl"]
            rows.append(dict(step=int(re.search(r"step(\d+)", p).group(1)), acc=acc,
                             aborted=bool(r.get("aborted", False)), layer=r["layer"],
                             chance=r["chance"], argv=r.get("argv", ""),
                             entity_acc=r.get("decode_ent_full")))
        argv = " ".join(x["argv"] for x in rows if x["argv"])
        K = int(re.search(r"--K (\d+)", argv).group(1)) if "--K" in argv else 6
        D = int(re.search(r"--D (\d+)", argv).group(1)) if "--D" in argv else 256
        n = int(re.search(r"--n (\d+)", argv).group(1)) if "--n " in argv else 1500
        n_test = n - int(0.7 * n)
        entry = dict(n_checkpoints=len(rows), dropped_duplicates=dropped, K=K, D=D, n_trials=n,
                     n_test=n_test, layer=sorted({x["layer"] for x in rows}),
                     pool_chance=rows[0]["chance"] if rows else None,
                     steps=[x["step"] for x in rows], acc=[round(x["acc"], 4) for x in rows],
                     n_aborted=sum(x["aborted"] for x in rows),
                     entity_acc=[x["entity_acc"] for x in rows],
                     entity_acc_min=min((x["entity_acc"] for x in rows if x["entity_acc"] is not None), default=None),
                     n_entity=sum(x["entity_acc"] is not None for x in rows))
        if len(rows) < 4:
            entry["excluded"] = "fewer than 4 distinct checkpoints"
        else:
            rho, p, kind = spearman_perm([x["step"] for x in rows], [x["acc"] for x in rows])
            fin = rows[-1]["acc"]
            entry.update(rho=round(rho, 4), p=p, p_kind=kind, final=round(fin, 4),
                         final_ci=[round(v, 4) for v in wilson(fin, n_test)],
                         max=round(max(x["acc"] for x in rows), 4),
                         min=round(min(x["acc"] for x in rows), 4))
        out[lab] = entry
    return out


def steer():
    out = {}
    for lab, pat in STEER:
        rows = []
        for d in sorted(glob.glob(os.path.join(RES, "dynamics", pat.format("*"))),
                        key=lambda q: int(re.search(r"em_step(\d+)", q).group(1))):
            if lab == "1.4B" and d.endswith("_p69"):
                continue
            st = json.load(open(os.path.join(d, "steer.json")))
            pl = json.load(open(os.path.join(d, "per_layer.json")))
            n = st["n"]
            n_test = n - 2 * (n // 3)
            assert pl["n_fail"] == st["n_fail"] and pl["lstar"] == st["lstar"]
            R = st["results"]
            row = dict(step=int(re.search(r"em_step(\d+)", d).group(1)), n=n, n_test=n_test,
                       n_wrong=st["n_fail"], acc=round(1 - st["n_fail"] / n_test, 4),
                       layer=st["lstar"], final_layer=pl["final_layer"], K=st["K"], D=st["D"],
                       probe_all=round(pl["per_layer_all"][st["lstar"]], 4),
                       probe_wrong=round(pl["per_layer_fail"][st["lstar"]], 4),
                       probe_wrong_final=round(pl["per_layer_fail"][pl["final_layer"]], 4),
                       probe_all_final=round(pl["per_layer_all"][pl["final_layer"]], 4),
                       gated_acc=round(R["gated_all"]["0.5"]["acc"], 4),
                       gated_base=round(R["gated_all"]["0.5"]["base"], 4),
                       gated_fired=round(R["gated_all"]["0.5"]["fired"], 4))
            assert abs(row["gated_base"] - row["acc"]) < 1e-9, "gated base must be the test accuracy"
            for a in ("0.5", "1.0"):
                row[f"gold_w_{a}"] = round(R["linear_w"][a]["repair"], 4)
                row[f"gold_mu_{a}"] = round(R["manifold_mu"][a]["repair"], 4)
                row[f"probe_w_{a}"] = round(R["linear_w_decoded"][a]["repair"], 4)
                row[f"probe_mu_{a}"] = round(R["manifold_mu_decoded"][a]["repair"], 4)
                row[f"random_{a}"] = round(R["random"][a]["repair"], 4)
                row[f"keep_probe_w_{a}"] = round(R["linear_w_decoded"][a]["damage_keep"], 4)
                row[f"keep_random_{a}"] = round(R["random"][a]["damage_keep"], 4)
            assert R["baseline"]["repair"] == 0.0 and R["baseline"]["damage_keep"] == 1.0
            row["probe_over_gold_w_0.5"] = round(row["probe_w_0.5"] / row["gold_w_0.5"], 4) \
                if row["gold_w_0.5"] > 0 else None
            rows.append(row)
        late = [r for r in rows if r["step"] >= 8000]
        fin = rows[-1]
        summary = dict(
            n_checkpoints=len(rows), steps=[r["step"] for r in rows],
            gold_w_late_range=[min(r["gold_w_0.5"] for r in late), max(r["gold_w_0.5"] for r in late)],
            gold_mu_late_range=[min(r["gold_mu_0.5"] for r in late), max(r["gold_mu_0.5"] for r in late)],
            random_max=max(max(r["random_0.5"], r["random_1.0"]) for r in rows),
            random_max_05=max(r["random_0.5"] for r in rows),
            probe_w_first=rows[0]["probe_w_0.5"], probe_w_8000=next(r["probe_w_0.5"] for r in rows if r["step"] == 8000),
            probe_w_final=fin["probe_w_0.5"], probe_mu_final=fin["probe_mu_0.5"],
            ratio_8000=next(r["probe_over_gold_w_0.5"] for r in rows if r["step"] == 8000),
            ratio_final=fin["probe_over_gold_w_0.5"],
            keep_probe_8000=next(r["keep_probe_w_0.5"] for r in rows if r["step"] == 8000),
            keep_probe_final=fin["keep_probe_w_0.5"], keep_random_final=fin["keep_random_0.5"],
            final=fin,
            final_probe_wrong_ci=[round(v, 4) for v in wilson(fin["probe_wrong"], fin["n_wrong"])],
            final_probe_wrong_final_ci=[round(v, 4) for v in wilson(fin["probe_wrong_final"], fin["n_wrong"])],
            final_probe_w_ci=[round(v, 4) for v in wilson(fin["probe_w_0.5"], fin["n_wrong"])],
            probe_w_rho=spearman_perm([r["step"] for r in rows], [r["probe_w_0.5"] for r in rows])[:2],
            gold_w_late_rho=spearman_perm([r["step"] for r in late], [r["gold_w_0.5"] for r in late])[:2],
            ratio_late_rho=spearman_perm([r["step"] for r in late], [r["probe_over_gold_w_0.5"] for r in late])[:2],
            gated_first=rows[0]["gated_acc"], gated_final=fin["gated_acc"], gated_fired_final=fin["gated_fired"],
            probe_all_final_layer=fin["probe_all_final"],
            depth_gap_first=round(rows[0]["probe_wrong"] - rows[0]["probe_wrong_final"], 4),
            depth_gap_final=round(fin["probe_wrong"] - fin["probe_wrong_final"], 4),
            depth_gap_rho=spearman_perm([r["step"] for r in rows],
                                        [r["probe_wrong"] - r["probe_wrong_final"] for r in rows])[:2])
        out[lab] = dict(rows=rows, summary=summary)
    return out


POST_STEPS = (1000, 8000, 32000, 143000)


def posterior():
    """Wrong answers of Pythia-1.4B split by whether the probe's top value is right and whether the model
    gives the right value at least 1/K probability among the K listed values (P2's saved batteries)."""
    rows = []
    for st in POST_STEPS:
        d = jl(f"dynamics/dev_step{st}/posterior.json")
        k = d["K"]
        wrong = [r for r in d["rows"] if r["model_wrong"]]
        for r in d["rows"]:
            assert r["model_wrong"] == (r["correct_ix"] != r["emitted_ix"])
        g = {"probe_right_output_low": 0, "probe_right_output_high": 0,
             "probe_wrong_output_low": 0, "probe_wrong_output_high": 0}
        for r in wrong:
            pk = "probe_right" if r["probe_top1"] else "probe_wrong"
            ok = "output_high" if r["model_p_correct"] >= 1 / k else "output_low"
            g[f"{pk}_{ok}"] += 1
        n_pr = g["probe_right_output_low"] + g["probe_right_output_high"]
        rows.append(dict(step=st, K=k, D=d["D"], cond=d["cond"], layer=d["layer"], n=len(d["rows"]),
                         n_wrong=len(wrong), groups=g,
                         share_probe_right=round(n_pr / len(wrong), 4),
                         share_probe_right_output_low=round(g["probe_right_output_low"] / len(wrong), 4),
                         share_output_high_given_probe_right=round(g["probe_right_output_high"] / n_pr, 4) if n_pr else None))
    return dict(rows=rows)


def k4_prompt_baseline():
    """Guess among the listed values on the K=4 'declarative' prompts, from the rendered prompts.
    neuron_legibility.py builds the trials with trilogy_tasks.build_declarative(n, K=4, D=0, seed=0)
    and renders VALUE_i as vals[i % 18], so a value word can repeat inside a prompt."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("trilogy_tasks", os.path.join(HERE, "trilogy_tasks.py"))
    tt = importlib.util.module_from_spec(spec)
    sys.modules["trilogy_tasks"] = tt           # its dataclasses look the module up while it loads
    spec.loader.exec_module(tt)
    n, k, nv = 1536, 4, 18
    trials = tt.build_declarative(n, K=k, D=0, seed=0)

    def vid(s):
        return int(re.search(r"(\d+)$", s).group(1))
    guess, rep_v, test = [], [], []
    for t in trials:
        vals = [vid(v) % nv for (_, _, v) in t.body]
        gold = vid(t.gold) % nv
        guess.append(sum(v == gold for v in vals) / k)
        rep_v.append(len(set(vals)) < k)
        test.append(t.fold == "test")
    g, r, te = np.array(guess), np.array(rep_v), np.array(test)
    return dict(n=n, n_test=int(te.sum()), guess_all=float(g.mean()), guess_test=float(g[te].mean()),
                repeat_all=float(r.mean()), repeat_test=float(r[te].mean()))


def k4():
    base = k4_prompt_baseline()
    rows = []
    for rel in K4_FILES:
        d = jl(rel)
        f = d["families"]["declarative"]
        rep = f["repair"]
        assert rep["noop"]["recovery_on_wrong"] == 0.0 and rep["noop"]["max_logit_drift"] == 0.0
        assert d["K"] == 4 and d["D"] == 0
        rows.append(dict(model=PRETTY[d["model"]], K=d["K"], D=d["D"], n_values=d["n_values"],
                         n_trials=f["n"], acc=f["accuracy"], n_test=f["n_test"], n_wrong=f["n_wrong_test"],
                         probe_wrong=f["p_acc_given_wrong"], baseline=f["present_set_baseline"],
                         probe_wrong_ci=[round(v, 4) for v in wilson(f["p_acc_given_wrong"], f["n_wrong_test"])],
                         gold=rep["gold"]["recovery_on_wrong"], probe=rep["decoded"]["recovery_on_wrong"],
                         random=rep["random_matched"]["recovery_on_wrong"],
                         wrong_target_keep=rep["wrong_target"]["correct_preserved"],
                         gold_keep=rep["gold"]["correct_preserved"],
                         random_keep=rep["random_matched"]["correct_preserved"],
                         disagree_auroc=f["disagreement_auroc"],
                         acc_test=round(1 - f["n_wrong_test"] / f["n_test"], 4)))
        assert f["n_test"] == base["n_test"], "the regenerated folds must match the saved test fold"
    pw = [r["probe_wrong"] for r in rows]
    moved = [1 - r["wrong_target_keep"] for r in rows]
    summary = dict(n_models=len(rows), probe_wrong_range=[min(pw), max(pw)], baseline=rows[0]["baseline"],
                   n_wrong_range=[min(r["n_wrong"] for r in rows), max(r["n_wrong"] for r in rows)],
                   ci_lo_min=min(r["probe_wrong_ci"][0] for r in rows),
                   wrong_target_moved_range=[round(min(moved), 4), round(max(moved), 4)],
                   gold_range=[min(r["gold"] for r in rows), max(r["gold"] for r in rows)],
                   by_model={r["model"]: r["probe_wrong"] for r in rows},
                   auroc_range=[min(r["disagree_auroc"] for r in rows), max(r["disagree_auroc"] for r in rows)],
                   prompt_baseline=base)
    cap = jl("capacity_trn1_summary.json")["models"]
    cap_ids = {m["model"] for m in (cap.values() if isinstance(cap, dict) else cap)}
    summary["n_in_capacity_set"] = sum(jl(rel)["model"] in cap_ids for rel in K4_FILES)
    return dict(rows=rows, summary=summary)


def f2(x):
    """Two decimals, half rounded up (0.565 prints as 0.57)."""
    return str(Decimal(repr(round(float(x), 10))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def up2(x):
    """An upper bound at two decimals, rounded up so that 'at most' stays true."""
    return f"{math.ceil(round(x * 100, 9)) / 100:.2f}"


def down2(x):
    """A lower bound at two decimals, rounded down so that 'at least' stays true."""
    return f"{math.floor(round(x * 100, 9)) / 100:.2f}"


def fp(p):
    if p < 1e-3:
        e = int(math.floor(math.log10(p)))
        return f"{p / 10 ** e:.1f}\\times10^{{{e}}}"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def fp_up(p):
    """A p-value bound ('p at most ...'), rounded up at the printed precision."""
    if p < 1e-3:
        e = int(math.floor(math.log10(p)))
        return f"{math.ceil(round(p / 10 ** e * 10, 9)) / 10:.1f}\\times10^{{{e}}}"
    if p < 0.01:
        return f"{math.ceil(round(p * 1000, 9)) / 1000:.3f}"
    return up2(p)


def emit_tex(res, tex_dir):
    L, S, K, P = res["ladder"], res["steer"], res["k4"], res["posterior"]["rows"]
    small_max = max(L[m]["max"] for m in SMALL)
    assert L["160M"]["n_checkpoints"] == L["410M"]["n_checkpoints"], "the text gives one count for both"
    big = [m for m in ("1B", "1.4B", "6.9B")]
    s14, s69 = S["1.4B"]["summary"], S["6.9B"]["summary"]
    f14 = s14["final"]
    M = {
        # ladder
        "DvBaseline": f2(1 / L["1B"]["K"]), "DvPoolChance": f2(L["1B"]["pool_chance"]),
        "DvLadderK": str(L["1B"]["K"]), "DvLadderTest": str(L["1B"]["n_test"]),
        "DvLadderTrials": f"{L['1B']['n_trials']:,}".replace(",", "{,}"),
        "DvSmallMax": up2(small_max),
        "DvSmallCkpt": str(L["160M"]["n_checkpoints"]),
        "DvSmallEntityMin": down2(min(L[m]["entity_acc_min"] for m in SMALL)),
        "DvSmallEntityN": str(sum(L[m]["n_entity"] for m in SMALL)),
        "DvSmallCkptAll": str(sum(L[m]["n_checkpoints"] for m in SMALL)),
        "DvSmallRhoPOneSixty": fp(L["160M"]["p"]), "DvSmallRhoOneSixty": f2(L["160M"]["rho"]),
        "DvBigFinalLo": f2(min(L[m]["final"] for m in big)), "DvBigFinalHi": f2(max(L[m]["final"] for m in big)),
        "DvBigRhoMin": down2(min(L[m]["rho"] for m in big)), "DvBigPMax": fp_up(max(L[m]["p"] for m in big)),
        "DvFourTenRho": f2(L["410M"]["rho"]), "DvFourTenP": fp(L["410M"]["p"]),
        "DvFourTenFirst": f2(L["410M"]["acc"][0]), "DvFourTenFinal": f2(L["410M"]["final"]),
        "DvCkptLo": str(min(L[m]["n_checkpoints"] for m in ("160M", "410M", "1B", "1.4B", "6.9B"))),
        "DvCkptHi": str(max(L[m]["n_checkpoints"] for m in ("160M", "410M", "1B", "1.4B", "6.9B"))),
        "DvOneBFirst": f2(L["1B"]["acc"][0]), "DvOneFourFirst": f2(L["1.4B"]["acc"][0]),
        "DvSixNineFirst": f2(L["6.9B"]["acc"][0]),
        # steering, Pythia-1.4B
        "DvStAcc": f2(f14["acc"]), "DvStTest": str(f14["n_test"]), "DvStWrong": str(f14["n_wrong"]),
        "DvStLayer": str(f14["layer"]), "DvStFinalLayer": str(f14["final_layer"]),
        "DvStProbeAll": f2(f14["probe_all"]), "DvStProbeWrong": f2(f14["probe_wrong"]),
        "DvStProbeWrongLo": f2(s14["final_probe_wrong_ci"][0]), "DvStProbeWrongHi": f2(s14["final_probe_wrong_ci"][1]),
        "DvStProbeWrongFinal": f2(f14["probe_wrong_final"]),
        "DvStGoldLo": f2(s14["gold_w_late_range"][0]), "DvStGoldHi": f2(s14["gold_w_late_range"][1]),
        "DvStGoldMuLo": f2(s14["gold_mu_late_range"][0]), "DvStGoldMuHi": f2(s14["gold_mu_late_range"][1]),
        "DvStProbeMuFinal": f2(s14["probe_mu_final"]),
        "DvStProbeWrongFirst": f2(S["1.4B"]["rows"][0]["probe_wrong"]), "DvStAccFirst": f2(S["1.4B"]["rows"][0]["acc"]),
        "DvStTrials": f"{f14['n']:,}".replace(",", "{,}"), "DvSixTrials": str(s69["final"]["n"]),
        "DvLadderTrain": f"{L['1B']['n_trials'] - L['1B']['n_test']:,}".replace(",", "{,}"),
        "DvAbortOneSixty": str(L["160M"]["n_aborted"]), "DvAbortFourTen": str(L["410M"]["n_aborted"]),
        "DvTwoEightKept": str(L["2.8B"]["n_checkpoints"]),
        "DvTwoEightAll": str(L["2.8B"]["n_checkpoints"] + L["2.8B"]["dropped_duplicates"]),
        "DvSixRandomMax": up2(s69["random_max_05"]),
        "DvKfTrials": f"{K['rows'][0]['n_trials']:,}".replace(",", "{,}"), "DvKfValues": str(K["rows"][0]["n_values"]),
        "DvStRandomMax": up2(s14["random_max_05"]),
        "DvStProbeFirst": f2(s14["probe_w_first"]), "DvStProbeEight": f2(s14["probe_w_8000"]),
        "DvStProbeFinal": f2(s14["probe_w_final"]),
        "DvStRatioEight": f2(s14["ratio_8000"]), "DvStRatioFinal": f2(s14["ratio_final"]),
        "DvStKeepEight": f2(s14["keep_probe_8000"]), "DvStKeepFinal": f2(s14["keep_probe_final"]),
        "DvStKeepRandom": f2(s14["keep_random_final"]),
        # steering, Pythia-6.9B (appendix)
        "DvSixAcc": f2(s69["final"]["acc"]), "DvSixProbeWrong": f2(s69["final"]["probe_wrong"]),
        "DvSixProbeFinal": f2(s69["probe_w_final"]), "DvSixProbeFirst": f2(s69["probe_w_first"]),
        "DvSixGoldLo": f2(s69["gold_w_late_range"][0]), "DvSixGoldHi": f2(s69["gold_w_late_range"][1]),
        # final checkpoints, K = 4, no distractors
        "DvKfN": str(K["summary"]["n_models"]), "DvKfLo": f2(K["summary"]["probe_wrong_range"][0]),
        "DvKfHi": f2(K["summary"]["probe_wrong_range"][1]), "DvKfBase": f2(K["summary"]["prompt_baseline"]["guess_test"]),
        "DvKfRepeat": f2(K["summary"]["prompt_baseline"]["repeat_all"]),
        "DvKfInTableOne": {7: "seven", 8: "all eight", 6: "six"}.get(K["summary"]["n_in_capacity_set"],
                                                                    str(K["summary"]["n_in_capacity_set"])),
        "DvKfPFourTen": f2(K["summary"]["by_model"]["Pythia-410M"]),
        "DvKfQZeroFive": f2(K["summary"]["by_model"]["Qwen2.5-0.5B"]),
        "DvKfMovedLo": f2(K["summary"]["wrong_target_moved_range"][0]),
        "DvKfMovedHi": f2(K["summary"]["wrong_target_moved_range"][1]),
        "DvKfWrongLo": str(K["summary"]["n_wrong_range"][0]), "DvKfWrongHi": str(K["summary"]["n_wrong_range"][1]),
        "DvKfAurocLo": f2(K["summary"]["auroc_range"][0]), "DvKfAurocHi": f2(K["summary"]["auroc_range"][1]),
        # self-gated steering and depth, Pythia-1.4B
        "DvGatedFinal": f2(s14["gated_final"]), "DvGatedFired": f2(s14["gated_fired_final"]),
        "DvGatedFirst": f2(s14["gated_first"]), "DvSixGatedFinal": f2(s69["gated_final"]),
        "DvSixGatedBase": f2(s69["final"]["acc"]),
        "DvStProbeAllFinalLayer": f2(s14["probe_all_final_layer"]),
        "DvDepthGapFirst": f2(s14["depth_gap_first"]), "DvDepthGapFinal": f2(s14["depth_gap_final"]),
        "DvDepthGapRho": f2(s14["depth_gap_rho"][0]), "DvDepthGapP": fp(s14["depth_gap_rho"][1]),
        # wrong answers split by probe and output, Pythia-1.4B posterior batteries
        "DvPostWrongFirst": str(P[0]["n_wrong"]), "DvPostWrongFinal": str(P[-1]["n_wrong"]),
        "DvPostN": str(P[-1]["n"]),
        "DvPostProbeRightFirst": f2(P[0]["share_probe_right"]), "DvPostProbeRightFinal": f2(P[-1]["share_probe_right"]),
        "DvPostLowFinal": f2(P[-1]["share_probe_right_output_low"]),
        "DvPostHighGivenRightFirst": f2(P[0]["share_output_high_given_probe_right"]),
        "DvPostHighGivenRightFinal": f2(P[-1]["share_output_high_given_probe_right"]),
    }
    lines = ["% Generated by analysis/p3_development.py from results/p3_development.json. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    with open(os.path.join(tex_dir, "numbers_development.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # Appendix table: the eight final-checkpoint models at K = 4.
    t = ["% Generated by analysis/p3_development.py. Do not edit.",
         "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
         "model & recall & wrong & probe on wrong & gold & probe & wrong value & \\\\",
         " & & answers & (95\\% interval) & repair & repair & keeps correct & AUROC \\\\", "\\midrule"]
    for r in K["rows"]:
        lo, hi = r["probe_wrong_ci"]
        t.append(f"{r['model']} & {f2(r['acc_test'])} & {r['n_wrong']} & {f2(r['probe_wrong'])} ({f2(lo)}--{f2(hi)}) & "
                 f"{f2(r['gold'])} & {f2(r['probe'])} & {f2(r['wrong_target_keep'])} & {f2(r['disagree_auroc'])} \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(tex_dir, "table_kfour.tex"), "w") as f:
        f.write("\n".join(t) + "\n")
    t = ["% Generated by analysis/p3_development.py. Do not edit.",
         "\\begin{tabular}{@{}rrrrrrrrrr@{}}", "\\toprule",
         " & & & \\multicolumn{2}{c}{probe on wrong} & \\multicolumn{3}{c}{wrong answers repaired} & & \\\\",
         "\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}",
         "step & layer & wrong & selected & final & gold & probe & random & model & steered \\\\", "\\midrule"]
    for r in S["1.4B"]["rows"]:
        step = f"{r['step']:,}".replace(",", "{,}")
        t.append(f"{step} & {r['layer']} & {r['n_wrong']} & {f2(r['probe_wrong'])} & {f2(r['probe_wrong_final'])} & "
                 f"{f2(r['gold_w_0.5'])} & {f2(r['probe_w_0.5'])} & {f2(r['random_0.5'])} & {f2(r['acc'])} & "
                 f"{f2(r['gated_acc'])} \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(tex_dir, "table_steer.tex"), "w") as f:
        f.write("\n".join(t) + "\n")
    t = ["% Generated by analysis/p3_development.py. Do not edit.",
         "\\begin{tabular}{@{}rrrrrr@{}}", "\\toprule",
         " & & \\multicolumn{2}{c}{probe right} & \\multicolumn{2}{c}{probe wrong} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
         "step & wrong answers & output low & output high & output low & output high \\\\", "\\midrule"]
    for r in P:
        g = r["groups"]
        t.append(f"{r['step']:,} & {r['n_wrong']} & {g['probe_right_output_low']} & {g['probe_right_output_high']} & "
                 f"{g['probe_wrong_output_low']} & {g['probe_wrong_output_high']} \\\\".replace(",", "{,}"))
    t += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(tex_dir, "table_errors.tex"), "w") as f:
        f.write("\n".join(t) + "\n")
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", default=None, help="write numbers_development.tex and table_kfour.tex here")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--from-summary", help="emit TeX from a saved summary without recomputing statistics")
    a = ap.parse_args()
    if a.from_summary:
        if not a.emit:
            ap.error("--from-summary requires --emit")
        emit_tex(json.load(open(a.from_summary)), a.emit)
        print("emitted saved development summary to", a.emit)
        return
    res = {"selftest": selftest()}
    print("[selftest]", res["selftest"], flush=True)
    if a.selftest:
        return
    res["ladder"] = ladder()
    res["steer"] = steer()
    res["k4"] = k4()
    res["posterior"] = posterior()
    for lab, e in res["ladder"].items():
        print(f"[ladder] {lab}: n={e['n_checkpoints']} dropped={e['dropped_duplicates']} "
              f"final={e.get('final')} max={e.get('max')} rho={e.get('rho')} p={e.get('p')} {e.get('excluded', '')}")
    for lab, e in res["steer"].items():
        s = e["summary"]
        print(f"[steer] {lab}: final acc={s['final']['acc']} probe_wrong={s['final']['probe_wrong']} "
              f"gold_late={s['gold_w_late_range']} probe first/8000/final={s['probe_w_first']}/{s['probe_w_8000']}/"
              f"{s['probe_w_final']} ratio={s['ratio_8000']}->{s['ratio_final']} random_max={s['random_max_05']}")
    print("[k4]", res["k4"]["summary"], flush=True)
    out = os.path.join(RES, "p3_development.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("wrote", out)
    if a.emit:
        M = emit_tex(res, a.emit)
        print(f"wrote {len(M)} macros to", os.path.join(a.emit, "numbers_development.tex"))


if __name__ == "__main__":
    main()
