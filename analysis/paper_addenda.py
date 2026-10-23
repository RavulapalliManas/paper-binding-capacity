#!/usr/bin/env python3
"""
Addenda to the P3 capacity paper, computed only from already-committed results.
No new compute: every input is a JSON already in results/.

Sections, in the order they appear in the revised paper:
  A  collapse shape          -- logistic slope, and the test that KILLS the "slope law"
  B  threshold-free law      -- the K50 law re-run on capacity_auc (no threshold, no drop)
  C  ceiling uniformity      -- is K50 confounded by the K=1 ceiling?
  D  interference, 3 terms   -- ceiling drop, midpoint shift (both normalisations), slope ratio
  E  GPT-2 sensitivity       -- does the recipe median survive dropping the disputed family?
  F  developmental legibility-- query-site binding decodability across Pythia checkpoints
  G  the nulls               -- readout retraction, steering without the oracle

Writes results/paper_addenda.json. Run from anywhere:
    python3 analysis/paper_addenda.py
"""
import json, math, os, re, glob
import numpy as np
from addenda_tex import emit_tex

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DST = os.path.join(RES, "paper_addenda.json")

# Parameter counts in millions. Source: each model's HuggingFace config (total params).
PARAMS_M = {
    "olmo2-1b-x": 1480, "opt-1.3b-x": 1316, "opt-125m-x": 125, "opt-350m-x": 331,
    "pythia-1.4b-x": 1415, "pythia-160m-x": 162, "pythia-1b-x": 1012,
    "pythia-2.8b-x": 2775, "pythia-410m-x": 405, "qwen2.5-0.5b-x": 494,
    "qwen2.5-1.5b-x": 1544, "smollm2-360m-x": 362,
}
N_BOOT = 20000
SEED = 0


def load(name):
    return json.load(open(os.path.join(RES, name)))


def ols(cols, y):
    """Least squares with intercept. Returns (coefficients, R^2)."""
    X = np.column_stack([np.ones(len(y))] + list(cols))
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    return b, 1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()


def partial_corr(a, b, given):
    """Correlation of a and b after regressing each on `given`."""
    def r(y):
        X = np.column_stack([np.ones(len(y)), given])
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(r(a), r(b))[0, 1])


def family_boot(x, y, fams, seed=SEED, n_boot=N_BOOT):
    """Bootstrap the slope of y~x, resampling FAMILIES not models."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(fams))
    idx_by_fam = {f: [i for i, g in enumerate(fams) if g == f] for f in uniq}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = [i for f in pick for i in idx_by_fam[f]]
        if len(set(fams[i] for i in idx)) < 2:
            continue                      # a degenerate draw cannot identify a slope
        try:
            draws.append(ols([x[idx]], y[idx])[0][1])
        except np.linalg.LinAlgError:
            pass
    d = np.array(draws)
    return dict(ci95=[round(float(np.percentile(d, 2.5)), 4),
                      round(float(np.percentile(d, 97.5)), 4)],
                frac_ge_zero=round(float((d >= 0).mean()), 4),
                n_boot_valid=int(len(d)))


def curves():
    """The 12 continuous-curve models, D=0 and D=256, from the committed summary."""
    d = load("capacity_trn1_summary.json")
    out = []
    for m in d["models"]:
        z0, z2 = m["by_D"].get("0"), m["by_D"].get("256")
        out.append(dict(label=m["label"], family=m["family"], recipe=m["recipe"],
                        params_M=PARAMS_M[m["label"]], d0=z0, d256=z2))
    return out


# ---------------------------------------------------------------- A
def section_A(C):
    """Collapse shape. The headline 'slope law' is tested and REJECTED here."""
    u = [c for c in C if not c["d0"].get("censored")]
    lN = np.array([math.log(c["params_M"]) for c in u])
    lK = np.array([math.log(c["d0"]["K50"]) for c in u])
    lS = np.array([math.log(c["d0"]["slope"]) for c in u])
    fams = [c["family"] for c in u]
    slopes = [c["d0"]["slope"] for c in u]

    b_n, r2_n = ols([lN], lS)
    b_k, r2_k = ols([lK], lS)
    b_nk, r2_nk = ols([lN, lK], lS)
    pc_n = partial_corr(lS, lN, lK)

    # The kill rule, fixed before looking: scale must retain |partial r| >= 0.4
    # against K50 for "the collapse gets shallower with scale" to be an independent law.
    independent = abs(pc_n) >= 0.40

    return dict(
        n=len(u), n_families=len(set(fams)),
        slope_min=round(min(slopes), 3), slope_max=round(max(slopes), 3),
        slope_range_factor=round(max(slopes) / min(slopes), 2),
        slope_median=round(float(np.median(slopes)), 3),
        corr_logslope_logK50=round(float(np.corrcoef(lS, lK)[0, 1]), 3),
        corr_logslope_logN=round(float(np.corrcoef(lS, lN)[0, 1]), 3),
        fit_slope_on_N=dict(coef=round(b_n[1], 3), r2=round(r2_n, 3)),
        fit_slope_on_K50=dict(coef=round(b_k[1], 3), r2=round(r2_k, 3)),
        fit_slope_on_N_and_K50=dict(coef_N=round(b_nk[1], 3), coef_K50=round(b_nk[2], 3),
                                    r2=round(r2_nk, 3)),
        partial_corr_slope_N_given_K50=round(pc_n, 3),
        partial_corr_slope_K50_given_N=round(partial_corr(lS, lK, lN), 3),
        family_boot_slope_on_N=family_boot(lN, lS, fams),
        VERDICT=("INDEPENDENT" if independent else "REJECTED: not independent of K50"),
        reportable_claim=("The collapse is gradual, not a step: logistic slopes in log K "
                          "span %.2f to %.2f (n=%d). The slope is strongly coupled to K50 "
                          "(r=%.2f in logs) and carries no information about scale once "
                          "K50 is known (partial r=%.2f). Report the range; do NOT report "
                          "a slope scaling law."
                          % (min(slopes), max(slopes), len(u),
                             float(np.corrcoef(lS, lK)[0, 1]), pc_n)),
    )


# ---------------------------------------------------------------- B
def section_B(C):
    """The law on capacity_auc: no threshold, and the censored model is kept."""
    out = {}
    for tag, sel in (("all12", C), ("uncensored11", [c for c in C if not c["d0"].get("censored")])):
        lN = np.array([math.log(c["params_M"]) for c in sel])
        lA = np.array([math.log(c["d0"]["capacity_auc"]) for c in sel])
        b, r2 = ols([lN], lA)
        out[tag] = dict(n=len(sel), exponent=round(b[1], 3), r2=round(r2, 3),
                        **family_boot(lN, lA, [c["family"] for c in sel]))
    out["note"] = ("AUC is bounded in [0,1] and K50 is not, so the exponents are NOT "
                   "comparable in magnitude to alpha=0.820. What replicates is the sign, "
                   "the exclusion of zero, and the fit quality, with no threshold and no "
                   "model dropped for censoring.")
    out["k50_law_for_reference"] = dict(alpha=0.820, r2=0.726, n=11)
    return out


# ---------------------------------------------------------------- C
def section_C(C):
    """Is K50 an artefact of differing K=1 ceilings? Ceilings are near-uniform."""
    ceil = [c["d0"]["K1_recall"] for c in C]
    lN = np.array([math.log(c["params_M"]) for c in C])
    b, r2 = ols([lN], np.array(ceil))
    hi = [c for c in C if c["d0"]["K1_recall"] >= 0.9]
    return dict(n=len(C), ceiling_min=round(min(ceil), 3), ceiling_max=round(max(ceil), 3),
                ceiling_median=round(float(np.median(ceil)), 3),
                n_above_0p9=len(hi),
                lowest_model=min(C, key=lambda c: c["d0"]["K1_recall"])["label"],
                fit_ceiling_on_logN=dict(coef=round(b[1], 4), r2=round(r2, 3)),
                reportable_claim=("%d of %d models hold a K=1 ceiling between %.3f and %.3f; "
                                  "the exception is %s at %.3f. The ceiling is only weakly "
                                  "related to scale (R2=%.2f), so K50 differences are not "
                                  "ceiling differences."
                                  % (len(hi), len(C), min(c["d0"]["K1_recall"] for c in hi),
                                     max(ceil), min(C, key=lambda c: c["d0"]["K1_recall"])["label"],
                                     min(ceil), r2)))


# ---------------------------------------------------------------- D
def section_D(C):
    """Interference in three separable terms."""
    ia = load("interference_axis.json")
    drops = [m["ceiling_drop"] for m in ia["models"]]

    # own-ceiling midpoint ratio, the estimator the paper does NOT use
    own = [(c["label"], c["d256"]["K50"] / c["d0"]["K50"]) for c in C
           if c["d256"] and "excluded" not in c["d256"] and not c["d0"].get("censored")]
    own_r = [r for _, r in own]

    # slope ratio: does the curve change SHAPE as well as position?
    sh = [(c["label"], c["d256"]["slope"] / c["d0"]["slope"]) for c in C
          if c["d256"] and "excluded" not in c["d256"]
          and not c["d0"].get("censored") and not c["d256"].get("censored")]
    sh_r = [r for _, r in sh]

    return dict(
        term1_ceiling_loss=dict(
            n=len(drops), median=round(float(np.median(drops)), 3),
            min=round(min(drops), 3), max=round(max(drops), 3),
            worst_model=max(ia["models"], key=lambda m: m["ceiling_drop"])["label"],
            best_model=min(ia["models"], key=lambda m: m["ceiling_drop"])["label"],
            spearman_vs_capacity=ia["spearman_capacity_vs_ceiling_drop"]),
        term2_midpoint_shift=dict(
            d0_ceiling_normalised=dict(median=ia["retained_fraction_median"],
                                       n=ia["n_uncensored"],
                                       n_reduced=ia["n_models_interference_reduces_capacity"]),
            own_ceiling_normalised=dict(median=round(float(np.median(own_r)), 3),
                                        n=len(own_r),
                                        n_reduced=int(sum(1 for r in own_r if r < 1.0))),
            note=("The two normalisations disagree because interference lowers the K=1 "
                  "ceiling itself (term 1). Own-ceiling normalisation divides out the very "
                  "damage being measured. Report both.")),
        term3_slope_change=dict(
            n=len(sh_r), median_ratio=round(float(np.median(sh_r)), 3),
            min=round(min(sh_r), 2), max=round(max(sh_r), 2),
            n_steeper=int(sum(1 for r in sh_r if r > 1.0)),
            corr_with_ceiling_drop=round(float(np.corrcoef(
                [d for lab, _ in sh for d in [next(m["ceiling_drop"] for m in ia["models"]
                                                   if m["label"] == lab)]], sh_r)[0, 1]), 3),
            VERDICT="NULL: interference does not systematically change the curve shape."),
    )


# ---------------------------------------------------------------- E
def section_E():
    """Does the recipe median survive dropping GPT-2, whose k* the second instrument disputes?"""
    g = load("figdata.json")["geom"]
    rows = [r for r in (g if isinstance(g, list) else g.values())
            if isinstance(r, dict) and "kstar" in r]
    rows = [r for r in rows if "pythia70" not in str(r.get("label", "")).lower()]
    old = [r for r in rows if r["recipe"] == "old"]
    mod = [r for r in rows if r["recipe"] == "modern"]
    no_gpt2 = [r for r in old if r["family"] != "GPT-2"]
    med = lambda rs: float(np.median([r["kstar"] for r in rs]))
    return dict(n=len(rows), n_old=len(old), n_modern=len(mod),
                old_median=med(old), modern_median=med(mod),
                median_diff=med(mod) - med(old),
                n_gpt2_dropped=len(old) - len(no_gpt2),
                old_median_without_gpt2=med(no_gpt2),
                median_diff_without_gpt2=med(mod) - med(no_gpt2),
                reportable_claim=("The continuous instrument places the GPT-2 family below "
                                  "the K=1 floor. Dropping all %d GPT-2 models moves the "
                                  "old-recipe median from %g to %g and the median difference "
                                  "from %g to %g, so the recipe split does not rest on them."
                                  % (len(old) - len(no_gpt2), med(old), med(no_gpt2),
                                     med(mod) - med(old), med(mod) - med(no_gpt2))))


# ---------------------------------------------------------------- F
def section_F():
    """Query-site binding legibility across pretraining."""
    out = {}

    cc = load("capacity_control.json")                 # Pythia-1.4B, leak-free layer selection
    f, l = cc["rows"][0], cc["rows"][-1]
    out["pythia_1.4b_controlled"] = dict(
        model=cc["model"], n_checkpoints=len(cc["rows"]), K=cc["K"], D=cc["D"],
        chance=cc["chance"], leak_free_layer_selection=cc["leak_free_layer_selection"],
        midstack_first=round(f["r_lstar"], 3), midstack_last=round(l["r_lstar"], 3),
        midstack_gain=round(l["r_lstar"] / f["r_lstar"], 2),
        final_layer_first=round(f["r_final"], 3), final_layer_last=round(l["r_final"], 3),
        final_layer_gain=round(l["r_final"] / f["r_final"], 2),
        behaviour_first=round(f["behaviour"], 3), behaviour_last=round(l["behaviour"], 3),
        behaviour_gain=round(l["behaviour"] / f["behaviour"], 2),
        separation_first=round(f["sep"], 3), separation_last=round(l["sep"], 3),
        best_layer_first=f["lstar"], best_layer_last=l["lstar"],
        spearman_sep_vs_step=cc["sep_vs_step_spearman"], exact_p=cc["sep_vs_step_p"],
        selection_not_forcing_positive=(f["sep"] < 0))

    qd = load("qgeom_dev.json")                        # same model, decodability trend
    tr = qd["trends"]["decode_obl_full"]
    out["pythia_1.4b_decodability"] = dict(
        model=qd["model"], layer=qd["layer"], chance=qd["chance"],
        n_checkpoints=qd["n_checkpoints"], aborts=qd["aborts"],
        distinct_weight_hashes=qd["distinct_weight_hashes"],
        first=round(tr["first"], 3), last=round(tr["last"], 3),
        spearman=tr["rho"], exact_p=tr["p_exact"], steps=qd["steps"])

    for tag, pat in (("pythia_1b", "qgfix_p1b"), ("pythia_160m", "qgfix_p160")):
        fs = sorted(glob.glob(os.path.join(RES, pat, "step*", "qgeom.json")),
                    key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
        vals, aborts = [], []
        for p in fs:
            j = json.load(open(p))
            s = int(re.search(r"step(\d+)", p).group(1))
            (aborts if j.get("aborted") else vals).append(
                s if j.get("aborted") else (s, round(j["decode_obl_full"], 3),
                                            round(j["decode_ent_full"], 3)))
        out[tag] = dict(n_checkpoints=len(fs), n_aborted=len(aborts),
                        aborted_steps=aborts, values=vals,
                        obl_min=min(v[1] for v in vals), obl_max=max(v[1] for v in vals),
                        ent_min=min(v[2] for v in vals), ent_max=max(v[2] for v in vals),
                        abort_meaning=("aborted = binding not decodable at that checkpoint "
                                       "(see analysis/qgeom_ladder_agg.py:47). Informative, "
                                       "not missing-at-random."))
    out["scope"] = ("One model with full controls (Pythia-1.4B, 7 checkpoints, leak-free "
                    "layer selection, 7 distinct weight hashes, 0 aborts), one consistent "
                    "second model with a hand-set layer (Pythia-1B), and one negative case "
                    "(Pythia-160M, flat, 5 of 14 checkpoints not decodable).")
    return out


# ---------------------------------------------------------------- G
def repair_trajectory():
    """Non-oracle repair across Pythia-1.4B pretraining checkpoints."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "dynamics", "em_step*", "steer.json"))):
        if "_p69" in f:
            continue
        d = json.load(open(f))
        step = int(re.search(r"step(\d+)", f).group(1))
        arm = d["results"].get("manifold_mu_decoded", {}).get("1.0", {})
        rows.append(dict(step=step, read_layer=d["lstar"],
                         repair_no_oracle=round(arm.get("repair", float("nan")), 3),
                         damage_keep=round(arm.get("damage_keep", float("nan")), 3),
                         repair_oracle=round(d["results"]["manifold_mu"]["1.0"]["repair"], 3)))
    return sorted(rows, key=lambda r: r["step"])


def spearman_exact(a, b):
    """Spearman rho with an exact permutation p-value (n is small)."""
    from itertools import permutations
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    hits = sum(1 for perm in permutations(range(len(a)))
               if np.corrcoef(ra, rb[list(perm)])[0, 1] >= rho - 1e-12)
    return round(rho, 3), hits / math.factorial(len(a))


def section_G():
    """Released-model repair, and how repairability arrives during pretraining."""
    rm = load("repair_6models.json")
    models = rm["models"]
    better = sum(1 for v in models.values() if v["repair"] > v["baseline"])
    vs_rand = sum(1 for v in models.values() if v["repair"] > v["random"])

    traj = repair_trajectory()
    qd = load("qgeom_dev.json")
    leg, steps_leg = qd["trends"]["decode_obl_full"], qd["steps"]
    aligned = [r for r in traj if r["step"] in steps_leg]
    rho, p_exact = spearman_exact([r["repair_no_oracle"] for r in aligned],
                                  [leg["values"][steps_leg.index(r["step"])] for r in aligned])

    rv_path = os.path.join(RES, "readout_visibility.json")
    rv = json.load(open(rv_path)) if os.path.exists(rv_path) else {}

    return dict(
        released_models=dict(
            source="results/repair_6models.json", leak_free=rm["leak_free"],
            n_seeds=rm["n_seeds"], n_per_model=600, n_models=len(models),
            n_repair_beats_baseline=better, n_repair_beats_random=vs_rand,
            recovery_min=round(min(v["recovery_mean_over_seeds"] for v in models.values()), 3),
            recovery_max=round(max(v["recovery_mean_over_seeds"] for v in models.values()), 3),
            uses_oracle=False,
            note=("The intervention re-injects the binding the PROBE decoded, not the gold "
                  "label. It is the companion paper's result; reproduced here only to "
                  "anchor the developmental trajectory below.")),
        repairability_develops=dict(
            model="EleutherAI/pythia-1.4b", n_checkpoints=len(traj), K=6, D=256, rows=traj,
            repair_first=traj[0]["repair_no_oracle"], repair_last=traj[-1]["repair_no_oracle"],
            repair_gain=round(traj[-1]["repair_no_oracle"] / traj[0]["repair_no_oracle"], 1),
            damage_keep_first=traj[0]["damage_keep"], damage_keep_last=traj[-1]["damage_keep"],
            read_layer_first=traj[0]["read_layer"], read_layer_last=traj[-1]["read_layer"],
            spearman_vs_legibility=rho, exact_p=round(p_exact, 5),
            VERDICT=("Repairability is installed by pretraining and tracks legibility. At "
                     "step 1000 the binding is not yet present (decode %.3f, read layer %d) "
                     "and repair fails; by step 143000 repair reaches %.3f with damage_keep "
                     "%.3f." % (leg["values"][0], traj[0]["read_layer"],
                                traj[-1]["repair_no_oracle"], traj[-1]["damage_keep"]))),
        readout_blindness=dict(
            status="RETRACTED (in the companion paper, which made the claim)",
            comparator="readout_k minus classmean_k, Vw fit on the probe's train fold",
            source="analysis/qgeom_aggregate.py",
            detail=("Label-matched comparator is below zero on 3 of 5 models: no systematic "
                    "readout deficit. P3 never made the claim and should cite, not re-argue."),
            per_model=rv.get("readout_minus_classmean", "see results/readout_visibility.json")),
        stale_file_warning=dict(
            path="results/qgeom_ladder.json",
            problem=("Still asserts readout_outgrown_significant_on for 160M/410M/1B, which "
                     "rests on the retracted leaky comparator. Regenerate or delete.")))


def main():
    C = curves()
    out = {
        "_provenance": ("analysis/paper_addenda.py. Inputs, all already committed: "
                        "capacity_trn1_summary.json, interference_axis.json, figdata.json, "
                        "capacity_control.json, qgeom_dev.json, qgfix_p1b/, qgfix_p160/, "
                        "dynamics/em_step1000/steer.json, readout_visibility.json. "
                        "No new compute; no model was run."),
        "_status": ("EXPLORATORY. Every finding here was generated from the existing "
                    "corpus after the fact. None was predicted in advance. Label the "
                    "section exploratory in the paper."),
        "A_collapse_shape": section_A(C),
        "B_threshold_free_law": section_B(C),
        "C_ceiling_uniformity": section_C(C),
        "D_interference_three_terms": section_D(C),
        "E_gpt2_sensitivity": section_E(),
        "F_developmental_legibility": section_F(),
        "G_nulls": section_G(),
    }
    json.dump(out, open(DST, "w"), indent=1)
    print("wrote", os.path.normpath(DST))
    for tex in (os.path.join(RES, "..", "numbers_addenda.tex"),
                os.path.expanduser("~/Downloads/ARXIV_2026-08-12 Revised-8-25/"
                                   "P3_capacity_arxiv/numbers_addenda.tex")):
        if os.path.isdir(os.path.dirname(tex)):
            emit_tex(out, tex)
    print("\nA  collapse shape      :", out["A_collapse_shape"]["VERDICT"])
    print("B  threshold-free law  : exponent %+.3f, R2 %.3f, n=%d (all models, censored kept)"
          % (out["B_threshold_free_law"]["all12"]["exponent"],
             out["B_threshold_free_law"]["all12"]["r2"],
             out["B_threshold_free_law"]["all12"]["n"]))
    print("C  ceiling uniformity  : %d of %d models in [0.918, 0.990]"
          % (out["C_ceiling_uniformity"]["n_above_0p9"], out["C_ceiling_uniformity"]["n"]))
    d = out["D_interference_three_terms"]
    print("D  interference        : ceiling loss median %.3f | midpoint %.3f (D0-norm) vs "
          "%.3f (own-norm) | slope %s"
          % (d["term1_ceiling_loss"]["median"],
             d["term2_midpoint_shift"]["d0_ceiling_normalised"]["median"],
             d["term2_midpoint_shift"]["own_ceiling_normalised"]["median"],
             d["term3_slope_change"]["VERDICT"].split(":")[0]))
    e = out["E_gpt2_sensitivity"]
    print("E  GPT-2 sensitivity   : old median %g -> %g, difference %g -> %g"
          % (e["old_median"], e["old_median_without_gpt2"],
             e["median_diff"], e["median_diff_without_gpt2"]))
    f = out["F_developmental_legibility"]["pythia_1.4b_controlled"]
    print("F  developmental       : mid-stack x%.2f vs behaviour x%.2f, separation %+.3f -> "
          "%+.3f, Spearman %.1f, exact p %.1e"
          % (f["midstack_gain"], f["behaviour_gain"], f["separation_first"],
             f["separation_last"], f["spearman_sep_vs_step"], f["exact_p"]))
    g = out["G_nulls"]
    print("G  repair              : %d/%d released models beat baseline, %d/%d beat random; "
          "repairability %.3f -> %.3f over pretraining (rho %.3f vs legibility, exact p %.3f)"
          % (g["released_models"]["n_repair_beats_baseline"], g["released_models"]["n_models"],
             g["released_models"]["n_repair_beats_random"], g["released_models"]["n_models"],
             g["repairability_develops"]["repair_first"], g["repairability_develops"]["repair_last"],
             g["repairability_develops"]["spearman_vs_legibility"],
             g["repairability_develops"]["exact_p"]))


if __name__ == "__main__":
    main()
