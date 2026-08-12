"""Emit paper/numbers.tex: every headline number, computed from committed data, as a LaTeX macro.

The papers then write \KstarRangeFactor instead of typing "8". A number that only exists as a literal in prose
can drift from the data with nothing to catch it, and in this project it has done so ten times -- including a
"12x" that I introduced myself while "correcting" a correct 8x.

Run:  python analysis/make_numbers.py        (writes paper/numbers.tex)
Check: python analysis/verify.py             (fails if numbers.tex is stale, or a guard trips)
"""
from __future__ import annotations
import json, os, re, sys
from math import comb

import numpy as np


def _sci(p):
    """LaTeX scientific notation for a p-value, e.g. 2.3\\times10^{-7}."""
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def _fisher2(a, b, c, d):
    """Exact two-sided Fisher p for the 2x2 table [[a, b], [c, d]] (hypergeometric sum)."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    def pr(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p0 = pr(a) * (1 + 1e-12)
    return sum(pr(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if pr(x) <= p0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_stats import sign_test, spearman_exact  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "numbers.tex")

# results/leakfree_rigor_16models.txt is retired: a single run whose configuration died with a preempted
# box. Nothing in this file reads it any more. See analysis/rigor_seeds.py and AUDIT.md sec. 2.5b.


def load(name):
    return json.load(open(os.path.join(RES, name)))


def steer_rows():
    rows = {}
    base = os.path.join(RES, "dynamics")
    for d in sorted(os.listdir(base)):
        if not d.startswith("steer_"):
            continue
        f = os.path.join(base, d, "steer.json")
        if not os.path.exists(f):
            continue
        g = json.load(open(f))["results"].get("gated_all", {}).get("0.5")
        if g:
            rows[d] = (g["base"], g["acc"])
    return rows


def build():
    N = {}          # macro -> (value string, source, human note)

    # ---------------- capacity (capacity_stats.json)
    cs = load("capacity_stats.json"); st = cs["stats"]
    N["KstarMin"] = (f"{cs['kstar_min']}", "capacity_stats.json", "min k* among measurable models")
    N["KstarMax"] = (f"{cs['kstar_max']}", "capacity_stats.json", "censored ceiling")
    N["KstarRangeFactor"] = (f"{cs['kstar_range_factor']:.0f}", "capacity_stats.json", "8x, NOT 12x")
    N["NModels"] = (f"{cs['n']}", "capacity_stats.json", "after excluding pythia-70m")
    N["NFamilies"] = (f"{cs['n_families']}", "capacity_stats.json", "")
    N["KstarOldMed"] = (f"{st['kstar']['old_median']:.0f}", "capacity_stats.json", "")
    N["KstarModMed"] = (f"{st['kstar']['modern_median']:.0f}", "capacity_stats.json", "")
    lo, hi = st["kstar"]["family_ci"]
    N["KstarFamCI"] = (f"[{lo:+.0f},\\,{hi:+.0f}]", "capacity_stats.json", "family-clustered")
    N["KstarFamP"] = (f"{st['kstar']['family_p']:.3f}", "capacity_stats.json", "")
    N["PackFamP"] = (f"{st['pack']['family_p']:.3f}", "capacity_stats.json", "")
    dlo, dhi = st["d_bind"]["family_ci"]
    N["DdimFamCI"] = (f"[{dlo:+.3f},\\,{dhi:+.2f}]", "capacity_stats.json", "touches zero -- does NOT survive")
    olo, ohi = st["offdiag"]["family_ci"]
    N["OffdiagFamCI"] = (f"[{olo:+.2f},\\,{ohi:+.3f}]", "capacity_stats.json", "touches zero -- does NOT survive")
    N["DbindFamP"] = (f"{st['d_bind']['family_p']:.3f}", "capacity_stats.json", "does NOT survive")
    N["OffdiagFamP"] = (f"{st['offdiag']['family_p']:.3f}", "capacity_stats.json", "does NOT survive")
    # ---- p3 macro-ization additions (2026-08-07) ----
    # p3 prints the model-level Mann-Whitney beside every family-clustered test on purpose: it is
    # the number a reader recomputes from the released table before the clustering is applied.
    N["KstarModelP"] = (f"{st['kstar']['model_p']:.1e}".replace("e-04", "\\times 10^{-4}"),
                        "capacity_stats.json", "model-level Mann-Whitney -- the UNclustered p")
    N["NOldModels"] = (f"{st['kstar']['n_old']}", "capacity_stats.json", "old-recipe models AFTER excluding pythia-70m")
    N["NModernModels"] = (f"{st['kstar']['n_modern']}", "capacity_stats.json", "modern-recipe models")
    N["NCensored"] = (f"{len(cs['censored'])}", "capacity_stats.json", "right-censored at the entity-pool ceiling")
    N["PackOldMed"] = (f"{st['pack']['old_median']:.3f}", "capacity_stats.json", "")
    N["PackModMed"] = (f"{st['pack']['modern_median']:.3f}", "capacity_stats.json", "")
    _plo, _phi = st["pack"]["family_ci"]
    N["PackFamCI"] = (f"[{_plo:+.3f},\\,{_phi:+.3f}]", "capacity_stats.json", "family-clustered -- SURVIVES, but tautological")
    N["DdimOldMed"] = (f"{st['d_bind']['old_median']:.2f}", "capacity_stats.json", "")
    N["DdimModMed"] = (f"{st['d_bind']['modern_median']:.2f}", "capacity_stats.json", "")
    N["OffdiagOldMed"] = (f"{st['offdiag']['old_median']:.3f}", "capacity_stats.json", "")
    N["OffdiagModMed"] = (f"{st['offdiag']['modern_median']:.3f}", "capacity_stats.json", "")
    _olo2, _ohi2 = st["offdiag"]["family_ci"]
    N["OffdiagFamCITab"] = (f"[{_olo2:+.3f},\\,{_ohi2:+.3f}]", "capacity_stats.json", "3 dp, p3 table column")
    # The Qwen ladder is the within-recipe scale effect, and it is NOT monotone (3B censored, 7B below it).
    N["QwenLadder"] = ("\\to".join(str(v) for v in cs["qwen_ladder"].values()),
                       "capacity_stats.json", "0.5B/1.5B/3B/7B -- NOT monotone, reported unsmoothed")

    # ---------------- capacity control (capacity_control.json)
    cc = load("capacity_control.json")
    N["RfinalFirst"] = (f"{cc['r_final_first']:.3f}", "capacity_control.json", "")
    N["RfinalLast"] = (f"{cc['r_final_last']:.3f}", "capacity_control.json", "")
    N["RfinalChanceX"] = (f"{cc['r_final_last']/cc['chance']:.1f}", "capacity_control.json", "x 1/|V| (25-way)")
    N["Dblock"] = (f"{cc['D']}", "capacity_control.json", "distractor tokens between declarations and query")
    # HONESTY: 9.6x is against the LOOSE 1/|V| vocabulary baseline. Against the 1/K present-set baseline the
    # paper claims to report throughout, the same final-layer decode is only 2.3x. Report both.
    N["RfinalPresent"] = (f"{1/cc['K']:.3f}", "capacity_control.json", "1/K present-set baseline, K=6")
    N["RfinalPresentX"] = (f"{cc['r_final_last']/(1/cc['K']):.1f}", "capacity_control.json", "x 1/K present-set")
    N["Chance"] = (f"{cc['chance']:.3f}", "capacity_control.json", "25-way")
    N["SepLast"] = (f"{cc['rows'][-1]['sep']:.3f}", "capacity_control.json", "")
    steps = [r["step"] for r in cc["rows"]]; seps = [r["sep"] for r in cc["rows"]]
    rho, pe, n, _ = spearman_exact(steps, seps)
    N["SepRho"] = (f"{rho:.3f}", "capacity_control.json", "")
    N["SepPexact"] = (f"{pe:.1e}".replace("e-04", "\\times 10^{-4}"), "exact_stats", f"n={n}; floor 2/{n}!")
    N["SepN"] = (f"{n}", "capacity_control.json", "")

    # ---------------- gap + certificate (rigor_seeds.json -- 16 models x 3 seeds)
    # Supersedes results/leakfree_rigor_16models.txt, a single run whose configuration died with a preempted
    # box. The COUNTS below are unstable across seeds and the paper must not lead with them; what carries the
    # claims is GapMean/GapCI and CertMean/CertCI, bootstrapped clustered on model.
    P = load("rigor_seeds.json")["paper"]
    RS = "rigor_seeds.json"
    PRES = P["present_baseline"]
    N["GapN"] = (f"{P['gap_n']}", RS, "gap-measurable; 2 censored by ceiling")
    N["CertN"] = (f"{P['cert_n']}", RS, "all models -- cert is an AUROC over all test trials")
    N["CensoredN"] = (f"{P['cert_n'] - P['gap_n']}", RS, "too few failures to measure pAcc|wrong")
    N["Present"] = (f"{PRES:.3f}", RS, "1/K, K=8")
    N["GapMedian"] = (f"{P['pa_median']:.2f}", RS, "median pAcc|wrong")
    N["GapMax"] = (f"{P['pa_max']:.2f}", RS, "")
    N["GapMaxModel"] = (P["pa_argmax"], RS, "was Pythia-1.4B under the retired table")
    N["GapPosSeeds"] = (", ".join(str(k) for k in P["gap_pos_per_seed"]), RS, "UNSTABLE across seeds")
    N["GapPosMin"] = (f"{min(P['gap_pos_per_seed'])}", RS, "")
    N["GapPosMax"] = (f"{max(P['gap_pos_per_seed'])}", RS, "")
    N["GapSignP"] = (f"{sign_test(min(P['gap_pos_per_seed']), P['gap_n']):.3f}", "exact_stats", "worst seed")
    N["GapMean"] = (f"{P['gap_mean']:+.3f}", RS, "clustered on model -- THIS carries the claim")
    N["GapCI"] = (f"[{P['gap_ci'][0]:+.3f}, {P['gap_ci'][1]:+.3f}]", RS, "")
    N["GapPos"] = (f"{P['gap_pos_seedmean']}", RS, "on seed-means, of GapN")
    N["GapPosP"] = (f"{P['gap_pos_seedmean_p']:.3f}", "exact_stats", "")
    N["CertPosSeeds"] = (", ".join(str(k) for k in P["cert_pos_per_seed"]), RS, "UNSTABLE across seeds")
    # Four rivals, each a seed-mean per model then a bootstrap clustered on model. The count is not the
    # evidence; the CI is. Predictive entropy is a STRONG baseline: the certificate does NOT clear it on 16.
    for macro, key in [("Cert", "self"), ("Ent", "entropy"), ("SelfCons", "selfcons"), ("RawProbe", "rawprobe")]:
        r = P["rivals"][key]
        note = "CI excludes 0" if r["ci_excludes_zero"] else "CI STRADDLES 0 -- no advantage"
        N[macro + "Mean"] = (f"{r['mean']:+.3f}", RS, note)
        N[macro + "CI"] = (f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]", RS, "")
        N[macro + "Pos"] = (f"{r['pos']}", RS, f"of {r['n']}")
        N[macro + "SignP"] = (f"{r['sign_p']:.3f}", "exact_stats", "")
    N["RawProbeNeg"] = (f"{P['rawprobe_neg']}", RS, "loses on; of CertN")
    N["CfMin"] = (f"{P['cf_min']:.2f}", RS, "")
    N["CfMax"] = (f"{P['cf_max']:.2f}", RS, "")
    N["NTrials"] = (f"{P['n_trials']}", RS, "")
    N["NSeeds"] = (f"{P['n_seeds']}", RS, "")
    N["CfPos"] = (f"{P['cf_pos']}", RS, f"of {P['cf_n']}, above {PRES}")
    _cfp = sign_test(P["cf_pos"], P["cf_n"])
    N["CfSignP"] = ((f"{_cfp:.1e}".replace("e-04", "\\times 10^{-4}") if _cfp < 1e-3 else f"{_cfp:.3f}"),
                    "exact_stats", "the one per-model tally that clears Bonferroni (reviewer-sim fix)")

    # ---------------- steering (dynamics/steer_*/steer.json)
    sr = steer_rows()
    d = np.array([acc - b for b, acc in sr.values()])
    N["SteerN"] = (f"{len(sr)}", "dynamics/steer_*/steer.json", "")
    N["SteerPos"] = (f"{int((d > 0).sum())}", "dynamics/steer_*/steer.json", "")
    N["SteerMean"] = (f"{d.mean():+.3f}", "dynamics/steer_*/steer.json", "")
    N["SteerSignP"] = (f"{sign_test(int((d > 0).sum()), len(sr)):.3f}", "exact_stats", "")

    # ---------------- prompt repair (repair_6models.json)
    rp = load("repair_6models.json")["models"]
    dr = np.array([v["repair"] - v["baseline"] for v in rp.values()])
    N["RepairN"] = (f"{len(rp)}", "repair_6models.json", "")
    N["RepairPos"] = (f"{int((dr > 0).sum())}", "repair_6models.json", "")
    N["RepairMean"] = (f"{dr.mean():+.2f}", "repair_6models.json", "")
    N["RepairMax"] = (f"{dr.max():+.2f}", "repair_6models.json", "")
    N["RepairSignP"] = (f"{sign_test(int((dr > 0).sum()), len(rp)):.3f}", "exact_stats", "count is WEAK")
    p14 = rp["Pythia-1.4B"]
    N["RandCtrlBase"] = (f"{p14['baseline']:.2f}", "repair_6models.json", "")
    N["RandCtrlAfter"] = (f"{p14['random']:.2f}", "repair_6models.json", "random injection HARMS")

    # ---------------- certificate-gated closed loop (repair_6models.json, "gated" column)
    # gate = the signed-coherence certificate below its own median (label-free; see
    # analysis/repair_rerun.py). The loop composes detect (certificate) + decode (probe) +
    # repair (re-injection) with no access to the correct binding anywhere in the pipeline.
    dg = np.array([v["gated"] - v["baseline"] for v in rp.values()])
    N["LoopN"] = (f"{len(rp)}", "repair_6models.json", "")
    N["LoopPos"] = (f"{int((dg > 0).sum())}", "repair_6models.json", "")
    N["LoopMean"] = (f"{dg.mean():+.2f}", "repair_6models.json", "certificate-gated, label-free")
    _fr = [g / r for g, r in zip(dg, dr) if r > 0]
    N["LoopHeadroomN"] = (f"{len(_fr)}", "repair_6models.json", "models where the ungated repair helps")
    N["LoopFracMin"] = (f"{100 * min(_fr):.0f}\\%", "repair_6models.json",
                        "min share of the ungated repair's recovery captured")
    N["LoopBeatsOracle"] = (f"{sum(1 for g, r in zip(dg, dr) if r > 0 and g >= r)}",
                            "repair_6models.json", "gated recovery >= the ungated repair's")
    _ceil = int(np.argmin(dr))
    N["LoopCeilGated"] = (f"{dg[_ceil]:+.2f}", "repair_6models.json", "ceiling model, gated loop")
    N["LoopCeilOracle"] = (f"{dr[_ceil]:+.2f}", "repair_6models.json", "ceiling model, ungated repair")

    # ---------------- repair seed sweep (repair_seeds.json)
    rs = load("repair_seeds.json")
    N["RepairSeeds"] = (f"{rs['n_seeds']}", "repair_seeds.json", "")
    N["RepairMeanSeeds"] = (f"{rs['mean_recovery']:+.3f}", "repair_seeds.json", "")
    lo, hi = rs["ci_seed_clustered"]
    N["RepairCI"] = (f"[{lo:+.3f},\\,{hi:+.3f}]", "repair_seeds.json", "seed-clustered, only 3 clusters")
    # Model-clustered CI over the same 18 runs: the model, not the seed, is the unit of
    # generalisation (arXiv release protocol, Blocker E -- the seed-clustered interval used the
    # wrong unit). Deterministic bootstrap, fixed seed, resampling models with seeds averaged.
    import numpy as _np_rc
    _rng = _np_rc.random.default_rng(20260807)
    _pm = [float(_np_rc.mean(v["recovery"])) for v in rs["per_model"].values()]
    _boot = [float(_np_rc.mean(_rng.choice(_pm, len(_pm), replace=True))) for _ in range(4000)]
    _mlo, _mhi = _np_rc.percentile(_boot, [2.5, 97.5])
    N["RepairCIModel"] = (f"[{_mlo:+.3f},\\,{_mhi:+.3f}]", "repair_seeds.json",
                          "model-clustered bootstrap (6 clusters), deterministic seed")
    # Reviewer-sim fix (2026-08-07): the 18/18 p=7.6e-6 treats seeds as independent, violating the
    # paper's own model-as-unit rule. The inference-grade statistic clusters on model: 6/6 models'
    # random-control shift is negative.
    _negm = sum(1 for v in rs["per_model"].values() if v["random_delta_mean"] < 0)
    N["RepairCtrlModelK"] = (f"{_negm}", "repair_seeds.json", "models whose mean random-control shift is negative")
    N["RepairCtrlModelN"] = (f"{len(rs['per_model'])}", "repair_seeds.json", "")
    N["RepairCtrlModelP"] = (f"{sign_test(_negm, len(rs['per_model'])):.3f}", "exact_stats",
                             "model-clustered sign test -- the generalization claim")
    rh = rs["repair_helps"]; rc = rs["random_control_harms"]
    N["RepairHelpsK"] = (f"{rh['k']}", "repair_seeds.json", "")
    N["RepairHelpsN"] = (f"{rh['n']}", "repair_seeds.json", "")
    N["RepairHelpsP"] = (f"{rh['sign_p']:.4f}", "exact_stats", "")
    N["RepairCtrlK"] = (f"{rc['k']}", "repair_seeds.json", "")
    N["RepairCtrlN"] = (f"{rc['n']}", "repair_seeds.json", "")
    N["RepairCtrlP"] = (f"{rc['sign_p']:.1e}".replace("e-06", "\\times 10^{-6}"), "exact_stats",
                        "the identification, 18/18")
    N["RepairCtrlShift"] = (f"{rc['mean_shift']:+.3f}", "repair_seeds.json", "")

    # ---------------- developmental accessibility (qgeom_dev.json), CORRECTED INSTRUMENT
    # chain-rule frame (*sd), train-only mu/sd, train-only Vw, per-trial distractors, 50-draw nulls,
    # probe C tuned per subspace. Two earlier retractions were themselves artifacts of the broken version.
    qd = load("qgeom_dev.json"); tr = qd["trends"]
    import numpy as _np

    def _fmt(pv):
        return (f"{pv:.1e}".replace("e-04", "\\times 10^{-4}").replace("e-03", "\\times 10^{-3}")
                if pv < 1e-2 else f"{pv:.2f}")

    for macro, key, note in [("Acc", "decode_obl_full", "monotone"),
                             ("RdAcc", "decode_readout_k", ""),
                             ("Cm", "decode_classmean_k", "the fair, label-matched comparator"),
                             ("RndNull", "decode_random_k", "the null rises too"),
                             ("AccGap", "gap", "full - readout_k; monotone on the corrected instrument"),
                             ("RdCm", "readout_minus_classmean", "the readout is OUTGROWN"),
                             ("Align", "alignment", "FLAT: the aim never improves")]:
        t = tr[key]
        N[macro + "First"] = (f"{t['first']:.3f}", "qgeom_dev.json", "")
        N[macro + "Last"] = (f"{t['last']:.3f}", "qgeom_dev.json", "")
        N[macro + "Rho"] = (f"{t['rho']:+.3f}", "qgeom_dev.json", "")
        N[macro + "P"] = (_fmt(t["p_exact"]), "exact_stats", note)
    N["AccCkpts"] = (f"{qd['n_checkpoints']}", "qgeom_dev.json", "")
    N["AccLayer"] = (f"{qd['layer']}", "qgeom_dev.json", "held fixed")
    _ks = qd["k_per_checkpoint"]; _nl = qd["alignment_nulls"]; _al = tr["alignment"]["values"]
    N["AccKMin"] = (f"{min(_ks)}", "qgeom_dev.json", "k is recomputed per checkpoint")
    N["AccKMax"] = (f"{max(_ks)}", "qgeom_dev.json", "")
    N["AlignNullMin"] = (f"{min(_nl):.4f}", "qgeom_dev.json", "k/d varies with k")
    N["AlignNullMax"] = (f"{max(_nl):.4f}", "qgeom_dev.json", "")
    _r = [a / n for a, n in zip(_al, _nl)]
    N["AlignRatioMin"] = (f"{min(_r):.1f}", "qgeom_dev.json", "")
    N["AlignRatioMax"] = (f"{max(_r):.1f}", "qgeom_dev.json", "")
    N["AlignRatioMean"] = (f"{_np.mean(_r):.0f}", "qgeom_dev.json", "NOT 14x; that was the leak")
    N["RdBelowCm"] = (f"{qd['readout_below_classmean_at']}", "qgeom_dev.json", "of AccCkpts")

    # ---------------- developmental sweep across model sizes (devsweep.json) -- retires "one model"
    _ds_path = os.path.join(RES, "devsweep.json")
    if os.path.exists(_ds_path):
        ds = load("devsweep.json")
        s14 = ds["sizes"].get("pythia-1.4b")
        if s14:                                    # the paper's reference model: replaces the hand-typed literals
            N["DevReprFirst"] = (f"{s14['repr_first']:.3f}", "devsweep.json", "1.4B representation clock, chance 0.040")
            N["DevReprLast"] = (f"{s14['repr_last']:.3f}", "devsweep.json", "was hand-typed 0.712")
            N["DevReadoutFirst"] = (f"{s14['readout_first']:.3f}", "devsweep.json", "1.4B readout clock, chance 1/6")
            N["DevReadoutLast"] = (f"{s14['readout_last']:.3f}", "devsweep.json", "was hand-typed 0.323")
            # xchance for the first->last sentence must pair with *_last (the endpoint), not the converged mean
            N["DevReprXchance"] = (f"{s14['repr_last'] / ds['probe_chance']:.0f}", "devsweep.json",
                                   "1.4B repr LAST in units of its chance")
            N["DevReadoutXchance"] = (f"{s14['readout_last'] / ds['readout_chance']:.1f}", "devsweep.json",
                                      "1.4B readout LAST in units of its chance")
        sc = ds.get("scaling")
        if sc:
            N["DevNSizes"] = (f"{sc['n_sizes']}", "devsweep.json", "160M/410M/1B/1.4B/6.9B (2.8B excluded, broken revisions)")
            N["DevReprXFirst"] = (f"{sc['repr_xchance'][0]:.1f}", "devsweep.json", "smallest model")
            N["DevReprXLast"] = (f"{sc['repr_xchance'][-1]:.1f}", "devsweep.json", "largest model")
            N["DevReadoutXFirst"] = (f"{sc['readout_xchance'][0]:.1f}", "devsweep.json", "")
            N["DevReadoutXLast"] = (f"{sc['readout_xchance'][-1]:.1f}", "devsweep.json", "")
            N["DevReadLeadsK"] = (f"{ds['read_leads_in']}", "devsweep.json", "sizes where repr outruns readout")
            N["DevReadLeadsN"] = (f"{ds['n_sizes']}", "devsweep.json", "")

    # ---------------- RL reproduces the gap as a rotation (rl_legibility/rotation_test.json)
    # A probe FROZEN on the base model loses what a REFIT probe keeps: over GRPO the binding rotates within the
    # layer, sliding out from under a fixed downstream consumer. This is the RL-time analogue of the
    # pretraining-time separation (the readout is outgrown in a basis rather than in a depth).
    rt = sorted(json.load(open(os.path.join(RES, "rl_legibility", "rotation_test.json"))),
                key=lambda r: r["step"])
    _st = [r["step"] for r in rt]; _fz = [r["frozen"] for r in rt]; _rf = [r["refit"] for r in rt]
    frho, fpe, fn, _ = spearman_exact(_st, _fz)
    rrho, rpe, rn, _ = spearman_exact(_st, _rf)
    RLJ = "rl_legibility/rotation_test.json"
    N["RlN"] = (f"{fn}", RLJ, "GRPO checkpoints")
    N["RlFrozenFirst"] = (f"{_fz[0]:.3f}", RLJ, "frozen probe, base")
    N["RlFrozenLast"] = (f"{_fz[-1]:.3f}", RLJ, "frozen probe, last step")
    N["RlRefitFirst"] = (f"{_rf[0]:.3f}", RLJ, "refit probe, base")
    N["RlRefitLast"] = (f"{_rf[-1]:.3f}", RLJ, "refit probe, last step")
    N["RlFrozenRho"] = (f"{frho:+.3f}", RLJ, "frozen decays monotonically")
    N["RlFrozenP"] = (f"{fpe:.1e}".replace("e-04", "\\times 10^{-4}"), "exact_stats", f"n={fn}; floor 2/{fn}!")
    N["RlRefitRho"] = (f"{rrho:+.3f}", RLJ, "refit decays less")
    N["RlRefitP"] = (f"{rpe:.3f}", "exact_stats", "")
    N["RlFrozenLoss"] = (f"{_fz[0] - _fz[-1]:.3f}", RLJ, "a fixed readout loses this")
    N["RlRefitLoss"] = (f"{_rf[0] - _rf[-1]:.3f}", RLJ, "an adaptive readout loses only this")

    # ---------------- causal-use control across pretraining (dynamics/em_step*/steer.json)
    # Self-gated steer toward the model's OWN decoded binding, repair measured on the failure set. It rises with
    # training (destructive-early, safe-late) while a matched-norm WRONG direction recovers almost nothing. This
    # is the causal control that the gap is a genuine USE failure, not a probe artefact.
    _cu_steps = [1000, 8000, 16000, 32000, 64000, 96000, 143000]
    _cu, _cu_rand = [], []
    for s in _cu_steps:
        r = json.load(open(os.path.join(RES, "dynamics", f"em_step{s}", "steer.json")))["results"]
        _cu.append(r["manifold_mu_decoded"]["0.5"]["repair"])
        _cu_rand.append(r["random"]["0.5"]["repair"])
    curho, cupe, cun, _ = spearman_exact(_cu_steps, _cu)
    CUJ = "dynamics/em_step*/steer.json"
    N["CuseN"] = (f"{cun}", CUJ, "pretraining checkpoints")
    N["CuseFirst"] = (f"{_cu[0]:.3f}", CUJ, "decoded repair on failures, earliest ckpt")
    N["CuseLast"] = (f"{_cu[-1]:.3f}", CUJ, "decoded repair on failures, final ckpt")
    N["CuseRho"] = (f"{curho:+.3f}", CUJ, "rises with training")
    N["CuseP"] = (f"{cupe:.1e}".replace("e-03", "\\times 10^{-3}"), "exact_stats", "")
    N["CuseRandLast"] = (f"{_cu_rand[-1]:.3f}", CUJ, "matched-norm WRONG direction barely repairs")

    # ---------------- cross-scale replication of the developmental spine
    # Turns the spine from one-model (1.4B) into a scale-robust claim: both the accessibility-opens curve and the
    # causal repairability replicate at 6.9B, and 410M is honestly too small to develop the binding.
    #
    # REPRODUCIBILITY NOTE (2026-07-14): the raw sources for this block -- dynamics/em_step*_p69/steer.json,
    # dynamics/dev_p69/dev.json, dynamics/dev_p410/dev.json -- were never committed; they exist only on the
    # analysis box. On a clone without them this block emits results/transcribed_p69_p410.json, a TRANSCRIBED
    # snapshot of the committed macro values (verify.check_provenance flags it; committing the raw dirs is a
    # NEEDS-SOURCE item in paper/CLAIM_LEDGER.md). When the raw dirs ARE present, the block recomputes from raw
    # and any drift from the snapshot surfaces as a numbers.tex diff under verify.check_numbers_fresh.
    _p69_raw_present = (
        os.path.exists(os.path.join(RES, "dynamics", "dev_p69", "dev.json"))
        and os.path.exists(os.path.join(RES, "dynamics", "dev_p410", "dev.json"))
        and all(os.path.exists(os.path.join(RES, "dynamics", f"em_step{s}_p69", "steer.json"))
                for s in _cu_steps)
    )
    if _p69_raw_present:
        # (a) causal-use control at 6.9B: dynamics/em_step*_p69/steer.json  (mirrors the 1.4B \Cuse curve)
        _cub = []
        for s in _cu_steps:
            r = json.load(open(os.path.join(RES, "dynamics", f"em_step{s}_p69", "steer.json")))["results"]
            _cub.append(r["manifold_mu_decoded"]["0.5"]["repair"])
        cubrho, cubpe, cubn, _ = spearman_exact(_cu_steps, _cub)
        CUBJ = "dynamics/em_step*_p69/steer.json"
        N["CuseBigN"] = (f"{cubn}", CUBJ, "6.9B pretraining checkpoints")
        N["CuseBigFirst"] = (f"{_cub[0]:.3f}", CUBJ, "6.9B decoded repair on failures, earliest ckpt")
        N["CuseBigLast"] = (f"{_cub[-1]:.3f}", CUBJ, "6.9B decoded repair on failures, final ckpt")
        N["CuseBigRho"] = (f"{cubrho:+.3f}", CUBJ, "rises with training at 6.9B")
        N["CuseBigP"] = (f"{cubpe:.1e}".replace("e-03", "\\times 10^{-3}"), "exact_stats", "")

        # (b) accessibility opens at 6.9B: dynamics/dev_p69/dev.json (probe chance 0.040, same as \Chance)
        dvb = json.load(open(os.path.join(RES, "dynamics", "dev_p69", "dev.json")))
        _st69 = [r["step"] for r in dvb]; _pa69 = [r["probe_acc"] for r in dvb]; _ba69 = [r["behav_acc"] for r in dvb]
        pbrho, pbpe, pbn, _ = spearman_exact(_st69, _pa69)
        DVBJ = "dynamics/dev_p69/dev.json"
        N["DevBigProbeFirst"] = (f"{_pa69[0]:.3f}", DVBJ, "6.9B best-layer decode, earliest ckpt")
        N["DevBigProbeLast"] = (f"{_pa69[-1]:.3f}", DVBJ, "6.9B best-layer decode, final ckpt")
        N["DevBigProbeXchance"] = (f"{_pa69[-1] / cc['chance']:.1f}", DVBJ, "6.9B probe LAST in units of 0.040 chance")
        N["DevBigProbeRho"] = (f"{pbrho:+.3f}", DVBJ, "accessibility opens with training at 6.9B")
        N["DevBigProbeP"] = (f"{pbpe:.3f}", "exact_stats", "")
        N["DevBigBehavFirst"] = (f"{_ba69[0]:.3f}", DVBJ, "6.9B behavioural acc, earliest ckpt")
        N["DevBigBehavLast"] = (f"{_ba69[-1]:.3f}", DVBJ, "6.9B behavioural acc, final ckpt -- lags the probe")

        # (c) honest negative at 410M: dynamics/dev_p410/dev.json -- too small to develop the binding
        dvs = json.load(open(os.path.join(RES, "dynamics", "dev_p410", "dev.json")))
        _st410 = [r["step"] for r in dvs]; _pa410 = [r["probe_acc"] for r in dvs]
        psrho, pspe, psn, _ = spearman_exact(_st410, _pa410)
        DVSJ = "dynamics/dev_p410/dev.json"
        N["DevSmallProbeLo"] = (f"{min(_pa410):.3f}", DVSJ, "410M probe stays near chance")
        N["DevSmallProbeHi"] = (f"{max(_pa410):.3f}", DVSJ, "410M never develops")
        N["DevSmallProbeRho"] = (f"{psrho:+.3f}", DVSJ, "no trend with training -- honest null")
        N["DevSmallProbeP"] = (f"{pspe:.2f}", "exact_stats", "n.s.")
    else:
        snap = load("transcribed_p69_p410.json")
        print("WARNING: raw dirs dynamics/{em_step*_p69, dev_p69, dev_p410} are absent (never committed). "
              "Emitting the TRANSCRIBED snapshot results/transcribed_p69_p410.json for the 16 cross-scale "
              "macros. See its _provenance field.", file=sys.stderr)
        for k, (v, src, note) in snap["macros"].items():
            N[k] = (v, src, note)

    # ---------------- developmental geometry LADDER across Pythia scales (qgeom_ladder.json)
    # The ROBUST law: the gap (accessibility - readout-accessibility, decode_obl_full - decode_readout_k) grows
    # over pretraining at EVERY usable scale. The "outgrown margin" (readout - classmean) is scale-DEPENDENT --
    # significant at 1B (and at 1.4B, from qgeom_dev.json) -- and it BREAKS at 6.9B, where the readout keeps
    # pace. 2.8B is censored (duplicate-weight checkpoints, the fabrication guard fired). p-values are the exact
    # permutation p already stored in the json (p_exact); we do NOT recompute them.
    ql = load("qgeom_ladder.json"); qsum = ql["_summary"]; QL = "qgeom_ladder.json"
    _usable = qsum["usable_scales"]                       # ['160M','410M','1B','6.9B']

    def _lad(scale, key, field):
        return ql[scale]["trends"][key][field]

    def _pfmt(pv):                                         # exact permutation p; keep small values off "0.00"
        if pv < 1e-2:
            return (f"{pv:.1e}".replace("e-06", "\\times 10^{-6}").replace("e-05", "\\times 10^{-5}")
                    .replace("e-04", "\\times 10^{-4}").replace("e-03", "\\times 10^{-3}"))
        return f"{pv:.2f}"
    N["LadderScales"] = (f"{len(_usable) + 1}", QL, "4 usable ladder scales + 1.4B (qgeom_dev.json)")
    _gap_rhos = [_lad(s, "gap", "rho") for s in _usable]
    N["LadderGapRhoLo"] = (f"{min(_gap_rhos):+.2f}", QL, "min gap rho across the ladder")
    N["LadderGapRhoHi"] = (f"{max(_gap_rhos):+.2f}", QL, "max gap rho across the ladder")
    # Per-scale checkpoint counts (reviewer-sim fix: Table 1 reported rho/p with no n; the 6.9B
    # null is load-bearing and its power must be visible).
    for _sc, _tg in (("160M", "SixteenM"), ("410M", "FourTenM"), ("1B", "OneB"), ("6.9B", "SixNineB")):
        N["LadN" + _tg] = (f"{ql[_sc]['n']}", QL, f"checkpoints at {_sc}")
    N["LadNOneFourB"] = (f"{load('qgeom_dev.json')['n_checkpoints']}", "qgeom_dev.json", "checkpoints at 1.4B")
    _lad_gap_sig = sum(1 for s in _usable if _lad(s, "gap", "p_exact") < 0.05)
    N["LadderGapSigTotal"] = (f"{_lad_gap_sig + 1}", QL, "usable gap-sig + 1.4B (now all 5)")
    _lad_out_sig = sum(1 for s in _usable if _lad(s, "readout_minus_classmean", "p_exact") < 0.05)
    N["OutgrownSigTotal"] = (f"{_lad_out_sig + 1}", QL, "160M/410M/1B + 1.4B = 4; breaks at 6.9B")
    N["OutgrownOneBRho"] = (f"{_lad('1B', 'readout_minus_classmean', 'rho'):+.2f}", QL, "1B: strongest outgrown point")
    N["OutgrownOneBP"] = (_pfmt(_lad('1B', 'readout_minus_classmean', 'p_exact')), QL, "exact p from json")
    N["OutgrownBreakRho"] = (f"{_lad('6.9B', 'readout_minus_classmean', 'rho'):+.2f}", QL, "6.9B: readout KEEPS PACE")
    N["OutgrownBreakP"] = (_pfmt(_lad('6.9B', 'readout_minus_classmean', 'p_exact')), QL, "n.s.")
    N["LadderCensored"] = (", ".join(qsum["censored"]), QL, "duplicate-weight checkpoints (fabrication guard)")
    for _tag, _s in [("SixteenM", "160M"), ("FourTenM", "410M"), ("OneB", "1B"), ("SixNineB", "6.9B")]:
        N["LadGapRho" + _tag] = (f"{_lad(_s, 'gap', 'rho'):+.2f}", QL, f"{_s} gap rho")
        N["LadMarRho" + _tag] = (f"{_lad(_s, 'readout_minus_classmean', 'rho'):+.2f}", QL, f"{_s} readout-classmean rho")
        N["LadAccRho" + _tag] = (f"{_lad(_s, 'accessibility', 'rho'):+.2f}", QL, f"{_s} accessibility rho")
        N["LadAccLast" + _tag] = (f"{_lad(_s, 'accessibility', 'last'):.2f}", QL, f"{_s} accessibility, final ckpt")
    # HONESTY: the GAP statistic grows at all 5 scales, but that conflates a rising representation with a
    # collapsing readout. The accessibility (the representation itself, its numerator) only rises SIGNIFICANTLY
    # -- and above the 1/K present-set baseline -- at 1B and 6.9B (usable) plus 1.4B. At 160M/410M accessibility
    # is flat/n.s. and hovers at or below 1/K, so the opening gap there is readout collapse under a near-chance
    # representation, not accessibility rising. Restrict the strong developmental claim to those scales.
    _lad_acc_sig = sum(1 for s in _usable if _lad(s, "accessibility", "p_exact") < 0.05)
    N["LadAccSigTotal"] = (f"{_lad_acc_sig + 1}", QL, "accessibility-sig scales: 1B,6.9B + 1.4B = 3")
    N["LadStrongScales"] = (f"{_lad_acc_sig + 1}", QL, "scales where the representation genuinely rises above 1/K")
    N["LadderPresent"] = (f"{1/6:.3f}", QL, "1/K present-set baseline, K=6 (same task as capacity_control)")

    # ---------------- OLMo-2 cross-family developmental confirmation (olmo_dev.json)
    # SAME synthetic entity-obligation battery, a SECOND model family. Accessibility rises over training and
    # clears the 1/K present-set baseline at 1B and 13B -- answering the review that the strong developmental
    # claim was Pythia-only. This is NOT a real-task result; the synthetic-task scoping stays. p is exact.
    if os.path.exists(os.path.join(RES, "olmo_dev.json")):
        _ol = load("olmo_dev.json"); OL = "olmo_dev.json"

        def _olp(pv):
            return (f"{pv:.1e}".replace("e-05", "\\times 10^{-5}").replace("e-04", "\\times 10^{-4}")
                    .replace("e-03", "\\times 10^{-3}")) if pv < 1e-2 else f"{pv:.2f}"
        N["OlmoNfam"] = ("2", OL, "developmental law now spans Pythia + OLMo-2 families")
        for _tag, _m in [("OneB", "OLMo-2-1B"), ("SevenB", "OLMo-2-7B"), ("ThirteenB", "OLMo-2-13B")]:
            d = _ol[_m]
            N["Olmo" + _tag + "Rho"] = (f"{d['probe_rho']:+.3f}", OL, f"{_m} accessibility vs step")
            N["Olmo" + _tag + "P"] = (_olp(d["probe_p"]), OL, "exact permutation p from json")
            N["Olmo" + _tag + "XK"] = (f"{d['probe_last_over_1oK']:.2f}", OL, f"{_m} final probe / (1/K present-set)")
            N["Olmo" + _tag + "N"] = (f"{d['n']}", OL, f"{_m} checkpoints")
            N["Olmo" + _tag + "Clears"] = (f"{d['clears_1oK_late']}", OL, "late checkpoints clearing 1/K")
            N["Olmo" + _tag + "ClearsN"] = (f"{d['n_late']}", OL, "late checkpoints")

    # ---------------- BOUNDARY: the gap does NOT transfer to real computed/tracked state (characterized)
    # (a) CRUXEval-O computed outputs (cruxeval_binding/): pre-registered KILL, established 0/n. The
    #     surface-literal control is the clincher -- the answer-slot features decode the given INPUT far better
    #     than the COMPUTED OUTPUT, so the signal is surface-dominated, not computed state.
    _cx = os.path.join("cruxeval_binding", "cruxeval_agg.json")
    if os.path.exists(os.path.join(RES, _cx)):
        cx = load(_cx); CX = "cruxeval_binding/cruxeval_agg.json"
        N["CruxN"] = (f"{cx['n_models']}", CX, "code models")
        N["CruxEstab"] = (f"{cx['k_established']}", CX, "gap established (pre-registered KILL fired)")
        N["CruxKill"] = ("triggered" if cx["kill"] else "not triggered", CX, "pre-registered KILL clause")
        _cin = [m["control_input"]["ratio"] for m in cx["per_model"].values()]
        _cout = [m["output_ratio"] for m in cx["per_model"].values()]
        N["CruxInputXcLo"] = (f"{min(_cin):.1f}", CX, "answer-slot decode of the given INPUT, x chance")
        N["CruxInputXcHi"] = (f"{max(_cin):.1f}", CX, "")
        N["CruxOutputXcLo"] = (f"{min(_cout):.1f}", CX, "answer-slot decode of the COMPUTED OUTPUT, x chance")
        N["CruxOutputXcHi"] = (f"{max(_cout):.1f}", CX, "surface-dominated: far below the input")
        # Censoring disclosure: every arm's wrong-trial count sits below the pre-registered floor,
        # so the fired KILL is reported as an underpowered null, not a boundary law (2026-08-07 rescope).
        _nf = [m["n_fail"] for m in cx["per_model"].values()]
        N["CruxFailLo"] = (f"{min(_nf)}", CX, "smallest per-model wrong-trial count")
        N["CruxFailHi"] = (f"{max(_nf)}", CX, "largest; all below the pre-registered floor")
        N["CruxFailFloor"] = ("40", "PREREG_cruxeval.md", "pre-registered minimum failure count")
    # (b) Boxes entity-tracking (boxes_binding/): declaration-confounded -- the probe reads the INITIAL
    #     declaration far better than the tracked state.
    import glob as _gb
    _bx = sorted(_gb.glob(os.path.join(RES, "boxes_binding", "*", "boxes.json")))
    _decl = []
    for f in _bx:
        c = json.load(open(f)).get("controls", {}).get("declaration", {})
        if "p_confound" in c:
            _decl.append((c["p_confound"], c["p_binding"]))
    if _decl:
        BX = "boxes_binding/*/boxes.json"
        N["BoxDeclLo"] = (f"{min(x for x, _ in _decl):.2f}", BX, "probe reads the initial declaration, low")
        N["BoxDeclHi"] = (f"{max(x for x, _ in _decl):.2f}", BX, "high")
        N["BoxTrackLo"] = (f"{min(y for _, y in _decl):.2f}", BX, "probe reads the tracked state, low")
        N["BoxTrackHi"] = (f"{max(y for _, y in _decl):.2f}", BX, "high -- declaration-confounded")

    # ---------------- readout visibility (readout_visibility.json)
    rv = load("readout_visibility.json")["models"]
    g = [v["steer_gain"] for v in rv.values()]; k = [v["readout_k"] for v in rv.values()]
    rho2, pe2, n2, _ = spearman_exact(k, g)
    N["ReadoutRho"] = (f"{rho2:+.2f}", "readout_visibility.json", "")
    N["ReadoutPexact"] = (f"{pe2:.2f}", "exact_stats", f"n={n2}, NOT significant")
    N["ReadoutN"] = (f"{n2}", "readout_visibility.json", "")
    N["ReadoutBeats"] = (f"{load('readout_visibility.json')['readout_beats_random_on_n_models']}",
                         "readout_visibility.json", "of ReadoutN; NOT a control -- Vg is label-derived")
    N["ReadoutBelowCm"] = (f"{load('readout_visibility.json')['readout_below_classmean_on_n']}",
                           "readout_visibility.json", "of ReadoutN; no systematic deficit")
    dsc = rv["DeepSeek-Coder-6.7B"]
    N["DscReadout"] = (f"{dsc['readout_k']:.3f}", "readout_visibility.json", "")
    N["DscRandom"] = (f"{dsc['random_k']:.3f}", "readout_visibility.json", "indistinguishable")
    N["DscFull"] = (f"{dsc['full']:.3f}", "readout_visibility.json", "")
    N["DscAlign"] = (f"{dsc['alignment']:.3f}", "readout_visibility.json", "")
    N["DscClassmean"] = (f"{dsc['classmean_k']:.3f}", "readout_visibility.json", "")
    N["DscRandomHi"] = (f"{dsc['random_k_ci'][1]:.3f}", "readout_visibility.json",
                        "readout_k falls BELOW the null's 95% upper bound")
    # ranges across the three models whose readout recovers real signal (fixes P1 sec5 stale literals)
    VIS = ["Pythia-1.4B", "Pythia-6.9B", "OLMo-2-7B"]
    rk = [rv[m]["readout_k"] for m in VIS]; rnd = [rv[m]["random_k"] for m in VIS]; ful = [rv[m]["full"] for m in VIS]
    N["ReadoutKVisLo"] = (f"{min(rk):.2f}", "readout_visibility.json", "readout_k, 3 visible models")
    N["ReadoutKVisHi"] = (f"{max(rk):.2f}", "readout_visibility.json", "")
    N["RandomKVisLo"] = (f"{min(rnd):.2f}", "readout_visibility.json", "")
    N["RandomKVisHi"] = (f"{max(rnd):.2f}", "readout_visibility.json", "")
    N["FullVisLo"] = (f"{min(ful):.2f}", "readout_visibility.json", "")
    N["FullVisHi"] = (f"{max(ful):.2f}", "readout_visibility.json", "")

    # ---------------- real-code corroboration (results/overnight_2026-07-11/rc_*.json)
    # HONESTY-CRITICAL: encodes that the probe does NOT beat the model's own rank-2 logit, so the merged
    # paper's LEAD cannot drift back to a "hidden knowledge" claim. This is a synthetic-task-is-not-an-artifact
    # corroboration, not new evidence of hidden knowledge.
    import glob as _glob
    _rc = [json.load(open(f)) for f in sorted(_glob.glob(os.path.join(RES, "overnight_2026-07-11", "rc_*.json")))]
    if _rc:
        _gap = [r["probe_acc_on_wrong"] for r in _rc]; _xf = [r["gap_xchance_fair"] for r in _rc]
        _r2 = [r["model_rank2_acc_on_wrong"] for r in _rc]
        _bd = [r["recency_probe_gold"] for r in _rc]; _rec = [r["recency_probe_last"] for r in _rc]
        _beats = sum(1 for r in _rc if r.get("probe_beats_model_output"))
        N["RcModels"] = (f"{len(_rc)}", "overnight rc", "3 families, code + non-code")
        N["RcGapLo"] = (f"{min(_gap):.2f}", "overnight rc", "probe-on-errors")
        N["RcGapHi"] = (f"{max(_gap):.2f}", "overnight rc", "")
        N["RcXfairLo"] = (f"{min(_xf):.1f}", "overnight rc", "x FAIR null, one over present-set size")
        N["RcXfairHi"] = (f"{max(_xf):.1f}", "overnight rc", "")
        N["RcRankTwoLo"] = (f"{min(_r2):.2f}", "overnight rc", "model's OWN rank-2 recovers gold")
        N["RcRankTwoHi"] = (f"{max(_r2):.2f}", "overnight rc", "")
        N["RcBeatsOut"] = (f"{_beats}", "overnight rc", "of RcModels: probe does NOT beat the logits (0)")
        N["RcBindRec"] = (f"{(sum(_bd) / sum(_rec)):.1f}", "overnight rc", "binding vs recency, on wrong trials")

    # ---------------- real EXECUTED code (realcode_binding/): MBPP+HumanEval under sys.settrace, trace-level
    # ground truth (a variable's executed final value, NOT readable from source). Pre-registered KILL. Counts
    # are read from the agg + globbed per-model jsons, so a re-aggregate (more models finishing) updates the
    # paper by just re-running this script.
    RCB = "realcode_binding/realcode_agg.json"
    if os.path.exists(os.path.join(RES, RCB)):
        _agg = load(RCB)
        N["RcPrimaryN"] = (f"{_agg['n_primary']}", RCB, "primary (non-censored) model x seed evaluations")
        N["RcGapEstab"] = (f"{_agg['gap_established_k']}", RCB, "established evaluations of RcPrimaryN")
        N["RcKill"] = ("not triggered" if not _agg["kill"] else "TRIGGERED", RCB, "pre-registered KILL clause")
        N["RcKillP"] = (f"{_agg['sign_p']:.3f}", RCB, "KILL sign test")
        _rcb = [json.load(open(f)) for f in
                sorted(_glob.glob(os.path.join(RES, "realcode_binding", "*", "realcode.json")))]
        RCJ = "realcode_binding/*/realcode.json"
        # distinct-MODEL view (Qwen has two seeds; deepseek s1 is censored) -- for honest per-model phrasing
        _prim = [d for d in _rcb if not d.get("censored", True)]
        _dist = sorted({d["model"] for d in _prim})
        _dist_estab = sorted({d["model"] for d in _prim if d.get("gap_established")})
        N["RcDistinctN"] = (f"{len(_dist)}", RCJ, "distinct code models with a primary evaluation")
        N["RcDistinctEstab"] = (f"{len(_dist_estab)}", RCJ, "distinct models clearing the pre-registered 2x bar")
        _xc = [d["gap_xchance"] for d in _prim if d.get("gap_established")]
        N["RcGapXcLo"] = (f"{min(_xc):.1f}", RCJ, "established-evaluation gap in units of probe chance, low")
        N["RcGapXcHi"] = (f"{max(_xc):.1f}", RCJ, "high")
        _xne = [d["gap_xchance"] for d in _prim if not d.get("gap_established")]
        if _xne:
            N["RcNotEstabXc"] = (f"{max(_xne):.1f}", RCJ, "best non-established model -- just misses 2x bar")
        _acc = [d["model_acc"] for d in _prim]
        N["RcAccLo"] = (f"{min(_acc):.2f}", RCJ, "behavioural accuracy, low")
        N["RcAccHi"] = (f"{max(_acc):.2f}", RCJ, "high")
        N["RcProbeChance"] = (f"{_prim[0]['probe_chance']:.2f}", RCJ, "1/|V|, kept distinct from behav 1/K")
        # causal-use: the model with the cleanest content-specific repair (max DEC-WRONG); report both
        _cu = [(d["steering"]["results"]["DEC"]["repair"], d["steering"]["results"]["WRONG"]["repair"])
               for d in _prim if d.get("steering", {}).get("results", {}).get("DEC")
               and d["steering"]["results"].get("WRONG")]
        _cu.sort(key=lambda t: t[0] - t[1], reverse=True)
        N["RcDecClean"] = (f"{_cu[0][0]:.2f}", RCJ, "DEC repair on the cleanest model (deepseek)")
        N["RcWrongClean"] = (f"{_cu[0][1]:.2f}", RCJ, "WRONG repair, same model -- DEC beats it there")
        # binding-vs-recency: the model with the largest binding-minus-last separation
        _rc2 = sorted(((d["recency_binding"], d["recency_last"]) for d in _prim
                       if "recency_binding" in d and "recency_last" in d), key=lambda t: t[0] - t[1], reverse=True)
        N["RcRecBind"] = (f"{_rc2[0][0]:.2f}", RCJ, "probe tracks executed value (max-separation model)")
        N["RcRecLast"] = (f"{_rc2[0][1]:.2f}", RCJ, "vs the most-recent token")

        # HONESTY (fix 1): the executed-code gap clears the loose 1/|V| vocabulary baseline but NOT the 1/K
        # present-set baseline. Against 1/K the established evaluations sit at ~1x and every 95% CI STRADDLES
        # it. Report both baselines; do not lead with the 1/|V| multiplier alone.
        _bc = _prim[0]["behav_chance"]                       # 1/K present-set chance (~0.31)
        _estab = [d for d in _prim if d.get("gap_established")]
        N["RcBehavChance"] = (f"{_bc:.2f}", RCJ, "1/K present-set baseline on executed code")
        _xk = [d["gap_on_failures"] / _bc for d in _estab]
        N["RcGapPresentXLo"] = (f"{min(_xk):.1f}", RCJ, "established gap vs 1/K present-set, low (~1x)")
        N["RcGapPresentXHi"] = (f"{max(_xk):.1f}", RCJ, "established gap vs 1/K present-set, high")
        _strad = sum(1 for d in _estab if d["gap_ci95"][0] <= _bc <= d["gap_ci95"][1])
        N["RcStraddleEstab"] = (f"{_strad}", RCJ, "established CIs that STRADDLE the 1/K baseline")
        N["RcStraddleN"] = (f"{len(_estab)}", RCJ, "established evaluations")

        # HONESTY (fix 4): the recency control is mixed on executed code -- do NOT report the max (0.44 vs 0.08).
        # Report the mean and the count of runs where the binding beats recency (it loses on Qwen s1).
        _bnd = [d["recency_binding"] for d in _prim if "recency_binding" in d]
        _lst = [d["recency_last"] for d in _prim if "recency_last" in d]
        N["RcRecBindMean"] = (f"{np.mean(_bnd):.2f}", RCJ, "mean binding-recovery across primary runs")
        N["RcRecLastMean"] = (f"{np.mean(_lst):.2f}", RCJ, "mean most-recent-token recovery")
        N["RcRecBindLo"] = (f"{min(_bnd):.2f}", RCJ, "binding-recovery range, low")
        N["RcRecBindHi"] = (f"{max(_bnd):.2f}", RCJ, "binding-recovery range, high")
        _win = sum(1 for d in _prim if d.get("recency_binding", 0) > d.get("recency_last", 0))
        N["RcRecBindWinK"] = (f"{_win}", RCJ, "runs where binding beats recency (mixed: loses on 1)")
        N["RcRecBindWinN"] = (f"{len(_prim)}", RCJ, "primary runs")

    N["DscSteer"] = (f"{dsc['steer_gain']:.3f}", "readout_visibility.json", "smallest of the 5")

    # ---------------- the phenomenon in the wild (unified paper sec. 1) -- BOTH files are TRANSCRIBED
    # corpus_stats.json is DeepSWE-v11's corpus (NOT gpt-oss); gptoss_g1_forensics.json is gpt-oss-20B's own
    # G1 run. Prose citing \Corpus* macros must attribute DeepSWE-v11; the two must never be conflated
    # (the retracted 0.16/0.83 pair was exactly that conflation). See each file's _provenance.
    co = load("corpus_stats.json"); COJ = "corpus_stats.json"
    N["CorpusN"] = (f"{co['n_attempts']:,}", COJ, "SWE-bench attempts, DeepSWE-v11 -- NOT gpt-oss")
    N["CorpusRecallFail"] = (f"{co['file_recall_fail']:.2f}", COJ, "gold-file recall on failures")
    N["CorpusRecallPass"] = (f"{co['file_recall_pass']:.2f}", COJ, "gold-file recall on passes")
    N["CorpusCovFail"] = (f"{co['edit_coverage_fail']:.2f}", COJ, "edit-coverage on failures")
    N["CorpusCovPass"] = (f"{co['edit_coverage_pass']:.2f}", COJ, "edit-coverage on passes")
    N["CorpusSawNeverEdited"] = (f"{co['saw_never_edited_pct']:.0f}", COJ, "% of failures: opened a gold file, never edited it")
    N["CorpusEditedAllFail"] = (f"{co['edited_all_still_failed_pct']:.1f}", COJ, "% edited every gold file, still failed")
    N["CorpusOmissionShare"] = (f"{co['omission_share_pct']:.0f}", COJ, "% of failures that are omission")
    N["CorpusWrongLogicShare"] = (f"{co['wrong_logic_share_pct']:.0f}", COJ, "% wrong logic, the comparator")
    N["CorpusApiShare"] = (f"{co['api_interface_share_pct']:.0f}", COJ, "% modal failing construct: api-interface wiring")

    g1 = load("gptoss_g1_forensics.json"); G1J = "gptoss_g1_forensics.json"
    N["GossSolveBare"] = (f"{g1['run1_bare']['solve']:.3f}", G1J, "gpt-oss-20B G1 run 1 (bare greedy)")
    N["GossSolveFixed"] = (f"{g1['run2_temp_retry']['solve']:.3f}", G1J, "run 2 (temp 1.0 + retry)")
    N["GossNeverEditBare"] = (f"{g1['run1_bare']['never_edit_frac']*100:.0f}", G1J, "% episodes never emitting an edit, run 1")
    N["GossNeverEditFixed"] = (f"{g1['run2_temp_retry']['never_edit_frac']*100:.0f}", G1J, "% never-edit after sampling fix")
    N["GossLocateEditRatio"] = (f"{g1['run1_bare']['locate_to_edit_ratio']}", G1J, "locate:edit tool-call ratio, run 1")
    N["GossSolveAmongEdited"] = (f"{g1['run2_temp_retry']['solve_among_edited']:.2f}", G1J, "solve rate among episodes that DID edit")

    # ---------------- gpt-oss-20B rows + OLMo post-training pair (figdata.json -- ORPHAN, unified-only)
    # figdata.json has no committed generator; verify.check_orphan_claims allows these macros ONLY in
    # unified.tex under the written ORPHAN_CLAIM_EXCEPTIONS entry, with the Reproducibility Statement
    # disclosing the unregenerable table. The OLMo pair REPLACES flagship.tex's untraced 0.57/0.50
    # (CLAIM_LEDGER sec. 1: figdata disagrees with those literals; this emits what the committed table says).
    FDJ = "figdata.json"
    fd = load(FDJ)
    _gaprow = {r["label"]: r for r in fd["gap"]}
    goss_g = _gaprow["gptoss20b"]
    N["GossPaccWrong"] = (f"{goss_g['pacc_wrong']:.2f}", FDJ, "gpt-oss-20B probe decode on its wrong trials")
    goss_r = next(r for r in fd["repair"] if r["label"] == "gptoss")
    N["GossRepairBase"] = (f"{goss_r['orig']:.3f}", FDJ, "gpt-oss-20B base recall, repair battery")
    N["GossRepairReinj"] = (f"{goss_r['reinj']:.3f}", FDJ, "after re-injecting its own decoded binding")
    N["GossRepairCtrl"] = (f"{goss_r['ctrl']:.3f}", FDJ, "after format-matched WRONG content -- harms")
    N["OlmoGapBase"] = (f"{_gaprow['olmo2-7b']['pacc_wrong']:.2f}", FDJ, "OLMo-2-7B base pAcc|wrong")
    N["OlmoGapInstr"] = (f"{_gaprow['olmo2-instruct']['pacc_wrong']:.2f}", FDJ, "OLMo-2-7B-Instruct pAcc|wrong -- alignment does not close the gap")

    # geometry correlates of k*, recomputed from the committed table over the 29 included models
    # (p3's r=+0.65 came from the unpreserved measurement script; these are the committed-table values)
    from scipy.stats import pearsonr, spearmanr
    _geo = [x for x in fd["geom"] if x["kstar"] >= 3]
    _ks = np.array([x["kstar"] for x in _geo], float)
    _db = np.array([x["d_bind"] for x in _geo], float)
    _od = np.array([x["offdiag"] for x in _geo], float)
    N["DdeclPearson"] = (f"{pearsonr(_ks, _db)[0]:+.2f}", FDJ, "k* vs entity-code dimension, model level")
    N["DdeclSpearman"] = (f"{spearmanr(_ks, _db)[0]:+.2f}", FDJ, "rank version")
    N["OffdiagPearson"] = (f"{pearsonr(_ks, _od)[0]:+.2f}", FDJ, "k* vs interference, model level")
    # These REPLACE p3's n=30 literals (+0.56 and -0.35). p3 sec. geometry printed its correlations
    # over the full 30-row table while claiming "across the 29 models"; on the n=29 set the paper
    # actually uses they are +0.64/+0.77/+0.54/-0.52/-0.34. CLAIM_LEDGER sec. 4's attribution of the
    # 0.65 to "the unpreserved measurement script" is SUPERSEDED: the printed values reproduce
    # exactly as the pre-exclusion computation over this committed table, so the discrepancy is a
    # MISSING EXCLUSION, not a lost artifact.
    import numpy as _np_g

    def _partial(x, y, z):
        """Partial correlation of x,y controlling z, off the inverse correlation matrix."""
        Pm = _np_g.linalg.inv(_np_g.corrcoef(_np_g.vstack([x, y, z])))
        return -Pm[0, 1] / _np_g.sqrt(Pm[0, 0] * Pm[1, 1])

    N["DdeclPartial"] = (f"{_partial(_ks, _db, _od):+.2f}", FDJ,
                         "k* vs entity-code dimension, controlling interference -- n=29, was 0.56 at n=30")
    N["OffdiagPartial"] = (f"{_partial(_ks, _od, _db):+.2f}", FDJ,
                           "k* vs interference, controlling dimension -- n=29, was -0.35 at n=30")
    # Query-site obligation decode, from the COMMITTED qgx2 battery (analysis/qgeom.py) -- no orphan
    # taint. REPLACES p3's 0.51--0.89: the committed range tops out at 0.8644; nothing reaches 0.89.
    import glob as _gb2
    QGX = "qgx2/*/qgeom.json"
    _qg = [json.load(open(p))["decode_obl_full"]
           for p in sorted(_gb2.glob(os.path.join(RES, "qgx2", "*", "qgeom.json")))]
    assert _qg, "results/qgx2/*/qgeom.json is missing -- QueryDecode* cannot be emitted"
    N["QueryDecodeN"] = (f"{len(_qg)}", QGX, "models with a committed query-site decode")
    N["QueryDecodeLo"] = (f"{min(_qg):.2f}", QGX, "retrieved obligation at the QUERY site, weakest model")
    N["QueryDecodeHi"] = (f"{max(_qg):.2f}", QGX, "strongest -- vs 0.017-0.054 at the declaration site")
    # ---- p3 macro-ization additions (2026-08-07): inside the figdata section so the ORPHAN
    # disclosure that already covers figdata covers these too. ----
    _g = {r["label"]: r for r in fd["geom"]}
    N["NModelsRaw"] = (f"{len(fd['geom'])}", FDJ, "models in the RAW table, BEFORE excluding pythia-70m")
    N["NOldRaw"] = (f"{sum(1 for r in fd['geom'] if r['recipe'] == 'old')}", FDJ, "old-recipe rows, raw table")
    N["NModernRaw"] = (f"{sum(1 for r in fd['geom'] if r['recipe'] == 'modern')}", FDJ, "modern-recipe rows, raw table")
    _PL = ["pythia-410m", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b"]
    _pl = [_g[m]["kstar"] for m in _PL]
    N["PythiaLadder"] = (",".join(str(v) for v in _pl), FDJ, "k* at 410M/1.4B/2.8B/6.9B/12B -- FLAT")
    N["PythiaKstarLo"] = (f"{min(_pl)}", FDJ, "")
    N["PythiaKstarHi"] = (f"{max(_pl)}", FDJ, "")
    N["PythiaKstarFirst"] = (f"{_pl[0]}", FDJ, "410M")
    N["PythiaKstarLast"] = (f"{_pl[-1]}", FDJ, "12B -- 29x the parameters, +1 binding")
    N["KstarOlmoE"] = (f"{_g['olmoe']['kstar']}", FDJ, "a MODERN model low in the range -- the classes overlap")
    N["KstarOptOneThree"] = (f"{_g['opt-1.3b']['kstar']}", FDJ, "an OLD model high in the range")
    _ol = {_g[m]["kstar"] for m in ("olmo2-7b", "olmo2-sft", "olmo2-dpo", "olmo2-instruct")}
    assert len(_ol) == 1, f"OLMo-2 alignment ladder is no longer flat: {_ol}"
    N["KstarOlmoLadder"] = (f"{_ol.pop()}", FDJ, "base/SFT/DPO/Instruct -- identical at every stage")
    N["PythiaSeventyCeiling"] = (f"{_g['pythia70']['ceiling']:.3f}", FDJ, "K=1 ceiling of the EXCLUDED model, vs 0.040 chance")
    N["PythiaSeventyKstar"] = (f"{_g['pythia70']['kstar']}", FDJ, "not a measurement -- its K=1 ceiling is at chance")
    _rob = fd["robust"]
    _K = {r["fixed_K"] for r in _rob}
    assert len(_K) == 1, f"robust curves no longer share one K: {_K}"
    N["RobustN"] = (f"{len(_rob)}", FDJ, "models with committed D-sweep curves")
    N["RobustK"] = (f"{_K.pop()}", FDJ, "K held fixed while D varies")

    # certificate transfer tally, from figdata "transfer" deltas (resolves flagship's ambiguous "5/6")
    _tr = fd["transfer"]
    _pure = ("cond-transfer", "vocab-transfer", "load-transfer")
    _combo = ("vocab+cond", "load+cond")
    N["TransferN"] = (f"{len(_tr)}", FDJ, "models in the zero-shot transfer battery")
    N["TransferPureK"] = (f"{sum(1 for r in _tr if all(r['tags'][a]['delta'] > 0 for a in _pure))}",
                          FDJ, "positive on ALL THREE single-axis shifts")
    N["TransferComboK"] = (f"{sum(1 for r in _tr if all(r['tags'][a]['delta'] > 0 for a in _pure + _combo))}",
                           FDJ, "still positive when shifts stack (OLMoE drops out)")

    # ---------------- the per-trial head-to-head we LOSE (headtohead_stats.json, TRANSCRIBED)
    hh = load("headtohead_stats.json"); HHJ = "headtohead_stats.json"
    _m = hh["models"].values()
    N["HtHN"] = (f"{hh['n_models']}", HHJ, "models in the positional head-to-head")
    N["HtHPosAurocLo"] = (f"{min(x['auroc_positional'] for x in _m):.3f}", HHJ, "positional model, weakest")
    N["HtHPosAurocHi"] = (f"{max(x['auroc_positional'] for x in _m):.3f}", HHJ, "positional model, strongest")
    N["HtHIncLo"] = (f"{min(x['overlap_increment'] for x in _m):+.4f}", HHJ, "overlap's increment, low")
    N["HtHIncHi"] = (f"{max(x['overlap_increment'] for x in _m):+.4f}", HHJ, "overlap's increment, high -- adds nothing")
    N["HtHFoilOvlLo"] = (f"{min(x['foil_max_overlap'] for x in _m):.2f}", HHJ, "emitted foil = max-overlap competitor, low")
    N["HtHFoilOvlHi"] = (f"{max(x['foil_max_overlap'] for x in _m):.2f}", HHJ, "at or below the 0.20 chance")
    N["HtHFoilOvlChance"] = (f"{next(iter(_m))['foil_ovl_chance']:.2f}", HHJ, "")
    # ---------------- p3 sec. confounds, from the COMMITTED per-model geometry (2026-08-07).
    # results/dynamics/geo_*/geom.json is written by analysis/dyn.py and IS regenerable, unlike
    # figdata.json, so these carry no orphan taint.
    def _geo(name):
        return json.load(open(os.path.join(RES, "dynamics", f"geo_{name}", "geom.json")))
    GEOJ = "dynamics/geo_*/geom.json"
    _q7, _q15, _q14b = _geo("q7"), _geo("q15"), _geo("q14b")
    N["PosVarOlmo"] = (f"{_geo('olmo2')['positional_variance_fraction']:.2f}", GEOJ,
                       "OLMo-2 is the ONE model where position does not dominate")
    N["DcovQsevenB"] = (f"{_q7['d_bind_cov']:.1f}", GEOJ, "covariance PR, Qwen2.5-7B -- the ARTIFACT")
    N["TopOneQsevenB"] = (f"{_q7['top1_dim_variance_fraction']*100:.0f}", GEOJ, "% of residual variance in ONE coordinate")
    N["DdimQonefiveB"] = (f"{_q15['d_bind']:.1f}", GEOJ, "standardized PR, Qwen2.5-1.5B")
    N["DdimQsevenB"] = (f"{_q7['d_bind']:.1f}", GEOJ, "standardized PR, Qwen2.5-7B -- the CORRECTED estimate")
    N["DdimQfourteenB"] = (f"{_q14b['d_bind']:.1f}", GEOJ, "standardized PR, Qwen2.5-14B -- scales with size")
    _c0, _c1 = _geo("step1000"), _geo("step143000")
    N["PosVarDevFirst"] = (f"{_c0['positional_variance_fraction']:.2f}", GEOJ, "Pythia-1.4B step 1000")
    N["PosVarDevLast"] = (f"{_c1['positional_variance_fraction']:.2f}", GEOJ, "step 143000 -- the confound is LEARNED")
    N["TopOneDevFirst"] = (f"{_c0['top1_dim_variance_fraction']:.3f}", GEOJ, "Pythia-1.4B step 1000")
    N["TopOneDevLast"] = (f"{_c1['top1_dim_variance_fraction']:.2f}", GEOJ, "step 143000 -- so does the massive activation")
    # ---- p3 confound ranges (2026-08-08): computed over ALL committed frozen-model geo_* dirs so the
    # prose scope matches the data. The old 99.3--99.9% literal silently dropped DeepSeek-Coder (98.1%);
    # "near 0.95" was one model (p14). geo_step*/geo_stg* are checkpoint/stage runs, not frozen models.
    _frozen = {}
    for _d in sorted(os.listdir(os.path.join(RES, "dynamics"))):
        _f = os.path.join(RES, "dynamics", _d, "geom.json")
        if _d.startswith("geo_") and not _d[4:].startswith(("step", "stg")) and os.path.exists(_f):
            _frozen[_d[4:]] = json.load(open(_f))
    _nono = {k: v for k, v in _frozen.items() if k != "olmo2"}
    N["GeoNModels"] = (f"{len(_frozen)}", GEOJ, "frozen models with committed declaration-site geometry")
    N["GeoNNonOlmo"] = (f"{len(_nono)}", GEOJ, "excluding OLMo-2, the one exception to position dominance")
    _pv = [v["positional_variance_fraction"] for v in _nono.values()]
    N["PosVarLoPct"] = (f"{100 * min(_pv):.1f}", GEOJ, "low end IS DeepSeek-Coder: 98.1, not the old 99.3")
    N["PosVarHiPct"] = (f"{100 * max(_pv):.1f}", GEOJ, "")
    N["PosVarOlmoPct"] = (f"{100 * _frozen['olmo2']['positional_variance_fraction']:.0f}", GEOJ,
                          "OLMo-2, the one exception, in the same percent units as the range")
    _ri = [v["interference_raw"] for v in _nono.values()]
    N["RawIntLo"] = (f"{min(_ri):.2f}", GEOJ, "raw pairwise cos^2 -- it is describing position")
    N["RawIntHi"] = (f"{max(_ri):.2f}", GEOJ, "")
    N["DrawHi"] = (f"{max(v['d_bind_raw'] for v in _nono.values()):.2f}", GEOJ, "raw PR is ~1 wherever position dominates")
    _res = [v["interference"] for v in _frozen.values()]
    N["ResidIntLo"] = (f"{min(_res):.3f}", GEOJ, "residual interference after removing the per-slot mean")
    N["ResidIntHi"] = (f"{max(_res):.3f}", GEOJ, "")
    # ---- p3 robustness sweep summary (figdata.json["robust"]). The old prose "from ~0.9 toward ~0.25"
    # described only the top curves: four of the eight models sit below 0.4 already at D=0, and those four
    # are exactly the models whose k* is below the sweep's K=8 -- itself the capacity != robustness point.
    # k* is joined on the "model" field against figdata["geom"] (the robust entries' own "recipe" tags are
    # wrong for mist/q05/q7 -- geom is the source of truth; see CLAIM_LEDGER sec. 8.9). The q05 sweep used
    # Qwen2.5-0.5B-Instruct while k*=3 was measured on the base 0.5B; the join maps it explicitly and the
    # paper discloses it. All 8 curves run D=0..256.
    _fd = load("figdata.json")
    _rk = {g["model"]: g["kstar"] for g in _fd["geom"]}
    _rk["Qwen/Qwen2.5-0.5B-Instruct"] = _rk["Qwen/Qwen2.5-0.5B"]
    _rob = _fd["robust"]
    _rc = [x["curve"] for x in _rob]
    assert all(c[0][0] == 0 and c[-1][0] == 256 for c in _rc), "robust curves must span D=0..256"
    _r0 = [c[0][1] for c in _rc]
    _r1 = [c[-1][1] for c in _rc]
    _lowk = [x["curve"][0][1] for x in _rob if _rk[x["model"]] < 8]
    _highk = [x["curve"][0][1] for x in _rob if _rk[x["model"]] >= 8]
    assert max(_lowk) < min(_highk), "the D=0 split must separate the k* groups exactly"
    N["RobustLowKN"] = (f"{len(_lowk)}", "figdata.json", "robust-sweep models with k* below the sweep's K=8")
    N["RobustLowKDzeroHi"] = (f"{max(_lowk):.2f}", "figdata.json", "highest D=0 recall among the k*<8 models")
    N["RobustHighKDzeroLo"] = (f"{min(_highk):.2f}", "figdata.json", "lowest D=0 recall among the k*>=16 models")
    N["RobustDmax"] = ("256", "figdata.json", "largest distractor block in the sweep")
    N["RobustDzeroLo"] = (f"{min(_r0):.2f}", "figdata.json", "D=0 recall, lowest of 8 -- already collapsed at K=8")
    N["RobustDzeroHi"] = (f"{max(_r0):.2f}", "figdata.json", "D=0 recall, highest of 8")
    N["RobustDzeroMed"] = (f"{float(np.median(_r0)):.2f}", "figdata.json", "D=0 recall, median of 8")
    N["RobustDendLo"] = (f"{min(_r1):.2f}", "figdata.json", "D=256 recall, lowest of 8")
    N["RobustDendHi"] = (f"{max(_r1):.2f}", "figdata.json", "D=256 recall, highest of 8")
    N["RobustDendMed"] = (f"{float(np.median(_r1)):.2f}", "figdata.json", "D=256 recall, median of 8")

    # ---------------- SFT builds the WM representation; RLVR erodes it (rlvr_sft_stages.json, TRANSCRIBED)
    # d_decl = the declaration-site / entity-code participation ratio (the metric GEOMETRY_DYNAMICS 2c/2h
    # calls d_bind), renamed per CLAIM_LEDGER sec. 9 (measured at the entity token; it is the entity code,
    # not the binding). Raw batteries committed under rl_legibility/ and dynamics/; a committed aggregator
    # retires this transcribed summary (verify.ORPHAN_OK; check_provenance warns).
    rs = load("rlvr_sft_stages.json"); RSJ = "rlvr_sft_stages.json"
    _sc = {r["stage"]: r for r in rs["stage_chain"]["stages"]}
    N["SftFailBase"] = (f"{_sc['base']['failures']}", RSJ, "OLMo-2-7B failures/400, base")
    N["SftFailSFT"] = (f"{_sc['SFT']['failures']}", RSJ, "failures/400 after SFT -- most of the drop is here")
    N["SftFailDPO"] = (f"{_sc['DPO']['failures']}", RSJ, "failures/400 after DPO")
    N["SftFailRLVR"] = (f"{_sc['RLVR']['failures']}", RSJ, "failures/400 after RLVR -- barely moves")
    N["SftProbeBase"] = (f"{_sc['base']['probe_acc']:.3f}", RSJ, "probe decodability, base")
    N["SftProbeSFT"] = (f"{_sc['SFT']['probe_acc']:.3f}", RSJ, "probe decodability after SFT -- jumps then flat")
    N["SftDdeclRLVRDelta"] = (f"{_sc['RLVR']['d_decl'] - _sc['DPO']['d_decl']:+.2f}", RSJ, "RLVR moves the declaration-site subspace by this (entity-code, not binding)")
    N["SftSpearFirst"] = (f"{_sc['base']['spearman_probe_model']:.3f}", RSJ, "Spearman(probe,model), base")
    N["SftSpearLast"] = (f"{_sc['RLVR']['spearman_probe_model']:.3f}", RSJ, "Spearman(probe,model), post-RLVR -- posteriors converge")
    _rl = rs["rl_steps"]["steps"]; _dd = [r["d_decl"] for r in _rl]
    N["RlStepN"] = (f"{len(_rl)}", RSJ, "collision-free RL-Zero-Code step branches")
    N["RlStepSpan"] = (f"{_rl[-1]['step'] - _rl[0]['step']:,}", RSJ, "RL steps spanned")
    N["RlStepProbeFirst"] = (f"{_rl[0]['probe_acc']:.3f}", RSJ, "probe decodability, first RL step")
    N["RlStepProbeLast"] = (f"{_rl[-1]['probe_acc']:.3f}", RSJ, "probe decodability, last RL step -- falls")
    N["RlStepRecallFirst"] = (f"{_rl[0]['recall']:.3f}", RSJ, "recall @ K=10, first RL step")
    N["RlStepRecallLast"] = (f"{_rl[-1]['recall']:.3f}", RSJ, "recall @ K=10, last RL step -- falls")
    N["RlStepMarginFirst"] = (f"{_rl[0]['margin']:+.3f}", RSJ, "capacity margin @ K=10, first RL step")
    N["RlStepMarginLast"] = (f"{_rl[-1]['margin']:+.3f}", RSJ, "capacity margin @ K=10, last -- crosses zero")
    N["RlStepDdeclLo"] = (f"{min(_dd):.2f}", RSJ, "declaration-site subspace dim, min over RL steps")
    N["RlStepDdeclHi"] = (f"{max(_dd):.2f}", RSJ, "declaration-site subspace dim, max -- frozen")
    N["RlStepDdeclPct"] = (f"{(max(_dd)-min(_dd))/2/(sum(_dd)/len(_dd))*100:.1f}", RSJ, "half-range of d_decl as % -- geometry frozen (+/-0.1%)")

    # ---------------- Trainium developmental replication (neuron_developmental.py + two_clocks.py)
    # Independent re-implementation of the trajectory on trn1: paired trials at every public
    # Pythia checkpoint, read layer fixed once on a selection fold. Onset claim gated by the
    # pre-declared falsification condition in two_clocks.py (claim ledger OR-3: NOT SUPPORTED).
    dv = load("developmental_trn1/dev-p14b.json"); DVJ = "developmental_trn1/dev-p14b.json"
    N["DevTrnAccFirst"] = (f"{dv['rows'][0]['accessibility_R']:.3f}", DVJ, "accessibility, step1000")
    N["DevTrnAccLast"] = (f"{dv['rows'][-1]['accessibility_R']:.3f}", DVJ, "accessibility, step143000")
    N["DevTrnBehavLast"] = (f"{dv['rows'][-1]['behaviour_U']:.3f}", DVJ, "behaviour, step143000")
    N["DevTrnProbeOnFail"] = (f"{dv['rows'][-1]['gap_given_wrong_G']:.3f}", DVJ, "p(probe right | model wrong), final ckpt")
    N["DevTrnBaseline"] = (f"{1.0/dv['K']:.3f}", DVJ, "1/K present-set baseline")
    tc = load("two_clocks.json")
    _tj = {x["label"]: x for x in tc["trajectories"]}
    _b, _s = _tj["dev-p14b"], _tj["dev-p410m"]
    N["DevTrnGapSlope"] = (f"{_b['gap_slope_per_log10_step']:+.3f}", "two_clocks.json", "gap slope per log10 step, 1.4B")
    N["DevTrnGapSlopeCILo"] = (f"{_b['gap_slope_ci95'][0]:.3f}", "two_clocks.json", "")
    N["DevTrnGapSlopeCIHi"] = (f"{_b['gap_slope_ci95'][1]:.3f}", "two_clocks.json", "")
    N["DevTrnSmallSlope"] = (f"{_s['gap_slope_per_log10_step']:+.3f}", "two_clocks.json", "gap slope, 410M -- n.s.")
    N["DevTrnSmallSlopeCILo"] = (f"{_s['gap_slope_ci95'][0]:.3f}", "two_clocks.json", "")
    N["DevTrnSmallSlopeCIHi"] = (f"{_s['gap_slope_ci95'][1]:.3f}", "two_clocks.json", "CI spans zero")
    N["OnsetLag"] = (f"{_b['lag_log10_steps']:+.3f}", "two_clocks.json", "onset lag in decades, 1.4B")
    N["OnsetLagCILo"] = (f"{_b['lag_ci95'][0]:.3f}", "two_clocks.json", "")
    N["OnsetLagCIHi"] = (f"{_b['lag_ci95'][1]:.3f}", "two_clocks.json", "CI spans zero -- onset claim NOT supported")

    # ---------------- Trainium capacity re-measurement (neuron_capacity.py -> agg_capacity.py)
    # POST-HOC power law per claim ledger BC-2; committed fit in capacity_alpha_fit.py.
    af = load("capacity_alpha_fit.json"); AFJ = "capacity_alpha_fit.json"
    N["CapAlpha"] = (f"{af['alpha']:.3f}", AFJ, "K50 = c*N^alpha, D=0, uncensored")
    N["CapAlphaCILo"] = (f"{af['alpha_ci95_family_clustered'][0]:.3f}", AFJ, "")
    N["CapAlphaCIHi"] = (f"{af['alpha_ci95_family_clustered'][1]:.3f}", AFJ, "")
    N["CapRsq"] = (f"{af['r2']:.2f}", AFJ, "")
    N["CapLofoLo"] = (f"{af['lofo_range'][0]:.2f}", AFJ, "leave-one-family-out alpha, min")
    N["CapLofoHi"] = (f"{af['lofo_range'][1]:.2f}", AFJ, "leave-one-family-out alpha, max")
    N["CapFitN"] = (f"{af['n_models']}", AFJ, "models entering the fit")
    ct = load("capacity_trn1_summary.json"); CTJ = "capacity_trn1_summary.json"
    _rc = ct["recipe_test_controlling_for_scale"]
    N["TrnRecipeCoef"] = (f"{_rc['recipe_coef_logK50']:+.3f}", CTJ, "recipe coef controlling scale -- interval spans zero")
    N["TrnRecipeCILo"] = (f"{_rc['recipe_ci95'][0]:.3f}", CTJ, "")
    N["TrnRecipeCIHi"] = (f"{_rc['recipe_ci95'][1]:.3f}", CTJ, "")
    N["TrnScaleCoef"] = (f"{_rc['log_params_coef']:+.3f}", CTJ, "log-params coef, same regression")
    N["TrnScaleCILo"] = (f"{_rc['log_params_ci95'][0]:.3f}", CTJ, "")
    N["TrnScaleCIHi"] = (f"{_rc['log_params_ci95'][1]:.3f}", CTJ, "excludes zero")
    N["TrnOptResolved"] = (f"{[m for m in ct['models'] if m['label']=='opt-1.3b-x'][0]['by_D']['0']['K50']:.1f}", CTJ, "opt-1.3b resolves past the old K=24 ceiling")
    ia = load("interference_axis.json"); IAJ = "interference_axis.json"
    N["TrnIntRetained"] = (f"{ia['retained_fraction_median']:.3f}", IAJ, "median retained capacity fraction under D=256")
    N["TrnIntReduced"] = (f"{ia['n_models_interference_reduces_capacity']}", IAJ, "")
    N["TrnIntN"] = (f"{ia['n_uncensored']}", IAJ, "")
    N["TrnIntRho"] = (f"{ia['spearman_capacity_vs_retained']['rho']:+.2f}", IAJ, "capacity does not predict robustness")
    N["TrnIntP"] = (f"{ia['spearman_capacity_vs_retained']['p']:.2f}", IAJ, "")
    cc = load("cross_campaign_capacity.json"); CCJ = "cross_campaign_capacity.json"
    _cb = cc["comparison"][0]
    N["BoundMeasured"] = (f"{_cb['measured_capacity']:.1f}", CCJ, "4.98M task-trained capacity")
    N["BoundPredicted"] = (f"{_cb['law_prediction']:.2f}", CCJ, "law prediction at 4.98M")
    N["BoundFactor"] = (f"{_cb['ratio']:.0f}", CCJ, "the regime boundary")
    N["CapLooMax"] = (f"{af['loo_max_abs_delta']:.2f}", AFJ, "max |delta alpha| deleting any single model")

    # ---------------- time law: formation cost vs binding load (absorbed from p6_laws)
    # Emerged runs only in both corpora; exponents are lower bounds (censoring).
    ea = load("extra_analyses.json"); EAJ = "extra_analyses.json"
    _tb = ea["B_emergence_time_vs_K"]
    N["TimeLawHCoef"] = (f"{_tb['coef']:.0f}", EAJ, "t = c*K^a, H100 clocks corpus")
    N["TimeLawHExp"] = (f"{_tb['exponent']:.2f}", EAJ, "")
    N["TimeLawHRsq"] = (f"{_tb['r2']:.2f}", EAJ, "")
    dvx = load("devaxis_analyses.json"); DXJ = "devaxis_analyses.json"
    _dl = dvx["emergence_law"]["devaxis"]
    N["TimeLawDCoef"] = (f"{_dl['coef']:.0f}", DXJ, "same law, Trainium codebase")
    N["TimeLawDExp"] = (f"{_dl['exponent']:.2f}", DXJ, "")
    N["TimeLawDRsq"] = (f"{_dl['r2']:.2f}", DXJ, "")
    _nk = [(float(k), v["median"]) for k, v in _dl["cells"].items() if int(k) != 8]
    _slope = np.polyfit(np.log10([k for k, _ in _nk]), np.log10([m for _, m in _nk]), 1)[0]
    N["TimeLawExpNoKeight"] = (f"{_slope:.1f}", DXJ, "devaxis exponent w/o the n=2 K=8 cell -- honest low end")
    _dbl = sorted([2 ** _tb["exponent"], 2 ** _dl["exponent"]])
    N["TimeLawDblLo"] = (f"{_dbl[0]:.0f}", EAJ + " + " + DXJ, "step-cost factor per doubling of K, low")
    N["TimeLawDblHi"] = (f"{_dbl[1]:.0f}", EAJ + " + " + DXJ, "high")
    _lt = ea["D_learnability_threshold"]["by_size"]
    N["LearnSmallN"] = (f"{_lt['1.31']['n']}", EAJ, "runs at 1.31M params")
    N["LearnSmallEmerged"] = (f"{_lt['1.31']['emerged']}", EAJ, "none form at 1.31M in measured budgets")
    N["SeedOverdisp"] = (f"{dvx['seed_heterogeneity']['overdispersion_ratio']:.2f}", DXJ, "seed variance / binomial")
    N["DevaxisRuns"] = (f"{dvx['n_runs_total']:,}", DXJ, "Trainium program total")

    # ---------------- supervision-supply gap law (X14; training_history.tex)
    gl = load("x14_gap_law.json"); GLJ = "x14_gap_law.json"
    _ga = gl["gap_axis"]
    for _gap, _mac in [("14", "GapCellUniform"), ("99", "GapCellPeriodic"),
                       ("990", "GapCellClustered"), ("10395", "GapCellBlocks")]:
        _a, _b = (int(x) for x in _ga[_gap].split("/"))
        N[_mac] = (_ga[_gap], GLJ, f"re-formed/complete at max gap {_gap}")
        N[_mac + "Pct"] = (f"{100 * _a / _b:.0f}", GLJ, "")
    N["GapFifty"] = (f"{gl['fit']['gap50']:.0f}", GLJ, "logistic 50% point in log gap")
    N["GapSpanFactor"] = (f"{10395 / 14:.0f}", GLJ, "max/min gap ratio at matched rate+totals")
    N["GapIncomplete"] = (f"{gl['incomplete_total']}", GLJ, "incomplete rescues excluded and counted")

    # ---------------- pre-registered dose-250 replication (training_history.tex)
    dz = load("d250_verdict.json"); DZJ = "d250_verdict.json"
    N["DzBaseN"] = (f"{dz['baseline']['n']}", DZJ, "pooled baseline")
    N["DzBaseCross"] = (f"{dz['baseline']['crossed']}", DZJ, "")
    N["DzBaseMed"] = (f"{dz['baseline']['median']:,.0f}", DZJ, "")
    N["DzChessN"] = (f"{dz['chess']['n']}", DZJ, "")
    N["DzChessCross"] = (f"{dz['chess']['crossed']}", DZJ, "")
    N["DzChessMed"] = (f"{dz['chess']['median']:,.0f}", DZJ, "")
    N["DzShufCross"] = (f"{dz['shuffle']['crossed']}", DZJ, "")
    N["DzShufCens"] = (f"{dz['shuffle']['cens']}", DZJ, "")
    N["DzShufMed"] = (f"{dz['shuffle']['median']:,.0f}", DZJ, "")
    N["DzPChess"] = (f"{dz['p_chess']:.2f}", DZJ, "verdict NOT CONFIRMED")
    N["DzPShuf"] = (f"{dz['p_shuffle']:.3f}", DZJ, "specificity arm slows learning")

    # ---------------- H100 re-derivation headliners (training_history.tex)
    hr = load("h100_reanalysis.json"); HRJ = "h100_reanalysis.json"
    _q1 = hr["Q1_warmup"]
    N["WuRatio"] = (f"{_q1['ratio']:.2f}", HRJ, "re-derived warmup handicap")
    N["WuP"] = (_sci(_q1["p"]), HRJ, "")
    N["WuSlowMed"] = (f"{_q1['strata']['500']['median']:,.0f}", HRJ, "warmup-500 baseline median")
    N["WuSlowN"] = (f"{_q1['strata']['500']['n']}", HRJ, "")
    N["WuSlowCross"] = (f"{_q1['strata']['500']['crossed']}", HRJ, "crosses eventually")
    N["WuFastMed"] = (f"{_q1['strata']['1']['median']:,.0f}", HRJ, "warmup-free baseline median")
    N["WuFastN"] = (f"{_q1['strata']['1']['n']}", HRJ, "")
    N["WuFastCross"] = (f"{_q1['strata']['1']['crossed']}", HRJ, "")
    _sv = {}
    for _r in hr["Q2_low_dose_w500"]:
        _sv.setdefault(_r["prime"], []).append(_r["saved"] / 1000)
    for _pr, _mac in [("chess", "SaveChess"), ("shuffle_chess", "SaveShuf"), ("chess_random", "SaveRand")]:
        N[_mac + "Lo"] = (f"{min(_sv[_pr]):.1f}", HRJ, f"{_pr} savings vs cold baseline, k-steps, min over doses")
        N[_mac + "Hi"] = (f"{max(_sv[_pr]):.1f}", HRJ, "max")
    _q5 = {(r["warmup"], r["dose"]): r for r in hr["Q5_harm"]}
    N["HarmChessFive"] = (_q5[(500, 5000)]["chess"], HRJ, "crossing at dose 5k, warmup-matched")
    N["HarmShufFive"] = (_q5[(500, 5000)]["shuffle"], HRJ, "")
    N["HarmPFive"] = (f"{_q5[(500, 5000)]['fisher_p']:.4f}", HRJ, "")
    N["HarmChessTen"] = (_q5[(500, 10000)]["chess"], HRJ, "")
    N["HarmShufTen"] = (_q5[(500, 10000)]["shuffle"], HRJ, "")
    N["HarmPTen"] = (f"{_q5[(500, 10000)]['fisher_p']:.4f}", HRJ, "")
    N["HarmChessTwenty"] = (_q5[(500, 20000)]["chess"], HRJ, "dose 20k: structure survives")
    N["HarmShufTwenty"] = (_q5[(500, 20000)]["shuffle"], HRJ, "dose 20k: statistics-only collapses")
    _ca, _cn = (int(x) for x in _q5[(500, 20000)]["chess"].split("/"))
    _sa, _sn = (int(x) for x in _q5[(500, 20000)]["shuffle"].split("/"))
    N["HarmPTwenty"] = (_sci(_fisher2(_ca, _cn - _ca, _sa, _sn - _sa)), HRJ,
                        "exact two-sided Fisher on the dose-20k cells")
    _arm = {(r["prime"], r["dose"]): r for r in hr["Q3Q4_w1"]["arms"]}
    N["SpeedChessB"] = (f"{_arm[('chess', 5000)]['median_phaseB']:,.0f}", HRJ, "phase-B median after chess d5000")
    N["SpeedShufB"] = (f"{_arm[('shuffle_chess', 5000)]['median_phaseB']:,.0f}", HRJ, "after shuffled chess d5000")
    N["SpeedRatio"] = (f"{_arm[('shuffle_chess', 5000)]['median_phaseB'] / _arm[('chess', 5000)]['median_phaseB']:.1f}", HRJ, "")
    N["SpeedShufP"] = (_sci(_arm[("shuffle_chess", 5000)]["p_phaseB_vs_base"]), HRJ, "phase-B vs warmup-matched base")
    N["SpeedChessP"] = (f"{_arm[('chess', 5000)]['p_phaseB_vs_base']:.2f}", HRJ, "not significant")
    _q6 = hr["Q6_strategy"]
    N["StratChessR"] = (_q6["chess"], HRJ, "re-derived strategy null")
    N["StratRandR"] = (_q6["random"], HRJ, "")
    N["StratP"] = (f"{_q6['p']:.2f}", HRJ, "")
    _q7 = hr["Q7_revival"]
    N["RevZero"] = (_q7["0.0"], HRJ, "revival at p=0")
    N["RevTwoTh"] = (_q7["0.002"], HRJ, "")
    N["RevFiveTh"] = (_q7["0.005"], HRJ, "")
    N["RevOne"] = (_q7["0.01"], HRJ, "full revival from p=0.01")
    _dr = {(r["K"], r["p"]): r for r in dvx["dose_response"]}
    N["RetKfZero"] = (f"{_dr[(4, 0.0)]['held']}/{_dr[(4, 0.0)]['n']}", DXJ, "K=4 retention at p=0")
    N["RetKfLow"] = (f"{_dr[(4, 0.01)]['held']}/{_dr[(4, 0.01)]['n']}", DXJ, "K=4 at p=0.01")
    N["RetKfTwo"] = (f"{_dr[(4, 0.02)]['held']}/{_dr[(4, 0.02)]['n']}", DXJ, "K=4 at p=0.02")
    N["RetKfThree"] = (f"{_dr[(4, 0.03)]['held']}/{_dr[(4, 0.03)]['n']}", DXJ, "K=4 at p=0.03")
    N["RetKtZero"] = (f"{_dr[(2, 0.0)]['held']}/{_dr[(2, 0.0)]['n']}", DXJ, "K=2 at p=0")
    N["RetKtTwo"] = (f"{_dr[(2, 0.02)]['held']}/{_dr[(2, 0.02)]['n']}", DXJ, "K=2 at p=0.02")
    N["RetKtThree"] = (f"{_dr[(2, 0.03)]['held']}/{_dr[(2, 0.03)]['n']}", DXJ, "K=2 at p=0.03 -- within its interval")

    # ---------------- token-denominated economics (p3_capacity.tex sec:cost, training_history.tex)
    tc = load("token_costs.json"); TCJ = "token_costs.json"
    _tl = tc["time_law_tokens"]
    N["TokLawHExp"] = (f"{_tl['h100_fit']['exponent']:.2f}", TCJ, "time law refit in token units, H100")
    N["TokLawHRsq"] = (f"{_tl['h100_fit']['r2']:.2f}", TCJ, "")
    N["TokLawDExp"] = (f"{_tl['devaxis_fit']['exponent']:.2f}", TCJ, "devaxis")
    N["TokLawDRsq"] = (f"{_tl['devaxis_fit']['r2']:.2f}", TCJ, "")
    N["TokLawDExpNoKeight"] = (f"{_tl['devaxis_exponent_no_K8']:.1f}", TCJ, "without the K=8 cell")
    N["TokKfourH"] = (f"{_tl['K4_formation_tokens']['h100'] / 1e6:.0f}", TCJ, "K=4 formation, millions of tokens, H100")
    N["TokKfourD"] = (f"{_tl['K4_formation_tokens']['devaxis'] / 1e6:.0f}", TCJ, "devaxis")
    _mt = tc["maintenance_tokens"]
    N["TokWindow"] = (f"{_mt['devaxis_K4_window_tokens'] / 1e6:.1f}", TCJ, "10k-step deprivation window, M tokens")
    N["TokTrickle"] = (f"{_mt['trickle_tokens'] / 1e6:.2f}", TCJ, "deciding supervised trickle, M tokens")
    N["TokTricklePct"] = (f"{_mt['trickle_fraction_of_window'] * 100:.1f}", TCJ, "trickle as % of window throughput")
    N["TokGapFifty"] = (f"{tc['gap_law_tokens']['gap50_tokens'] / 1e6:.2f}", TCJ, "gap50 in M unsupervised tokens")
    _pt = tc["priming_tokens"]
    _breq = _pt["baseline_requirement_tokens"]
    N["TokBaseReqLo"] = (f"{_breq['campaign_equalised_5875'] / 1e6:.1f}", TCJ, "task requirement, M tokens, campaign accounting")
    N["TokBaseReqHi"] = (f"{_breq['warmup_free_rederived'] / 1e6:.1f}", TCJ, "raw-record accounting")
    N["TokWuWaste"] = (f"{_pt['warmup_artifact_extra_tokens'] / 1e6:.1f}", TCJ, "warmup-500 extra cost per run, M tokens")
    N["TokWuWasteMult"] = (f"{_pt['warmup_artifact_extra_tokens'] / _breq['warmup_free_rederived']:.1f}", TCJ,
                           "warmup waste as a multiple of the task's own requirement")
    N["TokHarmDose"] = (f"{_pt['harm_dose20k_tokens'] / 1e6:.1f}", TCJ, "the dose-20k structure-free corpus, M tokens")
    return N


def main():
    N = build()
    lines = ["% GENERATED by analysis/make_numbers.py -- DO NOT EDIT.",
             "% Every value below is computed from a committed file in paper/results/.",
             "% If a paper types a literal instead of using these macros, verify.py will not protect it.",
             ""]
    w = max(len(k) for k in N)
    for k, (v, src, note) in N.items():
        pad = " " * (w - len(k))
        cmt = f"  % {src}" + (f" -- {note}" if note else "")
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}{pad}{cmt}")
    open(OUT, "w").write("\n".join(lines) + "\n")

    print(f"{'macro':<20}{'value':<16}source")
    print("-" * 78)
    for k, (v, src, note) in N.items():
        flag = "  <-- " + note if note and ("NOT" in note or "WEAK" in note or "12x" in note) else ""
        print(f"{k:<20}{v:<16}{src}{flag}")
    print(f"\nwrote {OUT}  ({len(N)} macros)")


if __name__ == "__main__":
    main()
