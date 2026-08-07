"""Is "read leads use" an artifact of probe capacity?

Chou et al. (arXiv:2605.27078, "Two Speeds of Learning: A Representation-Readout Decomposition") argue that
representation learning and readout calibration are BOTH active throughout training, and that the readout can
be train-biased *before* the representation matures. If they are right, our two-clock result may be an
artifact of the instrument: our representation clock is a linear probe that is REFIT at each checkpoint and
allowed to SHOP ACROSS LAYERS, while our readout clock is the model's own fixed, final-layer,
unembedding-mediated output. A probe with that much freedom is a strictly stronger readout than the model's
own, so r(t) > u(t) could hold by construction.

The control. Report three quantities, not two, all on the SAME population -- the trials the model got WRONG
(where its behavioural accuracy is 0 by construction):

    r_lstar   probe at the layer chosen on a VALIDATION fold, reported on a disjoint test fold.
              Upper bound on what is linearly represented anywhere in the stack.
    r_final   probe restricted to the FINAL layer -- the state the unembedding actually consumes.
              Capacity-matched in layer. No shopping.
    sep       r_lstar - r_final.  How much of "what is represented" never reaches the readout's layer.

If r_final collapses to chance, there are not two clocks: there is one clock and a probe with extra freedom.

Result (Pythia-1.4B, K=6 bindings, D=256 distractor tokens, 25-way obligation decode, chance 0.040):

    r_final rises 0.194 -> 0.384 over pretraining, ~10x chance, with NO layer selection.  The clocks survive.

    sep is <= 0 early (the final layer is as good as any layer) and OPENS after ~32k steps,
    growing monotonically to 0.317 (Spearman rho = 1.000 against step).

    So pretraining does not merely build the representation faster than the readout.  It PUSHES the binding
    away from the layer the readout consumes.  That is a sharper claim than "two clocks", and it is the
    pretraining-time analogue of the RL rotation result (a frozen consumer loses what a refit one keeps).

Honest limits, stated because they bound the claim:
  * r_* is a 25-way decode; behavioural accuracy is 6-way (restricted to the obligations present in context).
    The two are NOT directly comparable, and we never subtract them.  On failures, behaviour is 0 by
    construction, so the meaningful comparison is r_final against CHANCE, not against behaviour.
  * capacity-matched in LAYER is not capacity-matched in MAP: the unembedding is fixed, our probe is fit.
    A probe can read a direction the unembedding cannot. Closing that gap needs the gold obligation's rank
    under the model's own logits, which this dump does not contain.  See TWO_CLOCKS_PAPER_v2.md.
  * one model, one seed.  Pythia ships one seed per size, so the seed floor (arXiv:2606.25010) is not
    measurable here at all.  It comes from the controlled runs.
"""
from __future__ import annotations
import json, os

import numpy as np
from exact_stats import spearman_exact, min_attainable_p

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "dynamics")
STEPS = [1000, 8000, 16000, 32000, 64000, 96000, 143000]

# source: paper/results/dynamics/em_step*/steer.json -> results.gated_all["0.5"].base
BEHAVIOUR = {1000: 0.190, 8000: 0.255, 16000: 0.255, 32000: 0.305,
             64000: 0.258, 96000: 0.287, 143000: 0.323}


def main():
    rows, leakfree = [], True
    for s in STEPS:
        d = json.load(open(os.path.join(RES, f"em_step{s}", "per_layer.json")))
        pf, pv = d["per_layer_fail"], d["per_layer_fail_val"]
        lstar, final = d["lstar"], d["final_layer"]
        leakfree &= (pv.index(max(pv)) == lstar)      # layer chosen on validation, not on test
        rows.append(dict(step=s, lstar=lstar, final_layer=final,
                         r_lstar=pf[lstar], r_final=pf[final], sep=pf[lstar] - pf[final],
                         chance=d["chance"], n_fail=d["n_fail"], behaviour=BEHAVIOUR[s]))

    assert leakfree, "lstar is not the validation argmax -- r_lstar would be selection-biased"

    # scipy's spearmanr uses a t-approximation and returns exactly 0.0 for a perfect ordering.
    # With n=7 there are 7! orderings and only 2 reach |rho|=1, so the FLOOR is 2/5040 = 4.0e-4.
    rho, p, _n, _pasym = spearman_exact([r["step"] for r in rows], [r["sep"] for r in rows])
    out = dict(model="EleutherAI/pythia-1.4b", K=6, D=256, decode_classes=25,
               chance=rows[0]["chance"], leak_free_layer_selection=True,
               rows=rows, sep_vs_step_spearman=float(rho), sep_vs_step_p=float(p),
               r_final_first=rows[0]["r_final"], r_final_last=rows[-1]["r_final"])

    print(f"chance = {rows[0]['chance']:.3f}   (CONTROL PRINTED FIRST)")
    print(f"layer selection leak-free: {leakfree}\n")
    print(f"{'step':>7} {'lstar':>5} {'r_lstar':>8} {'r_final':>8} {'sep':>7} {'n_fail':>7}")
    for r in rows:
        print(f"{r['step']:>7} {r['lstar']:>5} {r['r_lstar']:>8.3f} {r['r_final']:>8.3f} "
              f"{r['sep']:>7.3f} {r['n_fail']:>7}")
    print(f"\nr_final: {rows[0]['r_final']:.3f} -> {rows[-1]['r_final']:.3f} "
          f"({rows[-1]['r_final']/rows[0]['r_final']:.2f}x chance-relative "
          f"{rows[-1]['r_final']/rows[0]['chance']:.1f}x)")
    print(f"separation vs step: Spearman rho={rho:.3f}, exact p={p:.2e} "
          f"(floor at n={len(rows)} is {min_attainable_p(len(rows)):.2e})")
    print("\nThe clocks survive capacity matching. And the binding MOVES AWAY from the readout's layer.")

    dst = os.path.join(HERE, "..", "results", "capacity_control.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
