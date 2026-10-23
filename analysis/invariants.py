"""Runtime invariants. Every one of these is a bug this repo actually shipped.

WHY THIS FILE EXISTS, and why it is not a bug-finding agent.

Adversarial agents read `analysis/qgeom.py` line by line and produced six real defects. They also **quoted the
exact line** that carried the worst bug in the project --

    M = np.stack([Hz[Y == c].mean(0) for c in cls])          # qgeom.py, the class-mean subspace Vw

-- while explaining what `pr_state_classmeans` computes, and saw nothing wrong. Nothing IS wrong with that
line in isolation. It becomes a test-label leak only when `Vw` is later used to RESTRICT a probe that is
scored on a held-out fold. The bug lives in the relationship between two distant intentions, not in any line,
and reading cannot see it.

Worse: the agent's own remedy was the delivery vehicle. It correctly said "readout beats a random subspace" is
near-tautological and recommended the class-mean subspace as the fair comparator. Building that comparator on
the existing (leaky) `Vw` promoted an inert bug into a headline.

What found it was a number that looked impossible. What would have found it *immediately* is an assertion.

A CAUTIONARY NOTE ABOUT THAT "IMPOSSIBLE" NUMBER, because it is the point of this module.
The leak surfaced as `decode_classmean_k` (0.349) exceeding `decode_obl_full` (0.247): a k-dimensional
projection out-decoding the d-dimensional space containing it. That was reported as impossible. **It is not.**
With L2 logistic regression the full space can represent any subspace solution at the same penalty (`V`
orthonormal implies `||V^T w|| = ||w||`), so its TRAINING objective is never worse -- but its TEST accuracy can
be, because the larger hypothesis class overfits noise directions. Restricting dimensions is not the same as
penalising norm.

Measured, leak-free, on Pythia-1.4B: at step 1000 `readout_k` = 0.171 beats `full` = 0.144, and `Vg` is fit on
a completely independent gradient stream. Nesting is therefore **NOT an invariant**, and encoding it as one
would fire on clean data forever.

So the invariants below are split:

  HARD  -- mathematically or definitionally necessary. Violation is a bug, always. Raise.
  SOFT  -- a strong smell with a known benign explanation. Return a diagnostic, never raise.

The leak's HARD invariant is not "nesting". It is: **a label-derived projection must be estimated from the
probe's training fold alone**, because otherwise it is not a restriction of the hypothesis class -- it is a
second, unregularised classifier that has seen the answers.

Usage:
    from invariants import (subspace_partition, chance_matches_arity, intervention_moved,
                            label_derived_projection_uses_train_only, p_above_permutation_floor,
                            in_unit_interval, nesting_smell)
Run `python analysis/invariants.py` for a self-test that reconstructs each historical bug.
"""
from __future__ import annotations
import math

import numpy as np


class InvariantError(AssertionError):
    """A necessary property of the measurement does not hold. The number is not wrong; it is meaningless."""


def _raise(name, msg, shipped):
    raise InvariantError(f"[{name}] {msg}\n    This invariant exists because we shipped: {shipped}")


# ---------------------------------------------------------------- HARD invariants


def label_derived_projection_uses_train_only(rows_used, tr, what="projection"):
    """A subspace that RESTRICTS a probe must be estimated from that probe's training fold.

    Otherwise it is not a restriction of the hypothesis class -- it is a second, unregularised classifier
    fit with the test fold's labels, and every accuracy measured inside it is optimistic.

    Label-DERIVED projections are the dangerous case: class means, LDA, supervised PCA, anything that touches
    `Y`. Label-BLIND ones (random subspaces, PCA of the unlabelled cloud, gradients from an independent trial
    stream) are safe -- which is exactly why the leak hid so long. The arm we distrusted was the clean one.
    """
    extra = set(np.asarray(rows_used).ravel().tolist()) - set(np.asarray(tr).ravel().tolist())
    if extra:
        _raise("train-only projection",
               f"{what} was estimated from {len(extra)} row(s) outside the probe's training fold "
               f"(e.g. {sorted(extra)[:5]}). It has seen the test labels.",
               "Vw (the class-mean subspace) fit on ALL rows, then used to restrict a probe scored on a "
               "held-out fold. It made the paper's fair comparator unfair, and it killed a headline.")


def subspace_partition(k, parts, what="B"):
    """Subspaces claimed to PARTITION a k-dimensional space must have dimensions summing to k.

    `parts` is {name: basis or dim}. Two generic k-dim subspaces of R^d intersect only at the origin, so
    `P_R B` and `(I - P_R) B` are BOTH rank k and cannot partition anything.
    """
    dims = {n: (v if isinstance(v, int) else len(v)) for n, v in parts.items()}
    tot = sum(dims.values())
    if tot != k:
        _raise("subspace partition",
               f"{what} has dim {k} but its claimed parts sum to {tot}: {dims}. They do not partition it.",
               "B_vis = basis(P_R B) and B_inv = basis((I-P_R) B), each rank 21 for a 21-dim B. "
               "21 + 21 != 21. The fix is principal vectors of (B, R).")


def chance_matches_arity(chance, n_options, metric="metric"):
    """A metric's chance level is 1/(number of options it actually chooses among).

    A probe classifying over the whole obligation pool has chance 1/|OBL|. A model doing an argmax restricted
    to the K obligations PRESENT in the prompt has chance 1/K. They are different numbers for different
    metrics and conflating them silently rescales every threshold built on them.
    """
    want = 1.0 / n_options
    if not math.isclose(chance, want, rel_tol=1e-6):
        _raise("chance level",
               f"{metric} chooses among {n_options} options, so its chance is {want:.4f}, not {chance:.4f}.",
               "PREREG_ablation.md set chance = 1/|OBL| = 0.04 for BOTH read-outs. model_acc is a K=6 forced "
               "choice, so its chance is 0.167. A total collapse (0.163) was scored as a FAILURE.")


def intervention_moved(H_before, H_after, P=None, rel_proj_max=2e-2, what="ablation"):
    """An intervention must actually change the quantity it claims to change.

    Returns (rel_change, rel_proj). Raises if the residual did not move, or if `P` was supposedly projected
    out and a material component of it survives. Thresholds are RELATIVE: the model runs in bf16 and a
    perfectly ablated residual retains ~1e-4 of the removed subspace.
    """
    H_before, H_after = np.asarray(H_before), np.asarray(H_after)
    moved = float(np.linalg.norm(H_after - H_before) / max(np.linalg.norm(H_before), 1e-12))
    if moved < 1e-6:
        _raise("intervention reached its target",
               f"{what} left the measured residual bit-identical (|dH|/|H| = {moved:.2e}).",
               "the ablation hook fired on the logits but `output_hidden_states[L]` was NOT the post-hook "
               "residual, so the probe read unablated activations. Probe accuracy was frozen at 0.644 across "
               "every arm, with sd exactly 0.000 over five random draws.")
    rel_proj = float("nan")
    if P is not None:
        P = np.asarray(P)
        rel_proj = float(np.linalg.norm(H_after @ P.T) / max(np.linalg.norm(H_after), 1e-12))
        if rel_proj > rel_proj_max:
            _raise("subspace actually removed",
                   f"{what} left {rel_proj:.2e} of the removed subspace in the residual "
                   f"(bar {rel_proj_max:.0e}).",
                   "the same hook bug -- and, later, an ABSOLUTE 1e-3 bar that rejected a correct ablation "
                   "because bf16 leaves ~1e-4 behind. Relative, not absolute.")
    return moved, rel_proj


def p_above_permutation_floor(p, n, what="p-value"):
    """At n points the smallest attainable two-sided permutation p is 2/n!. Anything below it is impossible."""
    floor = 2.0 / math.factorial(n)
    if p < floor * (1 - 1e-9):
        _raise("permutation floor",
               f"{what} = {p:.3e} at n={n}, but the floor is 2/{n}! = {floor:.3e}.",
               "'rho = 1.000, p < 1e-4' at n=7, where the floor is 3.97e-4. scipy returns exactly 0.0 for "
               "rho=1 because its t-approximation diverges.")


def in_unit_interval(name, v, lo=0.0, hi=1.0):
    """Accuracies, alignments and captured-variance fractions live in [0, 1]."""
    if not (lo - 1e-9 <= v <= hi + 1e-9):
        _raise("range", f"{name} = {v} is outside [{lo}, {hi}].",
               "nothing yet -- this one is prophylactic, and cheap.")


def variance_ratio_has_a_construction_matched_null(null_kind, subspace_is_label_derived, what="VIF"):
    """A removed-variance 'inflation factor' for a LABEL-DERIVED subspace must be measured against a null that
    holds the CONSTRUCTION fixed (same class-mean fit, permuted labels), not against the uniform-random k/d.

    `removed_var(S)/(k/d)` compares S to a UNIFORMLY RANDOM subspace. But a class-mean / difference-of-means
    subspace is a variance-seeking construction: fit at n<d, it concentrates variance ~ d/PR(Sigma) even under
    RANDOM labels. So the uniform null attributes to the concept an inflation the construction produces on its
    own. The only honest null permutes the labels and refits the same construction.
    """
    if subspace_is_label_derived and null_kind not in ("permutation", "construction-matched", "random-label"):
        _raise("variance-ratio null",
               f"{what} for a label-derived subspace was reported against the '{null_kind}' null. "
               f"A class-mean subspace fit at n<d inflates removed-variance ~d/PR even under permuted labels, "
               f"so only a label-permutation (construction-matched) null is valid.",
               "SPEC_vif.md reported VIF = removed_var/(k/d) = 32-113x as evidence the binding concept has "
               "anomalous variance. The pre-registered permutation gate (B2_vif_null_KILL.md) then showed "
               "VIF_honest = 0.84-1.03 on 7/7 checkpoints: the 'inflation' was the construction, not the concept.")


def whitening_is_well_conditioned(n, d, eff_rank, cond, what="whitening"):
    """A Sigma^{-1/2} used to whiten before ablation must be full-rank or shrunk: effective rank <= n, and a
    finite condition number. At n<d the empirical covariance is rank-deficient and 1/sqrt(lambda) amplifies
    pure sampling noise on the near-null directions, so the whitened concept direction points at noise.
    """
    if eff_rank > n:
        _raise("whitening rank",
               f"{what} claims effective rank {eff_rank} from n={n} samples in d={d}. The empirical covariance "
               f"has rank <= n-1; an effective rank above n means an unshrunk pseudo-inverse of a singular matrix.",
               "not shipped -- caught in the 48h plan's red-team before a whitened ablation arm was built. "
               "The fix is PCA-whitening in the top-r<n subspace, or Ledoit-Wolf shrinkage with gamma from X alone.")
    if not (cond == cond) or cond > 1e8:                       # NaN or explosive
        _raise("whitening conditioning",
               f"{what} covariance has condition number {cond:.3e} (bar 1e8). 1/sqrt(lambda) on the near-null "
               f"eigenvalues amplifies sampling noise; the whitened direction is dominated by it.",
               "not shipped -- prophylactic. Report cond(Sigma_hat) per model; shrink until it is finite.")


# ---------------------------------------------------------------- SOFT diagnostics (never raise)


def nesting_smell(full, subspaces, n_test, label_derived=(), z=2.0):
    """A LABEL-DERIVED subspace out-decoding the full space is a smell, not an impossibility.

    Returns a list of human-readable warnings. It is genuinely possible for a projection to beat the space
    containing it -- restricting dimensions is not the same as penalising norm, and the larger hypothesis
    class overfits noise directions. Measured leak-free: at step 1000, readout_k 0.171 > full 0.144, with Vg
    fit on an independent gradient stream.

    So we only complain when a LABEL-DERIVED subspace exceeds `full` by more than `z` binomial standard errors
    of the test set -- because that is the configuration in which a leak, rather than regularisation, is the
    likely cause. Even then: a warning, and a pointer at `label_derived_projection_uses_train_only`.
    """
    se = math.sqrt(0.25 / max(n_test, 1))
    out = []
    for name, acc in subspaces.items():
        if acc > full + z * se:
            kind = "LABEL-DERIVED" if name in label_derived else "label-blind"
            msg = (f"{name} ({acc:.3f}) exceeds full ({full:.3f}) by more than {z} SE ({z*se:.3f}).")
            if name in label_derived:
                msg += ("  It is label-derived: check that its projection was fit on the probe's TRAIN fold "
                        "only. This is how the Vw leak announced itself.")
            else:
                msg += ("  It is label-blind, so this is most likely regularisation, not a leak: a "
                        "lower-dimensional hypothesis class can generalise better. Report, do not panic.")
            out.append(f"[{kind}] " + msg)
    return out


# ---------------------------------------------------------------- self-test: reconstruct each shipped bug


def _selftest():
    import sys
    fails = []

    def expect_raise(fn, name):
        try:
            fn()
        except InvariantError as e:
            print(f"  CAUGHT  {name}\n          {str(e).splitlines()[0]}")
            return
        fails.append(name)
        print(f"  MISSED  {name}   <-- the invariant did not fire")

    print("Reconstructing each bug this repo shipped, and checking the invariant catches it.\n")
    expect_raise(lambda: label_derived_projection_uses_train_only(
        rows_used=np.arange(900), tr=np.arange(630), what="Vw (class-mean subspace)"),
        "Vw fit on all rows incl. the test fold's labels")
    expect_raise(lambda: subspace_partition(21, {"B_vis": 21, "B_inv": 21}),
        "B_vis and B_inv both rank 21 for a 21-dim B")
    expect_raise(lambda: chance_matches_arity(0.04, 6, metric="model_acc (argmax over K=6 present)"),
        "chance = 1/|OBL| used for a K-way forced choice")
    H = np.random.default_rng(0).standard_normal((50, 32))
    expect_raise(lambda: intervention_moved(H, H.copy(), what="ablation"),
        "ablation never reached the probe's residual")
    expect_raise(lambda: p_above_permutation_floor(1e-4, 7),
        "'p < 1e-4' at n=7, where the floor is 3.97e-4")
    expect_raise(lambda: in_unit_interval("alignment", 1.4),
        "an accuracy outside [0,1]")
    expect_raise(lambda: variance_ratio_has_a_construction_matched_null(
        "uniform", subspace_is_label_derived=True, what="VIF"),
        "VIF vs k/d (uniform null) for a class-mean subspace -- SPEC_vif's retracted headline")
    expect_raise(lambda: whitening_is_well_conditioned(n=1200, d=4096, eff_rank=4096, cond=1e12),
        "whitening at n<d with an unshrunk singular covariance")

    print("\nAnd the SOFT one, which must NOT fire on clean data:")
    clean = nesting_smell(full=0.144, subspaces={"readout_k": 0.171}, n_test=270,
                          label_derived=())          # Vg is label-blind wrt the probe's folds
    print(f"  leak-free step-1000 (readout_k 0.171 > full 0.144): "
          f"{'silent (correct)' if not clean else 'WARNED: ' + clean[0][:60]}")
    leaky = nesting_smell(full=0.247, subspaces={"classmean_k": 0.349}, n_test=270,
                          label_derived=("classmean_k",))
    print(f"  the historical leak (classmean_k 0.349 > full 0.247): "
          f"{'WARNED (correct)' if leaky else 'silent  <-- MISSED'}")
    if leaky:
        print(f"      {leaky[0]}")
    if not leaky:
        fails.append("nesting_smell on the leak")

    print()
    if fails:
        print(f"SELF-TEST FAILED on {len(fails)}: {fails}")
        return 1
    print("SELF-TEST PASSED: every shipped bug is caught, and the clean counterexample is not flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
