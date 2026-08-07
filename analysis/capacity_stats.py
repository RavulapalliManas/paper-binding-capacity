"""Every capacity number in the capacity paper, recomputed from the committed table.

Three corrections this script exists to enforce.

1. EXCLUDE MODELS THAT CANNOT HOLD ONE BINDING.  k* is the smallest K at which recall falls below the midpoint
   of the K=1 ceiling and chance.  If the K=1 ceiling is itself at chance, that midpoint is noise and k* is
   meaningless.  pythia-70m has a ceiling of 0.070 against a chance of 0.040.  Its k*=2 is not a measurement.
   It is also the sole source of the "12x range" the paper claimed: excluding it, k* spans 3..24, i.e. 8x.

2. THE RECIPE CLASSES ARE NOT DISJOINT IN RANGE.  OPT-6.7B is an old-recipe model at the censored ceiling
   (k*=24) and Qwen2.5-0.5B is a modern one at k*=3.  Both classes span 3..24.  "Old recipes cap at 2-6" is
   false.  The separation is in the MEDIAN, and that is what we report.

3. THIRTY MODELS ARE TEN FAMILIES.  Pythia and OLMo contribute six apiece.  Every comparison is therefore
   re-run with a bootstrap clustered on families and with a family-level test.

Reported, in this order: the exclusions, the range, the model-level test a reader would compute from the
released table, and then what that test becomes once the clustering is taken seriously.
"""
from __future__ import annotations
import json, os

import numpy as np
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "figdata.json")
DST = os.path.join(HERE, "..", "results", "capacity_stats.json")
CHANCE = 0.04
CEILING_FLOOR = 0.5          # a model must clear this at K=1 for its k* to mean anything
KSTAR_CENSOR = 24            # the entity pool caps the measurable k*
B = 10000


def main():
    g = json.load(open(SRC))["geom"]
    rng = np.random.default_rng(0)

    degenerate = [e for e in g if e["ceiling"] < CEILING_FLOOR]
    ok = [e for e in g if e["ceiling"] >= CEILING_FLOOR]
    censored = [e for e in ok if e["kstar"] >= KSTAR_CENSOR]

    print(f"CONTROLS FIRST   chance = {CHANCE}   ceiling floor = {CEILING_FLOOR}")
    print(f"  excluded, cannot hold one binding: "
          f"{[(e['label'], e['ceiling']) for e in degenerate] or 'none'}")
    print(f"  right-censored at k*={KSTAR_CENSOR} (entity-pool ceiling): "
          f"{[(e['label'], e['recipe']) for e in censored]}")
    print(f"  n = {len(ok)} models, {len({e['family'] for e in ok})} families\n")

    ks = [e["kstar"] for e in ok]
    print(f"RANGE  k* spans {min(ks)}..{max(ks)}  ->  {max(ks)/min(ks):.0f}x")
    allks = [e["kstar"] for e in g]
    print(f"       (including the degenerate model it would read {max(allks)/min(allks):.0f}x; "
          f"that factor is an artifact of one model that cannot do the task)\n")

    fams = sorted({e["family"] for e in ok})
    byf = {f: [e for e in ok if e["family"] == f] for f in fams}

    def arrs(key):
        o = np.array([e[key] for e in ok if e["recipe"] == "old"], float)
        m = np.array([e[key] for e in ok if e["recipe"] == "modern"], float)
        return o, m

    def famboot(key):
        out = []
        for _ in range(B):
            pick = rng.choice(fams, len(fams), replace=True)
            s = [e for f in pick for e in byf[f]]
            o = [e[key] for e in s if e["recipe"] == "old"]
            m = [e[key] for e in s if e["recipe"] == "modern"]
            if o and m:
                out.append(np.median(m) - np.median(o))
        out = np.array(out)
        return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())

    def famlevel(key):
        med = {f: float(np.median([e[key] for e in byf[f]])) for f in fams}
        rec = {f: byf[f][0]["recipe"] for f in fams}
        o = [med[f] for f in fams if rec[f] == "old"]
        m = [med[f] for f in fams if rec[f] == "modern"]
        return len(o), len(m), float(mannwhitneyu(m, o, alternative="two-sided")[1])

    rows = {}
    hdr = f"{'quantity':<10}{'old med':>9}{'modern med':>12}{'model p':>11}   {'family-clustered 95% CI':<26}{'family p':>9}"
    print(hdr); print("-" * len(hdr))
    for key in ["kstar", "pack", "d_bind", "offdiag"]:
        o, m = arrs(key)
        pm = float(mannwhitneyu(m, o, alternative="two-sided")[1])
        lo, hi, pz = famboot(key)
        no, nm, pf = famlevel(key)
        rows[key] = dict(old_median=float(np.median(o)), modern_median=float(np.median(m)),
                         n_old=len(o), n_modern=len(m), model_p=pm,
                         family_ci=[lo, hi], frac_boot_le_zero=pz,
                         family_p=pf, n_fam_old=no, n_fam_modern=nm,
                         survives_clustering=bool(lo > 0 or hi < 0))
        star = "" if (lo > 0 or hi < 0) else "   <-- CI touches zero"
        print(f"{key:<10}{np.median(o):>9.3f}{np.median(m):>12.3f}{pm:>11.1e}   "
              f"[{lo:+.3f}, {hi:+.3f}]{'':<8}{pf:>9.4f}{star}")

    print("\nREADING")
    print("  k* and packing efficiency survive family clustering.  Interference does not: its clustered")
    print("  interval touches zero.  Rest the recipe claim on capacity and packing; report interference")
    print("  as a weaker, consistent trend.")
    print("  Note pack = k*/d_bind contains k* in its numerator and is DESCRIPTIVE, not predictive.")
    print("  The non-tautological geometric predictors are d_bind and interference.")

    qwen = sorted([e for e in ok if e["family"] == "Qwen"], key=lambda e: e["label"])
    print("\nTHE QWEN LADDER, as actually measured (it is NOT monotone in scale)")
    for e in qwen:
        print(f"  {e['label']:<16} k* = {e['kstar']:<3} (ceiling {e['ceiling']})")

    out = dict(chance=CHANCE, ceiling_floor=CEILING_FLOOR, n=len(ok), n_families=len(fams),
               excluded=[e["label"] for e in degenerate],
               censored=[e["label"] for e in censored],
               kstar_min=min(ks), kstar_max=max(ks), kstar_range_factor=max(ks) / min(ks),
               qwen_ladder={e["label"]: e["kstar"] for e in qwen}, stats=rows)
    json.dump(out, open(DST, "w"), indent=1)
    print(f"\nwrote {DST}")


if __name__ == "__main__":
    main()
