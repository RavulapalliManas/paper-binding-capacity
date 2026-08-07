"""Compare the ksweep.py re-run against the frozen capacity table (figdata.json["geom"]).

Prints per-model k* (frozen vs re-run), the recipe medians both ways, and Spearman agreement.
The frozen table stays frozen; this is the agreement check the Reproducibility Statement's open
item exists to enable. Run after pulling results/ksweep_rerun/*.json from the GPU box:

    python analysis/compare_ksweep.py
"""
from __future__ import annotations
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def median(xs):
    xs = sorted(xs); n = len(xs)
    return None if n == 0 else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def main():
    frozen = {m["label"]: m for m in json.load(open(os.path.join(RES, "figdata.json")))["geom"]}
    rerun = {}
    for p in sorted(glob.glob(os.path.join(RES, "ksweep_rerun", "*.json"))):
        r = json.load(open(p))
        if r.get("done"):
            rerun[r["label"]] = r

    rows, pairs = [], []
    for label, f in sorted(frozen.items()):
        r = rerun.get(label)
        new_k = r["kstar"] if r else None
        cen = "^" if r and r.get("right_censored") else ""
        unusable = " (unusable: K=1 at chance)" if r and not r.get("usable") else ""
        rows.append(f"{label:15s} {f['recipe']:7s} frozen={f['kstar']:>3}  rerun={str(new_k):>4}{cen}{unusable}")
        if r and r.get("usable") and new_k is not None:
            pairs.append((f["kstar"], new_k, f["recipe"]))
    print("\n".join(rows))
    extra = sorted(set(rerun) - set(frozen))
    for label in extra:
        r = rerun[label]
        print(f"{label:15s} {r['recipe']:7s} frozen=  -  rerun={str(r['kstar']):>4}{'^' if r.get('right_censored') else ''}  (NEW)")

    if pairs:
        for tag in ("old", "modern"):
            fz = [a for a, _, rec in pairs if rec == tag]
            rr = [b for _, b, rec in pairs if rec == tag]
            if fz:
                print(f"\nmedian k* [{tag}]  frozen={median(fz)}  rerun={median(rr)}  (n={len(fz)})")
        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr([a for a, _, _ in pairs], [b for _, b, _ in pairs])
            print(f"\nSpearman(frozen, rerun) over n={len(pairs)}: rho={rho:.3f} (p={p:.4f}, asymptotic -- "
                  f"indicative only; use exact_stats for any paper claim)")
        except ImportError:
            pass
    print(f"\n{len(rerun)}/{len(frozen)} frozen models re-measured; {len(extra)} new.")


if __name__ == "__main__":
    main()
