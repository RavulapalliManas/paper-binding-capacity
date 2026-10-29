"""Which value does a model give when it answers wrong, and where in the list does that value sit?

Reads the pool logits that gpu_capacity_dev.py saved (no model is run).  The answer is the top pool
logit, as in the capacity measurements.  For each model, condition and load it reports:
  n, n_wrong
  listed      wrong answers that are the value of another listed entity (a swap)
  unlisted    wrong answers that are not listed in the prompt
  listed_exp  the share of listed answers expected if a wrong answer were a uniform draw from the pool
              values other than gold (mean over the wrong trials of #listed-not-gold / (|V| - 1))
  dist        among swaps, the slot distance d = j - qi from the queried entity to the entity whose
              value was given (nearest such slot if the value repeats): counts for d = -1, +1, |d| = 2,
              |d| = 3, |d| >= 4
  adj_exp     the share of swaps at |d| = 1 expected if the swapped slot were uniform over the other
              K - 1 slots (mean over swap trials of #neighbours(qi) / (K - 1))
  by_pos      recall by the queried slot, in five bins of qi / (K - 1), for K >= 8
Checks: listed + unlisted == n_wrong for every row (assert).

PEAK MEMORY.  One condition of one model at a time: 8,640 x 67 float32 logits plus 8,640 x 64 int16
values, under 5 MB.

Usage: python p3_error_anatomy.py --inp <capacity_fp32 root> --out anatomy/p3_error_anatomy.json
"""
import argparse
import glob
import json
import os

import numpy as np


def anatomy(LG, gold, K, qi, vals):
    nV = LG.shape[1]
    emitted = LG.argmax(1)
    rows = []
    for k in sorted(set(K.tolist())):
        m = np.where(K == k)[0]
        r = dict(K=int(k), n=len(m), n_wrong=0, listed=0, unlisted=0, listed_exp=None,
                 dist={"-1": 0, "+1": 0, "2": 0, "3": 0, ">=4": 0}, adj_exp=None, by_pos=None)
        lexp, aexp = [], []
        for i in m:
            if emitted[i] == gold[i]:
                continue
            r["n_wrong"] += 1
            v = vals[i, :k]
            lexp.append(len(set(v.tolist()) - {int(gold[i])}) / (nV - 1))
            slots = np.where(v == emitted[i])[0]
            if len(slots) == 0:
                r["unlisted"] += 1
                continue
            r["listed"] += 1
            d = int(slots[np.argmin(np.abs(slots - qi[i]))] - qi[i])
            key = {-1: "-1", 1: "+1"}.get(d, "2" if abs(d) == 2 else "3" if abs(d) == 3 else ">=4")
            r["dist"][key] += 1
            aexp.append((int(qi[i] > 0) + int(qi[i] < k - 1)) / (k - 1))
        assert r["listed"] + r["unlisted"] == r["n_wrong"], r
        r["listed_exp"] = round(float(np.mean(lexp)), 4) if lexp else None
        r["adj_exp"] = round(float(np.mean(aexp)), 4) if aexp else None
        if k >= 8:
            b = np.minimum((qi[m] / (k - 1) * 5).astype(int), 4)
            ok = emitted[m] == gold[m]
            r["by_pos"] = [round(float(ok[b == j].mean()), 4) if (b == j).any() else None for j in range(5)]
            r["by_pos_n"] = [int((b == j).sum()) for j in range(5)]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true", help="check adjacency counting and unlisted-error controls")
    ap.add_argument("--conds", default="D0,D256")
    a = ap.parse_args()
    if a.selftest:
        K = np.full(3, 3, dtype=np.int16)
        qi = np.arange(3, dtype=np.int16)
        values = np.tile(np.arange(3), (3, 1))
        logits = np.eye(4)[[1, 0, 1]]
        row = anatomy(logits, qi, K, qi, values)[0]
        assert row["listed"] == 3 and row["adj_exp"] == 0.6667, row
        row = anatomy(np.eye(4)[[3, 3, 3]], qi, K, qi, values)[0]
        assert row["listed"] == 0 and row["unlisted"] == 3 and row["adj_exp"] is None, row
        print("PASS: integer neighbor counts and unlisted-error negative control")
        return
    if not a.inp or not a.out:
        ap.error("--inp and --out are required unless --selftest is used")
    out = {}
    for jp in sorted(glob.glob(os.path.join(a.inp, "*", "main.json"))):
        rec = json.load(open(jp))
        with np.load(jp[:-5] + ".npz") as Z:
            for c in a.conds.split(","):
                rows = anatomy(Z[f"{c}_logits"], Z[f"{c}_gold"].astype(int), Z[f"{c}_K"].astype(int),
                               Z[f"{c}_qi"].astype(int), Z[f"{c}_vals"].astype(int))
                out[f"{rec['label']}|{c}"] = dict(label=rec["label"], cond=c,
                                                 n_obligations=rec["n_obligations"], rows=rows)
        print("done", rec["label"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(models=out, _provenance="analysis/p3_error_anatomy.py from gpu_capacity_dev.py "
                   "float32 outputs (pool logits)"), open(a.out, "w"), indent=1)
    print("wrote", a.out, len(out), "model-conditions", flush=True)


if __name__ == "__main__":
    main()
