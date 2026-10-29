"""Probe capacity next to output capacity, from the hidden states that gpu_capacity_dev.py saved.

For each model, checkpoint, condition and saved layer, a probe (StandardScaler + multinomial logistic
regression, C=0.5, as in the earlier probe runs) predicts the queried value from the last-token
residual stream.  It is fit with three folds over seeds (test seeds {0,1}, {2,3}, {4,5} in turn),
pooled over all loads, so every trial gets an out-of-fold prediction.  Per load it reports:
  probe recall          top value of the probe over the pool
  probe recall, listed  top value of the probe among the values listed in the prompt
  output recall         the published readout (top pool logit), and the same among listed values
  probe right on wrong  share of the model's wrong answers on which the probe's pool top value is gold
and K50 of the output curve and of the probe curve, with the estimator of the capacity measurements
(capacity_curve.fit_k50, chance 1/|V|).

CONTROLS, at layer L/2:
  shuffled   labels permuted within the training folds: must sit at chance over the pool
  prequery   the state at the last token before the question: it cannot know the queried entity,
             so among listed values it should sit near the guess 1/K + (K-1)/(K|V|)
  entity     the same states, predicting the queried entity instead of its value (positive control)
  foil       the same states, predicting the value of the next listed entity, which was not queried
             (loads K >= 2): a state that holds only the listed values reads it as well as the queried
             value; among listed values other than the queried one the guess is 1/(K-1)
  prequery is skipped when <cond>_preq holds the token index, as in the float32 runs of 2026-10-24.

SLOT, at every saved layer: the same states, predicting the list position of the queried entity
(0 to K-1).  The position is random per trial, so it is readable only if the model has retrieved
where the queried entity stands in the list; the guess is 1/K.  Read it only for K <= 16: positions
above 16 occur only in the K = 32 and 64 trials, about 6 training trials per position, and in the
planted test the slot probe falls to 0.89 at K = 32 for that reason alone.

PEAK MEMORY.  Per task: trials x d x 4 bytes (8,640 x 4,096 x 4 B = 0.14 GB) plus the model's
coefficients; --jobs tasks at once (default 24: about 4 GB).  One process writes the output file.

Usage: python p3_probe_capacity.py --inp <gpu_capacity_dev out root> --out results/p3_probe_capacity.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def oof_probe(X, y, seed, n_classes, rng=None):
    """Out-of-fold class probabilities over the pool (n x n_classes); classes absent in a fold get 0."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    P = np.zeros((len(y), n_classes), np.float32)
    # three folds over the seeds present: {0,1},{2,3},{4,5} with six seeds, {0},{1},{2} with three
    for grp in np.array_split(np.unique(seed), 3):
        te = np.isin(seed, grp)
        tr = ~te
        if not te.any():
            continue
        ytr = y[tr].copy()
        if rng is not None:
            ytr = rng.permutation(ytr)
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(X[tr]), ytr)
        P[np.ix_(te, clf.classes_)] = clf.predict_proba(sc.transform(X[te]))
    return P


def listed_top(S, vals):
    """Top-scoring value among the listed ones (vals: n x 64, padded with -1)."""
    v = np.where(vals >= 0, vals, 0)
    s = np.where(vals >= 0, np.take_along_axis(S, v, 1), -np.inf)
    return np.take_along_axis(v, s.argmax(1)[:, None], 1)[:, 0]


def task(path, cond, li, layer, control, oof_dir=None):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    Z = np.load(path)
    nV = len(Z["obl_words"])
    y, K, seed, vals = (Z[f"{cond}_gold"].astype(int), Z[f"{cond}_K"], Z[f"{cond}_seed"],
                        Z[f"{cond}_vals"].astype(int))
    if control == "prequery":
        X = Z[f"{cond}_preq"].astype(np.float32)
    elif control in ("logits", "centered"):
        # readout baselines from the model's own scores over the 67 values: a probe on them, and
        # the argmax after subtracting each value's mean score on the training folds
        X = Z[f"{cond}_logits"].astype(np.float32)
    else:
        X = Z[f"{cond}_hidden"][:, li, :].astype(np.float32)
    if control == "entity":
        ents = Z[f"{cond}_ents"].astype(int)
        y = ents[np.arange(len(y)), Z[f"{cond}_qi"].astype(int)]
        nV = len(Z["ent_words"])
    qi = Z[f"{cond}_qi"].astype(int)
    keep = np.ones(len(y), bool)
    if control == "foil":
        # the value of the next listed entity, which was not queried; needs K >= 2
        keep = K >= 2
        y = vals[np.arange(len(y)), (qi + 1) % np.maximum(K, 1)]
    if control == "slot":
        y, nV = qi, vals.shape[1]
    X, y, K, seed, vals, qi = X[keep], y[keep], K[keep], seed[keep], vals[keep], qi[keep]
    rng = np.random.default_rng(7) if control == "shuffled" else None
    if control == "centered":
        P = np.zeros_like(X)
        for grp in np.array_split(np.unique(seed), 3):
            te = np.isin(seed, grp)
            P[te] = X[te] - X[~te].mean(0)
    else:
        P = oof_probe(X, y, seed, nV, rng)
    top = P.argmax(1)
    out = dict(cond=cond, layer=int(layer), control=control, rows=[])
    LG = Z[f"{cond}_logits"][keep]
    gold = Z[f"{cond}_gold"][keep]
    emitted = LG.argmax(1)
    top_l = listed_top(P, vals) if control not in ("entity", "slot") else top
    em_l = listed_top(LG, vals)
    if oof_dir and control in (None, "logits", "centered"):
        # per-trial out-of-fold predictions, in the trial order of the input file
        lab = os.path.basename(os.path.dirname(path)) + "_" + os.path.basename(path)[:-4]
        tag = f"L{layer}" if control is None else control
        np.savez(os.path.join(oof_dir, f"{lab}_{cond}_{tag}.npz"), top=top.astype(np.int16),
                 top_listed=top_l.astype(np.int16), p_gold=P[np.arange(len(y)), y].astype(np.float32))
    for k in sorted(set(K.tolist())):
        m = K == k
        wrong = m & (emitted != gold)
        out["rows"].append(dict(
            K=int(k), n=int(m.sum()),
            probe=float((top[m] == y[m]).mean()),
            probe_listed=float((top_l[m] == y[m]).mean()),
            output=float((emitted[m] == gold[m]).mean()),
            output_listed=float((em_l[m] == gold[m]).mean()),
            n_wrong=int(wrong.sum()),
            probe_right_on_wrong=float((top[wrong] == y[wrong]).mean()) if wrong.any() else None))
    return path, out


def k50(rows, key, nV):
    from capacity_curve import fit_k50
    Ks = np.array([r["K"] for r in rows], float)
    rec = np.array([r[key] for r in rows], float)
    try:
        f = fit_k50(Ks, rec, 1.0 / nV, n_trials=np.array([r["n"] for r in rows]))
        return dict(K50=round(float(f["K50"]), 3), censored=bool(f["censored"]))
    except (ValueError, RuntimeError) as e:
        return dict(K50=None, censored=None, why=str(e)[:120])


def planted(root):
    """Two synthetic models with known answers, in the gpu_capacity_dev.py file format.
    bound:   the state holds the queried value, entity and list position plus the listed values
             -> value, entity and slot probes near 1
    unbound: the state holds only the listed values, not which one is queried -> probes at the guess
    The pre-question state holds only the listed values in both.  Shuffled labels must sit at chance."""
    rng = np.random.default_rng(0)
    nV, nE, d = 67, 67, 64
    E, F = rng.standard_normal((nV, d)), rng.standard_normal((nE, d))
    S = rng.standard_normal((64, d))
    Ks = [1, 2, 4, 8, 16, 32]
    for world in ("bound", "unbound"):
        rows = dict(K=[], seed=[], gold=[], qi=[], vals=[], ents=[], H=[], P=[])
        for K in Ks:
            for s in range(6):
                for _ in range(48):
                    v = rng.integers(0, nV, K)
                    e = rng.choice(nE, K, replace=False)
                    q = int(rng.integers(0, K))
                    bag = E[v].mean(0)
                    h = bag + rng.standard_normal(d) * 0.3
                    if world == "bound":
                        h = h + 2.0 * E[v[q]] + 2.0 * F[e[q]] + 2.0 * S[q]
                    pad = lambda x: np.concatenate([x, -np.ones(64 - K, int)])
                    rows["K"].append(K); rows["seed"].append(s); rows["gold"].append(v[q])
                    rows["qi"].append(q); rows["vals"].append(pad(v)); rows["ents"].append(pad(e))
                    rows["H"].append(np.stack([h] * 4)); rows["P"].append(bag + rng.standard_normal(d) * 0.3)
        A = {k: np.array(x) for k, x in rows.items()}
        LG = rng.standard_normal((len(A["K"]), nV)).astype(np.float32)
        os.makedirs(os.path.join(root, world), exist_ok=True)
        c = "D0"
        np.savez(os.path.join(root, world, "main.npz"), obl_words=np.array([f"v{i}" for i in range(nV)]),
                 ent_words=np.array([f"e{i}" for i in range(nE)]), layers=np.array([1, 2, 3, 4]),
                 **{f"{c}_logits": LG, f"{c}_hidden": A["H"].astype(np.float16),
                    f"{c}_preq": A["P"].astype(np.float16), f"{c}_K": A["K"], f"{c}_seed": A["seed"],
                    f"{c}_gold": A["gold"], f"{c}_qi": A["qi"], f"{c}_vals": A["vals"],
                    f"{c}_ents": A["ents"]})
        json.dump(dict(label=world, revision="main", embed_md5=world, n_layers=4, layers_saved=[1, 2, 3, 4],
                       preq_layer=2, conds=[c], n_obligations=nV), open(os.path.join(root, world, "main.json"), "w"))


def check_planted(out):
    R = json.load(open(out))["checkpoints"]
    nV = 67
    ok = True
    for world in ("bound", "unbound"):
        runs = R[f"{world}|main"]["runs"]
        for r in runs:
            for row in r["rows"]:
                K = row["K"]
                guess = 1 / K + (K - 1) / (K * nV)
                if r["control"] is None and r["layer"] == 2:
                    want = row["probe_listed"] >= 0.9 if world == "bound" else abs(row["probe_listed"] - guess) <= 0.08
                elif r["control"] == "prequery":
                    want = abs(row["probe_listed"] - guess) <= 0.08
                elif r["control"] == "shuffled":
                    want = row["probe"] <= 0.05
                elif r["control"] == "entity":
                    want = row["probe"] >= 0.9 if world == "bound" else row["probe"] <= 0.1
                elif r["control"] == "foil":
                    want = K < 2 or row["probe_listed"] <= 1 / (K - 1) + 0.08
                elif r["control"] == "slot" and r["layer"] == 2 and 4 <= K <= 16:
                    want = row["probe"] >= 0.9 if world == "bound" else row["probe"] <= 1 / K + 0.1
                else:
                    continue
                if not want:
                    ok = False
                    print(f"[planted] FAIL {world} control={r['control']} K={K} {row}", flush=True)
    print(f"[planted] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--planted", action="store_true", help="write two synthetic models to --inp, fit, check")
    ap.add_argument("--readout-baselines", action="store_true",
                    help="only a probe on the 67 value logits and the mean-centred logit argmax")
    ap.add_argument("--save-oof", default=None, help="directory for per-trial out-of-fold predictions")
    ap.add_argument("--controls-at", type=int, default=None,
                    help="run only the shuffled, foil and entity controls, at layers_saved[i]")
    ap.add_argument("--ctl-conds", default="D0,D256",
                    help="conditions that get controls and readout baselines (frame_test adds verbatim, shots2)")
    a = ap.parse_args()
    ctl_conds = tuple(a.ctl_conds.split(","))
    if a.planted:
        planted(a.inp)
    from joblib import Parallel, delayed
    jobs = []
    meta = {}
    for jp in sorted(glob.glob(os.path.join(a.inp, "*", "*.json"))):
        if os.path.basename(jp).startswith("steer_"):
            continue
        rec = json.load(open(jp))
        if rec.get("duplicate_of_previous"):
            continue
        npz = jp[:-5] + ".npz"
        meta[npz] = rec
        layers = rec["layers_saved"]
        with np.load(npz) as Z:
            has_preq = Z[f"{rec['conds'][0]}_preq"].ndim == 2
        if a.readout_baselines:
            for cond in [c for c in rec["conds"] if c in ctl_conds]:
                for ctl in ("logits", "centered"):
                    jobs.append((npz, cond, 0, -1, ctl))
            continue
        if a.controls_at is not None:
            li = a.controls_at
            for cond in [c for c in rec["conds"] if c in ctl_conds]:
                for ctl in ("shuffled", "entity", "foil"):
                    jobs.append((npz, cond, li, layers[li], ctl))
            continue
        for cond in rec["conds"]:
            for li, layer in enumerate(layers):
                jobs.append((npz, cond, li, layer, None))
                if cond in ctl_conds:
                    jobs.append((npz, cond, li, layer, "slot"))
            li = layers.index(rec["preq_layer"])
            if cond in ctl_conds:
                for ctl in ("shuffled", "entity", "foil") + (("prequery",) if has_preq else ()):
                    jobs.append((npz, cond, li, rec["preq_layer"], ctl))
    print(f"[probe-capacity] {len(jobs)} tasks over {len(meta)} checkpoints", flush=True)
    if a.save_oof:
        os.makedirs(a.save_oof, exist_ok=True)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(task)(*j, oof_dir=a.save_oof) for j in jobs)
    out = {}
    for path, r in res:
        rec = meta[path]
        nV = rec["n_obligations"]
        key = f"{rec['label']}|{rec['revision']}"
        o = out.setdefault(key, dict(label=rec["label"], revision=rec["revision"],
                                     n_layers=rec["n_layers"], embed_md5=rec["embed_md5"], runs=[]))
        if r["control"] is None:
            r["K50_probe"] = k50(r["rows"], "probe", nV)
            r["K50_output"] = k50(r["rows"], "output", nV)
        o["runs"].append(r)
    with open(a.out, "w") as f:
        json.dump(dict(checkpoints=out, _provenance="analysis/p3_probe_capacity.py from "
                       "gpu_capacity_dev.py outputs"), f, indent=1, sort_keys=True)
    print("wrote", a.out, flush=True)
    if a.planted:
        sys.exit(0 if check_planted(a.out) else 1)


if __name__ == "__main__":
    main()
