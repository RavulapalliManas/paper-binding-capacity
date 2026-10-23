"""Why activation steering fails and prompt re-injection works -- and a steering method that should work.

Two questions, one script.

(1) ROUTING FAILURE vs READOUT MISALIGNMENT.
    The gap has two candidate causes and they are distinguishable by probing at EVERY layer:
      routing failure      : the binding is present at read-layer l*, and destroyed by layers l*->L.
      readout misalignment : it reaches h_L but lands ~orthogonal to the obligation-difference
                             subspace span{W_U[o_i] - W_U[o_j]}, so the unembedding cannot see it.
    If decodability PERSISTS to h_L while the argmax is wrong  -> misalignment.
    If decodability DECAYS across l*->L                        -> routing failure.
    This matters because misalignment is the regime in which an activation-space fix can work at all.

(2) FLAT-VECTOR vs ALONG-MANIFOLD STEERING.
    The literature reports activation steering failing to fix this class of error (0/4 in 2603.18353,
    0/7 in 2604.13068) while prompt-level re-injection works (+0.20, ours). Goodfire's manifold-steering
    result says why that might be: a straight linear direction "cuts across the manifold and produces
    noisy, off-target effects." So we contrast, at the same layer, same norm, same trials:

      linear_w    delta = w_c - w_e   the probe's decision-boundary NORMAL. This is what "add a
                                      steering vector" means, and it points off the data manifold.
      manifold_mu delta = mu_c - mu_e the class-conditional CENTROID difference in activation space.
                                      A displacement between two points that lie on the manifold.
      random      matched-norm gaussian. The control that makes any effect above interpretable.
      unembed     delta = W_U[c] - W_U[e] applied at the FINAL layer. Trivially works; a ceiling, and a
                                      check that our intervention plumbing is correct.

    Reported on the FAILURE set (does it repair?) and on the CORRECT set (does it damage?). A method
    that repairs failures by destroying successes has not fixed anything.

Usage: python steer.py --model EleutherAI/pythia-6.9b --out results/steer_p69
"""
from __future__ import annotations
import argparse, json, os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probe_battery_v2 import Battery2, Cfg


def log(*a): print("[steer]", *a, flush=True)


def get_blocks(model):
    for path in ("model.layers", "gpt_neox.layers", "transformer.h", "model.decoder.layers"):
        o = model
        try:
            for p in path.split("."): o = getattr(o, p)
            return o
        except AttributeError:
            continue
    raise RuntimeError("cannot locate transformer blocks")


class Steer:
    def __init__(self, model, revision=None, dequantize=False, trust=False):
        self.b = Battery2(Cfg(model=model, out="/tmp/_unused", revision=revision,
                              dequantize=dequantize, trust_remote_code=trust))
        self.tok, self.model, self.dev = self.b.tok, self.b.model, self.b.dev
        self.ent, self.obl, self.obl_ids = self.b.ent, self.b.obl, self.b.obl_ids
        self.nL = self.b.nlayers
        self.blocks = get_blocks(self.model)
        self.q_pre = self.tok(" The task for", add_special_tokens=False)["input_ids"]
        self.q_suf = self.tok(" is:", add_special_tokens=False)["input_ids"]

    def build(self, rng, K, distract):
        ents = rng.sample(self.ent, K); obls = rng.sample(self.obl, K)
        ids = []
        for (e, _), (o, _) in zip(ents, obls):
            ids += self.tok(f"{e}:{o}.", add_special_tokens=False)["input_ids"]
        j = rng.randrange(K)
        seq = ids + distract + self.q_pre + [ents[j][1]] + self.q_suf
        return seq, obls[j][1], [o[1] for o in obls], j

    def _pad(self, seqs):
        maxlen = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), maxlen), self.tok.pad_token_id, dtype=torch.long)
        m = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, maxlen - len(s):] = torch.tensor(s); m[i, maxlen - len(s):] = 1
        return ids.to(self.dev), m.to(self.dev)

    @torch.no_grad()
    def collect(self, K, D, cond, n, seed=0, batch=24):
        rng = random.Random(9 + seed)
        distract = self.b._distract(cond, D, random.Random(1234), self.model.config.vocab_size) if D else []
        seqs, Y, PRES, SLOT = [], [], [], []
        for _ in range(n):
            s, y, pres, j = self.build(rng, K, distract)
            seqs.append(s); Y.append(y); PRES.append(pres); SLOT.append(j)
        F, LG = [], []
        for b in range(0, len(seqs), batch):
            ids, m = self._pad(seqs[b:b + batch])
            out = self.model(input_ids=ids, attention_mask=m, output_hidden_states=True)
            F.append(np.stack([h[:, -1, :].float().cpu().numpy() for h in out.hidden_states], axis=1))
            LG.append(out.logits[:, -1, :].float().cpu().numpy())
        return (np.concatenate(F), np.array(Y), np.array(PRES), np.array(SLOT),
                np.concatenate(LG), seqs)

    # ---------- (1) per-layer decodability: routing vs misalignment ----------
    def per_layer(self, F, Y, tr, te, fail_te, fail_va):
        """Probe every layer. Fit on tr. Report on te. Select l* on va (never on te)."""
        accs, accs_fail, accs_fail_va = [], [], []
        for L in range(F.shape[1]):
            sc = StandardScaler().fit(F[tr, L, :])
            try:
                clf = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(F[tr, L, :]), Y[tr])
                accs.append(float((clf.predict(sc.transform(F[te, L, :])) == Y[te]).mean()))
                accs_fail.append(float((clf.predict(sc.transform(F[fail_te, L, :])) == Y[fail_te]).mean())
                                 if fail_te.size else float("nan"))
                accs_fail_va.append(float((clf.predict(sc.transform(F[fail_va, L, :])) == Y[fail_va]).mean())
                                    if fail_va.size else float("nan"))
            except Exception:
                accs.append(float("nan")); accs_fail.append(float("nan")); accs_fail_va.append(float("nan"))
        return accs, accs_fail, accs_fail_va

    # ---------- (2) steering ----------
    def _hook(self, layer_ix, delta):
        """delta: [B, d] added to the LAST position of block `layer_ix`'s output."""
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h[:, -1, :] = h[:, -1, :] + delta.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return self.blocks[layer_ix].register_forward_hook(fn)

    @torch.no_grad()
    def run_steered(self, seqs, deltas, layer_ix, batch=16):
        """deltas: [n, d] numpy. Returns logits at last position."""
        outs = []
        for b in range(0, len(seqs), batch):
            ids, m = self._pad(seqs[b:b + batch])
            d = torch.tensor(deltas[b:b + batch], device=self.dev, dtype=torch.float32)
            h = self._hook(layer_ix, d)
            try:
                o = self.model(input_ids=ids, attention_mask=m)
                outs.append(o.logits[:, -1, :].float().cpu().numpy())
            finally:
                h.remove()
        return np.concatenate(outs)


def acc_from_logits(LG, PRES, Y, idx):
    ok = []
    for k, i in enumerate(idx):
        row = LG[k][PRES[i]]
        ok.append(int(PRES[i][int(row.argmax())] == Y[i]))
    return float(np.mean(ok)) if ok else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--cond", default="code"); ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dequantize", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    alphas = [float(x) for x in a.alphas.split(",")]

    s = Steer(a.model, a.revision, a.dequantize, a.trust_remote_code)
    F, Y, PRES, SLOT, LG, seqs = s.collect(a.K, a.D, a.cond, a.n, seed=a.seed, batch=a.batch)
    n = len(Y); idx = np.arange(n); np.random.RandomState(a.seed).shuffle(idx)
    n3 = n // 3; tr, va, te = idx[:n3], idx[n3:2 * n3], idx[2 * n3:]

    # who does the model get wrong?
    emitted = np.array([PRES[i][int(LG[i][PRES[i]].argmax())] for i in range(n)])
    wrong = emitted != Y
    fail_te = te[wrong[te]]; corr_te = te[~wrong[te]]; fail_va = va[wrong[va]]
    chance = 1.0 / len(s.obl)          # the probe is |OBL|-way, not K-way
    log(f"n={n} failures={wrong.sum()} ({wrong.mean():.3f})  test failures={len(fail_te)} correct={len(corr_te)}")
    log(f"probe is {len(s.obl)}-way; chance = {chance:.3f}")

    # ---- (1) per-layer decodability. l* selected on the VALIDATION failures, reported on test. ----
    accs, accs_fail, accs_fail_va = s.per_layer(F, Y, tr, te, fail_te, fail_va)
    Lstar = int(np.nanargmax(np.array(accs_fail_va)))       # <- selection never touches te
    LfinaI = F.shape[1] - 1
    log(f"PER-LAYER probe acc (all-test): {[round(x,3) for x in accs]}")
    log(f"PER-LAYER probe acc (failures, test): {[round(x,3) for x in accs_fail]}")
    log(f"  l* (selected on VAL failures) = {Lstar}/{LfinaI}   "
        f"test acc@l*={accs_fail[Lstar]:.3f}  test acc@final={accs_fail[-1]:.3f}  chance={chance:.3f}")
    ratio = (accs_fail[-1] - chance) / max(accs_fail[Lstar] - chance, 1e-9)
    log(f"  above-chance decodability retained at h_L = {ratio:.3f}")
    log("  CAVEAT: at h_L a linear probe can nearly read the model's own output, so on FAILURE trials it")
    log("  is biased toward predicting the emitted obligation. Low acc@final is what that predicts; we")
    log("  therefore read the SHAPE (peak mid-network, monotone decay) rather than the endpoint alone.")
    log(f"  => {'READOUT MISALIGNMENT (survives to h_L)' if ratio > 0.7 else 'ROUTING FAILURE (largely destroyed by l*->L)'}"
        + ("  [partial: still above chance]" if accs_fail[-1] > 1.5 * chance else ""))
    json.dump(dict(per_layer_all=accs, per_layer_fail=accs_fail, per_layer_fail_val=accs_fail_va,
                   lstar=Lstar, final_layer=LfinaI, chance=chance, retained_above_chance=ratio,
                   n_fail=int(len(fail_te))),
              open(f"{a.out}/per_layer.json", "w"), indent=2)

    # ---- probe + centroids at l* (fit on train only) ----
    sc = StandardScaler().fit(F[tr, Lstar, :])
    clf = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(F[tr, Lstar, :]), Y[tr])
    cls = {c: k for k, c in enumerate(clf.classes_)}
    W = clf.coef_ / np.where(sc.scale_ > 1e-8, sc.scale_, 1.0)     # decision-normal in RAW space
    mu = {}
    for c in clf.classes_:
        m = F[tr][Y[tr] == c, Lstar, :]
        if len(m): mu[c] = m.mean(0)
    hnorm = float(np.linalg.norm(F[te, Lstar, :], axis=1).mean())
    log(f"steering at layer {Lstar}; mean ||h||={hnorm:.1f}")

    def unit(v):
        nv = np.linalg.norm(v)
        return v / nv if nv > 1e-9 else v

    rng = np.random.RandomState(7 + a.seed)
    results = {}
    # baselines
    base_fail = acc_from_logits(np.stack([LG[i] for i in fail_te]), PRES, Y, fail_te)
    base_corr = acc_from_logits(np.stack([LG[i] for i in corr_te]), PRES, Y, corr_te)
    results["baseline"] = dict(repair=base_fail, damage_keep=base_corr)
    log(f"BASELINE  failures-fixed={base_fail:.3f}  correct-kept={base_corr:.3f}")

    # The probe's DECODED label on each trial, restricted to the K present obligations. This is what an
    # inference-time method actually has. Arms using Y[i] are ORACLE arms: they measure whether layer l*
    # causally controls the output, NOT whether a repair method exists.
    Praw = clf.predict_proba(sc.transform(F[:, Lstar, :]))
    decoded = np.empty(n, dtype=Y.dtype)
    for i in range(n):
        pres = PRES[i]
        pp = np.array([Praw[i][cls[p]] if p in cls else 0.0 for p in pres])
        decoded[i] = pres[int(pp.argmax())]
    dec_acc_fail = float((decoded[fail_te] == Y[fail_te]).mean())
    log(f"probe-decoded label on test failures is correct {dec_acc_fail:.3f} of the time "
        f"(K-way chance {1.0/a.K:.3f}) -- this caps any self-contained arm")

    methods = [("linear_w", True), ("manifold_mu", True), ("random", True),
               ("linear_w_decoded", False), ("manifold_mu_decoded", False)]
    for meth, is_oracle in methods:
        results[meth] = {"oracle": is_oracle}
        for al in alphas:
            def make(idxs):
                ds = []
                for i in idxs:
                    e = emitted[i]
                    c = Y[i] if is_oracle else decoded[i]     # oracle vs self-contained
                    if meth.startswith("linear_w"):
                        d = (W[cls[c]] if c in cls else 0.0) - (W[cls[e]] if e in cls else 0.0)
                    elif meth.startswith("manifold_mu"):
                        d = (mu[c] if c in mu else 0.0) - (mu[e] if e in mu else 0.0)
                    else:
                        d = rng.randn(F.shape[2])
                    d = np.asarray(d, dtype=float)
                    if d.ndim == 0 or not np.any(d): d = np.zeros(F.shape[2])
                    ds.append(unit(d) * al * hnorm)
                return np.stack(ds).astype(np.float32)

            lgf = s.run_steered([seqs[i] for i in fail_te], make(fail_te), Lstar, batch=max(4, a.batch // 2))
            rep = acc_from_logits(lgf, PRES, Y, fail_te)
            lgc = s.run_steered([seqs[i] for i in corr_te], make(corr_te), Lstar, batch=max(4, a.batch // 2))
            keep = acc_from_logits(lgc, PRES, Y, corr_te)
            results[meth][al] = dict(repair=rep, damage_keep=keep)
            tag = "ORACLE " if is_oracle else "self-cont"
            log(f"{tag} {meth:20s} a={al:<4} failures-fixed={rep:.3f}  correct-kept={keep:.3f} "
                f"(base {base_fail:.3f} / {base_corr:.3f})")

    # ---- the deployable number: self-gated steering over ALL test trials ----
    # delta = mu[decoded] - mu[emitted] is exactly ZERO whenever the probe agrees with the output, so the
    # method fires only on disagreement -- the certificate IS the gate, for free. Damage therefore comes
    # only from trials the model got right and the probe misread. This is the number to compare against
    # the prompt-level re-injection result, since both are self-contained and both act on every trial.
    base_all = acc_from_logits(np.stack([LG[i] for i in te]), PRES, Y, te)
    results["gated_all"] = {}
    for al in alphas:
        ds = []
        for i in te:
            c, e = decoded[i], emitted[i]
            if c == e: d = np.zeros(F.shape[2])
            else: d = unit((mu[c] if c in mu else 0.0) - (mu[e] if e in mu else 0.0)) * al * hnorm
            ds.append(np.asarray(d, dtype=float))
        lg = s.run_steered([seqs[i] for i in te], np.stack(ds).astype(np.float32), Lstar,
                           batch=max(4, a.batch // 2))
        acc = acc_from_logits(lg, PRES, Y, te)
        fired = float(np.mean([decoded[i] != emitted[i] for i in te]))
        results["gated_all"][al] = dict(acc=acc, base=base_all, fired=fired)
        log(f"GATED (self-contained, all trials) a={al:<4} acc {base_all:.3f} -> {acc:.3f} "
            f"({acc-base_all:+.3f})  fired on {fired:.3f} of trials")

    # ---- unembed ceiling at the final block ----
    try:
        WU = s.model.get_output_embeddings().weight.detach().float().cpu().numpy()
        results["unembed_finallayer"] = {}
        for al in alphas[:2]:
            ds = np.stack([unit(WU[Y[i]] - WU[emitted[i]]) * al * hnorm for i in fail_te]).astype(np.float32)
            lgf = s.run_steered([seqs[i] for i in fail_te], ds, len(s.blocks) - 1, batch=max(4, a.batch // 2))
            rep = acc_from_logits(lgf, PRES, Y, fail_te)
            results["unembed_finallayer"][al] = dict(repair=rep)
            log(f"unembed@L    a={al:<4} failures-fixed={rep:.3f}  (plumbing check / ceiling)")
    except Exception as e:
        log("unembed arm skipped:", type(e).__name__, str(e)[:60])

    json.dump(dict(model=a.model, revision=a.revision, seed=a.seed, K=a.K, D=a.D, lstar=Lstar,
                   n=n, n_fail=int(len(fail_te)), results=results),
              open(f"{a.out}/steer.json", "w"), indent=2)
    log("DONE", a.out)


if __name__ == "__main__":
    raise SystemExit(main())
