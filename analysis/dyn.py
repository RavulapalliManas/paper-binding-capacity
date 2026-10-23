"""Working memory as a dynamical system: the retrievability trajectory and what follows from it.

For a trial with K bindings and a FIXED distractor stream of length D, define for each binding i
and each distractor position t:

    r_i(t) = the model's logit margin for obligation o_i, queried about entity e_i,
             with the prompt truncated at distractor position t.

The distractor stream is held fixed across trials (only the binding set varies), so the drive is a
deterministic exogenous sequence and the conditional dynamics is autonomous. This is what makes the
finite-time sensitivity analysis below legitimate; a varying drive would make it a forced-system
problem, which is a much harder object.

Computes, from one shared set of trials:
  (A) traj      r_i(t), p_i(t) full-vocab and restricted-to-present-K            -> decay law
  (B) conserve  sum_i p_i(t) as K grows                                          -> is capacity a shared resource?
  (C) lambda    (1/D) log || d r_i(D) / d eps ||, eps at e_i's declaration       -> marginal stability
  (D) geom      d_bind (participation ratio) + off-diagonal cos^2 interference
  (E) post      probe posterior vs model posterior on the FAILURE set            -> is the gap real, or noisy retrieval?
  (F) cap       continuous capacity: recall & forced-choice margin vs K          -> Schaeffer defense

(E) is the existential one. Two worlds:
    use failure   : probe confident + separated on correct; model confidently wrong. Gap real.
    noisy retrieval: probe posterior ~= model posterior, slightly reweighted. Certificate is
                     substantially reading the model's own uncertainty.

Usage:
  python dyn.py --model EleutherAI/pythia-1.4b --out results/dyn_p14 [--revision step143000]
"""
from __future__ import annotations
import argparse, json, os, random, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probe_battery_v2 import Battery2, Cfg, ENTITIES, OBLIGATIONS, PROSE, CODE


def log(*a): print("[dyn]", *a, flush=True)


class Dyn:
    def __init__(self, model, revision=None, dequantize=False, trust=False, dtype="bfloat16"):
        self.b = Battery2(Cfg(model=model, out="/tmp/_unused", revision=revision,
                              dequantize=dequantize, trust_remote_code=trust, dtype=dtype))
        self.tok, self.model, self.dev = self.b.tok, self.b.model, self.b.dev
        self.ent, self.obl = self.b.ent, self.b.obl
        self.obl_ids = self.b.obl_ids
        self.nL = self.b.nlayers
        # query scaffold, built compositionally so every entity yields the SAME query length
        self.q_pre = self.tok(" The task for", add_special_tokens=False)["input_ids"]
        self.q_suf = self.tok(" is:", add_special_tokens=False)["input_ids"]

    # ---------- trial construction ----------
    def fixed_distractor(self, cond, D, seed=1234):
        """One deterministic distractor stream, reused across every trial."""
        rng = random.Random(seed)
        return self.b._distract(cond, D, rng, self.model.config.vocab_size) if D else []

    def build_decl(self, pairs):
        """Declaration token ids + the token index of each entity's declaration position."""
        ids, ent_pos = [], []
        for (e_str, _e_id), (o_str, _o_id) in pairs:
            part = self.tok(f"{e_str}:{o_str}.", add_special_tokens=False)["input_ids"]
            ent_pos.append(len(ids))          # entity token is first in the part
            ids += part
        return ids, ent_pos

    def query_ids(self, ent_tok_id):
        return self.q_pre + [ent_tok_id] + self.q_suf

    # ---------- batched forward ----------
    @torch.no_grad()
    def _logits_hidden(self, seqs, want_hidden=False):
        maxlen = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        m = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):                       # LEFT pad -> last pos is query end
            ids[i, maxlen - len(s):] = torch.tensor(s); m[i, maxlen - len(s):] = 1
        out = self.model(input_ids=ids.to(self.dev), attention_mask=m.to(self.dev),
                         output_hidden_states=want_hidden)
        lg = out.logits[:, -1, :].float().cpu().numpy()
        hs = None
        if want_hidden:
            hs = np.stack([h[:, -1, :].float().cpu().numpy() for h in out.hidden_states], axis=1)
        return lg, hs

    # ---------- (A)(B) trajectory + conservation ----------
    def trajectory(self, K, cond, D, tgrid, n_trials, seed=0, batch=64):
        rng = random.Random(9000 + seed)
        distract = self.fixed_distractor(cond, D)
        R = np.zeros((n_trials, K, len(tgrid)), dtype=np.float32)   # margin correct - best foil(present)
        Pf = np.zeros_like(R)                                        # full-vocab prob of correct obligation
        Pr = np.zeros_like(R)                                        # prob restricted to the K present obligations
        for tr in range(n_trials):
            ents = rng.sample(self.ent, K); obls = rng.sample(self.obl, K)
            pairs = list(zip(ents, obls))
            decl, _ = self.build_decl(pairs)
            present = [o[1] for o in obls]
            seqs, meta = [], []
            for ti, t in enumerate(tgrid):
                pre = decl + distract[:t]
                for i in range(K):
                    seqs.append(pre + self.query_ids(ents[i][1])); meta.append((i, ti))
            for b in range(0, len(seqs), batch):
                lg, _ = self._logits_hidden(seqs[b:b + batch])
                for r, (i, ti) in enumerate(meta[b:b + batch]):
                    row = lg[r]
                    cor = row[present[i]]
                    foils = [row[p] for j, p in enumerate(present) if j != i]
                    R[tr, i, ti] = cor - (max(foils) if foils else 0.0)
                    Pf[tr, i, ti] = float(np.exp(row[present[i]] - _lse(row)))
                    sub = row[present]
                    Pr[tr, i, ti] = float(np.exp(sub[i] - _lse(sub)))
            if (tr + 1) % 20 == 0: log(f"traj K={K} {cond} D={D}: {tr+1}/{n_trials}")
        return R, Pf, Pr

    # ---------- (C) finite-time sensitivity ----------
    def _grad_norm(self, seq, ent_pos_i, present, i, emb):
        ids = torch.tensor([seq], device=self.dev)
        E = emb(ids).detach()
        # additive perturbation localized on e_i's declaration token, via a one-hot mask so autograd
        # sees a clean leaf (index-assignment on a clone is fragile across transformers versions)
        pert = torch.zeros(E.shape[-1], device=self.dev, dtype=torch.float32, requires_grad=True)
        mask = torch.zeros(E.shape[:2], device=self.dev, dtype=torch.float32)
        mask[0, ent_pos_i] = 1.0
        E2 = E + (mask.unsqueeze(-1) * pert).to(E.dtype)
        o = self.model(inputs_embeds=E2, use_cache=False)
        row = o.logits[0, -1, :].float()
        foils = [row[p] for j, p in enumerate(present) if j != i]
        r = row[present[i]] - (torch.stack(foils).max() if foils else row.new_zeros(()))
        g, = torch.autograd.grad(r, pert)
        return float(g.float().norm().clamp_min(1e-12).item())

    def sensitivity(self, K, cond, D, tgrid, n_trials=24, seed=0):
        """Sensitivity profile S_i(t) = || d r_i(t) / d eps ||, eps at e_i's declaration token.

        The contraction rate is the log-slope of S_i(t). We return the full profile rather than a
        single number anchored at t=0, because t=0 turns out NOT to be the no-interference condition:
        with the query immediately following the declaration list the model is still in "continue the
        list" mode, so S_i(0) is depressed for reasons that have nothing to do with memory. Anchoring
        a rate there measures that framing artifact. The rate is fit downstream on the post-peak
        branch (see analyze_dyn.py).

            slope < 0   the declaration's influence on the output CONTRACTS   -> forgotten
            slope ~ 0   influence is preserved: marginally stable             -> the binding survives

        This is a finite-time input-output sensitivity of a DRIVEN system. It reduces to a Lyapunov
        exponent only in the autonomous limit and we do not call it one. The distractor stream is held
        fixed across trials so the drive is deterministic; per Stark's forced-system results the object
        characterized is conditional on that drive, not an intrinsic autonomous spectrum. Following
        Sagodi et al. (2408.00109), a merely SLOW mode and a genuinely MARGINAL one are different
        claims, so we report the spread across items and trials, not a thresholded count.
        """
        rng = random.Random(4100 + seed)
        distract = self.fixed_distractor(cond, D)
        emb = self.model.get_input_embeddings()
        out = np.zeros((n_trials, K, len(tgrid)), dtype=np.float32)
        for tr in range(n_trials):
            ents = rng.sample(self.ent, K); obls = rng.sample(self.obl, K)
            decl, ent_pos = self.build_decl(list(zip(ents, obls)))
            present = [o[1] for o in obls]
            for i in range(K):
                q = self.query_ids(ents[i][1])
                for ti, t in enumerate(tgrid):
                    out[tr, i, ti] = self._grad_norm(decl + distract[:t] + q, ent_pos[i], present, i, emb)
            if (tr + 1) % 8 == 0: log(f"sens K={K} D={D}: {tr+1}/{n_trials}")
        return out   # [trials, K, |tgrid|]  -- raw gradient norms

    # ---------- (D) declaration-site geometry ----------
    @torch.no_grad()
    def geometry(self, K, n_trials=160, seed=0, layer=None):
        """Effective dimension and interference of the BINDING subspace.

        Confound handled: declaration-site hidden states are dominated by a positional component --
        entity i always sits at slot i, so the raw vectors are nearly collinear (participation ratio
        ~1, cos^2 ~1) and any "geometry" read off them is really reading position. We therefore
        estimate the positional component as the per-slot mean ACROSS trials (position fixed, binding
        varies) and measure geometry on the residual, which is what actually carries the binding.

        Both numbers are returned so the confound is visible rather than assumed away.
        """
        rng = random.Random(770 + seed)
        L = layer if layer is not None else self.nL // 2
        raws = []
        for tr in range(n_trials):
            ents = rng.sample(self.ent, K); obls = rng.sample(self.obl, K)
            decl, ent_pos = self.build_decl(list(zip(ents, obls)))
            ids = torch.tensor([decl], device=self.dev)
            hs = self.model(input_ids=ids, output_hidden_states=True).hidden_states[L][0].float().cpu().numpy()
            raws.append(hs[ent_pos])                          # [K, d]
        Rw = np.stack(raws)                                   # [T, K, d]

        def pr(M, standardize=False):                         # participation ratio of rows of M
            M = M - M.mean(0, keepdims=True)
            if standardize:
                # Transformer residual streams carry a few "massive activation" dimensions whose
                # variance is orders of magnitude above the rest (attention sinks). A covariance PR
                # is dominated by them, so it reports ~1 for families that have them (Qwen, Llama)
                # regardless of the binding code. Standardizing per-dimension makes PR a property of
                # the correlation structure, which is what "effective dimension of the code" means.
                sd = M.std(0, keepdims=True)
                M = M / np.where(sd > 1e-8, sd, 1.0)
            s = np.linalg.svd(M, compute_uv=False); ev = s ** 2
            return float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.sum() > 0 else float("nan")

        def top1_var(M):
            M = M - M.mean(0, keepdims=True)
            v = M.var(0); return float(v.max() / (v.sum() + 1e-12))

        def mean_cos2(V):                                     # mean off-diagonal cos^2 among K rows
            Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            C = (Vn @ Vn.T) ** 2
            return float((C.sum() - np.trace(C)) / (K * (K - 1))) if K > 1 else float("nan")

        # (i) position-contaminated: within-trial centering only  -- what a naive reading gives
        d_raw = float(np.nanmean([pr(Rw[t]) for t in range(len(Rw))]))
        i_raw = float(np.nanmean([mean_cos2(Rw[t] - Rw[t].mean(0, keepdims=True)) for t in range(len(Rw))]))

        # (ii) position-removed: subtract the per-slot mean across trials, then pool
        slot_mean = Rw.mean(0, keepdims=True)                 # [1, K, d]  <- the positional component
        Res = Rw - slot_mean                                  # [T, K, d]  <- binding content
        P = Res.reshape(-1, Res.shape[-1])
        d_cov = pr(P)                                         # covariance PR: massive-activation dominated
        d_bind = pr(P, standardize=True)                      # correlation PR: the effective dimension
        i_bind = float(np.nanmean([mean_cos2(Res[t]) for t in range(len(Res))]))
        # how much of the raw variance was positional?  and how much sits in one outlier dimension?
        pos_frac = float((slot_mean[0].var(0).sum()) / (Rw.reshape(-1, Rw.shape[-1]).var(0).sum() + 1e-12))
        return dict(d_bind=d_bind, d_bind_cov=d_cov, interference=i_bind,
                    d_bind_raw=d_raw, interference_raw=i_raw,
                    positional_variance_fraction=pos_frac, top1_dim_variance_fraction=top1_var(P),
                    layer=L, K=K, n_trials=n_trials)

    # ---------- (E) probe posterior vs model posterior on failures ----------
    def posterior(self, K, cond, D, n_trials=900, seed=0, batch=32):
        """The existential check. Nested split: fit on tr, pick layer on va, report on te."""
        rng = random.Random(31337 + seed)
        distract = self.fixed_distractor(cond, D)
        seqs, Y, PRESENT = [], [], []
        for _ in range(n_trials):
            ents = rng.sample(self.ent, K); obls = rng.sample(self.obl, K)
            decl, _ = self.build_decl(list(zip(ents, obls)))
            j = rng.randrange(K)
            seqs.append(decl + distract[:D] + self.query_ids(ents[j][1]))
            Y.append(obls[j][1]); PRESENT.append([o[1] for o in obls])
        F, LG = [], []
        for b in range(0, len(seqs), batch):
            lg, hs = self._logits_hidden(seqs[b:b + batch], want_hidden=True)
            F.append(hs); LG.append(lg)
        F = np.concatenate(F); LG = np.concatenate(LG); Y = np.array(Y); PRESENT = np.array(PRESENT)

        n = len(Y); idx = np.arange(n); np.random.RandomState(seed).shuffle(idx)
        n3 = n // 3; tr, va, te = idx[:n3], idx[n3:2 * n3], idx[2 * n3:]
        best, bestL = -1, 1
        for L in range(1, F.shape[1]):
            sc = StandardScaler().fit(F[tr, L, :])
            try:
                m = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(F[tr, L, :]), Y[tr])
                s = m.score(sc.transform(F[va, L, :]), Y[va])
            except Exception: s = -1
            if s > best: best, bestL = s, L
        trva = np.concatenate([tr, va])
        sc = StandardScaler().fit(F[trva, bestL, :])
        clf = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(F[trva, bestL, :]), Y[trva])
        classes = list(clf.classes_); cls_ix = {c: k for k, c in enumerate(classes)}
        PP = clf.predict_proba(sc.transform(F[te, bestL, :]))

        rows = []
        for r, ti in enumerate(te):
            present = PRESENT[ti]; y = Y[ti]
            mrow = LG[ti][present]
            mp = np.exp(mrow - _lse(mrow))                     # model posterior over present K
            emitted = present[int(mp.argmax())]
            pp = np.array([PP[r][cls_ix[p]] if p in cls_ix else 0.0 for p in present])
            pp = pp / (pp.sum() + 1e-12)                        # probe posterior over present K
            ci = int(np.where(present == y)[0][0]); ei = int(mp.argmax())
            rows.append(dict(
                correct_ix=ci, emitted_ix=ei, model_wrong=bool(emitted != y),
                probe_top1=bool(int(pp.argmax()) == ci),
                probe_p_correct=float(pp[ci]), probe_p_emitted=float(pp[ei]),
                model_p_correct=float(mp[ci]), model_p_emitted=float(mp[ei]),
                probe_margin=float(pp[ci] - pp[ei]), model_margin=float(mp[ci] - mp[ei]),
                # is the probe posterior just the model's, reweighted?
                spearman=float(_spearman(pp, mp)), tv=float(0.5 * np.abs(pp - mp).sum()),
            ))
        return dict(layer=bestL, va_acc=float(best), K=K, D=D, cond=cond, rows=rows)

    # ---------- (F) continuous capacity ----------
    @torch.no_grad()
    def capacity(self, cond, D, Ks=(1, 2, 3, 4, 5, 6, 8, 10), n_trials=200, seed=0, batch=32):
        rng = random.Random(555 + seed)
        distract = self.fixed_distractor(cond, D)
        out = []
        for K in Ks:
            if K > min(len(self.ent), len(self.obl)): break
            seqs, Y, PRESENT = [], [], []
            for _ in range(n_trials):
                ents = rng.sample(self.ent, K); obls = rng.sample(self.obl, K)
                decl, _ = self.build_decl(list(zip(ents, obls)))
                j = rng.randrange(K)
                seqs.append(decl + distract[:D] + self.query_ids(ents[j][1]))
                Y.append(obls[j][1]); PRESENT.append([o[1] for o in obls])
            rec, marg = [], []
            for b in range(0, len(seqs), batch):
                lg, _ = self._logits_hidden(seqs[b:b + batch])
                for r in range(lg.shape[0]):
                    gi = b + r; present = PRESENT[gi]; row = lg[r][present]
                    y = Y[gi]; ci = list(present).index(y)
                    rec.append(int(row.argmax() == ci))
                    foils = np.delete(row, ci)
                    marg.append(float(row[ci] - foils.max()) if foils.size else float("nan"))
            out.append(dict(K=K, recall=float(np.mean(rec)), margin=float(np.nanmean(marg)),
                            margin_se=float(np.nanstd(marg) / np.sqrt(len(marg)))))
            log(f"cap K={K}: recall={out[-1]['recall']:.3f} margin={out[-1]['margin']:+.3f}")
        return out


def _lse(x):
    m = np.max(x); return m + np.log(np.exp(x - m).sum())

def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--cond", default="code"); ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-traj", type=int, default=100); ap.add_argument("--n-post", type=int, default=900)
    ap.add_argument("--n-lambda", type=int, default=24); ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--dequantize", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--skip", default="")           # comma list: traj,lambda,geom,post,cap
    a = ap.parse_args()
    skip = set(x for x in a.skip.split(",") if x)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    d = Dyn(a.model, revision=a.revision, dequantize=a.dequantize, trust=a.trust_remote_code)
    manifest = dict(model=a.model, revision=a.revision, K=a.K, D=a.D, cond=a.cond,
                    layers=d.nL, n_ent=len(d.ent), n_obl=len(d.obl), ts=time.time())
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=2)

    # (E) existential check first -- it can invalidate everything downstream
    if "post" not in skip:
        p = d.posterior(a.K, a.cond, a.D, n_trials=a.n_post, batch=max(8, a.batch // 2))
        json.dump(p, open(f"{a.out}/posterior.json", "w"), indent=2)
        fr = [r for r in p["rows"] if r["model_wrong"]]
        if fr:
            log(f"POSTERIOR  layer={p['layer']} va={p['va_acc']:.3f} | failures={len(fr)}/{len(p['rows'])}")
            log(f"  probe top1 on failures     = {np.mean([r['probe_top1'] for r in fr]):.3f}  (chance {1/a.K:.3f})")
            log(f"  probe p(correct)           = {np.mean([r['probe_p_correct'] for r in fr]):.3f}")
            log(f"  probe p(emitted)           = {np.mean([r['probe_p_emitted'] for r in fr]):.3f}")
            log(f"  model p(correct)           = {np.mean([r['model_p_correct'] for r in fr]):.3f}")
            log(f"  model p(emitted)           = {np.mean([r['model_p_emitted'] for r in fr]):.3f}")
            log(f"  spearman(probe,model)      = {np.mean([r['spearman'] for r in fr]):+.3f}   <- ~1 => noisy retrieval")
            log(f"  total-variation(probe,model)= {np.mean([r['tv'] for r in fr]):.3f}          <- ~0 => noisy retrieval")

    if "cap" not in skip:
        cap = d.capacity(a.cond, a.D, n_trials=200, batch=a.batch)
        json.dump(cap, open(f"{a.out}/capacity.json", "w"), indent=2)

    if "geom" not in skip:
        g = d.geometry(a.K)
        json.dump(g, open(f"{a.out}/geom.json", "w"), indent=2)
        log(f"GEOM d_bind={g['d_bind']:.2f} (cov-PR {g['d_bind_cov']:.2f}) interference={g['interference']:.4f} @L{g['layer']}")
        log(f"     confounds: positional var frac={g['positional_variance_fraction']:.3f} "
            f"| top-1 dim var frac={g['top1_dim_variance_fraction']:.3f} "
            f"| naive read would be d={g['d_bind_raw']:.2f} cos2={g['interference_raw']:.3f}")

    if "traj" not in skip:
        tgrid = [t for t in (0, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256) if t <= a.D]
        R, Pf, Pr = d.trajectory(a.K, a.cond, a.D, tgrid, a.n_traj, batch=a.batch)
        np.savez_compressed(f"{a.out}/traj.npz", R=R, Pf=Pf, Pr=Pr, tgrid=np.array(tgrid))
        log(f"TRAJ margin r(0)={R[:,:,0].mean():+.3f} -> r(D)={R[:,:,-1].mean():+.3f}")

        # Conservation test. C(t) = sum_i P(o_i | query e_i), each term restricted to the K present
        # obligations. Every term comes from a SEPARATE forward pass, so nothing forces C to any value.
        # chance = 1.0 for every K (each term = 1/K).  perfect retrieval = K.
        # C conserved as K grows  => bindings trade against a fixed resource; "capacity" is the right noun.
        # C growing with K        => per-item decay, no shared resource; "capacity" is a misnomer.
        cons = {}
        for K2 in (2, 3, 4, 6, 8):
            if K2 > min(len(d.ent), len(d.obl)): continue
            R2, Pf2, Pr2 = d.trajectory(K2, a.cond, a.D, tgrid, max(30, a.n_traj // 2), batch=a.batch)
            C = Pr2.sum(1)                                        # [trials, |tgrid|] items' worth
            cons[K2] = dict(
                C_items=[float(C[:, ti].mean()) for ti in range(len(tgrid))],
                C_items_se=[float(C[:, ti].std() / np.sqrt(C.shape[0])) for ti in range(len(tgrid))],
                sum_pfull=[float(Pf2[:, :, ti].sum(1).mean()) for ti in range(len(tgrid))],
                margin=[float(R2[:, :, ti].mean()) for ti in range(len(tgrid))])
            log(f"CONSERVE K={K2}: C(0)={cons[K2]['C_items'][0]:.3f}+-{cons[K2]['C_items_se'][0]:.3f} "
                f"-> C(D)={cons[K2]['C_items'][-1]:.3f}   (chance=1.000, perfect={K2})")
        json.dump(dict(tgrid=tgrid, cons=cons, note="C_items chance=1.0 for all K"),
                  open(f"{a.out}/conservation.json", "w"), indent=2)

    if "lambda" not in skip:
        sg = [t for t in (0, 16, 32, 64, 96, 128, 192, 256) if t <= a.D]
        S = d.sensitivity(a.K, a.cond, a.D, sg, n_trials=a.n_lambda)
        np.savez_compressed(f"{a.out}/sens.npz", S=S, tgrid=np.array(sg))
        m = S.mean((0, 1))
        pk = int(np.argmax(m))
        if pk < len(sg) - 1:                       # contraction rate on the post-peak branch
            slope = np.polyfit(np.array(sg[pk:], float), np.log(m[pk:] + 1e-12), 1)[0]
        else:
            slope = float("nan")
        log(f"SENS profile={np.round(m,3).tolist()} peak@t={sg[pk]} "
            f"post-peak contraction rate={slope:+.5f}/token")

    log(f"DONE {a.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
