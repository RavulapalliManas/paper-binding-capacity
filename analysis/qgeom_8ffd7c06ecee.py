"""Binding geometry at the QUERY site -- the only position where the retrieved binding exists.

Why not the declaration site. The obvious place to look is the entity token of each declaration. It is the
wrong place: the obligation token follows the entity token, and attention is causal, so that vector cannot
carry the binding. Measured directly, from the entity-token residual (per-slot mean removed): entity
identity decodes at 1.000 and the bound obligation at 0.017-0.054 against 0.040 chance, on four models. A
"binding subspace" read there is a subspace of entities.

At the query site, after "The task for <entity> is:", the retrieved obligation IS decodable -- that is the
probe the whole paper rests on. So the geometry belongs here.

Quantities, all in the layer-l residual space at the query's last position:

  decode_obl   probe accuracy for the obligation. The positive control: this must be high, or nothing else
               in this script means anything.
  W            the obligation-discriminative subspace: span of the class-mean differences (rank <= C-1).
  G            the readout-sensitive subspace: top-k directions of the gradients d(margin)/d(h).
  alignment    fraction of the class-mean spread lying inside G. Exact null for a random k-dim subspace
               is k/d, for any cloud.
  matched      decode the obligation from the projection onto G's top-k, and onto a RANDOM k. If the
               readout subspace carries no more obligation information than a random one of equal size,
               the binding really is invisible to the output.

Usage: python qgeom.py --model EleutherAI/pythia-6.9b --out results/qg_p69
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from probe_battery_v2 import Battery2, Cfg


def log(*a): print("[qgeom]", *a, flush=True)


def get_blocks(model):
    for path in ("model.layers", "gpt_neox.layers", "transformer.h", "model.decoder.layers"):
        o = model
        try:
            for p in path.split("."): o = getattr(o, p)
            return o
        except AttributeError: continue
    raise RuntimeError("no blocks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    # a checkpoint revision, so the accessibility triple can be traced over TRAINING and not only across
    # models.  Cfg already carries `revision`; steer.py and jacdim.py already pass it.  Without this flag the
    # developmental store-vs-access curve cannot be measured at all.
    ap.add_argument("--revision", default=None)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--cond", default="code")
    ap.add_argument("--n", type=int, default=1500)     # trials for the state geometry
    ap.add_argument("--n-grad", type=int, default=256) # trials for the readout directions
    ap.add_argument("--layer", type=int, default=-1)   # -1 => mid-depth
    # `pert` is added to the RAW hidden state, so g = d margin / d h_raw. Hz = (H-mu)/sd, therefore the
    # standardised-frame gradient is sd (*) g -- a COVECTOR transform. steer.py:203 does the inverse
    # correctly (`W = clf.coef_ / sc.scale_`). This script divided. "div" reproduces the old (wrong) numbers
    # for the A/B; "mul" is correct and is the default.
    ap.add_argument("--grad-frame", choices=("mul", "div"), default="mul")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--dequantize", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    b = Battery2(Cfg(model=a.model, out="/tmp/_unused", revision=a.revision, dequantize=a.dequantize,
                     trust_remote_code=a.trust_remote_code))
    # provenance: a checkpoint's weights must be hashed before it enters a trajectory plot.  Two public
    # suites ship duplicate weights across revisions (Pythia-2.8B; Olmo-3-RL-Zero-Code) and would corrupt one.
    _w = b.model.get_input_embeddings().weight.detach().float().cpu().numpy()
    embed_md5 = hashlib.md5(_w.tobytes()).hexdigest()[:16]
    code_md5 = hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
    log(f"revision={a.revision} embed_md5={embed_md5} code={code_md5}")
    tok, model, dev = b.tok, b.model, b.dev
    L = (b.nlayers // 2) if a.layer < 0 else a.layer
    blocks = get_blocks(model); hook_ix = max(0, L - 1)
    q_pre = tok(" The task for", add_special_tokens=False)["input_ids"]
    q_suf = tok(" is:", add_special_tokens=False)["input_ids"]
    def build(rng):
        # Resample the distractor block PER TRIAL. It used to be drawn once (random.Random(1234)) and reused
        # across all state and gradient trials, so any interaction between that one block and the binding
        # geometry was baked into the measurement rather than averaged out.
        distract = b._distract(a.cond, a.D, rng, model.config.vocab_size) if a.D else []
        ents = rng.sample(b.ent, a.K); obls = rng.sample(b.obl, a.K)
        ids = []
        for (e, _), (o, _) in zip(ents, obls):
            ids += tok(f"{e}:{o}.", add_special_tokens=False)["input_ids"]
        j = rng.randrange(a.K)
        return ids + distract + q_pre + [ents[j][1]] + q_suf, obls[j][1], [o[1] for o in obls], ents[j][1]

    # ---------- state: query-site hidden states ----------
    rng = random.Random(31)
    seqs, Y, PRES, QENT = [], [], [], []
    for _ in range(a.n):
        s, y, pres, qe = build(rng); seqs.append(s); Y.append(y); PRES.append(pres); QENT.append(qe)
    H = []
    with torch.no_grad():
        for i in range(0, len(seqs), a.batch):
            ch = seqs[i:i + a.batch]
            ml = max(len(s) for s in ch)
            ids = torch.full((len(ch), ml), tok.pad_token_id, dtype=torch.long)
            m = torch.zeros((len(ch), ml), dtype=torch.long)
            for r, s in enumerate(ch):
                ids[r, ml - len(s):] = torch.tensor(s); m[r, ml - len(s):] = 1
            hs = model(input_ids=ids.to(dev), attention_mask=m.to(dev),
                       output_hidden_states=True).hidden_states[L][:, -1, :]
            H.append(hs.float().cpu().numpy())
    H = np.concatenate(H); Y = np.array(Y); QENT = np.array(QENT)
    d = H.shape[1]

    # SPLIT FIRST. mu and sd are fit on TRAIN rows only; fitting them over train+test leaks test statistics
    # into every decode. Every sibling in this repo already does it this way (probe_battery_v2.py:136,
    # rigor_rerun.py:63). `sd` also defines the gradient frame below, so it must not see the test fold.
    idx = np.random.RandomState(0).permutation(len(H))
    tr, te = idx[:int(.7 * len(H))], idx[int(.7 * len(H)):]
    mu = H[tr].mean(0, keepdims=True)
    sd = H[tr].std(0, keepdims=True); sd = np.where(sd > 1e-6, sd, 1.0)
    Hz = (H - mu) / sd
    Hz = Hz - Hz[tr].mean(0, keepdims=True)
    chance = 1.0 / len(b.obl)

    # THE PROBE'S CAPACITY MUST NOT DEPEND ON THE SUBSPACE'S DIMENSION.
    # With a fixed C, a 2048-dim probe on ~1050 train rows is underdetermined while a 21-dim probe is not, so
    # the low-dimensional arms get a free regularisation advantage. Measured consequence: on DeepSeek the
    # class-mean subspace decoded at 0.349 against 0.253 for the FULL residual -- i.e. the "nested levels"
    # inverted, which is impossible in principle and was an artefact of the estimator. Tune C on the TRAIN
    # fold, per subspace, by cross-validation. Nothing here ever sees `te`.
    C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    _chosen_C = {}

    def acc(P, y=Y, tag=None):
        best_c, best_v = None, -1.0
        for C in C_GRID:
            v = cross_val_score(LogisticRegression(max_iter=3000, C=C), P[tr], y[tr],
                                cv=3, n_jobs=1).mean()
            if v > best_v: best_c, best_v = C, v
        clf = LogisticRegression(max_iter=3000, C=best_c).fit(P[tr], y[tr])
        if tag: _chosen_C[tag] = best_c
        return float(clf.score(P[te], y[te]))

    decode_obl = acc(Hz, tag='full')
    decode_ent = acc(Hz, QENT)
    log(f"POSITIVE CONTROL at the query site: decode obligation = {decode_obl:.3f} (chance {chance:.3f})")
    log(f"                                    decode queried entity = {decode_ent:.3f}")
    if decode_obl < 3 * chance:
        log("!! obligation is not decodable here; the rest of this script is meaningless. stopping.")
        json.dump(dict(model=a.model, revision=a.revision, layer=L, grad_frame=a.grad_frame, decode_obl=decode_obl, chance=chance,
                       embed_md5=embed_md5, code_md5=code_md5, aborted=True),
                  open(f"{a.out}/qgeom.json", "w"), indent=2)
        return

    # obligation-discriminative subspace: class means
    cls = np.unique(Y)
    # TRAIN ROWS ONLY. Vw is a LABEL-DERIVED projection; computing it from all rows leaks the test fold's
    # labels into the subspace the probe is then scored in. Measured consequence: decode_classmean_k came out
    # ABOVE decode_obl_full on DeepSeek (0.349 vs 0.247), which is impossible for a projection of the same
    # space -- the projection was doing supervised dimensionality reduction with test labels. Vg is safe: its
    # gradient cloud is an independent draw. This one was not.
    M = np.stack([Hz[tr][Y[tr] == c].mean(0) for c in cls])
    Mc = M - M.mean(0, keepdims=True)
    _, Sw, Vw = np.linalg.svd(Mc, full_matrices=False)
    pr_state = float((Sw ** 2).sum() ** 2 / ((Sw ** 2) ** 2).sum())

    # ---------- readout: gradient of the margin wrt h at the query position ----------
    rng2 = random.Random(99)
    G = []
    for t in range(a.n_grad):
        seq, y, pres, _ = build(rng2)
        tt = torch.tensor([seq], device=dev)
        pert = torch.zeros(d, device=dev, dtype=torch.float32, requires_grad=True)

        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            mk = torch.zeros(h.shape[:2], device=h.device, dtype=torch.float32); mk[0, -1] = 1.0
            h = h + (mk.unsqueeze(-1) * pert).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        hd = blocks[hook_ix].register_forward_hook(fn)
        try:
            o = model(input_ids=tt, use_cache=False)
            row = o.logits[0, -1, :].float()
            foils = [row[p] for p in pres if p != y]
            r = row[y] - (torch.stack(foils).max() if foils else row.new_zeros(()))
            g, = torch.autograd.grad(r, pert)
        finally:
            hd.remove()
        G.append(g.float().cpu().numpy())
        if (t + 1) % 64 == 0: log(f"grad {t+1}/{a.n_grad}")
    G = np.stack(G)
    G = (G * sd) if a.grad_frame == "mul" else (G / sd)   # into the standardised frame; see --grad-frame
    Gc = G - G.mean(0, keepdims=True)
    _, Sg, Vg = np.linalg.svd(Gc, full_matrices=False)
    pr_readout = float((Sg ** 2).sum() ** 2 / ((Sg ** 2) ** 2).sum())
    k = int(min(max(round(pr_readout), 2), Vg.shape[0], len(cls) - 1 + 24))
    np.savez(f"{a.out}/vg.npz", Vg=Vg[:k], Vw=Vw[:k], sd=sd, k=k, grad_frame=a.grad_frame)

    # alignment of the obligation-discriminative subspace with the readout subspace
    tot = (Mc ** 2).sum()
    inside = ((Mc @ Vg[:k].T) ** 2).sum()
    alignment = float(inside / tot); null = k / d
    rr = np.random.RandomState(2)
    rand_al = []
    for _ in range(3):
        Q, _ = np.linalg.qr(rr.randn(d, k)); rand_al.append(float(((Mc @ Q) ** 2).sum() / tot))
    rand_alignment = float(np.mean(rand_al))

    # matched decodability: readout-k vs random-k
    a_read = acc(Hz @ Vg[:k].T, tag='readout_k')
    # 3 draws gave a point with no dispersion, so "readout beats random" had no error bar. DeepSeek's
    # readout_k ~ random_k cannot be asserted without one.
    N_NULL = 50
    ra = []
    for _ in range(N_NULL):
        Q, _ = np.linalg.qr(rr.randn(d, k)); ra.append(acc(Hz @ Q))
    a_rand = float(np.mean(ra)); a_rand_sd = float(np.std(ra, ddof=1))
    a_rand_lo, a_rand_hi = [float(v) for v in np.percentile(ra, [2.5, 97.5])]

    # --- defect B: the FAIR baseline.
    # Vg is built from GOLD-label margin gradients, so it approximates the obligation-unembedding subspace.
    # Beating a LABEL-BLIND random subspace at decoding those same labels is near-tautological in sign.
    # The honest comparator is the class-mean subspace Vw[:k]: same dimension, equally label-informed.
    #   readout << classmean  =>  the readout genuinely fails to look where the binding is.
    #   readout ~= classmean  =>  it sees what any label-informed subspace of that size would.
    a_classmean = acc(Hz @ Vw[:k].T, tag='classmean_k')

    res = dict(model=a.model, revision=a.revision, embed_md5=embed_md5, code_md5=code_md5,
               argv=" ".join(sys.argv[1:]), aborted=False,
               layer=L, hidden=int(d), K=a.K, D=a.D, n=a.n, k=k,
               chance=chance, decode_obl_full=decode_obl, decode_ent_full=decode_ent,
               pr_state_classmeans=pr_state, pr_readout=pr_readout,
               grad_frame=a.grad_frame,
               alignment=alignment, alignment_random=rand_alignment, alignment_null=float(null),
               decode_readout_k=a_read, decode_random_k=a_rand,
               decode_random_k_sd=a_rand_sd, decode_random_k_ci=[a_rand_lo, a_rand_hi], n_null_draws=N_NULL,
               decode_classmean_k=a_classmean, probe_C=_chosen_C, C_grid=list(C_GRID),
               readout_beats_random=bool(a_read > a_rand_hi),
               readout_vs_classmean=float(a_read - a_classmean))
    json.dump(res, open(f"{a.out}/qgeom.json", "w"), indent=2)
    log(f"frame={a.grad_frame}  class-mean PR = {pr_state:.1f}  readout PR = {pr_readout:.1f}  k = {k}")
    log(f"decode: full {decode_obl:.3f} | readout_k {a_read:.3f} | classmean_k {a_classmean:.3f} "
        f"| random_k {a_rand:.3f} +- {a_rand_sd:.3f} (95% [{a_rand_lo:.3f},{a_rand_hi:.3f}], {N_NULL} draws)")
    log(f"  readout beats the random null: {a_read > a_rand_hi}   "
        f"readout - classmean = {a_read - a_classmean:+.3f}  (the fair, label-matched contrast)")
    log(f"alignment of obligation subspace with readout = {alignment:.3f}  "
        f"(random k-dim {rand_alignment:.3f}, null k/d {null:.4f})")
    log(f"decode obligation | readout-k = {a_read:.3f}   | random-k = {a_rand:.3f}   "
        f"| full = {decode_obl:.3f}   chance {chance:.3f}")
    log("DONE " + a.out)


if __name__ == "__main__":
    raise SystemExit(main())
