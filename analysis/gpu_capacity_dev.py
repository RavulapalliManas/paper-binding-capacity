"""Capacity at the output and in the hidden state across Pythia pretraining, on the capacity protocol.

WHAT THIS RUNS.  For one model and a list of checkpoints, every prompt of the published capacity grid
(neuron_capacity.py: 15 loads x D in {0, 256} x 6 seeds x 96 trials, padded on the left to 768
tokens, bfloat16).  The prompts come from the same generator and the same seeds, so the last
checkpoint replays the published prompts and must reproduce the published recall.  Per trial it saves:
  - the logits over the value pool at the last token (the published readout is their argmax);
  - the residual stream at the last token after four fixed layers, L/4, L/2, 3L/4 and L
    (hidden_states indices, as in the earlier probe runs: index L is after the final layer norm);
  - the residual stream after layer L/2 at the last token before the question.  A probe there cannot
    know which entity will be queried, so it measures what a probe gets from the listed values alone.
Probes are fit afterwards on cpu-angad (p3_probe_capacity.py).  This script only runs the model.

CONTROLS (--controls, run at the checkpoints given in --control-revisions):
  pre     the D=256 trials with the distractor block moved in front of the bindings: same bindings,
          same lines, same length; only the separation of the bindings from the question changes.
  lenfix  the D=0 trials with code-like lines put in front of the bindings until the prompt has at
          least as many tokens as the longest D=0 prompt at K=64: load varies at near-fixed length.

PEAK MEMORY.  GPU, per batch: hidden states of every layer, batch x 768 x d x (L+1) x 2 bytes
(Pythia-6.9B at batch 16: 16 x 768 x 4096 x 33 x 2 B = 3.3 GB).  Host, per checkpoint: trials x 5 x d
x 2 bytes (Pythia-6.9B: 17,280 x 5 x 4096 x 2 B = 0.71 GB), written before the next checkpoint loads.

ONE WRITER PER FILE.  One process per model.  Each checkpoint writes <out>/<label>/<revision>.npz and
.json; a checkpoint whose .json exists is skipped, so a restarted run resumes.

Usage:
  python gpu_capacity_dev.py --selftest
  python gpu_capacity_dev.py --model EleutherAI/pythia-1.4b --label pythia-1.4b \
      --revisions step1000,step2000,...,step143000 --controls pre,lenfix \
      --control-revisions step143000 --out /workspace/out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from neuron_capacity import (K_GRID, D_GRID, ENTITY_POOL, OBLIGATION_POOL,  # noqa: E402
                             FILLER, single_token, make_prompt)

SEQ, N_TRIALS, SEEDS = 768, 96, 6          # the published grid (results/capacity_x/*_w0.json)
COND_CODE = {"D0": 0, "D256": 1, "pre": 2, "lenfix": 3}


def cell_rng(K, D, s):
    """The published seed formula (neuron_capacity.main)."""
    return random.Random((7000 + s) * 100003 + K * 31 + D)


def code_line(rng):
    return (f"{rng.choice(FILLER)}_{rng.randrange(100)} = f({rng.choice(FILLER)}, "
            f"{rng.randrange(100)})")


def make_trial(rng, ents, obls, K, D, place="after"):
    """neuron_capacity.make_prompt with the same RNG calls in the same order, returning the parts."""
    e = rng.sample(ents, K)
    o = [rng.choice(obls) for _ in range(K)]
    qi = rng.randrange(K)
    decl = "\n".join(f"{a}: {b}" for a, b in zip(e, o))
    lines = [code_line(rng) for _ in range(max(1, D // 8))] if D > 0 else []
    if place == "after":
        body = decl + ("\n" + "\n".join(lines) if lines else "")
    else:
        body = ("\n".join(lines) + "\n" if lines else "") + decl
    return dict(body=body, prompt=f"{body}\nThe task for {e[qi]} is:", ents=e, vals=o, qi=qi)


def grid_trials(ents, obls, max_k, cond, n_trials=N_TRIALS, seeds=SEEDS):
    """All trials of one condition, in the published cell order (K, then seed)."""
    D = 0 if cond in ("D0", "lenfix") else 256
    place = "before" if cond == "pre" else "after"
    out = []
    for K in K_GRID:
        if K > max_k:
            continue
        for s in range(seeds):
            rng = cell_rng(K, D, s)
            for i in range(n_trials):
                t = make_trial(rng, ents, obls, K, D, place)
                t.update(K=K, seed=s, i=i, cond=cond)
                out.append(t)
    return out


def apply_lenfix(trials, tok, target):
    """Prepend code-like lines (their own RNG per trial) until each prompt has >= target tokens."""
    for t in trials:
        rng = random.Random(f"lenfix|{t['K']}|{t['seed']}|{t['i']}")
        pre = []
        body = t["body"]
        while len(tok(f"{body}\nThe task for x is:")["input_ids"]) < target:
            pre.append(code_line(rng))
            body = "\n".join(pre) + "\n" + t["body"]
        e = t["ents"][t["qi"]]
        t["body"], t["prompt"], t["n_prefix_lines"] = body, f"{body}\nThe task for {e} is:", len(pre)
    return trials


def selftest(tok=None):
    """make_trial must reproduce make_prompt exactly, for every published cell."""
    ents = ENTITY_POOL[:67]
    obls = OBLIGATION_POOL[:67]
    n = 0
    for K in K_GRID:
        for D in D_GRID:
            for s in range(SEEDS):
                r1, r2 = cell_rng(K, D, s), cell_rng(K, D, s)
                for _ in range(N_TRIALS):
                    p, g = make_prompt(r1, ents, obls, K, D)
                    t = make_trial(r2, ents, obls, K, D)
                    assert p == t["prompt"] and g == t["vals"][t["qi"]], (K, D, s)
                    n += 1
    # the moved block keeps bindings and lines: same multiset of lines
    r1, r2 = cell_rng(8, 256, 0), cell_rng(8, 256, 0)
    a, b = make_trial(r1, ents, obls, 8, 256), make_trial(r2, ents, obls, 8, 256, "before")
    assert sorted(a["body"].split("\n")) == sorted(b["body"].split("\n")) and a["vals"] == b["vals"]
    print(f"[selftest] make_trial == make_prompt on {n} trials; 'pre' keeps bindings and lines", flush=True)


def layer_set(n_layers):
    return sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers})


def embed_md5(model):
    w = model.get_input_embeddings().weight[:2048]
    return hashlib.md5(w.detach().float().cpu().numpy().tobytes()).hexdigest()


def encode(tok, trials, seq):
    """Left-padded ids to `seq`, and the index of the last token before the question."""
    # the published runs called tok(prompts, ...) with the default add_special_tokens=True, so OPT
    # (and any tokenizer that adds a BOS) saw it; keep that here
    full = tok([t["prompt"] for t in trials])["input_ids"]
    body = tok([t["body"] for t in trials])["input_ids"]
    n_long = sum(len(f) > seq for f in full)
    assert n_long == 0, f"{n_long} prompts exceed {seq} tokens; the question would be cut"
    pre = np.full(len(trials), -1, np.int64)
    for j, (f, b) in enumerate(zip(full, body)):
        if f[: len(b)] == b:
            pre[j] = seq - 1 - (len(f) - len(b))
    ids = np.full((len(trials), seq), tok.pad_token_id, np.int64)
    mask = np.zeros((len(trials), seq), np.int64)
    for j, f in enumerate(full):
        ids[j, seq - len(f):] = f
        mask[j, seq - len(f):] = 1
    ntok = np.array([len(f) for f in full], np.int32)
    return ids, mask, pre, ntok


def run_checkpoint(a, model, tok, obl_ids, trials, enc, layers, lmid, dev):
    import torch
    ids, mask, pre, ntok = enc
    n = len(trials)
    d = model.config.hidden_size
    LG = np.zeros((n, len(obl_ids)), np.float32)
    H = np.zeros((n, len(layers), d), np.float16)
    P = np.zeros((n, d), np.float16)
    oid = torch.tensor(obl_ids, device=dev)
    seq = ids.shape[1]
    # fixed: every batch padded to `seq`, as in the published runs.  dynamic: trials sorted by length
    # and each batch cut to its longest prompt (the arrays are left-padded, so the last T columns hold
    # every prompt of length <= T); in float32 the amount of padding does not change the result.
    order = np.argsort(ntok, kind="stable") if a.pad == "dynamic" else np.arange(n)
    with torch.no_grad():
        for i in range(0, n, a.batch):
            r = order[i:i + a.batch]
            T = int(ntok[r].max()) if a.pad == "dynamic" else seq
            b_ids = torch.from_numpy(ids[r][:, seq - T:]).to(dev)
            b_mask = torch.from_numpy(mask[r][:, seq - T:]).to(dev)
            kw = {}
            if model.config.model_type == "gpt2":
                # learned absolute positions: count from the first real token, not from the padding
                kw["position_ids"] = (b_mask.cumsum(-1) - 1).clamp(min=0)
            out = model(input_ids=b_ids, attention_mask=b_mask, output_hidden_states=True,
                        use_cache=False, **kw)
            LG[r] = out.logits[:, -1, :].float().index_select(1, oid).cpu().numpy()
            hs = out.hidden_states
            H[r] = torch.stack([hs[l][:, -1, :] for l in layers], 1).float().cpu().numpy()
            bp = torch.from_numpy(np.maximum(pre[r] - (seq - T), 0)).to(dev)
            P[r] = hs[lmid][torch.arange(len(bp), device=dev), bp, :].float().cpu().numpy()
            del out, hs
    return LG, H, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--label")
    ap.add_argument("--revisions", default="main")
    ap.add_argument("--controls", default="", help="comma list of: pre, lenfix, verbatim, shots2, taskcolon, xis, matched")
    ap.add_argument("--control-revisions", default="")
    ap.add_argument("--out", default="out")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cells", default="", help="validation only: comma list of K:D:seed to run")
    ap.add_argument("--pad", default="fixed", choices=["fixed", "dynamic"])
    ap.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (off: strict float32)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ents = [w for w, _ in single_token(tok, ENTITY_POOL)]
    obl = single_token(tok, OBLIGATION_POOL)
    obl_words, obl_ids = [w for w, _ in obl], [i for _, i in obl]
    max_k = min(len(ents), len(obl_words))
    w2i = {w: j for j, w in enumerate(obl_words)}
    e2i = {w: j for j, w in enumerate(ents)}

    conds = ["D0", "D256"]
    trials = {c: grid_trials(ents, obl_words, max_k, c) for c in conds}
    ctl = [c for c in a.controls.split(",") if c]
    if "pre" in ctl:
        trials["pre"] = grid_trials(ents, obl_words, max_k, "pre")
    if "lenfix" in ctl:
        k64 = [t for t in trials["D0"] if t["K"] == max(k for k in K_GRID if k <= max_k)]
        target = max(len(tok(t["prompt"])["input_ids"]) for t in k64)
        trials["lenfix"] = apply_lenfix(grid_trials(ents, obl_words, max_k, "lenfix"), tok, target)
    # the D0 lists with another question.  verbatim: the question repeats the list format ("X:").
    # shots2: two answered questions about other listed entities (fewer when K < 3) come first, as in
    # the 1- or 2-shot phone-book lookup of Jelassi et al. (2024); the examples come from their own RNG
    if "verbatim" in ctl:
        trials["verbatim"] = [dict(t, cond="verbatim", prompt=f"{t['body']}\n{t['ents'][t['qi']]}:")
                              for t in grid_trials(ents, obl_words, max_k, "D0")]
    if "shots2" in ctl:
        trials["shots2"] = []
        for t in grid_trials(ents, obl_words, max_k, "D0"):
            r = random.Random((9000 + t["seed"]) * 100003 + t["K"] * 31 + t["i"])
            ex = r.sample([j for j in range(t["K"]) if j != t["qi"]], min(2, t["K"] - 1))
            shots = "".join(f"\nThe task for {t['ents'][j]} is: {t['vals'][j]}" for j in ex)
            trials["shots2"].append(dict(t, cond="shots2", shots=ex,
                                         prompt=f"{t['body']}{shots}\nThe task for {t['ents'][t['qi']]} is:"))
    # three more wordings on the same D0 lists.  taskcolon: "The task for X:"; xis: "X is:";
    # matched: every list line in the question's own form, "The task for e is: v", then the published
    # question.  The two first separate the entity-colon adjacency from the frame words.
    if "taskcolon" in ctl:
        trials["taskcolon"] = [dict(t, cond="taskcolon", prompt=f"{t['body']}\nThe task for {t['ents'][t['qi']]}:")
                               for t in grid_trials(ents, obl_words, max_k, "D0")]
    if "xis" in ctl:
        trials["xis"] = [dict(t, cond="xis", prompt=f"{t['body']}\n{t['ents'][t['qi']]} is:")
                         for t in grid_trials(ents, obl_words, max_k, "D0")]
    if "matched" in ctl:
        trials["matched"] = []
        for t in grid_trials(ents, obl_words, max_k, "D0"):
            body = "\n".join(f"The task for {e} is: {v}" for e, v in zip(t["ents"], t["vals"]))
            trials["matched"].append(dict(t, cond="matched", body=body,
                                          prompt=f"{body}\nThe task for {t['ents'][t['qi']]} is:"))
    if a.cells:
        keep = {tuple(int(x) for x in c.split(":")) for c in a.cells.split(",")}
        for c in trials:
            D = 0 if c in ("D0", "lenfix", "verbatim", "shots2", "taskcolon", "xis", "matched") else 256
            trials[c] = [t for t in trials[c] if (t["K"], D, t["seed"]) in keep]
        trials = {c: v for c, v in trials.items() if v}
        conds = [c for c in conds if c in trials]
    enc = {c: encode(tok, trials[c], SEQ) for c in trials}

    def meta(c):
        T = trials[c]
        vals = np.full((len(T), 64), -1, np.int16)
        en = np.full((len(T), 64), -1, np.int16)
        for j, t in enumerate(T):
            vals[j, : t["K"]] = [w2i[v] for v in t["vals"]]
            en[j, : t["K"]] = [e2i[e] for e in t["ents"]]
        return dict(K=np.array([t["K"] for t in T], np.int16),
                    seed=np.array([t["seed"] for t in T], np.int8),
                    idx=np.array([t["i"] for t in T], np.int16),
                    qi=np.array([t["qi"] for t in T], np.int16),
                    gold=np.array([w2i[t["vals"][t["qi"]]] for t in T], np.int16),
                    vals=vals, ents=en, ntok=enc[c][3], preq_idx=enc[c][2])
    # the token index must not be saved as "preq": the arrays below use "<cond>_preq" for the
    # pre-question hidden state, and the runs of 2026-10-24 lost that state to this name clash

    metas = {c: meta(c) for c in trials}
    odir = os.path.join(a.out, a.label)
    os.makedirs(odir, exist_ok=True)
    prev_md5 = None
    for rev in a.revisions.split(","):
        jpath = os.path.join(odir, f"{rev}.json")
        if os.path.exists(jpath):
            prev_md5 = json.load(open(jpath)).get("embed_md5")
            print(f"[{a.label}] {rev} done, skipped", flush=True)
            continue
        t0 = time.time()
        kw = dict(revision=rev)
        try:
            model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt, **kw)
        model.to(a.device).eval()
        md5 = embed_md5(model)
        L = model.config.num_hidden_layers
        layers, lmid = layer_set(L), L // 2
        # OPT-350M projects its last hidden state to word_embed_proj_dim (512, not 1024); use the
        # residual stream before that projection, after layer L-1, so every saved vector has one size
        if getattr(model.config, "word_embed_proj_dim", model.config.hidden_size) != model.config.hidden_size:
            layers = sorted({l if l < L else L - 1 for l in layers})
        rec = dict(label=a.label, model=a.model, revision=rev, embed_md5=md5, n_layers=L,
                   layers_saved=layers, preq_layer=lmid, seq=SEQ, dtype=a.dtype, batch=a.batch,
                   pad=a.pad, tf32=a.tf32, position_ids_from_mask=model.config.model_type == "gpt2",
                   device=torch.cuda.get_device_name(0) if a.device == "cuda" else a.device,
                   torch=torch.__version__, transformers=transformers.__version__,
                   n_obligations=len(obl_words), max_k=max_k, load_sec=round(time.time() - t0, 1))
        if md5 == prev_md5:
            rec["duplicate_of_previous"] = True
            json.dump(rec, open(jpath, "w"), indent=1)
            print(f"[{a.label}] {rev} has the same embedding as the previous checkpoint; skipped",
                  flush=True)
            del model
            continue
        run_conds = list(conds) + ([c for c in ctl if c in trials]
                                   if rev in a.control_revisions.split(",") else [])
        arrays, cells = {}, []
        for c in run_conds:
            LG, H, P = run_checkpoint(a, model, tok, obl_ids, trials[c], enc[c], layers, lmid,
                                      a.device)
            m = metas[c]
            pred = LG.argmax(1)
            listed = np.where(m["vals"] >= 0, m["vals"], 0)
            lg_listed = np.where(m["vals"] >= 0, np.take_along_axis(LG, listed, 1), -np.inf)
            pred_listed = np.take_along_axis(listed, lg_listed.argmax(1)[:, None], 1)[:, 0]
            for K in sorted(set(m["K"].tolist())):
                for s in sorted(set(m["seed"].tolist())):
                    sel = (m["K"] == K) & (m["seed"] == s)
                    cells.append(dict(cond=c, K=int(K), seed=int(s), n=int(sel.sum()),
                                      recall=float((pred[sel] == m["gold"][sel]).mean()),
                                      recall_listed=float((pred_listed[sel] == m["gold"][sel]).mean())))
            arrays.update({f"{c}_logits": LG, f"{c}_hidden": H, f"{c}_preq": P,
                           **{f"{c}_{k}": v for k, v in m.items()}})
            print(f"[{a.label}] {rev} {c}: n={len(LG)} recall={float((pred == m['gold']).mean()):.3f}",
                  flush=True)
        np.savez(os.path.join(odir, f"{rev}.npz"), obl_words=np.array(obl_words),
                 ent_words=np.array(ents), layers=np.array(layers), **arrays)
        rec.update(conds=run_conds, cells=cells, elapsed_sec=round(time.time() - t0, 1),
                   _provenance="analysis/gpu_capacity_dev.py; prompts from neuron_capacity.make_prompt")
        json.dump(rec, open(jpath + ".tmp", "w"), indent=1)
        os.replace(jpath + ".tmp", jpath)                 # the .json marks a finished checkpoint
        prev_md5 = md5
        del model, arrays
        if a.device == "cuda":
            torch.cuda.empty_cache()
        print(f"[{a.label}] {rev} done in {rec['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
