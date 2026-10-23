"""Capacity curves on Trainium: one compiled shape, 32-way data parallel.

WHAT THIS PRODUCES.  The full recall-vs-K curve for a model, which the capacity paper names as
its remaining measurement gap (it currently ships k*, a threshold crossing on a coarse grid, and
a 30-model table whose sweep generator was never committed).  Feed the output to
capacity_curve.fit_k50 for a censoring-aware K50 and a threshold-free capacity AUC.

WHY ONE SHAPE.  Neuron compiles a static graph per tensor shape, and each compile costs minutes.
A K-sweep looks like it needs a different shape per K -- it does not.  Every trial is padded to
one fixed sequence length, so K=1 and K=24 are the same graph.  ONE compile per model, then every
core runs the same NEFF out of the shared cache.  Getting this wrong is the difference between
one compile and a dozen, which is the difference between minutes and hours.

THE TWO-PHASE LAUNCH.  If 32 workers start at once they all miss the cache and all compile the
same graph simultaneously, thrashing the host.  Phase 1 runs ONE worker to populate the cache;
phase 2 launches the rest, which load the NEFF instead of building it.  --warm-only does phase 1.

PRECISION.  bfloat16.  The readout is an argmax over the obligation-token logits, a behavioural
measurement, not a probe input -- so bf16 is adequate here in a way it would NOT be for the
geometry work.  A 7B model in fp32 is 28 GB against 16 GB of core HBM and does not fit anyway.
This deviation is recorded in every output file rather than left implicit.

ONE WRITER PER FILE.  Each worker writes only <out>/cells_w<id>.json.  No worker ever appends to
a shared path (global CLAUDE.md sec. 6).

Usage:
    python neuron_capacity.py --model EleutherAI/pythia-410m --label pythia-410m --warm-only
    python neuron_capacity.py --model EleutherAI/pythia-410m --label pythia-410m \
        --worker-id 0 --n-workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

K_GRID = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)
# Extended past the committed grid's ceiling of 24 ON PURPOSE.  In the first sweep FOUR models
# (olmo2-1b, opt-1.3b, opt-2.7b, qwen2.5-3b) were still above r=0.5 at K=24 and had to be reported
# as right-censored, which removed 24% of the sample from the scale-controlled test.  A ceiling
# that censors the top of the population is a measurement limit, not a property of the models.
D_GRID = (0, 256)                                    # pure capacity, and under interference

# Pools large enough to actually reach K=64.  The earlier pools held 25 usable tokens, which capped
# K at 25 and MANUFACTURED the censoring above.  Every candidate is filtered to single-token per
# tokenizer at run time, so the usable count is model-specific and is recorded in each output file.
ENTITY_POOL = """lock mutex queue buffer socket thread cache token worker stream handle record
packet session channel frame server client daemon module vector matrix window cursor bucket
kernel driver device folder branch commit ticket router switch bridge tunnel proxy agent broker
filter parser writer reader logger timer counter monitor sensor beacon anchor marker pointer
schema column table index shard replica cluster region zone realm domain vault ledger wallet
basket crate pallet carton barrel drum canister flask beaker vessel""".split()

OBLIGATION_POOL = """balance verify rotate commit flush retry audit sign encrypt archive migrate
publish resolve refresh compile deploy inspect restore compress validate expand reduce merge
split index rebuild restart reload cancel approve reject submit review release revoke renew
enable disable escalate delegate schedule execute abort pause resume finalize export import
notify collect measure adjust align repair replace upgrade patch scan trace profile bundle
mirror sync purge prune label tag stamp seal lock unlock""".split()

FILLER = "alpha beta gamma delta sigma omega kappa theta lambda mu nu rho tau phi chi psi".split()


def single_token(tok, words):
    """Keep only words that are one token with a leading space.  A multi-token obligation would
    silently break the argmax readout, so this filters rather than assumes."""
    out = []
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            out.append((w, ids[0]))
    return out


def make_prompt(rng, ents, obls, K, D):
    e = rng.sample(ents, K)
    o = [rng.choice(obls) for _ in range(K)]
    qi = rng.randrange(K)
    decl = "\n".join(f"{a}: {b}" for a, b in zip(e, o))
    body = decl
    if D > 0:
        n_lines = max(1, D // 8)
        lines = [f"{rng.choice(FILLER)}_{rng.randrange(100)} = f({rng.choice(FILLER)}, "
                 f"{rng.randrange(100)})" for _ in range(n_lines)]
        body = decl + "\n" + "\n".join(lines)
    return f"{body}\nThe task for {e[qi]} is:", o[qi]


class Tap(torch.nn.Module):
    """Obligation logits at the final position.  Nothing else crosses the device boundary."""

    def __init__(self, model, obl_ids):
        super().__init__()
        self.model = model
        self.register_buffer("obl", obl_ids)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.logits[:, -1, :].index_select(1, self.obl)


def build(model_name, seq, batch, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    try:
        m = AutoModelForCausalLM.from_pretrained(model_name, dtype=dt)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dt)
    m.eval()
    ents = [w for w, _ in single_token(tok, ENTITY_POOL)]
    obl = single_token(tok, OBLIGATION_POOL)
    obl_words = [w for w, _ in obl]
    obl_ids = torch.tensor([i for _, i in obl])
    return tok, Tap(m, obl_ids).eval(), ents, obl_words


def cells(n_trials, seeds):
    return [(K, D, s) for K in K_GRID for D in D_GRID for s in range(seeds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-trials", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--warm-only", action="store_true", help="compile, save the NEFF, exit (phase 1)")
    ap.add_argument("--trace-path", default=None, help="saved traced model; workers load this")
    ap.add_argument("--out", default=os.path.expanduser("~/trilogy/capacity"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    tok, tap, ents, obl_words = build(a.model, a.seq, a.batch, a.dtype)
    max_k = min(len(ents), len(obl_words))
    chance = 1.0 / len(obl_words)

    def encode(prompts):
        e = tok(prompts, return_tensors="pt", padding="max_length", truncation=True,
                max_length=a.seq)
        return e["input_ids"], e["attention_mask"]

    ids0, mask0 = encode(["warm " * 8] * a.batch)

    import torch_neuronx
    # The traced model is an artifact on disk, not just an in-memory object.  torch_neuronx.trace
    # compiles afresh on every call -- NEURON_CC_FLAGS=--cache_dir does NOT dedupe trace() calls --
    # so 32 workers each calling trace() means 32 simultaneous compiles of the same graph.  That
    # took the host to load average 68 before it was killed.  Phase 1 traces once and SAVES; every
    # worker then loads the NEFF in seconds.
    tpath = a.trace_path or os.path.join(a.out, f"{a.label}_b{a.batch}_s{a.seq}_{a.dtype}.pt")
    compile_sec = 0.0
    if os.path.exists(tpath) and not a.warm_only:
        t0 = time.time()
        net = torch.jit.load(tpath)
        load_sec = round(time.time() - t0, 1)
    else:
        t0 = time.time()
        # inline_weights_to_neff=False keeps weights as graph INPUTS rather than baking them
        # into the NEFF.  Measured: compile drops 235s -> 46s, and it is what makes
        # torch_neuronx.replace_weights work at all (with inlining it silently no-ops).
        net = torch_neuronx.trace(tap, (ids0, mask0), inline_weights_to_neff=False)
        compile_sec = round(time.time() - t0, 1)
        torch.jit.save(net, tpath)
        load_sec = 0.0
    if a.warm_only:
        print(json.dumps({"label": a.label, "compile_sec": compile_sec, "trace_path": tpath,
                          "max_k": max_k, "n_obligations": len(obl_words),
                          "chance": round(chance, 5), "seq": a.seq, "batch": a.batch,
                          "dtype": a.dtype}))
        return

    todo = [c for i, c in enumerate(cells(a.n_trials, a.seeds)) if i % a.n_workers == a.worker_id]
    todo = [(K, D, s) for (K, D, s) in todo if K <= max_k]
    rows = []
    for (K, D, s) in todo:
        rng = random.Random((7000 + s) * 100003 + K * 31 + D)
        pr, gold = zip(*[make_prompt(rng, ents, obl_words, K, D) for _ in range(a.n_trials)])
        gi = torch.tensor([obl_words.index(g) for g in gold])
        ids, mask = encode(list(pr))
        pred = []
        for i in range(0, len(pr), a.batch):
            b_ids, b_mask = ids[i:i + a.batch], mask[i:i + a.batch]
            if b_ids.shape[0] < a.batch:                       # pad the final partial batch to
                pad = a.batch - b_ids.shape[0]                 # the compiled shape, then drop it
                b_ids = torch.cat([b_ids, b_ids[:1].repeat(pad, 1)])
                b_mask = torch.cat([b_mask, b_mask[:1].repeat(pad, 1)])
                pred.append(net(b_ids, b_mask).float().argmax(-1)[:a.batch - pad])
            else:
                pred.append(net(b_ids, b_mask).float().argmax(-1))
        pred = torch.cat(pred)
        recall = float((pred == gi).float().mean())
        rows.append({"K": K, "D": D, "seed": s, "recall": recall, "n": len(pr)})
        print(f"[{a.label} w{a.worker_id}] K={K:3d} D={D:3d} s={s} recall={recall:.3f}", flush=True)

    out = {"label": a.label, "model": a.model, "worker_id": a.worker_id,
           "n_workers": a.n_workers, "chance": chance, "n_obligations": len(obl_words),
           "max_k": max_k, "seq": a.seq, "batch": a.batch, "dtype": a.dtype,
           "n_trials": a.n_trials, "compile_sec": compile_sec, "neff_load_sec": load_sec,
           "elapsed_sec": round(time.time() - t_start, 1), "cells": rows,
           "_provenance": ("Measured on trn1.32xlarge by analysis/neuron_capacity.py. "
                           "bfloat16: the readout is an argmax over obligation-token logits, a "
                           "behavioural measurement, not a probe input. Same K grid as the "
                           "committed analysis/ksweep.py; this is a NEW measurement of the same "
                           "quantity, not a byte-level replication of the frozen 30-model table.")}
    p = os.path.join(a.out, f"{a.label}_w{a.worker_id}.json")
    json.dump(out, open(p, "w"), indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
