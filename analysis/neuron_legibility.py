"""The legibility gap across STATE OPERATIONS, not just static declaration recall.

THE OBJECTION THIS ANSWERS.  The paper's result -- models often retain the correct query-specific
binding on trials where they answer wrong -- is measured on one operation: declare once, recall
later.  A reviewer can reasonably say that is a lookup task, and that "state" is doing no work in
the claim.  This measures the same quantity when the model must OVERWRITE state and when it must
SCOPE it, using generators whose controls are in analysis/trilogy_tasks.py.

  declarative  each entity assigned once; recall it
  mutable      each entity assigned twice; report the FINAL value.  Query recency is sampled
               uniformly, so "answer the most recent value" cannot score (trilogy_tasks asserts it)
  scoped       an entity is rebound inside a block; the query asks for its value OUTSIDE, so a
               most-recent-assignment heuristic is wrong by construction

PER MODEL AND FAMILY, on the TEST fold only:
  behaviour             accuracy
  legibility gap        p(probe correct | model wrong) minus the 1/K present-set baseline
  query counterfactual  same declarations, different queried entity -- does the probe track the
                        QUERY or just the declaration block?  This is the control that decides
                        whether the probe reads the binding or merely the entity code
  disagreement AUROC    signed probe-vs-output disagreement as an error detector
  repair                patch the probe-decoded target into the residual stream at the read layer

REPAIR ARMS (all on the same trials, matched norms):
  noop            patch the recipient's own state back in.  MUST reproduce baseline exactly, and
                  MUST give zero recovery, since every scored trial was wrong by construction
  decoded         the probe's decoded binding -- the non-oracle intervention
  wrong_target    a deliberately wrong obligation's target
  random_matched  norm-matched random direction
  gold            oracle upper bound
Reporting `decoded` without `wrong_target` and `random_matched` would not distinguish "restored
the binding" from "any large perturbation at this site knocks the model off its error".

LAYER SELECTION IS LEAK-FREE.  The read layer is chosen on the SELECTION fold and applied to the
test fold.  Choosing it on test and then quoting a test number is the standard way this measurement
is inflated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trilogy_tasks import build_declarative, build_mutable, build_scoped  # noqa: E402

FILLER = "alpha beta gamma delta sigma omega kappa theta".split()


def render(trial, ent_map, val_map, D, rng, explicit=False):
    """Abstract trial -> concrete prompt, using this tokenizer's single-token vocabulary.

    `explicit` states the task's semantics in the prompt instead of leaving them to be inferred.
    This exists because near-baseline accuracy on the mutable task is ambiguous between two very
    different failures: the model cannot TRACK an overwrite, or the model does not know that the
    LAST assignment is the one being asked about.  Without the explicit condition, "cannot track
    mutable state" is not a licensed reading of the number.
    """
    lines, depth = [], 0
    preamble = ""
    if explicit:
        preamble = {
            "mutable": "An entity may be assigned more than once. Its current value is the value "
                       "from its MOST RECENT assignment.\n",
            "scoped": "Assignments inside BEGIN/END apply only within that block. Outside the "
                      "block, the value from before the block is current.\n",
        }.get(trial.task_family, "")
    for op, e, v in trial.body:
        if op == "set":
            lines.append("    " * depth + f"{ent_map[e]}: {val_map[v]}")
        elif op == "begin":
            lines.append("    " * depth + "BEGIN"); depth += 1
        elif op == "end":
            depth -= 1; lines.append("    " * depth + "END")
    body = "\n".join(lines)
    if D > 0:
        body += "\n" + "\n".join(
            f"{rng.choice(FILLER)}_{rng.randrange(100)} = f({rng.choice(FILLER)}, {rng.randrange(100)})"
            for _ in range(max(1, D // 12)))
    q = ent_map[trial.query]
    stem = ("The current task for" if trial.task_family == "mutable"
            else "Outside the block, the task for" if trial.task_family == "scoped"
            else "The task for")
    return f"{preamble}{body}\n{stem} {q} is:", val_map[trial.gold]


class PatchBlock(torch.nn.Module):
    """Broadcast-add at the final position.

    Owns its own position mask and its own patch/alpha.  It must NOT hold a reference back to the
    Tap that wraps it: Tap holds the model, the model holds this block, and a back-reference makes
    the module graph CYCLIC -- state_dict() then recurses until it dies (measured: RecursionError
    after 986 frames during torch_neuronx.trace).  The parent sets `blk.patch`/`blk.alpha` before
    calling the model; plain tensor attributes on an nn.Module are not registered, so they add no
    edge to the graph.

    The add is a broadcast, never an in-place indexed write -- an in-place write at a traced index
    can silently become a no-op in a compiled graph, which is what the alpha=0 arm checks.
    """

    def __init__(self, inner, seq):
        super().__init__()
        self.inner = inner
        pos = torch.zeros(1, seq, 1); pos[0, -1, 0] = 1.0
        self.register_buffer("pos", pos)
        self.patch = None
        self.alpha = None

    def forward(self, hidden_states, *a, **kw):
        out = self.inner(hidden_states, *a, **kw)
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        if self.patch is not None:
            h = h + self.pos * (self.alpha * self.patch).unsqueeze(1)
        return ((h,) + tuple(out[1:])) if tup else h


class Tap(torch.nn.Module):
    def __init__(self, model, obl_ids, block):
        super().__init__()
        self.model = model
        self.block = block          # also reachable via self.model: a diamond, not a cycle
        self.register_buffer("obl", obl_ids)

    def forward(self, input_ids, attention_mask, patch, alpha):
        self.block.patch, self.block.alpha = patch, alpha
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, use_cache=False)
        hs = torch.stack(out.hidden_states, 0)[:, :, -1, :]
        return out.logits[:, -1, :].index_select(1, self.obl), hs


def blocks_of(model):
    for path in ("gpt_neox.layers", "model.layers", "transformer.h"):
        o = model
        try:
            for p in path.split("."):
                o = getattr(o, p)
            return o
        except AttributeError:
            continue
    raise RuntimeError("cannot locate transformer blocks")


def auroc(scores, labels):
    """Rank-based AUROC; labels 1 = error."""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--seq", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch-layer-frac", type=float, default=0.75)
    ap.add_argument("--families", default="declarative,mutable,scoped")
    ap.add_argument("--explicit", action="store_true",
                    help="state the task semantics in the prompt instead of "
                         "requiring the model to infer them")
    ap.add_argument("--out", default=os.path.expanduser("~/trilogy/legibility"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()

    ents = [w for w in """lock mutex queue buffer socket thread cache token worker stream handle
        record packet session channel frame server client daemon module""".split()
        if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
    vals = [w for w in """balance verify rotate commit flush retry audit sign encrypt archive
        migrate publish resolve refresh compile deploy inspect restore""".split()
        if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
    obl_ids = torch.tensor([tok.encode(" " + w, add_special_tokens=False)[0] for w in vals])
    chance = 1.0 / len(vals)

    layers = blocks_of(model)
    pl = max(0, min(len(layers) - 1, int(round(a.patch_layer_frac * (len(layers) - 1)))))
    blk = PatchBlock(layers[pl], a.seq)
    layers[pl] = blk
    tap = Tap(model, obl_ids, blk).eval()
    H = model.config.hidden_size

    import torch_neuronx
    import random as _r
    rng = _r.Random(0)
    ex_ids = tok(["warm"] * a.batch, return_tensors="pt", padding="max_length",
                 truncation=True, max_length=a.seq)
    net = torch_neuronx.trace(tap, (ex_ids["input_ids"], ex_ids["attention_mask"],
                                    torch.zeros(a.batch, H), torch.zeros(1)),
                              inline_weights_to_neff=False)
    compile_sec = round(time.time() - t0, 1)
    print(f"compiled in {compile_sec}s, patch layer {pl}/{len(layers)}", flush=True)

    from sklearn.linear_model import LogisticRegression
    out = {"label": a.label, "model": a.model, "K": a.K, "D": a.D, "seq": a.seq,
           "explicit_semantics": bool(a.explicit),
           "chance": round(chance, 5), "n_values": len(vals), "patch_layer": pl,
           "n_layers": len(layers), "compile_sec": compile_sec, "families": {}}

    builders = {"declarative": build_declarative, "mutable": build_mutable, "scoped": build_scoped}
    for fam in a.families.split(","):
        tr = builders[fam](a.n, K=a.K, D=a.D, seed=0)
        emap = {f"ENTITY_{i:02d}": ents[i % len(ents)] for i in range(64)}
        vmap = {f"VALUE_{i:02d}": vals[i % len(vals)] for i in range(64)}
        rendered = [render(t, emap, vmap, a.D, rng, a.explicit) for t in tr]
        prompts = [p for p, _ in rendered]
        gold = np.array([vals.index(g) for _, g in rendered])
        folds = np.array([t.fold for t in tr])
        enc = tok(prompts, return_tensors="pt", padding="max_length", truncation=True,
                  max_length=a.seq)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        n = len(prompts)

        def run(patch=None, alpha=0.0):
            LO, HS = [], []
            z = torch.zeros(a.batch, H)
            for i in range(0, n, a.batch):
                bi, bm = ids[i:i + a.batch], mask[i:i + a.batch]
                k = bi.shape[0]
                if k < a.batch:
                    bi = torch.cat([bi, bi[:1].repeat(a.batch - k, 1)])
                    bm = torch.cat([bm, bm[:1].repeat(a.batch - k, 1)])
                pv = z if patch is None else patch[i:i + a.batch]
                if pv.shape[0] < a.batch:
                    pv = torch.cat([pv, torch.zeros(a.batch - pv.shape[0], H)])
                lo, hs = net(bi, bm, pv, torch.tensor([alpha]))
                LO.append(lo.float()[:k]); HS.append(hs.float()[:, :k])
            return torch.cat(LO), torch.cat(HS, 1)

        base_lo, base_hs = run()
        pred = base_lo.argmax(-1).numpy()
        acc = float((pred == gold).mean())

        trf, self_, tef = folds == "probe", folds == "selection", folds == "test"
        best, best_s = 0, -1.0
        for L in range(base_hs.shape[0]):
            X = base_hs[L].numpy()
            p = LogisticRegression(max_iter=2000).fit(X[trf], gold[trf])
            s = float(p.score(X[self_], gold[self_]))
            if s > best_s:
                best_s, best, bp = s, L, p
        Xb = base_hs[best].numpy()
        wrong = tef & (pred != gold)
        pacc_wrong = (float((bp.predict(Xb[wrong]) == gold[wrong]).mean())
                      if wrong.sum() >= 25 else None)

        # SHUFFLED-LABEL NULL.  A probe fit on randomly permuted labels, same features, same folds,
        # same regulariser.  It MUST land at chance.  If it does not, something leaks -- the fold
        # split, the feature extraction, or the class balance -- and every accessibility number
        # above is inflated by that leak.  This is the cheapest possible check on the whole probe
        # pipeline and costs no forward passes.
        _rs = np.random.RandomState(0)
        y_shuf = gold.copy()
        y_shuf[trf] = _rs.permutation(gold[trf])
        shuf_probe = LogisticRegression(max_iter=2000).fit(Xb[trf], y_shuf[trf])
        shuf_acc = float((shuf_probe.predict(Xb[tef]) == gold[tef]).mean())

        proba = bp.predict_proba(Xb)
        pcls = bp.classes_
        pidx = {c: i for i, c in enumerate(pcls)}
        model_p = torch.softmax(base_lo, -1).numpy()
        emitted = pred
        sgn = np.array([proba[i, pidx.get(emitted[i], 0)] for i in range(n)])
        disagree = -sgn                                     # high = probe doubts the emitted token
        au = auroc(disagree[tef], (pred != gold)[tef].astype(int))

        fam_out = {"n": n, "accuracy": round(acc, 4), "n_test": int(tef.sum()),
                   "n_wrong_test": int(wrong.sum()), "read_layer": int(best),
                   "probe_selection_acc": round(best_s, 4),
                   "p_acc_given_wrong": None if pacc_wrong is None else round(pacc_wrong, 4),
                   "shuffled_label_probe_acc": round(shuf_acc, 4),
                   "shuffled_label_expected": round(1.0 / len(vals), 4),
                   "present_set_baseline": round(1.0 / a.K, 4),
                   "disagreement_auroc": None if au is None else round(au, 4), "repair": {}}

        if wrong.sum() >= 25:
            idx = np.where(wrong)[0]
            # THE INTERVENTION SITE MUST MATCH THE CAPTURE SITE, on both counts:
            #   (a) the patch is applied to the output of block `pl`, and hidden_states[i] is the
            #       INPUT to block i, so block pl's output is hidden_states[pl+1].  Off by one and
            #       the "identity" patch transplants a neighbouring layer.
            #   (b) the decoded target must live in the space we patch, so the repair probe is fit
            #       at pl+1 -- not at the probe-selected read layer `best`, which is chosen for
            #       decodability and is generally a different layer.
            # Both were wrong in the first run and the noop arm caught it: identity-patching gave
            # 0.128 recovery where it must give exactly 0.
            site = min(pl + 1, base_hs.shape[0] - 1)
            Xp = base_hs[site].numpy()
            rp = LogisticRegression(max_iter=2000).fit(Xp[trf], gold[trf])
            fam_out["repair_probe_layer"] = int(site)
            fam_out["repair_probe_selection_acc"] = round(float(rp.score(Xp[self_], gold[self_])), 4)
            mu = np.stack([Xp[(gold == c) & trf].mean(0) for c in range(len(vals))])
            dec = rp.predict(Xp)
            recip = base_hs[site]
            # THE PATCH IS ADDITIVE: PatchBlock computes h = h + pos * (alpha * patch).  So every
            # arm must supply a DELTA, not a target.  Supplying the target adds the class mean on
            # top of the activation that is already there (and supplying `recip` doubles it, which
            # is why the first run's noop arm scored 0.095 instead of 0).  The noop arm is now
            # literally zero, so it is an exact identity by construction and its logit drift is
            # the check that the patch path itself is faithful.
            tgt = torch.tensor(mu[dec], dtype=torch.float32)
            wrongc = (dec + 1) % len(vals)
            delta = tgt - recip
            g = torch.randn_like(recip)
            arms = {
                "noop": torch.zeros_like(recip),
                "decoded": delta,
                "wrong_target": torch.tensor(mu[wrongc], dtype=torch.float32) - recip,
                "random_matched": g / g.norm(dim=-1, keepdim=True) * delta.norm(dim=-1, keepdim=True),
                "gold": torch.tensor(mu[gold], dtype=torch.float32) - recip,
            }
            for name, vec in arms.items():
                lo2, _ = run(patch=vec, alpha=1.0)
                p2 = lo2.argmax(-1).numpy()
                rec = float((p2[idx] == gold[idx]).mean())
                keep = float((p2[tef & (pred == gold)] == gold[tef & (pred == gold)]).mean())
                e = {"recovery_on_wrong": round(rec, 4), "correct_preserved": round(keep, 4)}
                if name == "noop":
                    e["max_logit_drift"] = float(f"{(lo2 - base_lo).abs().max():.3e}")
                fam_out["repair"][name] = e
                print(f"  [{fam}] repair {name:<15} recovery={rec:.3f} preserved={keep:.3f}",
                      flush=True)

        # ---------------------------------------------------------------- QUERY COUNTERFACTUAL
        # THE CONTROL THAT DECIDES WHAT THE PROBE IS READING.  A probe at the query position can
        # score highly for two very different reasons: it reads the BINDING for the entity that
        # was asked about, or it reads something query-independent (an entity code, or a summary
        # of the declaration block) that happens to correlate with the answer.  Only the first
        # supports "the model represents the queried binding".
        #
        # The test: keep the declaration block BIT-IDENTICAL and change ONLY the queried entity.
        # If the probe tracks the binding, its decode must follow the query to the new entity's
        # value.  If it is reading the block, the decode will not move.
        #
        # Reported as tracking rate against the 1/K floor.  Declarative only: for mutable and
        # scoped the counterfactual gold depends on update order and scope depth, and getting that
        # rule subtly wrong would silently manufacture the answer.
        if fam == "declarative":
            cf_prompts, cf_gold, cf_src = [], [], []
            for i, t in enumerate(tr):
                pairs = [(e, v) for op, e, v in t.body if op == "set" and e != t.query]
                if not pairs:
                    continue
                e2, v2 = pairs[rng.randrange(len(pairs))]
                body = "\n".join(f"{emap[e]}: {vmap[v]}" for op, e, v in t.body if op == "set")
                cf_prompts.append(f"{body}\nThe task for {emap[e2]} is:")
                cf_gold.append(vals.index(vmap[v2]))
                cf_src.append(i)
            if cf_prompts:
                cf_gold = np.array(cf_gold); cf_src = np.array(cf_src)
                enc2 = tok(cf_prompts, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=a.seq)
                ids_o, mask_o, n_o = ids, mask, n
                ids, mask, n = enc2["input_ids"], enc2["attention_mask"], len(cf_prompts)
                cf_lo, cf_hs = run()
                ids, mask, n = ids_o, mask_o, n_o
                Xcf = cf_hs[best].numpy()
                cf_probe_pred = bp.predict(Xcf)
                cf_model_pred = cf_lo.argmax(-1).numpy()
                on_test = np.isin(cf_src, np.where(tef)[0])
                fam_out["query_counterfactual"] = {
                    "n": int(on_test.sum()),
                    "probe_tracks_new_query": round(
                        float((cf_probe_pred[on_test] == cf_gold[on_test]).mean()), 4),
                    "probe_stays_on_old_answer": round(
                        float((cf_probe_pred[on_test] == gold[cf_src][on_test]).mean()), 4),
                    "model_tracks_new_query": round(
                        float((cf_model_pred[on_test] == cf_gold[on_test]).mean()), 4),
                    "floor_1_over_K": round(1.0 / a.K, 4),
                }
                print(f"  [{fam}] query-counterfactual: probe follows new query "
                      f"{fam_out['query_counterfactual']['probe_tracks_new_query']:.3f}, "
                      f"stays on old {fam_out['query_counterfactual']['probe_stays_on_old_answer']:.3f} "
                      f"(floor {1.0/a.K:.3f})", flush=True)

        # -------------------------------------------------- PER-TRIAL EXPORT (for risk-coverage)
        # Summary AUROC cannot answer the operational question ("at 90% coverage, how much error
        # is removed?"), and it cannot be compared against the DUMBEST baseline: predicting error
        # from trial difficulty alone (K, the queried entity's serial position, prompt length).
        # If difficulty predicts error as well as probe-output disagreement does, the mechanistic
        # signal buys nothing operationally, and that has to be checkable rather than assumed.
        fam_out["per_trial"] = [{
            "fold": str(folds[i]),
            "correct": bool(pred[i] == gold[i]),
            "disagreement": float(disagree[i]),
            "model_margin": float(np.partition(model_p[i], -2)[-1] - np.partition(model_p[i], -2)[-2]),
            "model_entropy": float(-(model_p[i] * np.log(model_p[i] + 1e-12)).sum()),
            "probe_conf": float(proba[i].max()),
            "query_slot": int(tr[i].meta.get("query_slot", -1)),
            "n_tokens": int(mask[i].sum()),
            "K": int(a.K),
        } for i in range(n)]

        out["families"][fam] = fam_out
        print(f"[{a.label}/{fam}] acc={acc:.3f} p(acc|wrong)="
              f"{'na' if pacc_wrong is None else f'{pacc_wrong:.3f}'} "
              f"(baseline {1.0/a.K:.3f}) AUROC={au} L*={best}", flush=True)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    out["_provenance"] = ("Measured on trn1.32xlarge by analysis/neuron_legibility.py. Task "
                          "generators and folds from analysis/trilogy_tasks.py (folds are a hash "
                          "of the trial, so probe/selection/test membership is fixed). Read layer "
                          "chosen on the SELECTION fold, reported on TEST. Patch verified by the "
                          "noop arm, which must reproduce the unpatched logits.")
    p = os.path.join(a.out, f"{a.label}.json")
    json.dump(out, open(p, "w"), indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
