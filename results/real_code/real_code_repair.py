"""REPAIR on real code: re-inject the decoded value and flip a wrong output to correct.

P1's third claim, ported to a real code model:
  On a trial the model gets WRONG, add the difference-of-means direction for the CORRECT value into the
  residual at a mid layer, let the network finish, and the output flips to correct -- inference-time
  capability recovery. The control (P1's format-matched wrong-content) injects a DIFFERENT present value's
  direction and must NOT flip it to correct.

Reuses the task from real_code_gap.py. Injection is a forward hook at layer L* adding alpha*(mu_gold - mu_bar)
at the query token; the forward then continues through the remaining layers, so the network can USE it.
"""
import argparse, json, random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import real_code_gap as G           # make_trial, single_token_words, POOL, _vals_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/deepseek-coder-6.7b-base")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--reassign", type=int, default=20)
    ap.add_argument("--layer", type=int, default=16, help="inject here; downstream layers then USE it")
    ap.add_argument("--alphas", default="0,2,4,8,16")
    ap.add_argument("--max-wrong", type=int, default=400)
    ap.add_argument("--out", default="/tmp/rcg_repair.json")
    a = ap.parse_args()

    rng = random.Random(0); torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    words = G.single_token_words(tok, G.POOL)
    val_id = {w: tok.encode(w, add_special_tokens=False)[0] for w in words}
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda").eval()
    L = a.layer
    block = model.model.layers[L]                     # inject at the OUTPUT of this decoder block

    # 1) run all trials once, record residual at L (last token), model pred (restricted), gold, present-set
    trials = []
    Hs = []
    for t in range(a.n):
        prompt, gold, target, depth, _last = G.make_trial(rng, words, a.k, a.reassign)
        present = list({val_id[w] for w in set(G._vals_in(prompt, words))})
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model(**ids, output_hidden_states=True)
        h = o.hidden_states[L][0, -1].float().cpu().numpy()
        logits = o.logits[0, -1]
        pred = present[int(torch.stack([logits[i] for i in present]).argmax())]
        Hs.append(h)
        trials.append(dict(prompt=prompt, gold=gold, present=present, pred=pred,
                           correct=int(pred == val_id[gold])))
    Hs = np.stack(Hs)

    # 2) difference-of-means directions per value word, from CORRECT trials only (clean estimate)
    ok = np.array([t["correct"] for t in trials]).astype(bool)
    mu_bar = Hs[ok].mean(0)
    mu = {}
    for w in words:
        m = np.array([i for i, t in enumerate(trials) if t["correct"] and t["gold"] == w])
        if len(m) >= 3:
            mu[w] = Hs[m].mean(0) - mu_bar
    dirn = {w: torch.tensor(v, device="cuda", dtype=torch.float16) for w, v in mu.items()}

    wrong_idx = [i for i, t in enumerate(trials) if not t["correct"] and t["gold"] in mu]
    rng.shuffle(wrong_idx); wrong_idx = wrong_idx[:a.max_wrong]
    print(f"model_acc={ok.mean():.3f}  wrong usable={len(wrong_idx)}  inject@L{L}  alphas={a.alphas}")

    # 3) hook that adds a fixed vector at the LAST position of block L's output
    inj = {"vec": None}
    def hook(mod, inp, out):
        if inj["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] = h[:, -1, :] + inj["vec"]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    handle = block.register_forward_hook(hook)

    def run_restricted(prompt, present, vec):
        inj["vec"] = vec
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**ids).logits[0, -1]
        inj["vec"] = None
        return present[int(torch.stack([lg[i] for i in present]).argmax())]

    alphas = [float(x) for x in a.alphas.split(",")]
    res = {"model": a.model, "layer": L, "alphas": alphas, "n_wrong": len(wrong_idx),
           "model_acc": float(ok.mean()), "gold_flip": {}, "ctrl_flip": {}, "ctrl_to_ctrl": {}}
    for al in alphas:
        gold_hits = ctrl_help = ctrl_lands = 0
        for i in wrong_idx:
            t = trials[i]; g = t["gold"]
            # REPAIR: inject toward the correct value
            pg = run_restricted(t["prompt"], t["present"], al * dirn[g])
            gold_hits += int(pg == val_id[g])
            # CONTROL: inject toward a random PRESENT-but-wrong value w'
            alts = [w for w in words if w in mu and val_id[w] in t["present"] and w != g]
            if alts:
                wj = rng.choice(alts)
                pc = run_restricted(t["prompt"], t["present"], al * dirn[wj])
                ctrl_help += int(pc == val_id[g])          # did wrong-content injection accidentally fix it? (should NOT)
                ctrl_lands += int(pc == val_id[wj])        # did the injection actually steer to w'? (sanity: yes)
        nz = len(wrong_idx)
        res["gold_flip"][al] = gold_hits / nz
        res["ctrl_flip"][al] = ctrl_help / nz
        res["ctrl_to_ctrl"][al] = ctrl_lands / nz
        print(f"  alpha={al:5.1f}  REPAIR flips-to-correct={gold_hits/nz:.3f}   "
              f"wrong-content flips-to-correct={ctrl_help/nz:.3f} (control, want ~0)   "
              f"wrong-content lands-on-its-target={ctrl_lands/nz:.3f} (sanity, want high)")
    handle.remove()

    a0 = alphas[0]
    best = max(alphas, key=lambda x: res["gold_flip"][x])
    print(f"\nRESULT")
    print(f"  baseline (alpha=0) flip-to-correct = {res['gold_flip'][a0]:.3f}  (should be ~0: these are the model's errors)")
    print(f"  REPAIR   (alpha={best}) recovers    = {res['gold_flip'][best]:.3f}  of the model's OWN errors, at the query site")
    print(f"  wrong-content control                = {res['ctrl_flip'][best]:.3f}  (re-injecting a DIFFERENT value does not fix it)")
    if res["gold_flip"][best] > 0.15 and res["gold_flip"][best] > 3 * res["ctrl_flip"][best]:
        print("  => CAPABILITY RECOVERY on real code: the correct value was present and re-injecting it fixes the output,")
        print("     while re-injecting a wrong value does not. P1's repair generalizes off the synthetic task.")
    else:
        print("  => repair weak or non-specific. Honest.")
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
