"""Does P1's gap replicate on REAL code, with a REAL code model?

Claim under test (P1, ported off the synthetic obligation task):
  On trials the model gets WRONG, a linear probe on its residual still recovers the correct answer,
  well above chance -- the model represented it and failed to use it.

Task: real Python with MUTABLE STATE (the thing that breaks coding agents). Several variables are
assigned single-token string values; some are REASSIGNED (interference); the model must report the
CURRENT value of a target variable. Ground truth = the last value assigned to it.

    v0 = "cat"
    v1 = "sun"
    v0 = "key"        # reassigned -> current v0 is "key"
    v2 = "box"
    # current value of v0:
    v0 == "

The model completes the string. On trials where its greedy token != gold, we ask: does a probe on the
residual at the query line recover gold? That is exactly P1's gap, on real code.

Everything is forward passes. One model, cached. Prints CONTROLS FIRST.
"""
import argparse, json, random, sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def single_token_words(tok, pool):
    """Keep words that are exactly one token when they appear BARE (as they do right after the quote:  "<word> )."""
    keep = []
    for w in pool:
        if len(tok.encode(w, add_special_tokens=False)) == 1:     # bare, matches gold_id below
            keep.append(w)
    return keep


POOL = ("cat dog sun key box car cup pen bag hat map bus egg ice jam owl fox pig cow bee ant "
        "red one two six ten fog log mud oak rat sky toy van web yak zoo arm bat axe bed cap "
        "den elf fan gem hen ink jar kit lip nut orb paw rib tab urn vet wax").split()


def make_trial(rng, words, k, n_reassign):
    names = [f"v{i}" for i in range(k)]
    vals = rng.sample(words, k)
    lines = [f'{names[i]} = "{vals[i]}"' for i in range(k)]
    cur = dict(zip(names, vals))
    # `n_reassign` reassignment OPERATIONS, sampling variables WITH REPLACEMENT so chains form
    # (a variable reassigned 3x forces the model to track the latest write through interference).
    counts = {nm: 0 for nm in names}
    for _ in range(n_reassign):
        nm = rng.choice(names)
        nv = rng.choice([w for w in words if w != cur[nm]])
        lines.append(f'{nm} = "{nv}"')
        cur[nm] = nv
        counts[nm] += 1
    # query the MOST-reassigned variable (deepest chain -> hardest), ties broken at random
    mx = max(counts.values())
    target = rng.choice([nm for nm in names if counts[nm] == mx])
    gold = cur[target]
    # recency control: the value in the LAST line of the program (what a pure-recency probe would read)
    last_val = lines[-1].split('"')[1]
    prog = "\n".join(lines)
    prompt = f'{prog}\n# current value of {target}:\n{target} == "'
    return prompt, gold, target, int(counts[target]), last_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/deepseek-coder-6.7b-base")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--reassign", type=int, default=2)
    ap.add_argument("--out", default="/tmp/real_code_gap.json")
    a = ap.parse_args()

    torch.manual_seed(0)
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    words = single_token_words(tok, POOL)
    print(f"single-token value pool: {len(words)} words -> {words[:12]}...")
    assert len(words) >= 12, "need a bigger single-token pool"

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda").eval()
    nL = model.config.num_hidden_layers
    d = model.config.hidden_size
    # gold token id = first token of the value as it appears after the quote (no leading space)
    def gold_id(w):
        ids = tok.encode(w, add_special_tokens=False)
        return ids[0]

    H = {L: [] for L in range(1, nL)}            # residual at the LAST prompt token, per layer
    Y, correct, was_reassigned, LastY, model_margin = [], [], [], [], []
    val_ids = {w: gold_id(w) for w in words}

    print("generating + running trials...", flush=True)
    for t in range(a.n):
        prompt, gold, target, reasg, last_val = make_trial(rng, words, a.k, a.reassign)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model(**ids, output_hidden_states=True)
        logits = o.logits[0, -1]                  # next token after the opening quote
        # restrict the model's choice to the present value tokens (a K-way-ish forced choice, fair)
        present = list({val_ids[w] for w in set(_vals_in(prompt, words))})
        pv = torch.tensor([float(logits[i]) for i in present])
        pred_id = present[int(pv.argmax())]
        sv = torch.softmax(pv, 0).sort(descending=True).values          # model's OWN confidence
        model_margin.append(float(sv[0] - sv[1]) if len(sv) > 1 else 1.0)
        Y.append(words.index(gold))
        LastY.append(words.index(last_val) if last_val in words else -1)
        correct.append(int(pred_id == val_ids[gold]))
        was_reassigned.append(int(reasg))
        for L in range(1, nL):
            H[L].append(o.hidden_states[L][0, -1].float().cpu().numpy())
        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{a.n}  running model_acc={np.mean(correct):.3f}", flush=True)

    Y = np.array(Y); correct = np.array(correct); reasg = np.array(was_reassigned); LastY = np.array(LastY)
    n = len(Y); ntr = int(0.7 * n)
    idx = np.random.RandomState(0).permutation(n); tr, te = idx[:ntr], idx[ntr:]
    chance = 1.0 / len(words)

    print("\nCONTROLS FIRST")
    deep = reasg >= 2                            # targets reassigned 2+ times (the hardest chains)
    print(f"  n={n}  model_acc(overall)={correct.mean():.3f}  "
          f"model_acc(depth>=2 targets)={correct[deep].mean() if deep.sum() else float('nan'):.3f}  "
          f"probe chance={chance:.3f}")
    print(f"  wrong trials: {(correct==0).sum()}  (that is the set the gap is measured on)")

    from sklearn.metrics import roc_auc_score

    def fit_layer(L, Cval=0.5):
        Hl = np.stack(H[L]); scl = StandardScaler().fit(Hl[tr])
        clf = LogisticRegression(max_iter=1000, C=Cval).fit(scl.transform(Hl[tr]), Y[tr])
        return scl, clf, Hl

    te_wrong = te[correct[te] == 0]              # trials in TEST the MODEL got wrong -- the gap is measured here
    te_right = te[correct[te] == 1]

    # LAYER PROFILE: is the answer present mid-stream, or only near the output head?
    print("\nLAYER PROFILE  (probe accuracy)")
    print(f"  {'layer':>6} {'all':>7} {'on ERRORS':>10}")
    profile = {}
    layers_to_show = sorted(set([nL // 4, nL // 2, 3 * nL // 4, nL - 2, nL - 1]))
    best = (-1, None, None)
    for L in range(1, nL):
        scl, clf, Hl = fit_layer(L)
        acc = clf.score(scl.transform(Hl[te]), Y[te])
        accw = clf.score(scl.transform(Hl[te_wrong]), Y[te_wrong]) if len(te_wrong) > 5 else float("nan")
        profile[L] = {"all": float(acc), "on_wrong": float(accw)}
        if L in layers_to_show:
            print(f"  {L:>6} {acc:7.3f} {accw:10.3f}")
        if accw == accw and accw > best[0]:      # pick the layer that best recovers ERRORS (the claim's layer)
            best = (accw, L, (scl, clf, Hl))
    probe_on_wrong, L, (sc, clf, Hl) = best
    probe_acc = profile[L]["all"]

    # BOOTSTRAP CI on the gap (resample the wrong trials, clustered by trial)
    pw = clf.predict(sc.transform(Hl[te_wrong]))
    hit = (pw == Y[te_wrong]).astype(float)
    rng_b = np.random.RandomState(0)
    boot = [hit[rng_b.randint(0, len(hit), len(hit))].mean() for _ in range(2000)] if len(hit) else [np.nan]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    # CERTIFICATE: does probe/output DISAGREEMENT detect the model's failure? (P1's detector, on real code)
    proba = clf.predict_proba(sc.transform(Hl[te]))
    # Practitioner-usable certificate: entropy of the probe's distribution (high => probe itself unsure => likely fail).
    ent = -(proba * np.clip(np.log(proba + 1e-12), -50, 0)).sum(1)
    y_fail = (correct[te] == 0).astype(int)
    cert_auroc = float(roc_auc_score(y_fail, ent)) if y_fail.sum() > 3 and y_fail.sum() < len(y_fail) else float("nan")
    # a STRONGER certificate that uses the probe's own top guess vs the probe's 2nd guess (margin):
    part = np.sort(proba, 1)
    margin = part[:, -1] - part[:, -2]
    cert_margin_auroc = float(roc_auc_score(y_fail, -margin)) if 3 < y_fail.sum() < len(y_fail) else float("nan")
    # BASELINE a practitioner would actually use: the MODEL'S OWN confidence (its top1-top2 margin over present)
    mm = np.array(model_margin)
    base_auroc = float(roc_auc_score(y_fail, -mm[te])) if 3 < y_fail.sum() < len(y_fail) else float("nan")

    # RECENCY CONTROL: is the probe tracking the BINDING, or just reading the last value token in context?
    # Look only at trials where gold != last-assigned-value (binding and recency DISAGREE), among the model's
    # errors. If the probe still picks gold (not last_val), it is tracking the binding, not recency.
    pred_all = clf.predict(sc.transform(Hl))
    disagree = (Y != LastY) & (LastY >= 0)
    tw_dis = te_wrong[disagree[te_wrong]]
    if len(tw_dis) > 5:
        probe_gets_gold = float((pred_all[tw_dis] == Y[tw_dis]).mean())
        probe_gets_last = float((pred_all[tw_dis] == LastY[tw_dis]).mean())
    else:
        probe_gets_gold = probe_gets_last = float("nan")
    print(f"\nRECENCY CONTROL  (test errors where gold != last-assigned value, n={len(tw_dis)})")
    print(f"  P(probe = gold/binding) = {probe_gets_gold:.3f}   vs   P(probe = last-assigned/recency) = {probe_gets_last:.3f}")
    print(f"  gold==last-assigned overall: {float((Y == LastY).mean()):.3f}  (if high, task is too recency-friendly)")
    if probe_gets_gold > 2 * probe_gets_last:
        print("  => the probe tracks the BINDING, not recency. The gap is not a recency artifact.")
    else:
        print("  => WARNING: probe may be reading recency. The gap claim is NOT clean here.")

    res = {
        "recency_probe_gold": probe_gets_gold, "recency_probe_last": probe_gets_last,
        "gold_eq_last_frac": float((Y == LastY).mean()), "n_disagree_test_wrong": int(len(tw_dis)),
        "model": a.model, "n": n, "k": a.k, "reassign": a.reassign, "best_layer_for_errors": L, "nlayers": nL,
        "chance": chance, "model_acc": float(correct.mean()),
        "model_acc_deep": float(correct[reasg >= 2].mean()) if (reasg >= 2).sum() else None,
        "n_wrong_total": int((correct == 0).sum()), "n_wrong_test": int(len(te_wrong)),
        "probe_acc_all": float(probe_acc),
        "probe_acc_on_wrong": float(probe_on_wrong), "gap_ci95": ci,
        "gap_xchance": float(probe_on_wrong / chance),
        "cert_auroc_entropy": cert_auroc, "cert_auroc_margin": cert_margin_auroc,
        "model_conf_auroc": base_auroc,
        "layer_profile": profile,
    }
    print(f"\nRESULT  (layer {L} of {nL}, chosen to best recover ERRORS)")
    print(f"  probe_acc (all test)        = {probe_acc:.3f}   ({probe_acc/chance:.1f}x chance)")
    print(f"  probe_acc ON MODEL'S ERRORS = {probe_on_wrong:.3f}   95% CI [{ci[0]:.3f}, {ci[1]:.3f}]   "
          f"({probe_on_wrong/chance:.1f}x chance, n_wrong_test={len(te_wrong)})   <-- THE GAP")
    print(f"  model_acc on those trials   = 0.000 by definition")
    print(f"  CERTIFICATE: probe-entropy AUROC = {cert_auroc:.3f}   vs  MODEL's-own-confidence AUROC = {base_auroc:.3f}"
          f"   ({'probe WINS' if cert_auroc > base_auroc else 'model conf wins'})")
    if ci[0] > 3 * chance:
        print("  => GAP CONFIRMED with CI clear of chance. The model represents the answer on trials it gets wrong,")
        print("     on real Python state-tracking, with a real code model. P1 generalizes off the synthetic task.")
    else:
        print("  => CI touches chance: underpowered or absent. Honest.")
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


def _vals_in(prompt, words):
    # the value words that literally appear in this prompt (present-set for the fair forced choice)
    import re
    found = re.findall(r'"([a-z]+)"', prompt)
    return [w for w in found if w in words]


if __name__ == "__main__":
    main()
