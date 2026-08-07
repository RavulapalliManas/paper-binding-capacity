"""E1 (killer experiment): watch the reads-vs-uses gap OPEN over pretraining.
For each Pythia checkpoint (HF revision step{N}), on the binding battery measure:
  - probe_acc : linear-probe decodability of the bound obligation from hidden state (the WRITE)
  - behav_acc : model's own output accuracy on the recall query (the USE)
  - margin    : CONTINUOUS behavioral metric = logit(gold) - logit(best distractor), mean
                (the Schaeffer 2304.15004 metric-artifact defense)
C2 prediction: probe_acc rises BEFORE behav_acc -> the gap (probe_acc - behav_acc) OPENS mid-training.
Reloads the SAME architecture at each revision (weights only) so probe/geometry are comparable across time.
"""
import argparse, json, os, random, shutil
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))
from transformers import AutoModelForCausalLM
from probe_battery_v2 import Battery2, Cfg
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class Dev(Battery2):
    def trials(self, rng, K, D):
        recs = []
        for _ in range(self.cfg.n_trials):
            ents = rng.sample(self.ent, K); obls = rng.sample(self.obl, K)
            decl = ". ".join(f"{e[0].strip()}: {o[0].strip()}" for e, o in zip(ents, obls)) + ". "
            j = rng.randrange(K)
            seq = (self.tok(decl, add_special_tokens=False)["input_ids"]
                   + self._distract("random", D, rng, self.model.config.vocab_size)
                   + self.tok(f"The task for {ents[j][0].strip()} is:", add_special_tokens=False)["input_ids"])
            recs.append(dict(seq=seq, gold=obls[j][1]))
        return recs
    def run_layer(self, recs, L):
        H, LG = [], []
        for b in range(0, len(recs), self.cfg.batch):
            chunk = [r["seq"] for r in recs[b:b+self.cfg.batch]]
            m = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), m), self.tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros((len(chunk), m), dtype=torch.long)
            for i, s in enumerate(chunk): ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
            ids = ids.to(self.model.device); att = att.to(self.model.device)
            with torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=att, output_hidden_states=True)
            last = att.sum(1) - 1
            for i in range(len(chunk)):
                H.append(out.hidden_states[L][i, last[i]].float().cpu().numpy())
                LG.append(out.logits[i, last[i]].float().cpu().numpy())
        return np.array(H), np.array(LG)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--steps", required=True)      # comma list of ints
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--n-trials", type=int, default=400); ap.add_argument("--layer-frac", type=float, default=0.6)
    a = ap.parse_args()
    cfg = Cfg(model=a.base, out=a.out, seeds=1, n_trials=a.n_trials, batch=16, K=a.K)
    bat = Dev(cfg)                                   # loads base (final) once for tok/ent/obl
    nL = bat.model.config.num_hidden_layers; L = max(1, int(a.layer_frac * nL))
    oi = np.array(bat.obl_ids); os.makedirs(a.out, exist_ok=True)
    rng0 = random.Random(999); recs = bat.trials(rng0, a.K, a.D)   # SAME trials at every checkpoint
    gold = np.array([r["gold"] for r in recs]); ntr = int(0.7 * len(recs))
    rows = []
    seen_fps = []                                     # FABRICATION GUARD: no two checkpoints may share weights
    cache_dir = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                             "models--" + a.base.replace("/", "--"))
    for step in [int(x) for x in a.steps.split(",")]:
        try:
            del bat.model; torch.cuda.empty_cache()
            shutil.rmtree(cache_dir, ignore_errors=True)   # force a FRESH download of THIS revision (no stale cache)
            bat.model = AutoModelForCausalLM.from_pretrained(
                a.base, revision=f"step{step}", torch_dtype=torch.float16).to("cuda").eval()
        except Exception as e:
            print(f"[dev] step{step} LOAD FAIL {e}", flush=True); continue
        fp = float(bat.model.get_input_embeddings().weight[100, :20].sum().item())  # detect revision-ignored fallback
        if any(abs(fp - s) < 1e-6 for s in seen_fps):
            print(f"[dev] step{step} FABRICATION GUARD: fp={fp} duplicates a prior checkpoint -- the revision "
                  f"load returned identical weights (failed download / cache fallback). ROW DROPPED, not written.",
                  flush=True)
            continue
        seen_fps.append(fp)
        H, LG = bat.run_layer(recs, L)
        pred = oi[LG[:, oi].argmax(1)]
        gy = gold[ntr:]                                   # test-set golds
        behav_te = (pred[ntr:] == gy)
        behav = float(behav_te.mean())
        # continuous behavioral margin (metric-artifact defense): logit(gold) - best distractor, test set
        gi = np.array([np.where(oi == g)[0][0] for g in gy])
        lgo = LG[ntr:, oi]; goldl = lgo[np.arange(len(gy)), gi]
        tmp = lgo.copy(); tmp[np.arange(len(gy)), gi] = -1e9
        margin = float(np.mean(goldl - tmp.max(1)))
        sc = StandardScaler().fit(H[:ntr]); clf = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(H[:ntr]), gold[:ntr])
        ppred = clf.predict(sc.transform(H[ntr:]))
        probe = float((ppred == gy).mean())
        wrong = ~behav_te                                 # the reads-but-doesn't-use metric:
        pacc_wrong = float((ppred[wrong] == gy[wrong]).mean()) if wrong.sum() > 5 else float("nan")
        r = dict(step=step, probe_acc=probe, behav_acc=behav, pacc_wrong=pacc_wrong,
                 margin=margin, n_wrong=int(wrong.sum()), n=len(gy), fp=fp)
        rows.append(r); print("[dev]", json.dumps(r), flush=True)
        json.dump(rows, open(os.path.join(a.out, "dev.json"), "w"), indent=2)
    if rows:
        pw = [r["pacc_wrong"] for r in rows if r["pacc_wrong"] == r["pacc_wrong"]]
        print("\n[SUMMARY] steps=%d probe=[%.2f,%.2f] behav=[%.2f,%.2f] pAcc|wrong=[%.2f,%.2f]"
              % (len(rows), min(r["probe_acc"] for r in rows), max(r["probe_acc"] for r in rows),
                 min(r["behav_acc"] for r in rows), max(r["behav_acc"] for r in rows),
                 (min(pw) if pw else float("nan")), (max(pw) if pw else float("nan"))), flush=True)
    else:
        print("[SUMMARY] no rows", flush=True)
if __name__ == "__main__": raise SystemExit(main())
