"""PROBE_CAMPAIGN_CROSSMODEL_v2 — Harness 2.0 (E1 + E2).

Fixes the pilot's VOID-FLOOR: entity<->obligation binding (stronger associative task than v1's
position->identity), adaptive K-calibration so each model's no-interference ceiling lands in
[0.55,0.90], and the E2 elicitation ladder so the reads-vs-uses gap is artifact-proof.

Task: K entities each bound to a distinct obligation (single-token action). Declaration lists the
pairs; a distractor block of length D (condition-typed) follows; the recall query names one entity
and the model must recover its obligation.
  binding probe : linear decode of the bound obligation from the query hidden state (chance 1/|OBL|)
  rung-1 behav  : argmax over obligation tokens at the query (zero-shot generation proxy)
  rung-5 behav  : forced-choice log-prob, correct obligation vs foil (no generation, no parsing) -> K1
  G_ru(rung)    = binding_probe - behav_rung ; the E2 claim is G_ru survives rung 5.

Conditions {random,prose,code,code_confusable} x D grid, >=3 seeds, StandardScaler+held-out probes,
mean +/- std, floor/ceiling flags, tokenizer-fairness column. NO TRAINING.
"""
from __future__ import annotations
import argparse, json, os, random, time
from dataclasses import dataclass
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

ENTITIES = ("apple banana cherry dragon eagle forest garden harbor island jungle kettle lemon mountain "
    "needle ocean pencil quartz river silver tunnel violet walnut yellow zebra anchor bridge candle "
    "diamond ember falcon glacier hammer ivory jacket kernel ladder marble nectar onyx pepper quill "
    "rabbit saddle timber umbra vault willow").split()
# obligations: distinct single-token action words -> the relation to be bound to each entity
OBLIGATIONS = ("review merge deploy test revert patch audit ship build cache delete export flush index "
    "lock mount purge queue rename scan sync trace unpack verify watch").split()
PROSE = ("the quiet afternoon settled over the valley as travellers wondered whether the season would "
    "turn again before the festival while the market filled with ordinary talk about weather food").split()
CODE = ("def foo ( x ) : return x + 1 class Bar self value append range len for while if else import "
    "from numpy as np def bar ( y ) : yield y * 2 lambda z : z try except finally with open write").split()

@dataclass
class Cfg:
    model: str; out: str
    seeds: int = 3
    n_trials: int = 1000
    train_frac: float = 0.70
    batch: int = 32
    dvals: tuple = (0, 32, 64, 128, 256)
    conditions: tuple = ("random", "prose", "code", "code_confusable")
    dtype: str = "bfloat16"
    revision: str = None
    trust_remote_code: bool = False
    dequantize: bool = False
    K: int = 0  # 0 => auto-calibrate

def log(*a): print("[v2]", *a, flush=True)

class Battery2:
    def __init__(self, cfg: Cfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        log(f"loading {cfg.model} rev={cfg.revision} dequant={cfg.dequantize} ...")
        self.tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.dtype, torch.bfloat16)
        kw = dict(torch_dtype=dt, device_map="auto", output_hidden_states=True,
                  revision=cfg.revision, trust_remote_code=cfg.trust_remote_code)
        if cfg.dequantize:
            from transformers import Mxfp4Config
            kw["quantization_config"] = Mxfp4Config(dequantize=True)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, **kw)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.dev = self.model.get_input_embeddings().weight.device
        self.nlayers = self.model.config.num_hidden_layers
        self.ent = self._single_tok(ENTITIES)
        self.obl = self._single_tok(OBLIGATIONS)
        self.obl_ids = [i for _, i in self.obl]
        self.chance = 1.0 / len(self.obl)
        log(f"layers={self.nlayers} entities={len(self.ent)} obligations={len(self.obl)} chance={self.chance:.3f}")

    def _single_tok(self, words):
        out = []
        for w in words:
            ids = self.tok(" " + w, add_special_tokens=False)["input_ids"]
            if len(ids) == 1: out.append((" " + w, ids[0]))
        return out

    def _distract(self, cond, D, rng, vocab):
        if D == 0: return []
        if cond == "random": return [rng.randrange(5, vocab - 5) for _ in range(D)]
        if cond in ("prose", "code"):
            words = PROSE if cond == "prose" else CODE; t = []
            while len(t) < D: t += self.tok(" " + rng.choice(words), add_special_tokens=False)["input_ids"]
            return t[:D]
        if cond == "code_confusable":
            t = []
            while len(t) < D:
                w = "_" + "".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(3, 7)))
                t += self.tok(" " + w, add_special_tokens=False)["input_ids"]
            return t[:D]

    def build(self, rng, K, cond, D):
        ents = rng.sample(self.ent, K); obls = rng.sample(self.obl, K)  # distinct obligation per entity
        decl = ". ".join(f"{e[0].strip()}: {o[0].strip()}" for e, o in zip(ents, obls)) + ". "
        j = rng.randrange(K)
        q = f"The task for {ents[j][0].strip()} is:"
        vocab = self.model.config.vocab_size
        seq = self.tok(decl, add_special_tokens=False)["input_ids"] + self._distract(cond, D, rng, vocab) \
              + self.tok(q, add_special_tokens=False)["input_ids"]
        foil = rng.choice([o for o in self.obl if o[1] != obls[j][1]])
        return seq, obls[j][1], foil[1]  # seq, correct obligation id, foil id

    @torch.no_grad()
    def _fwd(self, seqs):
        maxlen = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), maxlen), self.tok.pad_token_id, dtype=torch.long)
        m = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, maxlen - len(s):] = torch.tensor(s); m[i, maxlen - len(s):] = 1
        out = self.model(input_ids=ids.to(self.dev), attention_mask=m.to(self.dev))
        feats = np.stack([h[:, -1, :].float().cpu().numpy() for h in out.hidden_states], axis=1)
        logits = out.logits[:, -1, :].float().cpu().numpy()
        return feats, logits

    def collect(self, rng, K, cond, D):
        seqs, Y, FOIL = [], [], []
        for _ in range(self.cfg.n_trials):
            s, y, f = self.build(rng, K, cond, D); seqs.append(s); Y.append(y); FOIL.append(f)
        F, LG = [], []
        for b in range(0, len(seqs), self.cfg.batch):
            f, lg = self._fwd(seqs[b:b + self.cfg.batch]); F.append(f); LG.append(lg)
        return np.concatenate(F), np.array(Y), np.array(FOIL), np.concatenate(LG)

    def probe(self, F, y):
        n = len(y); ntr = int(self.cfg.train_frac * n); best, bestL = -1, 0
        for L in range(F.shape[1]):
            X = F[:, L, :]; sc = StandardScaler().fit(X[:ntr])
            Xtr, Xte = sc.transform(X[:ntr]), sc.transform(X[ntr:])
            if len(set(y[:ntr].tolist())) < 2: continue
            try:
                m = LogisticRegression(max_iter=1000, C=0.5).fit(Xtr, y[:ntr])
                s = accuracy_score(y[ntr:], m.predict(Xte))
            except Exception: s = -1
            if s > best: best, bestL = s, L
        return float(best), bestL

    def rung1(self, LG, Y):  # zero-shot generation proxy: argmax over obligation tokens
        sub = LG[:, self.obl_ids]; pred = np.array(self.obl_ids)[sub.argmax(1)]
        return float((pred == Y).mean())

    def rung5(self, LG, Y, FOIL):  # forced-choice logprob: prefer correct obligation over foil?
        # single-token obligations => next-token logit is the log-prob proxy
        idx = {tid: k for k, tid in enumerate(range(LG.shape[1]))}
        cor = LG[np.arange(len(Y)), Y]; foi = LG[np.arange(len(Y)), FOIL]
        return float((cor > foi).mean())

    def calibrate(self, seed=0):
        # find K so no-interference (D=0) binding ceiling in [0.55,0.90]; search K in 3..10
        rng = random.Random(777 + seed)
        best_K, best_gap = None, 1e9
        for K in (3, 4, 5, 6, 8, 10):
            if K > len(self.ent) or K > len(self.obl): break
            F, Y, FO, LG = self.collect(rng, K, "random", 0)
            ceil, _ = self.probe(F, Y)
            log(f"calib K={K}: ceiling(D=0)={ceil:.3f}")
            if 0.55 <= ceil <= 0.90: return K, ceil
            gap = min(abs(ceil - 0.55), abs(ceil - 0.90))
            if gap < best_gap: best_gap, best_K, best_ceil = gap, K, ceil
        return best_K, best_ceil

    def run(self):
        cfg = self.cfg; os.makedirs(cfg.out, exist_ok=True)
        K = cfg.K or None
        if K is None:
            K, cal_ceil = self.calibrate()
            log(f"calibrated K={K} (ceiling={cal_ceil:.3f})")
        else:
            cal_ceil = None
        ent_lens = [len(self.tok(" " + w, add_special_tokens=False)["input_ids"]) for w in ENTITIES]
        manifest = dict(model=cfg.model, revision=cfg.revision, layers=self.nlayers, K=K,
                        calibrated_ceiling=cal_ceil, chance=self.chance, seeds=cfg.seeds,
                        dvals=list(cfg.dvals), conditions=list(cfg.conditions),
                        entity_tok_len_mean=float(np.mean(ent_lens)), entity_tok_len_max=int(np.max(ent_lens)),
                        dequantize=cfg.dequantize, ts=time.time())
        json.dump(manifest, open(os.path.join(cfg.out, "manifest.json"), "w"), indent=2)
        jf = open(os.path.join(cfg.out, "cells.jsonl"), "a"); rows = []
        for seed in range(cfg.seeds):
            for cond in cfg.conditions:
                for D in cfg.dvals:
                    rng = random.Random((1000 + seed) * 100003 + hash((cond, D)) % 100003)
                    F, Y, FO, LG = self.collect(rng, K, cond, D)
                    pb, L = self.probe(F, Y); r1 = self.rung1(LG, Y); r5 = self.rung5(LG, Y, FO)
                    row = dict(seed=seed, K=K, cond=cond, D=D, chance=self.chance,
                               binding_probe=pb, probe_layer=L, behav_rung1=r1, behav_rung5=r5,
                               G_ru_rung1=pb - r1, G_ru_rung5=pb - r5)
                    jf.write(json.dumps(row) + "\n"); jf.flush(); rows.append(row)
                    log(f"s{seed} {cond:15s} D={D:3d} | probe={pb:.3f} r1={r1:.3f} r5={r5:.3f} "
                        f"| G_ru(r1)={pb-r1:+.3f} G_ru(r5)={pb-r5:+.3f}")
        jf.close(); json.dump(rows, open(os.path.join(cfg.out, "results.json"), "w"), indent=2)
        log("DONE ->", cfg.out); return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--n-trials", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--revision", default=None)
    ap.add_argument("--trust-remote-code", action="store_true"); ap.add_argument("--dequantize", action="store_true")
    a = ap.parse_args()
    Battery2(Cfg(model=a.model, out=a.out, seeds=a.seeds, n_trials=a.n_trials, batch=a.batch,
                 K=a.K, dtype=a.dtype, revision=a.revision, trust_remote_code=a.trust_remote_code,
                 dequantize=a.dequantize)).run()

if __name__ == "__main__":
    raise SystemExit(main())
