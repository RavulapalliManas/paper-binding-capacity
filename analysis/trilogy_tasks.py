"""Abstract trials, deterministic folds, and the four state-operation generators.

WHY THIS EXISTS.  The three papers currently measure ONE state operation: a declaration is made
and later recalled.  "The model represents the binding but does not use it" is a claim about
state, and a single operation cannot tell state from static lookup.  This module adds the three
operations that separate them (declare / overwrite / scope) plus a computed-state task whose
answer token is randomised per trial, and it does so on a shared trial identity so that the same
abstract trial can be run against different tokenizers, different models, and different
checkpoints and still be THE SAME TRIAL.

TWO DESIGN RULES, both of which exist because the alternative silently breaks an inference:

1. ABSTRACT FIRST, TOKENS LAST.  A trial is generated over ENTITY_k / VALUE_k ids and only then
   resolved to a model's single-token strings.  Resolving first would make a cross-tokenizer
   comparison a comparison of different trials, and the resulting "model A beats model B" would
   be partly a statement about token pools.

2. THE FOLD IS A HASH OF THE TRIAL, NOT A SHUFFLE.  split(trial_id) is a pure function, so
   probe-training / selection / test membership is identical across every run, model, and
   checkpoint forever.  A fresh shuffle per run leaks the test set across checkpoints of the same
   trajectory, which is exactly where the developmental claims live.

THE RECENCY TRAP (mutable state).  If the queried entity is usually the most recently updated
one, "answer the last value you saw" scores well without any state tracking.  build_mutable
therefore samples the query uniformly over update recency and records `query_recency` on every
trial so the analysis can condition on it.  A mutable-state result reported without that column
is not interpretable.

THE PRESENCE TRAP (computed state).  If the answer token is a fixed word, a model can score above
chance by preferring tokens that appear in the prompt, or by parametric knowledge of the program.
build_codebook resamples the state->word map every trial and includes every codeword exactly once,
so token presence gives exactly 1/K and parametric knowledge gives nothing.

Self-check:  python trilogy_tasks.py
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, asdict

FOLDS = ("calibration", "probe", "selection", "test")
FOLD_WEIGHTS = (0.10, 0.30, 0.30, 0.30)


@dataclass
class Trial:
    """One abstract trial.  `body`/`query` are token-free templates over ENTITY_k / VALUE_k."""
    task_family: str
    body: list                      # list of (op, entity_id, value_id) or scope markers
    query: str                      # the entity id being asked about
    gold: str                       # the value id that is correct
    K: int
    D: int
    meta: dict = field(default_factory=dict)

    @property
    def trial_id(self) -> str:
        payload = json.dumps({"f": self.task_family, "b": self.body, "q": self.query,
                              "g": self.gold, "K": self.K, "D": self.D,
                              "m": {k: v for k, v in sorted(self.meta.items())
                                    if k not in ("fold",)}},
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def fold(self) -> str:
        """Deterministic split.  Uses a different hash prefix than trial_id so that trials which
        are adjacent in id space are not adjacent in fold space."""
        h = hashlib.sha256(("fold:" + self.trial_id).encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2 ** 64
        acc = 0.0
        for name, w in zip(FOLDS, FOLD_WEIGHTS):
            acc += w
            if u < acc:
                return name
        return FOLDS[-1]


def _rng(seed, *parts):
    """A generator whose stream depends on every part, so two cells never share a stream."""
    h = hashlib.sha256(("|".join(str(p) for p in (seed,) + parts)).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def build_declarative(n, K=8, D=256, seed=0):
    """A: each entity declared once, one queried.  The task the three papers already use."""
    out = []
    for i in range(n):
        r = _rng(seed, "decl", K, D, i)
        ents = [f"ENTITY_{j:02d}" for j in r.sample(range(64), K)]
        vals = [f"VALUE_{j:02d}" for j in r.sample(range(64), K)]
        qi = r.randrange(K)
        body = [("set", e, v) for e, v in zip(ents, vals)]
        out.append(Trial("declarative", body, ents[qi], vals[qi], K, D,
                         {"query_slot": qi, "seed": seed}))
    return out


def build_mutable(n, K=8, D=256, updates=2, seed=0):
    """B: every entity is assigned `updates` times; the answer is the FINAL value.

    Every entity receives the same number of updates, so "how many times was it mentioned" is
    uninformative.  The query is sampled uniformly over the recency of the entity's last update
    (query_recency = 0 means its final update was the last statement in the body), which is what
    stops a pure recency heuristic from scoring.
    """
    out = []
    for i in range(n):
        r = _rng(seed, "mut", K, D, updates, i)
        ents = [f"ENTITY_{j:02d}" for j in r.sample(range(64), K)]
        pool = [f"VALUE_{j:02d}" for j in r.sample(range(64), K * updates)]
        stmts = []
        assigned = {}
        for u in range(updates):
            order = ents[:]
            r.shuffle(order)
            for e in order:
                v = pool.pop()
                stmts.append(("set", e, v))
                assigned[e] = v
        # last-update position for each entity, then query uniformly over recency rank
        last_pos = {}
        for pos, (_, e, _) in enumerate(stmts):
            last_pos[e] = pos
        by_recency = sorted(ents, key=lambda e: -last_pos[e])   # 0 = most recent
        rank = r.randrange(K)
        q = by_recency[rank]
        out.append(Trial("mutable", stmts, q, assigned[q], K, D,
                         {"updates": updates, "query_recency": rank, "seed": seed}))
    return out


def build_scoped(n, K=6, D=256, depth=2, seed=0):
    """C: an entity is rebound inside a block; the query asks for its value OUTSIDE the block.

    The correct answer is the value that was live before the block opened, so a model that simply
    reports the most recent assignment is wrong by construction on every in-scope rebinding.
    """
    out = []
    for i in range(n):
        r = _rng(seed, "scope", K, D, depth, i)
        ents = [f"ENTITY_{j:02d}" for j in r.sample(range(64), K)]
        vals = [f"VALUE_{j:02d}" for j in r.sample(range(64), K + depth)]
        outer = dict(zip(ents, vals[:K]))
        stmts = [("set", e, v) for e, v in outer.items()]
        shadowed = ents[r.randrange(K)]
        for d in range(depth):
            stmts.append(("begin", None, None))
            stmts.append(("set", shadowed, vals[K + d]))
        for _ in range(depth):
            stmts.append(("end", None, None))
        query_inside = r.random() < 0.5
        if query_inside:                       # asked inside the innermost block
            stmts.insert(-depth, ("mark_query", None, None))
            gold = vals[K + depth - 1]
        else:
            gold = outer[shadowed]             # the block is closed; the outer value is live
        out.append(Trial("scoped", stmts, shadowed, gold, K, D,
                         {"depth": depth, "query_inside": query_inside, "seed": seed}))
    return out


def build_codebook(n, K=6, n_ops=3, modulus=6, seed=0):
    """D: compute a latent state, then read it out through a codebook resampled every trial.

    The program is a chain of modular register updates.  Afterwards the prompt supplies a fresh
    state->codeword map containing every codeword exactly once.  Token presence is therefore
    exactly 1/modulus and carries no signal, and no amount of memorised program-to-answer
    association helps, because the answer word for a given program changes every trial.
    """
    out = []
    for i in range(n):
        r = _rng(seed, "code", K, n_ops, modulus, i)
        z = r.randrange(modulus)
        ops, trace = [], [z]
        for _ in range(n_ops):
            kind = r.choice(["add", "mul"])
            c = r.randrange(1, modulus)
            z = (z + c) % modulus if kind == "add" else (z * c + r.randrange(modulus)) % modulus
            ops.append((kind, c))
            trace.append(z)
        words = [f"VALUE_{j:02d}" for j in r.sample(range(64), modulus)]
        out.append(Trial("codebook", [("prog", None, json.dumps(ops)),
                                      ("codebook", None, json.dumps(words))],
                         "final_state", words[z], modulus, 0,
                         {"latent_state": z, "n_ops": n_ops, "modulus": modulus,
                          "trace": trace, "seed": seed}))
    return out


# --------------------------------------------------------------------------------------------
def _selfcheck():
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("trial identity + folds")
    a = build_declarative(500, seed=0)
    b = build_declarative(500, seed=0)
    chk([t.trial_id for t in a] == [t.trial_id for t in b], "regenerating with the same seed gives the same ids")
    chk(len({t.trial_id for t in a}) == 500, "no duplicate trial ids in 500 trials")
    from collections import Counter
    c = Counter(t.fold for t in build_declarative(20000, seed=1))
    frac = {k: round(v / 20000, 3) for k, v in c.items()}
    chk(all(abs(frac.get(f, 0) - w) < 0.02 for f, w in zip(FOLDS, FOLD_WEIGHTS)),
        f"fold proportions within 2pp of target: {frac}")
    a2 = build_declarative(500, seed=0)
    chk(all(x.fold == y.fold for x, y in zip(a, a2)), "fold is a pure function of the trial")

    print("mutable state -- the recency trap")
    m = build_mutable(4000, K=8, seed=0)
    rc = Counter(t.meta["query_recency"] for t in m)
    chk(max(rc.values()) - min(rc.values()) < 0.25 * (4000 / 8),
        f"query recency near-uniform over 8 ranks: {sorted(rc.items())}")
    t0 = m[0]
    counts = Counter(e for _, e, _ in t0.body)
    chk(len(set(counts.values())) == 1, f"every entity updated equally often: {set(counts.values())}")
    lastval = {}
    for _, e, v in t0.body:
        lastval[e] = v
    chk(lastval[t0.query] == t0.gold, "gold is the final value of the queried entity")

    print("scoped state")
    s = build_scoped(400, seed=0)
    outside = [t for t in s if not t.meta["query_inside"]]
    chk(len(outside) > 100, f"both query positions represented ({len(outside)}/400 outside)")
    t1 = outside[0]
    inner = [v for op, e, v in t1.body if op == "set" and e == t1.query]
    chk(t1.gold == inner[0] and t1.gold != inner[-1],
        "outside-scope gold is the OUTER value, not the most recent assignment")

    print("codebook computed state -- the presence trap")
    cb = build_codebook(2000, seed=0)
    chk(len({t.gold for t in cb}) > 1, "gold word varies across trials")
    words0 = json.loads(cb[0].body[1][2])
    chk(len(set(words0)) == len(words0), "each codeword appears exactly once in the codebook")
    chk(all(t.gold in json.loads(t.body[1][2]) for t in cb), "gold is always present in the codebook")
    # the decisive one: the same program must not always give the same word
    from collections import defaultdict
    byprog = defaultdict(set)
    for t in cb:
        byprog[t.body[0][2]].add(t.gold)
    repeated = {k: v for k, v in byprog.items() if len(v) > 1}
    chk(len(repeated) > 0, f"the same program maps to different words across trials ({len(repeated)} programs)")
    zc = Counter(t.meta["latent_state"] for t in cb)
    chk(max(zc.values()) / min(zc.values()) < 1.6, f"latent states roughly balanced: {sorted(zc.items())}")

    print("\nSELF-CHECK " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
