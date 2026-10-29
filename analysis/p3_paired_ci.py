"""Paired differences between the hidden-state probe, the model's answer and a decoder on its own logits.

For one model, condition, load and saved layer, every prompt has three answers: the model's top logit
over the value pool (output), the out-of-fold probe at that layer (probe), and the out-of-fold
multinomial regression on the model's value logits (logit decoder), all from p3_probe_capacity.py
--save-oof.  The differences probe - output and probe - logit decoder are paired by prompt; the 95%
interval is a percentile bootstrap over prompts (10,000 resamples, fixed RNG).  Per-seed values are
reported next to it.

Usage: python p3_paired_ci.py --cap-root <dir with <label>/main.npz> --oof-dir <dir> --labels a,b
                              --layer-index 2 --K 16 --conds D0,D256 --out <json>
       python p3_paired_ci.py --selftest
PEAK MEMORY: one condition of one model, 8,640 prompts x 67 logits (2.3 MB).
"""
import argparse
import json
import os

import numpy as np


def export_release(root, destination):
    """Copy referenced summaries and compact recorded predictions, without fitting.

    Peak memory: one condition's candidate logits (8640 x 71 x 4 bytes),
    plus integer trial arrays and predictions; below 32 MB. Hidden states
    are never loaded. Full original arrays remain in the campaign store.
    """
    from pathlib import Path
    import hashlib
    import shutil

    root, destination = Path(root), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Release destination must be empty; choose a new directory. Existing files are not removed.")
    destination.mkdir(parents=True, exist_ok=True)
    exported = set()
    summary = json.loads((root / "paper_revision/p3_measured_revision.json").read_text())
    manifest = {"schema": 1, "files": {}, "predictions": [],
                "scope": "Recorded integer predictions and trial IDs; no hidden states or refitted classifiers.",
                "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for rel, digest in summary["input_sha256"].items():
        raw = (root / rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, rel
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        exported.add(rel)
    shutil.copy2(root / "PREREG_p3_heldout_probe.md", destination / "PREREG_p3_heldout_probe.md")
    exported.add("PREREG_p3_heldout_probe.md")
    heldout = {v["label"] for v in summary["heldout"].values()}
    for cohort in ("capacity_fp32", "frame_test", "frame3", "ckpt_traj"):
        for source in sorted((root / cohort).glob("*/*.json")):
            rec = json.loads(source.read_text())
            if "cells" not in rec:
                continue
            lab, revision = rec["label"], rec["revision"]
            relative = source.relative_to(root).as_posix()
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(source.read_bytes())
            exported.add(relative)
            oof = root / "probe_cap" / ({"frame_test": "oof_frame", "ckpt_traj": "oof_ckpt"}.get(
                cohort, "oof_heldout" if lab in heldout else "oof13"))
            arrays = {}
            with np.load(source.with_suffix(".npz"), allow_pickle=False) as raw:
                for cond in rec["conds"]:
                    for field in ("gold", "K", "seed", "idx", "qi", "vals", "ents"):
                        key = f"{cond}_{field}"
                        if key in raw:
                            arrays[key] = raw[key]
                    arrays[f"{cond}_native"] = raw[f"{cond}_logits"].argmax(1).astype(np.int16)
                    if cohort == "frame3":
                        continue
                    for tag in [*(f"L{l}" for l in rec["layers_saved"]), "logits", "centered"]:
                        prediction = oof / f"{lab}_{revision}_{cond}_{tag}.npz"
                        if prediction.exists():
                            with np.load(prediction, allow_pickle=False) as saved:
                                assert len(saved["top"]) == len(arrays[f"{cond}_gold"]), prediction
                                arrays[f"{cond}_{tag}"] = saved["top"]
            output = destination / "predictions" / cohort / lab / f"{revision}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, **arrays)
            exported.add(output.relative_to(destination).as_posix())
            manifest["predictions"].append({"metadata": relative,
                "file": output.relative_to(destination).as_posix(),
                "source_array": source.with_suffix(".npz").relative_to(root).as_posix()})
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    assert actual == exported, "Unexpected files appeared in the release destination"
    for relative in sorted(exported):
        manifest["files"][relative] = hashlib.sha256((destination / relative).read_bytes()).hexdigest()
    (destination / "release_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("exported", len(manifest["files"]), "files;", len(manifest["predictions"]), "compact prediction records")


def paired(a, b, n_boot=10000, seed=0):
    """Mean of a - b (boolean arrays, one entry per prompt) and its 95% percentile bootstrap interval."""
    d = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    bs = d[idx].mean(1)
    return dict(mean=round(float(d.mean()), 4), lo=round(float(np.percentile(bs, 2.5)), 4),
                hi=round(float(np.percentile(bs, 97.5)), 4))


def selftest():
    rng = np.random.default_rng(1)
    base = rng.random(576) < 0.3
    same = paired(rng.random(576) < 0.3, base)        # independent, same rate: no effect
    flip = base | (rng.random(576) < 0.25)          # about +0.17
    up = paired(flip, base)
    ok = same["lo"] <= 0 <= same["hi"] and up["lo"] > 0
    print(f"[selftest] same-rate arrays {same}; +{flip.mean() - base.mean():.3f} arrays {up}: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-root")
    ap.add_argument("--oof-dir")
    ap.add_argument("--labels")
    ap.add_argument("--layer-index", type=int, default=2)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--conds", default="D0,D256")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--export-release", help="export compact release records from this campaign root")
    ap.add_argument("--export-to", help="destination of --export-release")
    a = ap.parse_args()
    if a.export_release:
        if not a.export_to:
            ap.error("--export-release requires --export-to")
        export_release(a.export_release, a.export_to)
        return
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    out = {}
    for lab in a.labels.split(","):
        rec = json.load(open(os.path.join(a.cap_root, lab, "main.json")))
        layer = rec["layers_saved"][a.layer_index]
        with np.load(os.path.join(a.cap_root, lab, "main.npz")) as Z:
            for c in a.conds.split(","):
                gold, K, seed = Z[f"{c}_gold"].astype(int), Z[f"{c}_K"], Z[f"{c}_seed"]
                m = K == a.K
                outp = Z[f"{c}_logits"].argmax(1) == gold
                pr = np.load(os.path.join(a.oof_dir, f"{lab}_main_{c}_L{layer}.npz"))["top"] == gold
                lg = np.load(os.path.join(a.oof_dir, f"{lab}_main_{c}_logits.npz"))["top"] == gold
                out[f"{lab}|{c}"] = dict(
                    label=lab, cond=c, K=a.K, layer=int(layer), n=int(m.sum()),
                    output=round(float(outp[m].mean()), 4), probe=round(float(pr[m].mean()), 4),
                    logit_decoder=round(float(lg[m].mean()), 4),
                    probe_minus_output=paired(pr[m], outp[m]), probe_minus_logit=paired(pr[m], lg[m]),
                    per_seed={int(s): dict(output=round(float(outp[m & (seed == s)].mean()), 4),
                                           probe=round(float(pr[m & (seed == s)].mean()), 4),
                                           logit_decoder=round(float(lg[m & (seed == s)].mean()), 4))
                              for s in sorted(set(seed[m].tolist()))})
                r = out[f"{lab}|{c}"]
                print(f"{lab:14s} {c:5s} K={a.K} L{layer}: out {r['output']:.3f} logit-dec "
                      f"{r['logit_decoder']:.3f} probe {r['probe']:.3f} | probe-out {r['probe_minus_output']} "
                      f"| probe-logit {r['probe_minus_logit']}", flush=True)
    json.dump(dict(results=out, _provenance="analysis/p3_paired_ci.py"), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
