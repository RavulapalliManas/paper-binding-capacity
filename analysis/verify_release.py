"""Verify the released paper against compact recorded predictions and generated assets.

No language model is loaded and no classifier is fit. A paired bootstrap uses
at most 10,000 x 576 x 8 bytes (46 MB); all other inputs are read one record
at a time. Run from any directory with --help for the release commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from pypdf import PdfReader

from p3_paired_ci import paired


def verify(root, generated):
    measurements = root / "results/measurement"
    manifest = json.loads((measurements / "release_manifest.json").read_text())
    assert manifest["exporter_sha256"] == hashlib.sha256((root / "analysis/p3_paired_ci.py").read_bytes()).hexdigest()
    for relative, expected in manifest["files"].items():
        assert hashlib.sha256((measurements / relative).read_bytes()).hexdigest() == expected, relative
    expected_pairs = {}
    for name in ["heldout_paired_K16.json", "p3_paired_ci_13_K16.json"]:
        expected_pairs.update(json.loads((measurements / "probe_cap" / name).read_text())["results"])
    n_cells = n_pairs = n_probe_rows = n_decoder_rows = 0
    for entry in manifest["predictions"]:
        metadata = json.loads((measurements / entry["metadata"]).read_text())
        lab = metadata["label"]
        with np.load(measurements / entry["file"], allow_pickle=False) as saved:
            for cell in metadata["cells"]:
                cond, k, seed = cell["cond"], cell["K"], cell["seed"]
                mask = (saved[f"{cond}_K"] == k) & (saved[f"{cond}_seed"] == seed)
                assert mask.sum() == cell["n"], (entry["file"], cell)
                accuracy = (saved[f"{cond}_native"][mask] == saved[f"{cond}_gold"][mask]).mean()
                assert abs(accuracy-cell["recall"]) < 1e-10, (entry["file"], cell)
                n_cells += 1
            if entry["metadata"].startswith("capacity_fp32/"):
                for cond in ["D0", "D256"]:
                    r = expected_pairs[f"{lab}|{cond}"]
                    mask = saved[f"{cond}_K"] == r["K"]
                    gold = saved[f"{cond}_gold"][mask]
                    native = saved[f"{cond}_native"][mask] == gold
                    probe = saved[f"{cond}_L{r['layer']}"][mask] == gold
                    logits = saved[f"{cond}_logits"][mask] == gold
                    assert mask.sum() == r["n"]
                    for key, values in [("output", native), ("probe", probe), ("logit_decoder", logits)]:
                        assert round(float(values.mean()), 4) == r[key], (lab, cond, key)
                    assert paired(probe, native) == r["probe_minus_output"], (lab, cond, "probe-output")
                    assert paired(probe, logits) == r["probe_minus_logit"], (lab, cond, "probe-logits")
                    n_pairs += 1
            source = None
            if entry["metadata"].startswith("ckpt_traj/"):
                source = "p3_probe_ckpt.json"
            elif entry["metadata"].startswith("frame_test/"):
                source = "p3_probe_frame.json"
            if source:
                checkpoint = json.loads((measurements / "probe_cap" / source).read_text())["checkpoints"]
                runs = checkpoint[f"{lab}|{metadata['revision']}"]["runs"]
                for run in runs:
                    if run["control"] is not None:
                        continue
                    cond, layer = run["cond"], run["layer"]
                    for row in run["rows"]:
                        mask = saved[f"{cond}_K"] == row["K"]
                        score = (saved[f"{cond}_L{layer}"][mask] == saved[f"{cond}_gold"][mask]).mean()
                        assert mask.sum() == row["n"] and abs(score-row["probe"]) < 1e-10
                        n_probe_rows += 1
                if entry["metadata"].startswith("frame_test/"):
                    decoder_runs = json.loads((measurements / "probe_cap/p3_readout_frame.json").read_text())["checkpoints"]
                    for run in decoder_runs[f"{lab}|{metadata['revision']}"]["runs"]:
                        cond, tag = run["cond"], run["control"]
                        if tag not in ("logits", "centered"):
                            continue
                        for row in run["rows"]:
                            mask = saved[f"{cond}_K"] == row["K"]
                            score = (saved[f"{cond}_{tag}"][mask] == saved[f"{cond}_gold"][mask]).mean()
                            assert mask.sum() == row["n"] and abs(score-row["probe"]) < 1e-10
                            n_decoder_rows += 1
    if generated:
        for path in generated.glob("*.tex"):
            if (root / path.name).exists():
                assert path.read_bytes() == (root / path.name).read_bytes(), f"stale generated file: {path.name}"
        figures = ["fig_p3_wording.pdf", "fig_p3_measured.pdf", "fig_p3_heldout.pdf",
                   "fig_p3_checkpoint.pdf", "fig_p3_development.pdf"]
        for name in figures:
            assert (generated / name).stat().st_size > 1000, name
            release = PdfReader(root / "figs_p456" / name).metadata.subject
            rebuilt = PdfReader(generated / name).metadata.subject
            assert release and release.startswith("p3-series-sha256:"), f"missing data fingerprint: {name}"
            assert rebuilt == release, f"figure data/identity mismatch: {name}"
    paper = (root / "p3_capacity.tex").read_text()
    labels = re.findall(r"\\label\{([^}]+)\}", paper)
    assert len(labels) == len(set(labels))
    assert set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", paper)) <= set(labels)
    for name in re.findall(r"\\input\{([^}]+)\}", paper):
        assert (root / (name+".tex")).exists(), name
    for name in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", paper):
        assert (root / "figs_p456" / name).exists(), name
    assert "not yet been synchronized" not in paper
    print(f"PASS: {len(manifest['files'])} input hashes, {n_cells} native cells, "
          f"{n_pairs} paired comparisons, {n_probe_rows} pilot/checkpoint probe rows, "
          f"{n_decoder_rows} pilot decoder rows, generated TeX/figure fingerprints and paper references")


def source_zip(root, target, generated):
    paper = (root / "p3_capacity.tex").read_text()
    members = {"p3_capacity.tex", "neurips_2026.sty"}
    members.update(name+".tex" for name in re.findall(r"\\input\{([^}]+)\}", paper))
    members.update("figs_p456/"+name for name in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", paper))
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            path = root / name
            if generated and name.startswith("figs_p456/"):
                path = generated / Path(name).name
            archive.write(path, name)
    print("wrote self-contained source archive:", target, "members:", len(members))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--generated", type=Path, help="compare regenerated TeX and require regenerated figures")
    parser.add_argument("--source-zip", type=Path, help="package the paper's exact TeX/figure dependency closure")
    args = parser.parse_args()
    verify(args.root, args.generated)
    if args.source_zip:
        source_zip(args.root, args.source_zip, args.generated)


if __name__ == "__main__":
    main()
