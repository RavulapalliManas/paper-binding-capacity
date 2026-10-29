PYTHON ?= python3
BUILD ?= build

.PHONY: all assets verify pdf source
all: pdf source

assets:
	mkdir -p $(BUILD)/assets
	$(PYTHON) analysis/make_p3_revision_figs.py --measurement-root results/measurement --out $(BUILD)/assets
	$(PYTHON) analysis/make_p3_revision_figs.py --development-only --out $(BUILD)/assets
	$(PYTHON) analysis/p3_development.py --from-summary results/p3_development.json --emit $(BUILD)/assets
	$(PYTHON) analysis/p3_revision_checks.py --emit $(BUILD)/assets

verify: assets
	$(PYTHON) analysis/gpu_capacity_dev.py --selftest
	$(PYTHON) analysis/p3_paired_ci.py --selftest
	$(PYTHON) analysis/p3_error_anatomy.py --selftest
	$(PYTHON) analysis/capacity_curve.py
	$(PYTHON) analysis/verify_release.py --generated $(BUILD)/assets

pdf: verify
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) p3_capacity.tex > $(BUILD)/latex-pass1.txt
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) p3_capacity.tex > $(BUILD)/latex-pass2.txt
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) p3_capacity.tex > $(BUILD)/latex-pass3.txt
	$(PYTHON) -c 'import pathlib,re; s=pathlib.Path("$(BUILD)/p3_capacity.log").read_text(); assert not re.search(r"Overfull|LaTeX Warning|Package .* Warning|undefined|multiply defined",s), "LaTeX warnings in $(BUILD)/p3_capacity.log"'

source: verify
	$(PYTHON) analysis/verify_release.py --generated $(BUILD)/assets --source-zip $(BUILD)/P3_capacity_arxiv.zip
