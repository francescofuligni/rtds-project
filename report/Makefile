LATEXMK ?= latexmk

.PHONY: pdf

pdf:
	mkdir -p .build
	$(LATEXMK) -norc -pdf -outdir=.build -interaction=nonstopmode -halt-on-error -file-line-error main.tex
