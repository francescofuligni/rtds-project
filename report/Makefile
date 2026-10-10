LATEXMK ?= latexmk

.PHONY: pdf

pdf:
	mkdir -p .build
	$(LATEXMK) -norc -pdf -synctex=1 -outdir=.build -out2dir=. -interaction=nonstopmode -halt-on-error -file-line-error -e '@out2_exts = ( "pdf" );' main.tex
