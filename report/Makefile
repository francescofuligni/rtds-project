LATEXMK ?= latexmk
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex
PDFLATEX_FLAGS = -synctex=1 -output-directory=.build -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: pdf pdf-direct

# Prefer latexmk; BasicTeX installations can use the bounded direct recipe.
ifeq ($(shell command -v $(LATEXMK) 2>/dev/null),)
pdf: pdf-direct
else
pdf:
	mkdir -p .build
	$(LATEXMK) -norc -pdf -synctex=1 -outdir=.build -out2dir=. -interaction=nonstopmode -halt-on-error -file-line-error -e '@out2_exts = ( "pdf" );' main.tex
endif

pdf-direct:
	mkdir -p .build
	$(PDFLATEX) $(PDFLATEX_FLAGS) main.tex
	cd .build && BIBINPUTS=..: BSTINPUTS=..: $(BIBTEX) main
	$(PDFLATEX) $(PDFLATEX_FLAGS) main.tex
	$(PDFLATEX) $(PDFLATEX_FLAGS) main.tex
	cp .build/main.pdf main.pdf
