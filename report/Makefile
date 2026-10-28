LATEXMK ?= latexmk
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex
PDFLATEX_FLAGS = -synctex=1 -output-directory=.build -interaction=nonstopmode -halt-on-error -file-line-error

.DEFAULT_GOAL := pdf
.PHONY: pdf pdf-direct check-template

# Keep missing template assets from producing a misleading final PDF.
check-template:
	@for file in template.tex iclr2025_conference.sty iclr2025_conference.bst; do \
	  test -f "$$file" || { echo "Missing report template asset: $$file"; exit 1; }; \
	done

ifeq ($(shell command -v $(LATEXMK) 2>/dev/null),)
pdf: pdf-direct
else
pdf: check-template
	mkdir -p .build
	$(LATEXMK) -norc -pdf -synctex=1 -outdir=.build -out2dir=. -interaction=nonstopmode -halt-on-error -file-line-error -e '@out2_exts = ( "pdf" );' report.tex
endif

pdf-direct: check-template
	mkdir -p .build
	$(PDFLATEX) $(PDFLATEX_FLAGS) report.tex
	@if grep -q '\\citation{' .build/report.aux && grep -q '\\bibdata{' .build/report.aux; then \
	  cd .build && BIBINPUTS=..: BSTINPUTS=..: $(BIBTEX) report; \
	fi
	$(PDFLATEX) $(PDFLATEX_FLAGS) report.tex
	$(PDFLATEX) $(PDFLATEX_FLAGS) report.tex
	cp .build/report.pdf report.pdf
