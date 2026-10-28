# ICLR report template

The user supplied the ICLR 2025 Conference Submission source bundle on
2026-10-29, after selecting the ICLR option listed in the course instructions.
The bundle was imported locally; no browser or network download was used.
Only the ICLR style and bibliography style were copied. natbib and fancyhdr
are provided by the installed TeX distribution. Example
scientific text, sample references and the example PDF were not imported.

## Original assets

These hashes identify the original supplied assets. The bibliography style has
only trailing whitespace normalized. The ICLR style has one documented compatibility fallback for
its unused review-ruler font; all other course changes live in `template.tex`.

| File | SHA-256 |
| --- | --- |
| `iclr2025_conference.sty` | `9e280a9ff99be84ad190f5cafe047a1a32c2eeb9ca14232ddceae42d8489c697` |
| `iclr2025_conference.bst` | `2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5` |

The style requests a legacy `phvb` font metric missing from local BasicTeX.
The copied style falls back to the standard `cmssbx10` metric only if `phvb`
is unavailable. The final-mode report never displays that review ruler.
Original source files in the supplied directory are unchanged.

## Course adaptations

- One column, using the supplied ICLR layout and Times text.
- The supplied style forces 10-point text. `template.tex` overrides the main
  text to 11 pt with a 13.2 pt baseline and a spacing multiplier of 1, matching
  the course's 11-point, single-spaced requirement.
- Final mode displays the actual authors and removes review line numbering.
  The wrapper clears the conference-publication header; the course report is
  not presented as an ICLR submission or accepted conference paper.
- Latin Modern supplies scalable math/mono fonts; Times remains the text face.
  This avoids size substitutions and unavailable Courier on local BasicTeX.
- The course limit is six content pages, excluding references, rather than the
  conference's page limit. Recheck this after writing the report.

## Build

Run `make report` from the repository root. The final file is `report/report.pdf`;
all auxiliary files and SyncTeX stay in `report/.build/`. Alternatively, run
`make -C report pdf-direct` with pdfLaTeX and BibTeX. Sources and template assets
are versionable; the final PDF and auxiliary directory remain ignored.

The abstract, five course sections and contributions are intentionally empty.
The existing bibliography is retained but inactive until the first citation is
added. Uncomment the bibliography command in `report.tex` when it is needed.
Original source PDFs belong in `docs/papers/`; figure assets stay in `figures/`.
