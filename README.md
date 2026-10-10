# RTDS Project

Project for the **Research Topics in Data Science** course, within the Master's Degree in Computer Science at Stockholm University, in 2026.

The project investigates the reconstruction of a sensitive attribute from decision trees in a Rashomon set. It compares how reconstruction changes when the attacker knows neither of two selected attributes, only one, or both.

The literature review has been completed. The experimental protocol and implementation are currently being defined.

## Repository structure

```text
rtds-project/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   ├── rashomon/         # Rashomon-set construction
│   └── attack/           # attribute reconstruction
└── report/              # LaTeX report source and build configuration
```

## Requirements

Python 3.11 or later. Project metadata is defined in `pyproject.toml`; no dependencies have been added yet.

## Current status

The `src/rashomon/` and `src/attack/` directories contain only `.gitkeep` placeholders; no algorithms are implemented yet. The report has a blank `main.tex` and an empty `sections/` directory; section headings will be added once the required structure is confirmed.

## Report compilation

Requires `make`, `latexmk`, and `pdflatex`. Compile from the repository root:

```sh
make -C report
```

The Makefile calls `latexmk` to manage compilation passes and dependencies. Automatic compilation is handled by the editor; in VS Code, configure LaTeX Workshop to run this Makefile on save. Add `\input{sections/filename}` entries to `report/main.tex` when section files are introduced.

Experiment execution, Python testing, and linting commands will be documented when the corresponding implementation and tooling are introduced.
