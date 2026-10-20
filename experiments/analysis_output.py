"""Write analysis summaries while retaining terminal output."""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import TextIO


class _Tee:
    def __init__(self, terminal: TextIO, summary: TextIO):
        self.terminal = terminal
        self.summary = summary

    def write(self, text: str) -> int:
        self.terminal.write(text)
        return self.summary.write(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.summary.flush()


def run_analysis(analysis: Callable[[], None], summary_path: Path) -> None:
    """Save printed tables and metrics as UTF-8 text."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        summary_path.open("w", encoding="utf-8") as summary,
        redirect_stdout(_Tee(sys.stdout, summary)),
    ):
        analysis()
    print(f"Wrote {summary_path}")


def condition_labels(protocol: dict) -> dict[str, str]:
    """Return compact, protocol-aware names for background-knowledge conditions."""
    attr_a = protocol["attr_a"]
    attr_b = protocol["attr_b"]
    short_names = {
        "marital-status": "marital",
        "education": "education",
        "age": "age",
        "hours-per-week": "hours/week",
        "capital-gain": "capital gain",
    }
    short_a = short_names.get(attr_a, attr_a)
    short_b = short_names.get(attr_b, attr_b)
    return {
        "none": "No background",
        "A": f"A: {short_a}",
        "B": f"B: {short_b}",
        "A+B": f"A+B: {short_a} + {short_b}",
    }
