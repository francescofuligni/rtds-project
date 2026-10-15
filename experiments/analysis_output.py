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
