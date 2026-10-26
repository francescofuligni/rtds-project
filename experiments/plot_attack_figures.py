"""Command-line entry point for plotting.attack_figures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plotting.attack_figures import cli

if __name__ == "__main__":
    cli()
