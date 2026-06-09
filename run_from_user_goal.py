"""Minimal repo-root entrypoint for natural-language BenchForge runs."""

from pathlib import Path
import sys


PROJECT_PARENT = Path(__file__).resolve().parent.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from benchforge.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
