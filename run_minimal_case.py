"""Minimal runnable BenchForge example.

Run from the repo root:

    python run_minimal_case.py

This script reuses the existing CLI with a small English user goal and
lightweight targets. Defaults prefer a more likely available model route
(`deepseek-v3`) while still allowing manual overrides through CLI flags.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchforge.cli import main as cli_main


DEFAULT_USER_GOAL = (
    "Build a  benchmark to evaluate physics. "
)


def build_default_argv() -> list[str]:
    return [
        "--user-goal",
        DEFAULT_USER_GOAL,
        "--language",
        "en",
        "--candidate-model",
        "deepseek-v3.2",
        "--judge-model",
        "deepseek-v3.2",
        "--planner-model",
        "deepseek-v3.2",
        "--qa-target",
        "2",
        "--mc-target",
        "1",
        "--max-rounds",
        "1",
        "--min-selected-per-round",
        "1",
    ]


def main(argv: list[str] | None = None) -> int:
    return cli_main(argv if argv is not None else build_default_argv())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
