import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment.qa_agent.direct_generation import (
    _all_targets_met,
    _difficulty_targets,
    _should_stop_direct,
)


def test_direct_should_stop_when_min_difficulty_targets_met():
    diff_targets = _difficulty_targets(5, {"easy": 0.2, "medium": 0.5, "hard": 0.3})

    should_stop, reason = _should_stop_direct(
        candidate_count=9,
        diff_counts={"easy": 2, "medium": 4, "hard": 3},
        diff_targets=diff_targets,
        max_candidate_target=10,
        rounds_processed=3,
        max_rounds=5,
    )

    assert _all_targets_met({"easy": 2, "medium": 4, "hard": 3}, diff_targets) is True
    assert should_stop is True
    assert reason == "min_candidate_pool_sufficient"


def test_direct_should_stop_when_max_candidate_target_reached():
    diff_targets = _difficulty_targets(5, {"easy": 0.2, "medium": 0.5, "hard": 0.3})

    should_stop, reason = _should_stop_direct(
        candidate_count=10,
        diff_counts={"easy": 0, "medium": 10, "hard": 0},
        diff_targets=diff_targets,
        max_candidate_target=10,
        rounds_processed=3,
        max_rounds=5,
    )

    assert should_stop is True
    assert reason == "max_candidate_pool_reached"


def test_direct_should_stop_when_max_rounds_reached():
    diff_targets = _difficulty_targets(5, {"easy": 0.2, "medium": 0.5, "hard": 0.3})

    should_stop, reason = _should_stop_direct(
        candidate_count=1,
        diff_counts={"easy": 0, "medium": 1, "hard": 0},
        diff_targets=diff_targets,
        max_candidate_target=10,
        rounds_processed=10,
        max_rounds=5,
    )

    assert should_stop is True
    assert reason == "max_rounds_reached"
