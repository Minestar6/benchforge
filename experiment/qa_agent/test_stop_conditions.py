import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.qa_agent.executor import mode_should_stop
from agents.qa_agent.state import CandidateRecord, CandidateStatus, GlobalState, ModeState
from experiment.qa_agent.fixtures import make_blueprint, make_config


def _accepted_record(topic: str, difficulty: str, idx: int) -> CandidateRecord:
    return CandidateRecord(
        question_id=f"{topic}-{difficulty}-{idx}",
        question=f"{difficulty} question {idx}",
        answer="answer",
        topic=topic,
        difficulty=difficulty,
        status=CandidateStatus.ACCEPTED,
        source_round=1,
        source_strategy="normal_generate",
        chunk_ids=[f"chunk-{idx}"],
    )


def _make_mode_state(records: list[CandidateRecord], *, initial_done: bool = True) -> ModeState:
    state = ModeState(mode="qa")
    state.candidate_questions.extend(records)
    if initial_done:
        state.initial_coverage = {"Climate Change", "Artificial Intelligence"}
    return state


@pytest.mark.parametrize(
    ("records", "expected_reason"),
    [
            (
                [
                    *[_accepted_record("Climate Change", "easy", i) for i in range(2)],
                    *[_accepted_record("Climate Change", "medium", i) for i in range(4)],
                    *[_accepted_record("Artificial Intelligence", "hard", i) for i in range(3)],
                ],
                "min_candidate_pool_sufficient",
        ),
        (
            [_accepted_record("Climate Change", "medium", i) for i in range(10)],
            "max_candidate_pool_reached",
        ),
    ],
)
def test_mode_should_stop_on_new_candidate_pool_thresholds(records, expected_reason):
    blueprint = make_blueprint(count=5)
    config = make_config()
    mode_cfg = blueprint.modes["qa"]
    mode_state = _make_mode_state(records)

    should_stop, reason = mode_should_stop(
        mode_cfg=mode_cfg,
        mode_state=mode_state,
        global_state=GlobalState(),
        blueprint=blueprint,
        config=config,
    )

    assert should_stop is True
    assert reason == expected_reason


def test_mode_should_not_stop_before_initial_breadth_even_if_threshold_met():
    blueprint = make_blueprint(count=5)
    config = make_config()
    mode_cfg = blueprint.modes["qa"]
    records = [
        *[_accepted_record("Climate Change", "easy", i) for i in range(2)],
        *[_accepted_record("Climate Change", "medium", i) for i in range(4)],
        *[_accepted_record("Artificial Intelligence", "hard", i) for i in range(3)],
    ]
    mode_state = _make_mode_state(records, initial_done=False)

    should_stop, reason = mode_should_stop(
        mode_cfg=mode_cfg,
        mode_state=mode_state,
        global_state=GlobalState(),
        blueprint=blueprint,
        config=config,
    )

    assert should_stop is False
    assert reason is None
