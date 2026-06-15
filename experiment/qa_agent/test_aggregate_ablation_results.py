import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment.qa_agent.aggregate_ablation_results import compute_metrics


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_compute_metrics_reads_direct_report_fields(tmp_path: Path):
    run_dir = tmp_path / "run_a_direct"
    _write_json(
        run_dir / "metadata.json",
        {
            "run_id": "run_a_direct",
            "group_id": "A_direct",
            "seed": 42,
            "model_name": "fake",
            "count": 5,
            "max_rounds": 5,
            "topics": ["a", "b"],
        },
    )
    _write_json(
        run_dir / "generation_report.json",
        {
            "candidate_count": 8,
            "difficulty_counts": {"easy": 2, "medium": 4, "hard": 2},
            "stopped_reason": "max_rounds_reached",
            "min_target_reached": False,
            "max_target_reached": False,
            "token_stats": {"calls": 3, "input_tokens": 10, "output_tokens": 20},
        },
    )
    _write_json(run_dir / "direct_questions.json", [{"question": "q1"}])

    metrics = compute_metrics(run_dir)

    assert metrics["accepted_count"] == 8
    assert metrics["stopped_reason"] == "max_rounds_reached"
    assert metrics["min_target_reached"] is False
    assert metrics["max_target_reached"] is False


def test_compute_metrics_reads_mode_summary_fields(tmp_path: Path):
    run_dir = tmp_path / "run_d_full"
    _write_json(
        run_dir / "metadata.json",
        {
            "run_id": "run_d_full",
            "group_id": "D_full",
            "seed": 42,
            "model_name": "fake",
            "count": 5,
            "max_rounds": 5,
            "topics": ["a", "b"],
        },
    )
    _write_json(
        run_dir / "generation_report.json",
        {
            "modes": {
                "qa": {
                    "candidate_count": 9,
                    "difficulty_counts": {"easy": 2, "medium": 4, "hard": 3},
                    "stopped_reason": "min_candidate_pool_sufficient",
                    "min_target_reached": True,
                    "max_target_reached": False,
                }
            }
        },
    )

    metrics = compute_metrics(run_dir)

    assert metrics["accepted_count"] == 9
    assert metrics["stopped_reason"] == "min_candidate_pool_sufficient"
    assert metrics["min_target_reached"] is True
    assert metrics["max_target_reached"] is False
