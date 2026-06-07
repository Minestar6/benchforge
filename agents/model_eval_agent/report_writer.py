"""报告写入器：将聚合结果写入 CSV / JSON 文件。"""

import csv
import json
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def write_reports(
    questions: list[dict[str, Any]],
    dataset_scores: list[dict[str, Any]],
    automatic_scores: list[dict[str, Any]],
    judge_scores: list[dict[str, Any]],
    output_root: Path,
    dataset_quality_summary: dict[str, Any],
    dataset_quality_by_group: list[dict[str, Any]],
    model_overall_report: list[dict[str, Any]],
    model_by_dimension: dict[str, list[dict[str, Any]]],
) -> None:
    dataset_dir = output_root / "dataset_report"
    model_dir = output_root / "model_report"

    _write_json(dataset_dir / "dataset_quality_summary.json", dataset_quality_summary)
    _write_csv(dataset_dir / "dataset_quality_by_group.csv", dataset_quality_by_group)

    _write_csv(model_dir / "model_overall_report.csv", model_overall_report)

    for dim_name, rows in model_by_dimension.items():
        _write_csv(model_dir / f"model_by_{dim_name}.csv", rows)

    # model_aggregate_report.json：所有模型的综合摘要
    aggregate: dict[str, Any] = {}
    for row in model_overall_report:
        model = row["model_name"]
        mode = row["question_mode"]
        aggregate.setdefault(model, {})[mode] = {
            k: v for k, v in row.items()
            if k not in ("model_name", "question_mode")
        }
    _write_json(model_dir / "model_aggregate_report.json", aggregate)
