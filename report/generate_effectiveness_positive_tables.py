from __future__ import annotations

import csv
import json
from pathlib import Path


TASK_ID = "effectiveness_task_20260618_153413"
TASK_DIR = Path("/Users/zhaoziqing/Desktop/benchforge/runs") / TASK_ID
REPORT_JSON = TASK_DIR / "effectiveness" / "task_effectiveness_report.json"
OUT_DIR = Path("/Users/zhaoziqing/Desktop/benchforge/report/effectiveness_positive_tables")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def build_round_increment_table(report: dict):
    rows = []
    cumulative = 0
    for item in report["round_summaries"]:
        selected = int(item["selected_count"])
        cumulative += selected
        rows.append([item["run_id"], selected, cumulative])
    return rows


def build_verifier_stability_table():
    rows = []
    for rid in sorted([p.name for p in TASK_DIR.iterdir() if p.is_dir() and p.name.startswith("round_")]):
        rep = load_json(TASK_DIR / rid / "validation" / "validation_report.json")
        rows.append([
            rid,
            rep["total_candidates"],
            rep["citation_passed"],
            rep["llm_passed"],
            rep["final_selected"],
            rep["failed_by_stage"]["citation"],
            rep["failed_by_stage"]["llm"],
        ])
    return rows


def build_dataset_quality_table(report: dict):
    rows = []
    for name, block in [("B1", report["b1"]), ("Bfull", report["bfull"])]:
        d = block["evaluation_result"]["dataset_metrics"]
        rows.append([
            name,
            block["question_count"],
            d["citation_score"]["mean"],
            d["diversity_score"]["score"],
            d["diversity_score"]["embedding_dispersion"],
            d["diversity_score"]["cluster_entropy"],
        ])
    return rows


def build_qa_model_table(report: dict):
    rows = []
    for name, block in [("B1", report["b1"]), ("Bfull", report["bfull"])]:
        qa_model = block["evaluation_result"]["qa"]["model"]
        for model_name in ["deepseek-v4", "deepseek-v4-pro", "minimax"]:
            rows.append([name, model_name, qa_model.get(model_name)])
    return rows


def build_qa_judge_table():
    rows = []
    judge_rows = load_jsonl(TASK_DIR / "bfull_eval" / "evaluation" / "model_report" / "llm_judge_scores.jsonl")
    for row in judge_rows:
        if row["question_mode"] != "qa":
            continue
        metric = row["metric_name"]
        rows.append([
            metric,
            row["models"]["deepseek-v4"]["mean"],
            row["models"]["deepseek-v4-pro"]["mean"],
            row["models"]["minimax"]["mean"],
        ])
    return rows


def build_discriminative_table(report: dict):
    rows = []
    for name, block in [("B1", report["b1"]), ("Bfull", report["bfull"])]:
        s = block["evaluation_result"]["discriminative_signals"]
        rows.append([
            name,
            s["overall_model_gap"],
            s["best_vs_second_gap"],
            s["per_question_variance_mean"],
            s["discriminative_question_ratio"],
            s["easy_questions_too_easy_ratio"],
            s["all_models_fail_ratio"],
        ])
    return rows


def main():
    report = load_json(REPORT_JSON)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        OUT_DIR / "01_round_increment.csv",
        ["round", "selected_this_round", "selected_cumulative"],
        build_round_increment_table(report),
    )
    write_csv(
        OUT_DIR / "02_verifier_stability.csv",
        ["round", "total_candidates", "citation_passed", "llm_passed", "final_selected", "failed_citation", "failed_llm"],
        build_verifier_stability_table(),
    )
    write_csv(
        OUT_DIR / "03_dataset_quality_positive.csv",
        ["subset", "question_count", "citation_score_mean", "diversity_score", "embedding_dispersion", "cluster_entropy"],
        build_dataset_quality_table(report),
    )
    write_csv(
        OUT_DIR / "04_qa_model_scores.csv",
        ["subset", "model", "qa_score"],
        build_qa_model_table(report),
    )
    write_csv(
        OUT_DIR / "05_bfull_qa_judge_scores.csv",
        ["metric", "deepseek-v4", "deepseek-v4-pro", "minimax"],
        build_qa_judge_table(),
    )
    write_csv(
        OUT_DIR / "06_discriminative_signals_positive.csv",
        ["subset", "overall_model_gap", "best_vs_second_gap", "per_question_variance_mean", "discriminative_question_ratio", "easy_questions_too_easy_ratio", "all_models_fail_ratio"],
        build_discriminative_table(report),
    )


if __name__ == "__main__":
    main()
