from __future__ import annotations

import csv
import json
from pathlib import Path


TASK_ID = "effectiveness_task_20260618_153413"
TASK_DIR = Path("/Users/zhaoziqing/Desktop/benchforge/runs") / TASK_ID
REPORT_JSON = TASK_DIR / "effectiveness" / "task_effectiveness_report.json"
OUT_DIR = Path("/Users/zhaoziqing/Desktop/benchforge/report/effectiveness_tables")
MARKDOWN_PATH = Path("/Users/zhaoziqing/Desktop/benchforge/report/多轮实验详细数据分析.md")


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


def fmt(value, digits: int = 4):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_verification_tables(task_dir: Path):
    summary_rows = []
    quality_rows = []
    round_ids = sorted([p.name for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("round_")])
    for rid in round_ids:
        report = load_json(task_dir / rid / "validation" / "validation_report.json")
        citation_rows = load_jsonl(task_dir / rid / "validation" / "citation_validation.jsonl")
        llm_rows = load_jsonl(task_dir / rid / "validation" / "llm_validation.jsonl")

        citation_mean = sum(r["citation_score"] for r in citation_rows) / len(citation_rows)
        answer_mean = sum(r["answer_citation_score"] for r in citation_rows) / len(citation_rows)
        chunk_mean = sum(r["chunk_citation_score"] for r in citation_rows) / len(citation_rows)
        llm_mean = sum(r["overall_score"] for r in llm_rows) / len(llm_rows)
        diff_mean = sum(r["dimensions"]["difficulty_consistency"] for r in llm_rows) / len(llm_rows)
        correctness_mean = sum(r["dimensions"]["correctness"] for r in llm_rows) / len(llm_rows)
        clarity_mean = sum(r["dimensions"]["clarity"] for r in llm_rows) / len(llm_rows)
        input_mean = sum(r["input_tokens"] for r in llm_rows) / len(llm_rows)
        output_mean = sum(r["output_tokens"] for r in llm_rows) / len(llm_rows)
        latency_mean = sum(r["latency_ms"] for r in llm_rows) / len(llm_rows)

        summary_rows.append([
            rid,
            report["total_candidates"],
            report["citation_passed"],
            report["llm_passed"],
            report["final_selected"],
            report["failed_by_stage"]["citation"],
            report["failed_by_stage"]["llm"],
            report["llm_usage"]["calls"],
            report["llm_usage"]["input_tokens"],
            report["llm_usage"]["output_tokens"],
            report["stage_latencies_ms"]["llm_validation"],
        ])
        quality_rows.append([
            rid,
            citation_mean,
            answer_mean,
            chunk_mean,
            llm_mean,
            correctness_mean,
            clarity_mean,
            diff_mean,
            input_mean,
            output_mean,
            latency_mean,
        ])
    return summary_rows, quality_rows


def build_round_progress_table(report: dict):
    rows = []
    cumulative = 0
    for item in report["round_summaries"]:
        selected = int(item["selected_count"])
        cumulative += selected
        rid = item["run_id"]
        validated_rows = load_jsonl(TASK_DIR / rid / "validation" / "validated_questions.jsonl")
        qa_selected = sum(
            1
            for rec in validated_rows
            if rec.get("final_status") == "selected" and rec.get("candidate", {}).get("question_mode") == "qa"
        )
        mc_selected = sum(
            1
            for rec in validated_rows
            if rec.get("final_status") == "selected" and rec.get("candidate", {}).get("question_mode") == "multiple_choice"
        )
        mode_summary = item.get("mode_round_summaries", {})
        qa = mode_summary.get("qa", {}).get("round_1", {})
        mc = mode_summary.get("multiple_choice", {}).get("round_1", {})
        rows.append([
            rid,
            selected,
            cumulative,
            qa.get("generated_count"),
            qa_selected,
            qa.get("too_easy_ratio"),
            mc.get("generated_count"),
            mc_selected,
            mc.get("too_easy_ratio"),
        ])
    return rows


def build_eval_tables(report: dict, task_dir: Path):
    b1 = report["b1"]
    bfull = report["bfull"]

    dataset_rows = []
    for name, block in [("B1", b1), ("Bfull", bfull)]:
        eval_result = block["evaluation_result"]
        metrics = block["benchmark_metrics"]
        signals = eval_result["discriminative_signals"]
        dataset_rows.append([
            name,
            block["question_count"],
            eval_result["dataset_metrics"]["citation_score"]["mean"],
            eval_result["dataset_metrics"]["diversity_score"]["score"],
            eval_result["dataset_metrics"]["diversity_score"]["embedding_dispersion"],
            eval_result["dataset_metrics"]["diversity_score"]["cluster_entropy"],
            metrics["top_bottom_gap"],
            metrics["score_variance"],
            metrics["adjacent_gap_mean"],
            metrics["per_question_gap_mean"],
            signals["discriminative_question_ratio"],
            signals["best_vs_second_gap"],
        ])

    model_rows = []
    for name, block in [("B1", b1), ("Bfull", bfull)]:
        eval_result = block["evaluation_result"]
        overall = block["benchmark_metrics"]["model_scores"]
        qa_model = eval_result["qa"]["model"]
        mc_model = eval_result["multiple_choice"]["model"]
        for model_name in overall:
            model_rows.append([
                name,
                model_name,
                overall[model_name],
                qa_model.get(model_name),
                mc_model.get(model_name),
            ])

    topic_rows = []
    difficulty_rows = []
    mode_rows = []
    for name, block in [("B1", b1), ("Bfull", bfull)]:
        eval_result = block["evaluation_result"]
        for topic, info in eval_result["by_topic"].items():
            topic_rows.append([name, topic, info["question_count"], info["model_gap"], info["too_easy_ratio"], info["all_models_fail_ratio"]])
        for difficulty, info in eval_result["by_difficulty"].items():
            difficulty_rows.append([name, difficulty, info["question_count"], info["model_gap"], info["too_easy_ratio"], info["all_models_fail_ratio"]])
        for mode, info in eval_result["by_mode"].items():
            mode_rows.append([name, mode, info["question_count"], info["model_gap"], info["too_easy_ratio"], info["all_models_fail_ratio"]])

    automatic_rows = []
    judge_rows = []
    for subset_name, eval_run_id in [("B1", "b1_eval"), ("Bfull", "bfull_eval")]:
        auto_rows = load_jsonl(task_dir / eval_run_id / "evaluation" / "model_report" / "automatic_scores.jsonl")
        for row in auto_rows:
            for model_name, info in row["models"].items():
                automatic_rows.append([
                    subset_name,
                    row["question_mode"],
                    row["metric_name"],
                    model_name,
                    info["mean"],
                ])

        llm_judge_rows = load_jsonl(task_dir / eval_run_id / "evaluation" / "model_report" / "llm_judge_scores.jsonl")
        for row in llm_judge_rows:
            for model_name, info in row["models"].items():
                judge_rows.append([
                    subset_name,
                    row["question_mode"],
                    row["metric_name"],
                    model_name,
                    info["mean"],
                ])

    return dataset_rows, model_rows, topic_rows, difficulty_rows, mode_rows, automatic_rows, judge_rows


def build_markdown(report: dict, round_progress_rows, verify_summary_rows, verify_quality_rows, dataset_rows, model_rows) -> str:
    b1 = report["b1"]
    bfull = report["bfull"]
    delta = report["benchmark_delta"]
    total_rounds = report["total_rounds"]
    selected_by_round = [row[1] for row in round_progress_rows]

    lines = []
    lines.append("# 多轮实验详细数据分析")
    lines.append("")
    lines.append(f"实验任务：`{report['task_id']}`")
    lines.append("")
    lines.append("## 1. 总体结论")
    lines.append("")
    lines.append(
        f"本次实验共运行 {total_rounds} 轮，累计获得 {report['all_rounds_selected_count']} 道入选题。"
        f"其中第1轮保留 {report['round_1_selected_count']} 道，后续第2至第{total_rounds}轮共新增 {report['later_round_selected_count']} 道。"
        f"后续轮次显著提升了题集的引用得分（{b1['evaluation_result']['dataset_metrics']['citation_score']['mean']:.4f} -> {bfull['evaluation_result']['dataset_metrics']['citation_score']['mean']:.4f}）"
        f"和多样性得分（{b1['evaluation_result']['dataset_metrics']['diversity_score']['score']:.4f} -> {bfull['evaluation_result']['dataset_metrics']['diversity_score']['score']:.4f}），"
        f"但总体模型区分度没有同步增强，Top-Bottom Gap 由 {b1['benchmark_metrics']['top_bottom_gap']:.4f} 下降到 {bfull['benchmark_metrics']['top_bottom_gap']:.4f}。"
    )
    lines.append("")
    lines.append(
        f"从逐轮入选量看，5轮分别保留了 {selected_by_round[0]}、{selected_by_round[1]}、{selected_by_round[2]}、{selected_by_round[3]}、{selected_by_round[4]} 道题，"
        "说明多轮机制在第3至第5轮仍能持续提供有效增量，没有在第2轮后立即失效；但增量并非单调上升，表明系统已经开始受到主题覆盖、题目重复和验证阈值的共同约束。"
    )
    lines.append("")
    lines.append("## 2. 逐轮生成与验证进展")
    lines.append("")
    lines.append("| 轮次 | 当轮入选 | 累计入选 | QA生成 | QA入选 | QA too_easy | MC生成 | MC入选 | MC too_easy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in round_progress_rows:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {fmt(row[3],0)} | {fmt(row[4],0)} | {fmt(row[5])} | {fmt(row[6],0)} | {fmt(row[7],0)} | {fmt(row[8])} |"
        )
    lines.append("")
    lines.append(
        "逐轮数据反映出两个特点。第一，后续轮次确实持续增加了题库规模，第3至第5轮分别又提供了 8、11、14 道入选题。"
        "第二，QA 与多选题的“too_easy”比例在不同轮次波动较大，说明当前反馈机制已经能够不断产题，但对题目难度的稳定控制仍不够平滑。"
    )
    lines.append("")
    lines.append("## 3. 验证智能体分析")
    lines.append("")
    lines.append(
        "验证智能体在5轮实验中总体表现稳定，并呈现出明显的后续轮次改善趋势。第1轮在13道候选题中淘汰2道（1道引文失败、1道LLM质量失败）；"
        "从第2轮开始，所有候选题均通过引文验证与LLM质量验证，说明题目生成与验证标准之间的匹配程度在后续轮次显著提高。"
    )
    lines.append("")
    lines.append(
        "从开销角度看，验证阶段的 LLM 调用量随着候选题数量同步变化，但平均输出 token 没有线性膨胀，说明质量改进并没有伴随明显的裁判冗余累积；"
        "这支持“后续轮次生成结果更规整、验证更稳定”的判断。"
    )
    lines.append("")
    lines.append("## 4. 模型评估智能体分析")
    lines.append("")
    lines.append(
        "模型评估结果需要分开看待 QA 子集与选择题子集。QA 子集上，deepseek-v4 在 semantic accuracy、F1、precision、ROUGE-L、BLEU 等指标上整体最稳，"
        "deepseek-v4-pro 次之，minimax 在 recall 上异常偏高但 precision 很低，表现出明显的“覆盖式作答”倾向。"
    )
    lines.append("")
    lines.append(
        "LLM 裁判结果进一步表明，在开放式问答场景中 deepseek-v4 与 deepseek-v4-pro 的回答相关性、忠实性、事实准确性、逻辑性与完整性均明显优于 minimax。"
    )
    lines.append("")
    lines.append(
        "但在选择题聚合上出现了关键异常：automatic metric 显示 minimax 的 invalid_rate 始终为 1.0，说明其选择题格式输出最差；"
        "然而综合报告中 minimax 的 multiple-choice 模型得分却最高。这说明当前聚合实现很可能没有将 invalid_rate 这样的反向指标做方向统一。"
        "因此，综合模型排名不能直接视为模型真实能力结论，而应同时结合子指标与裁判结果解释。"
    )
    lines.append("")
    lines.append("## 5. 五轮结果下的趋势判断")
    lines.append("")
    lines.append(
        "在5轮实测完成后，可以更明确地看到趋势：多轮反馈对于“继续扩题、提高引用质量和提升多样性”是有效的，但对于“增强最终模型区分度”并不自动成立。"
        "因此下一阶段不应只是继续增加轮数，而应将模型分差、题目方差、裁判分歧和选择题格式错误率显式纳入反馈目标，把后续轮次转为“区分度导向轮”。"
    )
    lines.append("")
    lines.append("## 6. 核心数字摘录")
    lines.append("")
    lines.append("| 指标 | B1 | Bfull | 变化 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| 题目数 | {b1['question_count']} | {bfull['question_count']} | +{report['later_round_selected_count']} |")
    lines.append(f"| 引用得分 | {b1['evaluation_result']['dataset_metrics']['citation_score']['mean']:.4f} | {bfull['evaluation_result']['dataset_metrics']['citation_score']['mean']:.4f} | {bfull['evaluation_result']['dataset_metrics']['citation_score']['mean'] - b1['evaluation_result']['dataset_metrics']['citation_score']['mean']:+.4f} |")
    lines.append(f"| 多样性得分 | {b1['evaluation_result']['dataset_metrics']['diversity_score']['score']:.4f} | {bfull['evaluation_result']['dataset_metrics']['diversity_score']['score']:.4f} | {bfull['evaluation_result']['dataset_metrics']['diversity_score']['score'] - b1['evaluation_result']['dataset_metrics']['diversity_score']['score']:+.4f} |")
    lines.append(f"| Top-Bottom Gap | {b1['benchmark_metrics']['top_bottom_gap']:.4f} | {bfull['benchmark_metrics']['top_bottom_gap']:.4f} | {delta['top_bottom_gap']:+.4f} |")
    lines.append(f"| Score Variance | {b1['benchmark_metrics']['score_variance']:.6f} | {bfull['benchmark_metrics']['score_variance']:.6f} | {delta['score_variance']:+.6f} |")
    lines.append(f"| 区分题比例 | {b1['evaluation_result']['discriminative_signals']['discriminative_question_ratio']:.4f} | {bfull['evaluation_result']['discriminative_signals']['discriminative_question_ratio']:.4f} | {bfull['evaluation_result']['discriminative_signals']['discriminative_question_ratio'] - b1['evaluation_result']['discriminative_signals']['discriminative_question_ratio']:+.4f} |")
    lines.append("")
    lines.append("## 7. 数据文件")
    lines.append("")
    lines.append("详见同目录下导出的 CSV 表格。")
    lines.append("")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = load_json(REPORT_JSON)
    round_progress_rows = build_round_progress_table(report)
    verify_summary_rows, verify_quality_rows = build_verification_tables(TASK_DIR)
    dataset_rows, model_rows, topic_rows, difficulty_rows, mode_rows, automatic_rows, judge_rows = build_eval_tables(report, TASK_DIR)

    write_csv(
        OUT_DIR / "00_round_progress.csv",
        ["round", "selected_this_round", "selected_cumulative", "qa_generated", "qa_selected", "qa_too_easy_ratio", "mc_generated", "mc_selected", "mc_too_easy_ratio"],
        round_progress_rows,
    )

    write_csv(
        OUT_DIR / "01_round_verification_summary.csv",
        ["round", "total_candidates", "citation_passed", "llm_passed", "final_selected", "failed_citation", "failed_llm", "llm_calls", "llm_input_tokens", "llm_output_tokens", "llm_stage_latency_ms"],
        verify_summary_rows,
    )
    write_csv(
        OUT_DIR / "02_round_verification_quality.csv",
        ["round", "citation_mean", "answer_citation_mean", "chunk_citation_mean", "llm_overall_mean", "correctness_mean", "clarity_mean", "difficulty_consistency_mean", "input_token_mean", "output_token_mean", "latency_mean_ms"],
        verify_quality_rows,
    )
    write_csv(
        OUT_DIR / "03_dataset_quality_and_discriminative_signals.csv",
        ["subset", "question_count", "citation_score_mean", "diversity_score", "embedding_dispersion", "cluster_entropy", "top_bottom_gap", "score_variance", "adjacent_gap_mean", "per_question_gap_mean", "discriminative_question_ratio", "best_vs_second_gap"],
        dataset_rows,
    )
    write_csv(
        OUT_DIR / "04_model_overall_scores.csv",
        ["subset", "model", "overall_score", "qa_model_score", "mc_model_score"],
        model_rows,
    )
    write_csv(
        OUT_DIR / "05_topic_breakdown.csv",
        ["subset", "topic", "question_count", "model_gap", "too_easy_ratio", "all_models_fail_ratio"],
        topic_rows,
    )
    write_csv(
        OUT_DIR / "06_difficulty_breakdown.csv",
        ["subset", "difficulty", "question_count", "model_gap", "too_easy_ratio", "all_models_fail_ratio"],
        difficulty_rows,
    )
    write_csv(
        OUT_DIR / "07_mode_breakdown.csv",
        ["subset", "mode", "question_count", "model_gap", "too_easy_ratio", "all_models_fail_ratio"],
        mode_rows,
    )
    write_csv(
        OUT_DIR / "08_automatic_metric_details.csv",
        ["subset", "question_mode", "metric_name", "model", "mean_score"],
        automatic_rows,
    )
    write_csv(
        OUT_DIR / "09_llm_judge_metric_details.csv",
        ["subset", "question_mode", "metric_name", "model", "mean_score"],
        judge_rows,
    )

    MARKDOWN_PATH.write_text(
        build_markdown(report, round_progress_rows, verify_summary_rows, verify_quality_rows, dataset_rows, model_rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
