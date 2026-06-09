"""PlannerAgent 反馈解析模块（docs/plan-runtime-aligned.md § 7）。"""

import json
from pathlib import Path

from benchforge.utils.shared_state import load_shared_state
from benchforge.utils.llm_tracer import TraceReader

from .schema import (
    GeneratorFeedback,
    GeneratorFeedbackArtifacts,
    GeneratorFeedbackSummary,
    ModeGeneratorFeedback,
    ValidatorFeedback,
    ValidatorFeedbackSummary,
    ValidatorQualitySignals,
    EvaluatorFeedback,
    EvaluatorFeedbackSummary,
    DatasetSignals,
    RoundSpec,
)


def _trace_usage_for_agent(llm_calls_path: str | None, agent_name: str) -> tuple[int, int, int]:
    if not llm_calls_path or not Path(llm_calls_path).exists():
        return 0, 0, 0

    usage = TraceReader(llm_calls_path).cost_by_agent().get(
        agent_name,
        {"calls": 0, "input_tokens": 0, "output_tokens": 0},
    )
    return (
        usage.get("calls", 0),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )


def build_generator_feedback(
    shared_state_path: str | Path,
    round_spec: RoundSpec,
) -> GeneratorFeedback:
    """从 shared_state 和产物文件构建 GeneratorFeedback。

    数据来源：
    - shared_state.artifacts
    - generation_report.json
    - {mode}/mode_state.json
    - llm_calls.jsonl（可选，统计 qa_agent token）
    """
    state = load_shared_state(shared_state_path)
    run_dir = Path("runs") / state.task_id / state.run_id

    # 读 generation_report.json
    report_path = state.artifact("generation_report") or str(run_dir / "generation_report.json")
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    by_mode: dict[str, ModeGeneratorFeedback] = {}
    topic_coverage: dict[str, dict[str, int]] = {}

    for mode_name, mode_summary in report.get("modes", {}).items():
        # 读 mode_state.json
        mode_state_key = f"{mode_name}_mode_state"
        mode_state_path = state.artifact(mode_state_key) or str(run_dir / mode_name / "mode_state.json")
        if Path(mode_state_path).exists():
            with open(mode_state_path, encoding="utf-8") as f:
                mode_state = json.load(f)
        else:
            mode_state = {}

        candidate_count = mode_summary.get("candidate_count", 0)
        target = mode_summary.get("target_candidate_count", 1)
        by_mode[mode_name] = ModeGeneratorFeedback(
            candidate_count=candidate_count,
            target_candidate_count=target,
            fulfillment_rate=candidate_count / max(target, 1),
            stopped_reason=mode_summary.get("stopped_reason"),
            difficulty_counts=mode_state.get("difficulty_counts", {}),
            topic_counts=mode_state.get("topic_counts", {}),
        )

        # 合并 topic_counts 到 topic_coverage
        for topic, count in mode_state.get("topic_counts", {}).items():
            if topic not in topic_coverage:
                topic_coverage[topic] = {"candidate_count": 0}
            topic_coverage[topic]["candidate_count"] += count

    # 统计 llm token（从 llm_calls.jsonl 按 agent=qa_agent 聚合）
    _, llm_input_tokens, llm_output_tokens = _trace_usage_for_agent(
        state.artifact("llm_calls"),
        "qa_agent",
    )

    artifacts = GeneratorFeedbackArtifacts(
        shared_state_path=str(shared_state_path),
        generation_report=report_path,
        qa_candidate_pool=state.artifact("qa_candidate_pool"),
        multiple_choice_candidate_pool=state.artifact("multiple_choice_candidate_pool"),
        qa_mode_state=state.artifact("qa_mode_state"),
        multiple_choice_mode_state=state.artifact("multiple_choice_mode_state"),
        chunked_evidence=state.artifact("chunked_evidence"),
    )

    summary = GeneratorFeedbackSummary(
        total_candidates=report.get("total_candidates", 0),
        global_used_chunk_combinations=report.get("global_used_chunk_combinations", 0),
        global_failures=report.get("global_failures", 0),
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=llm_output_tokens,
    )

    return GeneratorFeedback(
        task_id=state.task_id,
        run_id=state.run_id,
        round_id=round_spec.round_id,
        status="success" if summary.total_candidates > 0 else "failed",
        artifacts=artifacts,
        summary=summary,
        by_mode=by_mode,
        topic_coverage=topic_coverage,
    )


def build_validator_feedback(
    shared_state_path: str | Path,
    round_id: int,
) -> ValidatorFeedback | None:
    """从 shared_state 和 validation 产物构建 ValidatorFeedback。

    数据来源：
    - validation_report.json
    - validated_questions.jsonl
    - weighted_selection.json
    - llm_calls.jsonl
    """
    state = load_shared_state(shared_state_path)
    run_dir = Path("runs") / state.task_id / state.run_id
    val_dir = run_dir / "validation"

    report_path = state.artifact("validation_report") or str(val_dir / "validation_report.json")
    if not Path(report_path).exists():
        return None

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    trace_calls, trace_input_tokens, trace_output_tokens = _trace_usage_for_agent(
        state.artifact("llm_calls"),
        "verify_agent",
    )

    summary = ValidatorFeedbackSummary(
        total_candidates=report.get("total_candidates", 0),
        citation_passed=report.get("citation_passed", 0),
        llm_passed=report.get("llm_passed", 0),
        final_selected=report.get("final_selected", 0),
        citation_pass_rate=report.get("citation_passed", 0) / max(report.get("total_candidates", 1), 1),
        llm_pass_rate_after_citation=report.get("llm_passed", 0) / max(report.get("citation_passed", 1), 1),
        final_selection_rate=report.get("final_selected", 0) / max(report.get("total_candidates", 1), 1),
        llm_calls=trace_calls or report.get("llm_usage", {}).get("calls", 0),
        llm_input_tokens=trace_input_tokens or report.get("llm_usage", {}).get("input_tokens", 0),
        llm_output_tokens=trace_output_tokens or report.get("llm_usage", {}).get("output_tokens", 0),
    )

    # 读 validated_questions.jsonl，统计各维度分布
    validated_path = state.artifact("validated_questions") or str(val_dir / "validated_questions.jsonl")
    by_final_status: dict[str, int] = {}
    by_mode: dict[str, dict[str, int]] = {}
    by_difficulty: dict[str, dict[str, int]] = {}
    by_topic: dict[str, dict[str, any]] = {}
    citation_scores_all = []
    citation_scores_selected = []
    llm_scores_all = []
    llm_scores_selected = []

    if Path(validated_path).exists():
        with open(validated_path, encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                final_status = q.get("final_status", "unknown")
                by_final_status[final_status] = by_final_status.get(final_status, 0) + 1

                candidate = q.get("candidate", {})
                mode = candidate.get("question_mode", "unknown")
                difficulty = candidate.get("estimated_difficulty", "unknown")
                topic = candidate.get("topic", "unknown")

                # by_mode
                if mode not in by_mode:
                    by_mode[mode] = {"selected": 0, "reserve": 0, "rejected": 0}
                if final_status == "selected":
                    by_mode[mode]["selected"] += 1
                elif final_status == "reserve":
                    by_mode[mode]["reserve"] += 1
                else:
                    by_mode[mode]["rejected"] += 1

                # by_difficulty
                if difficulty not in by_difficulty:
                    by_difficulty[difficulty] = {"selected": 0, "reserve": 0, "rejected": 0}
                if final_status == "selected":
                    by_difficulty[difficulty]["selected"] += 1
                elif final_status == "reserve":
                    by_difficulty[difficulty]["reserve"] += 1
                else:
                    by_difficulty[difficulty]["rejected"] += 1

                # by_topic
                if topic not in by_topic:
                    by_topic[topic] = {
                        "selected": 0,
                        "reserve": 0,
                        "rejected": 0,
                        "citation_scores": [],
                        "llm_scores": [],
                    }
                if final_status == "selected":
                    by_topic[topic]["selected"] += 1
                elif final_status == "reserve":
                    by_topic[topic]["reserve"] += 1
                else:
                    by_topic[topic]["rejected"] += 1

                # citation / llm scores
                citation_val = q.get("citation_validation", {})
                if citation_score := citation_val.get("citation_score"):
                    citation_scores_all.append(citation_score)
                    by_topic[topic]["citation_scores"].append(citation_score)
                    if final_status == "selected":
                        citation_scores_selected.append(citation_score)

                llm_val = q.get("llm_validation", {})
                if overall_score := llm_val.get("overall_score"):
                    llm_scores_all.append(overall_score)
                    by_topic[topic]["llm_scores"].append(overall_score)
                    if final_status == "selected":
                        llm_scores_selected.append(overall_score)

    # 汇总 by_topic 平均分
    for topic_data in by_topic.values():
        cit_scores = topic_data.pop("citation_scores", [])
        llm_scores = topic_data.pop("llm_scores", [])
        topic_data["avg_citation_score"] = sum(cit_scores) / len(cit_scores) if cit_scores else None
        topic_data["avg_llm_overall_score"] = sum(llm_scores) / len(llm_scores) if llm_scores else None

    # weighted_selection.json
    selection_path = state.artifact("weighted_selection") or str(val_dir / "weighted_selection.json")
    dropped_dup = 0
    dropped_overquota = 0
    if Path(selection_path).exists():
        with open(selection_path, encoding="utf-8") as f:
            sel = json.load(f)
            dropped_dup = len(sel.get("dropped_as_duplicate", []))
            dropped_overquota = len(sel.get("dropped_as_overquota", []))

    quality_signals = ValidatorQualitySignals(
        avg_citation_score_all=sum(citation_scores_all) / len(citation_scores_all) if citation_scores_all else None,
        avg_citation_score_selected=sum(citation_scores_selected) / len(citation_scores_selected) if citation_scores_selected else None,
        avg_llm_overall_score_all=sum(llm_scores_all) / len(llm_scores_all) if llm_scores_all else None,
        avg_llm_overall_score_selected=sum(llm_scores_selected) / len(llm_scores_selected) if llm_scores_selected else None,
        duplicate_rate=dropped_dup / max(summary.total_candidates, 1),
        overquota_rate=dropped_overquota / max(summary.llm_passed, 1),
    )

    return ValidatorFeedback(
        task_id=state.task_id,
        run_id=state.run_id,
        round_id=round_id,
        status="success" if summary.final_selected > 0 else "partial_success",
        artifacts={
            "validation_report": report_path,
            "validated_questions": validated_path,
            "weighted_selection": selection_path,
        },
        summary=summary,
        quality_signals=quality_signals,
        failed_by_stage=report.get("failed_by_stage", {}),
        by_final_status=by_final_status,
        by_mode=by_mode,
        by_difficulty=by_difficulty,
        by_topic=by_topic,
        selection_summary={
            "dropped_as_duplicate": dropped_dup,
            "dropped_as_overquota": dropped_overquota,
        },
    )


def build_evaluator_feedback(
    shared_state_path: str | Path,
    round_id: int,
) -> EvaluatorFeedback | None:
    """从 shared_state 和 evaluation 产物构建 EvaluatorFeedback。

    数据来源：
    - evaluation_report.json
    - dataset_quality_summary.json
    - model_overall_report.csv（需解析）
    - llm_calls.jsonl
    """
    state = load_shared_state(shared_state_path)
    run_dir = Path("runs") / state.task_id / state.run_id
    eval_dir = run_dir / "evaluation"

    report_path = state.artifact("evaluation_report") or str(eval_dir / "evaluation_report.json")
    if not Path(report_path).exists():
        return None

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    # dataset_quality_summary.json
    ds_summary_path = state.artifact("dataset_quality_summary") or str(eval_dir / "dataset_report" / "dataset_quality_summary.json")
    dataset_signals = DatasetSignals()
    if Path(ds_summary_path).exists():
        with open(ds_summary_path, encoding="utf-8") as f:
            ds = json.load(f)
            dataset_signals = DatasetSignals(
                citation_score_mean=ds.get("citation_score", {}).get("mean"),
                citation_pass_rate=ds.get("citation_pass_rate", {}).get("mean"),
                diversity_score=ds.get("diversity_score"),
                embedding_dispersion=ds.get("embedding_dispersion"),
                cluster_entropy=ds.get("cluster_entropy"),
            )

    # 统计 evaluator llm token
    _, llm_input_tokens, llm_output_tokens = _trace_usage_for_agent(
        state.artifact("llm_calls"),
        "model_eval_agent",
    )

    summary = EvaluatorFeedbackSummary(
        num_questions=report.get("num_questions", 0),
        num_models=report.get("num_models", 0),
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=llm_output_tokens,
    )

    # 简化版：不深度解析 CSV，只记录路径
    return EvaluatorFeedback(
        task_id=state.task_id,
        run_id=state.run_id,
        round_id=round_id,
        status="success",
        artifacts={
            "evaluation_report": report_path,
            "dataset_quality_summary": ds_summary_path,
            "model_overall_report": state.artifact("model_overall_report") or str(eval_dir / "model_report" / "model_overall_report.csv"),
            "model_aggregate_report": state.artifact("model_aggregate_report") or str(eval_dir / "model_report" / "model_aggregate_report.json"),
            "model_by_topic": state.artifact("model_by_topic") or str(eval_dir / "model_report" / "model_by_topic.csv"),
            "model_by_difficulty": state.artifact("model_by_difficulty") or str(eval_dir / "model_report" / "model_by_difficulty.csv"),
            "model_by_question_mode": state.artifact("model_by_question_mode") or str(eval_dir / "model_report" / "model_by_question_mode.csv"),
        },
        summary=summary,
        dataset_signals=dataset_signals,
        derived_performance_signals={},  # 后续可扩展
        dataset_metrics_ref={"summary_json": ds_summary_path},
        model_metrics_ref={
            "overall_csv": state.artifact("model_overall_report") or str(eval_dir / "model_report" / "model_overall_report.csv"),
        },
    )
