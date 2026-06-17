"""LLM Judge 执行器：并发评分，落盘 llm_judge_scores.jsonl + traces/llm_judge_traces.jsonl。"""

import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from benchforge.config.config import load_prompt
from benchforge.models.base import BaseModelClient
from benchforge.utils.artifact_store import ArtifactStore
from benchforge.utils.llm_tracer import LLMTracer

from .schema import JudgeConfig, JudgeMetricSpec, QuestionModeMetricPlan
from .model_runner import _format_multiple_choice_options
from .aggregation import build_mode_question_ids, compute_scores_and_mean


def _question_text_for_judge(question: dict[str, Any]) -> str:
    question_text = question.get("question", "")
    if question.get("question_mode") != "multiple_choice":
        return question_text
    choices = question.get("choices") or question.get("options") or {}
    options_text = _format_multiple_choice_options(choices)
    if not options_text:
        return question_text
    return f"{question_text}\n{options_text}"


def _evidence_text_for_judge(question: dict[str, Any]) -> str:
    citations = question.get("citations") or question.get("chunks") or []
    return "\n".join(
        c.get("text", c) if isinstance(c, dict) else str(c)
        for c in citations[:3]
    ) or "N/A"


def _metrics_text(metrics: list[JudgeMetricSpec]) -> str:
    return "\n".join(f"- {metric.name}: {metric.description}" for metric in metrics)


def _output_format_text(metrics: list[JudgeMetricSpec]) -> str:
    score_lines = []
    for index, metric in enumerate(metrics):
        suffix = "," if index < len(metrics) - 1 else ""
        score_lines.append(f'  "{metric.name}": <score 1-5>{suffix}')
    return "{\n" + "\n".join(score_lines) + "\n}"


def _render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def _render_system_prompt(
    template: str,
    metrics: list[JudgeMetricSpec],
) -> str:
    if not template:
        return ""
    return _render_template(
        template,
        {
            "metrics": _metrics_text(metrics),
            "output_format": _output_format_text(metrics),
        },
    )


def _render_user_prompt(
    template: str,
    question: dict[str, Any],
    prediction: str,
    metrics: list[JudgeMetricSpec],
) -> str:
    question_text = _question_text_for_judge(question)
    reference_answer = str(question.get("answer", ""))
    evidence = _evidence_text_for_judge(question)
    if not template:
        # 内置简单模板
        return (
            f"Question:\n{question_text}\n\n"
            f"Reference Answer:\n{reference_answer}\n\n"
            f"Evidence / Citations:\n{evidence}\n\n"
            f"Model Answer:\n{prediction}\n\n"
            f"Evaluation dimensions:\n{_metrics_text(metrics)}\n\n"
            f"Return JSON only:\n{_output_format_text(metrics)}"
        )
    return _render_template(
        template,
        {
            "citations": evidence,
            "question": question_text,
            "reference_answer": reference_answer,
            "model_answer": prediction,
            "metrics": _metrics_text(metrics),
            "output_format": _output_format_text(metrics),
        },
    )


def _parse_judge_output(text: str, metrics: list[JudgeMetricSpec]) -> tuple[dict[str, Any] | None, bool, str | None]:
    """返回 (parsed_output, parse_success, error)。"""
    # 提取 JSON
    json_str = text.strip()
    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        m = re.search(pattern, json_str, re.DOTALL)
        if m:
            json_str = m.group(1)
            break

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # 尝试提取 {...}
        m = re.search(r'\{.*\}', json_str, re.DOTALL)
        if not m:
            return None, False, f"JSON parse error: {json_str[:200]}"
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError as e:
            return None, False, str(e)

    scores_raw = data.get("scores")
    if not isinstance(scores_raw, dict):
        scores_raw = data
    scores: dict[str, float] = {}
    for metric in metrics:
        val = scores_raw.get(metric.name)
        if val is not None:
            try:
                scores[metric.name] = float(val)
            except (ValueError, TypeError):
                pass

    return {
        "scores": scores,
        "reason": data.get("reason", ""),
    }, True, None


async def _judge_single(
    client: BaseModelClient,
    judge_model_name: str,
    question: dict[str, Any],
    model_name: str,
    prediction: str,
    metrics: list[JudgeMetricSpec],
    system_prompt: str,
    user_template: str,
    judge_defaults: dict[str, Any],
    llm_trace_path: str,
    tracer: LLMTracer | None,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2,
) -> dict[str, Any]:
    rendered_system_prompt = _render_system_prompt(system_prompt, metrics)
    user_prompt = _render_user_prompt(user_template, question, prediction, metrics)
    messages = [{"role": "user", "content": user_prompt}]
    if rendered_system_prompt:
        messages = [{"role": "system", "content": rendered_system_prompt}] + messages

    trace_id = f"judge_{question['question_id']}_{model_name}"
    trace: dict[str, Any] = {
        "trace_id": trace_id,
        "question_id": question["question_id"],
        "question_mode": question.get("question_mode", ""),
        "model_name": model_name,
        "judge_model": judge_model_name,
        "input": {
            "question": question.get("question", ""),
            "question_with_choices": _question_text_for_judge(question),
            "reference_answer": question.get("answer", ""),
            "model_answer": prediction,
            "citations": question.get("citations", []),
            "metrics": [{"name": m.name, "description": m.description} for m in metrics],
        },
        "prompt": {"system": rendered_system_prompt, "user": user_prompt},
    }

    parsed_output = None
    parse_success = False
    error = None
    raw_output = ""
    llm_call_id = None
    retry_count = 0
    latency = 0.0
    input_tokens = 0
    output_tokens = 0
    api_model = getattr(client, "model_name", judge_model_name)

    async with semaphore:
        for attempt in range(max_retries + 1):
            retry_count = attempt
            t0 = time.monotonic()
            try:
                # 使用 Span 自动记录（优先），向后兼容 llm_trace_path
                if tracer is not None:
                    with tracer.span(
                        model=judge_model_name,
                        provider=getattr(client, "provider", ""),
                        tags={
                            "question_id": question["question_id"],
                            "candidate_model": model_name,
                            "metric_names": [m.name for m in metrics],
                        },
                    ):
                        resp = await client.complete(
                            model=api_model,
                            messages=messages,
                            llm_trace_path=llm_trace_path,
                            **judge_defaults,
                        )
                else:
                    resp = await client.complete(
                        model=api_model,
                        messages=messages,
                        llm_trace_path=llm_trace_path,
                        **judge_defaults,
                    )
                raw_output = resp.get("text", "")
                llm_call_id = resp.get("llm_call_id")
                input_tokens = resp.get("input_tokens", 0)
                output_tokens = resp.get("output_tokens", 0)
                latency = time.monotonic() - t0
                parsed_output, parse_success, error = _parse_judge_output(raw_output, metrics)
                if parse_success:
                    break
            except Exception as e:
                error = str(e)
                latency = time.monotonic() - t0
                if attempt >= max_retries:
                    break
                continue

    trace.update({
        "llm_call_id": llm_call_id,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "parse_success": parse_success,
        "error": error,
        "retry_count": retry_count,
        "latency_seconds": round(latency, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })

    return trace


async def run_llm_judge(
    questions: list[dict[str, Any]],
    model_responses: list[dict[str, Any]],
    metric_plans: dict[str, QuestionModeMetricPlan],
    judge_client: BaseModelClient,
    judge_model_name: str,
    judge_config: JudgeConfig,
    judge_defaults: dict[str, Any],
    output_dir: Path,
    llm_trace_path: str,
    tracer: LLMTracer | None = None,
    max_concurrency: int = 4,
) -> list[dict[str, Any]]:
    system_prompt = load_prompt(judge_config.prompt_system)
    user_template = load_prompt(judge_config.prompt_user)

    q_index = {q["question_id"]: q for q in questions}
    semaphore = asyncio.Semaphore(max_concurrency)

    # 只针对有 llm_judge_metrics 的题型
    tasks = []
    for resp in model_responses:
        if resp.get("error") or not resp.get("prediction"):
            continue
        mode = resp.get("question_mode", "")
        plan = metric_plans.get(mode)
        if not plan or not plan.llm_judge_metrics:
            continue
        question = q_index.get(resp["question_id"])
        if question is None:
            continue
        tasks.append((resp, plan.llm_judge_metrics, question))

    logger.info(f"[LLMJudgeExecutor] running {len(tasks)} judge calls")
    traces = await asyncio.gather(*[
        _judge_single(
            client=judge_client,
            judge_model_name=judge_model_name,
            question=question,
            model_name=resp["model_name"],
            prediction=resp["prediction"],
            metrics=metrics,
            system_prompt=system_prompt,
            user_template=user_template,
            judge_defaults=judge_defaults,
            llm_trace_path=llm_trace_path,
            tracer=tracer,
            semaphore=semaphore,
        )
        for resp, metrics, question in tasks
    ])

    # 落盘 traces
    traces_store = ArtifactStore(str(output_dir / "traces"))
    traces_store.append_jsonl("llm_judge_traces.jsonl", list(traces))
    prompts_store = ArtifactStore(str(output_dir / "traces"))
    prompts_store.append_jsonl(
        "llm_judge_prompts.jsonl",
        [
            {
                "trace_id": trace["trace_id"],
                "question_id": trace["question_id"],
                "question_mode": trace["question_mode"],
                "model_name": trace["model_name"],
                "judge_model": trace["judge_model"],
                "prompt": trace.get("rendered_prompt", trace.get("prompt", {})),
                "metrics": trace["input"]["metrics"],
            }
            for trace in traces
        ],
    )

    # 聚合为 llm_judge_scores.jsonl（metric×mode×models）
    mode_question_ids = build_mode_question_ids(questions)

    # (mode, metric_name) → {model_name: {qid: score}}
    buckets: dict[tuple[str, str], dict[str, dict[str, float | None]]] = defaultdict(lambda: defaultdict(dict))

    for trace in traces:
        if not trace["parse_success"] or not trace["parsed_output"]:
            continue
        mode = trace["question_mode"]
        model_name = trace["model_name"]
        qid = trace["question_id"]
        for metric_name, score in trace["parsed_output"]["scores"].items():
            buckets[(mode, metric_name)][model_name][qid] = score

    judge_scores: list[dict[str, Any]] = []
    for (mode, metric_name), model_data in buckets.items():
        plan = metric_plans.get(mode)
        metric_spec = next(
            (m for m in (plan.llm_judge_metrics if plan else []) if m.name == metric_name),
            None,
        )
        question_ids = mode_question_ids.get(mode, [])

        models_payload: dict[str, Any] = {}
        for model_name, qid_scores in model_data.items():
            _, mean = compute_scores_and_mean(qid_scores, question_ids)
            models_payload[model_name] = {
                "scores": [qid_scores.get(qid) for qid in question_ids],
                "mean": mean,
            }

        judge_scores.append({
            "type": "llm_judge_metric",
            "question_mode": mode,
            "metric_name": metric_name,
            "question_ids": question_ids,
            "models": models_payload,
        })

    store = ArtifactStore(str(output_dir))
    store.append_jsonl("llm_judge_scores.jsonl", judge_scores)
    logger.info(f"[LLMJudgeExecutor] done, {len(judge_scores)} metric rows, {len(traces)} traces")
    return judge_scores
