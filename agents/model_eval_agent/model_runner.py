"""模型推理执行器：并发调用候选模型，落盘 model_responses.jsonl。"""

import asyncio
import time
from pathlib import Path
from typing import Any

from loguru import logger

from benchforge.models.base import BaseModelClient
from benchforge.utils.artifact_store import ArtifactStore
from benchforge.utils.llm_tracer import LLMTracer


def _format_multiple_choice_options(choices: Any) -> str:
    if not choices:
        return ""
    if isinstance(choices, dict):
        return "\n".join(f"{k}. {v}" for k, v in choices.items())

    formatted: list[str] = []
    for idx, choice in enumerate(choices):
        text = str(choice)
        stripped = text.lstrip()
        if stripped.startswith(("A.", "B.", "C.", "D.", "(A)", "(B)", "(C)", "(D)")):
            formatted.append(text)
        else:
            formatted.append(f"{chr(65 + idx)}. {text}")
    return "\n".join(formatted)


async def _run_single(
    client: BaseModelClient,
    model_name: str,
    question: dict[str, Any],
    generation_defaults: dict[str, Any],
    llm_trace_path: str,
    tracer: LLMTracer | None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    prompt = _build_prompt(question)
    messages = [{"role": "user", "content": prompt}]

    async with semaphore:
        t0 = time.monotonic()
        try:
            # 使用 Span 自动记录（优先），向后兼容 llm_trace_path
            if tracer is not None:
                with tracer.span(
                    model=model_name,
                    provider=getattr(client, "provider", ""),
                    tags={
                        "question_id": question["question_id"],
                        "question_mode": question.get("question_mode", ""),
                    },
                ):
                    resp = await client.complete(
                        model=model_name,
                        messages=messages,
                        llm_trace_path=llm_trace_path,
                        **generation_defaults,
                    )
            else:
                resp = await client.complete(
                    model=model_name,
                    messages=messages,
                    llm_trace_path=llm_trace_path,
                    **generation_defaults,
                )
            prediction = resp.get("text", "").strip()
            error = None
        except Exception as e:
            prediction = ""
            error = str(e)
            resp = {}
        latency = time.monotonic() - t0

    return {
        "question_id": question["question_id"],
        "model_name": model_name,
        "question_mode": question.get("question_mode", ""),
        "prediction": prediction,
        "latency_seconds": round(latency, 3),
        "input_tokens": resp.get("input_tokens", 0),
        "output_tokens": resp.get("output_tokens", 0),
        "llm_call_id": resp.get("llm_call_id"),
        "error": error,
    }


def _build_prompt(question: dict[str, Any]) -> str:
    q_text = question.get("question", "")
    mode = question.get("question_mode", "")

    if mode == "multiple_choice":
        choices = question.get("choices") or question.get("options") or {}
        options_text = _format_multiple_choice_options(choices)
        return (
            f"{q_text}\n\n{options_text}\n\n"
            "Please answer with the option letter only (A, B, C, or D)."
        )
    else:
        citations = question.get("citations") or question.get("chunks") or []
        if citations:
            evidence = "\n\n".join(
                c.get("text", c) if isinstance(c, dict) else str(c)
                for c in citations[:3]
            )
            return f"Context:\n{evidence}\n\nQuestion: {q_text}\n\nAnswer:"
        return f"Question: {q_text}\n\nAnswer:"


async def run_candidate_models(
    questions: list[dict[str, Any]],
    models: list[tuple[str, BaseModelClient]],  # (model_name, client)
    generation_defaults: dict[str, Any],
    output_dir: Path,
    llm_trace_path: str,
    tracer: LLMTracer | None = None,
    max_concurrency: int = 8,
) -> list[dict[str, Any]]:
    """并发调用所有候选模型，返回并落盘 model_responses。"""
    store = ArtifactStore(str(output_dir))
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        _run_single(client, model_name, q, generation_defaults, llm_trace_path, tracer, semaphore)
        for model_name, client in models
        for q in questions
    ]

    logger.info(f"[ModelRunner] running {len(tasks)} inferences ({len(models)} models × {len(questions)} questions)")
    responses = await asyncio.gather(*tasks)
    store.append_jsonl("model_responses.jsonl", list(responses))

    # 汇总 tracer 统计
    if tracer is not None:
        usage = tracer.usage()
        logger.info(f"[ModelRunner] done, {len(responses)} responses, "
                     f"llm_usage={usage}")

    logger.info(f"[ModelRunner] done, saved {len(responses)} responses")
    return list(responses)
