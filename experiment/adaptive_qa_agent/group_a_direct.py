"""Group A: Direct Generation（基线）

流程：
  topic → search_wikipedia → fetch_wikipedia_page
  → 按 1000 字符切分
  → 每块调 LLM 分别生成 QA 和 MCQ 题目（极简 prompt，仅指定输出字段）
  → 达到 target_count 后停止该题型的 chunk 处理
  → 保存 candidate_pool + 汇总 token 统计 + generation_report

终止条件：每种题型生成 ≥ qa_count 道候选即停止（与 B/C/D 的可比口径）。
支持从 run_all.py 传入真实模型客户端，或独立运行时使用 FakeModelClient。
"""
import asyncio
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from benchforge.models.fake import FakeModelClient
from benchforge.utils.retrieval import search_wikipedia, fetch_wikipedia_page
from benchforge.utils.filter import parse_llm_response
from benchforge.utils.run_context import RunContext

CHUNK_SIZE = 1000

# ── 极简系统提示词（基线：仅描述任务 + 输出字段，无额外工作流） ──

_QA_SYSTEM = """\
You are a question generator. Generate question-answer pairs from the provided text.
Each generated question must include two additional fields: `difficulty` and `citations`.
`difficulty` must be one of: `easy`, `medium`, or `hard`.
`citations` must be a list of exact quotes copied from the source text that directly support the answer. Each quote should be sufficient to verify the answer and should not be paraphrased.
Generate questions in the following JSON format:

```json
[
  {
    "question": "The question text",
    "answer": "The correct answer",
    "difficulty": "easy | medium | hard",
    "citations": ["Exact quote 1 from source text"]
  }
]
```

Target distribution: {easy} easy, {medium} medium, {hard} hard questions.
Return ONLY the JSON array. No markdown, no explanation."""

_MCQ_SYSTEM = """\
You are a question generator. Generate multiple-choice questions from the provided text.
Each generated question must include two additional fields: `difficulty` and `citations`.
`difficulty` must be one of: `easy`, `medium`, or `hard`.
`citations` must be a list of exact quotes copied from the source text that directly support the answer. Each quote should be sufficient to verify the answer and should not be paraphrased.
Generate questions in the following JSON format:

```json
[
  {
    "question": "The question text",
    "choices": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "answer": "Option A text",
    "difficulty": "easy | medium | hard",
    "citations": ["Exact quote 1 from source text"]
  }
]
```

Target distribution: {easy} easy, {medium} medium, {hard} hard questions.
Return ONLY the JSON array. No markdown, no explanation."""

_USER_TEMPLATE = """\
<title>{title}</title>
<text>
{chunk}
</text>"""

MODE_CONFIG = [
    ("qa", _QA_SYSTEM, "qa"),
    ("mcq", _MCQ_SYSTEM, "mcq"),
]


def _chunk_text(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _difficulty_targets(qa_count: int, dist: dict) -> dict[str, int]:
    """按比例分配各难度目标数量，确保总和 = qa_count。"""
    total = sum(dist.values())
    targets = {d: max(1, math.floor(qa_count * v / total)) for d, v in dist.items()}
    # 补足舍入误差
    diff = qa_count - sum(targets.values())
    if diff > 0:
        for d in sorted(dist, key=lambda k: dist[k], reverse=True):
            targets[d] += 1
            diff -= 1
            if diff == 0:
                break
    return targets


def _all_targets_met(counts: dict[str, int], targets: dict[str, int]) -> bool:
    return all(counts.get(d, 0) >= t for d, t in targets.items())


class TokenTracker:
    """聚合 token 消耗统计。"""
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.errors = 0
        self.total_latency_ms = 0.0

    def record(self, response: dict, latency_ms: float = 0.0):
        self.calls += 1
        self.input_tokens += response.get("input_tokens", 0)
        self.output_tokens += response.get("output_tokens", 0)
        self.total_latency_ms += latency_ms

    def record_error(self):
        self.errors += 1

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "avg_latency_ms": round(self.total_latency_ms / max(1, self.calls)),
            "total_latency_ms": round(self.total_latency_ms),
        }


async def _generate_from_chunk(
    model_client, model_name: str, chunk: str, chunk_idx: int,
    doc_title: str, tracer, topic: str,
    mode_key: str, system_prompt: str, question_mode: str,
    tracker: TokenTracker,
) -> list[dict]:
    """对单个 chunk 调用 LLM 生成题目，返回 items 并记录 token。"""
    user_msg = _USER_TEMPLATE.format(
        title=doc_title,
        chunk=chunk,
    )
    span = tracer.span(
        model=model_name,
        tags={"topic": topic, "chunk_idx": chunk_idx, "mode": mode_key, "agent": "group_a_direct"},
    )
    t0 = time.perf_counter()
    try:
        async with span:
            response = await model_client.complete(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=2048,
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        tracker.record(response, latency_ms)
    except Exception:
        tracker.record_error()
        return []

    # 清理 LLM 响应：移除可能的 markdown 代码块包裹
    text = response["text"].strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    items = parse_llm_response(text)
    return [
        {
            **item,
            "chunk_idx": chunk_idx,
            "source": "direct",
            "question_mode": question_mode,
        }
        for item in items
        if isinstance(item, dict) and "question" in item and "answer" in item
    ]


async def run(
    model_client=None,
    model_name: str = "fake",
    topics: list[str] | None = None,
    task_id: str = "exp_task",
    language: str = "en",
    qa_count: int = 10,
    difficulty_distribution: dict | None = None,
    **kwargs,
):
    """直接生成基线。

    终止条件：每种题型在每个难度档（easy/medium/hard）均达到按比例分配的目标数量后停止。
    """
    if model_client is None:
        model_client = FakeModelClient(delay=0.0)
        model_name = "fake"
    if topics is None:
        topics = ["topic_a", "topic_b"]
    if difficulty_distribution is None:
        difficulty_distribution = {"easy": 0.2, "medium": 0.2, "hard": 0.6}

    diff_targets = _difficulty_targets(qa_count, difficulty_distribution)

    run_id = f"run_a_direct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_ctx = RunContext(task_id=task_id, run_id=run_id)
    tracer = run_ctx.create_tracer(agent="group_a_direct", stage="generation")
    base = Path("runs") / task_id / run_id / "adaptive"

    # ── 状态追踪 ──
    trackers: dict[str, TokenTracker] = {"qa": TokenTracker(), "mcq": TokenTracker()}
    all_qa: list[dict] = []
    all_mcq: list[dict] = []
    stopped_modes: set[str] = set()
    mode_candidates: dict[str, list[dict]] = {"qa": all_qa, "mcq": all_mcq}
    # 各模式下各难度的已收集数量
    diff_counts: dict[str, dict[str, int]] = {
        "qa": {d: 0 for d in diff_targets},
        "mcq": {d: 0 for d in diff_targets},
    }

    max_chunks_per_mode = qa_count
    chunks_processed: dict[str, int] = {"qa": 0, "mcq": 0}

    # 按难度目标构造每个 chunk 的 prompt 参数（目标数量提示）
    per_chunk_target = max(1, qa_count // max(1, max_chunks_per_mode))
    prompt_easy = max(1, math.floor(per_chunk_target * difficulty_distribution.get("easy", 0.2) / sum(difficulty_distribution.values())))
    prompt_medium = max(1, math.floor(per_chunk_target * difficulty_distribution.get("medium", 0.2) / sum(difficulty_distribution.values())))
    prompt_hard = max(1, math.ceil(per_chunk_target * difficulty_distribution.get("hard", 0.6) / sum(difficulty_distribution.values())))

    # 格式化 system prompt（填入难度分布数量）
    fmt = {"easy": prompt_easy, "medium": prompt_medium, "hard": prompt_hard}
    mode_config_rendered = [
        ("qa", _QA_SYSTEM.format(**fmt), "qa"),
        ("mcq", _MCQ_SYSTEM.format(**fmt), "mcq"),
    ]

    print(f"[Group A] target={qa_count} per mode, diff_targets={diff_targets}, "
          f"max_chunks={max_chunks_per_mode}, topics={topics}")

    for topic in topics:
        if len(stopped_modes) == len(mode_config_rendered):
            print(f"[Group A] both modes reached target, stopping topic loop.")
            break

        results = search_wikipedia(query=topic, language=language, max_pages=2)
        for result in results:
            doc = fetch_wikipedia_page(result=result, run_id=run_id, language=language)
            full_text = (doc.summary or "") + "\n" + (doc.content or "")
            chunks = _chunk_text(full_text, CHUNK_SIZE)
            doc_id = result.page_id if hasattr(result, "page_id") else topic
            doc_title = getattr(doc, "title", topic)

            _write_json(base / "chunks" / f"{doc_id}.json",
                        {"doc_id": doc_id, "topic": topic, "chunks": chunks, "chunk_count": len(chunks)})

            for mode_key, system_prompt, question_mode in mode_config_rendered:
                if mode_key in stopped_modes:
                    print(f"  [{mode_key}] already at target, skipped.")
                    continue

                candidates: list[dict] = []
                for idx, chunk in enumerate(chunks):
                    if chunks_processed[mode_key] >= max_chunks_per_mode:
                        print(f"  [{mode_key}] max_chunks ({max_chunks_per_mode}) reached, forcing stop.")
                        stopped_modes.add(mode_key)
                        break

                    items = await _generate_from_chunk(
                        model_client, model_name, chunk, idx,
                        doc_title=doc_title,
                        tracer=tracer, topic=topic,
                        mode_key=mode_key, system_prompt=system_prompt,
                        question_mode=question_mode, tracker=trackers[mode_key],
                    )
                    chunks_processed[mode_key] += 1
                    for item in items:
                        item["topic"] = topic
                        item["document_id"] = doc_id
                        d = item.get("difficulty", "medium")
                        if d in diff_counts[mode_key]:
                            diff_counts[mode_key][d] += 1
                    candidates.extend(items)
                    t = trackers[mode_key]
                    print(f"  [{mode_key}] chunk {idx + 1}/{len(chunks)} → {len(items)} items"
                          f" | counts={diff_counts[mode_key]} / targets={diff_targets}"
                          f" | chunks_used={chunks_processed[mode_key]}/{max_chunks_per_mode}"
                          f" | tokens in={t.input_tokens} out={t.output_tokens}")

                    # 终止检查：所有难度档均达到目标
                    if _all_targets_met(diff_counts[mode_key], diff_targets):
                        print(f"  [{mode_key}] all difficulty targets met, stopping chunks.")
                        stopped_modes.add(mode_key)
                        break

                mode_candidates[mode_key].extend(candidates)
                _write_json(base / mode_key / "per_doc" / f"{doc_id}.json", candidates)

            if len(stopped_modes) == len(mode_config_rendered):
                break

    # ── 保存汇总 candidate_pool ──
    for mode_key, candidates in mode_candidates.items():
        _write_json(base / mode_key / "candidate_pool.json", candidates)
        print(f"Group A [{mode_key}]: {len(candidates)} candidates → {base / mode_key / 'candidate_pool.json'}")

    # ── 生成最终报告（与 B/C/D 的 generation_report.json 对齐） ──
    total_input = sum(t.input_tokens for t in trackers.values())
    total_output = sum(t.output_tokens for t in trackers.values())
    total_calls = sum(t.calls for t in trackers.values())
    total_errors = sum(t.errors for t in trackers.values())

    report = {
        "task_id": task_id,
        "run_id": run_id,
        "group": "A - Direct Generation",
        "topics": topics,
        "target_per_mode": qa_count,
        "difficulty_targets": diff_targets,
        "max_chunks_per_mode": max_chunks_per_mode,
        "chunks_processed": chunks_processed,
        "stopped_modes": list(stopped_modes),
        "modes": {
            mode_key: {
                "candidate_count": len(candidates),
                "difficulty_counts": diff_counts[mode_key],
                "target_reached": _all_targets_met(diff_counts[mode_key], diff_targets),
                "token_stats": trackers[mode_key].summary(),
            }
            for mode_key, candidates in mode_candidates.items()
        },
        "token_total": {
            "calls": total_calls,
            "errors": total_errors,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        },
    }
    _write_json(base / "generation_report.json", report)
    print(f"Group A report: {base / 'generation_report.json'}")
    print(f"  Total tokens: {total_input + total_output} ({total_input} in / {total_output} out)"
          f" across {total_calls} calls ({total_errors} errors)")

    return mode_candidates


if __name__ == "__main__":
    asyncio.run(run())
