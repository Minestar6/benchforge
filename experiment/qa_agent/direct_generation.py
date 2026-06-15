"""Direct Generation 基线：复用 EvidenceManager，不经 planner/executor 直接生成。

与 B/C/D 组使用相同的 retrieval/chunking/evidence 管道，确保实验公平对比。
"""

import asyncio
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from benchforge.utils.run_context import RunContext
from benchforge.utils.filter import parse_llm_response


# ── 直接生成 prompt（极简，仅描述任务 + 输出字段） ──────────────────────────

_QA_SYSTEM = """\
You are a question generator. Generate question-answer pairs from the provided text.
Each generated question must include two additional fields: `estimated_difficulty` and `citations`.
`question`: The question text
`answer`: Complete, accurate answer to the question
`estimated_difficulty`: Difficulty rating from 1 (easiest) to 10 (hardest)
`citations`: Exact quotes from the source text that support the answer
Generate questions in the following JSON format:

```json
[
  {{
    "question": "The question text",
    "answer": "The correct answer",
    "estimated_difficulty": 5,
    "citations": ["Exact quote 1 from source text"]
  }}
]
```

Return ONLY the JSON array. No markdown, no explanation.
"""

_MCQ_SYSTEM = """\
You are a multiple-choice question generator. Generate multiple-choice questions from the provided text.
Each question must have exactly 4 options (A, B, C, D) with one correct answer.
Each generated question must include the following fields: `question`, `options`, `answer`, `estimated_difficulty`, and `citations`.
`question`: The question text
`options`: Must be an array of 4 strings, each prefixed with the letter (e.g. "A. ...", "B. ...").
`answer`: Must be the letter of the correct option: "A", "B", "C", or "D".
`estimated_difficulty`: Difficulty rating from 1 (easiest) to 10 (hardest)
`citations`: Exact quotes from the source text that support the answer
Generate questions in the following JSON format:

```json
[
  {{
    "question": "The question text",
    "options": ["A. Option A text", "B. Option B text", "C. Option C text", "D. Option D text"],
    "answer": "A",
    "estimated_difficulty": 5,
    "citations": ["Exact quote 1 from source text"]
  }}
]
```

Return ONLY the JSON array. No markdown, no explanation."""

_USER_TEMPLATE = """\
<title>{title}</title>
<text>
{chunk}
</text>"""


# ── 工具函数 ────────────────────────────────────────────────────────────────

def _difficulty_targets(count: int, dist: dict[str, float]) -> dict[str, int]:
    """按比例分配各难度目标数量，确保总和 = count。"""
    total = sum(dist.values())
    targets = {d: max(1, math.floor(count * v / total)) for d, v in dist.items()}
    diff = count - sum(targets.values())
    if diff > 0:
        for d in sorted(dist, key=lambda k: dist[k], reverse=True):
            targets[d] += 1
            diff -= 1
            if diff == 0:
                break
    return targets


def _all_targets_met(counts: dict[str, int], targets: dict[str, int]) -> bool:
    return all(counts.get(d, 0) >= t for d, t in targets.items())


def _max_candidate_target(count: int, multiplier: float) -> int:
    return math.ceil(count * multiplier)


def _should_stop_direct(
    *,
    candidate_count: int,
    diff_counts: dict[str, int],
    diff_targets: dict[str, int],
    max_candidate_target: int,
    rounds_processed: int,
    max_rounds: int,
) -> tuple[bool, str | None]:
    if candidate_count >= max_candidate_target:
        return True, "max_candidate_pool_reached"

    if _all_targets_met(diff_counts, diff_targets):
        return True, "min_candidate_pool_sufficient"

    if rounds_processed >= max_rounds:
        return True, "max_rounds_reached"

    return False, None


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


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


# ── LLM 调用 ────────────────────────────────────────────────────────────────

async def _generate_from_chunk(
    model_client,
    model_name: str,
    chunk_text: str,
    chunk_idx: int,
    doc_title: str,
    tracer,
    topic: str,
    system_prompt: str,
    tracker: TokenTracker,
) -> list[dict]:
    """对单个 chunk 调用 LLM 生成题目。"""
    user_msg = _USER_TEMPLATE.format(title=doc_title, chunk=chunk_text)

    span = tracer.span(
        model=model_name,
        tags={"topic": topic, "chunk_idx": chunk_idx, "agent": "group_a_direct"},
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
        }
        for item in items
        if isinstance(item, dict) and "question" in item and "answer" in item
    ]


# ── 主入口 ──────────────────────────────────────────────────────────────────

async def run_direct_generation(
    *,
    blueprint,
    model_client,
    model_name: str,
    output_dir: Path,
    config_path: str,
) -> dict:
    """直接生成基线：复用 EvidenceManager 获取证据，直接 prompt 生成题目。

    不经 planner、executor、candidate_pool、反馈循环。
    使用与 B/C/D 组相同的 retrieval/chunking 管道。
    """
    from benchforge.agents.qa_agent.config_loader import load_qa_agent_config
    from benchforge.agents.qa_agent.evidence_manager import EvidenceManager

    config_path = Path(config_path)
    agent_config, model_ref, retrieval_cfg, chunking_cfg, sum_chunking_cfg, multi_chunk_cfg = (
        load_qa_agent_config(config_path)
    )

    # ── 构造 EvidenceManager（与 B/C/D 组相同的证据管道） ──
    class _EvidenceConfig:
        __slots__ = ("retrieval", "chunking", "summarization_chunking", "multi_chunk")
        def __init__(self, retrieval, chunking, summarization_chunking, multi_chunk):
            self.retrieval = retrieval
            self.chunking = chunking
            self.summarization_chunking = summarization_chunking
            self.multi_chunk = multi_chunk
        def get_resolved_output_path(self) -> Path:
            return output_dir

    evidence_config = _EvidenceConfig(retrieval_cfg, chunking_cfg, sum_chunking_cfg, multi_chunk_cfg)
    evidence_mgr = EvidenceManager(evidence_config, model_client)

    # ── 准备证据（并行检索所有 topic） ──
    logger.info(f"[Direct] Preparing evidence for {len(blueprint.topics)} topics")

    async def _prepare_topic(topic: str):
        return topic, await evidence_mgr.prepare_evidence(topic, blueprint)

    topic_evidence = await asyncio.gather(*[_prepare_topic(t) for t in blueprint.topics])

    # ── 按难度分布计算目标 ──
    mode_name, mode_cfg = next(iter(blueprint.modes.items()))
    difficulty_dist = mode_cfg.difficulty_distribution
    diff_targets = _difficulty_targets(mode_cfg.count, difficulty_dist)

    # ── 构造 prompt（根据 mode 选择 QA 或 MCQ 模板） ──
    easy_n = diff_targets.get("easy", 0)
    medium_n = diff_targets.get("medium", 0)
    hard_n = diff_targets.get("hard", 0)
    _prompt_template = _MCQ_SYSTEM if mode_name == "multiple_choice" else _QA_SYSTEM
    system_prompt = _prompt_template.format(easy=easy_n, medium=medium_n, hard=hard_n)
    logger.info(f"[Direct] mode={mode_name}, using {'MCQ' if mode_name == 'multiple_choice' else 'QA'} prompt")

    # ── 初始化追踪 ──
    run_ctx = RunContext(task_id=blueprint.task_id, run_id=blueprint.run_id)
    tracer = run_ctx.create_tracer(agent="group_a_direct", stage="generation")
    tracker = TokenTracker()
    diff_counts: dict[str, int] = {d: 0 for d in diff_targets}
    all_candidates: list[dict] = []
    max_candidate_target = _max_candidate_target(
        mode_cfg.count,
        agent_config.candidate_pool.max_candidate_multiplier,
    )
    rounds_processed = 0
    stopped = False
    stop_reason = None

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[Direct] min_target={diff_targets}, max_target={max_candidate_target}, "
        f"max_rounds={mode_cfg.max_rounds}, topics={blueprint.topics}"
    )

    for topic, (chunks, evidence_pool) in topic_evidence:
        if stopped:
            break

        evidence_mgr.evidence_pools[topic] = evidence_pool

        # 按文档分组 chunks，获取文档标题
        chunks_by_doc: dict[str, list] = {}
        for chunk in chunks:
            chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

        for doc_id, doc_chunks in chunks_by_doc.items():
            if stopped:
                break

            source_doc = evidence_mgr.documents.get(doc_id)
            doc_title = source_doc.title if source_doc else topic

            for idx, chunk in enumerate(doc_chunks):
                if stopped:
                    break

                items = await _generate_from_chunk(
                    model_client, model_name,
                    chunk_text=chunk.text if hasattr(chunk, "text") else str(chunk),
                    chunk_idx=idx,
                    doc_title=doc_title,
                    tracer=tracer, topic=topic,
                    system_prompt=system_prompt,
                    tracker=tracker,
                )
                rounds_processed += 1

                for item in items:
                    item["topic"] = topic
                    item["document_id"] = doc_id
                    d_raw = item.get("estimated_difficulty", 5)
                    if isinstance(d_raw, (int, float)):
                        d = "easy" if d_raw <= 3 else "medium" if d_raw <= 6 else "hard"
                    else:
                        d = str(d_raw)
                    if d in diff_counts:
                        diff_counts[d] += 1

                all_candidates.extend(items)

                logger.info(
                    f"[Direct] chunk {idx + 1}/{len(doc_chunks)} → {len(items)} items "
                    f"| counts={diff_counts} / targets={diff_targets} "
                    f"| candidates={len(all_candidates)}/{max_candidate_target} "
                    f"| rounds={rounds_processed}/{mode_cfg.max_rounds}"
                )

                should_stop, stop_reason = _should_stop_direct(
                    candidate_count=len(all_candidates),
                    diff_counts=diff_counts,
                    diff_targets=diff_targets,
                    max_candidate_target=max_candidate_target,
                    rounds_processed=rounds_processed,
                    max_rounds=mode_cfg.max_rounds,
                )
                if should_stop:
                    logger.info(f"[Direct] stopping: {stop_reason}")
                    stopped = True
                    break

    # ── 保存结果 ──
    # 为每个候选添加 question_mode，便于下游统一加载
    for item in all_candidates:
        item["question_mode"] = mode_name

    _write_json(output_dir / "direct_questions.json", all_candidates)

    # 保存到 {mode}/candidate_pool.json，与 B/C/D 组保持一致
    mode_output_dir = output_dir / mode_name
    mode_output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(mode_output_dir / "candidate_pool.json", all_candidates)

    # 构建并保存 shared_state.json，使 verify_agent 可复用统一入口
    from benchforge.utils.shared_state import build_shared_state, save_shared_state as _save_ss
    _state = build_shared_state(
        task_id=blueprint.task_id,
        run_id=blueprint.run_id,
        blueprint=blueprint,
    )
    _save_ss(_state, output_dir)

    report = {
        "task_id": blueprint.task_id,
        "run_id": blueprint.run_id,
        "group": "A - Direct Generation",
        "topics": blueprint.topics,
        "target_per_mode": mode_cfg.count,
        "min_difficulty_targets": diff_targets,
        "max_candidate_target": max_candidate_target,
        "max_rounds": mode_cfg.max_rounds,
        "rounds_processed": rounds_processed,
        "difficulty_counts": diff_counts,
        "min_target_reached": _all_targets_met(diff_counts, diff_targets),
        "max_target_reached": len(all_candidates) >= max_candidate_target,
        "stopped_reason": stop_reason,
        "candidate_count": len(all_candidates),
        "token_stats": tracker.summary(),
    }
    _write_json(output_dir / "generation_report.json", report)

    logger.info(
        f"[Direct] Complete: {len(all_candidates)} candidates, "
        f"tokens={tracker.input_tokens + tracker.output_tokens} "
        f"({tracker.input_tokens} in / {tracker.output_tokens} out)"
    )

    return report
