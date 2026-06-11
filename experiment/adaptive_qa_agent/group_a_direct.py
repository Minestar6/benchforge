"""Group A: Direct Generation

流程：
  topic → search_wikipedia → fetch_wikipedia_page
  → 按 1000 字符切分
  → 每块调 LLM 生成题目列表（使用 qa_agent 标准格式）
  → 保存到 runs/{task_id}/{run_id}/adaptive/qa/candidate_pool.json
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from benchforge.models.fake import FakeModelClient
from benchforge.utils.retrieval import search_wikipedia, fetch_wikipedia_page
from benchforge.utils.filter import parse_llm_response
from benchforge.utils.run_context import RunContext

TASK_ID = "exp_task"
RUN_ID = "run_a_direct"
TOPICS = ["topic_a", "topic_b"]
CHUNK_SIZE = 1000
LANGUAGE = "en"

_SYSTEM_PROMPT = (Path(__file__).parent.parent.parent / "prompts/question_generator/qa_system_prompt.md").read_text()

_USER_TEMPLATE = (Path(__file__).parent.parent.parent / "prompts/question_generator/qa_user_prompt.md").read_text()


def _chunk_text(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


async def _generate_from_chunk(
    model_client, model_name: str, chunk: str, chunk_idx: int,
    doc_title: str, doc_summary: str, tracer, topic: str,
) -> list[dict]:
    user_msg = _USER_TEMPLATE.format(
        additional_instructions="Generate as many question-answer pairs as possible.",
        doc_title=doc_title,
        doc_summary=doc_summary,
        evidence_text=chunk,
    )
    span = tracer.span(
        model=model_name,
        tags={"topic": topic, "chunk_idx": chunk_idx, "agent": "group_a_direct"},
    )
    async with span:
        response = await model_client.complete(
            model=model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
    items = parse_llm_response(response["text"])
    return [
        {**item, "chunk_idx": chunk_idx, "source": "direct", "question_mode": item.get("question_mode", "qa")}
        for item in items
        if isinstance(item, dict) and "question" in item and "answer" in item
    ]


async def run(model_client=None, model_name: str = "fake"):
    if model_client is None:
        model_client = FakeModelClient(delay=0.0)
        model_name = "fake"

    run_ctx = RunContext(task_id=TASK_ID, run_id=RUN_ID)
    tracer = run_ctx.create_tracer(agent="group_a_direct", stage="generation")
    base = Path("runs") / TASK_ID / RUN_ID / "adaptive" / "qa"

    all_candidates = []
    for topic in TOPICS:
        results = search_wikipedia(query=topic, language=LANGUAGE, max_pages=2)
        for result in results:
            doc = fetch_wikipedia_page(result=result, run_id=RUN_ID, language=LANGUAGE)
            full_text = (doc.summary or "") + "\n" + (doc.content or "")
            chunks = _chunk_text(full_text, CHUNK_SIZE)

            # 保存中间文件：chunks
            doc_id = result.page_id if hasattr(result, "page_id") else topic
            chunks_path = base / "chunks" / f"{doc_id}.json"
            chunks_path.parent.mkdir(parents=True, exist_ok=True)
            chunks_path.write_text(json.dumps({"doc_id": doc_id, "topic": topic, "chunks": chunks}, ensure_ascii=False, indent=2))

            chunk_results = []
            for idx, chunk in enumerate(chunks):
                items = await _generate_from_chunk(
                    model_client, model_name, chunk, idx,
                    doc_title=getattr(doc, "title", topic),
                    doc_summary=doc.summary or "",
                    tracer=tracer, topic=topic,
                )
                for item in items:
                    item["topic"] = topic
                    item["document_id"] = doc_id
                chunk_results.extend(items)
                all_candidates.extend(items)

            # 保存中间文件：per-document results
            doc_out = base / "per_doc" / f"{doc_id}.json"
            doc_out.parent.mkdir(parents=True, exist_ok=True)
            doc_out.write_text(json.dumps(chunk_results, ensure_ascii=False, indent=2))

    out = base / "candidate_pool.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_candidates, ensure_ascii=False, indent=2))
    print(f"Group A: {len(all_candidates)} candidates → {out}")
    return all_candidates


if __name__ == "__main__":
    asyncio.run(run())
