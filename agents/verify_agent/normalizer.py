"""输入归一化层：将 candidate_pool.json / accepted_questions.jsonl 统一为 QuestionCandidate。"""

import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

from .schema import QuestionCandidate


def _stable_question_id(question: str, idx: int) -> str:
    return hashlib.md5(question.encode("utf-8")).hexdigest()[:12]


def _normalize_citations(raw: Any) -> list[str]:
    """citations 字段可能是 list[str] 或 list[dict]，统一转为 list[str]。"""
    if not raw:
        return []
    result = []
    for c in raw:
        if isinstance(c, str):
            result.append(c)
        elif isinstance(c, dict):
            result.append(c.get("text", c.get("citation", str(c))))
        else:
            result.append(str(c))
    return result


def _build_chunk_index(chunked_path: str | Path) -> dict[str, str]:
    """从 evidence/chunked.json (优先) 或 chunked.jsonl 构建 chunk_id -> chunk_text 内存索引。"""
    path = Path(chunked_path)
    if not path.exists():
        logger.warning(f"chunked evidence not found at {path}, chunk hydrate disabled")
        return {}

    index: dict[str, str] = {}

    def _index_doc(doc: dict) -> None:
        for chunk in doc.get("chunks", []):
            cid = chunk.get("chunk_id", "")
            text = chunk.get("chunk_text", "")
            if cid and text:
                index[cid] = text

    if path.suffix == ".json":
        # chunked.json: {topic: [doc_records]}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for topic, rows in data.items():
                for row in rows:
                    _index_doc(row)
        elif isinstance(data, list):
            for row in data:
                _index_doc(row)
    else:
        # chunked.jsonl: 每行一个 doc record
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _index_doc(doc)

    logger.info(f"Built chunk index with {len(index)} entries from {path}")
    return index


def _hydrate_chunks(candidate_raw: dict, chunk_index: dict[str, str]) -> list[str]:
    """若 chunks 为空，从 chunk_index 中根据 chunk_ids 回查。"""
    chunks = candidate_raw.get("chunks", [])
    if chunks:
        return [c if isinstance(c, str) else str(c) for c in chunks]

    chunk_ids = candidate_raw.get("chunk_ids", [])
    if not chunk_ids or not chunk_index:
        return []

    hydrated = []
    for cid in chunk_ids:
        text = chunk_index.get(cid, "")
        if text:
            hydrated.append(text)
    return hydrated


def _normalize_one(
    raw: dict,
    idx: int,
    task_id: str,
    run_id: str,
    chunk_index: dict[str, str],
) -> QuestionCandidate:
    question = raw.get("question", "")
    question_id = raw.get("question_id") or _stable_question_id(question, idx)

    citations = _normalize_citations(raw.get("citations", []))
    chunks = _hydrate_chunks(raw, chunk_index)

    chunk_ids = raw.get("chunk_ids", [])
    document_id = raw.get("document_id", "")
    if not document_id and chunk_ids:
        # 从第一个 chunk_id 推导 document_id：doc_xxx::chunk_yyy → doc_xxx
        first_cid = chunk_ids[0]
        document_id = first_cid.split("::")[0] if "::" in first_cid else first_cid

    return QuestionCandidate(
        question_id=question_id,
        task_id=task_id,
        run_id=run_id,
        topic=raw.get("topic", ""),
        question=question,
        answer=raw.get("answer", ""),
        choices=raw.get("choices") or raw.get("options"),
        question_mode=raw.get("question_mode", ""),
        question_type=raw.get("question_type", ""),
        required_capability=raw.get("required_capability", ""),
        estimated_difficulty=raw.get("estimated_difficulty"),
        citations=citations,
        document_id=document_id,
        chunk_ids=chunk_ids,
        chunks=chunks,
        generation_metadata={
            k: v for k, v in raw.items()
            if k not in {
                "question_id", "task_id", "run_id", "topic", "question", "answer",
                "choices", "options", "question_mode", "question_type", "required_capability",
                "estimated_difficulty", "citations", "document_id",
                "chunk_ids", "chunks",
            }
        },
    )


def _load_candidate_pool_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return list(data.values()) if isinstance(data, dict) else []


def _load_accepted_questions_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_and_normalize(
    input_paths: list[str],
    task_id: str,
    run_id: str,
    chunk_index_path: str = "",
) -> list[QuestionCandidate]:
    """加载并归一化所有输入题目，返回 QuestionCandidate 列表。"""
    # 构建 chunk_index（优先使用指定路径，否则自动推导）
    chunk_index: dict[str, str] = {}
    if chunk_index_path:
        chunk_index = _build_chunk_index(chunk_index_path)
    else:
        # 从 run_id 对应的目录自动查找 evidence/chunked.json（优先）或 chunked.jsonl
        for p in input_paths:
            candidate_path = Path(p)
            for parent in candidate_path.parents:
                chunked_json = parent / "evidence" / "chunked.json"
                chunked_jsonl = parent / "evidence" / "chunked.jsonl"
                if chunked_json.exists():
                    chunk_index = _build_chunk_index(chunked_json)
                    break
                elif chunked_jsonl.exists():
                    chunk_index = _build_chunk_index(chunked_jsonl)
                    break
            if chunk_index:
                break

    candidates: list[QuestionCandidate] = []
    global_idx = 0

    for path_str in input_paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"Input path not found: {path}")
            continue

        if path.suffix == ".json":
            raws = _load_candidate_pool_json(path)
        elif path.suffix == ".jsonl":
            raws = _load_accepted_questions_jsonl(path)
        else:
            logger.warning(f"Unknown input format: {path}")
            continue

        logger.info(f"Loaded {len(raws)} raw candidates from {path}")

        for raw in raws:
            c = _normalize_one(raw, global_idx, task_id, run_id, chunk_index)
            candidates.append(c)
            global_idx += 1

    # 去重 question_id（若出现碰撞，加后缀）
    seen: dict[str, int] = {}
    result: list[QuestionCandidate] = []
    for c in candidates:
        qid = c.question_id
        if qid in seen:
            seen[qid] += 1
            c = c.model_copy(update={"question_id": f"{qid}_{seen[qid]}"})
        else:
            seen[qid] = 0
        result.append(c)

    logger.info(f"Normalized {len(result)} candidates total")
    return result
