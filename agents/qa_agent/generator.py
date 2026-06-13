"""Generator 模块：根据 GenerationBatch 调用 LLM 生成题目。"""

from pathlib import Path
from typing import Any

from loguru import logger

# --- Prompt 模板缓存 ---

_prompt_cache: dict[str, str] = {}

_PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts" / "question_generator"


def _load_prompt(template_id: str) -> tuple[str, str]:
    """根据 prompt_template_id 加载 system 和 user prompt。"""
    if template_id in _prompt_cache:
        return _prompt_cache[template_id]

    if template_id == "qa_generation_v1":
        sys_key, user_key = "qa_system_prompt", "qa_user_prompt"
    elif template_id == "mcq_generation_v1":
        sys_key, user_key = "mcq_system_prompt", "mcq_user_prompt"
    else:
        sys_key, user_key = f"{template_id}_system", f"{template_id}_user"

    sys_path = _PROMPT_DIR / f"{sys_key}.md"
    user_path = _PROMPT_DIR / f"{user_key}.md"

    system = sys_path.read_text(encoding="utf-8") if sys_path.exists() else ""
    user = user_path.read_text(encoding="utf-8") if user_path.exists() else ""

    result = (system, user)
    _prompt_cache[template_id] = result
    return result


def _render_user_prompt(
    template: str,
    batch: Any,
    evidence_pool: Any,
    document_summary: str,
    additional_instructions: str = "",
) -> str:
    """用 batch 中的数据填充 user prompt 模板。"""
    evidence_lines: list[str] = []
    for cid in batch.single_chunk_ids:
        for u in (evidence_pool.single_chunks if evidence_pool else []):
            if getattr(u, "chunk_id", None) == cid:
                evidence_lines.append(getattr(u, "text", ""))
                break
    for uid in batch.multi_chunk_ids:
        for u in (evidence_pool.multi_chunks if evidence_pool else []):
            if getattr(u, "unit_id", None) == uid:
                for t in getattr(u, "texts", []):
                    evidence_lines.append(t)
                break
    evidence_text = "\n\n".join(evidence_lines)

    result = template
    for var, val in [
        ("additional_instructions", additional_instructions),
        ("doc_title", getattr(batch, "topic", "")),
        ("doc_summary", document_summary),
        ("evidence_text", evidence_text),
    ]:
        result = result.replace("{" + var + "}", str(val))
    return result


class Generator:
    """题目生成器。

    根据 GenerationBatch 中的参数（topic、mode、difficulty、chunk_ids 等）
    组装 prompt 并调用 LLM 生成题目。
    """

    async def generate(
        self,
        batch: Any,
        model_client: Any,
        evidence_pool: Any,
        document_summary: str,
        language: str,
        llm_trace_path: str,
    ) -> tuple[Any, Any, str]:
        """调用 LLM 生成题目。

        Returns:
            (raw_items, metadata, llm_call_id) 三元组
        """
        template_id = getattr(batch, "prompt_template_id", "qa_generation_v1")
        system_prompt, user_template = _load_prompt(template_id)

        user_prompt = _render_user_prompt(
            user_template,
            batch,
            evidence_pool,
            document_summary,
            getattr(batch, "additional_instructions", ""),
        )

        messages = [{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        model_name = getattr(model_client, "model_name", "gpt-4o-mini")

        try:
            response = await model_client.complete(
                model=model_name,
                messages=messages,
                temperature=getattr(model_client, "temperature", 0.7),
                max_tokens=getattr(model_client, "max_tokens", 2000),
                llm_trace_path=llm_trace_path,
            )
            raw_text = response.get("text", "")
            llm_call_id = response.get("llm_call_id", "")
            logger.debug(
                f"Generator: mode={batch.target_mode} "
                f"difficulty={batch.target_difficulty} "
                f"response_len={len(raw_text)}"
            )
        except Exception as e:
            logger.error(f"Generator LLM call failed: {e}")
            raise

        return raw_text, None, llm_call_id
