"""模型抽象基类。"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import uuid4


class BaseModelClient(ABC):
    """模型客户端抽象基类。"""

    def _new_llm_call_id(self) -> str:
        return f"llm_{uuid4().hex}"

    def _record_llm_call(
        self,
        llm_trace_path: str | None,
        llm_call_id: str,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        if not llm_trace_path:
            return

        path = Path(llm_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "llm_call_id": llm_call_id,
            "request": request,
            "response": response,
            "error": error,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """完成对话。

        Returns:
            {
                "text": str,          # 模型响应文本
                "input_tokens": int,  # 输入 token 数
                "output_tokens": int, # 输出 token 数
                "latency": float,     # 延迟（秒）
                "raw": Any,           # 原始响应
                "llm_call_id": str,   # 本次调用唯一ID
            }
        """
        pass

    @abstractmethod
    async def batch_complete(
        self,
        model: str,
        messages_list: list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """批量完成对话。"""
        pass
