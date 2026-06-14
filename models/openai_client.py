"""OpenAI ?? API ????"""

import time
from typing import Any

import openai

from benchforge.models.base import BaseModelClient


class OpenAIClient(BaseModelClient):
    """OpenAI ?? API ????"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """?????"""
        llm_trace_path = kwargs.pop("llm_trace_path", None)
        llm_call_id = self._new_llm_call_id()
        request_record = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
        }
        start_time = time.time()

        # ?????? Span???? request ??
        try:
            from benchforge.utils.llm_tracer import current_span
            _span = current_span()
            if _span is not None:
                _span.request = request_record
                _span.model = model
                _span.generation_params = {"temperature": temperature, "max_tokens": max_tokens}
        except Exception:
            pass

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            self._record_llm_call(
                llm_trace_path=llm_trace_path,
                llm_call_id=llm_call_id,
                request=request_record,
                response=None,
                error=str(exc),
            )
            raise

        latency = time.time() - start_time
        text = response.choices[0].message.content or ""
        raw = response.model_dump()

        result = {
            "text": text,
            "input_tokens": raw.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": raw.get("usage", {}).get("completion_tokens", 0),
            "latency": latency,
            "raw": raw,
            "llm_call_id": llm_call_id,
        }
        self._record_llm_call(
            llm_trace_path=llm_trace_path,
            llm_call_id=llm_call_id,
            request=request_record,
            response=result,
            error=None,
        )

        # ?? Span ? response
        try:
            from benchforge.utils.llm_tracer import current_span
            _span = current_span()
            if _span is not None:
                _span.response = result
        except Exception:
            pass

        return result

    async def batch_complete(
        self,
        model: str,
        messages_list: list[list[dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        concurrency: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """???????"""
        import asyncio

        semaphore = asyncio.Semaphore(concurrency)

        async def single_call(messages):
            async with semaphore:
                return await self.complete(model, messages, temperature, max_tokens, **kwargs)

        return await asyncio.gather(*[single_call(msgs) for msgs in messages_list])
