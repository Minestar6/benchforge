"""本地模型客户端。

支持：
1. Ollama 本地推理服务
2. vLLM 本地推理服务
3. HuggingFace transformers 直接加载
"""

import json
import time
from typing import Any

import httpx
from benchforge.models.base import BaseModelClient


class OllamaClient(BaseModelClient):
    """Ollama 本地推理服务客户端。"""

    provider = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        """初始化 Ollama 客户端。"""
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    def _extract_provider_specific_params(self, kwargs: dict) -> dict:
        """Ollama 的参数在 options 子字典中，且命名不同。"""
        opts = kwargs.get("options", {})
        result = {}
        if "num_predict" in opts:
            result["max_tokens"] = opts["num_predict"]
        if "temperature" in opts:
            result["temperature"] = opts["temperature"]
        if "top_p" in opts:
            result["top_p"] = opts["top_p"]
        if "seed" in opts:
            result["seed"] = opts["seed"]
        if "stop" in opts:
            result["stop"] = opts["stop"]
        return result

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """完成对话。"""
        start_time = time.time()

        # 构建请求
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        payload["options"].update(kwargs.get("options", {}))

        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            error = None
        except Exception as exc:
            data = {}
            error = str(exc)

        latency = time.time() - start_time

        if error is None:
            result = {
                "text": data.get("message", {}).get("content", ""),
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
                "latency": latency,
                "raw": data,
            }
        else:
            result = {
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency": latency,
                "raw": {},
            }

        llm_call_id = self._auto_record(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            kwargs=kwargs,
            response=result if error is None else None,
            error=error,
        )
        result["llm_call_id"] = llm_call_id

        if error:
            raise RuntimeError(error)

        return result

    async def batch_complete(
        self,
        model: str,
        messages_list: list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """批量完成对话。"""
        import asyncio

        tasks = [self.complete(model, msgs, **kwargs) for msgs in messages_list]
        return await asyncio.gather(*tasks)


class VLLMClient(BaseModelClient):
    """vLLM 本地推理服务客户端（兼容 OpenAI API）。

    内部委托给 OpenAIClient，因此构造函数与 OpenAIClient 保持一致。
    """

    provider = "vllm"

    def __init__(
        self,
        api_key: str = "empty",
        base_url: str = "http://localhost:8000/v1",
    ):
        """初始化 vLLM 客户端。"""
        self.base_url = base_url
        self.api_key = api_key

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """完成对话。"""
        from .openai_client import OpenAIClient

        client = OpenAIClient(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        return await client.complete(model, messages, temperature, max_tokens, **kwargs)

    async def batch_complete(
        self,
        model: str,
        messages_list: list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """批量完成对话。"""
        from .openai_client import OpenAIClient

        client = OpenAIClient(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        return await client.batch_complete(model, messages_list, **kwargs)


class TransformersClient(BaseModelClient):
    """HuggingFace Transformers 直接加载的客户端。"""

    provider = "transformers"

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        load_in_8bit: bool = False,
    ):
        """初始化 Transformers 客户端。"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
        except ImportError:
            raise ImportError("请安装 transformers 和 torch: pip install transformers torch")

        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=self.device,
            load_in_8bit=load_in_8bit,
        )

        # 创建生成管道
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # 构建提示模板
        if "chatglm" in model_name_or_path.lower():
            self.prompt_format = "chatglm"
        elif "qwen" in model_name_or_path.lower():
            self.prompt_format = "qwen"
        else:
            self.prompt_format = "default"

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """格式化消息为提示。"""
        if self.prompt_format == "chatglm":
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"[Round 0]\n问：{msg['content']}\n答："
                elif msg["role"] == "assistant":
                    prompt += f"{msg['content']}\n"
            return prompt
        elif self.prompt_format == "qwen":
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "assistant":
                    prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt
        else:
            prompt = ""
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "Assistant:"
            return prompt

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """完成对话。"""
        start_time = time.time()

        prompt = self._format_messages(messages)

        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                return_full_text=False,
                **kwargs,
            )
            text = outputs[0]["generated_text"]
            error = None
        except Exception as exc:
            text = ""
            outputs = []
            error = str(exc)

        latency = time.time() - start_time

        # 估算 token 数
        input_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = len(self.tokenizer.encode(text))

        result = {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": latency,
            "raw": {"prompt": prompt, "outputs": outputs},
        }

        llm_call_id = self._auto_record(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            kwargs=kwargs,
            response=result if error is None else None,
            error=error,
        )
        result["llm_call_id"] = llm_call_id

        if error:
            raise RuntimeError(error)

        return result

    async def batch_complete(
        self,
        model: str,
        messages_list: list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """批量完成对话。"""
        import asyncio

        tasks = [self.complete(model, msgs, **kwargs) for msgs in messages_list]
        return await asyncio.gather(*tasks)
