"""LLM 调用追踪器：Span 上下文管理器 + TraceReader 查询接口。

核心设计:
- Span.__enter__ 注册到 contextvars，BaseModelClient 自动感知
- Span.__exit__ 自动写入 llm_calls.jsonl（包含标准化的 generation_params）
- TraceReader 提供事后查询/聚合能力

Usage:
    tracer = run_context.create_tracer(agent="qa_agent", stage="generation")
    with tracer.span(model="gpt-4o", provider="openai", tags={"topic": "ML"}) as span:
        resp = await client.complete(model="gpt-4o", messages=[...])
        # span.response 可直接取 token 数
        print(span.response["input_tokens"])
"""

import contextvars
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

# 协程安全的当前 Span 栈
_CURRENT_SPAN: contextvars.ContextVar = contextvars.ContextVar("llm_span", default=None)


def current_span() -> "Span | None":
    """获取当前协程上下文中活跃的 Span。"""
    return _CURRENT_SPAN.get()


# ─── 生成参数标准化 ──────────────────────────────────────────────────────────

# 影响可复现性的标准参数集
_REPRODUCIBLE_PARAMS = (
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "stop",
    "presence_penalty",
    "frequency_penalty",
)

_DEFAULT_PARAMS = {
    "top_p": 1.0,
    "seed": None,
    "stop": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def normalize_generation_params(
    temperature: float,
    max_tokens: int,
    kwargs: dict[str, Any],
    provider_extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从客户端特定参数中提取标准化的生成参数。

    无论底层是 OpenAI / Ollama / vLLM / Transformers，
    都输出统一的参数名和结构。
    """
    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # 从 kwargs 提取标准参数
    for key in _REPRODUCIBLE_PARAMS:
        if key in kwargs and key not in ("temperature", "max_tokens"):
            params[key] = kwargs[key]

    # provider 特定映射
    if provider_extras:
        params.update(provider_extras)

    # 填充默认值
    for k, v in _DEFAULT_PARAMS.items():
        params.setdefault(k, v)

    return params


# ─── Span ────────────────────────────────────────────────────────────────────


@dataclass
class Span:
    """单次 LLM 调用的追踪 Span。上下文管理器。

    在 __exit__ 时自动写入 llm_calls.jsonl，调用方也可通过 span.response 实时获取结果。
    """

    tracer: "LLMTracer"
    model: str
    provider: str = ""
    tags: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None

    call_id: str = field(default_factory=lambda: f"llm_{uuid4().hex}")
    span_id: str = field(default_factory=lambda: uuid4().hex[:8])

    # 由 BaseModelClient 在调用期间填充
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    error: str | None = None
    generation_params: dict[str, Any] | None = None

    _start_time: float = field(default=0.0, init=False, repr=False)

    def __enter__(self) -> "Span":
        self._start_time = time.monotonic()
        _CURRENT_SPAN.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        _CURRENT_SPAN.set(None)
        self.tracer._write(self._build_record())
        return False  # 不吞异常

    async def __aenter__(self) -> "Span":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)

    def _build_record(self) -> dict[str, Any]:
        latency_ms = int((time.monotonic() - self._start_time) * 1000)
        return {
            # 调用标识
            "call_id": self.call_id,
            "parent_id": self.parent_id,
            "span_id": self.span_id,

            # 调用方身份
            "agent": self.tracer.agent,
            "stage": self.tracer.stage or "",

            # 模型身份
            "model": self.model,
            "provider": self.provider,

            # 标准化生成参数（可复现性核心字段）
            "params": self.generation_params or {},

            # 业务标签
            "tags": self.tags,

            # 完整原始数据
            "request": self.request,
            "response": self.response,
            "error": self.error,

            "latency_ms": latency_ms,
        }

    def child_span(self, **overrides) -> "Span":
        """创建子 Span（用于嵌套 LLM 调用，如 generation → summarization）。"""
        return Span(
            tracer=self.tracer,
            model=overrides.get("model", self.model),
            provider=overrides.get("provider", self.provider),
            tags=overrides.get("tags", {}),
            parent_id=self.call_id,
        )


# ─── LLMTracer ───────────────────────────────────────────────────────────────


@dataclass
class LLMTracer:
    """LLM 调用追踪器。与 RunContext 绑定，自动管理 trace_path。

    Usage:
        tracer = run_context.create_tracer(agent="qa_agent", stage="generation")
        with tracer.span(model="gpt-4o", provider="openai", tags={"topic": "ML"}):
            ...
    """

    run_context: Any     # RunContext
    agent: str
    stage: str = ""

    _records: list[dict] = field(default_factory=list, init=False, repr=False)

    @property
    def trace_path(self) -> str:
        return self.run_context.llm_trace_path

    def span(
        self,
        model: str,
        provider: str = "",
        tags: dict[str, Any] | None = None,
        parent_id: str | None = None,
    ) -> Span:
        """创建一个调用 Span。用作上下文管理器。"""
        return Span(
            tracer=self,
            model=model,
            provider=provider,
            tags=tags or {},
            parent_id=parent_id,
        )

    def _write(self, record: dict) -> None:
        """将记录写入 llm_calls.jsonl。"""
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._records.append(record)

    def usage(self) -> dict[str, int]:
        """从当前 tracer 已记录的调用中汇总 token 使用量。

        Returns:
            {"calls": N, "input_tokens": N, "output_tokens": N, "errors": N}
        """
        calls = 0
        input_tokens = 0
        output_tokens = 0
        errors = 0
        for r in self._records:
            calls += 1
            resp = r.get("response") or {}
            input_tokens += resp.get("input_tokens", 0)
            output_tokens += resp.get("output_tokens", 0)
            if r.get("error"):
                errors += 1
        return {
            "calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "errors": errors,
        }

    @property
    def records(self) -> list[dict]:
        return list(self._records)


# ─── TraceReader ─────────────────────────────────────────────────────────────


class TraceReader:
    """llm_calls.jsonl 的事后查询接口。

    Usage:
        reader = TraceReader(run_context.llm_trace_path)
        qa_calls = reader.calls(agent="qa_agent")
        print(reader.cost_by_model())
        reader.trace_question("Q001")
    """

    def __init__(self, trace_path: str):
        self.path = Path(trace_path)
        self._cache: list[dict] | None = None

    @property
    def _records(self) -> list[dict]:
        if self._cache is None:
            self._cache = []
            if self.path.exists():
                with open(self.path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self._cache.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
        return self._cache

    def all(self) -> list[dict]:
        """返回所有调用记录。"""
        return list(self._records)

    def calls(
        self,
        agent: str | None = None,
        stage: str | None = None,
        model: str | None = None,
        has_error: bool | None = None,
    ) -> list[dict]:
        """按条件过滤调用记录。"""
        results = []
        for r in self._records:
            if agent is not None and r.get("agent") != agent:
                continue
            if stage is not None and r.get("stage") != stage:
                continue
            if model is not None and r.get("model") != model:
                continue
            if has_error is True and not r.get("error"):
                continue
            if has_error is False and r.get("error"):
                continue
            results.append(r)
        return results

    def get(self, call_id: str) -> dict | None:
        """根据 call_id 精确查找一次调用。"""
        for r in self._records:
            if r.get("call_id") == call_id:
                return r
        return None

    def cost_by_model(self) -> dict[str, dict]:
        """按模型聚合 token 消耗。"""
        cost: dict[str, dict] = {}
        for r in self._records:
            m = r.get("model", "unknown")
            resp = r.get("response") or {}
            cost.setdefault(m, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0})
            cost[m]["calls"] += 1
            cost[m]["input_tokens"] += resp.get("input_tokens", 0)
            cost[m]["output_tokens"] += resp.get("output_tokens", 0)
            if r.get("error"):
                cost[m]["errors"] += 1
        return cost

    def cost_by_agent(self) -> dict[str, dict]:
        """按 agent 聚合 token 消耗。"""
        cost: dict[str, dict] = {}
        for r in self._records:
            a = r.get("agent", "unknown")
            resp = r.get("response") or {}
            cost.setdefault(a, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0})
            cost[a]["calls"] += 1
            cost[a]["input_tokens"] += resp.get("input_tokens", 0)
            cost[a]["output_tokens"] += resp.get("output_tokens", 0)
            if r.get("error"):
                cost[a]["errors"] += 1
        return cost

    def trace_question(self, question_id: str) -> list[dict]:
        """追溯一道题目的完整 LLM 调用链。

        在所有记录的 tags 中搜索 question_id 匹配的记录。
        """
        results = []
        for r in self._records:
            tags = r.get("tags") or {}
            if tags.get("question_id") == question_id:
                results.append(r)
        return sorted(results, key=lambda r: r.get("latency_ms", 0))

    def params_diff(self, other_path: str | Path) -> list[dict]:
        """对比两次运行的生成参数差异。"""
        other = TraceReader(str(other_path))
        diffs = []
        for r1 in self._records:
            r2 = other.get(r1.get("call_id", ""))
            if r2 is None:
                diffs.append({"call_id": r1["call_id"], "status": "only_in_current"})
                continue
            p1 = r1.get("params", {})
            p2 = r2.get("params", {})
            changed = {}
            for k in set(p1) | set(p2):
                if p1.get(k) != p2.get(k):
                    changed[k] = {"old": p1.get(k), "new": p2.get(k)}
            if changed:
                diffs.append({"call_id": r1["call_id"], "status": "params_changed", "changes": changed})
        # 检查仅在 other 中的记录
        other_ids = {r.get("call_id") for r in other.all()}
        current_ids = {r.get("call_id") for r in self._records}
        for oid in other_ids - current_ids:
            diffs.append({"call_id": oid, "status": "only_in_other"})
        return diffs

    def summary(self) -> dict[str, Any]:
        """生成调用摘要。"""
        agents = set()
        models = set()
        total_calls = 0
        total_errors = 0
        for r in self._records:
            total_calls += 1
            if r.get("agent"):
                agents.add(r["agent"])
            if r.get("model"):
                models.add(r["model"])
            if r.get("error"):
                total_errors += 1
        return {
            "total_calls": total_calls,
            "total_errors": total_errors,
            "agents": sorted(agents),
            "models": sorted(models),
            "cost_by_model": self.cost_by_model(),
            "cost_by_agent": self.cost_by_agent(),
        }
