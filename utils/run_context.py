"""统一运行上下文：路径生成、seed 管理、配置快照。

所有 Agent 共享同一个 RunContext 实例，确保:
- 路径规则一致（llm_calls.jsonl 只有一个拼装点）
- seed 可复现
- 配置可追溯（run_snapshot.json）
"""

import json
import random
import secrets
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class RunContext:
    """一次完整运行的统一上下文。

    Usage:
        ctx = RunContext(task_id="my_task", seed=42)
        ctx.snapshot_dependencies()
        ctx.snapshot_config(my_config)
        ctx.save_snapshot()

        # 所有路径从这里生成
        tracer = ctx.create_tracer(agent="qa_agent")
        ctx.llm_trace_path  # → runs/my_task/run_20260606_120000_abcd/llm_calls.jsonl
    """

    task_id: str
    run_id: str = ""
    base_dir: Path = Path("runs")
    seed: int | None = None

    # 配置快照（运行时自动填充）
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    dependency_versions: dict[str, str] = field(default_factory=dict)

    _rng: random.Random | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        if not self.run_id:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = secrets.token_hex(2)
            self.run_id = f"run_{ts}_{suffix}"
        if self.seed is not None:
            self._rng = random.Random(self.seed)
        else:
            self._rng = random.Random()

    # ─── 路径生成（唯一的路径拼装点） ───

    @property
    def run_dir(self) -> Path:
        """运行根目录: runs/{task_id}/{run_id}/"""
        return self.base_dir / self.task_id / self.run_id

    @property
    def llm_trace_path(self) -> str:
        """全局 LLM trace 文件路径。所有 Agent 共享此文件，靠 agent 字段区分。"""
        return str(self.run_dir / "llm_calls.jsonl")

    def agent_dir(self, agent_name: str) -> Path:
        """每个 agent 独立的输出目录。"""
        return self.run_dir / agent_name

    def session_dir(self, agent_name: str, session_id: str) -> Path:
        """带 session 的子目录（如 evaluation）。"""
        return self.agent_dir(agent_name) / session_id

    # ─── Seed 管理 ───

    def get_rng(self) -> random.Random:
        """获取运行级别的随机数生成器。"""
        return self._rng

    def get_seed(self) -> int:
        return self.seed or 0

    # ─── Tracer 工厂 ───

    def create_tracer(self, agent: str, stage: str = "") -> "LLMTracer":
        """为此 RunContext 创建一个 LLMTracer。

        Args:
            agent: qa_agent | model_eval_agent | verify_agent
            stage: generation | inference | judge | validation | summarization
        """
        from benchforge.utils.llm_tracer import LLMTracer
        return LLMTracer(run_context=self, agent=agent, stage=stage)

    # ─── 可复现性快照 ───

    def snapshot_config(self, config: Any) -> None:
        """保存配置快照（dataclass / dict），用于事后复现。"""
        if hasattr(config, "__dataclass_fields__"):
            from dataclasses import asdict
            self.config_snapshot = {
                k: str(v) for k, v in asdict(config).items()
            }
        elif isinstance(config, dict):
            self.config_snapshot = {k: str(v) for k, v in config.items()}

    def snapshot_dependencies(self) -> None:
        """记录当前 Python 及关键依赖的版本。"""
        try:
            from importlib.metadata import version
            self.dependency_versions = {"python": sys.version}
            for pkg in ("openai", "httpx", "numpy", "loguru", "pyyaml"):
                try:
                    self.dependency_versions[pkg] = version(pkg)
                except Exception:
                    self.dependency_versions[pkg] = "unknown"
        except ImportError:
            self.dependency_versions = {"python": sys.version, "note": "importlib.metadata unavailable"}

    def save_snapshot(self) -> Path:
        """将可复现性快照写入 run_dir/run_snapshot.json。"""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "seed": self.seed,
            "config": self.config_snapshot,
            "dependencies": self.dependency_versions,
        }
        path = self.run_dir / "run_snapshot.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"[RunContext] snapshot saved to {path}")
        return path
