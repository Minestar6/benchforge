"""流水线共享状态读写模块。

流水线模式下 agent 通过 shared_state.json 传递上下文：
- qa_agent 创建初始版本
- verify_agent / model_eval_agent 加载并回写自身产物
"""

import json
from pathlib import Path
from typing import Any

from benchforge.schemas import SharedState, AgentStatus


# ─── 加载 ────────────────────────────────────────────────────────

def load_shared_state(path: str | Path) -> SharedState:
    """从 shared_state.json 加载流水线共享状态。

    Args:
        path: shared_state.json 文件路径

    Returns:
        SharedState 模型实例
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return SharedState(**data)


# ─── 构建 ────────────────────────────────────────────────────────

def build_shared_state(
    task_id: str,
    run_id: str,
    blueprint: Any,
    base_dir: str | Path | None = None,
    round_id: int | None = None,
    round_spec_ref: str | None = None,
) -> SharedState:
    """从 qa_agent blueprint 构建初始 SharedState。

    Args:
        task_id: 任务 ID
        run_id: 运行 ID
        blueprint: qa_agent Blueprint 对象（dataclass）
        base_dir: 运行根目录，默认 runs/{task_id}/{run_id}

    Returns:
        初始 SharedState
    """
    from dataclasses import asdict

    base = Path(base_dir) if base_dir else Path("runs") / task_id / run_id

    artifacts: dict[str, str] = {}
    for mode in getattr(blueprint, "modes", []):
        artifacts[f"{mode}_candidate_pool"] = str(base / mode / "candidate_pool.json")

    artifacts.update({
        "chunked_evidence": str(base / "evidence" / "chunked.jsonl"),
        "llm_calls": str(base / "llm_calls.jsonl"),
        "generation_report": str(base / "generation_report.json"),
    })

    return SharedState(
        task_id=task_id,
        run_id=run_id,
        round_id=round_id,
        round_spec_ref=round_spec_ref,
        blueprint=asdict(blueprint),
        blueprint_cache=asdict(blueprint),
        artifacts=artifacts,
        agent_status={
            "generation": AgentStatus.COMPLETED,
            "verification": AgentStatus.PENDING,
            "evaluation": AgentStatus.PENDING,
        },
    )


# ─── 保存 ────────────────────────────────────────────────────────

def save_shared_state(state: SharedState, base_dir: str | Path) -> Path:
    """将 SharedState 写入 base_dir/shared_state.json。

    Args:
        state: SharedState 模型实例
        base_dir: 运行根目录（如 runs/{task_id}/{run_id}）

    Returns:
        写入的文件路径
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "shared_state.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, ensure_ascii=False, indent=2)
    return path


# ─── 回写工具 ────────────────────────────────────────────────────

def update_and_save(
    path: str | Path,
    agent: str,
    artifacts: dict[str, str] | None = None,
    status: AgentStatus = AgentStatus.COMPLETED,
) -> SharedState:
    """加载 → 更新 agent 状态和产物 → 原地回写。

    供 verify_agent / model_eval_agent 完成后调用。

    Args:
        path: shared_state.json 路径
        agent: 当前 agent 名称（如 "verification"）
        artifacts: 当前 agent 产出的文件映射
        status: 当前 agent 的结束状态

    Returns:
        更新后的 SharedState
    """
    state = load_shared_state(path)
    state.set_agent_status(agent, status)
    if artifacts:
        for key, value in artifacts.items():
            state.add_artifact(key, value)
    save_shared_state(state, Path(path).parent)
    return state
