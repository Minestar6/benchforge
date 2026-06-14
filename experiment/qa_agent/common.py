"""共用工具：run_id 构造、blueprint 构造、metadata 保存。"""

from pathlib import Path
import json

from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg


TASK_ID = "qa_ablation"
LANGUAGE = "en"

SMOKE_TOPICS = [
    "Climate Change",
    "Artificial Intelligence",
]

FORMAL_TOPICS = [
    "Climate Change",
    "Artificial Intelligence",
    "Quantum Computing",
    "Renewable Energy",
    "Human Immune System",
    "World War II",
]

DIFFICULTY_DISTRIBUTION = {
    "easy": 0.2,
    "medium": 0.5,
    "hard": 0.3,
}


def make_run_id(group_id: str, method: str) -> str:
    """构造 run_id，只与方案名有关。"""
    return f"{group_id}_{method}"


def build_blueprint(
    group_id: str,
    method: str,
    seed: int,
    topics: list[str],
    task_id: str = TASK_ID,
    language: str = LANGUAGE,
    mode: str = "qa",
    count: int = 50,
    max_rounds: int = 10,
    difficulty_distribution: dict[str, float] | None = None,
) -> Blueprint:
    if difficulty_distribution is None:
        difficulty_distribution = DIFFICULTY_DISTRIBUTION

    run_id = make_run_id(group_id=group_id, method=method)

    return Blueprint(
        task_id=task_id,
        run_id=run_id,
        language=language,
        topics=topics,
        modes={
            mode: ModeCfg(
                count=count,
                max_rounds=max_rounds,
                difficulty_distribution=difficulty_distribution,
            )
        },
    )


def save_metadata(
    output_dir: Path,
    *,
    group_id: str,
    method: str,
    seed: int,
    blueprint: Blueprint,
    config_path: str,
    model_name: str,
    extra: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_name, mode_cfg = next(iter(blueprint.modes.items()))

    metadata = {
        "group_id": group_id,
        "method": method,
        "seed": seed,
        "run_id": blueprint.run_id,
        "task_id": blueprint.task_id,
        "language": blueprint.language,
        "topics": blueprint.topics,
        "mode": mode_name,
        "count": mode_cfg.count,
        "max_rounds": mode_cfg.max_rounds,
        "difficulty_distribution": mode_cfg.difficulty_distribution,
        "config_path": config_path,
        "model_name": model_name,
    }

    if extra:
        metadata.update(extra)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
