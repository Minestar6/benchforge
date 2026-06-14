下面是可以直接交给 Claude Code 执行的**实验实现方案**。核心原则是：**Blueprint 控制任务目标；配置文件控制实验组差异；每个实验组、每个 seed、每个 mode 都有唯一 run_id。**

---

# BenchForge QA-Agent 消融实验实现方案

## 0. 当前代码基础判断

当前 `Blueprint` 已经包含：

```python
task_id
run_id
language
topics
modes
```

其中 `ModeCfg` 包含：

```python
count
max_rounds
difficulty_distribution
```

这些字段已经足够表达实验任务目标，不需要额外设计复杂蓝图。([GitHub][1])

当前 `config_loader.py` 明确说明：`Blueprint` 不从 yaml 加载，而是由调用方程序化构造后传入 `run_generation_agent(blueprint=...)`。([GitHub][2])

所以本实验应该遵循：

```text
Blueprint = 实验目标
qa_agent.yaml = 方法配置
experiment config = 消融开关
```

---

# 1. 最终实验组设计

保留四组主实验：

| Group | 名称                               | 目的                      |
| ----- | -------------------------------- | ----------------------- |
| A     | Direct Generation                | 不使用 qa_agent 主循环的直接生成基线 |
| B     | Multi-round No Feedback          | 控制“多轮生成本身”的收益           |
| C     | Feedback No Difficulty Evolution | 验证反馈调度的收益               |
| D     | Full Method                      | 验证完整方法，包含难度进化           |

不要继续把 B 设计为 single-round。当前 `run_all.py` 仍然使用 `group_b_single`，这会导致 B 和 C 同时差异为“轮次数 + 反馈机制”，实验变量不干净。([GitHub][3])

---

# 2. run_id 设计

run_id 只和方案名有关，格式：

```text
{group_id}_{method}
```

示例：

```text
A_direct
B_no_feedback
C_feedback_no_diff
D_full
```

工具函数：

```python
def make_run_id(
    group_id: str,
    method: str,
) -> str:
    return f"{group_id}_{method}"
```

---

# 3. Blueprint 设计

## 3.1 Smoke Test Blueprint

用于快速验证代码能跑通：

```python
TOPICS = [
    "Climate Change",
    "Artificial Intelligence",
]

MODES = {
    "qa": ModeCfg(
        count=20,
        max_rounds=5,
        difficulty_distribution={
            "easy": 0.2,
            "medium": 0.5,
            "hard": 0.3,
        },
    )
}
```

## 3.2 Formal Experiment Blueprint

用于正式实验：

```python
TOPICS = [
    "Climate Change",
    "Artificial Intelligence",
    "Quantum Computing",
    "Renewable Energy",
    "Human Immune System",
    "World War II",
]

MODES = {
    "qa": ModeCfg(
        count=50,
        max_rounds=10,
        difficulty_distribution={
            "easy": 0.2,
            "medium": 0.5,
            "hard": 0.3,
        },
    )
}
```

## 3.3 各组 Blueprint 规则

A/B/C/D 的以下字段必须一致：

```text
task_id
language
topics
mode name
count
difficulty_distribution
```

B/C/D 的 `max_rounds` 也应一致：

```python
max_rounds = 10
```

A 组虽然不使用 qa_agent 主循环，但也应该构造相同 Blueprint，用于记录 metadata 和控制目标问题数量。

---

# 4. 需要新增的配置结构

当前 `AgentConfig` 没有实验控制字段，只包含 candidate_pool、initial_breadth、planner、chunk_mix、generation_yield、chunk_limits、runtime、decision 等。([GitHub][1])

需要在：

```text
benchforge/agents/qa_agent/schema.py
```

新增：

```python
@dataclass
class ExperimentConfig:
    name: str = "full"

    # 是否允许根据上一轮反馈改变策略
    enable_feedback: bool = True

    # 是否允许 HARD_GENERATE 策略
    enable_hard_generate: bool = True

    # 是否允许根据难度缺口主动调节 difficulty
    enable_difficulty_adaptation: bool = True

    # 用于 B 组固定策略
    fixed_strategy: str | None = None

    # 用于 B 组固定难度
    fixed_difficulty: str | None = None

    # 是否禁用 initial breadth
    disable_initial_breadth: bool = False
```

然后修改 `AgentConfig`：

```python
@dataclass
class AgentConfig:
    candidate_pool: CandidatePoolConfig
    initial_breadth: InitialBreadthConfig
    planner: PlannerConfig
    chunk_mix: ChunkMixConfig
    generation_yield: dict[str, GenerationYield]
    chunk_limits: dict[str, ChunkLimitsForMode]
    runtime: RuntimeConfig
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
```

---

# 5. 修改 config_loader.py

当前 `config_loader.py` 只加载现有 agent 行为配置，没有加载 `experiment` 字段。([GitHub][2])

需要新增 import：

```python
from benchforge.agents.qa_agent.schema import ExperimentConfig
```

然后在构造 `AgentConfig` 时加入：

```python
experiment=ExperimentConfig(**raw.get("experiment", {})),
```

最终结构：

```python
agent_config = AgentConfig(
    candidate_pool=CandidatePoolConfig(**raw["candidate_pool"]),
    initial_breadth=InitialBreadthConfig(**raw["initial_breadth"]),
    planner=PlannerConfig(**raw["planner"]),
    chunk_mix=ChunkMixConfig(...),
    generation_yield={...},
    chunk_limits={...},
    runtime=RuntimeConfig(**raw["runtime"]),
    decision=DecisionConfig(**raw["decision"]) if "decision" in raw else DecisionConfig(),
    experiment=ExperimentConfig(**raw.get("experiment", {})),
)
```

---

# 6. 修改 planner.py

当前 `planner.py` 的 `select_strategy()` 会根据 `hard_gap_val` 或 `too_easy` 直接触发 `HARD_GENERATE`。([GitHub][4])

这会导致 C 组“无难度进化”名义上禁用，但实际仍可能进入 hard generation。因此必须在 planner 层加 gating。

## 6.1 修改 select_strategy()

在函数开头加入：

```python
exp_cfg = getattr(config, "experiment", None)
```

### B 组：无反馈

如果 `enable_feedback=False`，直接返回固定策略：

```python
if exp_cfg and not exp_cfg.enable_feedback:
    fixed = getattr(exp_cfg, "fixed_strategy", None)

    if fixed:
        try:
            return RoundStrategy(fixed), "experiment_fixed_strategy_no_feedback"
        except ValueError:
            return RoundStrategy.NORMAL_GENERATE, f"invalid_fixed_strategy={fixed}; fallback_normal"

    return RoundStrategy.NORMAL_GENERATE, "experiment_no_feedback_normal"
```

### C 组：有反馈但无难度进化

在 hard generate 判断处改成：

```python
hard_enabled = True
if exp_cfg is not None:
    hard_enabled = getattr(exp_cfg, "enable_hard_generate", True)

if hard_enabled and (hard_gap_val > hard_gap_threshold or too_easy > too_easy_threshold):
    return RoundStrategy.HARD_GENERATE, (
        f"hard_gap={hard_gap_val:.2f}, too_easy={too_easy:.2f}; "
        "use direct hard generation"
    )
```

然后在低通过率部分禁用 `FOCUS_DIFFICULTY`：

```python
difficulty_adaptation_enabled = True
if exp_cfg is not None:
    difficulty_adaptation_enabled = getattr(exp_cfg, "enable_difficulty_adaptation", True)

if acc_rate < accept_rate_threshold and len(mode_state.candidate_questions) > 0:
    if missing:
        return RoundStrategy.FOCUS_TOPIC, (
            f"low_accept_rate={acc_rate:.2f}, missing_topics={len(missing)}"
        )

    if difficulty_adaptation_enabled:
        return RoundStrategy.FOCUS_DIFFICULTY, f"low_accept_rate={acc_rate:.2f}"

    return RoundStrategy.NORMAL_GENERATE, (
        f"low_accept_rate={acc_rate:.2f}, difficulty_adaptation_disabled"
    )
```

最后在 difficulty gap 判断处也加开关：

```python
if difficulty_adaptation_enabled and acc_total > 0:
    diff_gaps = {
        d: mode_cfg.difficulty_distribution.get(d, 0) - acc_diff.get(d, 0) / acc_total
        for d in mode_cfg.difficulty_distribution
    }
    if max(diff_gaps.values()) > 0.1:
        return RoundStrategy.FOCUS_DIFFICULTY, f"difficulty_gap={diff_gaps}"
```

---

## 6.2 修改 build_adaptive_plan()

当前 `build_adaptive_plan()` 只要 strategy 是 `HARD_GENERATE`，就强制：

```python
difficulty="hard"
```

这部分可以保留，但必须保证 C 组不会从 `select_strategy()` 返回 `HARD_GENERATE`。

同时需要处理 B 组 fixed difficulty：

```python
exp_cfg = getattr(config, "experiment", None)
fixed_difficulty = getattr(exp_cfg, "fixed_difficulty", None) if exp_cfg else None

if fixed_difficulty:
    difficulty = fixed_difficulty
else:
    difficulty = choose_difficulty_for_mode(mode_cfg, mode_state)
```

也就是说，将原来的：

```python
difficulty = choose_difficulty_for_mode(mode_cfg, mode_state)
```

替换为：

```python
exp_cfg = getattr(config, "experiment", None)
fixed_difficulty = getattr(exp_cfg, "fixed_difficulty", None) if exp_cfg else None

difficulty = fixed_difficulty or choose_difficulty_for_mode(mode_cfg, mode_state)
```

---

## 6.3 修改 build_mode_round_plan()

当前 `build_mode_round_plan()` 会在 `initial_breadth.enabled=True` 且还有未覆盖 topic 时优先进入 initial breadth。([GitHub][4])

B 组无反馈实验应尽量关闭 initial breadth，否则仍然带有覆盖导向策略。

修改为：

```python
exp_cfg = getattr(config, "experiment", None)
disable_initial_breadth = (
    getattr(exp_cfg, "disable_initial_breadth", False)
    if exp_cfg is not None
    else False
)

if (
    not disable_initial_breadth
    and config.initial_breadth.enabled
    and mode_initial_breadth_not_done(mode_state, blueprint)
):
    return build_initial_breadth_plan(mode, mode_cfg, blueprint, config, mode_state)
```

---

# 7. 四组配置文件

建议新增目录：

```text
experiment/qa_agent/configs/
```

从当前默认 `qa_agent.yaml` 拷贝四份：

```text
qa_agent_a_direct.yaml
qa_agent_b_no_feedback.yaml
qa_agent_c_feedback_no_difficulty.yaml
qa_agent_d_full.yaml
```

## 7.1 A 组配置

A 不走 qa_agent 主循环，配置可以单独用于 direct baseline：

```yaml
experiment:
  name: direct_generation
  enable_feedback: false
  enable_hard_generate: false
  enable_difficulty_adaptation: false
  fixed_strategy: null
  fixed_difficulty: null
  disable_initial_breadth: true

direct_generation:
  max_chunks_per_topic: 6
  prompt_path: benchforge/prompts/qa_agent/direct_qa_generation.md
```

## 7.2 B 组配置

```yaml
experiment:
  name: multi_round_no_feedback
  enable_feedback: false
  enable_hard_generate: false
  enable_difficulty_adaptation: false
  fixed_strategy: normal_generate
  fixed_difficulty: medium
  disable_initial_breadth: true
```

B 组关键点：

```text
多轮
固定 normal_generate
固定 medium
不使用上一轮反馈
不 hard_generate
不 initial_breadth
```

## 7.3 C 组配置

```yaml
experiment:
  name: feedback_no_difficulty
  enable_feedback: true
  enable_hard_generate: false
  enable_difficulty_adaptation: false
  fixed_strategy: null
  fixed_difficulty: null
  disable_initial_breadth: false
```

C 组允许：

```text
NORMAL_GENERATE
FOCUS_TOPIC
EXPAND_EVIDENCE
INITIAL_BREADTH
```

C 组禁止：

```text
HARD_GENERATE
FOCUS_DIFFICULTY
强制 difficulty=hard
```

## 7.4 D 组配置

```yaml
experiment:
  name: full
  enable_feedback: true
  enable_hard_generate: true
  enable_difficulty_adaptation: true
  fixed_strategy: null
  fixed_difficulty: null
  disable_initial_breadth: false
```

D 组保留完整方法。

---

# 8. 实验脚本重构

当前 `run_all.py` 仍然调用：

```python
group_a_direct
group_b_single
group_c_nodiff
group_d_full
```

并且只设置固定 `TOPICS`、`QA_COUNT`、`DIFFICULTY_DISTRIBUTION`。([GitHub][3])

建议改为：

```text
experiment/qa_agent/
  run_all.py
  common.py
  group_a_direct.py
  group_b_no_feedback.py
  group_c_feedback_no_difficulty.py
  group_d_full.py
  direct_generation.py
  configs/
    qa_agent_a_direct.yaml
    qa_agent_b_no_feedback.yaml
    qa_agent_c_feedback_no_difficulty.yaml
    qa_agent_d_full.yaml
```

---

# 9. common.py

新增：

```python
from pathlib import Path
import json
import random

from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg


TASK_ID = "exp_qa_ablation"
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


def make_run_id(
    group_id: str,
    method: str,
) -> str:
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

    run_id = make_run_id(
        group_id=group_id,
        method=method,
    )

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
```

---

# 10. A 组：Direct Generation

A 组不要调用 `run_generation_agent()`。

新增：

```text
experiment/qa_agent/direct_generation.py
```

伪代码：

```python
async def run_direct_generation(
    *,
    blueprint,
    model_client,
    output_dir,
    config_path,
):
    """
    Direct baseline:
    - 使用相同 topics
    - 使用相同 count
    - 使用相同 retrieval/chunking 配置
    - 不使用 planner
    - 不使用 candidate_pool
    - 不使用 round feedback
    - 不使用 hard_generate
    """

    # 1. 读取 qa_agent_a_direct.yaml
    # 2. 准备 evidence
    # 3. 每个 topic 取固定数量 chunks
    # 4. 拼接 context
    # 5. 单次或按 topic 调用 direct prompt
    # 6. 保存 direct_questions.json
```

`group_a_direct.py`：

```python
from pathlib import Path

from common import build_blueprint, save_metadata
from direct_generation import run_direct_generation


GROUP_ID = "A"
METHOD = "direct"
CONFIG_PATH = "experiment/qa_agent/configs/qa_agent_a_direct.yaml"


async def run(
    *,
    model_client,
    model_name,
    seed,
    topics,
    task_id,
    language,
    qa_count,
    difficulty_distribution,
):
    blueprint = build_blueprint(
        group_id=GROUP_ID,
        method=METHOD,
        seed=seed,
        topics=topics,
        task_id=task_id,
        language=language,
        mode="qa",
        count=qa_count,
        max_rounds=1,
        difficulty_distribution=difficulty_distribution,
    )

    output_dir = Path("runs/qa_ablation") / blueprint.run_id

    save_metadata(
        output_dir,
        group_id=GROUP_ID,
        method=METHOD,
        seed=seed,
        blueprint=blueprint,
        config_path=CONFIG_PATH,
        model_name=model_name,
        extra={
            "enable_feedback": False,
            "enable_hard_generate": False,
            "enable_difficulty_adaptation": False,
            "is_direct_baseline": True,
        },
    )

    await run_direct_generation(
        blueprint=blueprint,
        model_client=model_client,
        output_dir=output_dir,
        config_path=CONFIG_PATH,
    )
```

---

# 11. B 组：Multi-round No Feedback

新增：

```text
experiment/qa_agent/group_b_no_feedback.py
```

```python
from pathlib import Path

from benchforge.agents.qa_agent.agent import run_generation_agent
from common import build_blueprint, save_metadata


GROUP_ID = "B"
METHOD = "no_feedback"
CONFIG_PATH = "experiment/qa_agent/configs/qa_agent_b_no_feedback.yaml"


async def run(
    *,
    model_client,
    model_name,
    seed,
    topics,
    task_id,
    language,
    qa_count,
    difficulty_distribution,
):
    blueprint = build_blueprint(
        group_id=GROUP_ID,
        method=METHOD,
        seed=seed,
        topics=topics,
        task_id=task_id,
        language=language,
        mode="qa",
        count=qa_count,
        max_rounds=10,
        difficulty_distribution=difficulty_distribution,
    )

    output_dir = Path("runs/qa_ablation") / blueprint.run_id

    save_metadata(
        output_dir,
        group_id=GROUP_ID,
        method=METHOD,
        seed=seed,
        blueprint=blueprint,
        config_path=CONFIG_PATH,
        model_name=model_name,
        extra={
            "enable_feedback": False,
            "enable_hard_generate": False,
            "enable_difficulty_adaptation": False,
            "fixed_strategy": "normal_generate",
            "fixed_difficulty": "medium",
            "disable_initial_breadth": True,
        },
    )

    await run_generation_agent(
        blueprint=blueprint,
        model_client=model_client,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
    )
```

注意：如果当前 `run_generation_agent()` 参数名不是 `config_path` / `output_dir`，Claude Code 需要按当前函数签名适配，但目标逻辑保持不变。

---

# 12. C 组：Feedback No Difficulty Evolution

新增：

```text
experiment/qa_agent/group_c_feedback_no_difficulty.py
```

```python
from pathlib import Path

from benchforge.agents.qa_agent.agent import run_generation_agent
from common import build_blueprint, save_metadata


GROUP_ID = "C"
METHOD = "feedback_no_diff"
CONFIG_PATH = "experiment/qa_agent/configs/qa_agent_c_feedback_no_difficulty.yaml"


async def run(
    *,
    model_client,
    model_name,
    seed,
    topics,
    task_id,
    language,
    qa_count,
    difficulty_distribution,
):
    blueprint = build_blueprint(
        group_id=GROUP_ID,
        method=METHOD,
        seed=seed,
        topics=topics,
        task_id=task_id,
        language=language,
        mode="qa",
        count=qa_count,
        max_rounds=10,
        difficulty_distribution=difficulty_distribution,
    )

    output_dir = Path("runs/qa_ablation") / blueprint.run_id

    save_metadata(
        output_dir,
        group_id=GROUP_ID,
        method=METHOD,
        seed=seed,
        blueprint=blueprint,
        config_path=CONFIG_PATH,
        model_name=model_name,
        extra={
            "enable_feedback": True,
            "enable_hard_generate": False,
            "enable_difficulty_adaptation": False,
            "disable_initial_breadth": False,
        },
    )

    await run_generation_agent(
        blueprint=blueprint,
        model_client=model_client,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
    )
```

---

# 13. D 组：Full Method

修改或新增：

```text
experiment/qa_agent/group_d_full.py
```

```python
from pathlib import Path

from benchforge.agents.qa_agent.agent import run_generation_agent
from common import build_blueprint, save_metadata


GROUP_ID = "D"
METHOD = "full"
CONFIG_PATH = "experiment/qa_agent/configs/qa_agent_d_full.yaml"


async def run(
    *,
    model_client,
    model_name,
    seed,
    topics,
    task_id,
    language,
    qa_count,
    difficulty_distribution,
):
    blueprint = build_blueprint(
        group_id=GROUP_ID,
        method=METHOD,
        seed=seed,
        topics=topics,
        task_id=task_id,
        language=language,
        mode="qa",
        count=qa_count,
        max_rounds=10,
        difficulty_distribution=difficulty_distribution,
    )

    output_dir = Path("runs/qa_ablation") / blueprint.run_id

    save_metadata(
        output_dir,
        group_id=GROUP_ID,
        method=METHOD,
        seed=seed,
        blueprint=blueprint,
        config_path=CONFIG_PATH,
        model_name=model_name,
        extra={
            "enable_feedback": True,
            "enable_hard_generate": True,
            "enable_difficulty_adaptation": True,
            "disable_initial_breadth": False,
        },
    )

    await run_generation_agent(
        blueprint=blueprint,
        model_client=model_client,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
    )
```

---

# 14. run_all.py 重构

当前 `run_all.py` 只跑四个 group，没有 seed 循环。([GitHub][3])

建议改成：

```python
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from benchforge.config.config import load_dotenv
from benchforge.models.openai_client import OpenAIClient

import group_a_direct
import group_b_no_feedback
import group_c_feedback_no_difficulty
import group_d_full


PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v3-2-251201")

TASK_ID = "exp_qa_ablation"
LANGUAGE = "en"

TOPICS = [
    "Climate Change",
    "Artificial Intelligence",
    "Quantum Computing",
    "Renewable Energy",
    "Human Immune System",
    "World War II",
]

QA_COUNT = int(os.getenv("QA_COUNT", "50"))
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "10"))

DIFFICULTY_DISTRIBUTION = {
    "easy": 0.2,
    "medium": 0.5,
    "hard": 0.3,
}

SEEDS = [42, 43, 44, 45, 46]

GROUPS = [
    ("A - Direct Generation", group_a_direct.run),
    ("B - Multi-round No Feedback", group_b_no_feedback.run),
    ("C - Feedback No Difficulty", group_c_feedback_no_difficulty.run),
    ("D - Full Method", group_d_full.run),
]


async def main():
    if not MODEL_API_KEY:
        print(f"[run_all] ERROR: MODEL_API_KEY not set. Expected .env at: {PROJECT_ROOT / '.env'}")
        sys.exit(1)

    model_client = OpenAIClient(
        api_key=MODEL_API_KEY,
        base_url=MODEL_BASE_URL,
        model_name=MODEL_NAME,
    )

    print(f"[run_all] Model: {MODEL_NAME}")
    print(f"[run_all] Topics: {TOPICS}")
    print(f"[run_all] Seeds: {SEEDS}")

    for seed in SEEDS:
        for name, run_fn in GROUPS:
            print(f"\n{'=' * 80}")
            print(f"[run_all] Starting {name}, seed={seed}")
            print(f"{'=' * 80}")

            try:
                await run_fn(
                    model_client=model_client,
                    model_name=MODEL_NAME,
                    seed=seed,
                    topics=TOPICS,
                    task_id=TASK_ID,
                    language=LANGUAGE,
                    qa_count=QA_COUNT,
                    difficulty_distribution=DIFFICULTY_DISTRIBUTION,
                )
                print(f"[run_all] Finished {name}, seed={seed}")

            except Exception as e:
                import traceback
                print(f"[run_all] FAILED: {name}, seed={seed}, error={e}")
                traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("[run_all] All experiments completed.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
```

run_id 只标识方案，不同 seed 共用同一个 run_id，seed 信息在 metadata.json 中记录：

```text
A_direct
B_no_feedback
C_feedback_no_diff
D_full
```

---

# 15. 输出目录规范

每次运行输出到：

```text
runs/qa_ablation/{run_id}/
```

例如：

```text
runs/qa_ablation/
  A_direct/
    metadata.json
    direct_questions.json
    llm_calls.jsonl

  B_no_feedback/
    metadata.json
    candidate_pool.jsonl
    round_summary.jsonl
    llm_calls.jsonl

  C_feedback_no_diff/
    metadata.json
    candidate_pool.jsonl
    round_summary.jsonl
    llm_calls.jsonl

  D_full/
    metadata.json
    candidate_pool.jsonl
    round_summary.jsonl
    llm_calls.jsonl
```

---

# 16. 必须保存的 metadata 字段

每组都保存：

```json
{
  "group_id": "C",
  "method": "feedback_no_diff",
  "seed": 42,
  "run_id": "C_feedback_no_diff",
  "task_id": "exp_qa_ablation",
  "language": "en",
  "topics": [
    "Climate Change",
    "Artificial Intelligence",
    "Quantum Computing",
    "Renewable Energy",
    "Human Immune System",
    "World War II"
  ],
  "mode": "qa",
  "count": 50,
  "max_rounds": 10,
  "difficulty_distribution": {
    "easy": 0.2,
    "medium": 0.5,
    "hard": 0.3
  },
  "enable_feedback": true,
  "enable_hard_generate": false,
  "enable_difficulty_adaptation": false,
  "disable_initial_breadth": false,
  "config_path": "experiment/qa_agent/configs/qa_agent_c_feedback_no_difficulty.yaml",
  "model_name": "deepseek-v3-2-251201"
}
```

---

# 17. 验证实验开关是否生效

Claude Code 需要添加一个检查脚本：

```text
experiment/qa_agent/check_ablation_integrity.py
```

检查逻辑：

## B 组

断言：

```text
strategy 不应出现 HARD_GENERATE
strategy 不应出现 FOCUS_TOPIC
strategy 不应出现 FOCUS_DIFFICULTY
strategy 不应出现 EXPAND_EVIDENCE
strategy 应主要是 NORMAL_GENERATE
difficulty 应固定为 medium
```

## C 组

断言：

```text
strategy 不应出现 HARD_GENERATE
strategy 不应出现 FOCUS_DIFFICULTY
可以出现 INITIAL_BREADTH
可以出现 FOCUS_TOPIC
可以出现 EXPAND_EVIDENCE
可以出现 NORMAL_GENERATE
```

## D 组

允许：

```text
HARD_GENERATE
FOCUS_DIFFICULTY
FOCUS_TOPIC
EXPAND_EVIDENCE
NORMAL_GENERATE
INITIAL_BREADTH
```

如果 C 组日志里出现 `HARD_GENERATE`，说明消融失败。

---

# 18. 最终统计脚本

新增：

```text
experiment/qa_agent/aggregate_ablation_results.py
```

读取：

```text
runs/qa_ablation/*/metadata.json
runs/qa_ablation/*/candidate_pool.jsonl
runs/qa_ablation/*/round_summary.jsonl
runs/qa_ablation/*/llm_calls.jsonl
```

输出：

```text
runs/qa_ablation/summary/
  ablation_summary.csv
  ablation_summary.md
```

至少统计：

```text
accepted_count
generated_count
accept_rate
easy_ratio
medium_ratio
hard_ratio
multi_chunk_ratio
duplicate_ratio
total_llm_calls
total_prompt_tokens
total_completion_tokens
total_tokens
tokens_per_accepted_question
strategy_counts
```

对比：

```text
B - A = 多轮收益
C - B = 反馈收益
D - C = 难度进化收益
```

---

# 19. Claude Code 执行顺序

建议按下面顺序实现：

```text
1. 修改 schema.py，新增 ExperimentConfig
2. 修改 config_loader.py，加载 experiment 配置
3. 修改 planner.py，实现 enable_feedback / enable_hard_generate / enable_difficulty_adaptation gating
4. 新增 experiment/qa_agent/configs/*.yaml
5. 新增 experiment/qa_agent/common.py
6. 重写 group_a_direct.py，使其不调用 run_generation_agent
7. 新增 direct_generation.py
8. 新增 group_b_no_feedback.py
9. 新增 group_c_feedback_no_difficulty.py
10. 修改 group_d_full.py
11. 重写 run_all.py，支持 seeds 和统一 run_id
12. 新增 check_ablation_integrity.py
13. 新增 aggregate_ablation_results.py
14. 先用 QA_COUNT=20、SEEDS=[42] 做 smoke test
15. 再用 QA_COUNT=50、SEEDS=[42,43,44,45,46] 做正式实验
```

---

# 20. 最终判断

这套实现后，实验结论会变得清楚：

```text
A Direct
  ↓
B Multi-round No Feedback
  ↓
C Feedback No Difficulty Evolution
  ↓
D Full Method
```

对应论文论证：

```text
B > A：多轮生成有效
C > B：反馈调度有效
D > C：难度进化有效
```

重点是：**每个实验组都有自己的 run_id（即 `{group_id}_{method}`），不同 seed 共用同一个 run_id，seed 等信息在 metadata.json 中区分。**

[1]: https://raw.githubusercontent.com/Minestar6/benchforge/develop1/agents/qa_agent/schema.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/Minestar6/benchforge/develop1/agents/qa_agent/config_loader.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/Minestar6/benchforge/develop1/experiment/qa_agent/run_all.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Minestar6/benchforge/develop1/agents/qa_agent/planner.py "raw.githubusercontent.com"
