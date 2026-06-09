# BenchForge 规划智能体技术方案

## 1. 目标

规划智能体负责根据用户输入的文本描述，生成多轮进化式题目构建计划。

系统每一轮执行：

```text
Planner
  -> GeneratorPlan
  -> 题目生成智能体
  -> GeneratorFeedback

Planner
  -> ValidatorPlan
  -> 题目验证智能体
  -> ValidatorFeedback

Planner
  -> EvaluatorPlan
  -> 模型评估智能体
  -> EvaluatorFeedback

Planner 根据反馈生成下一轮 RoundPlan
```

规划智能体不直接控制底层执行参数，例如 chunk size、retrieval top-k、prompt 路径、temperature、max concurrency。
规划智能体只控制能够根据反馈动态调整的策略级参数，例如主题描述、题量、难度比例、候选池倍率、多样性等级、验证强度、评估模型池等。

---

## 2. 核心设计原则

### 2.1 总蓝图极简

总蓝图只记录用户最终目标和跨轮次不应频繁变化的信息。

总蓝图不记录每轮具体策略，不记录验证阈值、评估模型、指标列表、候选池倍率、难度比例等动态字段。

### 2.2 PlannerState 保存跨轮次记忆

跨轮次状态不放进每个 run 的 `shared_state.json`，而是单独保存为：

```text
runs/{task_id}/planner_state.json
```

PlannerState 记录当前进度、主题记忆、难度校准、模型池状态、预算状态和历史决策。

### 2.3 RoundPlan 控制本轮策略

每一轮自动生成一个 `run_id`。
每个 `run_id` 绑定一个 `RoundPlan`。

RoundPlan 是本轮执行计划，包含本轮题量、主题描述、生成策略、验证策略、评估策略和预算分配。

### 2.4 AgentPlan 保持兼容当前实现

给每个智能体的计划只包含该智能体需要的字段。
固定验收要求和返回要求放在 agent 默认模板中，不要求 planner 每轮重复生成。

### 2.5 shared_state 只保存本轮公共状态

`shared_state.json` 保存当前 run 的共享上下文和 artifact 引用。
它不保存完整 PlannerState。

---

## 3. 数据结构总览

```text
GlobalBlueprint
  用户最终目标、题型数量、语言、最大进化轮次、初始主题、终止条件

PlannerState
  跨轮次动态状态、主题记忆、模型池、预算、历史反馈

RoundPlan
  本轮题量、本轮主题、本轮生成/验证/评估策略

GeneratorPlan
  分发给题目生成智能体的计划

ValidatorPlan
  分发给题目验证智能体的计划

EvaluatorPlan
  分发给模型评估智能体的计划

GeneratorFeedback
  题目生成智能体返回的结构化反馈

ValidatorFeedback
  题目验证智能体返回的结构化反馈

EvaluatorFeedback
  模型评估智能体返回的结构化反馈
```

---

# 4. GlobalBlueprint

## 4.1 用途

GlobalBlueprint 是全局目标账本。
它只描述最终要完成什么，不描述每一轮怎么做。

## 4.2 Schema

```yaml
GlobalBlueprint:
  task_id: str
  blueprint_id: str
  user_goal: str
  language: str
  max_evolution_rounds: int

  final_targets:
    by_mode:
      qa: int
      multiple_choice: int

  seed_topics:
    - str

  termination_conditions:
    max_rounds_reached: bool
    final_targets_reached: bool
    budget_exhausted: bool
    no_significant_gain: bool

  user_required_constraints:
    require_citation: bool | null
    require_capability_label: bool | null
    allowed_question_types: list[str] | null

  user_required_evaluation:
    candidate_models: list[str] | null
    required_metrics: list[str] | null
```

## 4.3 字段说明

| 字段                                                   | 类型                  | 说明           | 用途                        |
| ---------------------------------------------------- | ------------------- | ------------ | ------------------------- |
| `task_id`                                            | `str`               | 用户任务 ID      | 标识一次完整 benchmark 构建任务     |
| `blueprint_id`                                       | `str`               | 总蓝图 ID       | 标识该任务的全局目标                |
| `user_goal`                                          | `str`               | 用户原始目标描述     | 防止多轮进化后目标漂移               |
| `language`                                           | `str`               | 题目语言         | 所有轮次默认继承                  |
| `max_evolution_rounds`                               | `int`               | 最大进化轮次       | 控制最多执行多少轮                 |
| `final_targets.by_mode.qa`                           | `int`               | 最终 QA 题数量    | 作为全局进度账本                  |
| `final_targets.by_mode.multiple_choice`              | `int`               | 最终选择题数量      | 作为全局进度账本                  |
| `seed_topics`                                        | `list[str]`         | 初始主题         | 第一轮主题来源                   |
| `termination_conditions.max_rounds_reached`          | `bool`              | 达到最大轮次时停止    | 终止条件                      |
| `termination_conditions.final_targets_reached`       | `bool`              | 达到最终题量时停止    | 终止条件                      |
| `termination_conditions.budget_exhausted`            | `bool`              | 预算耗尽时停止      | 终止条件                      |
| `termination_conditions.no_significant_gain`         | `bool`              | 边际收益过低时停止    | 终止条件                      |
| `user_required_constraints.require_citation`         | `bool \| null`      | 用户是否明确要求引用   | 只有用户明确要求时填写               |
| `user_required_constraints.require_capability_label` | `bool \| null`      | 用户是否明确要求能力标签 | 只有用户明确要求时填写               |
| `user_required_constraints.allowed_question_types`   | `list[str] \| null` | 用户限定的题型      | 没有明确要求则为空                 |
| `user_required_evaluation.candidate_models`          | `list[str] \| null` | 用户指定必须评估的模型  | 没有明确要求则由 PlannerState 管理  |
| `user_required_evaluation.required_metrics`          | `list[str] \| null` | 用户指定必须计算的指标  | 没有明确要求则由 evaluator 默认配置决定 |

---

# 5. PlannerState

## 5.1 用途

PlannerState 是规划智能体的跨轮次记忆。
它保存每轮反馈聚合结果，用于生成下一轮计划。

PlannerState 不下发给子智能体。

## 5.2 保存路径

```text
runs/{task_id}/planner_state.json
```

## 5.3 Schema

```yaml
PlannerState:
  task_id: str
  blueprint_id: str
  current_round: int

  progress:
    by_mode:
      qa: int
      multiple_choice: int

  topic_memory:
    <topic>:
      current_description: str
      pass_rate_history: list[float]
      citation_score_history: list[float]
      model_error_rate_history: list[float]
      discrimination_history: list[float]
      suspicious_rate_history: list[float]
      duplicate_rate_history: list[float]
      last_action: explore | repair | expand | mutate | prune | maintain

  difficulty_state:
    easy:
      empirical_accuracy: float | null
    medium:
      empirical_accuracy: float | null
    hard:
      empirical_accuracy: float | null
    decision: str | null

  diversity_state:
    embedding_dispersion: float | null
    near_duplicate_rate: float | null

  model_pool:
    active:
      - str
    eliminated:
      - model: str
        round_id: int
        reason: str

  budget_state:
    tokens_used: int
    model_calls_used: int
    budget_pressure: low | medium | high

  decision_log:
    - round_id: int
      action: str
      reason: str
```

## 5.4 字段说明

| 字段                                           | 类型              | 说明                 | 用途                   |
| -------------------------------------------- | --------------- | ------------------ | -------------------- |
| `task_id`                                    | `str`           | 当前任务 ID            | 关联 GlobalBlueprint   |
| `blueprint_id`                               | `str`           | 总蓝图 ID             | 关联全局目标               |
| `current_round`                              | `int`           | 当前已完成轮次            | 生成下一轮 `round_id`     |
| `progress.by_mode.qa`                        | `int`           | 当前已通过验证的 QA 题数量    | 判断 QA 题缺口            |
| `progress.by_mode.multiple_choice`           | `int`           | 当前已通过验证的选择题数量      | 判断选择题缺口              |
| `topic_memory.<topic>.current_description`   | `str`           | 当前主题描述             | 下一轮主题描述的基础           |
| `pass_rate_history`                          | `list[float]`   | 该主题历史验证通过率         | 判断主题质量稳定性            |
| `citation_score_history`                     | `list[float]`   | 该主题历史引用分数          | 判断是否需要提高证据严格度        |
| `model_error_rate_history`                   | `list[float]`   | 该主题历史模型错误率         | 判断主题挑战度              |
| `discrimination_history`                     | `list[float]`   | 该主题历史模型区分度         | 判断是否值得扩展             |
| `suspicious_rate_history`                    | `list[float]`   | 该主题疑似坏题比例          | 防止把坏题当作难题            |
| `duplicate_rate_history`                     | `list[float]`   | 该主题重复率             | 判断是否提高多样性            |
| `last_action`                                | `enum`          | 上一轮对该主题的操作         | 避免重复无效操作             |
| `difficulty_state.easy.empirical_accuracy`   | `float \| null` | easy 题模型实证准确率      | 校准真实难度               |
| `difficulty_state.medium.empirical_accuracy` | `float \| null` | medium 题模型实证准确率    | 判断 medium 是否其实过简单    |
| `difficulty_state.hard.empirical_accuracy`   | `float \| null` | hard 题模型实证准确率      | 判断 hard 是否有效         |
| `difficulty_state.decision`                  | `str \| null`   | 难度调整决策             | 生成下一轮难度比例            |
| `diversity_state.embedding_dispersion`       | `float \| null` | 当前题库 embedding 分散度 | 判断多样性是否不足            |
| `diversity_state.near_duplicate_rate`        | `float \| null` | 近重复题比例             | 判断是否提高 novelty       |
| `model_pool.active`                          | `list[str]`     | 当前仍参与评估的模型         | 下一轮 EvaluatorPlan 使用 |
| `model_pool.eliminated`                      | `list[dict]`    | 已淘汰模型及原因           | 避免重复评估低价值模型          |
| `budget_state.tokens_used`                   | `int`           | 已消耗 token          | 判断预算压力               |
| `budget_state.model_calls_used`              | `int`           | 已消耗模型调用次数          | 判断预算压力               |
| `budget_state.budget_pressure`               | `enum`          | 预算压力等级             | 决定是否降级验证/评估          |
| `decision_log`                               | `list[dict]`    | 历史规划决策             | 用于可解释性和调试            |

---

# 6. RoundPlan

## 6.1 用途

RoundPlan 是每一轮的执行计划。
每轮自动生成一个 `run_id`，并为该 run 生成一个 RoundPlan。

RoundPlan 会被拆分成 GeneratorPlan、ValidatorPlan、EvaluatorPlan。

## 6.2 Schema

```yaml
RoundPlan:
  task_id: str
  blueprint_id: str
  round_id: int
  run_id: str
  phase: exploration | repair | exploitation | focused_generation | finalization

  round_targets:
    by_mode:
      qa: int
      multiple_choice: int
    difficulty_distribution:
      easy: float
      medium: float
      hard: float

  topic_plan:
    topics:
      - str
    topic_descriptions:
      <topic>: str
    topic_actions:
      <topic>: explore | repair | expand | mutate | prune | maintain

  generation_policy:
    candidate_pool_multiplier: float
    diversity_level: low | medium | high
    novelty_level: low | medium | high
    evidence_strictness: relaxed | standard | high
    reasoning_depth: shallow | medium | deep

  validation_policy:
    profile: fast | standard | strict
    llm_enabled: bool
    dedup_enabled: bool
    overrides: dict | null

  evaluation_policy:
    enabled: bool
    candidate_models:
      - str
    judge_enabled: bool
    eval_scope: new_questions_only | all_validated | sample
    metrics_profile: fast | standard | strict

  budget:
    token_limit: int | null
    model_call_limit: int | null
```

## 6.3 字段说明

| 字段                                      | 类型               | 说明               | 用途                          |
| --------------------------------------- | ---------------- | ---------------- | --------------------------- |
| `task_id`                               | `str`            | 当前任务 ID          | 关联全局任务                      |
| `blueprint_id`                          | `str`            | 总蓝图 ID           | 关联最终目标                      |
| `round_id`                              | `int`            | 当前轮次             | 控制进化流程                      |
| `run_id`                                | `str`            | 当前轮执行 ID         | 用于产物路径和 shared_state        |
| `phase`                                 | `enum`           | 当前轮阶段            | 决定探索、修复、集中生成等策略             |
| `round_targets.by_mode.qa`              | `int`            | 本轮希望通过验证的 QA 题数量 | 分配给 GeneratorPlan           |
| `round_targets.by_mode.multiple_choice` | `int`            | 本轮希望通过验证的选择题数量   | 分配给 GeneratorPlan           |
| `difficulty_distribution.easy`          | `float`          | 本轮 easy 题比例      | 根据实证难度动态调整                  |
| `difficulty_distribution.medium`        | `float`          | 本轮 medium 题比例    | 根据模型准确率动态调整                 |
| `difficulty_distribution.hard`          | `float`          | 本轮 hard 题比例      | 题目过简单时提高                    |
| `topic_plan.topics`                     | `list[str]`      | 本轮主题列表           | 传给生成智能体                     |
| `topic_descriptions`                    | `dict[str,str]`  | 每个主题的细化描述        | 承载动态主题选择结果                  |
| `topic_actions`                         | `dict[str,enum]` | 每个主题的规划动作        | 指示 repair、expand、mutate 等意图 |
| `candidate_pool_multiplier`             | `float`          | 候选池倍率            | 控制生成冗余，适应通过率和多样性            |
| `diversity_level`                       | `enum`           | 多样性等级            | 多样性不足时提高                    |
| `novelty_level`                         | `enum`           | 新颖性等级            | 重复率高时提高                     |
| `evidence_strictness`                   | `enum`           | 证据严格度            | 引用分数低时提高                    |
| `reasoning_depth`                       | `enum`           | 推理深度             | 题目过简单时提高                    |
| `validation_policy.profile`             | `enum`           | 验证档位             | 控制 fast/standard/strict     |
| `validation_policy.llm_enabled`         | `bool`           | 本轮是否启用 LLM 验证    | 根据质量风险和预算动态决定               |
| `validation_policy.dedup_enabled`       | `bool`           | 本轮是否启用去重         | 多样性不足或重复率高时启用               |
| `validation_policy.overrides`           | `dict \| null`   | 验证阈值覆盖项          | 仅在反馈触发时使用                   |
| `evaluation_policy.enabled`             | `bool`           | 本轮是否执行模型评估       | 探索轮也可以开启小规模评估               |
| `candidate_models`                      | `list[str]`      | 本轮参与评估的模型        | 根据模型池动态淘汰或保留                |
| `judge_enabled`                         | `bool`           | 本轮是否启用 LLM judge | 根据预算和评估需要决定                 |
| `eval_scope`                            | `enum`           | 评估范围             | 可只评估新增题以节省成本                |
| `metrics_profile`                       | `enum`           | 指标档位             | 使用 evaluator YAML 中的指标配置    |
| `budget.token_limit`                    | `int \| null`    | 本轮 token 上限      | 控制成本                        |
| `budget.model_call_limit`               | `int \| null`    | 本轮模型调用上限         | 控制成本                        |

---

# 7. GeneratorPlan

## 7.1 用途

GeneratorPlan 是下发给题目生成智能体的计划。
它尽量兼容当前 `qa_agent` 的 `Blueprint` 结构：`task_id / run_id / language / topics / modes`。

动态策略通过 `planner_overrides` 传入，不强行改变原始 Blueprint。

## 7.2 Schema

```yaml
GeneratorPlan:
  task_id: str
  run_id: str
  round_id: int

  blueprint:
    language: str
    topics:
      - str
    modes:
      qa:
        count: int
        max_rounds: int
        difficulty_distribution:
          easy: float
          medium: float
          hard: float
      multiple_choice:
        count: int
        max_rounds: int
        difficulty_distribution:
          easy: float
          medium: float
          hard: float

  planner_overrides:
    topic_descriptions:
      <topic>: str
    previous_gaps: dict | null
    candidate_pool_multiplier: float | null
    diversity_level: low | medium | high | null
    novelty_level: low | medium | high | null
    evidence_strictness: relaxed | standard | high | null
    reasoning_depth: shallow | medium | deep | null

  budget:
    token_limit: int | null
    model_call_limit: int | null
```

## 7.3 字段说明

| 字段                                                        | 类型              | 说明            | 用途                 |
| --------------------------------------------------------- | --------------- | ------------- | ------------------ |
| `task_id`                                                 | `str`           | 当前任务 ID       | 关联全局任务             |
| `run_id`                                                  | `str`           | 当前轮运行 ID      | 写入本轮 shared_state  |
| `round_id`                                                | `int`           | 当前轮次          | 日志和反馈关联            |
| `blueprint.language`                                      | `str`           | 目标语言          | 生成题目语言             |
| `blueprint.topics`                                        | `list[str]`     | 本轮主题          | 与当前实现保持兼容          |
| `blueprint.modes.qa.count`                                | `int`           | 本轮 QA 目标题量    | 生成智能体生成目标          |
| `blueprint.modes.qa.max_rounds`                           | `int`           | QA 内部最大生成轮数   | 可沿用 agent 默认值      |
| `blueprint.modes.qa.difficulty_distribution`              | `dict`          | QA 难度比例       | 本轮动态难度分配           |
| `blueprint.modes.multiple_choice.count`                   | `int`           | 本轮选择题目标题量     | 生成智能体生成目标          |
| `blueprint.modes.multiple_choice.max_rounds`              | `int`           | 选择题内部最大生成轮数   | 可沿用 agent 默认值      |
| `blueprint.modes.multiple_choice.difficulty_distribution` | `dict`          | 选择题难度比例       | 本轮动态难度分配           |
| `planner_overrides.topic_descriptions`                    | `dict[str,str]` | 主题细化描述        | 引导生成更有挑战度或更有证据支撑的题 |
| `planner_overrides.previous_gaps`                         | `dict \| null`  | 上一轮剩余缺口       | 用于定向补题             |
| `planner_overrides.candidate_pool_multiplier`             | `float \| null` | 候选池倍率覆盖       | 根据通过率、多样性、预算动态调整   |
| `planner_overrides.diversity_level`                       | `enum \| null`  | 多样性策略         | 指示生成更多子方向和来源覆盖     |
| `planner_overrides.novelty_level`                         | `enum \| null`  | 新颖性策略         | 指示避免重复已有题          |
| `planner_overrides.evidence_strictness`                   | `enum \| null`  | 证据严格度         | 引用不足时提高            |
| `planner_overrides.reasoning_depth`                       | `enum \| null`  | 推理深度          | 题目过简单时提高           |
| `budget.token_limit`                                      | `int \| null`   | 本轮生成 token 上限 | 控制成本               |
| `budget.model_call_limit`                                 | `int \| null`   | 本轮生成调用上限      | 控制成本               |

---

# 8. GeneratorFeedback

## 8.1 用途

GeneratorFeedback 用于告诉 planner：

1. 本轮是否生成足够候选题。
2. 哪些主题生成困难。
3. 是否存在来源不足。
4. 是否存在重复或多样性不足。
5. 是否需要下一轮补题。

## 8.2 Schema

```yaml
GeneratorFeedback:
  task_id: str
  run_id: str
  round_id: int
  status: success | partial_success | failed

  output_refs:
    candidate_questions: str

  summary:
    generated_count: int
    target_candidate_count: int
    completion_rate: float

  by_topic:
    <topic>:
      generated_count: int
      estimated_duplicate_rate: float | null
      source_coverage: good | medium | weak | unknown
      dominant_subtopic_ratio: float | null

  by_mode:
    qa:
      generated_count: int
    multiple_choice:
      generated_count: int

  by_difficulty:
    easy:
      generated_count: int
    medium:
      generated_count: int
    hard:
      generated_count: int

  unfilled_gaps: dict

  failure_reasons:
    <reason>: int

  cost:
    tokens: int
    model_calls: int
    latency_seconds: float | null
```

## 8.3 字段说明

| 字段                                        | 类型              | 说明              | planner 用途                          |
| ----------------------------------------- | --------------- | --------------- | ----------------------------------- |
| `status`                                  | `enum`          | 生成是否成功          | 判断是否继续验证                            |
| `output_refs.candidate_questions`         | `str`           | 候选题 artifact 路径 | ValidatorPlan 输入                    |
| `summary.generated_count`                 | `int`           | 实际生成候选题数        | 判断候选池是否足够                           |
| `summary.target_candidate_count`          | `int`           | 目标候选题数          | 对比生成完成度                             |
| `summary.completion_rate`                 | `float`         | 候选池完成率          | 低于阈值时补题或降低目标                        |
| `by_topic.<topic>.generated_count`        | `int`           | 每个主题生成题数        | 判断主题覆盖情况                            |
| `estimated_duplicate_rate`                | `float \| null` | 估计重复率           | 高时提高 `novelty_level`                |
| `source_coverage`                         | `enum`          | 来源覆盖强弱          | weak 时增加来源或提高证据约束                   |
| `dominant_subtopic_ratio`                 | `float \| null` | 单一子方向占比         | 高时提高 `diversity_level`              |
| `by_mode.qa.generated_count`              | `int`           | QA 候选题数         | 判断题型缺口                              |
| `by_mode.multiple_choice.generated_count` | `int`           | 选择题候选题数         | 判断题型缺口                              |
| `by_difficulty.easy.generated_count`      | `int`           | easy 候选题数       | 判断难度分布是否达标                          |
| `by_difficulty.medium.generated_count`    | `int`           | medium 候选题数     | 判断难度分布是否达标                          |
| `by_difficulty.hard.generated_count`      | `int`           | hard 候选题数       | hard 不足时调整主题或证据策略                   |
| `unfilled_gaps`                           | `dict`          | 生成阶段未完成的缺口      | 下一轮 GeneratorPlan 的 `previous_gaps` |
| `failure_reasons`                         | `dict[str,int]` | 失败原因计数          | 判断是来源不足、重复、还是题型生成困难                 |
| `cost.tokens`                             | `int`           | token 消耗        | 更新预算状态                              |
| `cost.model_calls`                        | `int`           | 模型调用次数          | 更新预算状态                              |
| `cost.latency_seconds`                    | `float \| null` | 延迟              | 运行效率分析                              |

---

# 9. ValidatorPlan

## 9.1 用途

ValidatorPlan 是下发给题目验证智能体的计划。
它不展开所有验证阈值，而是使用 `profile + overrides`。

## 9.2 Schema

```yaml
ValidatorPlan:
  task_id: str
  run_id: str
  round_id: int

  inputs:
    artifact_ref: str
    existing_pool_summary_ref: str | null

  validation_policy:
    profile: fast | standard | strict
    llm_enabled: bool
    dedup_enabled: bool
    overrides:
      min_citation_score: float | null
      min_overall_score: float | null
      dedup_threshold: float | null
      embedding_model: str | null
      min_embedding_dispersion: float | null
      max_near_duplicate_rate: float | null

  budget:
    token_limit: int | null
    model_call_limit: int | null
```

## 9.3 字段说明

| 字段                                   | 类型              | 说明                | 用途                      |
| ------------------------------------ | --------------- | ----------------- | ----------------------- |
| `task_id`                            | `str`           | 当前任务 ID           | 关联全局任务                  |
| `run_id`                             | `str`           | 当前轮运行 ID          | 读取本轮 shared_state       |
| `round_id`                           | `int`           | 当前轮次              | 反馈关联                    |
| `inputs.artifact_ref`                | `str`           | 待验证候选题 artifact   | 来自 GeneratorFeedback    |
| `inputs.existing_pool_summary_ref`   | `str \| null`   | 已有题库摘要            | 用于跨轮次去重                 |
| `validation_policy.profile`          | `enum`          | 验证档位              | 默认使用 YAML 中的 profile 配置 |
| `validation_policy.llm_enabled`      | `bool`          | 是否启用 LLM 验证       | 根据质量风险和预算动态决定           |
| `validation_policy.dedup_enabled`    | `bool`          | 是否启用 embedding 去重 | 重复率高时启用                 |
| `overrides.min_citation_score`       | `float \| null` | 引用最低分覆盖           | 引用不足时提高                 |
| `overrides.min_overall_score`        | `float \| null` | LLM 综合质量分覆盖       | 质量不稳定时提高                |
| `overrides.dedup_threshold`          | `float \| null` | 去重阈值覆盖            | 重复率高时降低阈值               |
| `overrides.embedding_model`          | `str \| null`   | embedding 模型覆盖    | 需要指定部署模型时使用             |
| `overrides.min_embedding_dispersion` | `float \| null` | 最低 embedding 分散度  | 多样性不足时设置                |
| `overrides.max_near_duplicate_rate`  | `float \| null` | 最大近重复率            | 多样性不足时设置                |
| `budget.token_limit`                 | `int \| null`   | 验证 token 上限       | 控制成本                    |
| `budget.model_call_limit`            | `int \| null`   | 验证模型调用上限          | 控制成本                    |

---

# 10. ValidatorFeedback

## 10.1 用途

ValidatorFeedback 用于告诉 planner：

1. 有多少题真正通过验证。
2. 哪些主题质量稳定。
3. 哪些主题是伪难或质量差。
4. 引用、LLM 质量、多样性是否达标。
5. 下一轮应该补哪些缺口。

## 10.2 Schema

```yaml
ValidatorFeedback:
  task_id: str
  run_id: str
  round_id: int
  status: success | partial_success | failed

  output_refs:
    validated_questions: str
    rejected_questions: str | null
    needs_fix_questions: str | null

  summary:
    input_count: int
    passed_count: int
    pass_rate: float

  remaining_gaps: dict

  by_topic:
    <topic>:
      input_count: int
      passed_count: int
      pass_rate: float
      avg_citation_score: float | null
      avg_llm_score: float | null
      near_duplicate_rate: float | null
      embedding_dispersion: float | null
      top_rejection_reasons:
        <reason>: int

  by_difficulty:
    easy:
      pass_rate: float
    medium:
      pass_rate: float
    hard:
      pass_rate: float

  by_mode:
    qa:
      pass_rate: float
    multiple_choice:
      pass_rate: float

  diversity_summary:
    near_duplicate_rate: float | null
    embedding_dispersion: float | null
    diversity_passed: bool | null

  rejection_summary:
    top_reasons:
      <reason>: int

  cost:
    tokens: int
    model_calls: int
    latency_seconds: float | null
```

## 10.3 字段说明

| 字段                                       | 类型              | 说明               | planner 用途                     |
| ---------------------------------------- | --------------- | ---------------- | ------------------------------ |
| `status`                                 | `enum`          | 验证是否成功           | 判断是否进入评估                       |
| `output_refs.validated_questions`        | `str`           | 通过题目 artifact    | EvaluatorPlan 输入               |
| `output_refs.rejected_questions`         | `str \| null`   | 拒收题 artifact     | 分析失败原因                         |
| `output_refs.needs_fix_questions`        | `str \| null`   | 可修复题 artifact    | 决定是否修复或重生成                     |
| `summary.input_count`                    | `int`           | 输入候选题数           | 计算通过率                          |
| `summary.passed_count`                   | `int`           | 通过题数             | 更新全局进度                         |
| `summary.pass_rate`                      | `float`         | 整体通过率            | 调整 `candidate_pool_multiplier` |
| `remaining_gaps`                         | `dict`          | 验证后剩余缺口          | 下一轮定向补题                        |
| `by_topic.<topic>.input_count`           | `int`           | 主题输入题数           | 分析主题覆盖                         |
| `by_topic.<topic>.passed_count`          | `int`           | 主题通过题数           | 分析主题有效性                        |
| `by_topic.<topic>.pass_rate`             | `float`         | 主题通过率            | 低时 repair，高时可 expand           |
| `avg_citation_score`                     | `float \| null` | 平均引用分            | 低时提高 `evidence_strictness`     |
| `avg_llm_score`                          | `float \| null` | 平均 LLM 质量分       | 低时提高验证强度或降低难度                  |
| `near_duplicate_rate`                    | `float \| null` | 近重复率             | 高时提高 `novelty_level`           |
| `embedding_dispersion`                   | `float \| null` | embedding 分散度    | 低时提高 `diversity_level`         |
| `top_rejection_reasons`                  | `dict`          | 该主题主要拒收原因        | 决定修复方向                         |
| `by_difficulty.easy.pass_rate`           | `float`         | easy 题通过率        | 判断基础质量                         |
| `by_difficulty.medium.pass_rate`         | `float`         | medium 题通过率      | 判断中等难度是否可控                     |
| `by_difficulty.hard.pass_rate`           | `float`         | hard 题通过率        | hard 过低时降低比例或加强证据              |
| `by_mode.qa.pass_rate`                   | `float`         | QA 通过率           | 调整 QA 候选池                      |
| `by_mode.multiple_choice.pass_rate`      | `float`         | 选择题通过率           | 调整选择题候选池                       |
| `diversity_summary.near_duplicate_rate`  | `float \| null` | 全局近重复率           | 控制多样性策略                        |
| `diversity_summary.embedding_dispersion` | `float \| null` | 全局 embedding 分散度 | 控制多样性策略                        |
| `diversity_summary.diversity_passed`     | `bool \| null`  | 多样性是否达标          | 决定是否继续生成                       |
| `rejection_summary.top_reasons`          | `dict`          | 全局拒收原因           | 决定下一轮修复方向                      |
| `cost.tokens`                            | `int`           | 验证 token 消耗      | 更新预算                           |
| `cost.model_calls`                       | `int`           | 验证模型调用次数         | 更新预算                           |
| `cost.latency_seconds`                   | `float \| null` | 验证延迟             | 运行效率分析                         |

---

# 11. EvaluatorPlan

## 11.1 用途

EvaluatorPlan 是下发给模型评估智能体的计划。
它不展开完整指标定义，而是使用 `metrics_profile`。
本轮评估模型由 PlannerState 的 active model pool 和用户显式要求共同决定。

## 11.2 Schema

```yaml
EvaluatorPlan:
  task_id: str
  run_id: str
  round_id: int

  inputs:
    artifact_ref: str

  evaluation_policy:
    enabled: bool
    candidate_models:
      - str
    judge_enabled: bool
    eval_scope: new_questions_only | all_validated | sample
    metrics_profile: fast | standard | strict

  budget:
    token_limit: int | null
    model_call_limit: int | null
```

## 11.3 字段说明

| 字段                          | 类型            | 说明             | 用途                           |
| --------------------------- | ------------- | -------------- | ---------------------------- |
| `task_id`                   | `str`         | 当前任务 ID        | 关联全局任务                       |
| `run_id`                    | `str`         | 当前轮运行 ID       | 读取本轮 shared_state            |
| `round_id`                  | `int`         | 当前轮次           | 反馈关联                         |
| `inputs.artifact_ref`       | `str`         | 待评估题目 artifact | 来自 ValidatorFeedback         |
| `evaluation_policy.enabled` | `bool`        | 是否执行评估         | 预算紧张或无有效题时可跳过                |
| `candidate_models`          | `list[str]`   | 本轮候选模型         | 动态淘汰低价值模型                    |
| `judge_enabled`             | `bool`        | 是否启用 LLM judge | 根据预算和评估需求决定                  |
| `eval_scope`                | `enum`        | 评估范围           | 常用 `new_questions_only` 节省成本 |
| `metrics_profile`           | `enum`        | 指标档位           | 使用 evaluator YAML 的指标组合      |
| `budget.token_limit`        | `int \| null` | 评估 token 上限    | 控制成本                         |
| `budget.model_call_limit`   | `int \| null` | 评估模型调用上限       | 控制成本                         |

---

# 12. EvaluatorFeedback

## 12.1 用途

EvaluatorFeedback 用于告诉 planner：

1. 哪些主题对模型更有挑战。
2. 哪些主题有模型区分度。
3. 哪些题太简单。
4. 哪些题可能是坏题。
5. 哪些模型可以淘汰。
6. 下一轮是否应提高 hard 比例。

## 12.2 Schema

```yaml
EvaluatorFeedback:
  task_id: str
  run_id: str
  round_id: int
  status: success | partial_success | failed

  output_refs:
    score_matrix: str
    evaluation_report: str

  by_topic:
    <topic>:
      question_count: int
      avg_model_accuracy: float
      model_error_rate: float
      model_discrimination: float
      too_easy_rate: float
      suspicious_rate: float

  by_difficulty:
    easy:
      avg_model_accuracy: float
    medium:
      avg_model_accuracy: float
    hard:
      avg_model_accuracy: float

  by_mode:
    qa:
      avg_model_accuracy: float
    multiple_choice:
      avg_model_accuracy: float

  by_capability:
    <capability>:
      avg_model_accuracy: float
      model_discrimination: float | null
      challenge_score: float | null

  model_summary:
    <model_name>:
      overall_score: float
      status_hint: keep | eliminate | uncertain
      weak_topics:
        - str
      weak_capabilities:
        - str

  item_diagnostics:
    too_easy_questions:
      - str
    suspicious_questions:
      - str
    high_discrimination_questions:
      - str

  cost:
    tokens: int
    model_calls: int
    latency_seconds: float | null
```

## 12.3 字段说明

| 字段                                                | 类型              | 说明            | planner 用途           |
| ------------------------------------------------- | --------------- | ------------- | -------------------- |
| `status`                                          | `enum`          | 评估是否成功        | 判断是否可更新主题策略          |
| `output_refs.score_matrix`                        | `str`           | 题目级分数矩阵       | 供报告和分析使用             |
| `output_refs.evaluation_report`                   | `str`           | 模型评估报告        | 供最终输出使用              |
| `by_topic.<topic>.question_count`                 | `int`           | 该主题参与评估题数     | 评估统计置信度              |
| `avg_model_accuracy`                              | `float`         | 模型平均准确率       | 高说明题可能太简单            |
| `model_error_rate`                                | `float`         | 模型错误率         | 高说明主题更有挑战            |
| `model_discrimination`                            | `float`         | 模型区分度         | 高说明主题更有评估价值          |
| `too_easy_rate`                                   | `float`         | 全模型都容易答对的比例   | 高时 mutate 到更难主题      |
| `suspicious_rate`                                 | `float`         | 疑似坏题比例        | 高时 repair 或回流验证      |
| `by_difficulty.easy.avg_model_accuracy`           | `float`         | easy 题实证准确率   | 校准 easy 难度           |
| `by_difficulty.medium.avg_model_accuracy`         | `float`         | medium 题实证准确率 | 高时说明 medium 过简单      |
| `by_difficulty.hard.avg_model_accuracy`           | `float`         | hard 题实证准确率   | 判断 hard 是否有效         |
| `by_mode.qa.avg_model_accuracy`                   | `float`         | QA 题平均准确率     | 调整 QA 难度             |
| `by_mode.multiple_choice.avg_model_accuracy`      | `float`         | 选择题平均准确率      | 调整选择题难度              |
| `by_capability.<capability>.avg_model_accuracy`   | `float`         | 能力维度准确率       | 判断弱能力方向              |
| `by_capability.<capability>.model_discrimination` | `float \| null` | 能力维度模型区分度     | 高时增加该能力题             |
| `by_capability.<capability>.challenge_score`      | `float \| null` | 能力挑战度         | 下一轮能力分布参考            |
| `model_summary.<model>.overall_score`             | `float`         | 模型整体得分        | 判断模型表现               |
| `status_hint`                                     | `enum`          | 模型保留/淘汰建议     | 更新 active model pool |
| `weak_topics`                                     | `list[str]`     | 该模型弱主题        | 生成诊断题                |
| `weak_capabilities`                               | `list[str]`     | 该模型弱能力        | 生成诊断题                |
| `too_easy_questions`                              | `list[str]`     | 太简单题目 ID      | 可从后续生成方向中减少类似题       |
| `suspicious_questions`                            | `list[str]`     | 疑似坏题 ID       | 回流 Validator 复核      |
| `high_discrimination_questions`                   | `list[str]`     | 高区分题 ID       | 提取高价值题型特征            |
| `cost.tokens`                                     | `int`           | 评估 token 消耗   | 更新预算状态               |
| `cost.model_calls`                                | `int`           | 评估调用次数        | 更新预算状态               |
| `cost.latency_seconds`                            | `float \| null` | 评估延迟          | 运行效率分析               |

---

# 13. shared_state 兼容设计

## 13.1 用途

`shared_state.json` 只保存当前 run 的公共状态和 artifact 引用。
它不保存完整 PlannerState。

## 13.2 Schema

```yaml
SharedState:
  task_id: str
  run_id: str

  blueprint:
    language: str
    topics:
      - str
    modes:
      qa:
        count: int
        max_rounds: int
        difficulty_distribution:
          easy: float
          medium: float
          hard: float
      multiple_choice:
        count: int
        max_rounds: int
        difficulty_distribution:
          easy: float
          medium: float
          hard: float

  artifacts:
    round_plan: str
    qa_candidate_pool: str | null
    multiple_choice_candidate_pool: str | null
    chunked_evidence: str | null
    validated_questions: str | null
    evaluation_report: str | null
    generator_feedback: str | null
    validator_feedback: str | null
    evaluator_feedback: str | null
```

## 13.3 字段说明

| 字段                                         | 类型            | 说明                | 用途              |
| ------------------------------------------ | ------------- | ----------------- | --------------- |
| `task_id`                                  | `str`         | 当前任务 ID           | 关联全局任务          |
| `run_id`                                   | `str`         | 当前轮运行 ID          | 当前 run 的唯一标识    |
| `blueprint.language`                       | `str`         | 本轮语言              | 供各 agent 读取     |
| `blueprint.topics`                         | `list[str]`   | 本轮主题              | 与当前生成智能体兼容      |
| `blueprint.modes`                          | `dict`        | 本轮题型目标            | 与当前生成智能体兼容      |
| `artifacts.round_plan`                     | `str`         | 本轮 RoundPlan 文件路径 | 供调试和复现          |
| `artifacts.qa_candidate_pool`              | `str \| null` | QA 候选题路径          | Validator 输入    |
| `artifacts.multiple_choice_candidate_pool` | `str \| null` | 选择题候选题路径          | Validator 输入    |
| `artifacts.chunked_evidence`               | `str \| null` | 证据块路径             | Validator 使用    |
| `artifacts.validated_questions`            | `str \| null` | 通过验证题目路径          | Evaluator 输入    |
| `artifacts.evaluation_report`              | `str \| null` | 评估报告路径            | Planner 和最终报告使用 |
| `artifacts.generator_feedback`             | `str \| null` | 生成反馈路径            | Planner 更新状态    |
| `artifacts.validator_feedback`             | `str \| null` | 验证反馈路径            | Planner 更新状态    |
| `artifacts.evaluator_feedback`             | `str \| null` | 评估反馈路径            | Planner 更新状态    |

---

# 14. 反馈驱动的计划更新规则

## 14.1 更新 `candidate_pool_multiplier`

### 输入反馈

来自 ValidatorFeedback：

```yaml
summary.pass_rate
diversity_summary.near_duplicate_rate
```

来自 PlannerState：

```yaml
budget_state.budget_pressure
```

### 规则

```python
base = 1.0 / max(pass_rate, 0.25)

if near_duplicate_rate is not None and near_duplicate_rate > 0.20:
    base += 0.3

if budget_pressure == "high":
    base -= 0.4

candidate_pool_multiplier = clamp(base, 1.2, 2.8)
```

### 调整含义

| 反馈情况  | 下一轮调整                 |
| ----- | --------------------- |
| 通过率低  | 提高候选池倍率               |
| 重复率高  | 提高候选池倍率，同时提高 novelty  |
| 预算压力高 | 降低候选池倍率               |
| 通过率极低 | 不只提高倍率，应 repair 主题或来源 |

---

## 14.2 更新 `diversity_level`

### 输入反馈

来自 ValidatorFeedback：

```yaml
diversity_summary.embedding_dispersion
diversity_summary.near_duplicate_rate
by_topic.<topic>.embedding_dispersion
by_topic.<topic>.near_duplicate_rate
```

### 规则

```python
if near_duplicate_rate > 0.25 or embedding_dispersion < 0.25:
    diversity_level = "high"
elif near_duplicate_rate > 0.15:
    diversity_level = "medium"
else:
    diversity_level = "medium"
```

### 同步调整

当 `diversity_level = high` 时，下一轮同时调整：

```yaml
generation_policy:
  diversity_level: high
  novelty_level: high

topic_descriptions:
  <topic>: "覆盖更多子方向，避免围绕同一事实或定义重复出题"

validation_policy:
  dedup_enabled: true
  overrides:
    dedup_threshold: 0.82
```

---

## 14.3 更新 `novelty_level`

### 输入反馈

来自 GeneratorFeedback：

```yaml
by_topic.<topic>.estimated_duplicate_rate
```

来自 ValidatorFeedback：

```yaml
by_topic.<topic>.near_duplicate_rate
rejection_summary.top_reasons.duplicate_question
```

### 规则

```python
if duplicate_rate > 0.25:
    novelty_level = "high"
elif duplicate_rate > 0.15:
    novelty_level = "medium"
else:
    novelty_level = "medium"
```

### 调整含义

| 反馈情况       | 下一轮动作                       |
| ---------- | --------------------------- |
| 重复率高       | 提高 novelty                  |
| 主题集中在单一子方向 | 修改 topic_description        |
| 重复来自已有题库   | 加强 existing_pool_summary 输入 |

---

## 14.4 更新 `evidence_strictness`

### 输入反馈

来自 ValidatorFeedback：

```yaml
by_topic.<topic>.avg_citation_score
rejection_summary.top_reasons.answer_not_supported_by_source
rejection_summary.top_reasons.weak_citation
```

### 规则

```python
if avg_citation_score < 0.60:
    evidence_strictness = "high"
elif weak_citation_count is high:
    evidence_strictness = "high"
else:
    evidence_strictness = "standard"
```

### 同步调整

```yaml
generation_policy:
  evidence_strictness: high

validation_policy:
  overrides:
    min_citation_score: 0.70

topic_descriptions:
  <topic>: "只生成有明确来源支持的问题，避免无证据推断"
```

---

## 14.5 更新 `reasoning_depth`

### 输入反馈

来自 EvaluatorFeedback：

```yaml
by_topic.<topic>.too_easy_rate
by_topic.<topic>.avg_model_accuracy
by_difficulty.medium.avg_model_accuracy
```

### 规则

```python
if too_easy_rate > 0.50 or avg_model_accuracy > 0.80:
    reasoning_depth = "deep"
elif avg_model_accuracy > 0.65:
    reasoning_depth = "medium"
else:
    reasoning_depth = "medium"
```

### 同步调整

```yaml
round_targets:
  difficulty_distribution:
    easy: 0.0
    medium: 0.35
    hard: 0.65

generation_policy:
  reasoning_depth: deep

topic_descriptions:
  <topic>: "避免基础定义题，优先生成需要比较、因果、多跳或证据推理的问题"
```

---

## 14.6 更新难度分布

### 输入反馈

来自 EvaluatorFeedback：

```yaml
by_difficulty.easy.avg_model_accuracy
by_difficulty.medium.avg_model_accuracy
by_difficulty.hard.avg_model_accuracy
```

### 规则

```python
if medium_accuracy > 0.80:
    increase_hard_ratio()

if hard_accuracy < 0.20 and suspicious_rate high:
    reduce_hard_ratio()
    increase_evidence_strictness()

if easy_accuracy > 0.90:
    reduce_easy_ratio()
```

### 示例

```yaml
difficulty_distribution:
  easy: 0.00
  medium: 0.30
  hard: 0.70
```

适用情况：

```text
中等题模型准确率过高，说明中等题实际偏简单。
```

---

## 14.7 更新主题描述

### 输入反馈

来自 ValidatorFeedback：

```yaml
by_topic.<topic>.pass_rate
by_topic.<topic>.avg_citation_score
by_topic.<topic>.top_rejection_reasons
```

来自 EvaluatorFeedback：

```yaml
by_topic.<topic>.model_error_rate
by_topic.<topic>.model_discrimination
by_topic.<topic>.too_easy_rate
by_topic.<topic>.suspicious_rate
```

### 动作分类

| 动作         | 条件                     | 下一轮处理       |
| ---------- | ---------------------- | ----------- |
| `repair`   | 通过率低，引用差，suspicious 高  | 收窄主题，强调证据支持 |
| `expand`   | 通过率高，区分度高，suspicious 低 | 增加该主题题量     |
| `mutate`   | 质量高但 too_easy 高        | 进化到更细、更难子方向 |
| `prune`    | 长期质量差或重复高              | 降低或移除该主题    |
| `maintain` | 质量和挑战度稳定               | 保持主题方向      |

### 规则

```python
if pass_rate < 0.35 or suspicious_rate > 0.20:
    action = "repair"
elif pass_rate > 0.60 and model_discrimination > 0.20:
    action = "expand"
elif pass_rate > 0.60 and too_easy_rate > 0.50:
    action = "mutate"
elif duplicate_rate > 0.30:
    action = "repair"
else:
    action = "maintain"
```

### 示例

```yaml
topic_descriptions:
  AI Safety: >
    上一轮质量稳定但模型准确率偏高。
    下一轮避免基础定义题，重点转向 deceptive alignment、
    mesa-optimization、scalable oversight 和因果推理题。

  Quantum Computing: >
    上一轮模型错误率高但 suspicious rate 也高。
    下一轮只围绕有明确来源支持的 quantum error correction 出题，
    避免无证据支撑的冷门断言。
```

---

## 14.8 更新 LLM 验证开关

### 输入反馈

来自 ValidatorFeedback：

```yaml
summary.pass_rate
by_topic.<topic>.avg_citation_score
rejection_summary.top_reasons
```

来自 EvaluatorFeedback：

```yaml
by_topic.<topic>.suspicious_rate
item_diagnostics.suspicious_questions
```

来自 PlannerState：

```yaml
budget_state.budget_pressure
current_round
max_evolution_rounds
```

### 规则

```python
if budget_pressure == "high" and current_round is early:
    llm_enabled = False
elif suspicious_rate > 0.15:
    llm_enabled = True
elif current_round is final_round:
    llm_enabled = True
else:
    llm_enabled = default_by_profile
```

### 解释

| 情况                | LLM 验证 |
| ----------------- | ------ |
| 早期探索且预算紧          | 可关闭    |
| suspicious rate 高 | 开启     |
| 最后一轮收敛            | 开启     |
| 引用验证稳定且预算紧        | 可抽样或关闭 |

---

## 14.9 更新模型池

### 输入反馈

来自 EvaluatorFeedback：

```yaml
model_summary.<model>.overall_score
model_summary.<model>.status_hint
```

### 规则

```python
for model in model_summary:
    if status_hint == "eliminate" and user did not require this model:
        move model from active to eliminated
```

### 示例

```yaml
PlannerState:
  model_pool:
    active:
      - qwen2.5-7b
      - deepseek-v3
    eliminated:
      - model: llama3.1-8b
        round_id: 2
        reason: "连续两轮表现明显低且对排序贡献小"
```

下一轮：

```yaml
EvaluatorPlan:
  evaluation_policy:
    candidate_models:
      - qwen2.5-7b
      - deepseek-v3
```

---

## 14.10 更新评估范围

### 输入反馈

来自 PlannerState：

```yaml
progress
budget_state
```

来自 ValidatorFeedback：

```yaml
output_refs.validated_questions
```

### 规则

```python
if budget_pressure == "high":
    eval_scope = "sample"
elif current_round > 1:
    eval_scope = "new_questions_only"
else:
    eval_scope = "new_questions_only"
```

### 解释

| 情况         | eval_scope           |
| ---------- | -------------------- |
| 正常多轮       | `new_questions_only` |
| 预算紧张       | `sample`             |
| 最终报告需要全量校准 | `all_validated`      |

---

# 15. Planner 工作流程

## 15.1 初始化

输入：

```text
用户文本描述
```

输出：

```text
GlobalBlueprint
PlannerState
RoundPlan(round_id=1)
```

步骤：

```python
def initialize_planner(user_text):
    global_blueprint = parse_user_goal(user_text)
    planner_state = init_state(global_blueprint)
    round_plan = create_initial_round_plan(global_blueprint, planner_state)
    return global_blueprint, planner_state, round_plan
```

第一轮通常使用：

```yaml
phase: exploration

round_targets:
  by_mode:
    qa: 较少数量
    multiple_choice: 较少数量

generation_policy:
  candidate_pool_multiplier: 1.5
  diversity_level: high
  novelty_level: medium
  evidence_strictness: standard
  reasoning_depth: medium

validation_policy:
  profile: standard
  llm_enabled: 视预算决定

evaluation_policy:
  enabled: true
  eval_scope: new_questions_only
```

---

## 15.2 每轮执行

```python
def run_round(round_plan):
    generator_plan = build_generator_plan(round_plan)
    generator_feedback = generator.run(generator_plan)

    validator_plan = build_validator_plan(round_plan, generator_feedback)
    validator_feedback = validator.run(validator_plan)

    if validator_feedback.summary.passed_count == 0:
        return generator_feedback, validator_feedback, None

    evaluator_plan = build_evaluator_plan(round_plan, validator_feedback)
    evaluator_feedback = evaluator.run(evaluator_plan)

    return generator_feedback, validator_feedback, evaluator_feedback
```

---

## 15.3 更新 PlannerState

```python
def update_planner_state(state, gen_fb, val_fb, eval_fb):
    update_progress(state, val_fb)
    update_topic_memory(state, gen_fb, val_fb, eval_fb)
    update_difficulty_state(state, eval_fb)
    update_diversity_state(state, val_fb)
    update_model_pool(state, eval_fb)
    update_budget_state(state, gen_fb, val_fb, eval_fb)
    append_decision_log(state)
    return state
```

---

## 15.4 生成下一轮 RoundPlan

```python
def create_next_round_plan(global_blueprint, planner_state):
    if should_stop(global_blueprint, planner_state):
        return None

    phase = decide_phase(global_blueprint, planner_state)
    round_targets = allocate_round_targets(global_blueprint, planner_state)
    topic_plan = select_topics_and_descriptions(planner_state)
    generation_policy = decide_generation_policy(planner_state)
    validation_policy = decide_validation_policy(planner_state)
    evaluation_policy = decide_evaluation_policy(planner_state)
    budget = allocate_round_budget(planner_state)

    return RoundPlan(
        task_id=global_blueprint.task_id,
        blueprint_id=global_blueprint.blueprint_id,
        round_id=planner_state.current_round + 1,
        run_id=generate_run_id(),
        phase=phase,
        round_targets=round_targets,
        topic_plan=topic_plan,
        generation_policy=generation_policy,
        validation_policy=validation_policy,
        evaluation_policy=evaluation_policy,
        budget=budget,
    )
```

---

# 16. 轮次目标分配策略

## 16.1 探索轮

适用于早期轮次。

```yaml
phase: exploration
round_targets:
  数量较少
generation_policy:
  diversity_level: high
  novelty_level: medium
  candidate_pool_multiplier: 1.5
```

目标：

```text
探索哪些主题质量好、哪些主题对模型有挑战。
```

---

## 16.2 修复轮

适用于验证通过率低、引用不足、suspicious rate 高的情况。

```yaml
phase: repair
generation_policy:
  evidence_strictness: high
  diversity_level: high
  candidate_pool_multiplier: 1.8
validation_policy:
  llm_enabled: true
```

目标：

```text
修复质量问题，补齐 remaining_gaps。
```

---

## 16.3 利用轮

适用于发现高质量、高区分度主题之后。

```yaml
phase: exploitation
generation_policy:
  reasoning_depth: deep
  novelty_level: high
  candidate_pool_multiplier: 1.8
```

目标：

```text
扩展高价值主题，增加中高难度题。
```

---

## 16.4 集中生成轮

适用于方向明确后。

```yaml
phase: focused_generation
round_targets:
  数量较多
difficulty_distribution:
  easy: 0.0
  medium: 0.3
  hard: 0.7
```

目标：

```text
快速扩充最终题库。
```

---

## 16.5 最终收敛轮

适用于最后一轮或接近目标时。

```yaml
phase: finalization
validation_policy:
  profile: strict
  llm_enabled: true
evaluation_policy:
  eval_scope: all_validated
```

目标：

```text
补齐缺口，严格验证，生成最终报告。
```

---

# 17. 停止条件

```python
def should_stop(global_blueprint, planner_state):
    if planner_state.current_round >= global_blueprint.max_evolution_rounds:
        return True

    if all_mode_targets_reached(global_blueprint, planner_state):
        return True

    if planner_state.budget_state.budget_pressure == "exhausted":
        return True

    if no_significant_gain(planner_state):
        return True

    return False
```

## 字段说明

| 条件                      | 说明                   |
| ----------------------- | -------------------- |
| `max_rounds_reached`    | 达到最大进化轮次             |
| `final_targets_reached` | 各题型最终通过数量都达标         |
| `budget_exhausted`      | token 或模型调用预算耗尽      |
| `no_significant_gain`   | 最近几轮新增高质量题过少，继续进化收益低 |

---

# 18. 最小落地版本

如果先做 MVP，只实现以下字段即可。

## 18.1 GlobalBlueprint MVP

```yaml
GlobalBlueprint:
  task_id: str
  blueprint_id: str
  user_goal: str
  language: str
  max_evolution_rounds: int
  final_targets:
    by_mode:
      qa: int
      multiple_choice: int
  seed_topics:
    - str
```

## 18.2 RoundPlan MVP

```yaml
RoundPlan:
  task_id: str
  run_id: str
  round_id: int
  phase: str

  round_targets:
    by_mode:
      qa: int
      multiple_choice: int
    difficulty_distribution:
      easy: float
      medium: float
      hard: float

  topic_plan:
    topics:
      - str
    topic_descriptions:
      <topic>: str

  generation_policy:
    candidate_pool_multiplier: float
    diversity_level: str
    evidence_strictness: str
    reasoning_depth: str

  validation_policy:
    profile: str
    llm_enabled: bool
    dedup_enabled: bool

  evaluation_policy:
    enabled: bool
    candidate_models:
      - str
    judge_enabled: bool
    eval_scope: str
```

## 18.3 Feedback MVP

```yaml
GeneratorFeedback:
  generated_count: int
  target_candidate_count: int
  by_topic: dict
  unfilled_gaps: dict
  failure_reasons: dict
  cost: dict

ValidatorFeedback:
  passed_count: int
  pass_rate: float
  remaining_gaps: dict
  by_topic: dict
  by_difficulty: dict
  diversity_summary: dict
  rejection_summary: dict
  cost: dict

EvaluatorFeedback:
  by_topic: dict
  by_difficulty: dict
  model_summary: dict
  item_diagnostics: dict
  cost: dict
```

---

# 19. 最终字段边界

## 19.1 放在 GlobalBlueprint

```text
用户目标
语言
最大进化轮次
最终题型数量
初始主题
终止条件
用户显式约束
```

## 19.2 放在 PlannerState

```text
当前进度
主题历史表现
难度校准
多样性状态
模型池状态
预算状态
历史决策
```

## 19.3 放在 RoundPlan

```text
本轮题量
本轮难度比例
本轮主题描述
candidate_pool_multiplier
diversity_level
novelty_level
evidence_strictness
reasoning_depth
验证 profile 和开关
评估模型池和 scope
本轮预算
```

## 19.4 放在 Agent YAML

```text
chunk size
retrieval top-k
rerank 参数
single_k / multi_k 推导细节
prompt 路径
temperature
max_tokens
max_retries
max_concurrency
默认验证阈值
默认指标列表
metric description / direction
```

## 19.5 放在 shared_state

```text
task_id
run_id
当前 run 的轻量 blueprint
artifact 路径
feedback 路径
```

---

# 20. 总结

规划智能体的核心职责是：

```text
根据用户目标建立极简总蓝图；
维护跨轮次 PlannerState；
每轮生成 RoundPlan；
把 RoundPlan 拆成三个 AgentPlan；
读取三个智能体的结构化反馈；
根据通过率、多样性、引用质量、模型挑战度、模型区分度、预算压力更新下一轮计划。
```

规划智能体不应该成为底层参数生成器。
它应该控制少量高价值策略参数：

```text
topic_descriptions
round_targets
difficulty_distribution
candidate_pool_multiplier
diversity_level
novelty_level
evidence_strictness
reasoning_depth
llm_enabled
dedup_enabled
candidate_models
judge_enabled
eval_scope
```

每个智能体应该返回 planner 能直接用于决策的聚合反馈，而不是完整日志。

最终形成：

```text
轻量总蓝图
+ 跨轮次 PlannerState
+ 每轮动态 RoundPlan
+ 三类 AgentPlan/Feedback
+ shared_state artifact 兼容
```
