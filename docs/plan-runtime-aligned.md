# BenchForge 运行对齐版规划方案

## 1. 目标

这份方案的目标不是继续扩展抽象 schema，而是给出一版和当前仓库实现严格对齐、可复现、可成功运行的规划协议。

约束如下：

1. 不要求重写 `qa_agent`、`verify_agent`、`model_eval_agent` 主体逻辑。
2. 允许增加一层很薄的 planner 适配层。
3. planner 只下发当前代码真实可消费的参数。
4. planner 的反馈只依赖当前运行后真实落盘的 artifact 和 report。

这意味着：

- planner 可以生成 `RoundSpec`。
- planner 可以为每轮生成三份 config patch。
- orchestrator 负责把 base config 和 patch 深度合并，落成每轮生效的 config。
- 子智能体继续按当前入口函数运行。

---

## 2. 当前三个智能体真实需要的参数

### 2.1 `qa_agent`

在统一编排协议下，`qa_agent` 的运行输入分成两部分：

1. `RoundSpec.blueprint`
2. `qa_agent.yaml` 加载出的 agent config

这里也要区分两层：

1. 协议层
2. 当前核心函数签名

协议层应当是：

- planner 先产出本轮 `RoundSpec`
- `RoundSpec.blueprint` 是本轮执行蓝图的唯一真源
- orchestrator 从 `RoundSpec.blueprint` 反序列化出 `Blueprint`
- 再调用现有 `run_generation_agent(blueprint=..., config=...)`
- 如需兼容当前下游实现，可把同一份蓝图镜像写入 `shared_state.blueprint_cache`

当前仓库中的 `qa_agent` 核心函数签名仍然是：

1. `Blueprint`
2. `qa_agent` config

因此这里的“统一编排协议”指的是：

- 计划真源在 `RoundSpec`
- 运行状态真源在 `shared_state`
- 当前核心函数仍接受 `Blueprint` 对象

当前 `Blueprint` 必填字段如下：

```yaml
Blueprint:
  task_id: str
  run_id: str
  language: str
  topics: list[str]
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
```

`qa_agent` 当前不会直接读取 planner 自定义字段，例如：

- `phase`
- `topic_actions`
- `diversity_level`
- `reasoning_depth`
- `allowed_question_types`

这些字段如果保留，只能存在于 planner 内部，不能直接期待 `qa_agent` 消费。

`qa_agent` 的 base config 当前由 `config/qa_agent.yaml` 提供。

这里要区分两层：

1. 运行时可 patch 的上界
2. planner 真正应根据反馈调整的最小参数集

运行时可 patch 的上界如下：

```yaml
GeneratorConfigPatch:
  model:
    name: str | null
    temperature: float | null
    max_tokens: int | null
    max_retries: int | null

  retrieval:
    saliency_top_k: int | null
    min_paragraph_tokens: int | null

  candidate_pool:
    target_multiplier: float | null

  initial_breadth:
    enabled: bool | null
    max_topics_per_round: int | null
    difficulty: easy | medium | hard | null

  planner:
    topics_per_round: int | null

  runtime:
    max_consecutive_empty_rounds_per_mode: int | null
    max_failures_per_mode: int | null
```

这几个 patch 字段都能被现有 loader 直接读取并生效。

但 planner 真正应该自适应控制的只建议保留：

```yaml
GeneratorAdaptivePatch:
  candidate_pool:
    target_multiplier: float | null

  initial_breadth:
    enabled: bool | null

  planner:
    topics_per_round: int | null
```

不建议交给 planner 动态生成的字段：

- `model.*`
- `retrieval.min_paragraph_tokens`
- `runtime.*`

这些更适合保留在 base YAML 中做默认工程配置。

`topics / count / difficulty_distribution` 不应放在 `GeneratorAdaptivePatch` 中：

1. 它们属于 `RoundSpec.blueprint`，而不是 `qa_agent` config patch。
2. `run_generation_agent(blueprint=..., config=...)` 当前只从函数参数接收蓝图，不会从 `qa_agent` config 中读取 `blueprint.*`。
3. 因此 planner 对 topic 和 mode 的调整必须直接写入 `RoundSpec.blueprint`，否则会形成双真源并被静默忽略。

### 2.2 `verify_agent`

`verify_agent` 的真实运行输入分成两部分：

1. `shared_state.json`
2. `verify_agent.yaml` 加载出的 config

当前 `verify_agent` 从 `shared_state.json` 中只恢复这些信息：

```yaml
SharedState:
  task_id: str
  round_id: int
  run_id: str
  round_spec_ref: str
  blueprint_cache: dict | null
  artifacts:
    qa_candidate_pool: str | null
    multiple_choice_candidate_pool: str | null
    chunked_evidence: str | null
```

统一协议下更合理的恢复顺序应当是：

1. 从 `shared_state.round_spec_ref` 找到 `RoundSpec`
2. 读取 `RoundSpec.blueprint`
3. 仅在兼容模式下，才回退到 `shared_state.blueprint_cache`

也就是说，`verify_agent` 当前真正需要的是：

- 候选题路径
- `Blueprint` 里的 `topics`
- `Blueprint` 里的 `modes.{mode}.count`
- `Blueprint` 里的 `modes.{mode}.difficulty_distribution`

`verify_agent` 的 base config 当前由 `config/verify_agent.yaml` 提供。

运行时可 patch 的上界如下：

```yaml
ValidatorConfigPatch:
  citation_validation:
    enabled: bool | null
    min_citation_score: float | null
    alpha: float | null
    beta: float | null
    citation_match_threshold: float | null

  llm_validation:
    enabled: bool | null
    model: str | null
    temperature: float | null
    max_tokens: int | null
    min_overall_score: float | null
    max_concurrency: int | null
    max_retries: int | null
    hard_floor:
      clarity: float | null
      answerability: float | null
      faithfulness: float | null
      mode_alignment: float | null
    prompt_system: str | null
    prompt_user: str | null

  selection:
    enabled: bool | null
    embedding_model: str | null
```

但 planner 真正应该自适应控制的只建议保留：

```yaml
ValidatorAdaptivePatch:
  citation_validation:
    min_citation_score: float | null

  llm_validation:
    enabled: bool | null
    min_overall_score: float | null
    hard_floor:
      clarity: float | null
      answerability: float | null
      faithfulness: float | null
      mode_alignment: float | null
```

不建议交给 planner 动态生成的字段：

- `llm_validation.model`
- `llm_validation.prompt_system`
- `llm_validation.prompt_user`
- `selection.embedding_model`

这些应该作为默认配置固定在 YAML 中。

### 2.3 `model_eval_agent`

`model_eval_agent` 的真实运行输入分成两部分：

1. `shared_state.json`
2. `model_eval_agent.yaml` 加载出的 config

当前 `model_eval_agent` 从 `shared_state.json` 中主要恢复：

```yaml
SharedState:
  task_id: str
  round_id: int
  run_id: str
  round_spec_ref: str
  artifacts:
    validated_questions: str | null
```

然后从 `validated_questions.jsonl` 中只读取 `final_status == "selected"` 的题目作为评估输入。

这意味着：

1. 如果没有 `validated_questions` artifact，评估无法启动。
2. 如果 `validated_questions.jsonl` 里没有 `selected` 题目，评估会报错。
3. orchestrator 必须在 `selected_question_ids` 为空时跳过评估。

`model_eval_agent` 的 base config 当前由 `config/model_eval_agent.yaml` 提供。

运行时可 patch 的上界如下：

```yaml
EvaluatorConfigPatch:
  dataset_evaluation:
    enabled: bool | null
    metrics:
      - name: str
        threshold: float | null

  models:
    candidate_model_names: list[str] | null
    judge_model_name: str | null
    generation_defaults:
      temperature: float | null
      top_p: float | null
      max_tokens: int | null
    judge_defaults:
      temperature: float | null
      top_p: float | null
      max_tokens: int | null

  metrics:
    qa:
      automatic_metrics:
        - name: str
          threshold: float | null
      llm_judge_metrics:
        - name: str
          description: str
    multiple_choice:
      automatic_metrics:
        - name: str
          threshold: float | null
      llm_judge_metrics:
        - name: str
          description: str

  judge:
    enabled: bool | null
    prompt_system: str | null
    prompt_user: str | null
```

但 planner 真正应该自适应控制的只建议保留：

```yaml
EvaluatorAdaptivePatch:
  models:
    candidate_model_names: list[str] | null

  judge:
    enabled: bool | null

  dataset_evaluation:
    enabled: bool | null
```

其中：

- `eval_profile` 不属于 runtime patch
- 它应放在 `RoundSpec.planner_hints` 中，由 orchestrator 翻译成固定的 metrics 组合和 token 上限

不建议交给 planner 动态生成的字段：

- `judge.prompt_system`
- `judge.prompt_user`
- 固定 automatic metric 的默认集合
- `llm_judge_metrics`

关于 `llm_judge_metrics` 的约束：

1. `v1` 只要求 `name + description`。
2. 这些 metric 定义应放在 `GlobalBlueprint.evaluation_requirements` 中全局固定。
3. 题目评估智能体的固定 prompt 模板统一规定所有 judge metric 都输出 `0.0 ~ 1.0`。
4. 所有 judge metric 必须遵守“分数越高越好”的语义。
5. 因此不再单独保留 `direction` 字段。
6. 名称应避免 `hallucination`、`error_rate` 这类天然更像“越低越好”的词，建议统一使用：
   - `correctness`
   - `faithfulness`
   - `completeness`
   - `groundedness`
   - `reasoning_quality`

---

## 3. 规划层应该保留什么，不应该保留什么

### 3.1 应保留

planner 层应该保留三类信息：

1. 全局目标
2. 每轮要生成的有效 `RoundSpec`
3. 每轮要应用到三份 agent config 上的 patch

### 3.2 不应直接下发给 agent

以下字段即使 planner 内部想保留，也不应直接下发给现有 agent：

- `diversity_level`
- `novelty_level`
- `evidence_strictness`
- `reasoning_depth`
- `validation_policy.profile`
- `evaluation_policy.metrics_profile`
- `evaluation_policy.eval_scope`
- `topic_actions`

这些字段如果要保留，只能作为 planner 内部决策标签，最终必须翻译成：

- `Blueprint` 的 `topics / count / max_rounds / difficulty_distribution`
- 三份 YAML patch 的具体字段

---

## 4. 新的最小可运行规划协议

## 4.1 `GlobalBlueprint`

`GlobalBlueprint` 只保存全局目标和默认运行基线，不保存每轮动态策略。

```yaml
GlobalBlueprint:
  task_id: str
  blueprint_id: str
  user_goal: str
  language: str

  seed_topics:
    - str

  final_targets:
    qa: int
    multiple_choice: int

  default_modes:
    qa:
      max_rounds: int
      difficulty_distribution:
        easy: float
        medium: float
        hard: float
    multiple_choice:
      max_rounds: int
      difficulty_distribution:
        easy: float
        medium: float
        hard: float

  evaluator_defaults:
    candidate_model_names:
      - str
    judge_model_name: str | null

  evaluation_requirements:
    automatic_metrics:
      qa:
        - str
      multiple_choice:
        - str
    llm_judge_metrics:
      qa:
        - name: str
          description: str
      multiple_choice:
        - name: str
          description: str

  stop_conditions:
    max_rounds: int
    min_selected_per_round: int
    max_total_tokens: int | null
```

说明：

- `final_targets` 直接表达最终要保留多少 QA 和选择题。
- 不再保留 `allowed_question_types`。
- 不再保留 `require_capability_label` 这种当前 agent 不稳定消费的字段。
- evaluator 的默认模型池可以放在这里，因为后续会被翻译成 `EvaluatorConfigPatch`。
- `evaluation_requirements` 表达用户明确要求必须保留的评估内容。
- `evaluation_requirements` 在一次 planning run 内应视为全局固定评估协议，不作为逐轮自适应项。
- 如果启用 LLM judge，judge metric 的定义必须来自这里，而不是每轮动态改写。

## 4.2 `PlannerState`

`PlannerState` 只保存当前实现真正需要的跨轮次信息。

```yaml
PlannerState:
  task_id: str
  blueprint_id: str
  current_round: int

  completed_targets:
    qa: int
    multiple_choice: int

  resource_usage:
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    by_agent:
      qa_agent:
        input_tokens: int
        output_tokens: int
      verify_agent:
        input_tokens: int
        output_tokens: int
      model_eval_agent:
        input_tokens: int
        output_tokens: int

  topic_backlog:
    active:
      - str
    deferred:
      - str

  model_pool:
    active:
      - str
    removed:
      - str

  run_history:
    - round_id: int
      run_id: str
      topics:
        - str
      qa_target: int
      multiple_choice_target: int
      selected_count: int
      evaluated: bool
```

说明：

- 不再保留复杂的 `topic_memory.*_history`。
- 不再保留抽象的 `phase` 和 `topic_action`。
- `model_pool` 只做增删，不做复杂淘汰原因推理。
- 最近一轮的 `run_id / selected_count / evaluated` 统一从 `run_history[-1]` 读取，不再单独维护 `last_run`。
- `resource_usage` 应从 `shared_state.artifacts.llm_calls` 读取并聚合，而不是依赖各 agent report 的局部 token 字段。
- 当前代码的模型调用会通过 `Span` 或 `llm_trace_path` 自动写入 `llm_calls.jsonl`，因此可以支持全链路 token 预算。
- 如果某个底层 provider 不返回 usage，相关调用会记为 `0`，这是 token 预算的唯一已知精度边界。

## 4.3 `RoundSpec`

`RoundSpec` 是 planner 每轮唯一需要产出的核心对象。

```yaml
RoundSpec:
  task_id: str
  blueprint_id: str
  round_id: int
  run_id: str
  objective: str

  blueprint:
    task_id: str
    run_id: str
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

  qa_agent_patch: GeneratorAdaptivePatch
  verify_agent_patch: ValidatorAdaptivePatch
  model_eval_agent_patch: EvaluatorAdaptivePatch
  planner_hints: RoundPlannerHints
```

`RoundSpec` 的设计原则：

1. `RoundSpec.blueprint` 必须可以直接反序列化构造成 `qa_agent.schema.Blueprint`。
2. 三份 patch 必须可以直接和三份 base YAML 深度合并。
3. 所有运行决策都必须体现在 `blueprint`、patch 或 `planner_hints` 里，不能只存在于解释文字中。
4. judge 指标定义来自 `GlobalBlueprint.evaluation_requirements.llm_judge_metrics`，实际 prompt 由 orchestrator 注入固定模板。

其中：

- `qa_agent_patch / verify_agent_patch / model_eval_agent_patch` 都必须是可直接深度合并的 runtime patch。
- `planner_hints` 只保存无法直接深度合并、必须由 orchestrator 二次翻译的 hint；当前仅保留 `eval_profile`。

```yaml
RoundPlannerHints:
  eval_profile: light | standard | full | null
```

## 4.4 `SharedState`

`SharedState` 只保存当前轮次运行状态，不再把完整蓝图作为主真源。

```yaml
SharedState:
  task_id: str
  round_id: int
  run_id: str

  round_spec_ref: str

  # 仅兼容当前 agent，非真源
  blueprint_cache: dict | null

  artifacts:
    <artifact_key>: str

  agent_status:
    generation: pending | running | completed | failed
    verification: pending | running | completed | failed
    evaluation: pending | running | completed | failed
```

设计原则：

1. `RoundSpec` 表达“本轮应该怎么跑”。
2. `SharedState` 表达“本轮已经跑到哪里”。
3. planner 只修改 `RoundSpec`，agent 只回写 `SharedState`。
4. `blueprint_cache` 只是兼容当前实现的派生副本，不是真源。
5. 当三个 agent 都支持 `round_spec_ref -> RoundSpec.blueprint` 恢复链路后，`blueprint_cache` 应在下一版本废弃并删除。

---

## 5. 每轮运行时必须落盘的文件

为了保证完全可复现，planner 和 orchestrator 每轮至少要保存以下文件：

```text
runs/{task_id}/{run_id}/planner/
  global_blueprint.json
  planner_state_before.json
  round_spec.json
  qa_agent.patch.yaml
  verify_agent.patch.yaml
  model_eval_agent.patch.yaml
  translated_model_eval.patch.yaml
  effective_qa_agent.yaml
  effective_verify_agent.yaml
  effective_model_eval_agent.yaml
  planner_state_after.json
```

约定：

1. `round_spec.json` 是本轮计划真源，其中 `RoundSpec.blueprint` 是唯一执行蓝图真源。
2. `translated_model_eval.patch.yaml` 是 orchestrator 根据 `RoundSpec.planner_hints` 衍生出的 evaluator 运行时 patch。
3. `effective_*.yaml` 是 base config 和 patch 深度合并后的结果。
4. 只要这四个文件存在：
   - `round_spec.json`
   - `effective_qa_agent.yaml`
   - `effective_verify_agent.yaml`
   - `effective_model_eval_agent.yaml`
   就能重放整轮运行。

## 5.1 `shared_state` 的职责

`shared_state.json` 不只是下游 agent 的恢复入口，也应该是跨 agent 输入输出协调的唯一公共状态。

建议约束为：

1. planner / orchestrator 在本轮开始前就应初始化 `shared_state.json`，并写入 `round_spec_ref`。
2. 所有跨阶段输入路径都优先从 `shared_state.artifacts` 读取。
3. 所有关键输出都必须回写到 `shared_state.artifacts`。
4. planner 的 feedback parser 也优先从 `shared_state.artifacts` 读取，而不是猜目录结构。
5. 全链路 token 预算统计的权威来源应是 `shared_state.artifacts.llm_calls`。

如果当前实现尚未支持从 `round_spec_ref` 解析蓝图，则允许同时写入：

```yaml
blueprint_cache: <RoundSpec.blueprint 的镜像副本>
```

但这只是兼容缓存，不应被视为计划真源。

推荐至少维护这些 artifact key：

```yaml
artifacts:
  qa_candidate_pool: str | null
  multiple_choice_candidate_pool: str | null
  qa_mode_state: str | null
  multiple_choice_mode_state: str | null
  chunked_evidence: str | null
  llm_calls: str | null
  generation_report: str | null
  validated_questions: str | null
  validation_report: str | null
  weighted_selection: str | null
  evaluation_report: str | null
  dataset_quality_summary: str | null
  model_overall_report: str | null
  model_aggregate_report: str | null
  model_by_topic: str | null
  model_by_difficulty: str | null
  model_by_question_mode: str | null
```

当前代码里已经有一部分机制：

- `verify_agent` 会回写 `validated_questions`
- `model_eval_agent` 会回写 `evaluation_report`

当前实现与统一编排协议之间的差异是：

- 现在仓库里的 `verify_agent` 仍直接从 `shared_state.blueprint` 取蓝图视图
- 统一编排协议下，应改为优先通过 `round_spec_ref -> RoundSpec.blueprint` 恢复蓝图，必要时回退到 `blueprint_cache`
- 全链路 token 预算应统一读取运行根目录下的 `llm_calls`；像 `model_eval_agent` 这种内部局部 trace 路径概念，后续应收敛到同一个公共 artifact key

如果要让 planner 完全依赖公共状态，后续应补充：

- 增加一个 `qa_agent` 适配入口：从 `round_spec_ref` 读取 `RoundSpec.blueprint` 后调用现有核心函数
- 给 `verify_agent` 增加同样的恢复逻辑
- `verify_agent` 回写 `validation_report`、`weighted_selection`
- `model_eval_agent` 回写 `dataset_quality_summary`、`model_overall_report` 等关键报告引用

`blueprint_cache` 的退场计划：

1. `v1` 保留 `blueprint_cache` 作为兼容副本。
2. `v2` 当 `qa_agent`、`verify_agent`、`model_eval_agent` 都支持 `round_spec_ref -> RoundSpec.blueprint` 恢复后，停止写入 `blueprint_cache`。
3. 下一次 schema 升级时，删除 `SharedState.blueprint_cache` 字段。

---

## 6. 运行流程

### 6.1 编排流程

```python
def run_round(round_spec):
    materialize_effective_agent_configs(round_spec)
    initialize_shared_state(round_spec)

    gen_report = run_qa_agent(round_spec)
    gen_feedback = build_generator_feedback(round_spec)

    if gen_feedback.summary.total_candidates == 0:
        return gen_feedback, None, None

    val_result = run_verify_agent(round_spec)
    val_feedback = build_validator_feedback(round_spec, val_result)

    if val_feedback.summary.final_selected == 0:
        return gen_feedback, val_feedback, None

    eval_result = run_model_eval_agent(round_spec)
    eval_feedback = build_evaluator_feedback(round_spec, eval_result)
    return gen_feedback, val_feedback, eval_feedback
```

其中 `materialize_effective_agent_configs(round_spec)` 必须先把 `round_spec.planner_hints` 翻译成衍生 patch，再与三份 base YAML 合并。

### 6.2 关键跳过规则

必须和当前实现保持一致：

1. 如果生成阶段 `total_candidates == 0`，跳过验证和评估。
2. 如果验证阶段 `final_selected == 0`，跳过评估。
3. 不能在没有 `selected` 题目的情况下调用 `model_eval_agent`。
4. `verify_agent` 和 `model_eval_agent` 的输入路径优先从 `shared_state.artifacts` 解析。

### 6.3 planner 跨轮主循环

planner 的核心逻辑不在单轮编排里，而在轮次边界。

```python
def run_planner(global_blueprint, planner_state):
    while True:
        if planner_state.current_round >= global_blueprint.stop_conditions.max_rounds:
            break

        if targets_satisfied(planner_state.completed_targets, global_blueprint.final_targets):
            break

        if token_budget_exhausted(
            planner_state.resource_usage,
            global_blueprint.stop_conditions.max_total_tokens,
        ):
            break

        summary = summarize_planner_state(planner_state)
        latest_feedback = load_latest_feedback_from_shared_state(planner_state)

        llm_topic_plan = propose_topics_with_llm(
            global_blueprint=global_blueprint,
            planner_state_summary=summary,
            latest_feedback=latest_feedback,
        )

        round_spec = build_next_round_spec(
            global_blueprint=global_blueprint,
            planner_state=planner_state,
            latest_feedback=latest_feedback,
            llm_topic_plan=llm_topic_plan,
        )

        gen_feedback, val_feedback, eval_feedback = run_round(round_spec)

        planner_state = update_planner_state(
            planner_state=planner_state,
            round_spec=round_spec,
            gen_feedback=gen_feedback,
            val_feedback=val_feedback,
            eval_feedback=eval_feedback,
        )
```

约束：

1. 终止条件只在 planner 进入下一轮前判断。
2. 一旦某个 `RoundSpec` 开始执行，就不允许因为全局终止条件中途截断子智能体。
3. LLM 负责生成下一轮 topic / subtopic / additional requirement 提案。
4. 题量缺口、阈值、倍率、开关等数值参数仍由规则系统决定。
5. `build_next_round_spec(...)` 必须产出可直合并 patch；只有 `eval_profile` 这类无法直接深度合并的 hint 才进入 `planner_hints`。
6. `token_budget_exhausted(...)` 必须基于 `shared_state.artifacts.llm_calls` 的聚合结果判断，而不是只看某个 agent 的局部 report。

---

## 7. 新的反馈定义

新反馈的原则是：

1. 尽量直接复用现有 report。
2. planner 只读取自己需要的聚合字段。
3. 动态明细尽量以 artifact 路径形式回传，而不是再定义一套膨胀 schema。
4. token 用量的全局聚合应直接从 `llm_calls` trace 读取，不应让 planner 依赖各 feedback summary 重复计数。

## 7.1 `GeneratorFeedback`

数据来源：

- `shared_state.artifacts.llm_calls`
- `shared_state.artifacts.generation_report`
- `shared_state.artifacts.qa_candidate_pool`
- `shared_state.artifacts.multiple_choice_candidate_pool`
- `shared_state.artifacts.qa_mode_state`
- `shared_state.artifacts.multiple_choice_mode_state`
- `shared_state.artifacts.chunked_evidence`

```yaml
GeneratorFeedback:
  task_id: str
  run_id: str
  round_id: int
  status: success | partial_success | failed

  artifacts:
    shared_state_path: str
    generation_report: str
    qa_candidate_pool: str | null
    multiple_choice_candidate_pool: str | null
    qa_mode_state: str | null
    multiple_choice_mode_state: str | null
    chunked_evidence: str | null

  summary:
    total_candidates: int
    global_used_chunk_combinations: int
    global_failures: int
    llm_input_tokens: int
    llm_output_tokens: int

  by_mode:
    qa:
      candidate_count: int
      target_candidate_count: int
      fulfillment_rate: float
      stopped_reason: str | null
      difficulty_counts: dict[str, int]
      topic_counts: dict[str, int]
    multiple_choice:
      candidate_count: int
      target_candidate_count: int
      fulfillment_rate: float
      stopped_reason: str | null
      difficulty_counts: dict[str, int]
      topic_counts: dict[str, int]

  topic_coverage:
    <topic>:
      candidate_count: int
```

说明：

- `topic_counts` 和 `difficulty_counts` 来自各 mode 的 `mode_state` artifact。
- 不再虚构 `estimated_duplicate_rate` 这类当前生成阶段拿不到的字段。
- `fulfillment_rate = candidate_count / max(target_candidate_count, 1)`。
- `topic_coverage` 是把两个 mode 的 `topic_counts` 汇总后得到的简单覆盖计数。
- `llm_input_tokens / llm_output_tokens` 应从 `llm_calls` 中按 `agent=qa_agent` 聚合得到，并覆盖生成调用与文档摘要调用。

## 7.2 `ValidatorFeedback`

数据来源：

- `shared_state.artifacts.llm_calls`
- `shared_state.artifacts.validation_report`
- `shared_state.artifacts.validated_questions`
- `shared_state.artifacts.weighted_selection`

```yaml
ValidatorFeedback:
  task_id: str
  run_id: str
  round_id: int
  status: success | partial_success | failed

  artifacts:
    validation_report: str
    validated_questions: str
    weighted_selection: str

  summary:
    total_candidates: int
    citation_passed: int
    llm_passed: int
    final_selected: int
    citation_pass_rate: float
    llm_pass_rate_after_citation: float
    final_selection_rate: float
    llm_calls: int
    llm_input_tokens: int
    llm_output_tokens: int

  quality_signals:
    avg_citation_score_all: float | null
    avg_citation_score_selected: float | null
    avg_llm_overall_score_all: float | null
    avg_llm_overall_score_selected: float | null
    duplicate_rate: float
    overquota_rate: float

  failed_by_stage:
    citation: int
    llm: int
    validator_error: int
    duplicate: int

  by_final_status:
    selected: int
    reserve: int
    duplicate: int
    rejected_citation: int
    rejected_llm: int
    validator_error: int

  by_mode:
    qa:
      selected: int
      reserve: int
      rejected: int
    multiple_choice:
      selected: int
      reserve: int
      rejected: int

  by_difficulty:
    easy:
      selected: int
      reserve: int
      rejected: int
    medium:
      selected: int
      reserve: int
      rejected: int
    hard:
      selected: int
      reserve: int
      rejected: int

  by_topic:
    <topic>:
      selected: int
      reserve: int
      rejected: int
      avg_citation_score: float | null
      avg_llm_overall_score: float | null

  selection_summary:
    dropped_as_duplicate: int
    dropped_as_overquota: int
```

说明：

1. `validation_report` 已经提供了总数、阶段通过数、失败分布；其 `llm_usage` 可与 `llm_calls` 的 `agent=verify_agent` 聚合结果交叉校验。
2. `validated_questions` 足以回算 `by_final_status / by_mode / by_difficulty / by_topic`，也足以从逐题记录中统计平均 `citation_score` 和 `overall_score`。
3. `weighted_selection` 足以拿到 `dropped_as_duplicate / dropped_as_overquota`。
4. 这些字段已经足以驱动下一轮生成/验证参数，不需要再发明更复杂的验证反馈 schema。

## 7.3 `EvaluatorFeedback`

数据来源：

- `shared_state.artifacts.llm_calls`
- `shared_state.artifacts.evaluation_report`
- `shared_state.artifacts.dataset_quality_summary`
- `shared_state.artifacts.model_overall_report`
- `shared_state.artifacts.model_aggregate_report`
- `shared_state.artifacts.model_by_topic`
- `shared_state.artifacts.model_by_difficulty`
- `shared_state.artifacts.model_by_question_mode`

```yaml
EvaluatorFeedback:
  task_id: str
  run_id: str
  round_id: int
  status: success | partial_success | failed

  artifacts:
    evaluation_report: str
    dataset_quality_summary: str
    model_overall_report: str
    model_aggregate_report: str
    model_by_topic: str
    model_by_difficulty: str
    model_by_question_mode: str

  summary:
    num_questions: int
    num_models: int
    llm_input_tokens: int
    llm_output_tokens: int

  dataset_signals:
    citation_score_mean: float | null
    citation_pass_rate: float | null
    diversity_score: float | null
    embedding_dispersion: float | null
    cluster_entropy: float | null

  derived_performance_signals:
    by_question_mode:
      qa:
        avg_exact_match: float | null
        avg_f1: float | null
        avg_judge_correctness: float | null
        avg_judge_faithfulness: float | null
      multiple_choice:
        avg_accuracy: float | null
        avg_judge_correctness: float | null
    by_difficulty:
      easy:
        avg_accuracy_like: float | null
      medium:
        avg_accuracy_like: float | null
      hard:
        avg_accuracy_like: float | null
    by_topic:
      <topic>:
        avg_accuracy_like: float | null
        avg_judge_correctness: float | null

  dataset_metrics_ref:
    summary_json: str

  model_metrics_ref:
    overall_csv: str
    aggregate_json: str
    by_topic_csv: str
    by_difficulty_csv: str
    by_question_mode_csv: str
```

说明：

1. `dataset_signals` 直接来自 `shared_state.artifacts.dataset_quality_summary`。
2. `derived_performance_signals` 由 planner parser 从 `shared_state.artifacts.model_overall_report`、`shared_state.artifacts.model_by_difficulty`、`shared_state.artifacts.model_by_topic` 按当前 active model pool 做均值聚合得到。
3. 当前实现没有 `by_capability` 聚合，因此新反馈不能宣称支持能力维度评估。
4. 评估原始指标仍保留在 artifact 中，planner 只消费少量归一化信号。
5. 如果 `judge.enabled = true`，则 orchestrator 应从 `GlobalBlueprint.evaluation_requirements.llm_judge_metrics` 读取并注入 evaluator prompt。
6. `llm_input_tokens / llm_output_tokens` 应从 `llm_calls` 中按 `agent=model_eval_agent` 聚合得到，并同时覆盖 inference 与 judge 两个 stage。

---

## 8. 反馈应该影响哪些参数

原则只有一句：

- 只有当某个反馈信号能稳定映射到下一轮一个可执行调整动作时，它才应该进入 planner 主反馈。

## 8.1 生成阶段反馈 -> 参数

| 反馈信号 | 计算方式 | 下一轮应影响的参数 | 解释 |
| --- | --- | --- | --- |
| `by_mode.*.fulfillment_rate` 低 | `candidate_count / target_candidate_count` | `candidate_pool.target_multiplier` | 候选池不足，先增加过生成倍率 |
| `topic_coverage` 过于集中 | 单一 topic 占比过高 | `blueprint.topics`、`initial_breadth.enabled`、`planner.topics_per_round` | 覆盖太窄时优先换 topic 和增加广度，不要只盲目放大候选池 |
| `stopped_reason` 为连续空轮 | 来自 `qa_mode_state / multiple_choice_mode_state` | `blueprint.topics`、`planner.topics_per_round` | 说明当前 topic 或配额分配有问题，应换 topic 或收缩并发主题数 |

## 8.2 验证阶段反馈 -> 参数

| 反馈信号 | 计算方式 | 下一轮应影响的参数 | 解释 |
| --- | --- | --- | --- |
| `final_selection_rate` 低且 `citation_pass_rate` 也低 | `final_selected / total_candidates`，`citation_passed / total_candidates` | `blueprint.topics`、`difficulty_distribution` | 现有生成题质量差，先换 topic 或降低 hard 比例，比直接放松 validator 更合理 |
| `final_selection_rate` 低但 `citation_pass_rate` 高、`llm_pass_rate_after_citation` 低 | 来自 summary | `difficulty_distribution`、必要时 `llm_validation.min_overall_score` | 说明题有证据但质量/可答性差，应优先调难度或在探索轮稍微放宽 LLM 门槛 |
| `duplicate_rate` 高 | `dropped_as_duplicate / total_candidates` | `blueprint.topics`、`planner.topics_per_round`、`candidate_pool.target_multiplier` | 重复太高时不要继续加倍生成，优先扩 topic，必要时下调倍率 |
| `overquota_rate` 高 | `dropped_as_overquota / max(llm_passed, 1)` | `candidate_pool.target_multiplier`、`mode.count` | 说明供给明显超过选题配额，可减少过生成 |
| `avg_citation_score_selected` 高但 `final_selected` 低 | 来自逐题记录 | `llm_validation.enabled`、`llm_validation.min_overall_score` | 在探索轮可以考虑关闭 LLM 验证或稍降门槛；在收敛轮不建议放松 |
| `by_topic.*.selected` 长期为 0 | 按 topic 聚合 | `blueprint.topics` | 直接移除或替换该 topic |
| `by_topic.*.avg_citation_score` 长期低 | 按 topic 聚合 | `blueprint.topics` | 当前系统下生成器无法直接调证据策略，最有效动作是换更可证实的 topic |

## 8.3 评估阶段反馈 -> 参数

| 反馈信号 | 计算方式 | 下一轮应影响的参数 | 解释 |
| --- | --- | --- | --- |
| `by_difficulty.easy.avg_accuracy_like` 很高 | 聚合 `shared_state.artifacts.model_by_difficulty` | `difficulty_distribution` | easy 太简单，降低 easy 比例 |
| `by_difficulty.medium.avg_accuracy_like` 很高 | 同上 | `difficulty_distribution` | medium 实际偏简单，提高 hard 比例 |
| `by_difficulty.hard.avg_accuracy_like` 极低且验证阶段质量也差 | 结合 evaluator + validator | `difficulty_distribution` | hard 很可能是伪难，降低 hard 比例 |
| `by_topic.*.avg_accuracy_like` 很高 | `shared_state.artifacts.model_by_topic` 聚合 | `blueprint.topics` | 该 topic 太简单，下一轮可替换成更细更难的 topic |
| `dataset_signals.diversity_score` 低 | `shared_state.artifacts.dataset_quality_summary` | `blueprint.topics`、`planner.topics_per_round` | 多样性差时优先扩 topic 覆盖 |
| `num_questions` 很少或预算紧 | 来自 summary 或外部预算 | `judge.enabled`、`planner_hints.eval_profile` | 题少时可用 full；预算紧时切到 light |
| 某模型长期与其他模型无区分价值 | 由 `shared_state.artifacts.model_overall_report` 观察 | `models.candidate_model_names` | 可收缩模型池，节省评估成本 |
## 8.4 不应由反馈自动影响的参数

这些参数默认固定在 YAML 中，不建议每轮自动调：

- prompt 路径
- judge metric 描述文本
- embedding model
- timeout / retry / concurrency 默认值
- retrieval 的底层分块参数
- 默认 automatic metric 集合

`llm_judge_metrics` 也属于这一类：

- 它们应在 `GlobalBlueprint.evaluation_requirements` 中全局固定
- 不能作为每轮自适应参数变化
- 只能由 orchestrator 注入固定 prompt 模板，而不是直接替换 prompt 文件路径

---

## 9. planner 如何把内部策略翻译成当前可运行参数

planner 内部可以保留较少的决策标签，但最终必须翻译成具体 patch。

planner 内部可以使用更多抽象标签辅助决策，但在 `RoundSpec` 里只应保留需要 orchestrator 二次翻译的最小集合。

默认只保留：

```yaml
RoundPlannerHints:
  eval_profile: light | standard | full | null
```

像 `generation_focus`、`validation_strictness`、`model_pool_action` 这类内部标签，应在 planner 内部先翻译成 `blueprint` 或 patch，再写入 `RoundSpec`，不要额外持久化，避免重复表达同一决策。

翻译规则如下。

### 9.1 `generation_focus`

```yaml
coverage:
  qa_agent_patch:
    candidate_pool:
      target_multiplier: 2.5
    initial_breadth:
      enabled: true
    planner:
      topics_per_round: 3

backfill:
  qa_agent_patch:
    candidate_pool:
      target_multiplier: 1.8
    initial_breadth:
      enabled: false
    planner:
      topics_per_round: 1

harder:
  round_spec.blueprint.modes:
    qa:
      difficulty_distribution:
        easy: 0.1
        medium: 0.4
        hard: 0.5
    multiple_choice:
      difficulty_distribution:
        easy: 0.1
        medium: 0.4
        hard: 0.5
  qa_agent_patch:
    candidate_pool:
      target_multiplier: 2.8
```

### 9.2 `validation_strictness`

```yaml
low:
  verify_agent_patch:
    citation_validation:
      min_citation_score: 0.60
    llm_validation:
      enabled: false

medium:
  verify_agent_patch:
    citation_validation:
      min_citation_score: 0.65
    llm_validation:
      enabled: true
      min_overall_score: 0.75

high:
  verify_agent_patch:
    citation_validation:
      min_citation_score: 0.75
    llm_validation:
      enabled: true
      min_overall_score: 0.80
      hard_floor:
        clarity: 0.7
        answerability: 0.75
        faithfulness: 0.8
        mode_alignment: 0.75
```

### 9.3 `eval_profile`

这里的含义是：

- planner 只产出 `eval_profile`
- orchestrator 再把 `eval_profile` 翻译成 `model_eval_agent` 真实配置字段
- 如果 `judge.enabled = true`，orchestrator 还必须从全局评估需求中注入 `llm_judge_metrics`

```yaml
light:
  model_eval_agent_patch:
    dataset_evaluation:
      enabled: true
    models:
      generation_defaults:
        max_tokens: 512
    judge:
      enabled: false

standard:
  model_eval_agent_patch:
    dataset_evaluation:
      enabled: true
    judge:
      enabled: true

full:
  model_eval_agent_patch:
    dataset_evaluation:
      enabled: true
    judge:
      enabled: true
    models:
      generation_defaults:
        max_tokens: 1024
      judge_defaults:
        max_tokens: 1200
```

`llm_judge_metrics` 的注入规则：

```yaml
global_blueprint:
  evaluation_requirements:
    llm_judge_metrics:
      qa:
        - name: correctness
          description: 判断回答是否正确覆盖问题要求的核心事实。
        - name: faithfulness
          description: 判断回答是否被给定证据支持。
      multiple_choice: []
```

orchestrator 执行时：

1. 保留固定 `prompt_system`
2. 保留固定 `prompt_user` 模板
3. 将 `GlobalBlueprint.evaluation_requirements.llm_judge_metrics` 序列化后注入模板变量，例如 `{judge_metrics_spec}`


### 9.4 `model_pool_action`

```yaml
keep:
  model_eval_agent_patch:
    models:
      candidate_model_names: planner_state.model_pool.active

shrink:
  model_eval_agent_patch:
    models:
      candidate_model_names: planner_state.model_pool.active[:2]

expand:
  model_eval_agent_patch:
    models:
      candidate_model_names: planner_state.model_pool.active + [new_model]
```

---

## 10. 一轮完整可运行示例

```yaml
RoundSpec:
  task_id: "bench_001"
  blueprint_id: "bp_001"
  round_id: 1
  run_id: "run_20260609_001"
  objective: "首轮覆盖 AI Safety 和 Quantum Computing，得到一批可验证题目"

  blueprint:
    task_id: "bench_001"
    run_id: "run_20260609_001"
    language: "en"
    topics:
      - "AI Safety"
      - "Quantum Computing"
    modes:
      qa:
        count: 12
        max_rounds: 4
        difficulty_distribution:
          easy: 0.2
          medium: 0.5
          hard: 0.3
      multiple_choice:
        count: 8
        max_rounds: 4
        difficulty_distribution:
          easy: 0.2
          medium: 0.5
          hard: 0.3

  qa_agent_patch:
    candidate_pool:
      target_multiplier: 2.5
    initial_breadth:
      enabled: true
    planner:
      topics_per_round: 2

  verify_agent_patch:
    citation_validation:
      min_citation_score: 0.65
    llm_validation:
      enabled: true
      min_overall_score: 0.75

  model_eval_agent_patch:
    dataset_evaluation:
      enabled: true
    models:
      candidate_model_names:
        - "kimi-k2"
        - "glm-4.7"
    judge:
      enabled: true

  planner_hints:
    eval_profile: full
```

---

## 11. 最小执行伪代码

下面这段伪代码和当前仓库入口函数是一致的：

```python
from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg
from benchforge.agents.qa_agent.config_loader import load_qa_agent_config
from benchforge.agents.qa_agent import run_generation_agent
from benchforge.agents.verify_agent.agent import run_verify_agent_from_shared_state
from benchforge.agents.verify_agent.config_loader import load_verify_agent_config
from benchforge.agents.model_eval_agent.agent import run_model_eval_agent_from_shared_state
from benchforge.agents.model_eval_agent.config_loader import load_model_eval_config


def execute_round(round_spec):
    translated_eval_patch = translate_eval_profile(
        planner_hints=round_spec["planner_hints"],
        global_blueprint=load_global_blueprint(round_spec["blueprint_id"]),
    )

    effective_qa_yaml = merge_yaml("config/qa_agent.yaml", round_spec["qa_agent_patch"])
    effective_verify_yaml = merge_yaml("config/verify_agent.yaml", round_spec["verify_agent_patch"])
    effective_eval_yaml = merge_yaml(
        "config/model_eval_agent.yaml",
        round_spec["model_eval_agent_patch"],
        translated_eval_patch,
    )

    # 先落盘 round_spec.json，再初始化 shared_state.round_spec_ref
    save_runtime_files(round_spec, effective_qa_yaml, effective_verify_yaml, effective_eval_yaml)
    initialize_shared_state(round_spec)

    # 1. generation
    run_generation_agent(...)

    gen_feedback = build_generator_feedback(...)
    if gen_feedback["summary"]["total_candidates"] == 0:
        return

    # 2. verification
    verify_result = run_verify_agent_from_shared_state(...)
    val_feedback = build_validator_feedback(...)
    if val_feedback["summary"]["final_selected"] == 0:
        return

    # 3. evaluation
    run_model_eval_agent_from_shared_state(...)
    eval_feedback = build_evaluator_feedback(...)
```

---

## 12. 结论

运行对齐版 planner 应该做的事情很简单：

1. 生成每轮可直接运行的 `Blueprint`
2. 只生成少量真正需要自适应的 config patch
3. 保存每轮生效的 blueprint 和 config
4. 从现有 report 中提取最小闭环反馈
5. 只根据这些反馈去调整下一轮的 `count / topics / difficulty_distribution / 少量验证与评估参数`

不应该继续扩展的内容包括：

- planner 专属但 agent 不消费的抽象策略字段
- 当前实现拿不到的数据字段
- capability 维度评估
- 评估范围 profile、指标 profile 这类二次包装
- 把 prompt、embedding、超时、重试这类默认工程参数也交给 planner 动态生成

这版方案的重点不是“概念完整”，而是“今天就能接到现有代码上成功跑起来”。
