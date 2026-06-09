# BenchForge 多智能体架构重构方案

## 1. 背景与目标

BenchForge 当前已经具备多阶段、多模块协作的雏形：

```text
User Request
  -> PlannerAgent
  -> QAAgent
  -> VerifyAgent
  -> ModelEvalAgent
  -> Feedback
  -> PlannerAgent next round or finalize
```

当前系统的优势是流程清晰、产物可落盘、每个阶段职责明确，适合 benchmark generation 这类强调可复现、可追踪、可调试的任务。

但当前实现仍偏向硬编码 pipeline，缺少统一的 Agent / Tool / Runtime 抽象，因此在架构表述上容易出现两个问题：

1. `QAAgent`、`VerifyAgent`、`ModelEvalAgent` 内部更像 pipeline，容易被质疑是否是真正的 agent。
2. 系统缺少统一注册机制，Planner 与 worker agents 的调用边界、输入输出 schema、artifact、observation、失败处理不够清晰。

本方案目标不是把现有稳定流程推翻重写，而是在保留当前 pipeline 稳定性的基础上，引入轻量级 multi-agent runtime 抽象，使 BenchForge 可以被准确描述为：

> A planner-driven hierarchical multi-agent workflow for benchmark generation.

中文表述为：

> BenchForge 采用规划智能体驱动的层次化多智能体工作流。PlannerAgent 作为监督/规划智能体负责全局决策和跨轮反馈；QAAgent、VerifyAgent 和 ModelEvalAgent 作为 pipeline-based worker agents，分别完成题目生成、题目验证和模型评估，并通过 AgentTool 接口被 Runtime 调用。

---

## 2. 总体设计原则

### 2.1 不立即整体迁移到外部多智能体框架

短期不建议直接把整个项目迁移到 LangGraph / CrewAI / AutoGen。

原因：

- BenchForge 需要强可复现性和确定性执行顺序。
- 当前已有大量 YAML 配置、shared_state、artifact、round 目录和 planner state。
- 直接迁移外部框架可能会增加复杂度，而不是立即提升系统质量。
- 当前最需要的是先把内部边界抽象清楚。

推荐路线：

```text
现有硬编码 orchestrator
  -> 轻量 AgentRegistry / ToolRegistry / Runtime
  -> LangGraph-compatible state design
  -> 可选 LangGraph backend
```

### 2.2 保留固定主流程，但注册化执行

主流程仍建议保持确定性：

```text
QAAgent -> VerifyAgent -> ModelEvalAgent
```

Planner 主要负责决策：

- 本轮 topic
- 本轮题量
- difficulty mix
- mode mix
- QA config patch
- Verify config patch
- Eval config patch
- 是否继续下一轮
- 是否停止
- 是否请求补充 evidence

Runtime 负责保证质量门控不被随意跳过：

```text
Planner decides strategy.
Runtime enforces workflow.
Worker agents execute pipelines.
Artifacts record facts.
Observations feed back to Planner.
```

### 2.3 区分 Agent、AgentTool、Tool、Runtime Service

不要把所有 `utils/` 函数都注册给 Planner。应建立分层：

```text
Agent:
  有目标、有配置、有输入输出、有 artifact、有 observation 的执行单元。

AgentTool:
  上层 agent / runtime 调用某个 worker agent 的高层接口。

Tool:
  某个 agent 内部使用的小能力，例如采样、过滤、指标计算。

Runtime Service:
  artifact store、state store、trace、path、config loading 等基础设施。
```

---

## 3. 推荐目标架构

```text
BenchForge Runtime
│
├── User Request
│
├── PlannerAgent / SupervisorAgent
│   ├── parse user request
│   ├── maintain PlannerState
│   ├── produce PlannerAction
│   ├── produce RoundSpec
│   ├── produce config patches
│   └── update strategy from Observations
│
├── AgentRegistry
│   ├── planner_agent
│   ├── qa_agent
│   ├── verify_agent
│   └── model_eval_agent
│
├── ToolRegistry
│   ├── planner-visible AgentTools
│   │   ├── run_qa_agent
│   │   ├── run_verify_agent
│   │   ├── run_model_eval_agent
│   │   └── inspect_round_feedback
│   │
│   ├── agent-private tools
│   │   ├── QA tools
│   │   ├── Verify tools
│   │   └── Eval tools
│   │
│   └── runtime services
│       ├── artifact_store
│       ├── state_store
│       ├── tracer
│       └── config_loader
│
├── Worker Agents
│   ├── QAAgent
│   ├── VerifyAgent
│   └── ModelEvalAgent
│
├── ArtifactStore
│   ├── round_spec.json
│   ├── candidate_questions.jsonl
│   ├── generation_report.json
│   ├── validated_questions.jsonl
│   ├── validation_report.json
│   ├── model_eval_report.json
│   └── final_benchmark.jsonl
│
└── Observation Store
    ├── GenerationObservation
    ├── ValidationObservation
    ├── EvaluationObservation
    └── PlannerDecisionTrace
```

---

## 4. 核心抽象定义

### 4.1 AgentSpec

每个 agent 都应该有统一描述：

```python
from dataclasses import dataclass
from typing import Callable, Any, Literal

@dataclass
class AgentSpec:
    name: str
    role: str
    description: str
    autonomy_level: Literal["supervisor", "worker", "evaluator"]
    input_schema: type
    output_schema: type
    configurable_fields: list[str]
    input_artifacts: list[str]
    output_artifacts: list[str]
    run: Callable[..., Any]
```

示例：

```python
AgentSpec(
    name="qa_agent",
    role="Generate candidate QA/MCQ items from evidence according to a round plan.",
    description="Pipeline-based worker agent for question generation.",
    autonomy_level="worker",
    input_schema=QARunInput,
    output_schema=GenerationObservation,
    configurable_fields=[
        "candidate_pool.target_multiplier",
        "planner.topics_per_round",
        "difficulty_mix",
        "initial_breadth.enabled",
    ],
    input_artifacts=[
        "round_spec.json",
        "evidence_pool.jsonl",
    ],
    output_artifacts=[
        "candidate_questions.jsonl",
        "generation_report.json",
    ],
    run=run_qa_agent,
)
```

### 4.2 ToolSpec

工具也应统一描述，但要区分可见性：

```python
@dataclass
class ToolSpec:
    name: str
    owner: str
    visibility: Literal[
        "planner_visible",
        "workflow_required",
        "agent_private",
        "runtime_service",
    ]
    description: str
    input_schema: type
    output_schema: type
    side_effects: list[str]
    run: Callable[..., Any]
```

### 4.3 PlannerAction

Planner 不应该只是隐式生成 patch，而应该显式输出行动：

```python
@dataclass
class PlannerAction:
    round_id: int
    action_type: Literal[
        "run_round",
        "retry_round",
        "expand_corpus",
        "stop",
        "fail",
    ]
    round_spec: "RoundSpec | None"
    qa_patch: dict
    verify_patch: dict
    eval_patch: dict
    reason: str
    expected_outputs: list[str]
```

### 4.4 Observation

Worker agent 运行后必须返回结构化 observation，而不是只写 artifact。

```python
@dataclass
class AgentObservation:
    agent_name: str
    round_id: int
    status: Literal["success", "partial", "failed", "skipped"]
    metrics: dict
    artifact_refs: dict
    warnings: list[str]
    errors: list[str]
    recommendation: dict
```

---

## 5. 每个智能体的重构方案

## 5.1 PlannerAgent / SupervisorAgent

### 定位

PlannerAgent 是系统中最高自治度的 agent，负责全局规划、策略决策和跨轮反馈。

它应该被描述为：

> Supervisor agent that transforms user benchmark-building goals into iterative round plans and adapts future rounds based on worker observations.

### 主要职责

PlannerAgent 应负责：

- 接收用户目标与约束。
- 初始化 BenchmarkPlan。
- 维护 PlannerState。
- 生成 RoundSpec。
- 决定本轮 target、topic、difficulty、mode。
- 生成 QA / Verify / Eval config patch。
- 根据 observation 调整下一轮策略。
- 判断是否继续、重试、扩展 evidence 或终止。

### 不应负责

PlannerAgent 不应直接负责：

- 具体 chunk sampling。
- 具体题目生成 prompt 执行。
- citation validation 细节。
- duplicate clustering 细节。
- model metric 计算。
- artifact 底层读写。

这些应属于 worker agents 或 runtime services。

### Planner 可见工具

Planner 可以看到：

```text
workflow-required AgentTools:
  - run_qa_agent
  - run_verify_agent
  - run_model_eval_agent

planner-selectable tools:
  - inspect_round_feedback
  - inspect_artifact_summary
  - expand_corpus_for_topics
  - retry_round_with_patch
```

其中 `run_qa_agent`、`run_verify_agent`、`run_model_eval_agent` 可以被注册为工具，但不一定让 Planner 自由决定是否跳过。Runtime 可以规定正式流程必须执行：

```text
QA -> Verify -> ModelEval
```

Planner 主要决定这些工具的参数，而不是随意跳过质量门控。

---

## 5.2 QAAgent / Generation Worker Agent

### 定位

QAAgent 是 pipeline-based generation worker agent。

它内部可以是稳定 pipeline，但对外应作为一个具有明确目标、输入输出和 observation 的 worker agent。

### 对外接口

```python
@dataclass
class QARunInput:
    run_id: str
    round_id: int
    round_spec: "RoundSpec"
    evidence_scope: dict
    qa_config: dict
    artifact_refs: dict
    planner_context: dict

@dataclass
class GenerationObservation:
    agent_name: str
    round_id: int
    status: str

    requested_qa: int
    requested_mc: int
    generated_qa: int
    generated_mc: int
    accepted_candidates: int
    rejected_candidates: int

    topic_coverage: dict
    difficulty_distribution: dict
    mode_distribution: dict
    failure_reasons: dict

    artifact_refs: dict
    warnings: list[str]
    errors: list[str]
    recommendation: dict
```

对外只暴露：

```text
run_qa_agent(QARunInput) -> GenerationObservation
```

### 内部工具

QAAgent 内部可以有私有工具：

```text
QAAgent private tools:
  - sample_evidence
  - reserve_chunk_combinations
  - build_multi_chunk_units
  - build_generation_prompt
  - call_generation_model
  - parse_generation_output
  - lightweight_filter
  - record_chunk_usage
  - write_candidate_pool
  - build_generation_report
```

这些工具不建议暴露给 Planner。

### 局部自主性

QAAgent 可以做局部自主决策：

- 某个 topic evidence 不足时换采样策略。
- 某个 difficulty 生成失败时局部重试。
- LLM 输出格式错误时尝试 repair。
- 候选题不足时补采样。
- 避免重复 chunk combination。

但 QAAgent 不应决定：

- 是否终止整个 benchmark。
- 是否跳过 VerifyAgent。
- 是否修改全局目标题量。

---

## 5.3 VerifyAgent / Quality Gate Agent

### 定位

VerifyAgent 是 quality gate worker agent，负责题目质量门控。

它可以是 pipeline-based，但必须输出清晰的题目状态、质量统计和失败原因，以便 Planner 进行下一轮决策。

### 对外接口

```python
@dataclass
class VerifyRunInput:
    run_id: str
    round_id: int
    candidate_artifact: str
    verify_config: dict
    quota_spec: dict
    artifact_refs: dict

@dataclass
class ValidationObservation:
    agent_name: str
    round_id: int
    status: str

    input_candidates: int
    citation_passed: int
    llm_passed: int
    duplicate_dropped: int
    overquota_dropped: int
    selected_count: int
    reserve_count: int

    final_status_counts: dict
    quality_score_summary: dict
    failure_reasons: dict

    artifact_refs: dict
    warnings: list[str]
    errors: list[str]
    recommendation: dict
```

对外只暴露：

```text
run_verify_agent(VerifyRunInput) -> ValidationObservation
```

### 内部工具

```text
VerifyAgent private tools:
  - load_candidate_pool
  - validate_citations
  - llm_validate_questions
  - normalize_question
  - deduplicate_exact
  - cluster_semantic_duplicates
  - rank_by_quality
  - select_by_quota
  - assign_final_status
  - write_validated_questions
  - build_validation_report
```

### 题目 final_status 建议

每一道题应有明确状态：

```text
selected
reserve
duplicate
citation_failed
llm_failed
overquota
invalid_format
```

不要把 `overquota`、`duplicate`、`failed` 都混成 `reserve`，否则 Planner 无法判断下一轮应该调整什么。

### Quality Gate 原则

VerifyAgent 不建议被 Planner 随意跳过。

Planner 可以调整：

- citation 阈值
- LLM validation 是否启用
- LLM validation score 阈值
- dedup threshold
- quota 策略
- selected / reserve 比例

但正式 benchmark 构建流程中，VerifyAgent 应作为 mandatory quality gate。

---

## 5.4 ModelEvalAgent / Evaluator Worker Agent

### 定位

ModelEvalAgent 是 evaluator worker agent。它的自治程度最低，但仍可作为 agent，因为它有独立目标、配置、输入输出、artifact 和 observation。

### 对外接口

```python
@dataclass
class ModelEvalRunInput:
    run_id: str
    round_id: int
    selected_questions_artifact: str
    candidate_models: list[str]
    judge_model: str | None
    eval_config: dict
    artifact_refs: dict

@dataclass
class EvaluationObservation:
    agent_name: str
    round_id: int
    status: str

    evaluated_questions: int
    evaluated_models: int
    model_score_summary: dict
    difficulty_by_model: dict
    discrimination_summary: dict
    invalid_rate_summary: dict
    dataset_metric_summary: dict

    artifact_refs: dict
    warnings: list[str]
    errors: list[str]
    recommendation: dict
```

对外只暴露：

```text
run_model_eval_agent(ModelEvalRunInput) -> EvaluationObservation
```

### 内部工具

```text
ModelEvalAgent private tools:
  - load_selected_questions
  - run_candidate_model
  - run_judge_model
  - score_exact_match
  - score_f1
  - score_semantic_similarity
  - compute_invalid_rate
  - compute_discrimination
  - compute_dataset_metrics
  - write_eval_report
  - build_eval_feedback
```

### 局部策略

ModelEvalAgent 可以处理：

- candidate model 不可用时 fail fast。
- 某个模型调用失败时 retry。
- eval profile 为 light / standard / full。
- 大数据集时 sample eval。
- 只对 selected questions 评估。

但不应直接决定下一轮题目方向，这应由 Planner 根据 EvaluationObservation 处理。

---

## 6. Runtime 重构方案

当前 `planner_agent/orchestrator.py` 建议逐步重构为 Runtime，而不是一次性删除。

### 6.1 Runtime 职责

Runtime 应负责：

- 加载 BenchmarkRequest。
- 初始化 run directory。
- 加载 AgentRegistry / ToolRegistry。
- 调用 PlannerAgent 生成 PlannerAction。
- 按 workflow 执行 worker AgentTools。
- 校验每个 agent 的输入输出 schema。
- 维护 ArtifactStore。
- 维护 shared state。
- 记录 trace。
- 处理失败、重试、跳过、恢复。
- 将 observations 反馈给 PlannerAgent。
- 生成 final result。

### 6.2 Runtime 主循环

```python
class BenchForgeRuntime:
    async def run(self, request: BenchmarkRequest) -> BenchmarkRunResult:
        state = self.initialize(request)

        while True:
            action = await self.planner.decide(state)

            if action.action_type == "stop":
                return await self.finalize(state)

            if action.action_type == "fail":
                return await self.fail(state, action.reason)

            if action.action_type == "expand_corpus":
                observation = await self.tool_registry.call(
                    "expand_corpus_for_topics",
                    action,
                )
                state = self.apply_observation(state, observation)
                continue

            if action.action_type in {"run_round", "retry_round"}:
                round_obs = await self.execute_round(state, action)
                state = self.apply_round_observations(state, round_obs)
                continue
```

### 6.3 execute_round

```python
async def execute_round(self, state, action):
    generation_obs = await self.tool_registry.call(
        "run_qa_agent",
        build_qa_input(state, action),
    )

    validation_obs = await self.tool_registry.call(
        "run_verify_agent",
        build_verify_input(state, action, generation_obs),
    )

    evaluation_obs = await self.tool_registry.call(
        "run_model_eval_agent",
        build_eval_input(state, action, validation_obs),
    )

    return {
        "generation": generation_obs,
        "validation": validation_obs,
        "evaluation": evaluation_obs,
    }
```

注意：这里仍保持固定顺序，但每个阶段通过 registry 调用。

---

## 7. ToolRegistry 设计

### 7.1 工具可见性

建议定义四类：

```python
class ToolVisibility:
    PLANNER_VISIBLE = "planner_visible"
    WORKFLOW_REQUIRED = "workflow_required"
    AGENT_PRIVATE = "agent_private"
    RUNTIME_SERVICE = "runtime_service"
```

### 7.2 Planner 可见工具

```text
Planner-visible:
  - inspect_round_feedback
  - inspect_artifact_summary
  - expand_corpus_for_topics
  - retry_round_with_patch
```

### 7.3 Workflow-required AgentTools

```text
Workflow-required:
  - run_qa_agent
  - run_verify_agent
  - run_model_eval_agent
```

这些工具注册在 ToolRegistry 中，但正式 workflow 中由 Runtime 强制按顺序执行，Planner 不应自由跳过。

### 7.4 Agent-private tools

```text
QAAgent private:
  - sample_evidence
  - parse_generation_output
  - lightweight_filter

VerifyAgent private:
  - validate_citations
  - llm_validate_questions
  - deduplicate
  - cluster_select

ModelEvalAgent private:
  - run_candidate_model
  - compute_metrics
```

### 7.5 Runtime services

```text
Runtime services:
  - load_artifact
  - save_artifact
  - load_shared_state
  - save_shared_state
  - trace_llm_call
  - resolve_run_path
```

---

## 8. utils 目录迁移建议

当前 `utils/` 不建议一次性全部改名。可以渐进迁移。

### 8.1 Runtime service 类

建议移动到 `runtime/`：

```text
utils/artifact_store.py -> runtime/artifact_store.py
utils/shared_state.py   -> runtime/state_store.py
utils/llm_tracer.py     -> runtime/tracing.py
utils/run_context.py    -> runtime/run_context.py
utils/paths.py          -> runtime/paths.py
```

### 8.2 QAAgent 私有工具类

建议移动到 `agents/qa_agent/tools.py` 或 `agents/qa_agent/private_tools/`：

```text
utils/sampling.py
utils/multi_chunk.py
utils/signals.py
utils/filter.py
utils/parsing.py
```

其中 `filter.py` 如果 VerifyAgent 也使用，可拆成：

```text
agents/qa_agent/tools/filtering.py
agents/verify_agent/tools/format_validation.py
```

### 8.3 ModelEvalAgent 私有工具类

建议移动到：

```text
agents/model_eval_agent/tools/metrics_auto.py
agents/model_eval_agent/tools/metrics_dataset.py
```

### 8.4 Evidence / Corpus 能力

如果后续证据管理越来越复杂，可以新增：

```text
agents/evidence_agent/
```

对应迁移：

```text
utils/retrieval.py
utils/chunking.py
utils/signals.py
```

短期也可以保留为 shared evidence tools，不强制新增 EvidenceAgent。

### 8.5 planning.py

`utils/planning.py` 建议拆分：

```text
Planner policy helpers:
  -> agents/planner_agent/policy_helpers.py

QA prompt helpers:
  -> agents/qa_agent/prompt_helpers.py
```

不建议整体注册成 tool。

---

## 9. LangGraph 接入建议

### 9.1 当前是否应该直接调用 LangGraph？

短期不建议强依赖 LangGraph。

当前更重要的是先完成：

```text
AgentSpec
ToolSpec
PlannerAction
Observation
Runtime
ArtifactStore
```

这些边界稳定后，再接 LangGraph 会更自然。

### 9.2 什么时候值得接入 LangGraph？

如果出现以下需求，建议引入 LangGraph backend：

- 长任务 checkpoint / resume。
- human-in-the-loop 审核。
- 运行中断后恢复。
- 图状态可视化。
- 复杂条件分支。
- streaming progress。
- 多轮 agent graph 观测和调试。
- 多 backend runtime 管理。

### 9.3 LangGraph-compatible State

即使短期不接 LangGraph，也建议把 state 设计成 graph-friendly：

```python
@dataclass
class BenchForgeGraphState:
    request: BenchmarkRequest
    planner_state: PlannerState
    current_round: int

    last_planner_action: PlannerAction | None
    round_spec: RoundSpec | None

    generation_observation: GenerationObservation | None
    validation_observation: ValidationObservation | None
    evaluation_observation: EvaluationObservation | None

    artifact_refs: dict
    errors: list[dict]
    final_result: BenchmarkRunResult | None
```

### 9.4 未来 LangGraph 节点设计

```text
initialize_node
planner_node
qa_node
verify_node
eval_node
feedback_node
finalize_node
error_node
```

边：

```text
START -> initialize
initialize -> planner
planner -> qa | finalize | error
qa -> verify
verify -> eval
eval -> feedback
feedback -> planner | finalize | error
finalize -> END
error -> planner | END
```

高层节点调用 worker agent，不要把每个小函数都变成 LangGraph 节点。

---

## 10. 对外服务设计

BenchForge 应至少提供三种接口。

### 10.1 Python SDK

```python
from benchforge import BenchForgeRuntime, BenchmarkRequest

request = BenchmarkRequest(
    name="finance_qa_benchmark",
    target_questions=500,
    modes=["qa", "mcq"],
    difficulty_mix={
        "easy": 0.2,
        "medium": 0.5,
        "hard": 0.3,
    },
    candidate_models=["model_a", "model_b"],
)

runtime = BenchForgeRuntime.from_config("configs/benchforge.yaml")
result = await runtime.run(request)
```

### 10.2 CLI

```bash
benchforge run \
  --config configs/benchforge.yaml \
  --request requests/my_benchmark.yaml \
  --out runs/exp_001
```

辅助命令：

```bash
benchforge resume --run runs/exp_001
benchforge inspect --run runs/exp_001 --round 3
benchforge validate --run runs/exp_001
benchforge export --run runs/exp_001 --format jsonl
```

### 10.3 HTTP API

如果要平台化，可提供：

```http
POST /runs
GET  /runs/{run_id}
GET  /runs/{run_id}/events
GET  /runs/{run_id}/artifacts
POST /runs/{run_id}/resume
POST /runs/{run_id}/cancel
```

示例请求：

```json
{
  "name": "finance_qa_benchmark",
  "target_questions": 500,
  "modes": ["qa", "mcq"],
  "difficulty_mix": {
    "easy": 0.2,
    "medium": 0.5,
    "hard": 0.3
  },
  "candidate_models": ["model_a", "model_b"],
  "constraints": {
    "require_citations": true,
    "max_rounds": 8
  }
}
```

---

## 11. 推荐目录结构

建议目标目录：

```text
benchforge/
  agents/
    base.py
    registry.py

    planner_agent/
      agent.py
      policy.py
      schema.py
      prompts.py
      tools.py

    qa_agent/
      agent.py
      schema.py
      pipeline.py
      tools.py
      prompts.py

    verify_agent/
      agent.py
      schema.py
      pipeline.py
      tools.py
      selector.py

    model_eval_agent/
      agent.py
      schema.py
      pipeline.py
      tools.py

  runtime/
    runtime.py
    workflow.py
    tool_registry.py
    artifact_store.py
    state_store.py
    observation.py
    tracing.py
    errors.py
    run_context.py

  tools/
    evidence.py
    retrieval.py
    chunking.py

  services/
    llm_client.py
    model_registry.py
    storage.py

  api/
    server.py
    schemas.py

  cli/
    main.py

  docs/
    multi_agent_refactor_plan.md
```

---

## 12. 落地路线

### Phase 1: 定义统一 schema

新增：

```text
runtime/observation.py
agents/base.py
agents/registry.py
runtime/tool_registry.py
```

定义：

```text
AgentSpec
ToolSpec
PlannerAction
AgentObservation
GenerationObservation
ValidationObservation
EvaluationObservation
BenchmarkRunResult
```

目标：不改业务逻辑，先统一类型边界。

---

### Phase 2: 注册现有 agents

把现有：

```text
QAAgent
VerifyAgent
ModelEvalAgent
PlannerAgent
```

注册到 AgentRegistry。

目标：保留现有调用逻辑，但让系统具备可发现的 agent 元数据。

---

### Phase 3: 包装 AgentTools

把现有 orchestrator 里的调用包装成：

```text
run_qa_agent
run_verify_agent
run_model_eval_agent
```

目标：Runtime 不再直接 import 每个 agent 的内部函数，而是通过 ToolRegistry 调用。

---

### Phase 4: Orchestrator 升级为 Runtime

把 `planner_agent/orchestrator.py` 中的职责拆到：

```text
runtime/runtime.py
runtime/workflow.py
runtime/artifact_store.py
runtime/state_store.py
runtime/tracing.py
```

目标：PlannerAgent 只负责决策，Runtime 负责执行和状态管理。

---

### Phase 5: 整理 utils

按归属迁移：

```text
runtime services
agent-private tools
shared evidence tools
```

目标：减少 `utils/` 作为大杂烩目录带来的边界混乱。

---

### Phase 6: 可选 LangGraph backend

新增：

```text
runtime/langgraph_backend.py
```

但保持默认 backend 为 simple runtime：

```python
runtime = BenchForgeRuntime(backend="simple")
```

未来可切换：

```python
runtime = BenchForgeRuntime(backend="langgraph")
```

---

## 13. 最终推荐结论

BenchForge 不应改成“只有 PlannerAgent 一个智能体，其他都是普通工具”的结构。

更合理的最终形态是：

```text
PlannerAgent:
  supervisor agent，高自治度，负责全局计划和反馈闭环。

QAAgent:
  pipeline-based generation worker agent，负责题目生成。

VerifyAgent:
  pipeline-based quality gate worker agent，负责验证、去重、筛选。

ModelEvalAgent:
  evaluator worker agent，负责模型评估和区分度反馈。

Runtime:
  负责执行固定 workflow、调用 AgentTools、管理 artifact、trace、schema 校验和失败处理。

ToolRegistry:
  注册 planner-visible tools、workflow-required AgentTools、agent-private tools 和 runtime services。

LangGraph:
  暂不强依赖，但状态和 Runtime 设计应保持 LangGraph-compatible。
```

一句话总结：

> 注册工具不是为了让 Planner 随意跳过流水线，而是为了让 BenchForge 从硬编码 pipeline 升级为可注册、可校验、可追踪、可扩展的 planner-driven hierarchical multi-agent workflow。
