# BenchForge 独立题目生成 Agent 技术方案 V2

## 0. 本版修订说明

本方案相对上一版做了两个关键调整：

1. 避免使用 `planner` 这一命名，防止和当前已有实现产生歧义。
2. 新方案目标是新增一个独立 Agent，不直接迁移或破坏现有 `qa_agent` / `utils` 实现。

因此本方案采用：

```text
新建独立 agent
复制 utils 中必要能力
在新 agent 内部封装适配层
保留旧实现可运行
```

---

## 1. 新 Agent 定位

新 Agent 建议命名为：

```text
AdaptiveQuestionGenerationAgent
```

或目录名：

```text
agents/adaptive_qa_agent/
```

它不是替换当前 `agents/qa_agent/`，而是作为新的实验性实现存在。

职责：

```text
根据目标配额、验证反馈、topic 覆盖、难度分布、chunk 使用状态，动态选择下一轮题目生成动作。
```

核心结构：

```text
Feedback-driven
Rule-based Decision
Tool Executor
Validation Feedback Loop
```

---

## 2. 为什么不再使用 Planner 命名

`planner` 容易和当前已有的：

```text
agents/qa_agent/planner.py
utils/planning.py
```

产生语义冲突。

因此新 Agent 中建议改名为：

```text
decision_engine.py
```

或：

```text
round_policy.py
```

推荐使用：

```text
decision_engine.py
```

它表达的是：

```text
根据当前状态选择下一轮动作
```

而不是传统意义上的全局规划器。

---

## 3. 推荐新目录结构

```text
agents/
  adaptive_qa_agent/
    __init__.py

    agent.py
    state.py
    actions.py
    decision_engine.py
    executor.py
    feedback.py
    stop_policy.py

    prompts/
      base_prompt.py
      strategy_constraints.py
      prompt_builder.py

    tools/
      evidence_group_builder.py
      retrieval_expansion_tool.py
      generation_tool.py
      grounded_question_evolution_tool.py

    local_utils/
      chunking.py
      retrieval.py
      multi_chunk.py
      filter.py
      signals.py
```

说明：

```text
local_utils/ 是从 utils 复制出来的适配版本
不会影响原 utils
不会影响当前 qa_agent
```

---

## 4. 与当前实现的兼容策略

当前实现保持不动：

```text
agents/qa_agent/
utils/
```

新增实现独立运行：

```text
agents/adaptive_qa_agent/
```

推荐入口：

```python
from agents.adaptive_qa_agent.agent import run_adaptive_generation_agent
```

不要修改旧入口：

```python
from agents.qa_agent.agent import run_generation_agent
```

这样可以做到：

```text
旧系统稳定运行
新系统独立实验
后续成熟后再切换入口
```

---

## 5. 从 utils 复制，而不是迁移

本方案不迁移 `utils` 文件。

采用复制策略：

```text
utils/chunking.py
→ agents/adaptive_qa_agent/local_utils/chunking.py

utils/retrieval.py
→ agents/adaptive_qa_agent/local_utils/retrieval.py

utils/multi_chunk.py
→ agents/adaptive_qa_agent/local_utils/multi_chunk.py

utils/filter.py
→ agents/adaptive_qa_agent/local_utils/filter.py

utils/signals.py
→ agents/adaptive_qa_agent/local_utils/signals.py
```

保留原文件：

```text
utils/chunking.py
utils/retrieval.py
utils/multi_chunk.py
utils/filter.py
utils/signals.py
```

原因：

```text
1. 不破坏当前 qa_agent
2. 新 Agent 可以自由改造 * 文件
3. 便于 A/B 对比
4. 方便后续逐步收敛公共抽象
```

---

## 6. 新 Agent 主循环

```text
run_adaptive_generation_agent()
  ↓
for each mode:
  ↓
run_mode_loop()
  ↓
stop_policy.should_stop()
  ↓
decision_engine.decide_next_action()
  ↓
executor.execute(action)
  ↓
validation_agent.validate()
  ↓
feedback.aggregate()
  ↓
state.update()
```

对应代码结构：

```python
while True:
    stop, reason = stop_policy.should_stop(mode_state, config)
    if stop:
        break

    action_plan = decision_engine.decide_next_action(
        state=mode_state,
        config=config,
        feedback=feedback_state,
    )

    result = await executor.execute(action_plan, mode_state)

    validation_report = await validate_candidates(result.candidates)

    feedback_state = aggregate_validation_feedback(validation_report)

    mode_state.update(result, validation_report, feedback_state)
```

---

## 7. StopPolicy 设计

终止条件继续兼容当前 `max_round` 思路。

文件：

```text
agents/adaptive_qa_agent/stop_policy.py
```

保留条件：

```text
max_rounds_reached
candidate_pool_sufficient
consecutive_empty_rounds_reached
failure_limit_reached
```

新增可选条件：

```text
target_distribution_satisfied
no_improvement_rounds_reached
hard_generation_budget_exhausted
```

示例：

```python
class StopPolicy:
    def should_stop(self, state, config):
        if state.round_in_mode >= config.max_rounds:
            return True, "max_rounds_reached"

        if state.accepted_count >= config.target_candidates:
            return True, "candidate_pool_sufficient"

        if state.consecutive_empty_rounds >= config.max_empty_rounds:
            return True, "consecutive_empty_rounds_reached"

        if state.failures_count >= config.max_failures:
            return True, "failure_limit_reached"

        return False, None
```

重点：

```text
不要让 LLM 决定是否停止
不要让 tool 内部绕过 stop policy
所有终止条件统一收敛到 StopPolicy
```

---

## 8. DecisionEngine 设计

文件：

```text
agents/adaptive_qa_agent/decision_engine.py
```

职责：

```text
根据结构化状态选择下一轮 Action
```

不叫 Planner。

输入：

```python
state
config
feedback_state
```

输出：

```python
ActionPlan
```

示例：

```python
@dataclass
class ActionPlan:
    action: str
    topics: list[str]
    difficulty: str
    target_count: int
    evidence_strategy: str | None = None
    prompt_strategy: str | None = None
    reason: str | None = None
```

核心规则：

```python
if feedback.answer_not_grounded_count > threshold:
    return ActionPlan(
        action="retrieve_more_documents",
        evidence_strategy="new_document_retrieval",
        reason="grounding failure is high",
    )

if feedback.evidence_insufficient_count > threshold:
    return ActionPlan(
        action="expand_evidence",
        evidence_strategy="same_topic_or_neighbor",
        reason="evidence is insufficient",
    )

if state.hard_gap > threshold and feedback.not_multihop_count > threshold:
    return ActionPlan(
        action="multihop_generation",
        difficulty="hard",
        evidence_strategy="multi_evidence_group",
        prompt_strategy="multihop_hard",
    )

if state.hard_gap > threshold:
    return ActionPlan(
        action="hard_generation",
        difficulty="hard",
        evidence_strategy="high_hard_score_group",
        prompt_strategy="hard_constraints",
    )

if state.topic_gap:
    return ActionPlan(
        action="topic_focused_generation",
        topics=state.missing_topics,
        prompt_strategy="topic_focus",
    )

if feedback.too_easy_count > threshold and state.grounded_medium_pool:
    return ActionPlan(
        action="grounded_question_evolution",
        difficulty="hard",
        prompt_strategy="evolution_hard",
    )

return ActionPlan(action="generate")
```

---

## 9. Action Space

文件：

```text
agents/adaptive_qa_agent/actions.py
```

建议定义：

```python
GENERATE = "generate"

HARD_GENERATION = "hard_generation"

MULTIHOP_GENERATION = "multihop_generation"

TOPIC_FOCUSED_GENERATION = "topic_focused_generation"

EXPAND_EVIDENCE = "expand_evidence"

RETRIEVE_MORE_DOCUMENTS = "retrieve_more_documents"

GROUNDED_QUESTION_EVOLUTION = "grounded_question_evolution"

STOP = "stop"
```

说明：

```text
Action 是业务级动作
不是底层函数名
不是 utils 函数名
```

---

## 10. Tool Executor 设计

文件：

```text
agents/adaptive_qa_agent/executor.py
```

职责：

```text
根据 ActionPlan 分发到具体 tool
```

示例：

```python
class AdaptiveActionExecutor:
    async def execute(self, action_plan, state):
        if action_plan.action == "generate":
            return await self.generation_tool.run(action_plan, state)

        if action_plan.action == "hard_generation":
            evidence_group = self.evidence_group_builder.build(action_plan, state)
            return await self.generation_tool.run(action_plan, state, evidence_group)

        if action_plan.action == "multihop_generation":
            evidence_group = self.evidence_group_builder.build(action_plan, state)
            return await self.generation_tool.run(action_plan, state, evidence_group)

        if action_plan.action == "expand_evidence":
            return await self.retrieval_expansion_tool.run(action_plan, state)

        if action_plan.action == "retrieve_more_documents":
            return await self.retrieval_expansion_tool.run(action_plan, state)

        if action_plan.action == "grounded_question_evolution":
            return await self.grounded_question_evolution_tool.run(action_plan, state)
```

---

## 11. EvidenceGroupBuilder

文件：

```text
agents/adaptive_qa_agent/tools/evidence_group_builder.py
```

来源：

```text
复制并改造 utils/multi_chunk.py
必要时调用 chunking.py / signals.py
```

职责：

```text
构建适合 hard / multi-hop 生成的 evidence group
```

输入：

```python
topic
difficulty
seed_chunks
strategy
```

策略：

```text
same_document_neighbor
same_entity
same_topic
high_hard_score
new_document_retrieval
multi_chunk_unit
```

输出：

```python
{
    "chunk_ids": [...],
    "evidence_units": [...],
    "strategy": "...",
    "expected_reasoning_type": "..."
}
```

当前阶段不强制做显式 ReasoningChainBuilder。

先采用：

```text
Evidence Group + Prompt 约束 + 验证智能体兜底
```

后续如果 hard / multi-hop 失败率高，再引入：

```text
ReasoningChainBuilder
```

---

## 12. Prompt 设计

不单独维护大量 hard prompt。

采用：

```text
Base Prompt
+ Strategy Constraints
```

目录：

```text
prompts/
  base_prompt.py
  strategy_constraints.py
  prompt_builder.py
```

### 12.1 Base Prompt

负责统一格式、字段、基础 grounding 要求。

### 12.2 Strategy Constraints

按策略注入：

```python
HARD_CONSTRAINTS

MULTIHOP_CONSTRAINTS

TOPIC_FOCUS_CONSTRAINTS

EVOLUTION_CONSTRAINTS
```

### 12.3 Hard Constraints

```text
生成 hard 题时必须满足：

1. 至少依赖两个 evidence。
2. 答案不能从单个 chunk 直接复制。
3. 必须包含比较、条件筛选、归纳、因果、反向定位之一。
4. 必须输出 difficulty_reason。
5. 必须输出 required_evidence_ids。
6. 必须输出 reasoning_path。
7. 答案必须能被给定 evidence 完全支持。
```

输出格式：

```json
{
  "question": "...",
  "answer": "...",
  "difficulty": "hard",
  "difficulty_reason": "...",
  "reasoning_type": "...",
  "required_evidence_ids": ["chunk_1", "chunk_3"],
  "reasoning_path": [
    "..."
  ]
}
```

---

## 13. 难度不足处理方案

难度不足先诊断原因，不直接改写问题。

### 13.1 hard 不足 + grounding 失败多

表现：

```text
hard_gap 大
answer_not_grounded_count 高
```

动作：

```text
RETRIEVE_MORE_DOCUMENTS
```

流程：

```text
检索新文档
→ 切 chunk
→ EvidenceGroupBuilder
→ hard_constraints 生成
→ 验证智能体复核
```

### 13.2 hard 不足 + evidence 不足

动作：

```text
EXPAND_EVIDENCE
```

流程：

```text
邻接 chunk
同 topic chunk
同实体 chunk
新文档 chunk
→ evidence group
→ hard generation
```

### 13.3 hard 不足 + not_multihop 多

动作：

```text
MULTIHOP_GENERATION
```

流程：

```text
EvidenceGroupBuilder
→ 注入 multihop constraints
→ 生成 required_evidence_ids / reasoning_path
→ 验证 multi-hop
```

### 13.4 grounded 但 too_easy 多

动作：

```text
GROUNDED_QUESTION_EVOLUTION
```

触发条件：

```python
grounded=True
answerable=True
evidence_sufficient=True
difficulty in ["easy", "medium"]
reject_reason == "too_easy"
```

禁止触发：

```text
answer_not_grounded
evidence_insufficient
answer_ambiguous
duplicate
format_error
```

---

## 14. GroundedQuestionEvolutionTool

文件：

```text
tools/grounded_question_evolution_tool.py
```

定位：

```text
兜底工具
不是 hard 题生成主路径
```

输入：

```python
{
    "original_question": "...",
    "original_answer": "...",
    "evidence_ids": [...],
    "target_difficulty": "hard"
}
```

输出：

```python
{
    "evolved_question": "...",
    "answer": "...",
    "difficulty": "hard",
    "difficulty_reason": "...",
    "required_evidence_ids": [...],
    "reasoning_path": [...]
}
```

---

## 15. FeedbackAggregator

文件：

```text
agents/adaptive_qa_agent/feedback.py
```

职责：

```text
把验证智能体输出转成 DecisionEngine 可消费的结构化信号
```

输入：

```python
validation_results
```

输出：

```python
@dataclass
class FeedbackState:
    accepted_count: int
    rejected_count: int

    too_easy_count: int
    not_multihop_count: int
    answer_not_grounded_count: int
    evidence_insufficient_count: int
    duplicate_count: int
    format_error_count: int

    difficulty_stats: dict
    topic_stats: dict
```

---

## 16. 与验证智能体关系

新 Agent 只做：

```text
生成
轻量格式检查
状态更新
下一轮动作选择
```

验证智能体负责：

```text
grounded
answerable
difficulty
multi-hop
duplicate
reference support
ambiguity
```

生成 Agent 消费验证报告，不重复实现深度质量判断。

---

## 17. 和 max_round 的统一

新 Agent 仍保留 max_round。

max_round 属于：

```text
StopPolicy
```

不是 DecisionEngine，也不是 Tool。

执行顺序：

```text
StopPolicy.should_stop()
↓
DecisionEngine.decide_next_action()
↓
Executor.execute()
```

这样能兼容现有终止条件：

```text
max_rounds
candidate_pool_sufficient
consecutive_empty_rounds
failure_limit
```

---

## 18. 与旧实现的切换方式

建议提供配置项：

```yaml
qa_generation_agent_type: legacy
```

或：

```yaml
qa_generation_agent_type: adaptive
```

入口：

```python
if config.qa_generation_agent_type == "legacy":
    return await run_generation_agent(...)

if config.qa_generation_agent_type == "adaptive":
    return await run_adaptive_generation_agent(...)
```

---

## 19. 推荐实施阶段

### Phase 1: 独立目录搭建

```text
创建 agents/adaptive_qa_agent/
复制 local_utils
实现 state/actions/stop_policy/decision_engine
```

### Phase 2: 复用现有生成逻辑

```text
generation_tool 先包装当前 generator
prompt_builder 注入 hard/multihop constraints
```

### Phase 3: EvidenceGroupBuilder

```text
复制 multi_chunk.py
改造成 evidence_group_builder.py
支持 hard/multihop evidence group
```

### Phase 4: Validation Feedback Loop

```text
接入验证智能体报告
实现 FeedbackAggregator
让 DecisionEngine 根据反馈选择动作
```

### Phase 5: Grounded Evolution

```text
实现 grounded_question_evolution_tool
只对 grounded easy/medium 题做兜底升级
```

---

## 20. 最终结论

新架构应当是：

```text
独立 AdaptiveQuestionGenerationAgent
+ DecisionEngine
+ StopPolicy
+ ToolExecutor
+ EvidenceGroupBuilder
+ Prompt Strategy Injection
+ Validation Feedback Loop
```

不应当：

```text
直接改旧 qa_agent
直接迁移 utils
继续使用 planner 命名
让 LLM 每轮自主决策
让 DifficultyEvolution 成为主路径
```

核心思想：

```text
用结构化状态驱动动作选择。
用 evidence group 和 prompt constraints 提升难度。
用验证智能体保证 grounding 和质量。
用独立目录保证新旧系统互不干扰。
```
