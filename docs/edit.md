# adaptive_qa_agent 重构方案

目标：在不改变当前核心行为的前提下，提升 `adaptive_qa_agent` 的可读性、可测试性、可扩展性和实验可解释性。

这个方案是收敛后的增量重构，不做大规模架构翻新，不引入多策略系统，不把当前实现过度抽象化。

## 重构原则

1. 保持行为基本不变

当前 `adaptive_qa_agent` 已经具备可运行的 `decision / state / executor / agent loop` 结构。本次重构的目标不是重写逻辑，而是把现有职责边界整理清楚。

2. 只做结构化封装，不做未来化设计

本次只解决当前真实存在的问题：

- `decision.py` 规则增长后维护成本会上升
- `state.py` 缺少统一的 metrics 导出
- `agent.py` 的 stop 条件散落在主循环中

不为“以后可能需要”提前引入复杂抽象。

3. 优先低风险、可增量落地

每一步都应该可以独立提交、独立测试、独立回滚。

## 实施顺序

按下面顺序推进：

1. `decision.py` 改成 `DecisionEngine`
2. `ModeState / FeedbackState` 增加轻量 metrics 导出
3. `agent.py` 的 stop 条件抽成 `StopPolicy`

这三个改动都属于低风险、小范围、收益明确的整理。

## 方案 1：DecisionEngine

### 目标

把当前 `decide()` 中的顺序规则改成结构化封装，让规则边界更清晰、更容易测试和扩展。

### 当前问题

现在的 `decide()` 本质上是一串带优先级的 `if-elif` 规则：

- grounding failure
- evidence insufficient
- hard + multihop gap
- hard gap
- topic gap
- evolve easy
- default

这个逻辑本身没有问题，但如果后续继续加规则，函数会越来越难读，也不利于单规则测试。

### 改法

将 `decision.py` 从函数式判断改成一个轻量 `DecisionEngine`：

```python
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Action:
    action_type: str
    topics: list[str] = field(default_factory=list)
    difficulty: str | None = None
    evidence_strategy: str | None = None
    reason: str = ""


class DecisionEngine:
    def __init__(self, adaptive_config):
        self.cfg = adaptive_config.decision
        self.rules: list[Callable] = [
            self._rule_grounding_failure,
            self._rule_evidence_insufficient,
            self._rule_hard_multihop_gap,
            self._rule_hard_gap,
            self._rule_topic_gap,
            self._rule_evolve_easy,
        ]

    def decide(self, state, feedback) -> Action:
        for rule in self.rules:
            action = rule(state, feedback)
            if action is not None:
                return action
        return Action("generate", reason="default")
```

每条规则拆成独立私有方法，例如：

- `_rule_grounding_failure`
- `_rule_evidence_insufficient`
- `_rule_hard_multihop_gap`
- `_rule_hard_gap`
- `_rule_topic_gap`
- `_rule_evolve_easy`

### 边界说明

这里的 `DecisionEngine` 只是“当前规则的结构化封装”，不是策略系统。

这次不要做：

- `RulePolicy`
- `LLMPolicy`
- `HybridPolicy`
- `PolicyRegistry`
- 多策略切换框架

原因很简单：当前问题只是规则组织方式，不是策略体系问题。现在做多策略抽象属于过度设计。

### agent.py 配套修改

将：

```python
from .decision import decide

action = decide(state, feedback, adaptive_config)
```

改成：

```python
from .decision import DecisionEngine

# 每个 mode 初始化一次，不要放进 while 循环里反复构建
decision_engine = DecisionEngine(adaptive_config)

while True:
    action = decision_engine.decide(state, feedback)
```

这里要明确：

- `DecisionEngine` 应该在每个 mode 的循环外初始化一次
- 不要在每轮 `while` 内重新 `DecisionEngine(adaptive_config)`
- `self.rules` 是固定规则列表，没有必要每轮重复构建

### Action 兼容性要求

`Action` 不是这次新引入的陌生接口，而是对当前 `decision.py` 中动作结构的延续和固定。

执行时必须遵守：

- 保持 `executor.py` 当前消费契约不变
- `action.action_type / topics / difficulty / evidence_strategy / reason` 这些字段继续可用
- 不要把动作返回值改成 `tuple / dict / str`

如果当前代码里已经存在 `Action` dataclass，则应直接保留并复用；如果不存在，也必须新增一个与现有 `executor` 消费方式兼容的结构，而不是顺手改动 executor 输入协议。

也就是说，这一步的目标是：

- 重构 `decide` 的组织方式
- 不改变动作对象的外部接口

而不是：

- 顺便修改 executor 的输入协议
- 顺便改 Action 的类型体系
- 让 executor 跟着做破坏性适配

应当保持类似下面的消费方式继续成立：

```python
# executor.py
if action.action_type == "retrieve_more":
    ...

topics = action.topics or state.active_topics
```

### 收益

- 行为基本不变
- 决策规则边界清晰
- 单条规则可以单独测试
- 未来新增规则时，修改点更集中
- 不引入对 `executor.py` 的额外破坏性变更

## 方案 2：轻量 Metrics 聚合

### 目标

在不重构现有状态模型的前提下，为实验分析、日志观测、论文数据整理提供统一指标导出。

### 当前问题

当前 `ModeState` 已经记录：

- accepted / rejected 候选
- difficulty 分布
- topic 分布
- failures
- consecutive_empty
- used chunk combinations
- recent action 信息

当前 `FeedbackState` 已经记录：

- feedback window
- total generated
- total accepted
- 各类 reject ratio

这些信息已经足够有价值，但缺少统一导出接口，导致日志分析和实验复盘不方便。

### 改法

不要新建“大一统状态类”，只在现有类上补导出方法。

#### FeedbackState

新增：

```python
class FeedbackState:
    def export_metrics(self) -> dict:
        return {
            "lifetime_total_generated": self.total_generated,
            "lifetime_total_accepted": self.total_accepted,
            "lifetime_accept_rate": self.total_accepted / max(1, self.total_generated),
            "recent_answer_not_grounded_ratio": self.ratio("answer_not_grounded"),
            "recent_evidence_insufficient_ratio": self.ratio("evidence_insufficient"),
            "recent_not_multihop_ratio": self.ratio("not_multihop"),
            "recent_too_easy_ratio": self.ratio("too_easy"),
            "recent_evolution_failed_ratio": self.ratio("evolution_failed"),
        }
```

#### ModeState

新增：

```python
@dataclass
class ModeState:
    def export_metrics(self) -> dict:
        accepted = self.accepted_count
        return {
            "mode_name": self.mode,
            "mode_round_in_progress": self.round_in_mode,
            "mode_target_candidates": self.target_candidates,
            "mode_accepted_count": accepted,
            "mode_accept_progress": accepted / max(1, self.target_candidates),
            "mode_hard_gap": self.hard_gap,
            "mode_missing_topic_count": len(self.missing_topics),
            "mode_missing_topic_ratio": len(self.missing_topics) / max(1, len(self.all_topics)),
            "mode_consecutive_empty": self.consecutive_empty,
            "mode_failures": self.failures,
            "mode_last_action_type": self.last_action_type,
            "mode_consecutive_same_action": self.consecutive_same_action,
            "mode_difficulty_counts": dict(self.difficulty_counts),
            "mode_topic_counts": dict(self.topic_counts),
        }
```

#### 统一导出函数

可以额外加一个轻量函数：

```python
def export_adaptive_metrics(state: ModeState, feedback: FeedbackState) -> dict:
    mode_metrics = state.export_metrics()
    feedback_metrics = feedback.export_metrics()

    conflict_keys = mode_metrics.keys() & feedback_metrics.keys()
    assert not conflict_keys, f"metrics key conflict: {sorted(conflict_keys)}"

    return {
        **mode_metrics,
        **feedback_metrics,
    }
```

### 命名约束

metrics 必须明确分层，避免语义混淆：

- `lifetime_*`：全局累计
- `recent_*`：基于 feedback window 的近期统计
- `mode_*`：当前 mode 内统计

例如：

```python
{
    "lifetime_total_generated": ...,
    "lifetime_total_accepted": ...,
    "lifetime_accept_rate": ...,

    "recent_answer_not_grounded_ratio": ...,
    "recent_evidence_insufficient_ratio": ...,
    "recent_not_multihop_ratio": ...,

    "mode_accepted_count": ...,
    "mode_accept_progress": ...,
    "mode_hard_gap": ...,
    "mode_missing_topic_ratio": ...,
}
```

这里的约束很重要：

- 不要把 `recent_*` 和 `lifetime_*` 混成同一语义层
- 不要让 mode 指标和全局累计字段重名
- 字段命名尽量一次定型，避免后续日志 schema 漂移
- `export_adaptive_metrics(...)` 合并前应显式检查 key 冲突，不能静默覆盖

### 收益

- 实验结果更可解释
- 日志更容易观测和对比
- 后续论文、中期汇报、ablation 分析更方便

## 方案 3：StopPolicy

### 目标

把停止条件从主循环中抽离，使 `agent.py` 更像编排逻辑，stop reason 更清楚。

### 当前问题

当前主循环开头有多个停止条件：

- `state.round_in_mode >= state.max_rounds`
- `state.accepted_count >= state.target_candidates`
- `state.consecutive_empty >= adaptive_config.stop.max_empty_rounds`
- `state.failures >= adaptive_config.stop.max_failures`

逻辑本身没有错，但它们和“主循环编排”属于不同职责，散在一起会让 agent 越来越重。

### 改法

新增 `stop_policy.py`：

```python
from dataclasses import dataclass


@dataclass
class StopDecision:
    should_stop: bool
    reason: str = ""


class StopPolicy:
    def __init__(self, adaptive_config):
        self.cfg = adaptive_config.stop

    def check(self, state) -> StopDecision:
        if state.round_in_mode >= state.max_rounds:
            return StopDecision(True, "max_rounds")

        if state.accepted_count >= state.target_candidates:
            return StopDecision(True, "target_reached")

        if state.consecutive_empty >= self.cfg.max_empty_rounds:
            return StopDecision(True, "max_empty_rounds")

        if state.failures >= self.cfg.max_failures:
            return StopDecision(True, "max_failures")

        return StopDecision(False)
```

这里需要明确边界：

- `max_empty_rounds` 和 `max_failures` 来自 `adaptive_config.stop`
- `max_rounds` 和 `target_candidates` 是当前 mode 的运行上下文，不应硬塞进 `adaptive_config.stop`

原因是：

- `target_candidates` 是由当前 mode 的目标数量动态计算出来的
- `max_rounds` 也是 mode 级限制，而不是全局 stop 配置

因此 `StopPolicy` 使用 `state.max_rounds` 和 `state.target_candidates` 是可以接受的，前提是文档明确它们代表“mode execution context”，而不是随意混放的全局配置。

如果后续还想进一步解耦，可以做成下面这种形式，但这不是这次重构的必选项：

```python
@dataclass
class StopContext:
    max_rounds: int
    target_candidates: int


class StopPolicy:
    def __init__(self, adaptive_config, context: StopContext):
        self.cfg = adaptive_config.stop
        self.context = context

    def check(self, state) -> StopDecision:
        if state.round_in_mode >= self.context.max_rounds:
            return StopDecision(True, "max_rounds")
        if state.accepted_count >= self.context.target_candidates:
            return StopDecision(True, "target_reached")
        ...
```

本次建议保持简单：

- 先保留 `state.max_rounds` 和 `state.target_candidates`
- 在文档中明确它们是 mode 级上下文
- 不要错误地把它们迁到 `adaptive_config.stop`

`StopDecision` 当前只作为简单数据容器，保持显式接口即可：

```python
stop = stop_policy.check(state)
if stop.should_stop:
    ...
```

然后 `agent.py` 中改为：

```python
from .stop_policy import StopPolicy

stop_policy = StopPolicy(adaptive_config)

while True:
    stop = stop_policy.check(state)
    if stop.should_stop:
        logger.info(f"[AdaptiveQAAgent] mode={mode} stopped: {stop.reason}")
        break
```

### 收益

- 主循环更干净
- stop reason 更集中
- 后续新增 stop 条件更容易
- 便于单元测试 stop 行为

## 非目标

这次重构明确不做下面这些事：

1. 不拆 `Executor` 为多文件 action framework

当前 `Executor` 分支数量还不算高，现阶段不是主要痛点。现在就引入 `actions/`、`ActionRegistry`、每动作一个类，收益不一定大于成本。

2. 不引入统一的 `BaseWorkerAgent.run(task)` 抽象

当前各 agent 的输入输出差异很大，强行统一很容易退化成未类型化的大字典接口，反而降低可维护性。

3. 不引入多策略决策框架

本次只做 `DecisionEngine`，不做 `RulePolicy / LLMPolicy / HybridPolicy / PolicyRegistry`。

4. 不重建一个“大一统 AdaptiveQAState”

当前 `ModeState + FeedbackState` 已经足够支撑这次重构目标。先补导出能力，不急着重做状态模型。

## 推荐落地步骤

建议分 3 个小 PR 或 3 个独立提交完成：

### 第一步：DecisionEngine

- 保留 `Action` 数据结构
- 引入 `DecisionEngine`
- 把当前 `decide()` 规则迁移为私有规则方法
- 保证行为不变
- 补单元测试

验收标准：

- 在现有测试和典型输入下，决策结果与重构前保持一致
- 每条规则至少有独立测试覆盖

### 第二步：Metrics 导出

- 为 `ModeState` 增加 `export_metrics()`
- 为 `FeedbackState` 增加 `export_metrics()`，其中 `lifetime_accept_rate` 直接内联计算
- 可选增加 `export_adaptive_metrics(...)`
- 在关键日志中输出统一 metrics

验收标准：

- 日志中能稳定输出 `lifetime_* / recent_* / mode_*` 三层字段
- 不存在命名冲突和语义混淆

### 第三步：StopPolicy

- 新增 `StopPolicy`
- 将 `agent.py` 中的 stop 判断迁移出去
- 保持 stop 行为一致
- 补单元测试

验收标准：

- 现有停止条件行为不变
- 日志能明确输出 stop reason

## 总结

这个重构方案的核心思想是：

- 用 `DecisionEngine` 结构化封装当前规则
- 用轻量 metrics 导出增强实验可解释性
- 用 `StopPolicy` 清理主循环职责

它不是一次大重写，也不是为未来做复杂架构预埋，而是一次贴合当前代码现实的小步重构。

如果交给 Claude Code 执行，应该严格遵守以下边界：

- 优先保持行为不变
- 每一步独立提交
- 每一步都补对应测试
- 不额外引入本方案未要求的抽象层
