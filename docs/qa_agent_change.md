# QA Agent 状态与反馈增强改造方案

## 目标

不再单独维护 `adaptive_qa_agent` 作为新的生成主流程，而是在现有 `qa_agent` 基础上引入：

- 更清晰的题目状态管理
- 基于内部反馈的规则化轮次调度
- 难度进化（Evolution）
- Evidence 扩展策略
- 更完整的过程记录与指标落盘

保持现有多轮生成框架不变，不引入新的独立 Agent 主流程。

---

# 一、总体架构

```text
QA Agent
│
├── EvidenceManager
├── Planner
├── ChunkSampler
├── Generator
├── StructuralFilter
├── Evolution
├── State
├── Feedback
│
└── Candidate Questions
          │
          ▼
Internal Round Feedback
          │
          ▼
Rule Planner（下一轮策略）
```

核心原则：

- `qa_agent` 仍然负责完整生成闭环
- 不新增独立的质量裁决 Agent
- 保留现有 structural filter 边界
- 新增的是状态表达、反馈聚合、规则调度策略和过程记录

---

# 二、改造约束

这次改造的前提不是重写 `qa_agent`，而是在不丢失现有优势的前提下做增强。

必须保留的现有能力：

- 初始广度覆盖 + 后续补齐的两阶段规划
- 基于目标缺口和剩余轮数的动态参数计算
- 一轮多 topic 并发执行
- chunk 组合去重和全局 chunk usage 记录
- 基于 config 的 chunk mix / generation yield 调参逻辑
- 现有 EvidenceManager 的检索、summary、chunking、pool building 能力

不应发生的退化：

- 不能把 Planner 退化成只返回一个策略名
- 不能丢掉现有 `single_k / multi_k / target_candidates_per_topic` 的动态计算
- 不能把并发多 topic 主路径改成低吞吐串行 agent loop
- 不能绕开现有 chunk 去重语义
- 不能让 `EXPAND_EVIDENCE` 和 `EVOLVE` 替代普通 chunk 采样生成主路径

---

# 三、职责划分

## QA Agent

负责：

- evidence 管理
- round planning
- chunk sampling
- question generation
- structural filtering
- difficulty evolution
- 状态管理
- 内部反馈聚合
- 下一轮策略调整
- 过程记录和指标落盘

不负责：

- 复杂外部编排
- 新的独立验证工作流

---

# 四、State 改造

状态上需要区分题目状态，但不建议维护三份独立题池。

推荐方案：

- 只保留一个 `candidate_questions`
- 每个题目带 `status`
- `accepted/rejected/evolved` 通过状态字段和派生视图获得

这样可以避免：

- 同一题在多个列表中重复维护
- 计数与列表不一致
- 后续进化题与原题关联困难

---

## 建议数据结构

新增 `state.py`

说明：`CandidateRecord.source_strategy` 类型为 `RoundStrategy`，该枚举定义在 `planner.py`。实现时应在 `state.py` 顶部 `from __future__ import annotations`，或直接 import `RoundStrategy`，避免前向引用运行时 `NameError`。

```python
from __future__ import annotations

class CandidateStatus(Enum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EVOLVED = "evolved"


@dataclass
class CandidateRecord:
    question_id: str
    question: str
    answer: str
    topic: str
    difficulty: str
    status: CandidateStatus
    source_round: int
    source_strategy: RoundStrategy  # defined in planner.py
    chunk_ids: list[str]
    reject_reason: str | None = None
    parent_question_id: str | None = None


@dataclass
class ModeState:
    mode: str
    target_count: int

    candidate_questions: list[CandidateRecord]

    consecutive_empty_rounds: int

    used_chunk_combinations: set
    trace: list[dict]
```

---

## 派生指标

新增指标：

```text
candidate_count
accepted_count
rejected_count
evolved_count

accept_rate

topic_accept_rate
difficulty_accept_rate
too_easy_ratio

consecutive_empty_rounds
```

说明：

- `accepted_count`、`rejected_count` 都应从单一题池按 `status` 派生
- `failure_reason_counts`、`accepted_topic_counts`、`accepted_difficulty_counts` 等都应通过方法实时派生
- 如需性能优化，可在落盘阶段缓存统计结果，但不应作为 `ModeState` 的长期冗余字段
- `ModeState` 应尽量保持“单池 + 少量过程状态”的轻量结构
- `status` 不应使用裸字符串，应统一使用 `CandidateStatus`

---

# 五、内部反馈对象

新增：

```text
qa_agent/feedback.py
```

统一 `qa_agent` 内部反馈对象。

这套反馈的目标不是驱动一个强 Agent 决策器，而是为规则 Planner 提供稳定、可测的流水线信号。

```python
@dataclass
class RoundFeedback:
    mode: str
    round_id: int
    strategy: "RoundStrategy"

    generated_count: int  # 本轮新增 generated 数量
    accepted_count: int   # 本轮新增 accepted 数量
    rejected_count: int   # 本轮新增 rejected 数量
    evolved_count: int    # 本轮新增 evolved 数量

    failure_reason_counts: dict

    generated_topic_counts: dict
    accepted_topic_counts: dict
    rejected_topic_counts: dict

    generated_difficulty_counts: dict
    accepted_difficulty_counts: dict
    rejected_difficulty_counts: dict

    accept_rate: float
    duplicate_chunk_ratio: float
    empty_round: bool
    too_easy_ratio: float
```

用途：

- 给 Planner 提供下一轮调度信号
- 给 State 提供结构化更新输入
- 给 trace / report / metrics 提供统一落盘格式

这里的反馈是 `qa_agent` 内部反馈，不依赖外部 Agent。

说明：

- `RoundFeedback` 表示的是“本轮增量反馈”，不是截至当前轮次的累计状态
- 截至当前轮次的累计值，例如是否满足 `accepted >= target_count`，应统一从 `ModeState` 读取
- 优先记录可观测运行信号，而不是复杂推断信号
- 先服务规则 Planner
- 后续如果要加 `LLM Advisor`，也应只消费这套压缩反馈，不直接读取全部原始状态

---

# 六、Planner 改造

现有 Planner 已具备：

- topic coverage
- difficulty coverage
- dynamic round plan

这些逻辑保留。

新增 `RoundFeedback` 对下一轮策略的影响。

当前方案只做 `Rule Planner`，不把 LLM 作为默认规划器。

Planner 的职责是：

- 根据 `ModeState + RoundFeedback` 选择下一轮策略
- 根据反馈调节下一轮执行参数
- 约束在有限动作集合内决策
- 保持主路径仍然是既有流水线

Planner 不负责：

- 执行 evidence 扩展
- 执行题目进化
- 调用 LLM 自由规划动作

核心要求：

- 必须继承现有 Planner 的动态参数调节能力
- 新逻辑应该是在现有 `ModeRoundPlan` 基础上扩字段，而不是只引入 `RoundStrategy`
- `Rule Planner` 的主要智能应体现在参数调优，而不仅是动作切换

---

## RoundPlan 而不只是 RoundStrategy

`RoundStrategy` 只负责定义这一轮的路径类型。

真正可执行的规划对象应该是参数化的 `RoundPlan`，保留现有动态参数，并在此基础上扩展增强动作参数。

```python
@dataclass(frozen=True)
class ModeRoundPlan:
    mode: str
    round_in_mode: int
    strategy: "RoundStrategy"
    difficulty: str
    topics: tuple[str, ...]

    single_k: int
    multi_k: int
    target_candidates_per_topic: int

    evidence_strategy: EvidenceStrategy = EvidenceStrategy.DEFAULT
    expand_docs: int = 0
    expand_topics: tuple[str, ...] = ()
    evolve_source_count: int = 0

    reason: str = ""
```

说明：

- `strategy` 决定这轮属于哪一类动作
- `single_k / multi_k / target_candidates_per_topic` 仍然是核心参数
- `evidence_strategy` 应为受限枚举，不应是自由字符串
- `expand_docs / expand_topics` 只在 `EXPAND_EVIDENCE` 时生效
- `evolve_source_count` 只在 `EVOLVE_TO_HARDER` 时生效
- `strategy` 不应使用裸字符串，应统一使用 `RoundStrategy`

建议定义：

```python
class EvidenceStrategy(Enum):
    DEFAULT = "default"
    HIGH_HARD_SCORE = "high_hard_score"
    MULTI_GROUP = "multi_group"
```

---

## 新增策略

```python
class RoundStrategy(Enum):
    NORMAL_GENERATE
    FOCUS_TOPIC
    FOCUS_DIFFICULTY
    EXPAND_EVIDENCE
    EVOLVE_TO_HARDER
```

说明：

- `NORMAL_GENERATE` 仍然是主路径
- `FOCUS_TOPIC` / `FOCUS_DIFFICULTY` 仍然走普通 chunk 采样生成
- `EXPAND_EVIDENCE` 是前置增强动作，之后仍回到普通采样生成
- `EVOLVE_TO_HARDER` 是补充动作，不替代直接生成 hard
- `EXPAND_EVIDENCE` 用于补充当前 topic 的证据，而不是重置整个 evidence pool
- `EVOLVE_TO_HARDER` 只允许从 accepted `medium` 题中取样，不能从 `easy` 题直接进化到 `hard`

建议分阶段启用：

- Phase 1: `NORMAL_GENERATE` / `FOCUS_TOPIC` / `FOCUS_DIFFICULTY`
- Phase 2: `EXPAND_EVIDENCE`
- Phase 3: `EVOLVE_TO_HARDER`
- Phase 4: 如需引入 multihop 专项策略，再新增 `FOCUS_DIFFICULTY_MULTIHOP`

---

## 参数调优

规则 Planner 不能只决定 `strategy`，还必须根据反馈调节参数。

至少应调节：

```text
topics
difficulty
single_k
multi_k
target_candidates_per_topic
evidence_strategy
expand_docs
expand_topics
evolve_source_count
```

调节原则：

- `missing_topics` 高：收缩 topic 范围，提升目标 topic 的 `target_candidates_per_topic`
- `difficulty_gap` 高：倾斜目标难度，并调整 single/multi chunk 配比
- `hard_gap` 高：不仅提升 `difficulty=hard`，还应切换到更偏 hard 的 `evidence_strategy`
- `accept_rate` 低：优先收缩范围和调整采样强度，而不是立刻扩检索
- `consecutive_empty_rounds` 高：才触发 `EXPAND_EVIDENCE`
- `duplicate_chunk_ratio` 高：优先切 topic / difficulty，必要时再扩 evidence
- `EVOLVE_TO_HARDER` 触发时，必须确认存在足够的 accepted `medium` surplus 可供进化

也就是说：

```text
RoundFeedback
-> choose strategy
-> tune parameters
-> build ModeRoundPlan
```

而不是：

```text
RoundFeedback
-> choose strategy name only
```

---

## 决策逻辑

```python
if accepted_count >= target_count:
    STOP

elif consecutive_empty_rounds > threshold:
    EXPAND_EVIDENCE

elif hard_gap > threshold:
    FOCUS_DIFFICULTY

elif hard_gap > 0 and too_easy_ratio > threshold and evolvable_medium_surplus >= evolve_source_count:
    EVOLVE_TO_HARDER

elif accept_rate < threshold:
    FOCUS_TOPIC or FOCUS_DIFFICULTY

elif missing_topics:
    FOCUS_TOPIC

elif difficulty_gap:
    FOCUS_DIFFICULTY

else:
    NORMAL_GENERATE
```

说明：

- `accepted_count >= target_count` 的停止判断应读取 `ModeState` 的累计 accepted 数，而不是 `RoundFeedback.accepted_count`
- `topic gap` 和 `evidence gap` 要分开处理
- `topic accept rate` 低，不一定意味着 evidence 不足，也可能只是采样未命中
- `EXPAND_EVIDENCE` 只应在规则信号明确表明普通生成效果不佳时触发
- `FOCUS_DIFFICULTY` 不等于只有 `difficulty="hard"`，还应联动参数和证据偏好
- `Evolution` 应在“已有足够 accepted medium 题且存在 surplus”前提下触发
- 第一阶段应优先验证 `FOCUS_TOPIC` 和 `FOCUS_DIFFICULTY` 是否足够解决缺口
- `multihop` 相关信号和策略不应在 Phase 1 默认启用，应在后续单独验证后再引入

---

# 七、Structural Filter 保留

继续保留现有 Filter。

职责：

```text
JSON解析
字段完整
空题过滤
格式过滤
重复过滤
```

不要扩展为：

```text
复杂质量判断
答案正确性判断
深层语义验证
```

Structural Filter 仍然只做轻量结构约束。

---

# 八、Difficulty Control

新增：

```text
qa_agent/evolution.py
```

难度控制不应只等于 `Evolution`。

更接近 `adaptive_qa_agent` 的合理实现应包含 4 层：

```text
1. difficulty target
2. hard gap tracking
3. hard generation strategies
4. evolution fallback
```

---

## 1. 难度目标

建议保留面向 mode 的难度目标概念，例如：

```python
target_hard_ratio: float
```

用途：

- 描述当前 mode 最终希望达到的 hard 占比
- 用于驱动 `hard_gap`
- 避免 Planner 只凭单轮感觉决定是否出 hard

---

## 2. Hard Gap

建议引入：

```python
hard_gap
```

语义：

```text
目标 hard 占比 - 当前 accepted hard 占比
```

说明：

- `hard_gap` 应基于 accepted 结果计算
- 它是是否需要继续补 hard 的核心信号
- 这比简单地“本轮生成 hard”更稳定

---

## 3. Hard Generation Strategies

`adaptive_qa_agent` 的经验表明，hard 不只是一个 difficulty 标签，而应该拆成不同的生成路径。

Phase 1 建议先保留一种：

```text
FOCUS_DIFFICULTY
```

对应语义：

- `FOCUS_DIFFICULTY`
  - 目标是补 hard gap
  - 使用更偏 hard 的证据选择或 prompt 约束

Phase 2/3 如验证确有必要，再引入多跳专项策略。此时更合理的方式是通过受限 `EvidenceStrategy` 表达，例如：

```text
high_hard_score
multi_group
```

而不是只写：

```text
difficulty = hard
```

---

## 4. Evolution Fallback

目标：

```text
medium -> hard
```

不是完全替代直接生成 hard，而是作为补充路径。

---

## Evolution 流程

```text
选择 accepted medium

↓

构造 evolve prompt

↓

生成 harder version

↓

structural filter

↓

写入 candidate pool

↓

标记 parent_question_id
```

建议：

- 只从 `accepted medium` 题中选择可进化样本
- 进化后的题目进入同一 `candidate_questions`
- 原题与新题通过 `parent_question_id` 关联
- 进化成功与否写入 round trace
- `Evolution` 应在 `too_easy_ratio` 偏高、且直接 hard 生成仍未补齐时再触发
- 不允许 `easy -> hard` 的跨级进化
- 更稳妥的门控方式是：只从“超出当前 medium 目标配额的 accepted medium”里取样

---

## 难度控制总结

建议最终按这条链路组织：

```text
target_hard_ratio
-> hard_gap
-> choose hard generation strategy
-> if needed, evolve accepted medium
```

也就是说，主线里的难度控制应包括：

- 难度目标
- 难度缺口
- 难度导向的参数调优
- 难度导向的证据策略
- Evolution 补救

而不是只保留一个 `evolution.py`

---

# 九、Evidence Expansion

新增能力：

```python
append_evidence()
```

触发条件建议：

```text
consecutive_empty_rounds 偏高

某 topic 长期生成不足

某 difficulty 长期补齐失败
```

流程：

```text
expand evidence

↓

retrieve additional documents

↓

deduplicate existing document_ids

↓

summarize new documents

↓

chunk new documents

↓

refresh topic evidence pool

↓

sample chunk

↓

generate
```

说明：

- 不重新初始化全部 Evidence
- 只对当前需要补强的 topic 做局部扩展
- `EXPAND_EVIDENCE` 应是局部修正动作，不是默认主路径
- Planner 只决定触发，不负责检索、总结、chunking 的具体实现
- 具体执行应由 `EvidenceManager` 完成
- 应尽量复用现有 `prepare_evidence()` 的检索、summary、chunking、pool building 路径
- 更合理的实现是新增 `expand_topic_evidence(topic, ...)`，而不是重写整个 evidence 生命周期

---

# 十、Agent 主循环

```text
prepare evidence

while not stop:

    build round plan

    sample chunks

    generate questions

    structural filter

    build round feedback

    update state

    build next round plan

    save trace / metrics / artifacts
```

重点变化：

- 每轮结束后产出 `RoundFeedback`
- Planner 消费的是内部反馈，不是外部验证反馈
- Planner 当前是规则 Planner，不使用 LLM 自由规划
- State 更新与落盘围绕单一题池进行

---

# 十一、实施顺序

## Phase 1

先改状态模型：

```text
单一 candidate pool
status 字段
failure_reason_counts 派生统计
parent_question_id
```

---

## Phase 2

新增 `feedback.py`

统一每轮反馈对象：

```text
RoundFeedback
```

同时落盘基础反馈：

```text
round_feedback.jsonl
```

---

## Phase 3

Planner 使用 `RoundFeedback` 和 `ModeState`：

```text
focus topic
focus difficulty
normal generate
```

---

## Phase 4

新增 `EXPAND_EVIDENCE`

支持：

```text
retrieve additional documents
deduplicate
summarize
chunk
refresh evidence pool
```

---

## Phase 5

新增 `evolution.py`

支持：

```text
medium -> hard
```

---

## Phase 6

补齐增强动作相关落盘与扩展指标：

```text
evolution_records
evidence_expansion_records
extended_mode_metrics
```

---

# 十二、按文件实施清单

## 1. `agents/qa_agent/state.py`

保留：

- `GlobalState.used_chunk_combinations`
- `GlobalState.chunk_usage_counts`
- `ModeState.round_in_mode`
- `ModeState.trace`
- 现有轻量状态结构

新增：

- `CandidateRecord`
- `CandidateStatus`
- 单池题目状态 `status`
- `parent_question_id`
- 面向反馈聚合的派生统计方法
- 面向反馈聚合的失败原因统计方法

不要做：

- 拆成三份长期独立维护的题池
- 让状态更新依赖外部验证流程

目标：

- 保留现有轻量状态模型
- 在不打乱现有 stop / trace 语义的前提下支持更细粒度反馈

---

## 2. `agents/qa_agent/planner.py`

保留：

- `mode_candidate_target()`
- `remaining_mode_rounds()`
- `resolve_chunk_mix()`
- `compute_dynamic_chunk_k()`
- `choose_difficulty_for_mode()`
- `choose_low_coverage_topics_for_mode()`
- `initial_breadth -> adaptive supplement` 两阶段结构

新增：

- `RoundStrategy`
- 基于 `RoundFeedback` 的规则判断
- 在现有 `ModeRoundPlan` 上扩展 `expand_docs / expand_topics / evolve_source_count`
- 参数调优逻辑，而不是只选策略名

不要做：

- 替换现有动态参数计算
- 把 Planner 改成自由决策式 LLM loop
- 让 `EXPAND_EVIDENCE` 和 `EVOLVE` 覆盖普通生成主路径

目标：

- 保留现有 Planner 的核心优势
- 新能力通过增量字段和规则分支接入

---

## 3. `agents/qa_agent/evidence_manager.py`

保留：

- `prepare_evidence()`
- 检索 + summary + chunking + evidence pool building 主路径
- URL 去重和文档缓存逻辑

新增：

- `expand_topic_evidence(topic, ...)`
- 面向单 topic 的局部扩检索
- 对新增文档做去重、summary、chunking
- 将新增 chunk append 到现有 topic evidence pool

不要做：

- 重写整个 evidence 生命周期
- 每次扩 evidence 都重建所有 topic 的 evidence pool
- 把 evidence expansion 逻辑塞进 Planner

目标：

- 复用现有成熟能力
- 只补局部扩展入口

---

## 4. `agents/qa_agent/executor.py`

保留：

- 一轮多 topic 并发执行
- gather 后串行状态合并
- 现有 chunk reservation / duplicate protection
- quota 截断逻辑
- structural filter 位置不变

新增：

- `RoundFeedback` 聚合
- `EXPAND_EVIDENCE` 的执行分支
- `EVOLVE_TO_HARDER` 的执行分支
- 增强 trace 字段

不要做：

- 把执行器改成串行 agent action loop
- 破坏现有并发 topic 执行模型
- 绕开现有 `sample_chunks()` 去重语义

目标：

- 增强执行路径
- 不降低吞吐和多样性

---

## 5. `agents/qa_agent/storage.py`

保留：

- 现有 mode/global/report 落盘结构
- 下游 `shared_state.json` 兼容性

新增：

- `RoundFeedback` 落盘
- 派生 metrics 落盘
- `evolution_records`
- `evidence_expansion_records`

不要做：

- 打破现有输出目录约定
- 让下游依赖新的强耦合 artifact 才能运行

目标：

- 增加可观测性
- 保持现有运行产物兼容

---

## 6. `agents/qa_agent/feedback.py`

新增文件，职责：

- 定义 `RoundFeedback`
- 提供 round result -> feedback 的聚合函数
- 输出可供 Planner 直接消费的压缩信号

设计要求：

- 以可观测计数和比例为主
- 避免掺入过多语义推断
- 保持和 `executor.py` 输出对齐

---

## 7. `agents/qa_agent/evolution.py`

新增文件，职责：

- 只从超出当前配额的 `accepted medium` 题中挑选可进化样本
- 生成更高难度版本
- 与原题建立 `parent_question_id` 关联

设计要求：

- 低频触发
- 不替代普通 chunk 生成
- 输出仍回流到单一 candidate pool
- 不允许从 `easy` 题直接进化到 `hard`

---

# 十三、Artifacts / Intermediate Files

这次改造不应丢掉现有 `qa_agent` 已经保存的关键中间过程文件。

现有主线已具备的文件应继续保留：

```text
runs/{task_id}/{run_id}/llm_calls.jsonl
runs/{task_id}/{run_id}/generation_report.json
runs/{task_id}/{run_id}/global_state.json
runs/{task_id}/{run_id}/used_chunks.json
runs/{task_id}/{run_id}/shared_state.json

runs/{task_id}/{run_id}/evidence/chunked.jsonl
runs/{task_id}/{run_id}/evidence/single_units.json
runs/{task_id}/{run_id}/evidence/multi_units.json

runs/{task_id}/{run_id}/{mode}/candidate_pool.json
runs/{task_id}/{run_id}/{mode}/failures.json
runs/{task_id}/{run_id}/{mode}/mode_state.json
```

说明：

- `llm_calls.jsonl` 继续作为模型调用总记录
- `chunked.jsonl / single_units.json / multi_units.json` 继续作为证据准备阶段的核心产物
- `candidate_pool.json` 仍然是下游消费的主输入之一

---

## 新增文件

为了支持规则 Planner 和增强动作复盘，建议新增：

```text
runs/{task_id}/{run_id}/{mode}/round_plan.jsonl
runs/{task_id}/{run_id}/{mode}/round_feedback.jsonl
runs/{task_id}/{run_id}/{mode}/round_trace.jsonl
runs/{task_id}/{run_id}/{mode}/mode_metrics.json
```

如果启用增强动作，再新增：

```text
runs/{task_id}/{run_id}/{mode}/evolution_records.jsonl
runs/{task_id}/{run_id}/evidence/evidence_expansion_records.jsonl
```

---

## 文件用途

### `llm_calls.jsonl`

用途：

- 记录所有模型调用
- 统计 token 使用
- 关联 `llm_call_id` 回溯某轮生成

要求：

- 继续复用现有 tracing 机制
- 不为规则 Planner 另起一套调用记录文件

### `evidence/chunked.jsonl`

用途：

- 保存文档、summary、chunk 文本
- 支持下游回查 chunk_text
- 支持 evidence expansion 后的统一证据视图

要求：

- 不建议拆成初始 / 扩展两套主文件
- 更合理的做法是追加写入，并在记录中标记：

```text
source = initial | expanded
topic = ...
document_id = ...
```

### `evidence/single_units.json` / `multi_units.json`

用途：

- 保存证据池构造结果
- 支持后续排查 hard/easy 倾向、single/multi 采样分布

要求：

- 继续沿用现有结构
- 若 evidence expansion 更新了 evidence pool，应同步刷新

### `{mode}/candidate_pool.json`

用途：

- 保存单池候选题目列表
- 继续作为下游统一消费输入

要求：

- 每题保留 `status`
- 保留 `chunk_ids`
- 若存在 evolve 题，保留 `parent_question_id`

### `{mode}/round_plan.jsonl`

用途：

- 记录每轮 Planner 实际输出的 `ModeRoundPlan`
- 复盘“这一轮为什么选了这些参数”

建议字段：

```text
round_in_mode
strategy
topics
difficulty
single_k
multi_k
target_candidates_per_topic
evidence_strategy
expand_docs
expand_topics
evolve_source_count
reason
```

### `{mode}/round_feedback.jsonl`

用途：

- 记录每轮 `RoundFeedback`
- 复盘“这一轮生成效果到底怎样”

要求：

- Phase 2 即落盘
- `accepted_count / rejected_count` 明确表示本轮增量，不是累计值

### `{mode}/round_trace.jsonl`

用途：

- 聚合 round plan、execution result、round feedback 的完整视图
- 作为人工调试和回归分析的主文件

建议字段：

```text
round_in_mode
round_plan
execution_summary
round_feedback
mode_metrics_snapshot
```

### `{mode}/mode_metrics.json`

用途：

- 保存 mode 级累计派生指标快照
- 供 report 和调试快速读取

建议内容：

```text
candidate_count
accepted_count
rejected_count
accept_rate
topic_distribution
difficulty_distribution
hard_gap
stopped_reason
```

### `{mode}/evolution_records.jsonl`

用途：

- 记录每次 evolve 的输入输出关系

建议字段：

```text
round_in_mode
source_question_ids
source_difficulties
generated_question_ids
accepted_count
```

### `evidence/evidence_expansion_records.jsonl`

用途：

- 记录每次 `EXPAND_EVIDENCE` 的具体扩展结果

建议字段：

```text
round_in_mode
topics
queries
new_document_ids
new_chunk_ids
new_single_units
new_multi_units
```

---

## 分阶段落盘建议

### Phase 1

保留现有落盘，并适配单池状态：

```text
candidate_pool.json
mode_state.json
failures.json
generation_report.json
llm_calls.jsonl
chunked.jsonl
single_units.json
multi_units.json
```

### Phase 2

新增基础反馈与计划落盘：

```text
round_plan.jsonl
round_feedback.jsonl
mode_metrics.json
```

### Phase 3

新增完整轮级追踪：

```text
round_trace.jsonl
```

### Phase 4+

新增增强动作落盘：

```text
evidence_expansion_records.jsonl
evolution_records.jsonl
```

---

## 核心判断

这次方案最关键的新文件不是 `llm_calls.jsonl` 或 `chunked.jsonl`，因为它们现有主线已经具备。

真正必须补齐的是：

```text
round_plan.jsonl
round_feedback.jsonl
round_trace.jsonl
```

因为它们直接决定：

- 规则 Planner 是否可复盘
- 参数调优是否可解释
- 新方案是否比 `adaptive_qa_agent` 更容易调试

---

# 十四、最终结论

不建议继续把能力分叉到新的 `adaptive_qa_agent` 主流程。

更合理的方案是：

```text
保留现有 QA Agent 多轮框架

新增：
    单池状态管理
    内部反馈对象
    Rule Planner
    Evidence Expansion
    Evolution
    Trace / Metrics 增强
```

形成：

```text
Generation
+
Round Feedback
+
Rule Planner Adjustment
```

这样改动更集中，兼容性更高，也更容易把 `adaptive_qa_agent` 中真正有价值的能力回收进主线。
