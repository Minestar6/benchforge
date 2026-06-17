# Planner Agent 完整技术方案

## 1. 文档目标

本文档给出 `agents/planner_agent` 的完整技术方案，目标是让 planner 成为一个可解释、可测试、可扩展的上层控制器，用于指导以下三个下游智能体：

- 题目生成智能体
- 题目验证智能体
- 难度/模型评估智能体

本方案不包含 `adaptive_qa_agent`。

本文档重点回答四个问题：

1. planner 应控制哪些参数
2. planner 需要哪些反馈
3. planner 如何根据反馈设计下一轮计划
4. planner 内部哪些能力应做成工具，哪些决策需要 LLM 参与

## 2. 总体设计目标

Planner 的定位不是“生成题目的大模型”，而是“评估主导的控制器”。

### 2.1 主目标

持续提高题集对候选模型能力的区分度。

### 2.2 次目标

- 保持题目质量和可验证性
- 保持 topic 覆盖与数据集结构稳定
- 保持每轮计划可执行，不让生成侧频繁失稳

### 2.3 控制原则

每轮规划固定按以下顺序进行：

1. 先看评估反馈，判断当前题集是否真正区分模型能力
2. 再看验证反馈，判断这些区分度是否建立在高质量题目之上
3. 最后看生成反馈，判断下一轮计划是否有可执行性

这意味着 planner 的优先级是：

`评估目标 > 质量护栏 > 生成产能`

## 3. 总体架构

推荐采用：

`智能体壳 + 工具内核 + 定点 LLM 决策`

而不是：

`让 LLM 每轮看一堆反馈后自由输出 patch`

### 3.1 组件分层

#### A. Planner 主循环

负责：

- 维护 `PlannerState`
- 调用单轮 orchestrator
- 接收反馈
- 调用工具链生成下一轮计划

#### B. 工具层

负责：

- 汇总反馈
- 做诊断分类
- 调整参数
- 翻译 patch
- 产出结构化计划

#### C. LLM 参与层

只参与两类高价值非确定性决策：

1. 基于反馈的主题自适应搜索
2. 基于用户目标的首轮评估指标生成

除这两类外，planner 的核心闭环不依赖 LLM 自由发挥。

## 4. Planner 应控制的参数

Planner 应控制高杠杆、可解释、当前下游可承接的参数。

## 4.1 生成智能体参数

推荐控制面：

```yaml
generation_plan:
  topics: list[str]
  mode_targets:
    qa:
      count: int
      difficulty_distribution:
        easy: float
        medium: float
        hard: float
    multiple_choice:
      count: int
      difficulty_distribution:
        easy: float
        medium: float
        hard: float
  candidate_pool:
    target_multiplier: float
  planner:
    topics_per_round: int
  structure:
    single_multi_mix: str | null
    hard_ratio_target: float | null
```

说明：

- `topics` 是最重要的控制面
- `difficulty_distribution` 决定区分度结构
- `candidate_pool.target_multiplier` 决定验证侧是否有足够筛选余量
- `topics_per_round` 控制 topic 扩张广度

当前不建议纳入正式控制面的参数：

- `per_topic_quota`
- retrieval 底层超参
- runtime 底层超参
- chunking 细节

如果下游没有显式承接接口，这些字段不应出现在 planner 正式设计中。

## 4.2 验证智能体参数

推荐控制面：

```yaml
validation_plan:
  citation_validation:
    enabled: bool
    min_citation_score: float
    min_chunk_citation_score: float
    min_answer_citation_score: float
  llm_validation:
    enabled: bool
    min_overall_score: float
  selection:
    mode: off | light | strict
    semantic_similarity_threshold: float
```

控制原则：

- 验证侧主要是质量护栏
- 不是 planner 的主优化面
- 仅当质量明显下滑、重复明显升高、或需要排查边界样本时才主动调

## 4.3 评估智能体参数

推荐控制面：

```yaml
evaluation_plan:
  profile: light | standard | full
  models:
    candidate_model_names: list[str]
    judge_model_name: str | null
  judge:
    enabled: bool
  metrics:
    qa:
      automatic_metrics: list[str]
    multiple_choice:
      automatic_metrics: list[str]
      llm_judge_metrics: list[str]
```

说明：

- `qa` 不需要 planner 指定 `llm_judge_metrics`
- `dataset_metrics` 不应由 planner 每轮指定，直接使用评估智能体内部固定配置
- planner 主要控制：
  - 评估档位
  - 候选模型集合
  - judge 是否启用
  - multiple_choice 的 judge 指标集合

## 5. 首轮冻结的评估指标

这是本方案的关键约束。

### 5.1 结论

LLM 评估指标应在第 0 轮或第 1 轮前，根据用户目标生成，并写入状态；后续轮次默认遵守，不应在中途漂移。

### 5.2 原因

如果每轮都重生成评估指标，会导致：

- 目标函数漂移
- 多轮反馈不可比
- planner 可能“追着指标变化跑”，而不是持续优化同一目标

### 5.3 应冻结的字段

首轮确定并冻结：

- `evaluation_requirements.automatic_metrics`
- `evaluation_requirements.llm_judge_metrics`
- 各 metric 的语义定义和 description

后续允许轮次变化的字段：

- `profile`
- `judge.enabled`
- `candidate_model_names`
- `judge_model_name`

### 5.4 当前代码的契合点

当前 `agents/planner_agent/blueprint_synthesizer.py` 已经具备从 `user_goal` 生成 `evaluation_requirements` 的入口。

推荐进一步明确：

- `GlobalBlueprint.evaluation_requirements` 为首轮冻结真源
- `PlannerState` 中只记录引用，不复制出可漂移版本
- 后续 `RoundSpec` 不重写指标定义，只调整评估档位和模型池

## 6. Planner 输入输出协议

## 6.1 输入

Planner 每轮至少接收：

- `GlobalBlueprint`
- `PlannerState`
- `GeneratorFeedback`
- `ValidatorFeedback`
- `EvaluatorFeedback`

## 6.2 输出

Planner 每轮至少输出两份对象：

1. 给程序消费的结构化计划
2. 给人检查的解释文本

推荐结构：

```yaml
next_round_plan:
  objective:
    primary: increase_model_separation
    secondary:
      - maintain_validation_quality
      - maintain_dataset_stability
  diagnosis_ref: str
  generation_plan: ...
  validation_plan: ...
  evaluation_plan: ...

planner_explanation:
  diagnosis: str
  why: list[str]
  actions: list[str]
```

## 7. 反馈计划

Planner 需要三类反馈。

## 7.1 生成反馈

用途：

- 判断当前计划是否执行出来
- 判断哪些 topic 与 difficulty 真正可产出

推荐字段：

```yaml
generator_feedback:
  round_id: int
  summary:
    total_candidates: int
    global_failures: int
    llm_input_tokens: int
    llm_output_tokens: int
  by_mode:
    qa:
      candidate_count: int
      target_candidate_count: int
      fulfillment_rate: float
      stopped_reason: str | null
      difficulty_counts:
        easy: int
        medium: int
        hard: int
      topic_counts:
        <topic>: int
    multiple_choice:
      candidate_count: int
      target_candidate_count: int
      fulfillment_rate: float
      stopped_reason: str | null
      difficulty_counts:
        easy: int
        medium: int
        hard: int
      topic_counts:
        <topic>: int
  topic_coverage:
    <topic>:
      candidate_count: int
      evidence_doc_count: int | null
      evidence_chunk_count: int | null
  failure_summary:
    no_evidence: int
    empty_generation: int
    parse_error: int
```

## 7.2 验证反馈

用途：

- 判断质量是否过关
- 判断哪些题被质量护栏挡住

推荐字段：

```yaml
validator_feedback:
  round_id: int
  summary:
    total_candidates: int
    citation_passed: int
    llm_passed: int
    final_selected: int
    citation_pass_rate: float
    llm_pass_rate_after_citation: float
    final_selection_rate: float
  quality_signals:
    avg_citation_score_selected: float | null
    avg_llm_overall_score_selected: float | null
    duplicate_rate: float
    overquota_rate: float
  by_topic:
    <topic>:
      selected: int
      rejected: int
      avg_citation_score: float | null
      avg_llm_overall_score: float | null
  by_difficulty:
    easy:
      selected: int
      rejected: int
    medium:
      selected: int
      rejected: int
    hard:
      selected: int
      rejected: int
  by_mode:
    qa:
      selected: int
      reserve: int
      rejected: int
    multiple_choice:
      selected: int
      reserve: int
      rejected: int
```

## 7.3 评估反馈

用途：

- 判断题集是否真正拉开模型能力差距
- 决定下一轮方向

推荐字段：

```yaml
evaluator_feedback:
  round_id: int
  summary:
    num_questions: int
    num_models: int
    llm_input_tokens: int
    llm_output_tokens: int
  discriminative_signals:
    overall_model_gap: float | null
    best_vs_second_gap: float | null
    top_vs_bottom_gap: float | null
    per_question_variance_mean: float | null
    discriminative_question_ratio: float | null
    easy_questions_too_easy_ratio: float | null
    all_models_fail_ratio: float | null
  by_topic:
    <topic>:
      question_count: int
      model_gap: float | null
      too_easy_ratio: float | null
      all_fail_ratio: float | null
  by_difficulty:
    easy:
      question_count: int
      model_gap: float | null
    medium:
      question_count: int
      model_gap: float | null
    hard:
      question_count: int
      model_gap: float | null
  by_mode:
    qa:
      question_count: int
      model_gap: float | null
    multiple_choice:
      question_count: int
      model_gap: float | null
```

## 8. Planner 内部标准对象

推荐在 `planner_agent` 内部增加以下 schema。

## 8.1 RoundFeedbackBundle

```python
@dataclass
class RoundFeedbackBundle:
    round_id: int
    generator: GeneratorFeedback | None
    validator: ValidatorFeedback | None
    evaluator: EvaluatorFeedback | None
```

## 8.2 RoundDiagnosis

```python
@dataclass
class RoundDiagnosis:
    label: str
    confidence: float
    problems: list[str]
    evidence: dict[str, Any]
```

推荐标签：

- `high_quality_low_separation`
- `low_quality_low_separation`
- `high_separation_low_quality`
- `high_separation_imbalanced_distribution`
- `generation_capacity_insufficient`

## 8.3 NextRoundControlPlan

```python
@dataclass
class NextRoundControlPlan:
    qa_count: int
    mc_count: int
    qa_difficulty_distribution: dict[str, float]
    mc_difficulty_distribution: dict[str, float]
    candidate_pool_multiplier: float
    topics_per_round: int

    citation_enabled: bool
    min_citation_score: float
    min_chunk_citation_score: float
    min_answer_citation_score: float

    llm_validation_enabled: bool
    min_overall_score: float
    selection_mode: str
    semantic_similarity_threshold: float

    eval_profile: str
    judge_enabled: bool
    candidate_model_names: list[str]
    judge_model_name: str | None
    mc_llm_judge_metrics: list[str]
```

## 8.4 TopicPlan

```python
@dataclass
class TopicPlan:
    selected_topics: list[str]
    dropped_topics: list[str]
    rationale: list[str]
```

## 8.5 PlannerNarrative

```python
@dataclass
class PlannerNarrative:
    summary: str
    diagnosis_text: str
    next_round_rationale: list[str]
```

## 9. 工具化模块设计

Planner 最好做成“工具驱动”，并把工具边界显式写清。

## 9.1 feedback_aggregator

### 作用

将三类反馈整理成统一输入。

### 输入

- `GeneratorFeedback`
- `ValidatorFeedback`
- `EvaluatorFeedback`

### 输出

- `RoundFeedbackBundle`

### 实现建议

- 纯函数
- 不依赖 LLM
- 只做字段归一化和缺失值处理

## 9.2 round_diagnoser

### 作用

根据反馈输出诊断标签。

### 输入

- `PlannerState`
- `RoundFeedbackBundle`

### 输出

- `RoundDiagnosis`

### 实现建议

- 规则引擎或纯函数
- 使用阈值判断
- 不直接修改参数

### 核心规则示例

- `overall_model_gap` 低且 `citation_pass_rate` 高
  -> `high_quality_low_separation`
- `overall_model_gap` 低且 `citation_pass_rate` 低
  -> `low_quality_low_separation`
- `overall_model_gap` 高但 `avg_llm_overall_score_selected` 低
  -> `high_separation_low_quality`
- `fulfillment_rate` 低且 `empty_generation` 高
  -> `generation_capacity_insufficient`

## 9.3 plan_adjuster

### 作用

根据诊断结果生成下一轮控制计划。

### 输入

- `PlannerState`
- `GlobalBlueprint`
- `RoundDiagnosis`

### 输出

- `NextRoundControlPlan`

### 实现建议

- 规则系统
- 根据诊断标签映射调参动作
- 所有调参应有上下界

### 示例规则

- `high_quality_low_separation`
  - `hard +0.1`
  - `easy -0.1`
  - `topics_per_round -1`
  - `eval_profile -> full`

- `low_quality_low_separation`
  - 不提升 hard
  - `candidate_pool_multiplier +0.2`
  - 收窄 topic
  - `selection_mode -> strict`

- `high_separation_low_quality`
  - 保留 topic 方向
  - 收紧 citation 阈值
  - 收紧 llm_validation
  - 降低部分 count

## 9.4 patch_translator

### 作用

将控制计划翻译成当前下游真正可消费的 `RoundSpec`。

### 输入

- `NextRoundControlPlan`
- `TopicPlan`
- `PlannerState`
- `GlobalBlueprint`

### 输出

- `RoundSpec`

### 实现建议

- 纯函数
- 不做二次推理
- 只做结构转换

### 产出内容

- `RoundSpec.blueprint`
- `qa_agent_patch`
- `verify_agent_patch`
- `model_eval_agent_patch`

## 9.5 planner_explainer

### 作用

生成给人看的诊断说明和行动解释。

### 输入

- `RoundDiagnosis`
- `NextRoundControlPlan`

### 输出

- `PlannerNarrative`

### 实现建议

- 可选使用 LLM
- 若不用 LLM，也可以模板化生成

## 10. 需要 LLM 参与的工具设计

本方案只保留两类 LLM 参与决策。

## 10.1 主题自适应搜索

这是 planner 中最值得使用 LLM 的部分。

### 设计目标

参考 `reference/code/AutoBencher` 的思想，根据历史反馈对主题做自适应搜索，而不是固定轮转 topic backlog。

### AutoBencher 可借鉴的思想

AutoBencher 的核心做法不是简单随机扩 topic，而是：

1. 基于历史表现找弱点或目标区间
2. 生成候选 category
3. 搜索相关 category / related pages
4. 做覆盖性与去重筛选

在 BenchForge 中，推荐改造成：

`反馈中的 topic 表现 -> 语义扩展 -> 候选 topic 池 -> 排序筛选 -> 下一轮 topic`

### 推荐工具拆分

#### A. topic_feedback_summarizer

输入：

- `ValidatorFeedback.by_topic`
- `EvaluatorFeedback.by_topic`

输出：

```yaml
topic_status:
  strong_topics: list[str]
  weak_topics: list[str]
  noisy_topics: list[str]
  oversaturated_topics: list[str]
```

#### B. topic_expander

输入：

- `GlobalBlueprint.user_goal`
- `strong_topics`
- `weak_topics`
- `oversaturated_topics`

输出：

```yaml
candidate_topics:
  - topic: str
    source: strong_neighbor | weak_replacement | broaden | narrow
    rationale: str
```

#### C. topic_ranker

输入：

- `candidate_topics`
- 历史 topic 使用情况
- topic 表现统计

输出：

- `TopicPlan`

### 哪些部分用 LLM

`topic_expander` 适合用 LLM。

它的任务是：

- 为高区分 topic 找相邻细分主题
- 为低区分 topic 找替代主题
- 为“过难导致全失败”的 topic 找上位回退主题

### 哪些部分不用 LLM

- topic 表现统计
- topic 去重
- 历史重复惩罚
- 按轮次预算筛选最终 topics

这些应由工具层完成。

### LLM 提示输入

推荐输入：

- 用户目标
- 首轮冻结的评估指标
- 当前强/弱 topic 列表
- 各 topic 的 `model_gap / too_easy_ratio / all_fail_ratio`
- 历史已使用 topics

### LLM 输出

只输出：

- 候选 topic
- 扩展来源
- 简短理由

不要让 LLM 直接输出 `count`、阈值或 patch。

## 10.2 首轮评估指标生成

这是第二个应由 LLM 参与的决策。

### 设计目标

根据用户目标在首轮生成：

- 自动评估指标
- LLM judge 指标定义

并写入 `GlobalBlueprint.evaluation_requirements`。

### 推荐工具

#### evaluation_metric_designer

输入：

- `UserIntent.user_goal`
- `language`
- 已支持的 mode 列表

输出：

```yaml
evaluation_requirements:
  automatic_metrics:
    qa: list[str]
    multiple_choice: list[str]
  llm_judge_metrics:
    qa: list[metric_def]
    multiple_choice: list[metric_def]
```

### 实现建议

- 当前 `blueprint_synthesizer.py` 已经部分承担此职责
- 建议将“指标生成”从“蓝图整体生成”中逻辑上拆出
- 即便代码不拆文件，也应拆成独立函数

### 约束

- 只在首轮运行
- 后续轮次不重生成
- 输出必须严格受支持的 mode 和 metric 类型约束

## 11. 不建议交给 LLM 的决策

以下决策不建议由 LLM 直接做：

- `min_citation_score` 怎么改
- `min_overall_score` 怎么改
- `hard/easy` 比例改多少
- `count` 增减多少
- `selection.mode` 是否切 strict
- `profile` 切 light/standard/full

这些都应由工具层规则系统控制。

原因：

- 可测试
- 可回溯
- 多轮行为稳定
- 不会因 prompt 抖动导致策略漂移

## 12. 下一轮计划生成逻辑

Planner 不应直接“读反馈 -> 出 patch”，而应走两步：

1. 诊断
2. 调整

### 12.1 诊断分类

推荐五类：

#### A. high_quality_low_separation

特征：

- `citation_pass_rate` 高
- `final_selection_rate` 尚可
- `overall_model_gap` 低

动作：

- 提高 `hard`
- 降低 `easy`
- topic 缩窄到更专门方向
- `topics_per_round` 变小

#### B. low_quality_low_separation

特征：

- 质量差
- 区分度也差

动作：

- 不增加 hard
- 缩窄 topic
- 提高 `candidate_pool.target_multiplier`
- 收紧 `selection`

#### C. high_separation_low_quality

特征：

- 区分度有
- 质量不够

动作：

- 保留 topic 方向
- 收紧验证阈值
- 降低部分 count

#### D. high_separation_imbalanced_distribution

特征：

- 区分度集中于少数 topic 或少数 hard 题

动作：

- 保留强 topic
- 删弱 topic
- 补邻近 topic
- 适度补 medium

#### E. generation_capacity_insufficient

特征：

- 计划没有执行出来

动作：

- 降低 count
- 减少 topics_per_round
- 移除弱证据 topic

## 13. llm_validation 启停策略

### 13.1 结论

不建议因为“过滤数少”就直接关闭 `llm_validation`。

### 13.2 原因

过滤数少可能意味着：

- 候选题本来就干净
- 阈值太宽松
- 样本太少

这三种原因对应完全不同的动作，不能统一处理成“关闭”。

### 13.3 推荐策略

- 默认 `enabled = true`
- 允许降级使用，但不建议长期彻底关闭
- 仅在连续多轮质量稳定且其影响极小时，允许切为弱使用模式

在当前控制面中，如果下游还不支持“抽检模式”，则 planner 仍建议保持 `enabled = true`。

## 14. 与当前代码结构的映射

推荐映射如下：

### 14.1 schema.py

新增或扩展：

- `RoundFeedbackBundle`
- `RoundDiagnosis`
- `NextRoundControlPlan`
- `TopicPlan`
- `PlannerNarrative`

### 14.2 feedback.py

扩展：

- 评估反馈中的区分度字段
- topic 级 gap 统计
- difficulty 级 gap 统计

### 14.3 planner.py

拆分：

- `diagnose_last_round()`
- `adjust_next_round_plan()`
- `select_topics_for_next_round()`
- `build_round_spec_from_plan()`

### 14.4 blueprint_synthesizer.py

加强：

- 把 `evaluation_requirements` 明确为首轮冻结目标函数
- 可逻辑拆出 `evaluation_metric_designer`

## 15. 最小可落地版本

如果先做最小版本，建议优先实现：

1. 评估反馈新增：
   - `overall_model_gap`
   - `discriminative_question_ratio`
   - `easy_questions_too_easy_ratio`
   - `all_models_fail_ratio`
   - `by_topic.model_gap`

2. 诊断标签：
   - `high_quality_low_separation`
   - `low_quality_low_separation`
   - `high_separation_low_quality`
   - `generation_capacity_insufficient`

3. 工具层：
   - `feedback_aggregator`
   - `round_diagnoser`
   - `plan_adjuster`
   - `patch_translator`

4. LLM 参与层：
   - 首轮评估指标生成
   - topic 自适应扩展

## 16. 结论

对于当前 BenchForge，planner 最合理的完整实现应是：

- 用首轮冻结的评估指标定义目标函数
- 用工具层实现反馈聚合、诊断、调参和 patch 翻译
- 用 LLM 仅处理 topic 自适应搜索和首轮评估指标生成
- 用评估反馈主导下一轮方向，用验证反馈约束质量，用生成反馈校正产能

其核心闭环应明确为：

`首轮用户目标 -> 生成固定评估指标 -> 每轮执行 -> 聚合反馈 -> 诊断 -> topic 自适应搜索 -> 调参 -> 下一轮计划`

如果后续实现遵守这个边界，则 planner 的职责、控制面、反馈设计和 LLM 参与方式都是合理且稳定的。
