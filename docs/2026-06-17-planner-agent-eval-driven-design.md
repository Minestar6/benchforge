# Planner Agent 评估主导闭环设计

## 1. 目标

本文档定义 `agents/planner_agent` 如何指导以下三个下游智能体：

- 题目生成智能体
- 题目验证智能体
- 难度/模型评估智能体

设计目标不是让 planner 直接决定题目内容，而是让它成为一个评估主导的控制器：

- 主目标：持续提高题集对候选模型能力的区分度
- 次目标：保持题目质量、可验证性和数据集分布稳定
- 控制原则：先看评估反馈决定方向，再看验证反馈设质量护栏，最后看生成反馈校正产能

本文档只讨论这三个智能体，不包含 `adaptive_qa_agent`。

## 2. 角色定义

### 2.1 Planner Agent

Planner 负责四件事：

1. 诊断上一轮数据集的问题
2. 生成下一轮计划
3. 将计划翻译成三个下游智能体可消费的参数
4. 汇总反馈并更新下一轮控制参数

Planner 不负责：

- 直接生成题目
- 直接做题目验证
- 直接执行模型评估

### 2.2 下游智能体职责

- 生成智能体负责执行题目生成计划
- 验证智能体负责执行质量筛选和去重
- 评估智能体负责产出区分度和难度校准相关指标

Planner 与下游的接口应是显式控制面，而不是隐式依赖某些中间文件格式。

## 3. 设计原则

### 3.1 评估主导

如果 planner 的目标是“区分模型能力”，则最关键的反馈不应是：

- 生成了多少题
- 通过验证多少题

而应是：

- 这些题是否真的拉开了模型表现差距
- 差距主要出现在哪些 topic、mode、difficulty 上
- 数据集是否被大量“所有模型都能做”或“所有模型都做不出”的题污染

### 3.2 验证护栏

评估结果不能单独决定下一轮方向。若某轮题看起来有区分度，但引用质量差、题干不清晰、重复过多，则不能直接沿用该方向。

验证反馈的职责是给 planner 设置质量边界：

- 低质量的区分度不能被当成有效信号
- 高质量但低区分度的数据集，需要往更难、更窄、更结构化的方向调整

### 3.3 生成侧只调高杠杆参数

Planner 不应细粒度控制生成智能体的全部工程参数。应只调高杠杆且可解释的参数：

- topic
- count
- difficulty_distribution
- candidate_pool 倍率
- topic 扩张范围
- single/multi 结构

这样才能保持闭环稳定，且便于解释每轮调整原因。

## 4. Planner 应控制的参数

## 4.1 生成智能体参数

推荐 planner 可控参数：

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

优先级最高的生成参数：

1. `topics`
2. `difficulty_distribution`
3. `single_multi_mix`
4. `candidate_pool.target_multiplier`

原因：

- topic 决定模型能力切分面
- difficulty 决定是否会出现“全对”或“全错”
- single/multi 决定题目的推理深度和证据整合强度
- candidate_pool 决定验证阶段是否有足够筛选空间

说明：

- `per_topic_quota` 不应作为当前 planner 的标准控制参数
- 当前文档中不再将其纳入 planner 控制面
- 若未来生成智能体显式支持按 topic 配额生成，再单独引入
- 当前阶段通过 `topics` 和 `topics_per_round` 间接控制 topic 分布更合理

不建议 planner 高频调整的参数：

- model temperature
- retrieval 底层超参
- runtime 超时
- chunking 细节

这些更适合作为工程默认配置，而不是闭环控制变量。

## 4.2 验证智能体参数

推荐 planner 可控参数：

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

- 验证侧主要用来设护栏，不作为主优化面
- 只有当质量明显下滑或重复明显升高时，planner 才应主动调整验证阈值
- 如果质量已经稳定但区分度不足，应先调生成计划，再考虑收紧验证

推荐使用方式：

- `min_citation_score`：控制证据质量下界
- `min_overall_score`：控制题目清晰度和可回答性下界
- `selection.mode`：控制最终入选时的严格程度
- `semantic_similarity_threshold`：控制重复题和近重复题压制强度

## 4.3 评估智能体参数

推荐 planner 可控参数：

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

控制原则：

- 评估侧参数不是为了“让评估更漂亮”，而是为了让 planner 更准确地判断区分度
- 当 planner 进入探索轮时，可以用 `light`
- 当 planner 怀疑数据集有潜在高价值但证据不足时，应升到 `full`

推荐保留给 planner 的核心能力：

- 控制候选模型池
- 控制是否启用 judge
- 控制 multiple_choice 的 judge 指标集合
- 控制评估档位

说明：

- `qa` 不需要 planner 指定 `llm_judge_metrics`
- `dataset_metrics` 不需要 planner 每轮显式指定，直接使用评估智能体内部固定配置即可
- planner 更应关注评估档位和候选模型集合，而不是反复改底层 dataset metric 列表

## 5. 计划如何下发

Planner 不应只下发三个 patch，而应生成一份显式的 `RoundPlan`。

推荐结构：

```json
{
  "round_id": 2,
  "objective": {
    "primary": "increase_model_separation",
    "secondary": [
      "maintain_validation_quality",
      "maintain_dataset_diversity"
    ]
  },
  "diagnosis": {
    "last_round_problem": "high_quality_but_low_model_separation",
    "evidence": [
      "overall_model_gap_small",
      "easy_questions_too_easy_ratio_high",
      "topicwise_gap_concentrated"
    ]
  },
  "generation_plan": {
    "topics": ["retrieval robustness", "citation grounding"],
    "mode_targets": {
      "qa": {
        "count": 16,
        "difficulty_distribution": {"easy": 0.1, "medium": 0.4, "hard": 0.5}
      },
      "multiple_choice": {
        "count": 8,
        "difficulty_distribution": {"easy": 0.1, "medium": 0.5, "hard": 0.4}
      }
    },
    "candidate_pool": {"target_multiplier": 2.4},
    "planner": {"topics_per_round": 2},
    "structure": {
      "single_multi_mix": "more_multi",
      "hard_ratio_target": 0.45
    }
  },
  "validation_plan": {
    "citation_validation": {
      "enabled": true,
      "min_citation_score": 0.70,
      "min_chunk_citation_score": 0.88,
      "min_answer_citation_score": 0.78
    },
    "llm_validation": {
      "enabled": true,
      "min_overall_score": 0.78
    },
    "selection": {
      "mode": "strict",
      "semantic_similarity_threshold": 0.92
    }
  },
  "evaluation_plan": {
    "profile": "full",
    "models": {
      "candidate_model_names": ["model_a", "model_b", "model_c"],
      "judge_model_name": "judge_x"
    },
    "judge": {
      "enabled": true
    },
    "focus_metrics": [
      "overall_model_gap",
      "hard_question_gap",
      "topicwise_gap",
      "discriminative_question_ratio"
    ]
  }
}
```

这个结构有三个作用：

1. 让 planner 输出“诊断”而不只是“命令”
2. 让每轮计划具备解释性
3. 让下游 patch 生成变成机械翻译，而不是再做二次推理

## 6. Planner 必须接收哪些反馈

Planner 应接收三类反馈：

- 生成反馈
- 验证反馈
- 评估反馈

三者的职责不同，不应混在一个 summary 里。

## 6.1 生成反馈

生成反馈回答的问题是：

“本轮计划是否被成功执行出来了？”

推荐字段：

```yaml
generator_feedback:
  summary:
    total_candidates: int
    generation_success_rate: float
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
      single_multi_ratio: dict[str, float] | null
    multiple_choice:
      ...
  topic_coverage:
    topic_a:
      candidate_count: int
      evidence_doc_count: int | null
      evidence_chunk_count: int | null
  failure_summary:
    no_evidence: int
    empty_generation: int
    parse_error: int
```

Planner 最关心的信号：

- `fulfillment_rate`
- `difficulty_counts`
- `topic_counts`
- `single_multi_ratio`
- `failure_summary`

如果这些信号异常，说明下一轮再怎么调验证和评估都无效，必须先解决产能问题。

## 6.2 验证反馈

验证反馈回答的问题是：

“这批题的质量、可验证性和去重情况是否过关？”

推荐字段：

```yaml
validator_feedback:
  summary:
    total_candidates: int
    citation_passed: int
    llm_passed: int
    final_selected: int
    citation_pass_rate: float
    llm_pass_rate_after_citation: float
    final_selection_rate: float
  quality_signals:
    avg_citation_score_all: float | null
    avg_citation_score_selected: float | null
    avg_llm_overall_score_all: float | null
    avg_llm_overall_score_selected: float | null
    duplicate_rate: float
    overquota_rate: float
  by_topic:
    topic_a:
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

Planner 最关心的信号：

- `citation_pass_rate`
- `final_selection_rate`
- `duplicate_rate`
- `by_difficulty`
- `by_topic`

验证反馈主要用于判断：

- 质量是否足以支撑评估反馈
- 哪些 topic 或 difficulty 虽然生成出来了，但质量不够

## 6.3 评估反馈

评估反馈是本设计的核心。

评估反馈回答的问题是：

“这轮最终入选题是否真的区分了模型能力？”

推荐字段：

```yaml
evaluator_feedback:
  summary:
    num_questions: int
    num_models: int
    llm_input_tokens: int
    llm_output_tokens: int
  dataset_signals:
    citation_pass_rate: float | null
    diversity_score: float | null
    embedding_dispersion: float | null
    cluster_entropy: float | null
  discriminative_signals:
    overall_model_gap: float | null
    best_vs_second_gap: float | null
    top_vs_bottom_gap: float | null
    per_question_variance_mean: float | null
    discriminative_question_ratio: float | null
    easy_questions_too_easy_ratio: float | null
    all_models_fail_ratio: float | null
    judge_disagreement_rate: float | null
  by_topic:
    topic_a:
      question_count: int
      model_gap: float | null
      variance_mean: float | null
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

如果没有这些聚合信号，planner 无法判断：

- 数据集是真的有区分度
- 还是只是恰好某些模型在噪声题上分数波动

当前实现中最值得新增的是：

- `overall_model_gap`
- `discriminative_question_ratio`
- `easy_questions_too_easy_ratio`
- `all_models_fail_ratio`
- `by_topic.model_gap`
- `by_difficulty.model_gap`

## 7. 下一轮计划如何修改

Planner 不应直接从数值映射到 patch，而应先做诊断分类。

推荐使用四类诊断：

## 7.1 高质量但低区分度

特征：

- `citation_pass_rate` 高
- `final_selection_rate` 不低
- `overall_model_gap` 小
- `easy_questions_too_easy_ratio` 高

结论：

- 数据集质量没有问题
- 但题目过于基础或过于同质

下一轮调整：

- 提高 `hard` 比例
- 减少 easy 配额
- 缩窄到更专门 topic
- 增加 multi-evidence / multi-hop 题比例
- 评估档位提升到 `full`

## 7.2 低质量且低区分度

特征：

- `citation_pass_rate` 低
- `llm_pass_rate_after_citation` 低
- `duplicate_rate` 高
- `overall_model_gap` 也低

结论：

- 题目本身不稳定
- 当前评估结果不可信

下一轮调整：

- 不优先提高难度
- 缩窄 topic
- 提高 `candidate_pool.target_multiplier`
- 收紧验证阈值
- 增强去重
- 优先把“可验证、可读、可筛选”的题做稳

## 7.3 高区分度但质量差

特征：

- `overall_model_gap` 高
- 但 `citation_pass_rate` 或 `avg_llm_overall_score_selected` 低

结论：

- 存在有潜力的方向
- 但当前题目质量不足，不能直接放大量

下一轮调整：

- 保留 topic 方向
- 收紧验证阈值
- 增强 dedup 和 selection strictness
- 降低每 topic count
- 保留难度方向但压制噪声

## 7.4 高区分度但分布失衡

特征：

- `overall_model_gap` 高
- 但 `topicwise_gap` 过度集中
- 或 `by_difficulty` 只有 hard 真正拉开模型

结论：

- 数据集局部有效，整体结构失衡

下一轮调整：

- 保留高价值 topic 的一部分
- 引入相邻 topic 扩展
- 加强 per-topic quota
- 在保持 hard 的同时补充 medium 层的区分题

## 8. 推荐调参规则

建议先用规则系统，不必第一阶段就依赖 planner LLM 直接调参。

```yaml
rules:
  - if:
      overall_model_gap < g1
      and citation_pass_rate >= c_good
      and final_selection_rate >= s_good
    then:
      increase_hard_ratio: +0.1
      decrease_easy_ratio: -0.1
      narrow_topics: true
      prefer_multi_evidence: true

  - if:
      easy_questions_too_easy_ratio > e1
    then:
      reduce_easy_questions: true
      shift_to_specialized_topics: true

  - if:
      all_models_fail_ratio > f1
    then:
      decrease_hard_ratio: -0.1
      reduce_extreme_multi_hop: true

  - if:
      duplicate_rate > d1
    then:
      increase_semantic_dedup: true
      reduce_per_topic_quota: true

  - if:
      citation_pass_rate < c1
    then:
      do_not_increase_difficulty: true
      narrow_topics: true
      increase_candidate_pool_multiplier: true

  - if:
      topicwise_gap_concentrated == true
    then:
      keep_best_topics: true
      expand_adjacent_topics: true
      enforce_topic_balance: true
```

规则系统的优点：

- 可解释
- 易于测试
- 便于后续替换成 LLM 诊断器

## 9. Planner 决策顺序

每轮固定按以下顺序：

1. 看评估反馈，判断是否达成区分度目标
2. 看验证反馈，判断这些区分度是否建立在合格题目上
3. 看生成反馈，判断下一轮是否有足够产能支撑新计划
4. 产出诊断标签
5. 基于诊断标签生成下一轮 `RoundPlan`

不建议的顺序：

- 先看生成多少题，再决定是否成功
- 先根据验证通过率调难度，再看评估是否有区分度

这会让 planner 优化成“更容易通过验证”，而不是“更能区分模型能力”。

## 10. 对现有 planner_agent 的改造建议

若基于当前 `agents/planner_agent` 实现演进，建议分三步：

### 10.1 从 `RoundSpec` 升级为显式 `RoundPlan`

保留：

- `blueprint`
- `qa_agent_patch`
- `verify_agent_patch`
- `model_eval_agent_patch`

新增：

- `objective`
- `diagnosis`
- `generation_plan`
- `validation_plan`
- `evaluation_plan`
- `expected_feedback_focus`

### 10.2 扩充 `EvaluatorFeedback`

当前 planner 若只拿到 report 路径和少量 dataset signal，不足以驱动下一轮。

应新增：

- 区分度摘要
- 按 topic / mode / difficulty 的 gap 统计
- 过易题比例
- 全失败题比例

### 10.3 将 planner 决策函数拆成两段

推荐函数边界：

1. `diagnose_last_round(gen_fb, val_fb, eval_fb) -> Diagnosis`
2. `build_next_round_plan(state, diagnosis, blueprint) -> RoundPlan`

这样比把所有逻辑堆进 `_build_next_round_spec()` 更稳定，也更容易测试。

## 11. 最小可落地版本

如果只做最小版本，建议先实现以下能力：

1. 评估反馈中新增：
   - `overall_model_gap`
   - `discriminative_question_ratio`
   - `easy_questions_too_easy_ratio`
   - `all_models_fail_ratio`
2. planner 新增诊断标签：
   - `high_quality_low_separation`
   - `low_quality_low_separation`
   - `high_separation_low_quality`
   - `high_separation_imbalanced_distribution`
3. planner 只调以下参数：
   - `topics`
   - `count`
   - `difficulty_distribution`
   - `candidate_pool.target_multiplier`
   - `llm_validation.min_overall_score`
   - `selection.mode`
   - `evaluation profile`

这是最小但完整的评估主导闭环。

## 12. 结论

对于“质量优先，最终目标是区分模型能力”的场景，planner 最合理的实现不是：

- 以生成数量为核心优化目标
- 或以验证通过率为核心优化目标

而应是：

- 以模型区分度为主目标
- 以验证质量为护栏
- 以生成产能为执行约束

因此 planner 的核心闭环应定义为：

`评估反馈 -> 诊断区分度问题 -> 生成下一轮计划 -> 验证反馈做质量约束 -> 再次评估`

如果后续实现与这个闭环一致，则 planner 的职责边界、参数控制面和反馈回路就是合理的。
