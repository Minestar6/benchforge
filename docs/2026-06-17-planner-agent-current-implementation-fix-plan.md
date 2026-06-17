# Planner Agent 基于当前实现的修复计划

## 1. 文档目标

本文档不是重新设计一套全新的 planner 架构，而是基于当前仓库已经存在的实现，给出一份面向目标的修复计划。

目标是让当前 `agents/planner_agent` 真正实现以下能力：

1. 将 `evaluation_report` 作为模型评估智能体的正式反馈真源
2. 根据主题评估结果指导下一轮主题生成
3. 将题目数量、题型、难度分配交给独立工具来决定
4. 保证首轮由用户目标生成的评估指标在后续轮次中不漂移
5. 保持 planner 的控制逻辑主要由工具驱动，而不是完全依赖 LLM

本文档严格针对当前代码现状，不讨论 `adaptive_qa_agent`。

## 2. 当前实现现状总结

当前 planner 已经具备以下基础：

- 有 `GlobalBlueprint`
- 有 `PlannerState`
- 有 `RoundDiagnosis`
- 有 `NextRoundControlPlan`
- 有基于 `evaluation_report` 构建的 `EvaluatorFeedback`
- 有 topic 工具链雏形：
  - `summarize_topic_feedback`
  - `expand_topic_candidates`
  - `rank_topics`

这说明当前系统已经从“纯产能规划”进化到“部分评估驱动规划”。

但仍有几类核心问题没有收口。

## 3. 需要修复的核心问题

## 3.1 评估反馈真源已经明确，但消费边界还不够清晰

当前 `evaluation_report.json` 已经成为最核心的评估产物，并且已经包含：

- `discriminative_signals`
- `by_topic`
- `by_difficulty`
- `by_mode`

这一步是正确的。

但 planner 内部当前仍然通过 `EvaluatorFeedback.derived_performance_signals` 再包一层，虽然不是错误，但后续要进一步明确：

- `evaluation_report.json` 是评估真源
- `EvaluatorFeedback` 继续保留，但它只是 `evaluation_report` 的抽取/适配视图
- `build_evaluator_feedback()` 只是 adapter，不是另一套协议

修复目标：

- 明确 `evaluation_report` 为唯一评估真源
- 明确 `EvaluatorFeedback` 的定位：
  - 它不是第二套评估协议
  - 它不是独立真源
  - 它只是 planner 内部消费 `evaluation_report` 的轻量视图
- `EvaluatorFeedback` 只保留 planner 真正需要的轻量抽取字段

## 3.2 topic 规划已经工具化，但边界仍需通过约束和测试保持干净

当前已经存在：

- `summarize_topic_feedback`
- `expand_topic_candidates`
- `rank_topics`

这是正确方向。

当前实现已经基本走上正确方向，但仍有两个需要通过文档约束和单测防回归的点：

1. topic 数量预算来自上游控制计划，后续不应重新引入二次裁剪
2. topic 工具链虽然存在，但其输出和 planner 后续逻辑之间仍需通过单一真源约束保持稳定

修复目标：

- 目标状态下，`adjust_next_round_plan()` 这一类上游控制器只负责给出 `topic_budget`
- `rank_topics()` 只在预算内选最终 topics
- `_build_next_round_spec()` 继续保持不做额外裁剪或重解释

## 3.3 数量、题型、难度分配还没有独立成单独工具

当前 `adjust_next_round_plan()` 既在做诊断后的数量控制，又在顺带调整 topic budget、验证阈值、评估档位。

虽然已经比旧版好很多，但还没有达到你想要的结构：

- 主题生成由主题评估驱动
- 数量/题型/难度分配由另一个工具负责

修复目标：

采用两阶段收口：

### 近期最小拆分

先将当前 `adjust_next_round_plan()` 拆成：

1. `question_difficulty_evolver`
2. runtime 控制逻辑

### 后续完整拆分

再把 topic 选择从主流程中完全收口为：

1. `topic_adaptive_retriever`
2. `question_difficulty_evolver`

两者分别负责不同决策面。

## 3.4 首轮评估指标冻结机制仍需防回归收口

当前代码已经明显好于早期版本。这里更准确的说法不是“当前仍在漂移”，而是“需要防止后续改动重新引入漂移”。历史上最容易出问题的两条路径是：

1. `translate_eval_profile()` 会根据 `GlobalBlueprint.evaluation_requirements` 注入 metrics
2. `_build_next_round_spec()` 一度承担过组装之外的评估配置职责

当前更准确的风险表述应是：

- 必须防止后续演进时重新引入“后续轮次改写评估定义”的行为

修复目标：

- 将 `GlobalBlueprint.evaluation_requirements` 明确为冻结真源
- 所有后续轮次仅引用它，不再重建 metrics
- 删除任何“name -> description=name”的重写逻辑

## 3.5 当前 feedback schema 信息偏多，但策略消费偏少

从当前实现看：

- 生成反馈较多报表字段，真正被 planner 用到的很少
- 验证反馈最丰富，但当前消费不充分
- 评估反馈最接近主目标，应该继续作为核心反馈来源

修复目标：

- 明确“当前策略必需字段”
- 将大量路径引用和调试字段降为 debug-only 或 artifact-only
- 保留未来会用到的 `by_difficulty / by_mode`，但不把它们强塞进当前最小控制闭环

## 4. 当前阶段目标架构：4 个核心规划工具，先保证闭环

当前阶段不追求把所有规划能力都拆成独立工具，而是优先落地最小闭环。

本阶段只正式实现 4 个核心规划工具：

1. `user_goal_analyzer`
2. `topic_adaptive_retriever`
3. `question_difficulty_evolver`
4. `control_parameter_tuner`

其余能力先降级处理：

- `evaluation_report_adapter` 先作为轻量 adapter / helper 保留，不作为重工具推进
- `blueprint_decomposer` 先不做正式工具，只保留轻量阶段判断
- `round_spec_builder` 作为落地辅助器保留，不作为核心规划工具并列讨论

## 4.1 user_goal_analyzer

### 作用

分析用户输入，生成首轮 topic、首轮评估配置与首轮生成策略。

### 输入

- `user_goal`
- 自动指标枚举
- 题目策略枚举

### 输出

- `initial_topics`
- `initial_generation_strategy`
- `automatic_metrics_by_type`
- `llm_eval_enabled`
- `qa_llm_eval_metrics`

### LLM 参与

需要。

### 负责内容

- 初始 topic 列表
- 首轮生成题目的策略标签
- 题目类型对应的自动指标
- 是否开启 LLM 评估
- QA 侧的大模型评估指标

## 4.2 blueprint_decomposer

### 作用

把总体蓝图分解成每一轮的阶段性角色和粗粒度策略提示。

### 当前阶段状态

当前不作为正式独立工具实现，只保留为轻量阶段判断逻辑。

### 输入

- `GlobalBlueprint`
- `PlannerState`
- 剩余轮数
- 当前累计完成量
- token 预算

### 输出

- `RoundStrategy`

### 负责内容

- 当前轮是 `explore / exploit / repair / finalize`
- 当前轮的大致策略提示
- 当前轮是否偏探索、偏收敛、偏修复
- 当前轮可以给出非常粗的策略提示，例如更保守或更扩张，但不直接给预算值

### 不负责内容

- 不直接产出最终 `qa_count`
- 不直接产出最终 `mc_count`
- 不直接产出最终难度分布
- 不直接产出最终 `topic_budget`

也就是说，`blueprint_decomposer` 不是数值控制器，它只负责“轮次阶段策略”。

### LLM 参与

可选，但建议工具主导。

## 4.3 evaluation_report_adapter

### 作用

将 `evaluation_report.json` 转成 planner 可读的轻量结构。

### 当前阶段状态

当前保留为轻量 adapter / helper，不作为正式核心工具推进。

### 输入

- `evaluation_report.json`

### 输出

- `EvaluatorFeedback`

### 原则

- `evaluation_report` 是真源
- adapter 不定义第二套独立评估协议

## 4.4 topic_adaptive_retriever

### 作用

根据模型评估中的主题总结和检索到的相关主题，生成下一轮目标主题列表。

### 输入

- 主题评估总结
- 检索到的相关主题
- 用户目标

### 输出

- `target_topics`

### 内部子工具

- `topic_feedback_summarizer`
- `topic_expander`
- `topic_ranker`

### LLM 参与

需要，用于 topic 扩展。

### 负责内容

- 根据主题评估结果保留、替换、扩展主题
- 结合检索到的相关主题筛出更优主题

### 不负责内容

- 不决定 `qa_count`
- 不决定 `mc_count`
- 不决定难度分布
- 不决定验证阈值
- 不决定候选池倍率

也就是说，`topic_adaptive_retriever` 是唯一主题选择器，但不是数值控制器。

## 4.5 question_difficulty_evolver

### 作用

根据上一轮题目计划、题目生成反馈、题目验证反馈，决定下一轮题目计划：

- 题目类型分配
- 题目数目
- 难度比例
- `topic_budget`
- 候选池倍率
- 计划中的最小/最大指标

### 输入

- 上一轮题目计划
- `GeneratorFeedback`
- `ValidatorFeedback`

### 输出

- `next_round_question_plan`

### LLM 参与

不建议。

### 原则

- 这是数值控制工具，不是语义工具
- 主要由规则和阈值系统驱动

### 职责边界

`question_difficulty_evolver` 是唯一题目计划演化器。

这里的“题目计划演化”特指：题量、题型、难度分布、`topic_budget`、候选池倍率、计划中的最小/最大指标等题目规划配额。
与之对应，其他运行控制参数应由 `control_parameter_tuner` 负责，而不是继续留在题目计划演化器中。

它负责：

- `qa_count`
- `mc_count`
- QA / MC 难度分布
- `topic_budget`
- `candidate_pool_multiplier`
- 计划中的最小/最大指标

因此：

- `blueprint_decomposer` 不再产出最终 count / distribution
- `topic_adaptive_retriever` 不再决定数量
- `round_spec_builder` 不再做任何数值推理

## 4.6 control_parameter_tuner

### 作用

根据相关反馈自动调整其他计划中的控制指标。

### 输入

- 上一轮题目计划
- `GeneratorFeedback`
- `ValidatorFeedback`
- `EvaluatorFeedback`

### 输出

- `next_round_control_plan`

### LLM 参与

不建议。

### 原则

- 这是运行控制参数调优工具，不是目标函数定义工具
- 优先由规则和阈值系统驱动
- 允许自动调整控制参数，但不允许自动漂移评估指标定义

### 负责内容

- `citation_validation.enabled`
- `citation_validation.min_citation_score`
- `citation_validation.min_chunk_citation_score`
- `citation_validation.min_answer_citation_score`
- `llm_validation.enabled`
- `llm_validation.min_overall_score`
- `selection.mode`
- `semantic_similarity_threshold`
- `evaluation.profile`
- `judge.enabled`

### 不负责内容

- 不改变 `llm_judge_metrics`
- 不改变 `automatic_metrics`
- 不改变 metric descriptions
- 不决定 `qa_count`
- 不决定 `mc_count`
- 不决定 topic 选择

也就是说，它只调“运行控制参数”，不改“目标函数定义”。

## 4.7 round_spec_builder

### 作用

把 4 个规划工具的输出合并成最终 `RoundSpec`。

### 输入

- `target_topics`
- `next_round_question_plan`
- `next_round_control_plan`
- 首轮或冻结评估配置
- `GlobalBlueprint`
- `PlannerState`

### 输出

- `RoundSpec`

### 原则

- 只做结构翻译
- 不做二次推理
- 不做隐式裁剪

### 对 `PlannerState` 的使用约束

`round_spec_builder` 可以接收 `PlannerState`，但只能用于：

- 读取必要标识信息
- 补全 `task_id / run_id / round_id`
- 拼装 blueprint 所需静态上下文

不允许：

- 基于 state 再次裁剪 topics
- 基于 state 再次推导 count
- 基于 state 再次推导难度分布
- 基于 state 再次改变评估指标定义

否则它会从“组装器”重新退化为“隐藏控制器”。

## 5. 当前实现下，每个智能体真正需要的反馈

本节的目标不是删除当前 schema 中的所有扩展字段，而是明确：

- 当前 planner 真正依赖哪些字段形成控制闭环
- 哪些字段只是调试、追溯或未来增强用

后续修复应优先保证“必须字段”稳定，再决定是否裁剪扩展字段。

## 5.1 生成智能体反馈

### 当前策略最小必需字段

- `task_id`
- `run_id`
- `round_id`
- `status`
- `summary.total_candidates`
- `summary.global_failures`
- `by_mode.fulfillment_rate`
- `topic_coverage[topic].candidate_count`

### 当前可保留但非策略必需字段

- `artifacts.*`
- `llm tokens`
- `global_used_chunk_combinations`
- `by_mode.topic_counts`
- `by_mode.difficulty_counts`

### 修复建议

- 保留完整字段供分析和调试
- 但在 planner 内部再抽一层 `GenerationControlSignals`

### 这些字段为什么必须

- `task_id/run_id/round_id/status`
  - 保证统一反馈接口可追踪
- `summary.total_candidates`
  - 判断本轮是否完全没有有效生成
- `summary.global_failures`
  - 判断是否进入 `generation_capacity_insufficient`
- `by_mode.fulfillment_rate`
  - 判断本轮题量计划是否没产出来
- `topic_coverage.candidate_count`
  - 识别低产能 topic，支撑 `low_yield_topics`

## 5.2 验证智能体反馈

### 当前策略最小必需字段

- `task_id`
- `run_id`
- `round_id`
- `status`
- `summary.citation_pass_rate`
- `summary.final_selected`
- `quality_signals.avg_llm_overall_score_selected`
- `quality_signals.duplicate_rate`
- `by_topic.selected/rejected/avg_citation_score/avg_llm_overall_score`
- `by_mode.selected`

### 当前建议保留的扩展字段

- `by_difficulty`
- `selection_summary`

### 修复建议

- 当前不要删 `by_difficulty`
- 后续让 `question_difficulty_evolver` 真正消费它

### 这些字段为什么必须

- `task_id/run_id/round_id/status`
  - 保证统一反馈接口可追踪
- `summary.citation_pass_rate`
  - 判断整体质量护栏是否过关
- `summary.final_selected`
  - 判断本轮是否有足够的最终有效题目
- `quality_signals.avg_llm_overall_score_selected`
  - 判断是否出现“高区分但低质量”
- `quality_signals.duplicate_rate`
  - 判断是否需要收紧去重或减少 topic 扩张
- `by_topic.selected/rejected/avg_*`
  - 支撑 topic 噪声判断
- `by_mode.selected`
  - 用于更新 `completed_targets`

## 5.3 模型评估智能体反馈

### 当前策略最小必需字段

- `task_id`
- `run_id`
- `round_id`
- `status`
- `summary.num_questions`
- `summary.num_models`
- `derived_performance_signals.overall_model_gap`
- `derived_performance_signals.easy_questions_too_easy_ratio`
- `derived_performance_signals.all_models_fail_ratio`
- `derived_performance_signals.by_topic.<topic>.model_gap`
- `derived_performance_signals.by_topic.<topic>.too_easy_ratio`
- `derived_performance_signals.by_topic.<topic>.all_models_fail_ratio`

### 下一阶段闭环预留字段

- `derived_performance_signals.by_difficulty`
- `derived_performance_signals.by_mode`
- `derived_performance_signals.discriminative_question_ratio`
- `derived_performance_signals.best_vs_second_gap`
- `derived_performance_signals.per_question_variance_mean`

### 修复建议

- `evaluation_report` 继续作为核心真源
- `EvaluatorFeedback` 继续保留，但只作为 `evaluation_report` 的抽取视图
- 将 `by_difficulty` 和 `by_mode` 后续接入 `question_difficulty_evolver`

### 这些字段为什么必须

- `task_id/run_id/round_id/status`
  - 保证统一反馈接口可追踪
- `summary.num_questions/num_models`
  - 判断本轮评估样本是否足够支撑诊断
- `derived_performance_signals.overall_model_gap`
  - planner 当前主目标信号
- `derived_performance_signals.easy_questions_too_easy_ratio`
  - 判断低区分度是否因为题太简单
- `derived_performance_signals.all_models_fail_ratio`
  - 判断低区分度是否因为题太难或题目噪声过大
- `derived_performance_signals.by_topic.<topic>.model_gap`
  - 指导下一轮主题生成
- `derived_performance_signals.by_topic.<topic>.too_easy_ratio`
  - 识别过饱和、应下调的 topic
- `derived_performance_signals.by_topic.<topic>.all_models_fail_ratio`
  - 识别应回退或替换的 topic
- `derived_performance_signals.by_difficulty.*.model_gap`
  - 当前建议保留，后续支撑下一轮难度分配工具
- `derived_performance_signals.by_mode.*.model_gap`
  - 当前建议保留，后续支撑 QA / MC 配比调整

### `EvaluatorFeedback` 的保留原则

保留 `EvaluatorFeedback`，但需要明确其职责边界：

- 真源：`evaluation_report.json`
- 视图：`EvaluatorFeedback`
- 适配函数：`build_evaluator_feedback()`

也就是说：

- `evaluation_report` 负责完整表达评估结果
- `EvaluatorFeedback` 负责抽取 planner 当前控制必需的字段
- planner 后续新增字段时，应优先从 `evaluation_report` 增加，再决定是否纳入 `EvaluatorFeedback`

不建议让 `EvaluatorFeedback` 重新承担以下职责：

- 重新定义评估指标
- 重新定义评估 schema
- 与 `evaluation_report` 形成双真源

## 6. 需要修改的实现边界

## 6.1 用 `evaluation_report` 作为正式评估反馈真源

### 当前状态

已经基本成立。

### 修复动作

1. 在文档和代码注释中明确：
   - `evaluation_report.json` 是评估真源
2. `EvaluatorFeedback` 保留，但明确为抽取视图
3. `build_evaluator_feedback()` 只做：
   - 字段抽取
   - token 聚合
   - artifact 路径引用
4. 不再在 planner 层重新定义评估指标结构

### 额外边界

后续如果新增：

- `GenerationControlSignals`
- `next_round_question_plan`
- `target_topics`
- `next_round_control_plan`

需要明确它们都只是 planner 内部控制视图：

- 不替代原始 feedback schema
- 不替代 `evaluation_report.json`
- 不形成第三套对外协议

## 6.2 统一 topic 级失败信号命名

### 当前问题

- 全局是 `all_models_fail_ratio`
- topic 级有 `all_fail_ratio`

### 修复动作

统一成：

- 全局：`all_models_fail_ratio`
- topic 级：`all_models_fail_ratio`
- difficulty 级：`all_models_fail_ratio`
- mode 级：`all_models_fail_ratio`

### 迁移策略

短期兼容读取：

- 先读 `all_models_fail_ratio`
- 缺失时 fallback `all_fail_ratio`

长期：

- 所有产物只写统一命名

## 6.3 主题工具链收口

### 当前问题

topic 工具链已经存在，但预算和裁剪逻辑尚未完全收口。

### 修复动作

1. `question_difficulty_evolver` 作为唯一数值控制器决定 `topic_budget`
2. `topic_ranker` 在预算内输出最终 topics
3. `round_spec_builder` 不再做二次裁剪

### 明确边界

- `topic_adaptive_retriever` 决定“选哪些主题”
- `question_difficulty_evolver` 决定“这一轮允许几个主题”

不允许：

- `topic_adaptive_retriever` 自行再推导预算
- `round_spec_builder` 再按 topic 数量做二次裁剪

## 6.4 将数量、题型、难度分配独立成工具

### 当前问题

当前 `adjust_next_round_plan()` 同时处理：

- 难度分配
- 数量
- topic budget
- 验证阈值
- 评估档位

### 修复动作

按两阶段拆分：

### 近期最小拆分

1. `question_difficulty_evolver`
2. 从 `adjust_next_round_plan()` 中拆出 runtime 控制逻辑

### 后续完整拆分

1. `question_difficulty_evolver`
2. `control_parameter_tuner`
3. `topic_adaptive_retriever`

### 建议输入

- 上一轮计划
- `GeneratorFeedback`
- `ValidatorFeedback`
- `EvaluatorFeedback`
- 当前剩余目标缺口

### 建议输出

- `qa_count`
- `mc_count`
- `qa_difficulty_distribution`
- `mc_difficulty_distribution`
- `topic_budget`
- `candidate_pool_multiplier`

### 额外说明

若实现上希望进一步避免 `question_difficulty_evolver` 变胖，可以直接将：

- `validation_runtime`
- `evaluation_runtime`
- 以及其他运行控制参数

统一下沉到独立的 `control_parameter_tuner`，而不继续挂在 `next_round_question_plan` 上。

### 当前建议

按当前代码现实，最值得先做的不是把全部 planner 一次性拆完，而是先把 `adjust_next_round_plan()` 至少拆成两部分：

1. `question_difficulty_evolver`
2. runtime 控制逻辑

这样可以先把“题目计划演化”和“运行时护栏”分开。这里的 runtime 控制逻辑是阶段性描述，最终正式工具名统一收敛为 `control_parameter_tuner`。后续再把 topic 选择从主流程里完全收口为 `topic_adaptive_retriever`，并把 runtime 强度以及其他控制参数正式下沉到 `control_parameter_tuner`。

## 6.5 落实首轮评估指标冻结机制

### 当前问题

当前代码已经基本以 `GlobalBlueprint.evaluation_requirements` 作为运行时真源。这里更准确的问题不是“当前仍在漂移”，而是“需要通过约束和单测防止后续回归重新引入漂移”。

### 需要重点防回归的两处

1. `translate_eval_profile()`
2. `_build_next_round_spec()`

### 修复原则

- `GlobalBlueprint.evaluation_requirements` 是唯一冻结真源
- 后续轮次只允许：
  - 切 `profile`
  - 开关 judge
  - 切 candidate models
- 不允许：
  - 重写 metric 列表
  - 重写 metric description
  - 用 name 退化 description

### 最小实现动作

1. planner 整体后续轮次只能引用 `GlobalBlueprint.evaluation_requirements` 这一运行时真源，不重新生成
2. `round_spec_builder` / `_build_next_round_spec()` 继续保持不承担 metric-definition 职责
3. 如后续确认存在 blueprint 运行时被改写的风险，再考虑补：
   - `frozen_evaluation_requirements`
   - 或 `evaluation_requirements_hash`

当前阶段不强制要求先上 hash 机制。

## 7. 分阶段修复路线

## 第一阶段：收口真源和协议

目标：

- 让评估反馈真源和冻结机制先稳定

动作：

1. 明确 `evaluation_report` 为真源
2. 统一区分度字段命名
3. 用单测防止 metric definition / description 退化路径回归
4. 明确后续轮次不能改写评估定义

## 第二阶段：补充剩余辅助能力

目标：

- 在 4 个核心工具闭环稳定后，再补齐辅助能力

动作：

1. 保留 `evaluation_report_adapter` 为轻量 helper
2. 将 runtime 控制逻辑正式收敛为 `control_parameter_tuner`
3. 如确有必要，再补 `blueprint_decomposer`
4. 继续保持 `topic_adaptive_retriever` / `question_difficulty_evolver` / `control_parameter_tuner` / `round_spec_builder` 的边界稳定

## 第三阶段：完善 topic 自适应搜索

目标：

- 让 topic 规划真正依赖主题评估分数和主题轨迹

动作：

1. 让 `topic_feedback_summarizer` 以 `evaluation_report` 为主输入
2. 保留 `ValidatorFeedback` 和 `GeneratorFeedback` 作为约束输入
3. 强化 `topic_expander`：
   - 强主题扩邻近子主题
   - 弱主题找替代
   - 高失败主题回退
4. 强化 `topic_ranker`：
   - 历史复用惩罚
   - 预算内排序
   - 不再二次裁剪

## 第四阶段：测试不变量

应优先补的单测不是全量功能测试，而是三个关键不变量：

1. `metrics` 冻结
   - 后续轮次不能改写评估指标定义
2. `topic_budget` 单一真源
   - 只能由 `question_difficulty_evolver` 决定
3. `round_spec_builder` 不做裁剪
   - 只组装，不再改变上游 planner 已决定的结果

## 第五阶段：提升数量与难度控制质量

目标：

- 让难度和数量计划真正成为闭环控制器

动作：

1. 读取上一轮“计划值 vs 实际值”
2. 读取生成阶段难度产能偏差
3. 读取验证阶段不同难度保留率
4. 读取评估阶段不同难度区分度
5. 输出下一轮题量与难度分配

## 8. 给当前实现的最小修复清单

如果只做当前目标所需的最小修复，建议按以下顺序：

1. 实现 `user_goal_analyzer`
2. 实现 `topic_adaptive_retriever`
3. 实现 `question_difficulty_evolver`
4. 实现 `control_parameter_tuner`
5. 将 `_build_next_round_spec()` 收口为 `round_spec_builder`
6. 保留 `build_evaluator_feedback()`，但明确 `derived_performance_signals` 只是 `evaluation_report` 的 adapter 视图
7. 统一 `all_models_fail_ratio` 命名
8. 用单测防止评估指标定义与 description 退化路径回归
9. 补单测，重点覆盖：
   - metrics 冻结
   - topic_budget 单一真源
   - round_spec_builder 不做裁剪

## 9. 结论

基于当前实现，planner 已经具备一个较好的骨架：

- 有评估驱动的雏形
- 有 topic 工具链雏形
- 有诊断与控制计划的雏形

接下来真正要做的，不是再大幅重构一套新系统，而是：

- 收口评估真源
- 收口冻结机制
- 收口 topic 预算边界
- 将题量/类型/难度分配正式独立成工具

当前阶段的目标架构应是：

1. `user_goal_analyzer`：用户描述 + 指标/策略枚举 -> 初始 topics、初始生成策略、自动指标配置、LLM 评估开关与 QA 大模型评估指标
2. `topic_adaptive_retriever`：主题评估总结 + 检索到的相关主题 -> 目标主题列表
3. `question_difficulty_evolver`：上一轮题目计划 + 生成/验证反馈 -> 新一轮题目计划
4. `control_parameter_tuner`：相关反馈 -> 自动调控其他计划中的控制指标

辅助但暂不独立推进的部分：

- `evaluation_report_adapter`：保留为轻量 helper
- `blueprint_decomposer`：先不独立实现
- `round_spec_builder`：作为落地辅助器保留，不作为核心规划工具并列

如果按此计划推进，当前实现可以平滑收敛到你想要的目标，而不需要推倒重来。
