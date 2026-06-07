# 多智能体自动评价框架设计草案

本文基于 `plan.md` 的原始设想，整理当前更合理的框架修正、职责边界和数据字段建议。目标是为后续实现规划智能体、题目生成智能体、题目验证智能体、模型评估智能体和分析智能体提供稳定的数据契约。

## 1. 对当前方案的判断

`plan.md` 的总体方向是合理的：它把 `BenchAgents`、`AutoBencher`、`YourBench`、`EvalTree` 分别映射到多轮评估、主题扩展、文档出题和能力树分析，这个组合逻辑成立。

但当前描述仍然偏功能愿景，落实为工程系统时需要补齐三个核心问题：

1. 全局状态由谁维护。
2. 每个智能体之间传递什么数据。
3. 多轮反馈如何触发重新规划、继续执行或终止。

如果不先定义这三点，后续很容易出现状态冲突，例如题目生成智能体、验证智能体和分析智能体各自维护不同版本的能力标签、题目质量状态和下一轮主题方向。

## 2. 推荐架构选择

建议采用：

```text
强中心化规划智能体 + 子智能体局部有限自治
```

也就是：

- `规划智能体` 是唯一的全局真相源。
- `规划智能体` 独占维护和修改总蓝图。
- 其他智能体只接收裁剪后的任务单。
- 子智能体可以在任务单约束范围内选择局部策略。
- 子智能体不能直接修改总蓝图，只能返回结构化结果。
- 是否继续、回炉、扩展主题、进入评估或停止，都由规划智能体决定。

这个选择更适合当前方案，原因是系统目标不是单次题目生成，而是多轮自动评价闭环。题目生成、验证、模型评估和能力分析之间依赖很强，如果每个智能体都自主改状态，会导致全局策略漂移。

## 3. 原方案中的主要冲突与修正

### 3.1 规划智能体职责过宽但必须保留中心地位

原始描述中，规划智能体承担了蓝图生成、任务分解、反馈吸收和新计划生成。这是合理的，但需要明确它不是普通执行智能体，而是系统状态机和调度器。

修正后职责：

- 生成 `Blueprint`。
- 根据 `Blueprint` 生成各智能体的 `TaskSpec`。
- 接收各智能体的 `TaskResult`。
- 更新 `Blueprint`。
- 决定下一轮任务、重试、回炉、分析或停止。
- 维护全局预算、轮次、覆盖率、题库版本和决策日志。

### 3.2 题目生成智能体与分析智能体的能力标签边界需要拆开

原方案写到题目生成阶段参考 `EvalTree` 为题目添加能力描述属性，后期再构建能力树。这有潜在冲突。

问题在于，生成阶段的能力标签通常只是初步判断。如果直接当作最终标签，分析阶段构建的能力树会被生成模型的偏差污染。

修正后职责：

- 题目生成智能体产出候选能力标签、候选能力描述和证据。
- 题目验证智能体检查能力标签是否格式有效、是否与题目内容明显冲突。
- 分析智能体基于验证通过的题目、候选标签和评估结果，生成最终能力树和弱点树。

### 3.3 题目验证智能体需要成为质量闸门

原方案中题目验证智能体负责验证题目并反馈数量、难度等状态。这个方向正确，但需要更明确地把它定义为质量闸门。

修正后职责：

- 检查题目 schema 是否有效。
- 检查题目是否可回答。
- 检查答案是否唯一或评分规则是否明确。
- 检查引用和证据是否支持答案。
- 检查题目重复度。
- 检查语言、难度、题型、主题和能力分布是否满足当前任务要求。
- 输出通过、拒收、需修复、需重生的结构化结果。

### 3.4 模型评估智能体不应直接决定下一轮策略

原方案中模型评估智能体“反馈给规划智能，进行下一层主题筛选”。这里需要注意，评估智能体可以给建议，但不应直接改计划。

修正后职责：

- 运行模型评估。
- 输出题目级、主题级、语言级、题型级、能力级指标。
- 记录调用次数、token、延迟、成本、错误。
- 标记低置信度结果、无效题目或模型拒答。
- 给出下一轮建议，但最终是否采纳由规划智能体决定。

### 3.5 分析智能体负责后验结构化，不负责调度

分析智能体适合处理能力树、弱点树、覆盖缺口和评估报告。它不应该直接安排新任务。

修正后职责：

- 基于题目元数据和模型评估结果生成能力树。
- 生成模型弱点树。
- 识别覆盖不足的能力节点。
- 识别区分度低、噪声高或样本不足的区域。
- 向规划智能体提交结构化建议。

## 4. 核心数据对象

建议定义三个核心对象：

```text
Blueprint  -> 全局蓝图，由规划智能体维护
TaskSpec   -> 单次任务单，由规划智能体下发
TaskResult -> 任务回执，由子智能体返回
```

### 4.1 Blueprint

`Blueprint` 是全局真相源，保存用户目标、评估范围、数据策略、能力策略、验证策略、预算、轮次状态和共享产物。

建议字段：

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `blueprint_id` | `str` | 全局蓝图 ID |
| `version` | `int` | 蓝图版本，每次规划更新时递增 |
| `status` | `draft/running/paused/completed/failed` | 当前运行状态 |
| `user_goal` | `str` | 用户原始目标和最终产出要求 |
| `evaluation_scope` | `EvaluationScope` | 评估范围 |
| `dataset_policy` | `DatasetPolicy` | 题库规模、比例和采样策略 |
| `question_policy` | `QuestionPolicy` | 题目格式、证据和答案要求 |
| `capability_policy` | `CapabilityPolicy` | 能力标签和能力树策略 |
| `validation_policy` | `ValidationPolicy` | 题目质量闸门 |
| `evaluation_policy` | `EvaluationPolicy` | 模型评估方式和指标 |
| `analysis_policy` | `AnalysisPolicy` | 能力树、弱点树和报告生成策略 |
| `budget_policy` | `BudgetPolicy` | token、时间、成本、调用次数预算 |
| `iteration_policy` | `IterationPolicy` | 多轮迭代和停止条件 |
| `current_round` | `int` | 当前轮次 |
| `shared_artifacts` | `list[ArtifactRef]` | 共享产物引用 |
| `global_metrics` | `dict` | 当前全局统计 |
| `decision_log` | `list[DecisionRecord]` | 规划决策日志 |
| `next_actions` | `list[str]` | 规划智能体下一步动作 |
| `stop_conditions` | `list[str]` | 停止条件 |

### 4.2 EvaluationScope

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `domains` | `list[str]` | 评估主题或领域 |
| `languages` | `list[str]` | 目标语言 |
| `target_models` | `list[str]` | 被评估模型 |
| `source_policy` | `provided_docs/web_retrieval/hybrid` | 文档来源策略 |
| `allow_multihop` | `bool` | 是否允许多跳题 |
| `allow_cross_document` | `bool` | 是否允许跨文档题 |
| `external_retrieval_policy` | `dict` | 外部检索限制、白名单、最大页数等 |

### 4.3 DatasetPolicy

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `target_question_count` | `int` | 总目标题量 |
| `per_round_question_count` | `int` | 每轮新增题量 |
| `question_type_distribution` | `dict[str, float]` | 题型比例 |
| `difficulty_distribution` | `dict[str, float]` | 难度比例 |
| `language_distribution` | `dict[str, float]` | 语言比例 |
| `domain_distribution` | `dict[str, float]` | 主题比例 |
| `deduplication_required` | `bool` | 是否强制去重 |
| `min_valid_question_rate` | `float` | 每轮最低有效题比例 |

### 4.4 QuestionPolicy

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `allowed_question_types` | `list[str]` | 允许的题型 |
| `answer_format` | `str` | 答案格式 |
| `require_source_evidence` | `bool` | 是否要求证据 |
| `require_citation` | `bool` | 是否要求引用 |
| `allow_ambiguous_questions` | `bool` | 是否允许开放歧义 |
| `scoring_rubric_required` | `bool` | 开放题是否需要评分规则 |
| `max_question_length` | `int \| None` | 题干长度限制 |
| `max_answer_length` | `int \| None` | 答案长度限制 |

### 4.5 CapabilityPolicy

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `taxonomy_source` | `generated/evaltree_like/user_defined/hybrid` | 能力体系来源 |
| `allow_multi_label` | `bool` | 是否允许多能力标签 |
| `label_granularity` | `coarse/medium/fine` | 标签粒度 |
| `generator_labels_are_final` | `bool` | 生成阶段标签是否视为最终标签，建议默认 `false` |
| `min_questions_per_capability` | `int` | 每个能力节点最少题数 |
| `capability_schema` | `dict \| None` | 用户自定义能力 schema |

### 4.6 ValidationPolicy

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `require_schema_validity` | `bool` | 是否强制 schema 合法 |
| `require_answerable` | `bool` | 是否必须可回答 |
| `require_unique_answer` | `bool` | 是否要求唯一答案 |
| `require_source_evidence` | `bool` | 是否要求证据支持 |
| `deduplicate_threshold` | `float` | 判重阈值 |
| `allowed_auto_fixes` | `list[str]` | 允许自动修复的错误类型 |
| `reject_reasons` | `list[str]` | 拒收原因枚举 |
| `min_batch_pass_rate` | `float` | 单批最低通过率 |

### 4.7 EvaluationPolicy

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `metrics` | `list[str]` | 指标，例如 accuracy、exact_match、judge_score |
| `judge_model` | `str \| None` | 可选裁判模型 |
| `repeat_count` | `int` | 每题重复评估次数 |
| `temperature` | `float \| None` | 推理温度 |
| `record_token_usage` | `bool` | 是否记录 token |
| `record_latency` | `bool` | 是否记录延迟 |
| `record_cost` | `bool` | 是否记录成本 |
| `failure_handling` | `dict` | 超时、拒答、解析失败处理方式 |

### 4.8 AnalysisPolicy

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `build_capability_tree` | `bool` | 是否构建能力树 |
| `build_weakness_tree` | `bool` | 是否构建弱点树 |
| `min_samples_per_node` | `int` | 节点最小样本数 |
| `weakness_threshold` | `float \| None` | 弱点判定阈值 |
| `compare_models` | `bool` | 是否进行模型间对比 |
| `report_formats` | `list[str]` | 输出格式，例如 json、markdown、html |

### 4.9 BudgetPolicy

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `max_tokens` | `int \| None` | token 上限 |
| `max_cost_usd` | `float \| None` | 成本上限 |
| `max_wall_time_seconds` | `int \| None` | 总耗时上限 |
| `max_model_calls` | `int \| None` | 模型调用次数上限 |
| `per_task_timeout_seconds` | `int \| None` | 单任务超时 |

### 4.10 IterationPolicy

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `max_rounds` | `int` | 最大轮次 |
| `stop_when_target_count_reached` | `bool` | 达到目标题量后停止 |
| `stop_when_budget_exhausted` | `bool` | 预算耗尽后停止 |
| `min_new_information_gain` | `float \| None` | 低于新增收益阈值时停止 |
| `allow_topic_expansion` | `bool` | 是否允许扩展主题 |
| `allow_topic_pruning` | `bool` | 是否允许裁剪主题 |

## 5. TaskSpec 字段建议

`TaskSpec` 是规划智能体下发给某个子智能体的单次任务单。它不应该包含完整 `Blueprint`，只包含该任务需要的裁剪信息。

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `task_id` | `str` | 任务 ID |
| `blueprint_id` | `str` | 所属蓝图 ID |
| `blueprint_version` | `int` | 下发时的蓝图版本 |
| `round_id` | `int` | 所属轮次 |
| `agent_type` | `planner/question_generator/question_validator/model_evaluator/analyzer` | 目标智能体类型 |
| `objective` | `str` | 单次任务目标 |
| `inputs` | `dict` | 输入数据或产物引用 |
| `constraints` | `dict` | 本任务限制条件 |
| `acceptance_criteria` | `list[str]` | 验收标准 |
| `output_schema` | `dict` | 期望输出 schema |
| `budget` | `BudgetPolicy \| None` | 本任务预算 |
| `priority` | `low/normal/high/critical` | 优先级 |
| `depends_on` | `list[str]` | 依赖任务 |
| `retry_policy` | `dict` | 重试策略 |
| `return_requirements` | `list[str]` | 必须返回的统计、证据和错误信息 |

### 5.1 题目生成任务的 TaskSpec 应包含

| 字段区域 | 建议内容 |
| --- | --- |
| `inputs` | 主题、文档引用、上一轮覆盖缺口、已有题目摘要、候选能力体系 |
| `constraints` | 题量、语言、题型、难度、能力标签、是否多跳、是否跨文档 |
| `acceptance_criteria` | 输出题目数达到要求、每题有答案和证据、候选能力标签格式有效 |
| `return_requirements` | 题目 JSON、来源证据、生成失败原因、token 和调用统计 |

### 5.2 题目验证任务的 TaskSpec 应包含

| 字段区域 | 建议内容 |
| --- | --- |
| `inputs` | 待验证题目批次、验证策略、已有题库摘要 |
| `constraints` | 判重阈值、最低通过率、允许自动修复范围 |
| `acceptance_criteria` | 每题给出通过、拒收、需修复或需重生状态 |
| `return_requirements` | 通过题目集、拒收题目集、修复建议、批次质量指标 |

### 5.3 模型评估任务的 TaskSpec 应包含

| 字段区域 | 建议内容 |
| --- | --- |
| `inputs` | 验证通过题目集、目标模型、评分规则、评估配置 |
| `constraints` | 重复次数、超时、温度、最大并发、失败处理 |
| `acceptance_criteria` | 题目级结果完整、失败项有原因、指标可聚合 |
| `return_requirements` | 题目级分数、主题级指标、能力级指标、成本和延迟统计 |

### 5.4 分析任务的 TaskSpec 应包含

| 字段区域 | 建议内容 |
| --- | --- |
| `inputs` | 题目元数据、能力标签、模型评估结果、历史轮次指标 |
| `constraints` | 最小节点样本量、弱点阈值、报告格式 |
| `acceptance_criteria` | 输出能力树、弱点树、覆盖缺口和下一轮建议 |
| `return_requirements` | 结构化树、报告、低置信度节点、建议补题区域 |

## 6. TaskResult 字段建议

`TaskResult` 是子智能体向规划智能体返回的结构化回执。它是规划智能体更新 `Blueprint` 的唯一输入来源。

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `task_id` | `str` | 对应任务 ID |
| `blueprint_id` | `str` | 所属蓝图 ID |
| `blueprint_version` | `int` | 执行时依据的蓝图版本 |
| `round_id` | `int` | 所属轮次 |
| `agent_type` | `str` | 返回结果的智能体类型 |
| `status` | `pending/running/succeeded/failed/partial` | 任务状态 |
| `summary` | `str` | 简短结果摘要 |
| `produced_artifacts` | `list[ArtifactRef]` | 产物引用 |
| `metrics` | `dict` | 指标统计 |
| `issues` | `list[TaskIssue]` | 问题列表 |
| `recommendations` | `list[str]` | 下一步建议 |
| `evidence` | `list[dict]` | 关键证据 |
| `needs_replan` | `bool` | 是否建议重新规划 |
| `suggested_next_tasks` | `list[TaskSpec]` | 可选的下一步任务建议 |

### 6.1 TaskIssue 字段

| 字段 | 类型建议 | 说明 |
| --- | --- | --- |
| `severity` | `info/warning/error/critical` | 严重级别 |
| `code` | `str` | 机器可读错误码 |
| `message` | `str` | 人类可读说明 |
| `affected_items` | `list[str]` | 受影响题目、文档或任务 ID |
| `suggested_fix` | `str \| None` | 可选修复建议 |

## 7. 建议的最小执行流

```text
User Goal
  -> Planner creates Blueprint
  -> Planner emits TaskSpec(question_generator)
  -> QuestionGenerator returns TaskResult
  -> Planner updates Blueprint
  -> Planner emits TaskSpec(question_validator)
  -> Validator returns TaskResult
  -> Planner decides regenerate, continue, evaluate, analyze, or stop
  -> Planner emits TaskSpec(model_evaluator)
  -> ModelEvaluator returns TaskResult
  -> Planner emits TaskSpec(analyzer)
  -> Analyzer returns TaskResult
  -> Planner updates Blueprint and chooses next round or final report
```

## 8. 需要优先落地的工程约束

1. 先实现 `Blueprint`、`TaskSpec`、`TaskResult` 的 Pydantic schema。
2. 每个智能体都只通过 schema 接收输入和返回结果。
3. 规划智能体是唯一能更新 `Blueprint` 的组件。
4. 每次更新 `Blueprint` 都写入 `decision_log`。
5. 每个任务必须有 `acceptance_criteria` 和 `output_schema`。
6. 每个 `TaskResult` 必须包含 `status`、`metrics`、`issues` 和 `needs_replan`。
7. 能力标签在生成阶段默认只是候选标签，最终能力归因由分析阶段完成。

## 9. 后续实现顺序建议

推荐按以下顺序开发：

1. 定义 `schemas/blueprint.py`、`schemas/task.py`、`schemas/artifact.py`。
2. 实现规划智能体的蓝图创建和任务下发逻辑。
3. 实现题目生成智能体的最小闭环，先支持本地文档输入。
4. 实现题目验证智能体，作为题库质量闸门。
5. 实现模型评估智能体，先支持单模型、单轮评估。
6. 实现分析智能体，先生成能力覆盖和弱点摘要，再扩展到树结构。
7. 最后加入多轮策略、主题扩展、预算控制和报告生成。
