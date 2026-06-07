# 规划智能体 TaskSpec 字段定义

规划智能体向各子智能体下发 `TaskSpec`，只包含**战略性决策**（做什么、做多少、质量标准）。
**战术细节**（怎么做、用什么参数）留在各智能体自身 YAML 中。

`TaskSpec` 通用结构：

| 字段 | 类型 | 说明 |
|---|---|---|
| `task_id` | `str` | 任务 ID |
| `blueprint_id` | `str` | 所属蓝图 ID |
| `blueprint_version` | `int` | 下发时的蓝图版本 |
| `round_id` | `int` | 所属轮次 |
| `agent_type` | `str` | 目标智能体类型 |
| `objective` | `str` | 单次任务目标（人类可读） |
| `inputs` | `dict` | 输入数据引用 |
| `constraints` | `dict` | 限制条件 |
| `acceptance_criteria` | `list[str]` | 验收标准 |
| `output_schema` | `dict` | 期望输出 schema |
| `budget` | `BudgetPolicy \| None` | 本任务预算 |
| `return_requirements` | `list[str]` | 必须返回的信息 |

---

## 1. 题目生成智能体（GeneratorTaskSpec）

规划智能体控制**生成什么题目**，生成智能体自己决定**怎么生成**。

### inputs

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `topics` | `list[str]` | ✓ | 本轮要覆盖的主题，来自 Blueprint 或上轮覆盖缺口 |
| `language` | `str` | ✓ | 目标语言，如 `"en"` |
| `source_refs` | `list[str]` | | 文档路径或 URL 列表。空则使用 Wikipedia 检索 |
| `previous_gaps` | `dict[str, dict]` | | 上一轮覆盖缺口 `{topic: {mode: {difficulty: gap_count}}}`，用于自适应补题 |
| `existing_questions_summary` | `list[str]` | | 已有题目的摘要（避免重复生成），如题目文本哈希或关键词列表 |
| `capability_taxonomy` | `dict \| None` | | 候选能力体系 schema，用于为题目标注候选能力标签 |

### constraints

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `mode_targets` | `dict[str, ModeTarget]` | ✓ | 每种题型的目标，`ModeTarget = {count: int, difficulty_distribution: {easy: 0.3, medium: 0.4, hard: 0.3}}` |
| `allowed_question_types` | `list[str]` | | 允许的题型枚举。空则允许全部 |
| `require_citation` | `bool` | | 是否要求每题附引用证据，默认 `true` |
| `require_capability_label` | `bool` | | 是否要求每题标注候选能力，默认 `true` |
| `allow_multihop` | `bool` | | 是否允许多跳推理题，默认 `false` |
| `allow_cross_document` | `bool` | | 是否允许跨文档题，默认 `false` |
| `max_question_length` | `int \| None` | | 题干最大长度（字符） |
| `max_answer_length` | `int \| None` | | 答案最大长度（字符） |

### acceptance_criteria

```yaml
- "输出题目数 ≥ min(mode_targets 各题型 count 之和 × 0.8, 1)"
- "每题包含 question、answer、citations、required_capability 字段"
- "候选能力标签格式有效（非空字符串）"
- "每题 estimated_difficulty 在 {easy, medium, hard} 范围内"
```

### return_requirements

```yaml
- "生成的题目列表（JSON，每题为 QuestionRecord schema）"
- "每题对应的来源证据（chunk_ids 和 chunk 文本）"
- "生成失败原因（如有），按 topic 分组"
- "token 用量统计（input_tokens, output_tokens）"
- "模型调用次数和延迟"
```

---

## 2. 题目验证智能体（ValidatorTaskSpec）

规划智能体控制**质量标准**，验证智能体自己决定**怎么验证**。

### inputs

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `artifact_ref` | `str` | ✓ | 待验证题目的 shared_state artifact key，如 `"qa_candidate_pool"` |
| `blueprint_ref` | `str` | | blueprint 的 artifact key，用于获取 modes 的目标分布以选题 |
| `existing_pool_summary` | `str` | | 已有题库摘要路径，用于跨批次去重 |

### constraints

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `citation_enabled` | `bool` | | 是否启用引用验证，默认 `true` |
| `min_citation_score` | `float` | | 引用匹配最低分数（0-1），低于此值拒收，默认 `0.65` |
| `llm_enabled` | `bool` | | 是否启用 LLM 质量评分，默认 `true` |
| `min_overall_score` | `float` | | LLM 综合质量最低分（0-1），默认 `0.75` |
| `hard_floors` | `dict[str, float]` | | 各维度硬底线，任一维度低于底线即拒收。如 `{clarity: 0.6, answerability: 0.7, faithfulness: 0.7, mode_alignment: 0.7}` |
| `dedup_threshold` | `float` | | 判重相似度阈值（0-1），默认 `0.85` |
| `min_batch_pass_rate` | `float` | | 单批最低通过率（0-1），低于此值触发重新规划，默认 `0.5` |
| `allowed_auto_fixes` | `list[str]` | | 允许自动修复的错误类型，如 `["citation_format", "difficulty_label"]`。空列表表示不自动修复 |

### acceptance_criteria

```yaml
- "每道题输出明确的 passed / rejected / needs_fix / needs_regeneration 状态"
- "rejected 题目附带具体拒收原因（如 question_is_ambiguous、answer_not_supported_by_source）"
- "通过题目总数不低于 mode_targets 的各题型 count 之和"
```

### return_requirements

```yaml
- "通过题目集（ValidatedQuestionRecord 列表）"
- "拒收题目集（含拒收原因和证据）"
- "需修复题目集（含修复建议）"
- "批次质量指标（通过率、各维度平均分、拒收原因分布）"
- "判重结果（重复题目对）"
```

---

## 3. 模型评估智能体（EvaluatorTaskSpec）

规划智能体控制**评估什么模型、测什么指标**，评估智能体自己决定**怎么测**。

### inputs

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `artifact_ref` | `str` | ✓ | 验证通过题目的 shared_state artifact key，如 `"validated_questions"` |
| `candidate_models` | `list[str]` | ✓ | 被评估模型列表，对应 `model_registry.yaml` 中的逻辑名 |
| `judge_model` | `str \| None` | | 裁判模型逻辑名。`null` 表示不启用 LLM Judge |
| `registry_path` | `str` | | `model_registry.yaml` 路径，默认 `config/model_registry.yaml` |

### constraints

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `generation_params` | `dict` | ✓ | 候选模型推理参数，`{temperature: 0.0, top_p: 1.0, max_tokens: 1024}` |
| `judge_params` | `dict` | | 裁判模型调用参数，`{temperature: 0.0, top_p: 1.0, max_tokens: 1200}` |
| `repeat_count` | `int` | | 每题重复评估次数，默认 `1` |
| `max_concurrency` | `int` | | 最大并发推理数，默认 `8` |
| `record_token_usage` | `bool` | | 是否记录 token 用量，默认 `true` |
| `record_latency` | `bool` | | 是否记录延迟，默认 `true` |
| `record_cost` | `bool` | | 是否记录成本（需要价格配置），默认 `false` |
| `failure_handling` | `dict` | | 失败处理策略，`{timeout_seconds: 60, max_retries: 2, on_failure: "skip"}` |

### metrics

每种题型独立配置，按 `question_mode` 分组：

| 字段 | 类型 | 说明 |
|---|---|---|
| `metrics.{mode}.automatic` | `list[str]` | 自动指标名列表，如 `["accuracy", "exact_match", "f1", "bertscore"]` |
| `metrics.{mode}.judge` | `list[JudgeMetric]` | LLM 裁判指标，`{name, description, direction}`。`direction` 为 `"higher_is_better"` 或 `"lower_is_better"` |

```yaml
# 示例
metrics:
  qa:
    automatic: [exact_match, f1, bertscore, rouge_l]
    judge:
      - {name: correctness, description: "是否正确回答", direction: higher_is_better}
      - {name: faithfulness, description: "是否被证据支持", direction: higher_is_better}
      - {name: hallucination, description: "是否包含无根据内容", direction: lower_is_better}
  multiple_choice:
    automatic: [accuracy]
    judge: []
```

### acceptance_criteria

```yaml
- "每道题对每个候选模型产出完整的推理结果"
- "失败项有明确的错误原因（超时、拒答、解析失败）"
- "所有指定指标均已计算并可聚合"
- "LLM Judge 评分完成（如启用）"
```

### return_requirements

```yaml
- "题目级分数矩阵（question × model × metric）"
- "主题级聚合指标（topic × model × metric 均值）"
- "难度级聚合指标（difficulty × model × metric 均值）"
- "题型级聚合指标（question_mode × model × metric 均值）"
- "能力级聚合指标（capability × model × metric 均值）"
- "token 用量、延迟、成本统计（per model）"
- "低置信度结果标记列表"
- "无效题目或模型拒答列表"
```

---

## 完整示例：一轮评估的 TaskSpec 下发序列

```python
# 规划智能体 → 题目生成智能体
GeneratorTaskSpec(
    task_id="gen_round_1",
    objective="为 AI Safety 和 Quantum Computing 主题生成首批 QA 题目",
    inputs={
        "topics": ["AI Safety", "Quantum Computing"],
        "language": "en",
    },
    constraints={
        "mode_targets": {
            "qa": {"count": 10, "difficulty_distribution": {"easy": 0.3, "medium": 0.4, "hard": 0.3}},
            "multiple_choice": {"count": 6, "difficulty_distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2}},
        },
        "require_citation": True,
        "require_capability_label": True,
    },
    acceptance_criteria=["输出题目数 ≥ 13", "每题含完整字段"],
    return_requirements=["题目JSON", "来源证据", "token统计"],
)

# 规划智能体 → 题目验证智能体
ValidatorTaskSpec(
    task_id="verify_round_1",
    objective="验证首批生成题目的质量",
    inputs={
        "artifact_ref": "qa_candidate_pool",
    },
    constraints={
        "citation_enabled": True,
        "min_citation_score": 0.65,
        "llm_enabled": True,
        "min_overall_score": 0.75,
        "hard_floors": {"clarity": 0.6, "answerability": 0.7, "faithfulness": 0.7, "mode_alignment": 0.7},
        "dedup_threshold": 0.85,
    },
    acceptance_criteria=["每题有明确状态", "通过数 ≥ 目标数"],
    return_requirements=["通过题目集", "拒收题目集", "质量指标"],
)

# 规划智能体 → 模型评估智能体
EvaluatorTaskSpec(
    task_id="eval_round_1",
    objective="评估 qwen2.5-7b 和 llama3.1-8b 在首轮题库上的表现",
    inputs={
        "artifact_ref": "validated_questions",
        "candidate_models": ["qwen2.5-7b", "llama3.1-8b"],
        "judge_model": "gpt-4o-mini-judge",
    },
    constraints={
        "generation_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024},
        "judge_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1200},
        "record_token_usage": True,
        "record_latency": True,
    },
    metrics={
        "qa": {
            "automatic": ["exact_match", "f1", "bertscore"],
            "judge": [
                {"name": "correctness", "description": "是否正确回答", "direction": "higher_is_better"},
                {"name": "faithfulness", "description": "是否被证据支持", "direction": "higher_is_better"},
                {"name": "hallucination", "description": "是否包含无根据内容", "direction": "lower_is_better"},
            ],
        },
        "multiple_choice": {
            "automatic": ["accuracy"],
            "judge": [],
        },
    },
    acceptance_criteria=["每题对每模型有结果", "所有指标已计算"],
    return_requirements=["分数矩阵", "聚合指标", "token/延迟统计"],
)
```
