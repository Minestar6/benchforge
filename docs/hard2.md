# 6_14_2 版本 QA Agent 问题修复技术方案

## 1. 文档目标

本方案针对 `develop1` 分支最新提交 `6_14_2` 中的 QA Agent 修改进行二次修复。

当前版本已经基本完成了以下目标：

1. 引入 `HARD_GENERATE` 策略；
2. hard 轮根据 `mode + difficulty` 路由到 hard 专有 prompt；
3. 删除/禁用了题目进化主流程；
4. 扩展 `CandidateRecord`，保留 prompt 输出字段和执行元数据；
5. 删除了一部分没有执行落点的冗余字段。

但目前代码仍存在几个问题：

1. `_generate_for_topic()` 中 `filter_failures` 使用了尚未定义的 `chunk_id_list` 和 `chunk_texts`，可能导致运行失败；
2. `HARD_GENERATE` 分支没有强制提高 multi-chunk evidence 比例；
3. `hard_gap_threshold` 被读取但没有使用，hard 生成触发可能过于激进；
4. 题目进化相关字段和统计仍有残留，需要后续清理；
5. candidate metadata 保存方向正确，但需要确认 accepted / rejected 的字段构造顺序和日志输出稳定。

本方案目标是给出当前版本的最小安全修复路径。

---

## 2. 当前 6_14_2 已完成内容

### 2.1 hard prompt 路由已基本完成

当前代码已新增：

```python
def resolve_prompt_template_id(mode: str, difficulty: str) -> str:
    mode = str(mode or "").lower()
    difficulty = str(difficulty or "").lower()

    if mode == "qa":
        return "qa_hard_generation_v1" if difficulty == "hard" else "qa_generation_v1"

    if mode == "multiple_choice":
        return (
            "mcq_hard_generation_v1"
            if difficulty == "hard"
            else "mcq_generation_v1"
        )

    return "qa_generation_v1"
```

并且 `_make_batch()` 已经改为：

```python
prompt_template_id=resolve_prompt_template_id(
    round_plan.mode,
    round_plan.difficulty,
)
```

这说明 hard 轮已经能够真正使用：

```text
qa_hard_generation_v1
mcq_hard_generation_v1
```

这一步是正确的。

---

### 2.2 generator hard prompt 映射已完成

当前 `generator.py` 中已经支持：

```python
qa_hard_generation_v1 -> qa_hard_system_prompt + qa_user_prompt
mcq_hard_generation_v1 -> mcq_hard_system_prompt + mcq_user_prompt
```

这也符合预期。

---

### 2.3 `CandidateRecord` 已扩展

当前 `CandidateRecord` 已保存：

```text
question_mode
question_type
required_capability
estimated_difficulty
citations
thought_process
llm_call_id
chunks
generation_round
raw_item
```

这解决了之前 prompt 输出字段进入 `candidate_pool.json` 时丢失的问题。

---

### 2.4 MCQ alias 兼容已加入

当前代码已经新增：

```python
get_answer(q)
get_choices(q)
normalize_item_for_record(q, fallback_mode)
```

可以兼容：

```text
correct_answer -> answer
options -> choices
```

这部分方向正确。

---

## 3. 当前必须修复的问题

## 问题一：`filter_failures` 使用了未定义变量

### 3.1 问题描述

当前 `_generate_for_topic()` 中存在类似顺序：

```python
accepted_questions, rejected_questions = _question_filter.filter_questions(raw_questions)

filter_failures = [
    {
        "question": item.get("question", ""),
        "reason": reason,
        "raw_item": dict(item),
        "topic": topic,
        "chunk_ids": chunk_id_list,
        "chunks": chunk_texts,
        "llm_call_id": llm_call_id,
        "generation_round": round_plan.round_in_mode,
    }
    for item, reason in rejected_questions
]

chunk_id_list = raw_chunk_ids(chunks)

chunk_text_map: dict[str, str] = {}
...
```

这里 `filter_failures` 使用了：

```text
chunk_id_list
chunk_texts
```

但这两个变量在后面才定义。

如果 `rejected_questions` 非空，会触发：

```text
UnboundLocalError
NameError
```

即使当前某次运行没有 rejected，也只是暂时没有触发，不代表代码安全。

---

### 3.2 修复目标

调整 `_generate_for_topic()` 中变量构造顺序：

1. 先 parse / filter；
2. 再构造 `chunk_id_list`；
3. 再构造 `chunk_text_map`；
4. 再构造 `chunk_texts`；
5. 然后构造 `filter_failures`；
6. 最后给 accepted questions 补充字段。

---

### 3.3 推荐修复代码

将 `_generate_for_topic()` 中相关片段改成：

```python
raw_questions = parse_questions(raw_items)
accepted_questions, rejected_questions = _question_filter.filter_questions(raw_questions)

chunk_id_list = raw_chunk_ids(chunks)

chunk_text_map: dict[str, str] = {}
for u in chunks:
    if hasattr(u, "chunk_ids") and hasattr(u, "texts"):
        for cid, txt in zip(u.chunk_ids, u.texts):
            chunk_text_map[cid] = txt
    elif hasattr(u, "chunk_id"):
        chunk_text_map[u.chunk_id] = getattr(u, "text", "")

chunk_texts = [chunk_text_map.get(cid, "") for cid in chunk_id_list]

filter_failures = [
    {
        "question": item.get("question", ""),
        "reason": reason,
        "raw_item": dict(item),
        "topic": topic,
        "chunk_ids": chunk_id_list,
        "chunks": chunk_texts,
        "llm_call_id": llm_call_id,
        "generation_round": round_plan.round_in_mode,
    }
    for item, reason in rejected_questions
]

for q in accepted_questions:
    q["chunk_ids"] = chunk_id_list
    q["chunks"] = chunk_texts
    q["topic"] = topic
    q["llm_call_id"] = llm_call_id
    q["generation_round"] = round_plan.round_in_mode
```

---

### 3.4 验收标准

运行后：

1. 出现 rejected question 时不会报错；
2. rejected 记录中保留：

```json
{
  "raw_item": {...},
  "chunk_ids": [...],
  "chunks": [...],
  "llm_call_id": "...",
  "generation_round": 1
}
```

3. accepted 记录中也保留：

```json
{
  "chunk_ids": [...],
  "chunks": [...],
  "llm_call_id": "...",
  "generation_round": 1
}
```

---

## 问题二：`HARD_GENERATE` 分支缺少 multi-chunk 偏好增强

### 4.1 问题描述

hard 题通常需要：

```text
multi-hop evidence
cross-chunk synthesis
comparison
limitation analysis
causal reasoning
```

因此 hard 轮应该尽量提高 multi-chunk evidence 的比例。

之前普通 difficulty 分支中存在类似逻辑：

```python
if difficulty == "hard":
    multi_k = max(multi_k, single_k)
```

但当前新增的 `HARD_GENERATE` 分支中，如果没有这句，就可能出现：

```text
hard prompt + evidence 不够复杂
```

这样 hard prompt 虽然生效，但看到的证据仍然偏单段，难题质量会受影响。

---

### 4.2 修复目标

在 `planner.py` 的 `HARD_GENERATE` 分支中，`compute_dynamic_chunk_k()` 返回后补充：

```python
multi_k = max(multi_k, single_k)
```

---

### 4.3 推荐修复代码

找到 `build_adaptive_plan()` 中：

```python
if strategy == RoundStrategy.HARD_GENERATE:
    ...
    single_k, multi_k, tgt = compute_dynamic_chunk_k(...)
    return ModeRoundPlan(...)
```

改成：

```python
if strategy == RoundStrategy.HARD_GENERATE:
    topics = tuple(
        choose_low_coverage_topics(
            blueprint,
            mode_state,
            config.planner.topics_per_round,
        )
    )

    single_k, multi_k, tgt = compute_dynamic_chunk_k(
        mode=mode,
        mode_cfg=mode_cfg,
        difficulty="hard",
        selected_topic_count=max(1, len(topics)),
        mode_state=mode_state,
        blueprint=blueprint,
        config=config,
    )

    multi_k = max(multi_k, single_k)

    return ModeRoundPlan(
        mode=mode,
        round_in_mode=mode_state.round_in_mode,
        strategy=strategy,
        difficulty="hard",
        topics=topics,
        single_k=single_k,
        multi_k=multi_k,
        target_candidates_per_topic=tgt,
        reason=reason,
    )
```

---

### 4.4 验收标准

`round_plan.jsonl` 中 hard_generate 轮应满足：

```text
multi_k >= single_k
difficulty = hard
strategy = hard_generate
```

---

## 问题三：`hard_gap_threshold` 定义了但没有使用

### 5.1 问题描述

当前 `select_strategy()` 中读取了：

```python
hard_gap_threshold = getattr(decision_cfg, "hard_gap_threshold", 0.2)
```

但 hard generation 触发条件可能写成：

```python
if hard_gap_val > 0 or too_easy > too_easy_threshold:
    return RoundStrategy.HARD_GENERATE, ...
```

这会导致：

```text
只要 hard_gap 有一点点大于 0，就进入 HARD_GENERATE
```

这可能过于激进，导致 hard 生成轮过多，影响整体题目分布。

---

### 5.2 修复目标

使用配置中的 `hard_gap_threshold`：

```python
if hard_gap_val > hard_gap_threshold or too_easy > too_easy_threshold:
    return RoundStrategy.HARD_GENERATE, ...
```

---

### 5.3 推荐修复代码

将：

```python
if hard_gap_val > 0 or too_easy > too_easy_threshold:
    return RoundStrategy.HARD_GENERATE, (
        f"hard_gap={hard_gap_val:.2f}, too_easy={too_easy:.2f}; "
        "use direct hard generation"
    )
```

改成：

```python
if hard_gap_val > hard_gap_threshold or too_easy > too_easy_threshold:
    return RoundStrategy.HARD_GENERATE, (
        f"hard_gap={hard_gap_val:.2f}, too_easy={too_easy:.2f}; "
        "use direct hard generation"
    )
```

---

### 5.4 验收标准

当：

```text
hard_gap_val <= hard_gap_threshold
too_easy <= too_easy_threshold
```

时，不应触发 `HARD_GENERATE`。

当：

```text
hard_gap_val > hard_gap_threshold
```

或：

```text
too_easy > too_easy_threshold
```

时，才触发 `HARD_GENERATE`。

---

## 问题四：题目进化残留代码需要分阶段清理

### 6.1 当前状态

当前版本主流程已经不再依赖题目进化，但仍可能残留：

```text
RoundStrategy.EVOLVE_TO_HARDER
_execute_evolve_round()
CandidateStatus.EVOLVED
ModeState.evolved_count
ModeState.evolvable_surplus()
RoundFeedback.evolved_count
build_round_feedback(..., evolved_seed_count=...)
```

这些残留短期内不一定导致运行错误，但会造成：

1. 代码语义不清晰；
2. 后续维护者误以为题目进化仍然可用；
3. feedback / state 中存在无效统计；
4. planner 的策略空间看起来比实际更复杂。

---

### 6.2 清理原则

不要在修复运行 bug 的同一次提交中大规模删除所有进化代码。建议分两步：

#### 第一步：当前提交只修运行问题

先修：

```text
filter_failures 变量顺序
multi_k hard 偏好
hard_gap_threshold 判断
```

确保 e2e 跑通。

#### 第二步：单独提交清理进化残留

确认 hard-only generation 稳定后，再删除：

```text
RoundStrategy.EVOLVE_TO_HARDER
_execute_evolve_round()
CandidateStatus.EVOLVED
ModeState.evolved_count
ModeState.evolvable_surplus()
RoundFeedback.evolved_count
build_round_feedback(..., evolved_seed_count=...)
```

---

### 6.3 建议保留项

如果担心历史 run 文件兼容，可以暂时保留：

```python
parent_question_id: str | None = None
```

但标注为 deprecated：

```python
parent_question_id: str | None = None  # Deprecated: evolution is disabled.
```

如果只关注新实验，可以后续删除。

---

## 7. 推荐执行顺序

## Step 1：修复 `_generate_for_topic()` 变量顺序

修改文件：

```text
agents/qa_agent/executor.py
```

目标：

```text
先定义 chunk_id_list / chunk_texts
再构造 filter_failures
```

这是最高优先级，因为它可能导致运行失败。

---

## Step 2：修复 `HARD_GENERATE` 的 multi-k 逻辑

修改文件：

```text
agents/qa_agent/planner.py
```

目标：

```python
multi_k = max(multi_k, single_k)
```

只加在 `HARD_GENERATE` 分支中。

---

## Step 3：修复 `hard_gap_threshold` 使用

修改文件：

```text
agents/qa_agent/planner.py
```

目标：

```python
if hard_gap_val > hard_gap_threshold or too_easy > too_easy_threshold:
    return RoundStrategy.HARD_GENERATE, ...
```

---

## Step 4：运行静态检查

执行：

```bash
python -m compileall agents benchforge utils
```

如果你的项目中没有 `benchforge` 包目录或目录名称不同，根据实际目录调整。

---

## Step 5：运行最小 e2e

运行当前项目用于 QA agent 的 e2e 生成命令。

重点检查：

```text
round_plan.jsonl
round_feedback.jsonl
candidate_pool.json
mode_metrics.json
```

---

## Step 6：确认后再清理进化残留

如果 Step 1-5 正常，再单独提交删除 evolve 相关代码。

---

## 8. 最小修复 patch 摘要

### 8.1 `executor.py`

修复前：

```python
accepted_questions, rejected_questions = _question_filter.filter_questions(raw_questions)

filter_failures = [
    {
        "chunk_ids": chunk_id_list,
        "chunks": chunk_texts,
        ...
    }
    for item, reason in rejected_questions
]

chunk_id_list = raw_chunk_ids(chunks)
...
```

修复后：

```python
accepted_questions, rejected_questions = _question_filter.filter_questions(raw_questions)

chunk_id_list = raw_chunk_ids(chunks)

chunk_text_map: dict[str, str] = {}
for u in chunks:
    if hasattr(u, "chunk_ids") and hasattr(u, "texts"):
        for cid, txt in zip(u.chunk_ids, u.texts):
            chunk_text_map[cid] = txt
    elif hasattr(u, "chunk_id"):
        chunk_text_map[u.chunk_id] = getattr(u, "text", "")

chunk_texts = [chunk_text_map.get(cid, "") for cid in chunk_id_list]

filter_failures = [
    {
        "question": item.get("question", ""),
        "reason": reason,
        "raw_item": dict(item),
        "topic": topic,
        "chunk_ids": chunk_id_list,
        "chunks": chunk_texts,
        "llm_call_id": llm_call_id,
        "generation_round": round_plan.round_in_mode,
    }
    for item, reason in rejected_questions
]

for q in accepted_questions:
    q["chunk_ids"] = chunk_id_list
    q["chunks"] = chunk_texts
    q["topic"] = topic
    q["llm_call_id"] = llm_call_id
    q["generation_round"] = round_plan.round_in_mode
```

---

### 8.2 `planner.py`

修复 hard generation 触发条件：

```python
if hard_gap_val > hard_gap_threshold or too_easy > too_easy_threshold:
    return RoundStrategy.HARD_GENERATE, (
        f"hard_gap={hard_gap_val:.2f}, too_easy={too_easy:.2f}; "
        "use direct hard generation"
    )
```

修复 hard generation chunk 策略：

```python
single_k, multi_k, tgt = compute_dynamic_chunk_k(
    mode=mode,
    mode_cfg=mode_cfg,
    difficulty="hard",
    selected_topic_count=max(1, len(topics)),
    mode_state=mode_state,
    blueprint=blueprint,
    config=config,
)

multi_k = max(multi_k, single_k)
```

---

## 9. 验收标准

修复完成后必须满足：

### 9.1 编译检查通过

```bash
python -m compileall agents benchforge utils
```

无语法错误。

---

### 9.2 hard_generate 正常出现

`round_plan.jsonl` 中出现：

```json
{
  "strategy": "hard_generate",
  "difficulty": "hard"
}
```

---

### 9.3 hard 轮使用 hard prompt

round results 或 trace 中出现：

```json
{
  "prompt_template_id": "qa_hard_generation_v1"
}
```

或：

```json
{
  "prompt_template_id": "mcq_hard_generation_v1"
}
```

---

### 9.4 rejected 不再导致运行失败

即使有 rejected question，也不会因为：

```text
chunk_id_list
chunk_texts
```

未定义而崩溃。

---

### 9.5 candidate_pool 字段完整

`candidate_pool.json` 中 accepted 题目应包含：

```json
{
  "estimated_difficulty": 8,
  "difficulty": "hard",
  "citations": ["..."],
  "question_type": "...",
  "required_capability": "...",
  "thought_process": "...",
  "llm_call_id": "...",
  "chunks": ["..."],
  "raw_item": {...}
}
```

rejected 题目应包含：

```json
{
  "status": "rejected",
  "reject_reason": "...",
  "raw_item": {...},
  "chunk_ids": ["..."],
  "chunks": ["..."],
  "llm_call_id": "..."
}
```

---

### 9.6 hard generation 不应过度触发

当：

```text
hard_gap_val <= hard_gap_threshold
too_easy <= too_easy_threshold
```

时，不应触发 `HARD_GENERATE`。

---

## 10. 后续清理建议

当前最小修复通过后，建议下一次提交清理：

```text
RoundStrategy.EVOLVE_TO_HARDER
_execute_evolve_round()
CandidateStatus.EVOLVED
ModeState.evolved_count
ModeState.evolvable_surplus()
RoundFeedback.evolved_count
build_round_feedback(..., evolved_seed_count=...)
```

如果需要兼容旧数据，可以暂时保留：

```text
parent_question_id
```

但标注 deprecated。

---

## 11. 最终结论

`6_14_2` 的总体修改方向是正确的：

1. hard prompt 路由已接入；
2. candidate_pool 字段丢失问题基本解决；
3. 冗余 batch 字段已经大量清理；
4. 题目进化主流程已经被替换为 direct hard generation。

但当前版本还不能直接认为稳定，必须先修：

```text
1. filter_failures 使用未定义 chunk_id_list / chunk_texts；
2. HARD_GENERATE 分支补 multi_k = max(multi_k, single_k)；
3. hard_gap 判断使用 hard_gap_threshold。
```

修完这三点后，再跑 e2e 验证 hard prompt 是否真正改善 hard 题比例。
