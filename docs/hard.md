# QA Agent 三问题修复执行计划

## 目标

本方案用于修复 `develop1` 分支中 QA Agent 题目生成流程的三个问题：

1. **困难题目生成不足**
   - 当前 hard 轮没有真正调用 hard 专有 prompt。
   - 当前题目进化策略成本较高，且效果不稳定。
   - 本版本决定删除/禁用题目进化，仅使用 hard 专有 prompt 直接生成困难题目。

2. **冗余字段过多**
   - 当前一些字段被传入 `GenerationBatch` 或 `ModeRoundPlan`，但没有进入 prompt，也没有影响 chunk 检索、状态更新或日志分析。
   - 需要清理没有真实执行落点的字段。

3. **题目生成字段中途丢失**
   - prompt 要求输出的字段，例如 `question_mode`、`thought_process`、`question_type`、`required_capability`、`estimated_difficulty`、`citations` 等，在写入 `candidate_pool.json` 时丢失。
   - 需要扩展 `CandidateRecord` 并修复字段映射逻辑。

当前修改原则：

- 不修改 prompt 内容。
- 不增加 hard filter。
- 不做题目进化。
- 先修 hard prompt 路由，再修字段丢失，最后清理冗余字段。
- 优先兼容当前实现，避免一次性大范围删除导致新问题。

---

## 一、当前问题定位

### 1. hard 轮没有真正使用 hard prompt

当前 `executor._make_batch()` 中 prompt 路由写死为：

```python
prompt_template_id=(
    "mcq_generation_v1" if round_plan.mode == "multiple_choice" else "qa_generation_v1"
)
```

这意味着即使：

```python
round_plan.difficulty == "hard"
```

也仍然会使用普通 prompt，而不是已有的：

```text
qa_hard_system_prompt.md
mcq_hard_system_prompt.md
```

因此日志中出现了：

```text
planner 已经选择 hard
但实际生成仍然大量 easy / medium
```

核心原因之一就是 hard prompt 没有真正接入生成流程。

---

### 2. 题目进化策略不适合当前版本

当前代码中存在：

```text
RoundStrategy.EVOLVE_TO_HARDER
_execute_evolve_round()
evolve_source_count
parent_question_id
evolvable_surplus()
evolved_count
```

但本版本不想做题目进化。  
因此应该把 hard 缺口修复改成：

```text
hard_gap 大 / too_easy_ratio 高
    -> HARD_GENERATE
    -> 直接调用 hard prompt 生成困难题
```

而不是：

```text
medium parent question
    -> evolve_to_harder
    -> 生成新题
```

---

### 3. 冗余字段没有执行落点

当前一些字段被传入，但没有真实作用。

例如 `GenerationBatch` 中：

```text
remaining_count
requested_min_questions
requested_target_questions
```

这些字段目前：

- 不进入 prompt；
- 不影响 generator；
- 不影响 chunk 检索；
- 不影响 filter；
- 不影响状态更新。

因此属于冗余字段。

另外，`ModeRoundPlan` 中类似：

```text
evidence_strategy
expand_docs
evolve_source_count
```

也存在类似问题。  
其中 `evidence_strategy` 当前没有传入 `sample_chunks()`，所以不是实际检索控制参数。

---

### 4. prompt 输出字段写入 candidate_pool 时丢失

QA prompt 要求输出结构类似：

```json
[
  {
    "question": "The question text",
    "answer": "Complete, accurate answer to the question",
    "question_mode": "qa",
    "thought_process": "Explain why this question effectively tests understanding of the document content",
    "question_type": "The type of question (factual, analytical, conceptual, etc.)",
    "required_capability": "Describe the capability required to answer this question",
    "estimated_difficulty": 5,
    "citations": ["Exact quote 1 from source text", "Exact quote 2 from source text"]
  }
]
```

但当前 `CandidateRecord` 只保存了较少字段，例如：

```text
question_id
question
answer
topic
difficulty
status
source_round
source_strategy
chunk_ids
reject_reason
parent_question_id
choices
```

因此这些字段会丢失：

```text
question_mode
thought_process
question_type
required_capability
estimated_difficulty 原始分数
citations
llm_call_id
chunks
raw_item
```

其中 `estimated_difficulty` 当前只被压缩成：

```text
easy / medium / hard
```

原始 1-10 分数没有保存。

---

## 二、总体修改策略

修改按三个阶段执行。

### 阶段 1：修复 hard prompt 路由

目标：

```text
planner 不再返回 EVOLVE_TO_HARDER
hard 缺口触发 HARD_GENERATE
executor 根据 mode + difficulty 选择 prompt_template_id
generator 支持 hard prompt template id 映射
```

### 阶段 2：修复 candidate_pool 字段丢失

目标：

```text
CandidateRecord 保存 prompt 输出元数据
CandidateRecord 保存执行元数据
accepted / rejected 都保存 raw_item
MCQ alias 字段 options / correct_answer 不再丢失
```

### 阶段 3：清理冗余字段

目标：

```text
删除 GenerationBatch 中没有执行落点的数量字段
删除 ModeRoundPlan 中未落地字段
最后清理 evolve 相关代码
```

---

# 三、阶段 1：修复 hard prompt 路由

## 1. 修改 `agents/qa_agent/planner.py`

### 1.1 新增 `HARD_GENERATE`

找到 `RoundStrategy`，新增：

```python
HARD_GENERATE = "hard_generate"
```

如果当前存在：

```python
EVOLVE_TO_HARDER = "evolve_to_harder"
```

第一阶段可以先保留，不要立即删除，以减少兼容风险。

---

### 1.2 不再返回 `EVOLVE_TO_HARDER`

找到策略选择逻辑中类似：

```python
if (
    hard_gap_val > 0
    and too_easy > too_easy_threshold
    and mode_state.evolvable_surplus("medium", med_ratio) > 0
):
    return RoundStrategy.EVOLVE_TO_HARDER, ...
```

修改为：

```python
if hard_gap_val > 0 or too_easy > too_easy_threshold:
    return RoundStrategy.HARD_GENERATE, (
        f"hard_gap={hard_gap_val:.2f}, too_easy={too_easy:.2f}; "
        "use direct hard generation"
    )
```

要求：

- 不再依赖 `medium surplus`。
- 不再调用 `evolvable_surplus()`。
- 不再设置 `evolve_source_count`。

---

### 1.3 新增 `HARD_GENERATE` plan 构造

在 `build_adaptive_plan()` 或对应构造 round plan 的函数中增加分支：

```python
if strategy == RoundStrategy.HARD_GENERATE:
    topics = tuple(
        choose_low_coverage_topics(
            blueprint,
            mode_state,
            config.planner.topics_per_round,
        )
    )

    single_k, multi_k, target_per_topic = compute_dynamic_chunk_k(
        mode=mode,
        mode_cfg=mode_cfg,
        difficulty="hard",
        selected_topic_count=max(1, len(topics)),
        mode_state=mode_state,
        blueprint=blueprint,
        config=config,
    )

    return ModeRoundPlan(
        mode=mode,
        round_in_mode=mode_state.round_in_mode,
        strategy=RoundStrategy.HARD_GENERATE,
        difficulty="hard",
        topics=topics,
        single_k=single_k,
        multi_k=multi_k,
        target_candidates_per_topic=target_per_topic,
        reason=reason,
    )
```

如果当前代码没有 `choose_low_coverage_topics()`，不要新造复杂逻辑，直接复用当前 `FOCUS_TOPIC` 或 `FOCUS_DIFFICULTY` 中已有的 topic 选择逻辑。

---

### 1.4 验收标准

生成的新 `round_plan.jsonl` 中不应再出现：

```json
"strategy": "evolve_to_harder"
```

应出现：

```json
{
  "strategy": "hard_generate",
  "difficulty": "hard"
}
```

---

## 2. 修改 `agents/qa_agent/executor.py`

### 2.1 禁用 evolve 分支

当前 `execute_mode_round_plan()` 中可能有：

```python
if round_plan.strategy == RoundStrategy.EVOLVE_TO_HARDER:
    return await _execute_evolve_round(...)
```

第一阶段建议不要物理删除 `_execute_evolve_round()`，只删除或禁用触发路径。

推荐改法：

```python
# EVOLVE_TO_HARDER is disabled in the current version.
# Planner should not emit this strategy.
# All rounds, including hard generation, use the normal topic generation path.
```

然后让 hard 轮继续走普通 `_generate_for_topic()` 流程。

---

### 2.2 新增 prompt template resolver

在 `_make_batch()` 附近新增：

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

---

### 2.3 修改 `_make_batch()`

把原来的写死逻辑：

```python
prompt_template_id=(
    "mcq_generation_v1" if round_plan.mode == "multiple_choice" else "qa_generation_v1"
)
```

改成：

```python
prompt_template_id=resolve_prompt_template_id(
    round_plan.mode,
    round_plan.difficulty,
)
```

第一阶段可以暂时保留 batch 中的旧字段，避免 schema 尚未清理时报错：

```python
return GenerationBatch(
    topic=topic,
    target_mode=round_plan.mode,
    target_difficulty=round_plan.difficulty,
    remaining_count=round_plan.target_candidates_per_topic,
    single_chunk_ids=single_ids,
    multi_chunk_ids=multi_ids,
    prompt_template_id=resolve_prompt_template_id(
        round_plan.mode,
        round_plan.difficulty,
    ),
    requested_min_questions=round_plan.target_candidates_per_topic,
    requested_target_questions=round_plan.target_candidates_per_topic,
)
```

这些冗余字段在第三阶段再删除。

---

### 2.4 记录 prompt_template_id 到日志

在 `_generate_for_topic()` 返回结果中加入：

```python
"prompt_template_id": batch.prompt_template_id,
```

在 `round_results.append(...)` 中同步写入：

```python
"prompt_template_id": res.get("prompt_template_id"),
```

---

### 2.5 验收标准

hard 轮日志中应该能看到：

```json
{
  "strategy": "hard_generate",
  "difficulty": "hard",
  "prompt_template_id": "qa_hard_generation_v1"
}
```

或选择题模式下：

```json
{
  "prompt_template_id": "mcq_hard_generation_v1"
}
```

---

## 3. 修改 `agents/qa_agent/generator.py`

### 3.1 支持 hard prompt 映射

找到 `_load_prompt()` 或 prompt template 映射逻辑。

增加：

```python
if template_id == "qa_generation_v1":
    sys_key, user_key = "qa_system_prompt", "qa_user_prompt"

elif template_id == "qa_hard_generation_v1":
    sys_key, user_key = "qa_hard_system_prompt", "qa_user_prompt"

elif template_id == "mcq_generation_v1":
    sys_key, user_key = "mcq_system_prompt", "mcq_user_prompt"

elif template_id == "mcq_hard_generation_v1":
    sys_key, user_key = "mcq_hard_system_prompt", "mcq_user_prompt"

else:
    sys_key, user_key = f"{template_id}_system", f"{template_id}_user"
```

不要修改 prompt 文件内容。

---

### 3.2 验收标准

运行 hard 轮时不应报错寻找：

```text
qa_hard_generation_v1_system.md
qa_hard_generation_v1_user.md
```

而应该加载已有文件：

```text
qa_hard_system_prompt.md
qa_user_prompt.md
```

选择题 hard 轮应加载：

```text
mcq_hard_system_prompt.md
mcq_user_prompt.md
```

---

# 四、阶段 2：修复 candidate_pool 字段丢失

## 4. 修改 `agents/qa_agent/state.py`

### 4.1 扩展 `CandidateRecord`

找到 `CandidateRecord`，新增字段。所有新增字段都必须有默认值，以兼容旧 candidate_pool。

需要引入：

```python
from typing import Any
from dataclasses import field
```

建议结构：

```python
@dataclass
class CandidateRecord:
    question_id: str
    question: str
    answer: str
    topic: str
    difficulty: str
    status: CandidateStatus
    source_round: int
    source_strategy: str
    chunk_ids: list[str]

    reject_reason: str | None = None
    parent_question_id: str | None = None
    choices: list[Any] | None = None

    # Prompt output metadata
    question_mode: str | None = None
    question_type: str | None = None
    required_capability: str | None = None
    estimated_difficulty: int | float | str | None = None
    citations: list[Any] = field(default_factory=list)
    thought_process: str | None = None

    # Execution metadata
    llm_call_id: str | None = None
    chunks: list[str] = field(default_factory=list)
    generation_round: int | None = None

    # Preserve original model output
    raw_item: dict[str, Any] = field(default_factory=dict)
```

注意：

- `parent_question_id` 第一版可以保留，避免兼容问题。
- 不要把新增字段设为必填。

---

### 4.2 验收标准

旧 candidate_pool 如果没有这些字段，加载不应失败。  
新 candidate_pool 中 accepted 和 rejected 记录都应该包含新增字段。

---

## 5. 修改 `agents/qa_agent/executor.py` 的字段映射

### 5.1 新增标准化 helper

在 `executor.py` 中增加：

```python
def get_question_mode(q: dict, fallback_mode: str) -> str:
    return q.get("question_mode") or fallback_mode


def get_answer(q: dict) -> str:
    return q.get("answer") or q.get("correct_answer") or ""


def get_choices(q: dict) -> list | None:
    return q.get("choices") or q.get("options")


def normalize_item_for_record(q: dict, fallback_mode: str) -> dict:
    """Normalize common aliases without dropping original fields."""
    item = dict(q)

    if not item.get("question_mode"):
        item["question_mode"] = fallback_mode

    if "answer" not in item and "correct_answer" in item:
        item["answer"] = item.get("correct_answer")

    if "choices" not in item and "options" in item:
        item["choices"] = item.get("options")

    citations = item.get("citations", [])
    if citations is None:
        citations = []
    elif isinstance(citations, str):
        citations = [citations]
    item["citations"] = citations

    return item
```

说明：

- 不强依赖 `LightweightFilter.normalize_question()`，避免它丢弃未知字段。
- 这里只做最小 alias 兼容。
- 原始字段通过 `raw_item` 完整保存。

---

### 5.2 修改 `update_mode_state()`

把 accepted 记录映射改成保存完整元数据。

示例实现：

```python
def update_mode_state(
    mode_state: ModeState,
    topic: str,
    round_plan: ModeRoundPlan,
    parsed_questions: list[dict],
    filter_failures: list[dict] | None = None,
    parent_question_id: str | None = None,
) -> list[CandidateRecord]:
    """Append new CandidateRecords; return the list of newly added records."""
    new_records: list[CandidateRecord] = []

    for q in parsed_questions:
        raw_item = dict(q)
        q_norm = normalize_item_for_record(q, round_plan.mode)

        raw_estimated = (
            q_norm.get("estimated_difficulty")
            or q_norm.get("difficulty")
            or round_plan.difficulty
        )
        difficulty = normalize_difficulty(raw_estimated)
        question_mode = get_question_mode(q_norm, round_plan.mode)

        record = CandidateRecord(
            question_id=q_norm.get("question_id") or str(uuid.uuid4()),
            question=q_norm.get("question", ""),
            answer=get_answer(q_norm),
            topic=q_norm.get("topic") or topic,
            difficulty=difficulty,
            status=CandidateStatus.ACCEPTED,
            source_round=round_plan.round_in_mode,
            source_strategy=round_plan.strategy.value,
            chunk_ids=q_norm.get("chunk_ids", []),
            parent_question_id=parent_question_id,
            choices=get_choices(q_norm) if question_mode == "multiple_choice" else None,

            question_mode=question_mode,
            question_type=q_norm.get("question_type"),
            required_capability=q_norm.get("required_capability"),
            estimated_difficulty=q_norm.get("estimated_difficulty"),
            citations=q_norm.get("citations", []),
            thought_process=q_norm.get("thought_process"),

            llm_call_id=q_norm.get("llm_call_id"),
            chunks=q_norm.get("chunks", []),
            generation_round=q_norm.get("generation_round"),
            raw_item=raw_item,
        )

        mode_state.candidate_questions.append(record)
        new_records.append(record)

    for item in (filter_failures or []):
        raw = dict(item.get("raw_item") or item)
        q_norm = normalize_item_for_record(raw, round_plan.mode)

        raw_estimated = (
            q_norm.get("estimated_difficulty")
            or q_norm.get("difficulty")
            or round_plan.difficulty
        )
        question_mode = get_question_mode(q_norm, round_plan.mode)

        record = CandidateRecord(
            question_id=q_norm.get("question_id") or str(uuid.uuid4()),
            question=q_norm.get("question", item.get("question", "")),
            answer=get_answer(q_norm),
            topic=item.get("topic") or q_norm.get("topic") or topic,
            difficulty=normalize_difficulty(raw_estimated),
            status=CandidateStatus.REJECTED,
            source_round=round_plan.round_in_mode,
            source_strategy=round_plan.strategy.value,
            chunk_ids=item.get("chunk_ids") or q_norm.get("chunk_ids", []),
            reject_reason=item.get("reason", "filter_rejected"),
            parent_question_id=None,

            choices=get_choices(q_norm) if question_mode == "multiple_choice" else None,

            question_mode=question_mode,
            question_type=q_norm.get("question_type"),
            required_capability=q_norm.get("required_capability"),
            estimated_difficulty=q_norm.get("estimated_difficulty"),
            citations=q_norm.get("citations", []),
            thought_process=q_norm.get("thought_process"),

            llm_call_id=item.get("llm_call_id") or q_norm.get("llm_call_id"),
            chunks=item.get("chunks") or q_norm.get("chunks", []),
            generation_round=item.get("generation_round") or q_norm.get("generation_round"),
            raw_item=raw,
        )

        mode_state.candidate_questions.append(record)
        new_records.append(record)

    return new_records
```

---

### 5.3 验收标准

`candidate_pool.json` 中 accepted 题目应保留：

```json
{
  "question_mode": "qa",
  "question_type": "...",
  "required_capability": "...",
  "estimated_difficulty": 8,
  "citations": ["..."],
  "thought_process": "...",
  "llm_call_id": "...",
  "chunks": ["..."],
  "raw_item": {...}
}
```

---

## 6. 修改 `_generate_for_topic()` 中 rejected 字段保留

### 6.1 目标

rejected 题目也保留原始输出、chunk 信息、llm_call_id，方便调试。

---

### 6.2 调整字段构造顺序

当前可能先构造 `filter_failures`，再构造 chunk 信息。  
需要把 chunk id 和 chunk text 的构造提前。

建议：

```python
chunk_id_list = raw_chunk_ids(chunks)

chunk_text_map: dict[str, str] = {}
for u in chunks:
    if hasattr(u, "chunk_ids") and hasattr(u, "texts"):
        for cid, txt in zip(u.chunk_ids, u.texts):
            chunk_text_map[cid] = txt
    elif hasattr(u, "chunk_id"):
        chunk_text_map[u.chunk_id] = getattr(u, "text", "")

chunk_texts = [chunk_text_map.get(cid, "") for cid in chunk_id_list]
```

accepted question 补充：

```python
for q in accepted_questions:
    q["chunk_ids"] = chunk_id_list
    q["chunks"] = chunk_texts
    q["topic"] = topic
    q["llm_call_id"] = llm_call_id
    q["generation_round"] = round_plan.round_in_mode
```

rejected question 补充：

```python
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
```

---

### 6.3 验收标准

`candidate_pool.json` 中 rejected 题目应包含：

```json
{
  "status": "rejected",
  "reject_reason": "...",
  "raw_item": {...},
  "chunk_ids": [...],
  "chunks": [...],
  "llm_call_id": "..."
}
```

---

# 五、阶段 3：清理冗余字段

这一阶段建议在阶段 1 和阶段 2 e2e 跑通后再执行。

## 7. 清理 `GenerationBatch`

### 7.1 删除字段

找到 `GenerationBatch` 定义，删除：

```python
remaining_count
requested_min_questions
requested_target_questions
```

保留：

```python
topic
target_mode
target_difficulty
single_chunk_ids
multi_chunk_ids
prompt_template_id
additional_instructions
```

建议最终结构：

```python
class GenerationBatch(BaseModel):
    topic: str
    target_mode: str
    target_difficulty: str
    single_chunk_ids: list[str] = Field(default_factory=list)
    multi_chunk_ids: list[str] = Field(default_factory=list)
    prompt_template_id: str
    additional_instructions: str = ""
```

---

### 7.2 同步修改 `_make_batch()`

删除参数：

```python
remaining_count=round_plan.target_candidates_per_topic
requested_min_questions=round_plan.target_candidates_per_topic
requested_target_questions=round_plan.target_candidates_per_topic
```

最终：

```python
return GenerationBatch(
    topic=topic,
    target_mode=round_plan.mode,
    target_difficulty=round_plan.difficulty,
    single_chunk_ids=single_ids,
    multi_chunk_ids=multi_ids,
    prompt_template_id=resolve_prompt_template_id(
        round_plan.mode,
        round_plan.difficulty,
    ),
)
```

---

### 7.3 验收标准

全局搜索不应再有有效引用：

```text
remaining_count
requested_min_questions
requested_target_questions
```

---

## 8. 清理 `ModeRoundPlan` 中未落地字段

### 8.1 删除字段

如果当前 `ModeRoundPlan` 中存在以下字段，删除：

```python
evidence_strategy
expand_docs
evolve_source_count
```

建议最终结构：

```python
@dataclass(frozen=True)
class ModeRoundPlan:
    mode: str
    round_in_mode: int
    strategy: RoundStrategy
    difficulty: str
    topics: tuple[str, ...]
    single_k: int
    multi_k: int
    target_candidates_per_topic: int
    reason: str = ""
    expand_topics: tuple[str, ...] = ()
```

说明：

- 如果还保留 `EXPAND_EVIDENCE`，则保留 `expand_topics`。
- 如果连 `EXPAND_EVIDENCE` 也暂时不用，可以后续再删 `expand_topics`。

---

### 8.2 验收标准

全局搜索不应再有有效代码引用：

```text
evidence_strategy
expand_docs
evolve_source_count
```

---

## 9. 最后清理 evolve 相关代码

这一步放到最后执行。

### 9.1 删除或废弃

如果确认不再做题目进化，删除：

```text
RoundStrategy.EVOLVE_TO_HARDER
_execute_evolve_round()
ModeState.evolvable_surplus()
CandidateStatus.EVOLVED
RoundFeedback.evolved_count
build_round_feedback(..., evolved_seed_count=...)
```

### 9.2 保守兼容

如果担心旧日志兼容：

```text
CandidateRecord.parent_question_id 可以暂时保留，标注 deprecated。
```

如果只跑新实验，可以删除。

---

### 9.3 验收标准

全局搜索不应再有执行路径引用：

```text
EVOLVE_TO_HARDER
_execute_evolve_round
evolvable_surplus
evolved_count
evolved_seed_count
```

---

# 六、测试与验收

## 10. 静态检查

执行：

```bash
python -m compileall agents benchforge utils
```

如果项目有 pytest：

```bash
pytest
```

如果没有完整测试，至少跑 qa_agent 相关测试或最小 e2e。

---

## 11. 全局搜索检查

执行：

```bash
grep -R "EVOLVE_TO_HARDER" -n .
grep -R "evolve_source_count" -n .
grep -R "requested_min_questions" -n .
grep -R "requested_target_questions" -n .
grep -R "remaining_count" -n .
grep -R "evidence_strategy" -n .
grep -R "expand_docs" -n .
```

预期：

- 阶段 1 后：`EVOLVE_TO_HARDER` 可能还存在，但 planner 不应返回。
- 阶段 3 后：上述字段不应有有效代码引用。

---

## 12. 运行一次 e2e

运行当前项目用于题目生成的 e2e 命令。

生成后检查以下文件。

---

### 12.1 检查 `round_plan.jsonl`

应出现：

```json
{
  "strategy": "hard_generate",
  "difficulty": "hard"
}
```

不应出现：

```json
{
  "strategy": "evolve_to_harder"
}
```

---

### 12.2 检查 round results / trace

hard 轮应有：

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

### 12.3 检查 `candidate_pool.json`

每道 accepted 题目应至少包含：

```json
{
  "question": "...",
  "answer": "...",
  "question_mode": "qa",
  "question_type": "...",
  "required_capability": "...",
  "estimated_difficulty": 8,
  "difficulty": "hard",
  "citations": ["..."],
  "thought_process": "...",
  "chunk_ids": ["..."],
  "chunks": ["..."],
  "llm_call_id": "...",
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

# 七、Claude Code 推荐执行顺序

## Step 1：Hard prompt 路由

修改文件：

```text
agents/qa_agent/planner.py
agents/qa_agent/executor.py
agents/qa_agent/generator.py
```

目标：

```text
EVOLVE_TO_HARDER 不再被 planner 返回
hard difficulty 使用 qa_hard_generation_v1 / mcq_hard_generation_v1
```

完成后跑一次最小 e2e。

---

## Step 2：修复 candidate_pool 字段丢失

修改文件：

```text
agents/qa_agent/state.py
agents/qa_agent/executor.py
```

目标：

```text
CandidateRecord 保存 prompt metadata
CandidateRecord 保存 execution metadata
CandidateRecord 保存 raw_item
accepted 和 rejected 都不丢字段
```

完成后跑一次最小 e2e，检查 `candidate_pool.json`。

---

## Step 3：清理冗余字段

修改文件：

```text
benchforge/schemas.py 或 GenerationBatch 定义文件
agents/qa_agent/executor.py
agents/qa_agent/planner.py
agents/qa_agent/feedback.py
agents/qa_agent/state.py
```

目标：

```text
删除 remaining_count / requested_min_questions / requested_target_questions
删除 evidence_strategy / expand_docs / evolve_source_count
最后删除 evolve 相关代码
```

完成后跑静态检查和 e2e。

---

# 八、最终验收标准

修改完成后必须满足：

```text
1. hard 轮不再进入题目进化逻辑；
2. hard 轮实际使用 hard prompt；
3. 不新增 hard filter；
4. 不修改 prompt 内容；
5. candidate_pool 保存 prompt 输出字段；
6. candidate_pool 保存 llm_call_id、chunks、raw_item；
7. MCQ 的 options / correct_answer alias 不再丢失；
8. rejected 题目也保留 raw_item；
9. 删除没有执行落点的 batch 数量字段；
10. e2e 能正常跑完。
```

最关键的两个检查点：

```text
round_plan.jsonl:
  strategy = hard_generate

candidate_pool.json:
  estimated_difficulty / citations / question_type / required_capability / raw_item 不再丢失
```

---

# 九、注意事项

## 1. 不要第一版就大规模物理删除

第一版优先禁用：

```text
EVOLVE_TO_HARDER
```

而不是立刻删除所有相关代码。  
确认 e2e 通过后，再逐步删除相关字段和函数。

---

## 2. 不要删除 chunk 检索核心参数

以下字段不能删除：

```text
mode
difficulty
topics
single_k
multi_k
target_candidates_per_topic
single_chunk_ids
multi_chunk_ids
prompt_template_id
```

其中：

```text
single_k / multi_k
```

控制 chunk 数量和 multi-chunk 偏好。

```text
difficulty
```

既影响 evidence sampling，也影响 hard prompt 路由。

---

## 3. 不要加 hard filter

当前版本不做：

```text
if target_difficulty == hard and generated_difficulty != hard:
    reject
```

只通过日志观察 hard prompt 的实际效果。

---

## 4. 不要修改 prompt 内容

当前只修改代码路由，不改：

```text
qa_system_prompt.md
qa_hard_system_prompt.md
qa_user_prompt.md
mcq_system_prompt.md
mcq_hard_system_prompt.md
mcq_user_prompt.md
```

---

## 5. 一定要保留 raw_item

即使已经把当前 prompt 字段映射到 `CandidateRecord`，仍然要保存：

```python
raw_item: dict[str, Any]
```

这样以后 prompt 新增字段时，不会再次出现“中途丢失”。

---

# 十、最终结果预期

修改后的题目生成链路应为：

```text
planner
  hard_gap / too_easy_ratio 触发 HARD_GENERATE
  difficulty = hard
  计算 topics / single_k / multi_k / target_candidates_per_topic

executor
  sample_chunks(topic, mode, difficulty, single_k, multi_k)
  make GenerationBatch
  hard difficulty -> qa_hard_generation_v1 / mcq_hard_generation_v1

generator
  qa_hard_generation_v1 -> qa_hard_system_prompt.md + qa_user_prompt.md
  mcq_hard_generation_v1 -> mcq_hard_system_prompt.md + mcq_user_prompt.md

filter
  只做结构过滤
  不做 hard reject

state
  保存完整 prompt metadata
  保存 execution metadata
  保存 raw_item

feedback
  统计 accepted / rejected / difficulty / topic
  不再统计 evolve
```

一句话总结：

> 当前版本的核心修复不是增加更复杂的进化机制，而是让 hard 轮真正调用 hard prompt，并确保生成结果的完整元数据被保存下来。
