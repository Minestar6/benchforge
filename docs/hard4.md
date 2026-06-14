# BenchForge develop1 QA Agent 七项问题修复计划

> 适用分支：`develop1`  
> 目标：修复 QA Agent 当前实现中 max_tokens 不生效、multi_chunk 数量过少、题目被截断浪费、candidate_pool 字段缺失、LLM 调用链路不可回查、evolution 残留、chunked.jsonl 格式不统一等问题。  
> 建议执行方式：按本文“执行顺序”分批提交，避免一次提交改动过大导致难以定位回归问题。

---

## 0. 当前问题总览

当前 develop1 分支仍存在以下 7 个问题：

1. `max_tokens` 配置未真实生效，LLM 调用时可能仍使用默认或后端超大上限。
2. `multi_chunk units` 数量过少，当前实现把 multi 数量压成约 `single / h_max`。
3. LLM 生成题目被 `remaining_quota` 大量截断，导致生成成本浪费。
4. 题目字段没有完整保存到 `candidate_pool.json`。
5. `llm_calls.jsonl` 中同时有内部 `call_id` 和响应里的 `llm_call_id`，但 `candidate_pool` 没有保存可直接回查 trace 的 ID。
6. 日志 schema 和代码中仍有 evolution 残留。
7. `chunked.jsonl` 应改为统一的 `chunked.json`。

---

## 1. 修复 `max_tokens` 配置未真实生效

### 1.1 问题说明

`config/qa_agent.yaml` 中配置了：

```yaml
model:
  max_tokens: 1024
```

`config_loader.py` 已经能读取到 `model_ref.max_tokens`，但 `agents/qa_agent/agent.py` 加载模型后，没有把 `model_ref.max_tokens` 写入 `model_client`。

而 `agents/qa_agent/generator.py` 当前调用逻辑通常是：

```python
response = await model_client.complete(
    prompt=prompt,
    temperature=getattr(model_client, "temperature", 0.7),
    max_tokens=getattr(model_client, "max_tokens", 2000),
)
```

因此如果 `model_client` 上没有 `max_tokens` 属性，就会退回默认 `2000`。如果底层 client 又没有把 `max_tokens` 继续传给后端，就可能出现日志中的：

```text
Generated=9897
Generated=10198
```

说明实际生成 token 远超预期。

### 1.2 修改文件

```text
agents/qa_agent/agent.py
agents/qa_agent/generator.py
具体 ModelClient 实现文件，例如 models / llm / utils 中的 client 类
config/qa_agent.yaml
```

### 1.3 修改方案

#### 1.3.1 在 `agent.py` 加载模型后注入运行参数

找到模型加载逻辑：

```python
model_client = _resolve_model_client_fn(model_ref.name, registry_path)
```

改为：

```python
model_client = _resolve_model_client_fn(model_ref.name, registry_path)

setattr(model_client, "model_name", model_ref.name)
setattr(model_client, "temperature", model_ref.temperature)
setattr(model_client, "max_tokens", model_ref.max_tokens)
setattr(model_client, "max_retries", model_ref.max_retries)
```

如果 `model_ref.max_tokens` 可能为 `None`，则使用安全默认值：

```python
if model_ref.max_tokens is not None:
    setattr(model_client, "max_tokens", model_ref.max_tokens)
```

#### 1.3.2 确保 `Generator.generate()` 真实传参

保留或改为显式变量：

```python
temperature = getattr(model_client, "temperature", 0.7)
max_tokens = getattr(model_client, "max_tokens", 2000)

response = await model_client.complete(
    prompt=prompt,
    temperature=temperature,
    max_tokens=max_tokens,
)
```

#### 1.3.3 检查底层 `ModelClient.complete()`

必须确保参数继续传到底层后端。

OpenAI 兼容接口：

```python
response = await client.chat.completions.create(
    model=self.model_name,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens,
)
```

本地推理框架常见参数：

```python
response = backend.generate(
    prompt,
    temperature=temperature,
    max_new_tokens=max_tokens,
)
```

注意：很多本地后端使用 `max_new_tokens`，不是 `max_tokens`。

### 1.4 日志增强

在 `llm_calls.jsonl` 里记录真实请求参数：

```json
{
  "requested_temperature": 0.7,
  "requested_max_tokens": 1024,
  "actual_prompt_tokens": 5230,
  "actual_completion_tokens": 998,
  "finish_reason": "stop"
}
```

如果底层 response 没有 usage 信息，也至少记录：

```json
{
  "requested_max_tokens": 1024
}
```

### 1.5 验收标准

运行一次 e2e 后检查：

1. `llm_calls.jsonl` 每条记录都有 `requested_max_tokens`。
2. `requested_max_tokens` 等于 `qa_agent.yaml` 中的值。
3. 底层日志不应再出现明显超过配置的生成量，例如 `Generated=9897`，除非配置本身就是 10000。
4. 如果仍超过，需要继续检查底层 client 是否忽略了 `max_tokens/max_new_tokens`。

---

## 2. 修复 `multi_chunk units` 数量过少

### 2.1 问题说明

当前 `utils/multi_chunk.py` 的目标数计算逻辑仍然类似：

```python
doc_target = max(1, n // factor)

if doc_target * effective_h_max > n:
    doc_target = max(1, n // effective_h_max)
```

这会把 multi units 数量压成约：

```text
multi_units ≈ single_chunks / h_max
```

例如单文档 20 个 chunk，`h_max=4`，最后可能只有 5 个 multi units。

但如果允许 `h_min=2, h_max=4`，同文档理论组合空间应该是：

```text
C(n,2) + C(n,3) + C(n,4)
```

例如 `n=20`：

```text
C(20,2)=190
C(20,3)=1140
C(20,4)=4845
total=6175
```

不需要全部生成，但应该从这个空间中采样足够数量，而不是压缩成 `n/4`。

### 2.2 修改文件

```text
utils/multi_chunk.py
agents/qa_agent/evidence_manager.py
config/qa_agent.yaml
```

### 2.3 修改方案

#### 2.3.1 修改 target count 计算

在 `utils/multi_chunk.py` 中引入：

```python
from math import comb
```

重写或修改 `calculate_yourbench_target_count()`：

```python
from math import comb

def calculate_yourbench_target_count(
    single_chunks,
    h_min: int = 2,
    h_max: int = 4,
    num_multihops_factor: int = 2,
    max_units: int | None = None,
) -> int:
    if not single_chunks:
        return 0

    chunks_by_doc: dict[str, list] = {}
    for chunk in single_chunks:
        chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

    total = 0
    factor = max(1, int(num_multihops_factor))

    for _, doc_chunks in chunks_by_doc.items():
        n = len(doc_chunks)
        if n < h_min:
            continue

        upper_h = min(h_max, n)
        theoretical = sum(comb(n, h) for h in range(h_min, upper_h + 1))

        # 关键：按 single 数量放大，而不是按 h_max 压缩
        doc_target = min(theoretical, n * factor)
        total += doc_target

    if max_units is not None:
        total = min(total, int(max_units))

    return max(0, total)
```

如果当前函数是 class method，请保留 `self` 参数。

#### 2.3.2 修改候选组合尝试数

找到类似逻辑：

```python
n_combinations = max(1, n // combinations_per_doc_factor)
```

改成：

```python
n_combinations = min(
    target_count,
    max(1, n * combinations_per_doc_factor),
)
```

为避免随机重复导致凑不满，可把采样循环放大：

```python
max_attempts = max(n_combinations * 20, target_count * 20)

for _ in range(max_attempts):
    ...
    if len(units) >= target_count:
        break
```

#### 2.3.3 expansion 时用“旧 + 新”重建 multi pool

当前扩展检索可能只用新增 chunks 构建 multi，导致 expansion 的 multi 仍很少。

在 `agents/qa_agent/evidence_manager.py` 中，扩展检索生成 `new_single_units` 后，改成：

```python
new_single_units = build_evidence_pool_from_chunks(
    all_chunks,
    topic,
    document_summary="",
)

if evidence_pool:
    all_single_units_for_multi = evidence_pool.single_chunks + new_single_units
else:
    all_single_units_for_multi = new_single_units

multi_units = self._build_multi_units(
    all_single_units_for_multi,
    self.document_summaries,
    expansion=True,
)

if evidence_pool:
    evidence_pool.single_chunks.extend(new_single_units)
    evidence_pool.multi_chunks = multi_units
else:
    evidence_pool = EvidencePool(
        topic=topic,
        single_chunks=new_single_units,
        multi_chunks=multi_units,
    )
```

注意：只要 `build_multi_chunk_units_smart()` 内部仍然按 `document_id` 分组，就不会跨文档。

### 2.4 配置建议

```yaml
multi_chunk:
  enabled: true
  h_min: 2
  h_max: 4
  num_multihops_factor: 2
  combinations_per_doc_factor: 2
  max_units_per_topic: 300
  max_units_per_expansion: 300
```

### 2.5 验收标准

如果某 topic 下单文档有：

```text
single_chunks=20
```

则 multi units 应接近：

```text
20 * 2 = 40
```

不应再是 5 左右。

检查：

```text
evidence/multi_units.json
generation_report.json
round_feedback.jsonl
```

---

## 3. 修复 LLM 生成题目被 `remaining_quota` 大量截断

### 3.1 问题说明

当前 `agents/qa_agent/executor.py` 中存在类似逻辑：

```python
remaining_quota = target - mode_state.accepted_count

if remaining_quota <= 0:
    parsed_questions = []
elif len(parsed_questions) > remaining_quota:
    parsed_questions = parsed_questions[:remaining_quota]
```

这会导致：

```text
目标还差 2 题
LLM 本轮生成并解析出 10 题
最终只保存 2 题
其余 8 题直接浪费
```

这与当前设计目标不一致。`target_candidate_count` 应该是最小候选题数量，而不是最大容量。

### 3.2 修改文件

```text
agents/qa_agent/executor.py
agents/qa_agent/state.py
agents/qa_agent/feedback.py
agents/qa_agent/storage.py
```

### 3.3 修改方案

#### 3.3.1 删除提前截断

删除：

```python
remaining_quota = ...
parsed_questions = parsed_questions[:remaining_quota]
```

改为：

```python
parsed_questions = res["parsed_questions"]
```

本轮 LLM 生成并通过过滤的题目全部进入 `candidate_pool`。

#### 3.3.2 保持停止下一轮的逻辑

停止条件应该是：

```python
if mode_state.accepted_count >= target_candidate_count:
    stop_reason = "candidate_pool_sufficient"
```

但含义是：

```text
本轮已保存所有生成题；
如果当前候选池已达到最小目标；
则不再启动下一轮 LLM 调用。
```

不要在本轮内部裁掉题目。

#### 3.3.3 round feedback 中记录超额情况

增加字段：

```json
{
  "target_candidate_count": 12,
  "accepted_count_after_round": 18,
  "over_target_count": 6
}
```

可选实现：

```python
over_target_count = max(0, mode_state.accepted_count - target_candidate_count)
```

### 3.4 验收标准

假设：

```text
target=12
before=10
current_round_valid=8
```

修复后：

```text
after=18
```

不是：

```text
after=12
```

且不会触发下一轮 LLM 调用。

---

## 4. 修复 CandidatePool 字段保存不完整

### 4.1 问题说明

当前内存中的 `CandidateRecord` 可能已经新增了部分字段，但 `storage.py` 的序列化逻辑仍可能只保存基础字段，导致 `candidate_pool.json` 中看不到完整字段。

需要确保：

1. accepted 题完整保存。
2. rejected 题也完整保存，包括拒绝原因和原始字段。
3. `raw_item` 永远保存原始 LLM item，避免字段丢失。

### 4.2 修改文件

```text
agents/qa_agent/state.py
agents/qa_agent/executor.py
agents/qa_agent/storage.py
utils/filter.py
```

### 4.3 修改方案

#### 4.3.1 CandidateRecord 字段

确保 `CandidateRecord` 至少包含：

```python
from dataclasses import dataclass, field
from typing import Any

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

    question_mode: str | None = None
    question_type: str | None = None
    required_capability: str | None = None
    estimated_difficulty: int | float | str | None = None
    citations: list[Any] = field(default_factory=list)
    thought_process: str | None = None

    llm_call_id: str | None = None
    trace_call_id: str | None = None
    chunks: list[str] = field(default_factory=list)
    generation_round: int | None = None

    raw_item: dict[str, Any] = field(default_factory=dict)
```

新增字段必须有默认值，保证旧 candidate_pool 可兼容加载。

#### 4.3.2 normalize_item_for_record

在 `executor.py` 中增加或检查：

```python
def normalize_item_for_record(q: dict, fallback_mode: str) -> dict:
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

原则：只做 alias 标准化，不删除未知字段。

#### 4.3.3 storage.py 序列化完整字段

如果当前 `_serialize()` 手写字段，应改成完整输出。

示例：

```python
def serialize_candidate(record: CandidateRecord) -> dict:
    return {
        "question_id": record.question_id,
        "question": record.question,
        "answer": record.answer,
        "topic": record.topic,
        "difficulty": record.difficulty,
        "status": record.status.value if hasattr(record.status, "value") else record.status,
        "source_round": record.source_round,
        "source_strategy": record.source_strategy,
        "chunk_ids": record.chunk_ids,

        "reject_reason": record.reject_reason,
        "parent_question_id": record.parent_question_id,
        "choices": record.choices,

        "question_mode": record.question_mode,
        "question_type": record.question_type,
        "required_capability": record.required_capability,
        "estimated_difficulty": record.estimated_difficulty,
        "citations": record.citations,
        "thought_process": record.thought_process,

        "llm_call_id": record.llm_call_id,
        "trace_call_id": record.trace_call_id,
        "chunks": record.chunks,
        "generation_round": record.generation_round,

        "raw_item": record.raw_item,
    }
```

### 4.4 验收标准

`candidate_pool.json` 中每条 accepted/rejected 都应包含：

```json
{
  "question_type": "...",
  "required_capability": "...",
  "estimated_difficulty": "...",
  "citations": [],
  "thought_process": "...",
  "llm_call_id": "...",
  "trace_call_id": "...",
  "chunks": [],
  "generation_round": 1,
  "raw_item": {}
}
```

---

## 5. 修复 `call_id / llm_call_id` 链路不完整

### 5.1 问题说明

当前有两套 ID：

```text
call_id      = BenchForge LLMTracer 内部 span id
llm_call_id  = 底层模型服务返回 id
```

二者来源不同是合理的。

但是 `candidate_pool` 没有保存对应 `llm_calls.jsonl.call_id` 的字段，导致：

```text
candidate_pool item
→ 无法直接定位 llm_calls.jsonl 中完整 prompt/response
```

### 5.2 修改文件

```text
utils/llm_tracer.py
agents/qa_agent/generator.py
agents/qa_agent/executor.py
agents/qa_agent/state.py
agents/qa_agent/storage.py
```

### 5.3 字段定义

统一命名：

```text
trace_call_id = BenchForge 内部 trace id，对应 llm_calls.jsonl.call_id
llm_call_id   = 底层模型服务返回 id，例如 OpenAI/vLLM/SGLang request id
```

### 5.4 修改方案

#### 5.4.1 Generator 返回两个 ID

当前 `generator.generate()` 返回 `llm_call_id`。

需要增加 `trace_call_id`，但更推荐由 executor 负责创建 span 并传入。

示例：

```python
async def generate(..., trace_call_id: str | None = None):
    response = await model_client.complete(...)
    llm_call_id = response.get("llm_call_id")
    ...
    return raw_items, diagnostics, llm_call_id, trace_call_id
```

#### 5.4.2 executor.py 中保存 span.call_id

当前可能有：

```python
async with tracer.span(...) as span:
    ...
```

或：

```python
span = tracer.span(...)
```

需要显式保存：

```python
trace_call_id = None

if tracer is not None:
    span = tracer.span(...)
    trace_call_id = span.call_id
    async with span:
        raw_items, diagnostics, llm_call_id = await generator.generate(...)
else:
    raw_items, diagnostics, llm_call_id = await generator.generate(...)
```

然后给每道题写入：

```python
for q in parsed_questions:
    q["trace_call_id"] = trace_call_id
    q["llm_call_id"] = llm_call_id
```

过滤失败也写：

```python
filter_failures.append({
    ...
    "trace_call_id": trace_call_id,
    "llm_call_id": llm_call_id,
})
```

#### 5.4.3 CandidateRecord 和 storage 保存

见第 4 节，添加并序列化：

```python
trace_call_id: str | None = None
llm_call_id: str | None = None
```

### 5.5 验收标准

任意候选题：

```json
{
  "trace_call_id": "llm_xxx",
  "llm_call_id": "chatcmpl_xxx"
}
```

可以通过：

```text
candidate.trace_call_id == llm_calls.jsonl.call_id
```

直接找到对应 prompt、response、token usage、latency。

---

## 6. 清理 Evolution 残留逻辑与日志 schema

### 6.1 问题说明

当前代码中仍有以下残留：

```text
RoundStrategy.EVOLVE_TO_HARDER
CandidateStatus.EVOLVED
ModeState.evolved_count
ModeState.evolvable_surplus()
RoundFeedback.evolved_count
_execute_evolve_round()
```

虽然题目进化策略已停用，但日志和 schema 仍保留，会造成维护混乱。

### 6.2 修改文件

```text
agents/qa_agent/state.py
agents/qa_agent/planner.py
agents/qa_agent/executor.py
agents/qa_agent/feedback.py
agents/qa_agent/storage.py
tests/
```

### 6.3 修改方案

#### 6.3.1 删除枚举

删除：

```python
CandidateStatus.EVOLVED
RoundStrategy.EVOLVE_TO_HARDER
```

如果担心旧数据加载报错，可在加载旧文件时做兼容映射：

```python
if status == "evolved":
    status = "accepted"
```

或者：

```python
if status == "evolved":
    status = "rejected"
    reject_reason = "legacy_evolved_status"
```

建议映射为 accepted 仅用于旧日志兼容，不再新写出。

#### 6.3.2 删除 state 字段

删除：

```python
ModeState.evolved_count
ModeState.evolvable_surplus()
```

更新所有引用。

#### 6.3.3 删除 executor evolve 分支

删除：

```python
_execute_evolve_round()
```

删除任何分支：

```python
if strategy == RoundStrategy.EVOLVE_TO_HARDER:
    ...
```

#### 6.3.4 删除 feedback 输出

删除：

```python
RoundFeedback.evolved_count
build_round_feedback(... evolved_seed_count ...)
```

新日志不再输出：

```text
evolved_count
```

### 6.4 验收标准

新 run 的以下文件中不再出现：

```text
evolved_count
EVOLVED
EVOLVE_TO_HARDER
evolve_to_harder
```

检查文件：

```text
round_feedback.jsonl
mode_state.json
generation_report.json
candidate_pool.json
```

---

## 7. 将 `chunked.jsonl` 改为 `chunked.json`

### 7.1 问题说明

当前 evidence 输出包含：

```text
chunked.jsonl
single_units.json
multi_units.json
```

`chunked.jsonl` 与其他 evidence 文件格式不统一，不利于整体读取、回查和调试。

### 7.2 修改文件

```text
agents/qa_agent/agent.py
agents/qa_agent/evidence_manager.py
utils/storage 或 EvidenceStore 相关文件
tests/
```

### 7.3 新格式设计

推荐：

```json
{
  "Python": [
    {
      "document_id": "...",
      "document_title": "...",
      "document_url": "...",
      "document_summary": "...",
      "source": "initial_retrieval",
      "chunks": [
        {
          "chunk_id": "...",
          "chunk_index": 0,
          "chunk_text": "..."
        }
      ]
    }
  ],
  "Machine learning": []
}
```

优点：

1. 按 topic 分组。
2. 每个 document 下保留 chunks。
3. 与 `single_units.json`、`multi_units.json` 一样可整体读取。
4. 后续 candidate 的 `chunk_ids` 可以回查原始 chunk。

### 7.4 新增 helper

在 `evidence_manager.py` 或 storage 工具中新增：

```python
def load_chunked_json(evidence_dir: Path) -> dict:
    json_path = evidence_dir / "chunked.json"
    jsonl_path = evidence_dir / "chunked.jsonl"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 兼容旧 jsonl
    if jsonl_path.exists():
        data: dict[str, list[dict]] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                topic = row.get("topic", "unknown")
                data.setdefault(topic, []).append(row)
        return data

    return {}
```

新增合并写入：

```python
def merge_chunked_json(
    evidence_dir: Path,
    topic: str,
    rows: list[dict],
) -> None:
    path = evidence_dir / "chunked.json"
    data = load_chunked_json(evidence_dir)

    data.setdefault(topic, [])

    existing_keys = {
        (
            row.get("document_id"),
            row.get("source", ""),
        )
        for row in data[topic]
    }

    for row in rows:
        key = (
            row.get("document_id"),
            row.get("source", ""),
        )
        if key not in existing_keys:
            data[topic].append(row)
            existing_keys.add(key)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
```

### 7.5 替换旧写入

把：

```python
append_jsonl("chunked.jsonl", rows)
```

替换为：

```python
merge_chunked_json(
    evidence_dir=evidence_dir,
    topic=topic,
    rows=chunked_rows,
)
```

初始 evidence 和 expansion evidence 都要替换。

### 7.6 测试同步

搜索测试中的：

```text
chunked.jsonl
```

全部改为：

```text
chunked.json
```

但保留一个兼容测试：

```text
旧 chunked.jsonl 可以被 load_chunked_json() 读取
```

### 7.7 验收标准

新 run 目录中应有：

```text
evidence/chunked.json
evidence/single_units.json
evidence/multi_units.json
```

不再生成：

```text
evidence/chunked.jsonl
```

旧 run 的 `chunked.jsonl` 仍可被兼容读取。

---

## 8. 推荐执行顺序

### 第一批：直接影响生成成本与题目保存

```text
1. max_tokens 真实生效
3. 删除 remaining_quota 截断
4. CandidatePool 保存完整字段
5. trace_call_id / llm_call_id 链路补全
```

建议提交信息：

```bash
git commit -m "Fix QA generation token config and candidate tracing"
```

### 第二批：修复 evidence / multi_chunk

```text
2. multi_chunk 数量修复
7. chunked.json 格式统一
```

建议提交信息：

```bash
git commit -m "Improve multi-chunk evidence generation and storage"
```

### 第三批：清理历史残留

```text
6. Evolution 残留清理
```

建议提交信息：

```bash
git commit -m "Remove deprecated QA evolution workflow remnants"
```

---

## 9. 最终验收清单

完成后运行一次 e2e，检查：

### 9.1 max_tokens

```text
llm_calls.jsonl 中 requested_max_tokens == qa_agent.yaml model.max_tokens
底层生成 token 不超过配置上限
```

### 9.2 multi_chunk

```text
single_chunks=20 时 multi_units 约为 40
multi_units 不再固定为 5 或 10 附近
```

### 9.3 题目不再被截断

```text
target=12
生成后 candidate_pool 可以是 13、15、18
不强行裁剪回 12
```

### 9.4 candidate_pool 字段完整

每条题包含：

```text
question_type
required_capability
estimated_difficulty
citations
thought_process
llm_call_id
trace_call_id
raw_item
```

### 9.5 trace 可回查

```text
candidate.trace_call_id == llm_calls.jsonl.call_id
```

### 9.6 无 evolution 残留

```text
grep -R "evolved_count\|EVOLVED\|EVOLVE_TO_HARDER" runs/latest
```

应无结果。

### 9.7 evidence 文件统一

```text
evidence/chunked.json
evidence/single_units.json
evidence/multi_units.json
```

不存在：

```text
evidence/chunked.jsonl
```

---

## 10. 风险提示

1. 删除 `remaining_quota` 截断后，candidate_pool 数量可能超过目标，这是预期行为。
2. multi_chunk 数量增加后，采样空间变大，但 prompt 长度也可能增加；需要结合 `sample_chunks()` 控制每次送入 LLM 的 evidence 数量。
3. 删除 evolution 字段可能影响旧 run 加载，建议保留旧字段读取兼容，但新输出不再写。
4. `chunked.json` 改造需要同步测试，否则旧测试会继续找 `chunked.jsonl`。
5. `max_tokens` 修复必须检查到底层 client，而不是只在 `generator.py` 传参。

