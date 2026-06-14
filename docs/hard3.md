# BenchForge QA Agent 修复方案（2026-06-14）

## 目标

本次修复解决以下 7 个核心问题：

1. max_tokens 配置未真实生效
2. multi_chunk units 数量过少
3. LLM 生成题目被 remaining_quota 大量截断
4. CandidatePool 未保存完整题目字段
5. trace_call_id / llm_call_id 链路不完整
6. Evolution 残留逻辑与日志 schema
7. chunked.jsonl 改为 chunked.json

---

# 1. max_tokens 配置未真实生效

## 问题

当前：

```yaml
model:
  max_tokens: 1024
```

虽然被读取：

```python
model_ref.max_tokens
```

但未写回：

```python
model_client
```

Generator 实际调用：

```python
max_tokens=getattr(model_client, "max_tokens", 2000)
```

因此 yaml 配置可能失效。

日志中出现：

```text
Generated=9897
Generated=10198
```

说明底层推理后端实际未受到 1024 限制。

---

## 修改文件

```text
agents/qa_agent/agent.py
agents/qa_agent/generator.py
所有 ModelClient 实现
```

---

## 修改方案

### agent.py

模型加载后立即注入配置：

```python
model_client = _resolve_model_client_fn(
    model_ref.name,
    registry_path,
)

setattr(model_client, "model_name", model_ref.name)
setattr(model_client, "temperature", model_ref.temperature)
setattr(model_client, "max_tokens", model_ref.max_tokens)
setattr(model_client, "max_retries", model_ref.max_retries)
```

---

### Generator.generate()

保留：

```python
max_tokens=getattr(model_client, "max_tokens", 2000)
```

---

### ModelClient.complete()

确保：

```python
backend.generate(
    ...
    max_new_tokens=max_tokens,
)
```

或：

```python
chat.completions.create(
    ...
    max_tokens=max_tokens,
)
```

真正传给后端。

---

## 日志增强

llm_calls.jsonl 增加：

```json
{
  "requested_max_tokens": 1024,
  "actual_completion_tokens": 998,
  "finish_reason": "stop"
}
```

---

## 验收

不再出现：

```text
Generated=9897
Generated=10198
```

除非显式配置超大 token。

---

# 2. multi_chunk units 数量过少

## 问题

当前 multi_chunk 数量约：

```text
multi ≈ single / h_max
```

严重低于理论组合空间。

---

## 修改文件

```text
utils/multi_chunk.py
agents/qa_agent/evidence_manager.py
config/qa_agent.yaml
```

---

## 修改方案 A

重写：

```python
calculate_yourbench_target_count()
```

核心逻辑：

```python
theoretical = Σ C(n,h)

doc_target = min(
    theoretical,
    n * num_multihops_factor,
)
```

其中：

```python
h ∈ [h_min, h_max]
```

---

## 修改方案 B

修改：

```python
n_combinations
```

当前：

```python
n // combinations_per_doc_factor
```

改为：

```python
n * combinations_per_doc_factor
```

---

## 修改方案 C

扩展检索后：

不要只使用新增 chunk。

改成：

```python
old_single_chunks
+
new_single_chunks
```

重新构建 multi pool。

---

## 推荐配置

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

---

## 验收

例如：

```text
single_chunks=20
```

得到：

```text
multi_chunks≈40
```

而不是：

```text
5
```

---

# 3. LLM 生成题目被 remaining_quota 截断

## 问题

当前：

```python
parsed_questions = parsed_questions[:remaining_quota]
```

导致：

```text
LLM生成10题
只差2题
最终保存2题
浪费8题
```

---

## 修改文件

```text
agents/qa_agent/executor.py
```

---

## 修改方案

删除：

```python
remaining_quota = ...
```

相关截断逻辑。

---

## 新逻辑

本轮：

```text
LLM生成多少
保存多少
```

即：

```python
parsed_questions = res["parsed_questions"]
```

直接进入：

```python
update_mode_state(...)
```

---

## Candidate Target 语义修改

原来：

```text
最大容量
```

改成：

```text
最小目标
```

---

## 停止条件

保持：

```python
candidate_pool_count >= target
```

但：

```text
停止下一轮
```

而不是：

```text
裁掉本轮
```

---

## 验收

目标：

```text
12
```

当前：

```text
10
```

本轮生成：

```text
8
```

最终：

```text
18
```

而不是：

```text
12
```

---

# 4. CandidatePool 保存完整字段

## 问题

部分字段未进入 CandidateRecord。

---

## 修改文件

```text
agents/qa_agent/state.py
agents/qa_agent/executor.py
```

---

## CandidateRecord

新增：

```python
question_type
required_capability
estimated_difficulty

citations
thought_process

llm_call_id
trace_call_id

chunks
generation_round

raw_item
```

---

## normalize_item_for_record

禁止丢弃未知字段。

只做：

```python
alias转换
```

例如：

```python
correct_answer
→ answer
```

---

## 验收

candidate_pool.json 中每条记录包含：

```json
{
  "question_type": "...",
  "required_capability": "...",
  "estimated_difficulty": "...",

  "citations": [],
  "thought_process": "...",

  "llm_call_id": "...",
  "trace_call_id": "...",

  "raw_item": {}
}
```

---

# 5. trace_call_id / llm_call_id 链路补全

## 问题

当前：

```text
candidate_pool
↓
只能回查 llm_call_id
```

无法回查：

```text
llm_calls.jsonl
```

中的完整 trace。

---

## 修改文件

```text
utils/llm_tracer.py
agents/qa_agent/generator.py
agents/qa_agent/executor.py
agents/qa_agent/state.py
```

---

## 定义

```text
trace_call_id
=
BenchForge内部trace id

llm_call_id
=
底层模型返回id
```

---

## Generator

返回：

```python
return {
    ...
    "trace_call_id": trace_call_id,
    "llm_call_id": llm_call_id,
}
```

---

## CandidateRecord

新增：

```python
trace_call_id: str | None
```

---

## 验收

可以通过：

```text
candidate.trace_call_id
```

直接定位：

```text
llm_calls.jsonl.call_id
```

查看原始 prompt。

---

# 6. Evolution 残留逻辑清理

## 问题

当前仍保留：

```text
EVOLVE_TO_HARDER
EVOLVED
evolved_count
```

虽然实际已停用。

---

## 修改文件

```text
agents/qa_agent/executor.py
agents/qa_agent/state.py
agents/qa_agent/planner.py
agents/qa_agent/feedback.py
```

---

## 删除

```python
RoundStrategy.EVOLVE_TO_HARDER
```

```python
CandidateStatus.EVOLVED
```

```python
ModeState.evolved_count
```

```python
_execute_evolve_round()
```

---

## 删除日志字段

```text
evolved_count
```

---

## 验收

以下文件不再出现：

```text
EVOLVED
EVOLVE_TO_HARDER
evolved_count
```

包括：

```text
round_feedback.jsonl
mode_state.json
generation_report.json
```

---

# 7. chunked.jsonl → chunked.json

## 问题

当前：

```text
chunked.jsonl
```

与：

```text
single_units.json
multi_units.json
```

格式不统一。

---

## 修改文件

```text
agents/qa_agent/evidence_manager.py
```

---

## 新格式

```json
{
  "Python": [
    {
      "document_id": "...",
      "document_title": "...",
      "document_url": "...",

      "document_summary": "...",

      "chunks": [
        {
          "chunk_id": "...",
          "chunk_index": 0,
          "chunk_text": "..."
        }
      ]
    }
  ]
}
```

---

## 新函数

新增：

```python
_merge_chunked_json()
```

负责：

```text
读取
合并
去重
写回
```

---

## 兼容逻辑

优先：

```text
chunked.json
```

否则：

```text
chunked.jsonl
```

自动转换。

---

## 验收

最终目录：

```text
evidence/

chunked.json
single_units.json
multi_units.json
```

不再生成：

```text
chunked.jsonl
```

---

# 推荐执行顺序

## 第一批（最高优先级）

```text
1. max_tokens真实生效
3. 删除remaining_quota截断
4. CandidatePool保存完整字段
5. trace_call_id链路
```

---

## 第二批

```text
2. multi_chunk数量修复
7. chunked.json统一
```

---

## 第三批

```text
6. Evolution残留清理
```

---

# 预期收益

修复后：

```text
LLM生成题目利用率:
30% → 90%+

multi_chunk数量:
5 → 40+

trace可追踪性:
部分缺失 → 完整链路

日志结构:
历史遗留 → 简洁一致
```

