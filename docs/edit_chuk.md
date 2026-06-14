# qa_agent chunk 修改方案：YourBench 风格 multi_chunk 生成与扩充 evidence 持久化

## 1. 修改目标

本方案只处理 qa_agent 中与 evidence / chunk 相关的两个问题：

1. **multi_units 数量过少**
   - 当前初始构建时使用 `target_count=min(10, len(single_units)//2)`。
   - 当前扩充检索时使用 `target_count=min(5, len(single_units)//2)`。
   - 这会导致 multi_units 数量明显偏少，hard QA 或多跳题生成时容易复用少量组合。

2. **EXPAND_EVIDENCE 后新增 evidence 没有持久化**
   - 当前扩充检索后，只把新增的 `single_units` / `multi_units` 追加到内存中的 `evidence_pool`。
   - 新增的 chunks、single_units、multi_units 没有写回 `runs/{task_id}/{run_id}/evidence/`。
   - 后续 verify_agent 如果只根据 `candidate_pool.json` 中的 `chunk_ids` 回查 evidence，可能找不到扩充检索产生的 chunk 正文。

本方案暂时不修改：

- `round_plan.single_k / multi_k`
- 题目生成 prompt
- candidate 字段保存逻辑
- sampler 的 single/multi 严格执行问题

---

## 2. 目标效果

### 2.1 multi_units 数量机制

从当前固定上限：

```text
初始 evidence:
  multi_units <= min(10, len(single_units)//2)

扩充检索:
  multi_units <= min(5, len(single_units)//2)
```

改为接近 YourBench 的机制：

```text
按每篇文档的 chunk 数量动态生成 multi_units
由 h_min / h_max / num_multihops_factor 控制组合数量
再由 max_units_per_topic / max_units_per_expansion 控制全局上限
```

### 2.2 扩充检索 evidence 持久化

每次 `EXPAND_EVIDENCE` 后，新增内容写回：

```text
runs/{task_id}/{run_id}/evidence/chunked.jsonl
runs/{task_id}/{run_id}/evidence/single_units.json
runs/{task_id}/{run_id}/evidence/multi_units.json
```

这样后续阶段可以通过 `chunk_ids` 回查 evidence。

---

## 3. 需要修改的文件

建议修改以下文件：

```text
config/qa_agent.yaml
agents/qa_agent/schema.py
agents/qa_agent/config_loader.py
agents/qa_agent/agent.py
utils/multi_chunk.py
agents/qa_agent/evidence_manager.py
```

---

## 4. 修改 `config/qa_agent.yaml`

新增配置段：

```yaml
multi_chunk:
  h_min: 2
  h_max: 5
  num_multihops_factor: 1
  max_units_per_topic: 80
  max_units_per_expansion: 40
  combinations_per_doc_factor: 1
```

字段含义：

| 字段 | 含义 |
|---|---|
| `h_min` | 每个 multi unit 至少包含几个 chunk |
| `h_max` | 每个 multi unit 最多包含几个 chunk |
| `num_multihops_factor` | YourBench 风格数量缩放因子，越小生成越多 |
| `max_units_per_topic` | 初始 evidence 构建时每个 topic 最多保留多少 multi_units |
| `max_units_per_expansion` | 每次扩充检索最多新增多少 multi_units |
| `combinations_per_doc_factor` | 当前 MultiChunkBuilder 内部已有参数，建议设为 1，使每篇文档产生更多候选组合 |

---

## 5. 修改 `agents/qa_agent/schema.py`

新增配置 dataclass：

```python
from dataclasses import dataclass

@dataclass
class MultiChunkConfig:
    h_min: int = 2
    h_max: int = 5
    num_multihops_factor: int = 1
    max_units_per_topic: int = 80
    max_units_per_expansion: int = 40
    combinations_per_doc_factor: int = 1
```

如果 `schema.py` 已经集中定义 qa_agent config dataclass，则将该类放在其他 config class 附近即可。

---

## 6. 修改 `agents/qa_agent/config_loader.py`

### 6.1 import 新配置类

将 `MultiChunkConfig` 加入 import：

```python
from benchforge.agents.qa_agent.schema import (
    AgentConfig,
    CandidatePoolConfig,
    InitialBreadthConfig,
    PlannerConfig,
    ChunkMixConfig,
    ChunkMixDifficulty,
    ModeAdjustment,
    GenerationYield,
    ChunkLimitsForMode,
    ChunkKLimit,
    RuntimeConfig,
    DecisionConfig,
    MultiChunkConfig,
)
```

### 6.2 修改返回值签名

原本返回：

```python
def load_qa_agent_config(
    path: str | Path,
) -> tuple[AgentConfig, ModelRef, RetrievalConfig, ChunkingConfig, SummarizationChunkingConfig]:
```

改为：

```python
def load_qa_agent_config(
    path: str | Path,
) -> tuple[
    AgentConfig,
    ModelRef,
    RetrievalConfig,
    ChunkingConfig,
    SummarizationChunkingConfig,
    MultiChunkConfig,
]:
```

### 6.3 解析 YAML 中的 multi_chunk

在函数内增加：

```python
multi_chunk_cfg = MultiChunkConfig(**raw.get("multi_chunk", {}))
```

### 6.4 修改 return

原本：

```python
return (
    agent_config,
    model_ref,
    retrieval_cfg,
    chunking_cfg,
    sum_chunking_cfg,
)
```

改为：

```python
return (
    agent_config,
    model_ref,
    retrieval_cfg,
    chunking_cfg,
    sum_chunking_cfg,
    multi_chunk_cfg,
)
```

---

## 7. 修改 `agents/qa_agent/agent.py`

### 7.1 接收新的配置返回值

原本：

```python
agent_config, model_ref, retrieval_cfg, chunking_cfg, sum_chunking_cfg = (
    load_qa_agent_config(config_path)
)
```

改为：

```python
agent_config, model_ref, retrieval_cfg, chunking_cfg, sum_chunking_cfg, multi_chunk_cfg = (
    load_qa_agent_config(config_path)
)
```

### 7.2 扩展 `_EvidenceConfig`

原本可能类似：

```python
class _EvidenceConfig:
    __slots__ = ("retrieval", "chunking", "summarization_chunking")

    def __init__(self, retrieval, chunking, summarization_chunking):
        self.retrieval = retrieval
        self.chunking = chunking
        self.summarization_chunking = summarization_chunking
```

改为：

```python
class _EvidenceConfig:
    __slots__ = (
        "retrieval",
        "chunking",
        "summarization_chunking",
        "multi_chunk",
    )

    def __init__(self, retrieval, chunking, summarization_chunking, multi_chunk):
        self.retrieval = retrieval
        self.chunking = chunking
        self.summarization_chunking = summarization_chunking
        self.multi_chunk = multi_chunk
```

### 7.3 创建 evidence_config 时传入 multi_chunk_cfg

原本：

```python
evidence_config = _EvidenceConfig(
    retrieval_cfg,
    chunking_cfg,
    sum_chunking_cfg,
)
```

改为：

```python
evidence_config = _EvidenceConfig(
    retrieval_cfg,
    chunking_cfg,
    sum_chunking_cfg,
    multi_chunk_cfg,
)
```

---

## 8. 修改 `utils/multi_chunk.py`

在 `MultiChunkBuilder` 中新增 YourBench 风格的 target_count 计算方法：

```python
def calculate_yourbench_target_count(
    self,
    single_chunks: list[SingleChunkUnit],
    h_min: int = 2,
    h_max: int = 5,
    num_multihops_factor: int = 1,
    max_units: int | None = None,
) -> int:
    """
    按 YourBench 风格计算 multi-chunk 数量。

    设计原则：
    - 不枚举所有组合；
    - 按每篇文档的 chunk 数生成受控数量；
    - 最终使用 max_units 控制全局上限。
    """
    if not single_chunks:
        return 0

    chunks_by_doc: dict[str, list[SingleChunkUnit]] = {}
    for chunk in single_chunks:
        chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

    total = 0
    factor = max(1, int(num_multihops_factor))

    for _doc_id, doc_chunks in chunks_by_doc.items():
        n = len(doc_chunks)
        if n < h_min:
            continue

        effective_h_max = min(h_max, n)

        # YourBench 风格基础数量：按文档 chunk 数缩放
        doc_target = max(1, n // factor)

        # 防止每组过大时生成过多无效组合
        if doc_target * effective_h_max > n:
            doc_target = max(1, n // effective_h_max)

        total += doc_target

    if max_units is not None:
        total = min(total, int(max_units))

    return max(0, total)
```

说明：

- 该方法只负责计算目标数量。
- 具体组合生成仍然复用现有 `build_multi_chunk_units_smart()`。
- 这样改动最小，不需要重写 multi-chunk 构造逻辑。

---

## 9. 修改 `agents/qa_agent/evidence_manager.py`

### 9.1 新增 import

在文件顶部增加：

```python
import json
from pathlib import Path
from typing import Any
```

如果这些 import 已经存在，不要重复添加。

### 9.2 在 `EvidenceManager` 类中新增 helper 方法

```python
def _multi_chunk_cfg(self) -> Any:
    return getattr(self.config, "multi_chunk", None)


def _resolve_multi_target_count(
    self,
    single_units: list[Any],
    *,
    expansion: bool = False,
) -> int:
    mcfg = self._multi_chunk_cfg()

    h_min = getattr(mcfg, "h_min", 2)
    h_max = getattr(mcfg, "h_max", 5)
    num_multihops_factor = getattr(mcfg, "num_multihops_factor", 1)

    max_units = getattr(
        mcfg,
        "max_units_per_expansion" if expansion else "max_units_per_topic",
        40 if expansion else 80,
    )

    return self.multi_chunk_builder.calculate_yourbench_target_count(
        single_chunks=single_units,
        h_min=h_min,
        h_max=h_max,
        num_multihops_factor=num_multihops_factor,
        max_units=max_units,
    )


def _build_multi_units(
    self,
    single_units: list[Any],
    doc_summaries: dict[str, str],
    *,
    expansion: bool = False,
) -> list[Any]:
    mcfg = self._multi_chunk_cfg()

    h_min = getattr(mcfg, "h_min", 2)
    h_max = getattr(mcfg, "h_max", 5)
    combinations_per_doc_factor = getattr(mcfg, "combinations_per_doc_factor", 1)

    target_count = self._resolve_multi_target_count(
        single_units,
        expansion=expansion,
    )

    return self.multi_chunk_builder.build_multi_chunk_units_smart(
        single_units,
        doc_summaries,
        target_count=target_count,
        h_min=h_min,
        h_max=h_max,
        combinations_per_doc_factor=combinations_per_doc_factor,
    )
```

### 9.3 修改初始 evidence 构建逻辑

在 `_build_evidence_pool()` 中，将原逻辑：

```python
multi_units = self.multi_chunk_builder.build_multi_chunk_units_smart(
    single_units,
    doc_summaries,
    target_count=min(10, len(single_units) // 2),
)
```

替换为：

```python
multi_units = self._build_multi_units(
    single_units,
    doc_summaries,
    expansion=False,
)
```

### 9.4 修改扩充检索中的 multi_units 构建逻辑

在 `expand_retrieval()` 中，将原逻辑：

```python
multi_units = self.multi_chunk_builder.build_multi_chunk_units_smart(
    single_units,
    doc_summaries,
    target_count=min(5, len(single_units) // 2),
)
```

替换为：

```python
multi_units = self._build_multi_units(
    single_units,
    doc_summaries,
    expansion=True,
)
```

---

## 10. 扩充检索后持久化 evidence

### 10.1 在 `EvidenceManager` 中新增持久化方法

```python
def _append_expanded_evidence_to_disk(
    self,
    topic: str,
    chunks: list[Any],
    single_units: list[Any],
    multi_units: list[Any],
) -> None:
    """
    将 EXPAND_EVIDENCE 新增的 chunks / single_units / multi_units
    持久化到 runs/{task_id}/{run_id}/evidence。

    这样后续 verify_agent 可以通过 chunk_ids 回查 evidence。
    """
    if not hasattr(self.config, "get_resolved_output_path"):
        return

    evidence_dir = Path(self.config.get_resolved_output_path()) / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # 1. append chunked.jsonl
    chunks_by_doc: dict[str, list[Any]] = {}
    for chunk in chunks:
        chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

    chunked_rows: list[dict[str, Any]] = []

    for doc_id, doc_chunks in chunks_by_doc.items():
        doc_chunks_sorted = sorted(
            doc_chunks,
            key=lambda c: getattr(c, "chunk_index", 0),
        )
        source_doc = self.documents.get(doc_id)

        chunked_rows.append(
            {
                "document_id": doc_id,
                "topic": topic,
                "document_title": source_doc.title if source_doc else "",
                "document_url": source_doc.url if source_doc else "",
                "document_text": source_doc.content if source_doc else "",
                "document_summary": self.document_summaries.get(doc_id, ""),
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "chunk_text": c.text,
                    }
                    for c in doc_chunks_sorted
                ],
                "source": "expand_retrieval",
            }
        )

    if chunked_rows:
        with open(evidence_dir / "chunked.jsonl", "a", encoding="utf-8") as f:
            for row in chunked_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 2. merge single_units.json
    single_path = evidence_dir / "single_units.json"
    if single_path.exists():
        with open(single_path, "r", encoding="utf-8") as f:
            single_data = json.load(f)
    else:
        single_data = {}

    single_data.setdefault(topic, [])
    single_data[topic].extend(
        [
            {
                "chunk_id": u.chunk_id,
                "document_id": u.document_id,
                "text": getattr(u, "text", ""),
                "qa_score": u.qa_score,
                "mcq_score": u.mcq_score,
                "hard_score": u.hard_score,
                "source": "expand_retrieval",
            }
            for u in single_units
        ]
    )

    with open(single_path, "w", encoding="utf-8") as f:
        json.dump(single_data, f, ensure_ascii=False, indent=2)

    # 3. merge multi_units.json
    multi_path = evidence_dir / "multi_units.json"
    if multi_path.exists():
        with open(multi_path, "r", encoding="utf-8") as f:
            multi_data = json.load(f)
    else:
        multi_data = {}

    multi_data.setdefault(topic, [])
    multi_data[topic].extend(
        [
            {
                "unit_id": u.unit_id,
                "document_id": u.document_id,
                "chunk_ids": list(u.chunk_ids),
                "texts": list(getattr(u, "texts", [])),
                "qa_score": u.qa_score,
                "mcq_score": u.mcq_score,
                "hard_score": u.hard_score,
                "source": "expand_retrieval",
            }
            for u in multi_units
        ]
    )

    with open(multi_path, "w", encoding="utf-8") as f:
        json.dump(multi_data, f, ensure_ascii=False, indent=2)
```

### 10.2 在 `expand_retrieval()` 中保存新 evidence

在 `expand_retrieval()` 中，新增 chunks / units 构建完并追加到内存 evidence_pool 后，增加：

```python
self._append_expanded_evidence_to_disk(
    topic=topic,
    chunks=all_chunks,
    single_units=single_units,
    multi_units=multi_units,
)
```

推荐位置：

```python
# 扩展到现有证据池
if evidence_pool:
    evidence_pool.single_chunks.extend(single_units)
    evidence_pool.multi_chunks.extend(multi_units)

# 新增：持久化扩充 evidence
self._append_expanded_evidence_to_disk(
    topic=topic,
    chunks=all_chunks,
    single_units=single_units,
    multi_units=multi_units,
)

return ExpandResult(
    new_chunks=len(all_chunks),
    new_single_units=len(single_units),
    new_multi_units=len(multi_units),
)
```

---

## 11. 补充修复：扩充检索时保存 document 对象

当前初始 `prepare_evidence()` 通常会执行：

```python
self.documents[document.document_id] = document
```

但扩充检索里可能没有保存新 document。为了让 `_append_expanded_evidence_to_disk()` 能写出 `document_title / document_url / document_text`，需要在 `expand_retrieval()` 中 fetch document 成功后增加：

```python
self.documents[document.document_id] = document
```

建议位置：

```python
document = await self.fetch_wikipedia_page(...)

if document.status.value == "failed":
    continue

self.documents[document.document_id] = document
```

---

## 12. 去重注意事项

扩充检索可能重复抓取已经存在的文档或 chunk。为了避免 evidence 文件无限重复，建议后续增加去重逻辑：

```python
existing_chunk_ids = {
    unit.chunk_id
    for unit in evidence_pool.single_chunks
}
```

然后对 `single_units` 做过滤：

```python
single_units = [
    u for u in single_units
    if u.chunk_id not in existing_chunk_ids
]
```

multi_units 可以根据组合去重：

```python
existing_multi_combos = {
    tuple(sorted(u.chunk_ids))
    for u in evidence_pool.multi_chunks
}

multi_units = [
    u for u in multi_units
    if tuple(sorted(u.chunk_ids)) not in existing_multi_combos
]
```

本轮最小修改可以先不做，但建议后续加上。

---

## 13. 验收标准

修改完成后，执行一次包含 `EXPAND_EVIDENCE` 的 qa_agent 运行，检查：

### 13.1 multi_units 数量

检查：

```text
runs/{task_id}/{run_id}/evidence/multi_units.json
```

预期：

- 每个 topic 下的 multi_units 数量不再固定最多 10。
- 扩充检索后对应 topic 下会新增 multi_units。
- 数量受 `max_units_per_topic` 和 `max_units_per_expansion` 控制。

### 13.2 扩充 evidence 是否落盘

检查：

```text
runs/{task_id}/{run_id}/evidence/chunked.jsonl
runs/{task_id}/{run_id}/evidence/single_units.json
runs/{task_id}/{run_id}/evidence/multi_units.json
```

预期：

- `chunked.jsonl` 中出现 `"source": "expand_retrieval"` 的记录。
- `single_units.json` 中出现 `"source": "expand_retrieval"` 的记录。
- `multi_units.json` 中出现 `"source": "expand_retrieval"` 的记录。

### 13.3 chunk_id 可回查

从 `candidate_pool.json` 中找一个扩充检索后生成题目的 `chunk_ids`，确认这些 `chunk_id` 可以在以下文件中查到：

```text
evidence/chunked.jsonl
evidence/single_units.json
evidence/multi_units.json
```

### 13.4 日志一致性

检查 `generation_report.json` 或 `round_feedback.jsonl`：

- `new_multi_units` 数量应随检索内容变化。
- 不应始终固定为 5 或 10 附近。
- hard QA 场景下 multi_units 可用数量应明显增加。

---

## 14. 总结

本次修改的核心是：

```text
1. multi_units 数量不再使用固定的 min(10, single//2) 或 min(5, single//2)
2. 改为 YourBench 风格：按每篇文档 chunk 数、h_min/h_max、num_multihops_factor 动态计算
3. 扩充检索后的 chunks / single_units / multi_units 写回 evidence 目录
4. 保证 verify_agent 或后续评估可以通过 chunk_ids 回查完整 evidence
```

该方案只处理 chunk / evidence 层，不改变题目生成 prompt，不改变 round_plan 的 single_k / multi_k，不改变 sampler 执行逻辑。
