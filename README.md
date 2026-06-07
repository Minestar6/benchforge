# QuestionGeneratorAgent 使用文档

## 概述

`QuestionGeneratorAgent` 是基于设计文档（2026-05-21）重构的问题生成智能体，实现了状态机驱动的多主题题目生成。

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -e .

# 设置环境变量
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
```

### 2. 创建生成计划

```python
from benchforge.schemas import GenerationPlan, QuestionModeTarget

plan = GenerationPlan(
    run_id="run_example_001",
    goal="生成历史与产业制度相关评测题",
    topics=["Fordism", "Taylorism"],
    mode_targets={
        "multiple_choice": QuestionModeTarget(
            count=6,
            difficulty_distribution={"easy": 0.25, "medium": 0.5, "hard": 0.25},
        ),
        "qa": QuestionModeTarget(
            count=4,
            difficulty_distribution={"easy": 0.125, "medium": 0.5, "hard": 0.375},
        ),
    },
    max_rounds_per_topic=3,
    max_total_rounds=8,
    language="en",
)
```

### 3. 执行生成

```bash
# 直接运行示例
python example_question_generator.py
```

或使用 Python 代码：

```python
import asyncio
from benchforge.agents.question_generator import QuestionGeneratorAgent
from benchforge.models import OpenAIClient

async def main():
    model_client = OpenAIClient(
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
    )

    agent = QuestionGeneratorAgent(model_client=model_client)
    report = await agent.execute(plan)

    print(f"状态: {report.status}")
    print(f"完成: {report.final_counts}")

asyncio.run(main())
```

## 架构说明

### 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| 数据模型 | `benchforge/schemas/core.py` | 定义所有核心数据结构 |
| 信号计算 | `benchforge/utils/signals.py` | 规则特征计算 |
| 多 chunk 构建 | `benchforge/utils/multi_chunk.py` | 多证据单元构建 |
| 采样策略 | `benchforge/utils/sampling.py` | 广度探索和缺口驱动采样 |
| 轻量过滤 | `benchforge/utils/filter.py` | 结构级过滤 |
| 计划工具 | `benchforge/utils/planning.py` | 计划编译和状态更新 |
| 主代理 | `benchforge/agents/question_generator.py` | 状态机和调度逻辑 |

### 执行流程

```
1. 计划编译
   ├─ 展开 mode × difficulty 目标
   ├─ 按主题均分
   └─ 初始化 TopicState

2. 主题串行执行
   ├─ 准备证据
   │  ├─ 检索文档
   │  ├─ 生成摘要
   │  ├─ 切分 chunk
   │  └─ 构建证据池
   └─ 多轮生成
      ├─ 识别主缺口
      ├─ 选择采样策略
      ├─ 生成题目
      └─ 更新状态

3. 全局补题（可选）
   └─ 选择缺口最大主题继续生成

4. 输出报告和结果
```

### 证据池管理

**单证据单元（SingleChunkUnit）**：
- 每个独立的文本片段
- 计算三种分数：mcq_score、qa_score、hard_score
- 维护使用次数用于去重

**多证据单元（MultiChunkUnit）**：
- 同文档内的 2-3 个连续片段组合
- 适合生成高难度题目
- 智能组合基于信号分数

### 采样策略

**广度探索采样（BroadExplorationSampling）**：
- 优先低使用、高基础分数的单元
- 用于主题首轮或无明显缺口时

**缺口驱动采样（GapDrivenSampling）**：
- 根据 target_mode 和 target_difficulty 调整权重
- hard 题优先多 chunk 单元
- 用于填补特定缺口

## 输出说明

### 输出目录结构

```
runs/{run_id}/
├── source_documents.jsonl       # 检索的源文档
├── single_chunk_pool_{topic}.jsonl  # 单证据单元
├── multi_chunk_pool_{topic}.jsonl   # 多证据单元
└── accepted_questions.jsonl     # 生成的题目
```

### 生成报告

`GenerationReport` 包含：
- `final_counts`: 最终完成数量
- `remaining_gaps`: 剩余缺口
- `topic_states`: 各主题详细状态
- `status`: completed / partial

## 配置选项

### 生成计划参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `run_id` | 运行 ID | 必填 |
| `goal` | 生成目标描述 | 必填 |
| `topics` | 主题列表 | 必填 |
| `mode_targets` | 模式和难度目标 | 必填 |
| `max_rounds_per_topic` | 单主题最大轮次 | 3 |
| `max_total_rounds` | 全局最大轮次 | 8 |
| `language` | 语言 | "en" |
| `retrieval_policy` | 检索策略 | "wikipedia_first" |

### 难度分布说明

`difficulty_distribution` 的键必须是 `easy`、`medium`、`hard`，值必须为浮点数且总和为 1.0。

```python
difficulty_distribution={"easy": 0.25, "medium": 0.5, "hard": 0.25}
```

## 常见问题

**Q: 如何只生成选择题？**

```python
mode_targets={
    "multiple_choice": QuestionModeTarget(count=10, difficulty_distribution={...})
}
```

**Q: 如何调整检索文档数量？**

使用配置文件或设置环境变量。

**Q: 如何控制生成的题目质量？**

- 调整 `max_rounds_per_topic` 增加轮次
- 使用更高性能的模型
- 调整难度分布

## 技术文档

完整设计文档位于：`docs/superpowers/specs/2026-05-21-question-generator-agent-design.md`