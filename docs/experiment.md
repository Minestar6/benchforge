# AdaptiveQAAgent 第一阶段有效性验证实验方案 V2

## 1. 实验目标

本阶段不追求论文级统计显著性，而是验证 AdaptiveQAAgent 核心机制是否有效运行，并初步验证其价值。

核心研究问题：

### RQ1

AdaptiveQAAgent 是否比直接生成获得更高质量的数据？

---

### RQ2

多轮反馈机制是否真正改善生成结果？

---

### RQ3

难度自适应机制是否能够修复 Hard Question 不足问题？

---

### RQ4

Feedback → Decision → Action → State Update 闭环是否真正发生？

---

## 2. 实验配置

### 2.1 Topics

采用两个主题：

```yaml
topics:
  - topic_a
  - topic_b
```

原因：

* 保留 Topic Gap 机制
* 控制实验复杂度
* 避免单 Topic 导致 Topic 控制失效

---

### 2.2 Question Types

```yaml
qa:
  count: 5

mcq:
  count: 5
```

目标：

```text
QA Accepted = 5

MCQ Accepted = 5

Total Accepted = 10
```

---

### 2.3 Difficulty Distribution

```yaml
easy: 0.2

medium: 0.5

hard: 0.3
```

对应：

### QA

```text
easy = 1

medium = 2~3

hard = 1~2
```

### MCQ

```text
easy = 1

medium = 2~3

hard = 1~2
```

---

### 2.4 Multi-Round Configuration

```yaml
qa:
  max_rounds: 10

mcq:
  max_rounds: 10
```

目的：

`count` 降低后 `remaining` 始终偏小，每轮生成量减少，配合拉长 `max_rounds`，保证多轮运行，能够观察反馈闭环。

说明：

每轮生成量由 `remaining = target_candidates - accepted_count` 动态推算，代码中无 `per_round_generate_num` 字段，不可配置。

---

### 2.5 Random Seed

```yaml
seed: 1
```

仅用于流程验证。

---

# 3. 实验分组

---

## Group A：Direct Generation

### 流程

```text
Evidence
    ↓
Generate
    ↓
Save
```

特点：

```text
无 Verifier

无 Feedback

无 Adaptive Loop
```

---

## Group B：Single-round Agent

### 流程

```text
Generate
    ↓
Verify
    ↓
Accept / Reject
```

配置：

```yaml
max_rounds: 1
```

特点：

```text
有 Verifier

无 Feedback Loop
```

---

## Group C：Multi-round Agent (No Difficulty Adaptation)

### 保留

```text
retrieve_more

expand_evidence

topic_gap

feedback_mapper
```

### 关闭

```text
hard_gap

hard_multihop_gap

evolution
```

建议：

通过提高阈值使规则永不触发，无需新增配置字段：

```yaml
# Group C 的 adaptive_config (decision 部分)
decision:
  hard_gap_threshold: 1.0        # hard_gap 最大值 <= target_hard_ratio < 1，永不触发
  too_easy_ratio: 1.0            # feedback ratio 最大为 1，永不触发
```

---

### 流程

```text
Generate
    ↓
Verify
    ↓
Feedback
    ↓
Decision
    ↓
Generate
```

---

## Group D：Full AdaptiveQAAgent

启用全部机制：

```text
retrieve_more

expand_evidence

topic_gap

hard_gap

hard_multihop_gap

evolution
```

完整闭环：

```text
Generate
    ↓
Verify
    ↓
Feedback
    ↓
Decision
    ↓
Action
    ↓
Generate
```

---

# 4. 主结果指标（Main Results）

用于回答：

```text
RQ1
RQ2
RQ3
```

---

## 4.1 Acceptance Rate ↑

定义：

```text
accepted_questions
/
generated_questions
```

意义：

```text
单位生成成本下能够获得多少有效题目
```

越高越好。

---

## 4.2 Hard Ratio ↑

定义：

```text
accepted_hard
/
accepted_total
```

意义：

```text
最终 Hard Question 占比
```

越接近目标越好。

---

## 4.3 Hard Gap ↓

定义：

```text
target_hard_ratio
-
actual_hard_ratio
```

例如：

```text
target = 0.3

actual = 0.2

hard_gap = 0.1
```

意义：

```text
难度控制误差
```

越小越好。

---

## 4.4 Topic Balance ↓

定义：

```text
max(topic_count)
-
min(topic_count)
```

例如：

```text
Topic A = 9

Topic B = 1

Balance = 8
```

意义：

```text
Topic 分布是否均衡
```

越小越好。

---

# 5. 机制分析指标（Mechanism Analysis）

用于回答：

```text
RQ4
```

即：

```text
Agent 为什么有效
```

---

## 5.1 Too Easy Rate ↓

定义：

```text
too_easy
/
generated
```

意义：

```text
生成结果中被判定过于简单的比例
```

用于分析：

```text
evolution 是否发挥作用
```

---

## 5.2 Not Multi-Hop Rate ↓

定义：

```text
not_multihop
/
generated
```

意义：

```text
复杂推理能力是否提升
```

用于分析：

```text
hard_multihop_gap 是否有效
```

---

## 5.3 Grounding Failure Rate ↓

定义：

```text
answer_not_grounded
/
generated
```

意义：

```text
答案是否真正来自证据
```

用于分析：

```text
retrieve_more

expand_evidence
```

是否有效。

---

## 5.4 Action Trigger Count

统计：

```text
retrieve_more

expand_evidence

generate_hard

evolution
```

触发次数。

意义：

```text
系统是否真正使用了这些动作
```

---

## 5.5 Action Success Rate

定义：

```text
成功修复次数
/
动作触发次数
```

例如：

```text
generate_hard:

trigger = 10

success = 7

success_rate = 70%
```

意义：

```text
动作是否真正有效
```

---

# 6. 效率指标（Efficiency）

---

## 6.1 Generated ↓

定义：

```text
总生成题目数
```

意义：

```text
达到目标所需生成量
```

越少越好。

---

## 6.2 Rounds ↓

定义：

```text
达到目标数量所需轮次
```

意义：

```text
收敛速度
```

越少越好。

---

# 7. 主结果表

| Method | Accept Rate ↑ | Hard Ratio ↑ | Hard Gap ↓ | Topic Balance ↓ |
| ------ | ------------- | ------------ | ---------- | --------------- |
| Direct |               |              |            |                 |
| Single |               |              |            |                 |
| NoDiff |               |              |            |                 |
| Full   |               |              |            |                 |

---

# 8. Feedback Analysis

| Method | Too Easy ↓ | Not Multi-Hop ↓ | Grounding Failure ↓ |
| ------ | ---------- | --------------- | ------------------- |
| Direct |            |                 |                     |
| Single |            |                 |                     |
| NoDiff |            |                 |                     |
| Full   |            |                 |                     |

---

# 9. Efficiency Analysis

| Method | Generated ↓ | Rounds ↓ |
| ------ | ----------- | -------- |
| Direct |             |          |
| Single |             |          |
| NoDiff |             |          |
| Full   |             |          |

---

# 10. 成功标准

认为第一阶段实验成功需满足：

### S1

四组实验全部运行完成。

---

### S2

Single-round 仅运行一轮。

---

### S3

Multi-round 至少运行两轮。

---

### S4

NoDiff 不触发：

```text
hard_gap

hard_multihop_gap

evolution
```

---

### S5

Full 至少触发：

```text
hard_gap

hard_multihop_gap

evolution
```

中的一种。

---

### S6

日志中能够观察到：

```text
Feedback
    ↓
Decision
    ↓
Action
    ↓
State Update
```

完整链路。

---

满足以上条件后进入第二阶段正式实验：

```yaml
topics: 5

qa:
  count: 50

mcq:
  count: 50

seeds:
  - 1
  - 2
  - 3
```
