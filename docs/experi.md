下面这份文档可以直接交给 Claude Code 执行。

---

# BenchForge QA-Agent Ablation Experiment Design

## Objective

Design a rigorous ablation study to demonstrate the effectiveness of the QA generation framework implemented in `develop1`.

The goal is to separately measure the contribution of:

1. Multi-round generation
2. Feedback-driven planning
3. Difficulty evolution (hard question generation)

rather than only comparing "single-round vs multi-round".

---

# Experimental Groups

## Group A — Direct Generation Baseline

### Purpose

Evaluate the performance of pure LLM generation without the QA-Agent framework.

### Components Disabled

* Planner
* Feedback
* Candidate Pool
* Multi-round execution
* Difficulty evolution
* Topic adaptation

### Workflow

```text
Retrieve Evidence
      ↓
Concatenate Chunks
      ↓
Single LLM Generation
      ↓
Output Questions
```

### Expected Meaning

Represents the simplest baseline.

---

## Group B — Multi-Round Without Feedback

### Purpose

Measure whether merely increasing generation rounds improves quality.

### Components Enabled

* Multi-round execution
* Candidate accumulation

### Components Disabled

* Planner feedback
* Difficulty adaptation
* Hard generation

### Workflow

```text
Round1 NORMAL
Round2 NORMAL
Round3 NORMAL
...
Round10 NORMAL
```

Every round uses identical generation settings.

No information from previous rounds is used.

### Expected Meaning

If B > A:

```text
Multi-round sampling itself contributes.
```

---

## Group C — Feedback Without Difficulty Evolution

### Purpose

Measure the contribution of feedback-driven planning.

### Components Enabled

* Multi-round execution
* Feedback analysis
* Topic coverage adaptation
* Evidence expansion

### Components Disabled

* Hard generation
* Difficulty evolution

### Allowed Planner Strategies

```python
NORMAL_GENERATE
FOCUS_TOPIC
EXPAND_EVIDENCE
```

### Forbidden Strategies

```python
HARD_GENERATE
FOCUS_DIFFICULTY
```

### Expected Meaning

If C > B:

```text
Feedback-driven planning contributes.
```

---

## Group D — Full Method

### Purpose

Evaluate the complete proposed method.

### Components Enabled

* Multi-round execution
* Feedback analysis
* Topic adaptation
* Evidence expansion
* Difficulty evolution
* Hard question generation

### Allowed Planner Strategies

```python
NORMAL_GENERATE
FOCUS_TOPIC
EXPAND_EVIDENCE
HARD_GENERATE
FOCUS_DIFFICULTY
```

### Expected Meaning

If D > C:

```text
Difficulty evolution contributes.
```

---

# Final Comparison Chain

```text
A Direct Generation
    ↓
B Multi-Round No Feedback
    ↓
C Feedback No Difficulty Evolution
    ↓
D Full Method
```

Interpretation:

```text
B − A = gain from multi-round generation

C − B = gain from feedback planning

D − C = gain from difficulty evolution
```

---

# Required Code Refactoring

---

# Step 1: Add Experiment Configuration

File:

```text
benchforge/agents/qa_agent/config.py
```

Add:

```python
@dataclass
class ExperimentConfig:
    name: str = "full"

    enable_feedback: bool = True

    enable_hard_generate: bool = True

    enable_difficulty_adaptation: bool = True

    fixed_strategy: str | None = None

    fixed_difficulty: str | None = None
```

Then add:

```python
experiment: ExperimentConfig = field(
    default_factory=ExperimentConfig
)
```

inside AgentConfig.

---

# Step 2: Modify Planner

File:

```text
benchforge/agents/qa_agent/planner.py
```

Current logic:

```python
hard_gap > threshold
        ↓
HARD_GENERATE
```

must become:

```python
exp_cfg = config.experiment
```

---

## Case 1

No Feedback

```python
if not exp_cfg.enable_feedback:
    return (
        RoundStrategy.NORMAL_GENERATE,
        "no_feedback"
    )
```

This implements Group B.

---

## Case 2

Feedback But No Difficulty Evolution

```python
if (
    exp_cfg.enable_feedback
    and not exp_cfg.enable_difficulty_adaptation
):
```

Allowed:

```python
NORMAL_GENERATE
FOCUS_TOPIC
EXPAND_EVIDENCE
```

Forbidden:

```python
HARD_GENERATE
FOCUS_DIFFICULTY
```

This implements Group C.

---

## Case 3

Full Method

```python
if exp_cfg.enable_hard_generate:
```

retain existing logic:

```python
HARD_GENERATE
```

This implements Group D.

---

# Step 3: Create True Direct Baseline

Current Group A still invokes:

```python
run_generation_agent(...)
```

which is not a true baseline.

A dedicated pipeline should be created.

File:

```text
experiment/qa_agent/direct_generation.py
```

---

## Pseudocode

```python
async def run_direct_generation(
    topic,
    blueprint,
    model_client,
):
```

### Retrieve evidence

```python
evidence = await evidence_manager.prepare_evidence(
    topic,
    blueprint,
)
```

### Merge chunks

```python
context = "\n\n".join(
    chunk.text
    for chunk in evidence.chunks
)
```

### Single prompt

```python
prompt = direct_generation_prompt(
    context=context,
    count=blueprint.target_questions,
)
```

### Generate

```python
questions = await model_client.generate(
    prompt
)
```

### Save

```python
direct_questions.json
```

No planner.

No rounds.

No feedback.

No candidate pool.

---

# Step 4: New Experiment Scripts

Directory:

```text
experiment/qa_agent/groups/
```

---

## group_a_direct.py

```python
experiment.name = "direct"

run_direct_generation(...)
```

---

## group_b_multiround_no_feedback.py

```python
experiment.enable_feedback = False

experiment.enable_hard_generate = False

experiment.enable_difficulty_adaptation = False
```

```python
max_rounds = 10
```

---

## group_c_feedback_no_difficulty.py

```python
experiment.enable_feedback = True

experiment.enable_hard_generate = False

experiment.enable_difficulty_adaptation = False
```

```python
max_rounds = 10
```

---

## group_d_full.py

```python
experiment.enable_feedback = True

experiment.enable_hard_generate = True

experiment.enable_difficulty_adaptation = True
```

```python
max_rounds = 10
```

---

# Step 5: Unified Experiment Runner

File:

```text
experiment/qa_agent/run_all.py
```

---

## Structure

```python
EXPERIMENTS = [
    ("A", run_group_a_direct),
    ("B", run_group_b_no_feedback),
    ("C", run_group_c_feedback),
    ("D", run_group_d_full),
]
```

---

## Execution

```python
for seed in seeds:
    for name, runner in EXPERIMENTS:
        await runner(
            topic=topic,
            seed=seed,
        )
```

Recommended:

```python
seeds = [
    42,
    43,
    44,
    45,
    46,
]
```

Five independent runs.

---

# Step 6: Save Metadata

Each run should produce:

```json
{
  "group": "C",
  "experiment_name": "feedback_no_difficulty",

  "seed": 42,

  "max_rounds": 10,

  "enable_feedback": true,

  "enable_hard_generate": false,

  "enable_difficulty_adaptation": false
}
```

Stored as:

```text
metadata.json
```

---

# Step 7: Metrics

The following metrics should be collected.

---

## Dataset Quality

### Question Count

```text
accepted_count
```

### Acceptance Rate

```text
accept_rate
```

### Diversity

```text
diversity_score
```

### Deduplication

```text
duplicate_ratio
```

### Topic Coverage

```text
topic_coverage
```

### Citation Quality

```text
citation_score
```

---

## Difficulty Metrics

### Easy

```text
easy_ratio
```

### Medium

```text
medium_ratio
```

### Hard

```text
hard_ratio
```

### Multi-chunk Questions

```text
multi_chunk_ratio
```

---

## Efficiency Metrics

### Total Questions Generated

```text
generated_count
```

### Cost

```text
total_tokens
```

### Efficiency

```text
accepted_count / total_tokens
```

---

# Step 8: Statistical Analysis

For each metric:

```text
mean
std
```

across five seeds.

---

## Significance Testing

Recommended:

```python
scipy.stats.ttest_ind
```

Compare:

```text
A vs B
B vs C
C vs D
```

Report:

```text
p-value
```

---

# Expected Claims

After experiments, the paper can support:

### Claim 1

```text
Multi-round generation improves dataset quality.
```

Supported by:

```text
B > A
```

---

### Claim 2

```text
Feedback-driven planning improves coverage and diversity.
```

Supported by:

```text
C > B
```

---

### Claim 3

```text
Difficulty evolution improves hard-question generation.
```

Supported by:

```text
D > C
```

---

# Recommended Final Experimental Setup

```text
A = Direct Generation

B = Multi-Round Generation Without Feedback

C = Multi-Round Feedback Without Difficulty Evolution

D = Full BenchForge Method
```

这是目前 develop1 架构下变量控制最干净、论文说服力最强、代码改动量最小的一套消融实验方案。
