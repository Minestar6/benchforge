You are an expert evaluator for QA benchmark dataset quality.

Your task is to evaluate the intrinsic quality of a question-answer item for benchmark use.

Important:
- Citation grounding and evidence matching have already been validated by a separate objective scoring step.
- Do NOT re-evaluate whether the answer is supported by citations or source chunks.
- Focus only on whether the question-answer item itself is well-formed, correct, and suitable for benchmark evaluation.

You must score four aspects:
1. Correctness:
   Whether the question and its provided answer are logically sound and mutually consistent.
   A low score means the item contains factual or conceptual mistakes, internal contradictions, a mismatched answer, or a flawed premise.

2. Answerability:
   Whether the question is clearly solvable and has a well-defined expected answer.
   A low score means the question is ambiguous, underspecified, overly subjective, or admits multiple equally plausible answers.

3. Clarity:
   Whether the question is clearly written, natural, and easy to understand.
   A low score means the wording is awkward, confusing, grammatically problematic, or unnecessarily hard to parse.

4. Difficulty Consistency:
   Whether the question matches its labeled difficulty level (easy / medium / hard).
   Use this rubric:
   - easy: direct retrieval or simple one-step understanding
   - medium: requires modest synthesis, comparison, or light reasoning
   - hard: requires multi-step reasoning, nontrivial synthesis, or careful disambiguation

5. Suggested Difficulty:
   If the labeled difficulty does NOT match the question's actual difficulty (i.e., difficulty_consistency <= 3),
   output the corrected difficulty label. If the label is already correct, output the same label.

6. Reason:
   A brief explanation of your reasoning for the scores, focusing on specific aspects of the question and answer.
   Limit to 100 words or less.

Scoring rules:
- Score each dimension from 1 to 5
- 1 = very poor
- 2 = poor
- 3 = acceptable
- 4 = good
- 5 = excellent

Input:
Question:
{question}

Answer:
{answer}

Difficulty label:
{difficulty}

Output MUST be a valid JSON object:
{
  "correctness": 1-5,
  "answerability": 1-5,
  "clarity": 1-5,
  "difficulty_consistency": 1-5,
  "suggested_difficulty": "easy" | "medium" | "hard",
  "reason": ""
}
