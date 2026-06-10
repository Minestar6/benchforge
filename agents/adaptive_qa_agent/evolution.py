import json
from pathlib import Path
from typing import TYPE_CHECKING

from agents.verify_agent.schema import QuestionCandidate

if TYPE_CHECKING:
    from .state import CandidateRef


class EvolutionTool:
    """将简单题进化为更难版本。"""

    def __init__(self, model_client, blueprint, adaptive_config, prompts, validation_dir: Path | None = None):
        self.model_client = model_client
        self.blueprint = blueprint
        self.adaptive_config = adaptive_config
        self.prompts = prompts
        self.validation_dir = validation_dir

    async def evolve(self, candidates: list["CandidateRef"], mode: str) -> list[QuestionCandidate]:
        originals = self._recover_candidates(candidates)
        if not originals:
            return []

        results = []
        for cand in originals:
            evolved = await self._evolve_one(cand, mode)
            if evolved:
                results.append(evolved)
        return results

    def _recover_candidates(self, refs: list["CandidateRef"]) -> list[QuestionCandidate]:
        if not self.validation_dir:
            return []
        jsonl_path = self.validation_dir / "validated_questions.jsonl"
        if not jsonl_path.exists():
            return []

        ids = {r.question_id for r in refs}
        found = []
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("question_id") in ids:
                        cand_data = rec.get("candidate", {})
                        found.append(QuestionCandidate.model_validate(cand_data))
                except Exception:
                    continue
        return found

    async def _evolve_one(self, cand: QuestionCandidate, mode: str) -> QuestionCandidate | None:
        gen_cfg = self.adaptive_config.generation
        model = gen_cfg.model or getattr(self.model_client, "model_name", None)

        prompt = f"""You have a simple question that needs to be evolved into a harder, multi-hop version.

Original Question: {cand.question}
Original Answer: {cand.answer}
Topic: {cand.topic}

Create a harder version that requires multi-step reasoning. Return JSON:
[{{"question": "...", "answer": "...", "estimated_difficulty": "hard"}}]"""

        try:
            response = await self.model_client.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=gen_cfg.temperature,
                max_tokens=1024,
            )
            items = json.loads(response["text"]) if response["text"].strip().startswith("[") else [json.loads(response["text"])]
            if not items or "question" not in items[0]:
                return None
            item = items[0]
            return QuestionCandidate(
                question_id=f"{cand.question_id}_evolved",
                task_id=cand.task_id,
                run_id=cand.run_id,
                topic=cand.topic,
                question=item["question"],
                answer=item["answer"],
                question_mode=mode,
                estimated_difficulty=item.get("estimated_difficulty", "hard"),
                chunk_ids=cand.chunk_ids,
            )
        except Exception:
            return None
