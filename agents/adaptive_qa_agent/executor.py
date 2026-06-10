import asyncio
import json

from agents.verify_agent.schema import QuestionCandidate


class Executor:
    def __init__(self, evidence_manager, model_client, evolution_tool,
                 blueprint, config, adaptive_config, prompts):
        self.evidence_manager = evidence_manager
        self.model_client = model_client
        self.evolution_tool = evolution_tool
        self.blueprint = blueprint
        self.config = config
        self.adaptive_config = adaptive_config
        self.prompts = prompts
        self.sem = asyncio.Semaphore(adaptive_config.execution.concurrency)

    async def run(self, action, state) -> list[QuestionCandidate]:
        topics = action.topics or state.active_topics

        if action.action_type == "retrieve_more":
            await self._concurrent(topics, self._retrieve, action, state)
            return await self._concurrent(topics, self._generate, action, state)

        if action.action_type == "expand_evidence":
            await self._concurrent(topics, self._expand, action, state)
            return await self._concurrent(topics, self._generate, action, state)

        if action.action_type == "evolve":
            return await self._evolve(state)

        return await self._concurrent(topics, self._generate, action, state)

    async def _concurrent(self, items, fn, action, state) -> list[QuestionCandidate]:
        async def bounded(item):
            async with self.sem:
                return await fn(item, action, state)
        results = await asyncio.gather(*[bounded(i) for i in items])
        return [q for batch in results for q in batch]

    async def _generate(self, topic, action, state) -> list[QuestionCandidate]:
        difficulty = action.difficulty or "medium"
        evidence_pool = self.evidence_manager.evidence_pools.get(topic)
        if not evidence_pool:
            return []

        batch = self.evidence_manager.sample(
            evidence_pool=evidence_pool,
            topic=topic,
            target_mode=state.mode,
            target_difficulty=difficulty,
            prefer_multi_chunk=(action.evidence_strategy == "multi_group"),
            round_num=state.round_in_mode + 1,
            remaining=max(1, state.target_candidates - state.accepted_count),
        )
        if not batch or not (batch.single_chunk_ids or batch.multi_chunk_ids):
            return []

        chunk_ids = list(batch.single_chunk_ids or []) + list(batch.multi_chunk_ids or [])
        if state.is_chunk_combination_used(chunk_ids):
            return []
        state.record_chunk_combination(chunk_ids)

        evidence_text = self.evidence_manager.get_evidence_text(batch, evidence_pool)
        document_summary = self.evidence_manager.get_document_summary(batch, evidence_pool)

        messages = self.prompts.build_generation_prompt(
            topic=topic,
            difficulty=difficulty,
            mode=state.mode,
            evidence_text=evidence_text,
            document_summary=document_summary,
            language=self.blueprint.language,
            constraints=action.evidence_strategy,
        )

        gen_cfg = self.adaptive_config.generation
        model = gen_cfg.model or getattr(self.model_client, "model_name", None)
        response = await self.model_client.complete(
            model=model,
            messages=messages,
            temperature=gen_cfg.temperature,
            max_tokens=gen_cfg.max_tokens,
        )

        return self._parse_response(response["text"], topic, state, batch)

    def _parse_response(self, text, topic, state, batch) -> list[QuestionCandidate]:
        try:
            items = json.loads(text) if text.strip().startswith("[") else [json.loads(text)]
        except json.JSONDecodeError:
            return []

        candidates = []
        chunk_ids = list(batch.single_chunk_ids or []) + list(batch.multi_chunk_ids or [])
        for i, item in enumerate(items):
            if not isinstance(item, dict) or "question" not in item or "answer" not in item:
                continue
            candidates.append(QuestionCandidate(
                question_id=f"{self.blueprint.run_id}_{state.mode}_{topic}_{state.round_in_mode}_{i}",
                task_id=self.blueprint.task_id,
                run_id=self.blueprint.run_id,
                topic=topic,
                question=item["question"],
                answer=item["answer"],
                question_mode=state.mode,
                estimated_difficulty=item.get("difficulty", "medium"),
                chunk_ids=chunk_ids,
            ))
        return candidates

    async def _retrieve(self, topic, action, state) -> list[QuestionCandidate]:
        evidence_pool = self.evidence_manager.evidence_pools.get(topic)
        await self.evidence_manager.expand_retrieval(
            topic=topic,
            queries=[topic],
            language=self.blueprint.language,
            run_id=self.blueprint.run_id,
            evidence_pool=evidence_pool,
        )
        return []

    async def _expand(self, topic, action, state) -> list[QuestionCandidate]:
        evidence_pool = self.evidence_manager.evidence_pools.get(topic)
        await self.evidence_manager.expand_retrieval(
            topic=topic,
            queries=[f"{topic} details", f"{topic} related"],
            language=self.blueprint.language,
            run_id=self.blueprint.run_id,
            evidence_pool=evidence_pool,
        )
        return []

    async def _evolve(self, state) -> list[QuestionCandidate]:
        if not self.evolution_tool:
            return []
        easy_pool = [c for c in state.candidates
                     if c.status == "accepted" and c.difficulty in ("easy", "medium")]
        if not easy_pool:
            return []
        return await self.evolution_tool.evolve(easy_pool[:5], state.mode)
