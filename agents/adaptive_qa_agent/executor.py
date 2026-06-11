import asyncio
import json

from agents.verify_agent.schema import QuestionCandidate


class Executor:
    def __init__(self, evidence_manager, model_client, evolution_tool,
                 blueprint, config, adaptive_config, prompts, tracer=None):
        self.evidence_manager = evidence_manager
        self.model_client = model_client
        self.evolution_tool = evolution_tool
        self.blueprint = blueprint
        self.config = config
        self.adaptive_config = adaptive_config
        self.prompts = prompts
        self.tracer = tracer
        self.sem = asyncio.Semaphore(adaptive_config.execution.concurrency)
        self.last_run_details: dict = {}

    async def run(self, action, state) -> list[QuestionCandidate]:
        topics = action.topics or state.active_topics
        self.last_run_details = {
            "action_type": action.action_type,
            "topics": list(topics),
            "steps": [],
            "candidate_ids": [],
            "candidate_count": 0,
        }

        if action.action_type == "retrieve_more":
            _, retrieve_meta = await self._concurrent(topics, self._retrieve, action, state)
            candidates, generate_meta = await self._concurrent(topics, self._generate, action, state)
            self.last_run_details["steps"] = [
                {"step": "retrieve_more", "items": retrieve_meta},
                {"step": "generate", "items": generate_meta},
            ]
            self.last_run_details["candidate_ids"] = [c.question_id for c in candidates]
            self.last_run_details["candidate_count"] = len(candidates)
            return candidates

        if action.action_type == "expand_evidence":
            _, expand_meta = await self._concurrent(topics, self._expand, action, state)
            candidates, generate_meta = await self._concurrent(topics, self._generate, action, state)
            self.last_run_details["steps"] = [
                {"step": "expand_evidence", "items": expand_meta},
                {"step": "generate", "items": generate_meta},
            ]
            self.last_run_details["candidate_ids"] = [c.question_id for c in candidates]
            self.last_run_details["candidate_count"] = len(candidates)
            return candidates

        if action.action_type == "evolve":
            candidates, evolve_meta = await self._evolve(state, action)
            self.last_run_details["steps"] = [{"step": "evolve", "items": [evolve_meta]}]
            self.last_run_details["candidate_ids"] = [c.question_id for c in candidates]
            self.last_run_details["candidate_count"] = len(candidates)
            return candidates

        candidates, generate_meta = await self._concurrent(topics, self._generate, action, state)
        self.last_run_details["steps"] = [{"step": "generate", "items": generate_meta}]
        self.last_run_details["candidate_ids"] = [c.question_id for c in candidates]
        self.last_run_details["candidate_count"] = len(candidates)
        return candidates

    async def _concurrent(self, items, fn, action, state) -> tuple[list[QuestionCandidate], list[dict]]:
        async def bounded(item):
            async with self.sem:
                return await fn(item, action, state)
        results = await asyncio.gather(*[bounded(i) for i in items])
        candidates = [q for batch, _ in results for q in batch]
        meta = [item_meta for _, item_meta in results]
        return candidates, meta

    async def _generate(self, topic, action, state) -> tuple[list[QuestionCandidate], dict]:
        difficulty = action.difficulty or "medium"
        evidence_pool = self.evidence_manager.evidence_pools.get(topic)
        if not evidence_pool:
            return [], {"topic": topic, "status": "missing_evidence_pool", "candidate_count": 0}

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
            return [], {"topic": topic, "status": "empty_batch", "candidate_count": 0}

        chunk_ids = list(batch.single_chunk_ids or []) + list(batch.multi_chunk_ids or [])
        if state.is_chunk_combination_used(chunk_ids):
            return [], {
                "topic": topic,
                "status": "duplicate_chunk_combination",
                "chunk_ids": chunk_ids,
                "candidate_count": 0,
            }
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
        if self.tracer is not None:
            async with self.tracer.span(
                model=model or "unknown",
                provider=getattr(self.model_client, "provider", ""),
                tags={
                    "topic": topic,
                    "mode": state.mode,
                    "round": state.round_in_mode,
                    "action_type": action.action_type,
                },
            ):
                response = await self.model_client.complete(
                    model=model,
                    messages=messages,
                    temperature=gen_cfg.temperature,
                    max_tokens=gen_cfg.max_tokens,
                )
        else:
            response = await self.model_client.complete(
                model=model,
                messages=messages,
                temperature=gen_cfg.temperature,
                max_tokens=gen_cfg.max_tokens,
            )

        candidates = self._parse_response(
            response["text"],
            topic,
            state,
            batch,
            action.action_type,
            response.get("llm_call_id"),
        )
        return candidates, {
            "topic": topic,
            "status": "generated",
            "chunk_ids": chunk_ids,
            "difficulty": difficulty,
            "evidence_strategy": action.evidence_strategy,
            "candidate_ids": [c.question_id for c in candidates],
            "candidate_count": len(candidates),
            "llm_call_id": response.get("llm_call_id"),
        }

    def _parse_response(self, text, topic, state, batch, action_type, llm_call_id) -> list[QuestionCandidate]:
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
                generation_metadata={
                    "round_in_mode": state.round_in_mode,
                    "action_type": action_type,
                    "llm_call_id": llm_call_id,
                },
            ))
        return candidates

    async def _retrieve(self, topic, action, state) -> tuple[list[QuestionCandidate], dict]:
        evidence_pool = self.evidence_manager.evidence_pools.get(topic)
        await self.evidence_manager.expand_retrieval(
            topic=topic,
            queries=[topic],
            language=self.blueprint.language,
            run_id=self.blueprint.run_id,
            evidence_pool=evidence_pool,
        )
        return [], {"topic": topic, "queries": [topic], "status": "retrieved"}

    async def _expand(self, topic, action, state) -> tuple[list[QuestionCandidate], dict]:
        evidence_pool = self.evidence_manager.evidence_pools.get(topic)
        await self.evidence_manager.expand_retrieval(
            topic=topic,
            queries=[f"{topic} details", f"{topic} related"],
            language=self.blueprint.language,
            run_id=self.blueprint.run_id,
            evidence_pool=evidence_pool,
        )
        return [], {
            "topic": topic,
            "queries": [f"{topic} details", f"{topic} related"],
            "status": "expanded",
        }

    async def _evolve(self, state, action) -> tuple[list[QuestionCandidate], dict]:
        if not self.evolution_tool:
            return [], {"status": "missing_evolution_tool", "candidate_count": 0}
        easy_pool = [c for c in state.candidates
                     if c.status == "accepted" and c.difficulty in ("easy", "medium")]
        if not easy_pool:
            return [], {"status": "empty_evolution_pool", "candidate_count": 0}
        candidates = await self.evolution_tool.evolve(easy_pool[:5], state.mode)
        for cand in candidates:
            cand.generation_metadata.update({
                "round_in_mode": state.round_in_mode,
                "action_type": action.action_type,
            })
        return candidates, {
            "status": "evolved",
            "source_question_ids": [c.question_id for c in easy_pool[:5]],
            "candidate_ids": [c.question_id for c in candidates],
            "candidate_count": len(candidates),
        }
