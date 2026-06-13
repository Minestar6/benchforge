"""Chunk sampling helpers: raw_chunk_ids, dedup, record usage, sample_chunks wrapper."""

import logging
from typing import Any

from .state import GlobalState

logger = logging.getLogger(__name__)


def raw_chunk_ids(chunks: list[Any]) -> list[str]:
    ids = []
    for unit in chunks:
        if hasattr(unit, "chunk_ids") and isinstance(unit.chunk_ids, list):
            # MultiChunkUnit: chunk_ids is a list[str]
            ids.extend(unit.chunk_ids)
        elif hasattr(unit, "chunk_id"):
            ids.append(unit.chunk_id)
        else:
            logger.warning("raw_chunk_ids: unrecognized chunk type %s, skipping", type(unit))
            continue
    return sorted(set(ids))


def _unit_combos(chunks: list[Any]) -> list[tuple[str, ...]]:
    """Decompose a list of units into per-unit combos.

    Each SingleChunkUnit becomes a single-element combo (chunk_id,).
    Each MultiChunkUnit becomes a combo of its sorted chunk_ids.
    Single and multi units are NEVER merged; they stay separate combos.
    """
    combos = []
    for unit in chunks:
        if hasattr(unit, "chunk_ids") and isinstance(unit.chunk_ids, list):
            if unit.chunk_ids:
                combos.append(tuple(sorted(unit.chunk_ids)))
        elif hasattr(unit, "chunk_id"):
            combos.append((unit.chunk_id,))
    return combos


def record_global_chunk_usage(
    global_state: GlobalState,
    chunks: list[Any],
    max_size: int,
) -> None:
    unit_combos = _unit_combos(chunks)
    for combo in unit_combos:
        if combo not in global_state.used_chunk_combinations:
            global_state.used_chunk_combinations.add(combo)
            global_state.used_chunk_combination_order.append(combo)

        for cid in combo:
            global_state.chunk_usage_counts[cid] = global_state.chunk_usage_counts.get(cid, 0) + 1

    while len(global_state.used_chunk_combinations) > max_size:
        oldest = global_state.used_chunk_combination_order.popleft()
        global_state.used_chunk_combinations.discard(oldest)


def sample_chunks(
    evidence_manager: Any,
    topic: str,
    mode: str,
    difficulty: str,
    single_k: int,
    multi_k: int,
    global_used_combinations: set[tuple[str, ...]],
    global_chunk_usage_counts: dict[str, int],
    blocked_combinations: set[tuple[str, ...]] | None = None,
    round_num: int = 1,
) -> tuple[list[Any], bool]:
    evidence_pool = evidence_manager.evidence_pools.get(topic)
    if not evidence_pool:
        return [], False

    last_chunks: list[Any] = []

    for _ in range(5):
        batch = evidence_manager.sample(
            evidence_pool=evidence_pool,
            topic=topic,
            target_mode=mode,
            target_difficulty=difficulty,
            prefer_multi_chunk=(multi_k > single_k),
            round_num=round_num,
            remaining=max(single_k + multi_k, 1),
        )

        single_units = [u for u in evidence_pool.single_chunks if u.chunk_id in batch.single_chunk_ids]
        multi_units = [u for u in evidence_pool.multi_chunks if u.unit_id in batch.multi_chunk_ids]
        chunks = single_units + multi_units

        unit_combos_retry = _unit_combos(chunks)
        if not unit_combos_retry:
            continue
        blocked = blocked_combinations if blocked_combinations is not None else global_used_combinations
        if all(c not in blocked for c in unit_combos_retry):
            return chunks, False

        last_chunks = chunks

    return last_chunks, True
