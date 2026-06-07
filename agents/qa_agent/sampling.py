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


def record_global_chunk_usage(
    global_state: GlobalState,
    chunks: list[Any],
    max_size: int,
) -> None:
    chunk_ids = raw_chunk_ids(chunks)
    combo = tuple(chunk_ids)

    if combo not in global_state.used_chunk_combinations:
        global_state.used_chunk_combinations.add(combo)
        global_state.used_chunk_combination_order.append(combo)

    for cid in chunk_ids:
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

        combo = tuple(raw_chunk_ids(chunks))
        if combo not in global_used_combinations:
            return chunks, False

        last_chunks = chunks

    return last_chunks, True
