"""Tests for generic LLM call tracing in model clients."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from benchforge.models.fake import FakeModelClient


@pytest.mark.asyncio
async def test_fake_model_client_records_llm_call_jsonl(tmp_path):
    trace_path = tmp_path / "runs" / "task_x" / "run_y" / "llm_calls.jsonl"
    client = FakeModelClient(delay=0)

    response = await client.complete(
        model="fake-model",
        messages=[{"role": "user", "content": "chunk_id: doc_a::chunk_0"}],
        llm_trace_path=str(trace_path),
    )

    assert response["llm_call_id"]
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["llm_call_id"] == response["llm_call_id"]
    assert record["request"]["model"] == "fake-model"
    assert record["request"]["messages"][0]["content"] == "chunk_id: doc_a::chunk_0"
    assert record["response"]["text"] == response["text"]
    assert record["error"] is None
