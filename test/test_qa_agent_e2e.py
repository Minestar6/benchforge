"""qa_agent end-to-end integration tests.

Tests the new public API: run_generation_agent(blueprint, config_path).

Usage:
    # Fast: fake model only (no API key needed)
    pytest test/test_qa_agent_e2e.py -v -k fake

    # Real model (requires .env with CUSTOM_API_KEY / CUSTOM_API_BASE_URL)
    pytest test/test_qa_agent_e2e.py -v -k real

    # All
    pytest test/test_qa_agent_e2e.py -v
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

# ============================================================================
# Fixtures
# ============================================================================

CONFIG_PATH = project_root / "config" / "qa_agent.yaml"


@pytest.fixture(autouse=True)
def cwd_to_project_root():
    """Run every test from project root so relative paths (runs/) land correctly."""
    original = os.getcwd()
    os.chdir(project_root)
    yield
    os.chdir(original)


@pytest.fixture
def blueprint():
    """Minimal blueprint for fast iteration."""
    from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg

    return Blueprint(
        task_id="test_e2e",
        run_id="run_001",
        language="en",
        topics=["Python (programming language)"],
        modes={
            "qa": ModeCfg(
                count=4,
                max_rounds=2,
                difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
            ),
            "multiple_choice": ModeCfg(
                count=2,
                max_rounds=2,
                difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
            ),
        },
    )


@pytest.fixture(autouse=True)
def clean_runs():
    """Remove previous test output before each test."""
    run_dir = Path("runs") / "test_e2e"
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    yield


# ============================================================================
# Fake-model test — always safe, no network / API key
# ============================================================================

class _FakeClient:
    """Returns valid QA / MCQ JSON that passes the LightweightFilter."""
    provider = "fake"
    model_name = "fake-e2e"
    temperature = 0.7
    max_tokens = 2000
    call_count = 0

    async def complete(self, model: str, messages: list[dict], **kwargs) -> dict:
        self.call_count += 1
        content = messages[-1].get("content", "") if messages else ""

        if "<final_summary>" in content:
            text = "<final_summary>Python is a high-level programming language.</final_summary>"
        elif "multiple_choice" in content.lower():
            text = json.dumps([
                {
                    "question": "Which of the following is a key feature of Python?",
                    "choices": ["(A) Readable syntax", "(B) Compiled only", "(C) No standard library", "(D) Static typing only"],
                    "answer": "A",
                    "question_mode": "multiple_choice",
                    "thought_process": "Tests understanding of Python design philosophy.",
                    "question_type": "conceptual",
                    "required_capability": "understanding language features",
                    "estimated_difficulty": 3,
                    "citations": ["Python has a design philosophy that emphasizes code readability."],
                }
            ])
        else:
            text = json.dumps([
                {
                    "question": "What is Python's design philosophy?",
                    "answer": "Python emphasizes code readability with its notable use of significant whitespace.",
                    "question_mode": "qa",
                    "thought_process": "Tests understanding of Python's core design principles.",
                    "question_type": "factual",
                    "required_capability": "understanding language philosophy",
                    "estimated_difficulty": 3,
                    "citations": ["Python has a design philosophy that emphasizes code readability."],
                }
            ])
        return {"text": text, "input_tokens": 100, "output_tokens": 150, "latency": 0.01,
                "raw": {}, "llm_call_id": f"fake_{self.call_count:04d}"}

    async def batch_complete(self, model, messages_list, **kwargs):
        return [await self.complete(model, msgs, **kwargs) for msgs in messages_list]


@pytest.mark.asyncio
async def test_fake_model_produces_report(blueprint):
    """Fake model — verifies report structure and artifact creation."""
    import agents.qa_agent.agent as agent_mod
    from benchforge.agents.qa_agent import run_generation_agent

    client = _FakeClient()
    agent_mod._resolve_model_client_fn = lambda name, path: client

    report = await run_generation_agent(
        blueprint=blueprint,
        config_path=str(CONFIG_PATH),
    )

    # Report structure
    assert report["task_id"] == "test_e2e"
    assert report["run_id"] == "run_001"
    assert "modes" in report
    assert "total_candidates" in report
    assert report["total_candidates"] >= 0

    for mode in ("qa", "multiple_choice"):
        assert mode in report["modes"]
        assert "candidate_count" in report["modes"][mode]
        assert "target_candidate_count" in report["modes"][mode]
        assert "stopped_reason" in report["modes"][mode]

    # Artifact files
    run_dir = Path("runs") / "test_e2e" / "run_001"
    assert run_dir.exists()
    assert (run_dir / "shared_state.json").exists(), "shared_state.json missing"
    assert (run_dir / "qa" / "candidate_pool.json").exists(), "qa pool missing"
    assert (run_dir / "multiple_choice" / "candidate_pool.json").exists(), "mcq pool missing"
    assert (run_dir / "evidence" / "chunked.json").exists(), "chunked evidence missing"
    assert (run_dir / "generation_report.json").exists(), "generation report missing"

    # shared_state is valid JSON
    state = json.loads((run_dir / "shared_state.json").read_text(encoding="utf-8"))
    assert state["task_id"] == "test_e2e"
    assert state["agent_status"]["generation"] == "completed"


# ============================================================================
# Real-model test — requires .env, optionally skipped
# ============================================================================

def _api_key_available() -> bool:
    # Load .env from project root
    env_path = project_root / ".env"
    if not env_path.exists():
        return False
    from benchforge.config.config import load_dotenv
    load_dotenv(env_path)
    return bool(os.getenv("CUSTOM_API_KEY"))


requires_api_key = pytest.mark.skipif(
    not _api_key_available(),
    reason="CUSTOM_API_KEY not set in .env",
)


@pytest.mark.asyncio
@requires_api_key
async def test_real_model_produces_report(blueprint):
    """Real model (deepseek-v3) — full Wikipedia retrieval + LLM generation."""
    from benchforge.agents.qa_agent import run_generation_agent

    report = await run_generation_agent(
        blueprint=blueprint,
        config_path=str(CONFIG_PATH),
    )

    assert report["total_candidates"] >= 0
    assert report["modes"]["qa"]["stopped_reason"] is not None
    assert report["modes"]["multiple_choice"]["stopped_reason"] is not None

    run_dir = Path("runs") / "test_e2e" / "run_001"
    assert run_dir.exists()
    assert (run_dir / "shared_state.json").exists()

    # Candidate pools contain valid data
    qa_pool = json.loads((run_dir / "qa" / "candidate_pool.json").read_text(encoding="utf-8"))
    assert isinstance(qa_pool, list)
    if qa_pool:
        q = qa_pool[0]
        assert q.get("question")
        assert q.get("answer")
        assert q.get("difficulty") in ("easy", "medium", "hard")
        assert q.get("status") in ("accepted", "rejected")

    print(f"\n  [real] QA accepted: {report['modes']['qa']['candidate_count']}/{report['modes']['qa']['target_candidate_count']}")
    print(f"  [real] MCQ accepted: {report['modes']['multiple_choice']['candidate_count']}/{report['modes']['multiple_choice']['target_candidate_count']}")
    print(f"  [real] Stopped: qa={report['modes']['qa']['stopped_reason']}, mcq={report['modes']['multiple_choice']['stopped_reason']}")
    print(f"  [real] Output: {run_dir}")


@pytest.mark.asyncio
@requires_api_key
async def test_real_model_minimal_single_mode(blueprint):
    """Real model with a single mode (QA only) — validates fast path."""
    from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg
    from benchforge.agents.qa_agent import run_generation_agent

    minimal = Blueprint(
        task_id="test_e2e",
        run_id="run_minimal",
        language="en",
        topics=["Python (programming language)"],
        modes={
            "qa": ModeCfg(
                count=2,
                max_rounds=1,
                difficulty_distribution={"easy": 0.5, "medium": 0.5, "hard": 0.0},
            ),
        },
    )

    report = await run_generation_agent(
        blueprint=minimal,
        config_path=str(CONFIG_PATH),
    )

    assert report["total_candidates"] >= 0
    run_dir = Path("runs") / "test_e2e" / "run_minimal"
    assert (run_dir / "shared_state.json").exists()
    assert (run_dir / "qa" / "candidate_pool.json").exists()

    # No MCQ dir should be created
    assert not (run_dir / "multiple_choice" / "candidate_pool.json").exists()

    print(f"\n  [real-minimal] QA: {report['modes']['qa']['candidate_count']}/{report['modes']['qa']['target_candidate_count']}")
    print(f"  [real-minimal] Stopped: {report['modes']['qa']['stopped_reason']}")
