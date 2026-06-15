"""统一入口：加载真实模型，依次运行四组消融实验。

用法：
  cd experiment/qa_agent
  python run_all.py

前置条件（项目根目录 .env）：
  CUSTOM_API_KEY=<your-api-key>
  CUSTOM_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
  MODEL_NAME=deepseek-v3-2-251201
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from benchforge.config.config import load_dotenv
from benchforge.models.openai_client import OpenAIClient

import group_a_direct
import group_b_no_feedback
import group_c_feedback_no_difficulty
import group_d_full

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_API_KEY = os.getenv("CUSTOM_API_KEY", "")
MODEL_BASE_URL = os.getenv("CUSTOM_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v3-2-251201")

TASK_ID = "qa_ablation"
LANGUAGE = "en"

TOPICS = [
    "Artificial Intelligence",
    "Quantum Computing",
    "World War II",
]

QA_COUNT = int(os.getenv("QA_COUNT", "50"))
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "10"))

DIFFICULTY_DISTRIBUTION = {
    "easy": 0.2,
    "medium": 0.3,
    "hard": 0.5,
}

SEEDS = [42]

GROUPS = [
    ("A - Direct Generation", group_a_direct.run),
    # ("B - Multi-round No Feedback", group_b_no_feedback.run),
    # ("C - Feedback No Difficulty", group_c_feedback_no_difficulty.run),
    # ("D - Full Method", group_d_full.run),
]


async def main():
    if not MODEL_API_KEY:
        print(f"[run_all] ERROR: CUSTOM_API_KEY not set. Expected .env at: {PROJECT_ROOT / '.env'}")
        sys.exit(1)

    model_client = OpenAIClient(
        api_key=MODEL_API_KEY,
        base_url=MODEL_BASE_URL,
        model_name=MODEL_NAME,
    )

    # 注入 model_client 到 qa_agent，使 B/C/D 组绕过 registry lookup
    import benchforge.agents.qa_agent.agent as agent_mod
    agent_mod._resolve_model_client_fn = lambda name, path: model_client

    print(f"[run_all] Model: {MODEL_NAME}")
    print(f"[run_all] Topics: {TOPICS}")
    print(f"[run_all] Seeds: {SEEDS}")

    for seed in SEEDS:
        # seed 体现在 task_id 中，使不同 seed 的输出目录隔离
        seed_task_id = f"{TASK_ID}/seed_{seed}"

        for name, run_fn in GROUPS:
            print(f"\n{'=' * 80}")
            print(f"[run_all] Starting {name}, seed={seed}")
            print(f"{'=' * 80}")

            try:
                await run_fn(
                    model_client=model_client,
                    model_name=MODEL_NAME,
                    seed=seed,
                    topics=TOPICS,
                    task_id=seed_task_id,
                    language=LANGUAGE,
                    qa_count=QA_COUNT,
                    max_rounds=MAX_ROUNDS,
                    difficulty_distribution=DIFFICULTY_DISTRIBUTION,
                )
                print(f"[run_all] Finished {name}, seed={seed}")

            except Exception as e:
                import traceback
                print(f"[run_all] FAILED: {name}, seed={seed}, error={e}")
                traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("[run_all] All experiments completed.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
