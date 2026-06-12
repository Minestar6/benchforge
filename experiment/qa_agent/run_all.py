"""统一入口：加载真实模型，依次运行四组实验。

用法：
  cd experiment/qa_agent
  python run_all.py

前置条件（项目根目录 .env）：
  MODEL_API_KEY=<your-api-key>
  MODEL_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
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
import group_b_single
import group_c_nodiff
import group_d_full

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v3-2-251201")

if not MODEL_API_KEY:
    print(f"[run_all] ERROR: MODEL_API_KEY not set. Expected .env at: {PROJECT_ROOT / '.env'}")
    sys.exit(1)

TOPICS = ["Climate Change", "Artificial Intelligence"]
TASK_ID = "exp_task_qa"
LANGUAGE = "en"
QA_COUNT = 50
DIFFICULTY_DISTRIBUTION = {"easy": 0.2, "medium": 0.5, "hard": 0.3}

GROUPS = [
    ("A - Direct Generation",         group_a_direct.run),
    ("B - Single-round Agent",         group_b_single.run),
    ("C - Multi-round No Difficulty",  group_c_nodiff.run),
    ("D - Full qa_agent",              group_d_full.run),
]

SHARED_KWARGS = dict(
    topics=TOPICS, task_id=TASK_ID, language=LANGUAGE,
    qa_count=QA_COUNT, difficulty_distribution=DIFFICULTY_DISTRIBUTION,
)


async def main():
    model_client = OpenAIClient(api_key=MODEL_API_KEY, base_url=MODEL_BASE_URL, model_name=MODEL_NAME)
    print(f"[run_all] Model : {MODEL_NAME}\n[run_all] Topics: {TOPICS}")

    for name, run_fn in GROUPS:
        print(f"\n{'='*60}\n[run_all] Starting Group {name}\n{'='*60}")
        try:
            await run_fn(model_client=model_client, model_name=MODEL_NAME, **SHARED_KWARGS)
            print(f"[run_all] Group {name} done.")
        except Exception as e:
            import traceback
            print(f"[run_all] Group {name} FAILED: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}\n[run_all] All groups completed.\n{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
