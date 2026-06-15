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
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from benchforge.config.config import load_dotenv
from benchforge.models.openai_client import OpenAIClient

import group_a_direct
import group_b_no_feedback
import group_c_feedback_no_difficulty
import group_d_full

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── 日志配置：同时输出到 stderr（控制台）和文件 ─────────────────────────────
LOG_DIR = PROJECT_ROOT / "runs" / "qa_agent_exp" / "run_all_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)
logger.add(
    LOG_PATH,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    encoding="utf-8",
)
logger.info(f"Log file: {LOG_PATH}")

MODEL_API_KEY = os.getenv("CUSTOM_API_KEY", "")
MODEL_BASE_URL = os.getenv("CUSTOM_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v3-2-251201")

TASK_ID = "ques_generate"
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

MODES = ["qa", "multiple_choice"]

GROUPS = [
    # ("A - Direct Generation", group_a_direct.run),
    ("B - Multi-round No Feedback", group_b_no_feedback.run),
    # ("C - Feedback No Difficulty", group_c_feedback_no_difficulty.run),
    # ("D - Full Method", group_d_full.run),
]


async def main():
    if not MODEL_API_KEY:
        logger.error(f"CUSTOM_API_KEY not set. Expected .env at: {PROJECT_ROOT / '.env'}")
        sys.exit(1)

    model_client = OpenAIClient(
        api_key=MODEL_API_KEY,
        base_url=MODEL_BASE_URL,
        model_name=MODEL_NAME,
    )

    # 注入 model_client 到 qa_agent，使 B/C/D 组绕过 registry lookup
    import benchforge.agents.qa_agent.agent as agent_mod
    agent_mod._resolve_model_client_fn = lambda name, path: model_client

    # 本次运行统一时间戳，防止多次运行互相覆盖
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Modes: {MODES}")
    logger.info(f"Topics: {TOPICS}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Timestamp: {session_ts}")

    for seed in SEEDS:
        seed_task_id = f"{TASK_ID}/seed_{seed}"

        for mode in MODES:
            print(f"\n{'#' * 80}")
            logger.info(f"MODE: {mode}")
            print(f"{'#' * 80}")

            for name, run_fn in GROUPS:
                print(f"\n{'=' * 80}")
                logger.info(f"Starting {name} [{mode}], seed={seed}")
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
                        mode=mode,
                        timestamp=session_ts,
                    )
                    logger.info(f"Finished {name} [{mode}], seed={seed}")

                except Exception as e:
                    logger.opt(exception=True).error(f"FAILED: {name} [{mode}], seed={seed}, error={e}")

    print(f"\n{'=' * 80}")
    logger.info("All experiments completed.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
