"""统一入口：使用 model_registry.yaml 解析模型，依次运行四组消融实验。

模型身份从各 group 的 YAML 配置中的 model.name → model_registry.yaml 解析，
与 verify_agent 等组件保持一致的注册表模式。

用法：
  cd experiment/qa_agent
  python run_all.py

前置条件（项目根目录 .env）：
  CUSTOM_API_KEY=<your-api-key>
  CUSTOM_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
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

import group_a_direct
import group_b_no_feedback
import group_c_feedback_no_difficulty
import group_d_full

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY", "")
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
    ("A - Direct Generation", group_a_direct.run),
    ("B - Multi-round No Feedback", group_b_no_feedback.run),
    ("C - Feedback No Difficulty", group_c_feedback_no_difficulty.run),
    ("D - Full Method", group_d_full.run),
]


async def main():
    if not CUSTOM_API_KEY:
        logger.error(f"CUSTOM_API_KEY not set. Expected .env at: {PROJECT_ROOT / '.env'}")
        sys.exit(1)

    # 本次运行统一时间戳，防止多次运行互相覆盖
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "runs" / TASK_ID / "run_all_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_all_{session_ts}.log"
    logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        encoding="utf-8",
    )
    logger.info(f"Log file: {log_path}")

    logger.info(f"Modes: {MODES}")
    logger.info(f"Topics: {TOPICS}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Timestamp: {session_ts}")
    logger.info("Model resolution via model_registry.yaml (model.name in each group config)")

    for seed in SEEDS:
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
                        seed=seed,
                        topics=TOPICS,
                        task_id=TASK_ID,
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
