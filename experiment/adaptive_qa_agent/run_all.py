"""统一入口：加载 DeepSeek 真实模型，依次运行四组实验，汇总结果。

用法：
  cd experiment/adaptive_qa_agent
  python run_all.py

前置条件：
  项目根目录 .env 文件需包含:
    MODEL_API_KEY=<your-deepseek-api-key>
    MODEL_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
    MODEL_NAME=deepseek-v3-2-251201
"""
import asyncio
import os
import sys
from pathlib import Path

# ── 路径设置（必须在业务导入之前，绕开 benchforge/agents/__init__.py 的循环导入） ──
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # master/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))          # benchforge/
sys.path.insert(0, str(Path(__file__).parent))                        # adaptive_qa_agent/

from benchforge.config.config import load_dotenv
from benchforge.models.openai_client import OpenAIClient

import group_a_direct
import group_b_single
import group_c_nodiff
import group_d_full

# ── 加载 .env ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v3-2-251201")

if not MODEL_API_KEY:
    print("[run_all] ERROR: MODEL_API_KEY not set in .env file")
    print(f"[run_all] Expected .env at: {PROJECT_ROOT / '.env'}")
    sys.exit(1)

# ── 统一实验蓝图参数 ───────────────────────────────────────────────────────────
TOPICS = ["Climate Change", "Artificial Intelligence"]
TASK_ID = "exp_task_50"
LANGUAGE = "en"
QA_COUNT = 50
DIFFICULTY_DISTRIBUTION = {"easy": 0.2, "medium": 0.2, "hard": 0.6}

GROUPS = [
    ("A - Direct Generation",           group_a_direct.run),
    ("B - Single-round Agent",          group_b_single.run),
    ("C - Multi-round No Difficulty",   group_c_nodiff.run),
    ("D - Full AdaptiveQAAgent",        group_d_full.run),
]

# ── 传递给各组的通用参数 ──
SHARED_KWARGS = {
    "topics": TOPICS,
    "task_id": TASK_ID,
    "language": LANGUAGE,
    "qa_count": QA_COUNT,
    "difficulty_distribution": DIFFICULTY_DISTRIBUTION,
}


async def main():
    # 创建 DeepSeek 模型客户端（OpenAI 兼容协议）
    model_client = OpenAIClient(
        api_key=MODEL_API_KEY,
        base_url=MODEL_BASE_URL,
        model_name=MODEL_NAME,
    )
    print(f"[run_all] Model : {MODEL_NAME}")
    print(f"[run_all] URL   : {MODEL_BASE_URL}")
    print(f"[run_all] Topics: {TOPICS}")

    for name, run_fn in GROUPS:
        print(f"\n{'=' * 60}")
        print(f"[run_all] Starting Group {name}")
        print(f"{'=' * 60}")
        try:
            await run_fn(
                model_client=model_client,
                model_name=MODEL_NAME,
                **SHARED_KWARGS,
            )
            print(f"[run_all] Group {name} done.")
        except Exception as e:
            print(f"[run_all] Group {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"[run_all] All groups completed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
