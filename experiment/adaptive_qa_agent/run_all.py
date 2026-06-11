"""统一入口：依次运行四组实验，汇总结果。

用法：
  cd /Users/zhaoziqing/Desktop/benchforge/experiment/adaptive_qa_agent
  python run_all.py
"""
import asyncio
import sys
from pathlib import Path

# 必须 insert Desktop 层级，绕开 benchforge/agents/__init__.py 的循环导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import group_a_direct
import group_b_single
import group_c_nodiff
import group_d_full

GROUPS = [
    ("A - Direct Generation",          group_a_direct.run),
    ("B - Single-round Agent",          group_b_single.run),
    ("C - Multi-round No Difficulty",   group_c_nodiff.run),
    ("D - Full AdaptiveQAAgent",        group_d_full.run),
]


async def main():
    for name, run_fn in GROUPS:
        print(f"\n{'='*60}\n[run_all] Starting Group {name}\n{'='*60}")
        await run_fn()
        print(f"[run_all] Group {name} done.")


if __name__ == "__main__":
    asyncio.run(main())
