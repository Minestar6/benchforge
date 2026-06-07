"""路径工具函数。

提供项目根目录查找等功能，避免 Path(__file__).parent.parent.parent 在测试和工具中重复出现。
"""

from pathlib import Path

# 项目根目录相对于本文件的位置：utils/paths.py → utils → benchforge → root
_PROJECT_ROOT = Path(__file__).parent.parent


def get_project_root() -> Path:
    """返回 benchforge 项目根目录（包含 __init__.py 的目录）。

    替代所有 Path(__file__).parent.parent.parent 的重复模式。

    Returns:
        项目根目录的 Path 对象
    """
    return _PROJECT_ROOT
