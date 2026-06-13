"""Register custom pytest markers and set up path for direct submodule imports."""
import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async (requires pytest-asyncio)")
