"""experiment/qa_agent/verify_all.py tests."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_verify_all_uses_case_verify_config_by_default():
    import experiment.qa_agent.verify_all as verify_all_module

    assert verify_all_module.CONFIG_PATH.endswith("experiment/qa_agent/configs/verify_agent.yaml")
