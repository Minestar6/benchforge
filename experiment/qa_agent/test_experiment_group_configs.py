import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.qa_agent.config_loader import load_qa_agent_config


CONFIG_DIR = Path(__file__).parent / "configs"


def _load_experiment_config(filename: str):
    config, *_ = load_qa_agent_config(CONFIG_DIR / filename)
    return config.experiment


def test_terminal_hard_repair_only_enabled_for_group_d():
    assert _load_experiment_config("qa_agent_a_direct.yaml").enable_terminal_hard_repair is False
    assert _load_experiment_config("qa_agent_b_no_feedback.yaml").enable_terminal_hard_repair is False
    assert _load_experiment_config("qa_agent_c_feedback_no_difficulty.yaml").enable_terminal_hard_repair is False
    assert _load_experiment_config("qa_agent_d_full.yaml").enable_terminal_hard_repair is True
