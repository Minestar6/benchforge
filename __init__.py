"""BenchForge - 基准测试题目自动生成框架。"""

from benchforge.schemas import (
    SourceDocument,
    SourceChunk,
    QuestionRecord,
    QuestionType,
    Difficulty,
    QuestionStatus,
)
from benchforge.utils import search_wikipedia, fetch_wikipedia_page
from benchforge.utils.artifact_store import ArtifactStore
from benchforge.agents import run_generation_agent, VerifyAgent
from benchforge.agents.planner_agent import UserIntent, synthesize_global_blueprint, run_planner
from benchforge.app import run_benchforge
from benchforge.models import BaseModelClient, OpenAIClient, FakeModelClient, ModelLoader
from benchforge.config import QuestionGeneratorConfig

__all__ = [
    "SourceDocument", "SourceChunk", "QuestionRecord", "QuestionType", "Difficulty", "QuestionStatus",
    "search_wikipedia", "fetch_wikipedia_page",
    "ArtifactStore",
    "run_generation_agent", "VerifyAgent",
    "UserIntent", "synthesize_global_blueprint", "run_planner", "run_benchforge",
    "BaseModelClient", "OpenAIClient", "FakeModelClient", "ModelLoader",
    "QuestionGeneratorConfig",
]

__version__ = "0.2.0"
