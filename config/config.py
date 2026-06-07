"""配置加载和管理模块。

提供统一的：
- .env 文件加载（load_dotenv）
- ${VAR:default} 环境变量展开（expand_env_vars / expand_env_recursive）
- 模型配置（ModelConfig）
- 各类智能体配置模型
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


def load_dotenv(env_path: str | Path) -> None:
    """加载 .env 文件到 os.environ（不覆盖已存在的值）。

    所有 agent config_loader 通过此函数统一加载项目根目录的 .env 文件，
    避免 _load_dotenv 在多个模块中重复实现。

    Args:
        env_path: .env 文件路径
    """
    env_path = Path(env_path)
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def expand_env_vars(text: str) -> str:
    """展开环境变量。

    支持 ${VAR:default} 语法：
    ${OPENAI_API_KEY} - 使用环境变量，不存在则空字符串
    ${OPENAI_BASE_URL:https://api.openai.com/v1} - 使用环境变量，不存在则使用默认值

    Args:
        text: 包含环境变量引用的文本

    Returns:
        展开后的文本
    """
    def replace_match(match):
        var_expr = match.group(1)
        if ":" in var_expr:
            var_name, default = var_expr.split(":", 1)
            return os.getenv(var_name, default)
        else:
            return os.getenv(var_expr, "")

    pattern = r'\$\{([^}]+)\}'
    return re.sub(pattern, replace_match, text)


def expand_env_recursive(data: Any) -> Any:
    """递归展开数据结构中的环境变量。"""
    if isinstance(data, str):
        return expand_env_vars(data)
    elif isinstance(data, dict):
        return {k: expand_env_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [expand_env_recursive(item) for item in data]
    else:
        return data


def load_prompt(path: str | Path) -> str:
    """加载提示词模板文件。

    所有 agent 的 prompt 加载通过此函数统一处理，
    避免 _load_prompt / _load_prompt_template 在多个模块中重复实现。

    Args:
        path: 提示词文件路径（相对或绝对）

    Returns:
        文件内容；文件不存在时返回空字符串
    """
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    import sys
    print(f"Warning: Prompt file not found: {p}", file=sys.stderr)
    return ""


class ModelConfig(BaseModel):
    """模型配置（统一格式，所有模块共用）。

    支持从环境变量自动获取默认值：
    - provider=openai → OPENAI_API_KEY / OPENAI_BASE_URL
    - provider=ollama → http://localhost:11434
    - provider=vllm  → http://localhost:8000
    """
    model_name: str = "gpt-4o-mini"
    provider: str = "openai"  # openai, ollama, vllm
    base_url: str | None = None
    api_key: str | None = None
    max_concurrent_requests: int = 4
    temperature: float = 0.7
    max_tokens: int = 2000
    max_retries: int = 3
    extra_parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def apply_env_defaults(self):
        """从环境变量填充 provider 特定的默认值。"""
        if self.provider == "openai":
            if self.api_key is None:
                self.api_key = os.getenv("OPENAI_API_KEY", "")
            if self.base_url is None:
                self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        elif self.provider == "ollama":
            if self.base_url is None:
                self.base_url = "http://localhost:11434"
        elif self.provider == "vllm":
            if self.base_url is None:
                self.base_url = "http://localhost:8000"
        return self


class RetrievalConfig(BaseModel):
    """检索配置。"""
    max_pages: int = 5            # saliency 关闭时的回退页面数（不暴露在 YAML）
    request_timeout: int = 10
    saliency_rerank: bool = False
    saliency_top_k: int = 3
    saliency_start_date: str = "2022010100"
    saliency_end_date: str = "2025010100"
    min_paragraph_tokens: int = 20


class ChunkingConfig(BaseModel):
    """题目生成分块配置。"""
    chunk_size: int = 2048
    overlap: int = 300
    encoding: str = "cl100k_base"


class SummarizationChunkingConfig(BaseModel):
    """文档总结分段配置。"""
    chunk_size: int = 8192
    overlap: int = 512
    encoding: str = "cl100k_base"


class GenerationConfig(BaseModel):
    """生成配置。"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    questions_per_chunk: int = 2
    question_mode: str = "open-ended"
    prompt_template_id: str = "question_generation_v1"
    allowed_question_types: list[str] = Field(default_factory=list)
    difficulty_distribution: dict[str, float] = Field(default_factory=dict)


class ValidationConfig(BaseModel):
    """验证配置。"""
    require_citations: bool = True
    min_citation_length: int = 20
    require_capability: bool = True
    deduplicate: bool = True


class PromptConfig(BaseModel):
    """提示配置。"""
    template_path: str = "./prompts/question_generation.md"
    system_prompt: str = "你是一个专业的问答题目生成专家。"


class OutputConfig(BaseModel):
    """输出配置。"""
    save_raw_responses: bool = True
    save_source_documents: bool = True
    output_format: str = "jsonl"


class LoggingConfig(BaseModel):
    """日志配置。"""
    level: str = "INFO"
    log_file: str = "./logs/${run_id}.log"


class RunConfig(BaseModel):
    """运行配置。"""
    task_id: str = "task_001"
    run_id: str = "run_001"
    output_path: str = "./runs/${task_id}/${run_id}"
    language: str = "en"
    domain: str | None = None


class QuestionGeneratorConfig(BaseModel):
    """问题生成智能体统一配置。"""

    # 运行时配置
    run: RunConfig = Field(default_factory=RunConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    summarization_chunking: SummarizationChunkingConfig = Field(default_factory=SummarizationChunkingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path, task_id: str | None = None, run_id: str | None = None) -> "QuestionGeneratorConfig":
        """从 YAML 文件加载配置。

        Args:
            config_path: 配置文件路径
            task_id: 可选，覆盖配置中的 task_id
            run_id: 可选，覆盖配置中的 run_id
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        # 展开环境变量
        raw_data = expand_env_recursive(raw_data)

        # 如果提供了 task_id/run_id，覆盖配置中的值
        if task_id is not None:
            if "run" in raw_data:
                raw_data["run"]["task_id"] = task_id
            else:
                raw_data["run"] = {"task_id": task_id}

        if run_id is not None:
            if "run" in raw_data:
                raw_data["run"]["run_id"] = run_id
            else:
                raw_data["run"] = {"run_id": run_id}

        return cls.model_validate(raw_data)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return self.model_dump()

    def save_yaml(self, output_path: str | Path) -> None:
        """保存为 YAML 文件。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def get_resolved_output_path(self) -> Path:
        """获取解析后的输出路径（替换 ${task_id} 和 ${run_id}）。"""
        path = self.run.output_path.replace("${task_id}", self.run.task_id)
        path = path.replace("${run_id}", self.run.run_id)
        return Path(path)

    def get_resolved_log_path(self) -> Path:
        """获取解析后的日志路径（替换 ${run_id}）。"""
        path = self.logging.log_file.replace("${run_id}", self.run.run_id)
        return Path(path)