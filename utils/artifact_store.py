"""Artifact 存储模块。"""

import json
from pathlib import Path
from typing import Any

from loguru import logger


class ArtifactStore:
    """Artifact 存储类。"""

    def __init__(self, base_path: str):
        """初始化 Artifact 存储。"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def append_jsonl(
        self,
        filename: str,
        records: list[Any],
    ) -> int:
        """追加记录到 JSONL 文件。"""
        filepath = self.base_path / filename
        count = 0

        with open(filepath, "a", encoding="utf-8") as f:
            for record in records:
                if hasattr(record, "model_dump"):
                    data = record.model_dump()
                elif hasattr(record, "dict"):
                    data = record.dict()
                else:
                    data = record

                f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
                count += 1

        logger.info(f"Appended {count} records to {filepath}")
        return count

    def read_jsonl(
        self,
        filename: str,
    ) -> list[dict[str, Any]]:
        """读取 JSONL 文件。"""
        filepath = self.base_path / filename
        if not filepath.exists():
            return []

        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        return records

    def save_json(
        self,
        filename: str,
        data: Any,
    ) -> None:
        """保存 JSON 文件。"""
        filepath = self.base_path / filename

        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "dict"):
            data = data.dict()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Saved JSON to {filepath}")

    def load_json(
        self,
        filename: str,
    ) -> Any:
        """加载 JSON 文件。"""
        filepath = self.base_path / filename
        if not filepath.exists():
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def exists(self, filename: str) -> bool:
        """检查文件是否存在。"""
        return (self.base_path / filename).exists()

    def list_files(self, pattern: str = "*.jsonl") -> list[str]:
        """列出匹配的文件。"""
        return [f.name for f in self.base_path.glob(pattern)]