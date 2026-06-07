"""BenchForge 工具模块。"""

from benchforge.utils.retrieval import (
    search_wikipedia,
    fetch_wikipedia_page,
    get_document_source,
)
from benchforge.utils.chunking import chunk_document

__all__ = [
    "search_wikipedia",
    "fetch_wikipedia_page",
    "get_document_source",
    "chunk_document",
]
