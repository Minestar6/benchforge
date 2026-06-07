"""文档分块处理管道。"""

import tiktoken

from benchforge.schemas import SourceDocument, SourceChunk


def chunk_document(
    document: SourceDocument,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[SourceChunk]:
    """将文档切分成 token 块。

    Args:
        document: 源文档
        chunk_size: 每块的 token 数量
        overlap: 块之间的重叠 token 数量

    Returns:
        分块列表
    """
    if not document.content or not document.content.strip():
        return []

    text = document.content
    chunks_text = _split_into_token_chunks(text, chunk_size, overlap)

    chunks = []
    for i, chunk_text in enumerate(chunks_text):
        chunk_id = f"{document.document_id}::chunk_{i:04d}"
        chunk = SourceChunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            chunk_index=i,
            text=chunk_text,
            metadata={
                "title": document.title,
                "url": document.url,
                "topic": document.topic,
                "language": document.language,
            },
        )
        chunks.append(chunk)

    return chunks


def _split_into_token_chunks(
    text: str,
    chunk_tokens: int = 1200,
    overlap: int = 150,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """将文本分割成基于 token 的块。

    Args:
        text: 输入文本
        chunk_tokens: 每块最大 token 数
        overlap: 重叠 token 数
        encoding_name: tiktoken 编码名称

    Returns:
        文本块列表
    """
    if not text or not text.strip():
        return []

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text, disallowed_special=())
    stride = chunk_tokens - overlap

    chunks = []
    for i in range(0, len(tokens), stride):
        chunk_tokens_slice = tokens[i : i + chunk_tokens]
        chunk_text = enc.decode(chunk_tokens_slice)
        chunks.append(chunk_text)

    return chunks