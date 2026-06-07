"""文本解析工具。"""

import json
import re
from typing import Any

from loguru import logger


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """从文本中提取 JSON 数组。

    支持的格式：
    1. 纯 JSON 数组
    2. ```json 代码块
    3. <output_json> 标签

    Args:
        text: 模型输出的文本

    Returns:
        解析后的 JSON 数组，解析失败返回空列表
    """
    if not text or not isinstance(text, str):
        return []

    # 1. 尝试提取 <output_json> 标签内容
    json_str = _extract_tag_content(text, "output_json")
    if json_str.strip():
        result = _attempt_json_parse(json_str)
        if result is not None:
            return result

    # 2. 尝试提取 ```json 代码块
    json_str = _extract_fenced_json(text)
    if json_str:
        result = _attempt_json_parse(json_str)
        if result is not None:
            return result

    # 3. 尝试提取 JSON 数组（花括号包围）
    json_str = _extract_json_by_brackets(text)
    if json_str:
        result = _attempt_json_parse(json_str)
        if result is not None:
            return result

    logger.warning("Failed to extract JSON array from text")
    return []


def _extract_tag_content(text: str, tag: str) -> str:
    """提取 XML 标签内容。"""
    pattern = rf"<{tag}\s*>([\s\S]*?)</{tag}>"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""


def _extract_fenced_json(text: str) -> str:
    """提取 ```json 代码块。"""
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""


def _extract_json_by_brackets(text: str) -> str:
    """通过括号提取 JSON。"""
    # 尝试匹配最外层的数组
    stack = []
    start_idx = -1

    for i, char in enumerate(text):
        if char == "[":
            if not stack:
                start_idx = i
            stack.append(char)
        elif char == "]":
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    return text[start_idx : i + 1].strip()

    return ""


def _attempt_json_parse(json_str: str) -> list[dict[str, Any]] | None:
    """尝试解析 JSON 字符串。"""
    try:
        json_str = _strip_backticks(json_str)
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return parsed
        return None
    except Exception as e:
        logger.debug(f"JSON parse failed: {e}")
        return None


def _strip_backticks(text: str) -> str:
    """移除代码块标记。"""
    if not text or not isinstance(text, str):
        return ""

    # 移除开头和结尾的 ``` 或 ```json
    pattern = r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(pattern, text)
    if match:
        return match.group(1).strip()

    return text.strip()