"""轻量过滤模块（基于设计文档 16，兼容 Yourbench 字段）。"""

from typing import Any

from benchforge.schemas import QuestionMode, Difficulty


class LightweightFilter:
    """轻量过滤器（兼容 Yourbench 字段）。

    只做结构级过滤，不做正式语义验证。
    """

    def __init__(self):
        """初始化过滤器。"""
        self.valid_modes = {mode.value for mode in QuestionMode}
        self.valid_difficulties = {diff.value for diff in Difficulty}
        self.valid_question_types = {
            "factual", "analytical", "conceptual", "application-based",
            "clarification", "counterfactual", "edge-case", "true-false",
            "open-ended", "false-premise",
        }

    def filter_question(self, item: dict[str, Any]) -> tuple[bool, str]:
        """过滤单个问题。

        Args:
            item: 问题字典

        Returns:
            (是否通过, 原因)
        """
        # 1. 检查必需字段
        required_fields = ["question", "answer"]
        for field in required_fields:
            if field not in item or not item.get(field):
                return False, f"缺失或空字段: {field}"

        # 2. 检查问题模式
        mode = item.get("question_mode", "")
        if mode and mode not in self.valid_modes:
            return False, f"非法问题模式: {mode}"

        # 3. 检查难度字段（支持 1-10 数字或字符串）
        difficulty = item.get("estimated_difficulty", "")
        if difficulty:
            if isinstance(difficulty, int):
                if difficulty < 1 or difficulty > 10:
                    return False, f"难度必须在 1-10 之间: {difficulty}"
            else:
                diff_str = str(difficulty).lower()
                if diff_str not in self.valid_difficulties:
                    return False, f"非法难度字段: {difficulty}"

        # 4. 检查引用
        citations = item.get("citations", [])
        if not citations:
            return False, "引用为空"

        # 检查每个引用
        for cit in citations:
            if isinstance(cit, dict):
                if not cit.get("text"):
                    return False, "引用文本为空"
            elif isinstance(cit, str):
                if not cit:
                    return False, "引用文本为空"
            else:
                return False, f"非法引用类型: {type(cit)}"

        # 5. 检查 multiple_choice 特有字段
        if mode == "multiple_choice":
            # 检查选项（Yourbench 用 choices，旧版用 options）
            choices = item.get("choices", []) or item.get("options", [])
            if not choices or len(choices) != 4:
                return False, "multiple_choice 需要恰好 4 个选项"

            # 检查正确答案
            correct_answer = item.get("answer") or item.get("correct_answer")
            if correct_answer is None:
                return False, "multiple_choice 缺少正确答案"

            # 检查答案格式（应该是字母 A/B/C/D）
            if isinstance(correct_answer, int):
                if correct_answer < 0 or correct_answer >= len(choices):
                    return False, f"正确答案索引越界: {correct_answer}"
            elif isinstance(correct_answer, str):
                answer_upper = correct_answer.upper()
                if answer_upper in ["A", "B", "C", "D", "E", "F"]:
                    idx = ord(answer_upper) - ord("A")
                    if idx >= len(choices):
                        return False, f"正确答案字母越界: {correct_answer}"
                else:
                    # 允许直接是答案文本
                    pass

        # 6. 检查问题长度（太短可能无意义）
        question_text = item.get("question", "")
        if len(question_text.strip()) < 10:
            return False, "问题过短"

        # 7. 检查答案长度（MCQ 答案是单字母，跳过长度检查）
        if mode != "multiple_choice":
            answer_text = item.get("answer", "")
            if len(answer_text.strip()) < 5:
                return False, "答案过短"

        # 8. 检查 question_type（Yourbench 字段）
        question_type = item.get("question_type", "").lower()
        if question_type and question_type not in self.valid_question_types:
            return False, f"非法问题类型: {question_type}"

        return True, "通过"

    def filter_questions(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
        """批量过滤问题。

        Args:
            items: 问题列表

        Returns:
            (通过的问题列表, 未通过的问题及原因列表)
        """
        passed = []
        failed = []

        for item in items:
            is_valid, reason = self.filter_question(item)
            if is_valid:
                passed.append(item)
            else:
                failed.append((item, reason))

        return passed, failed

    def normalize_question(self, item: dict[str, Any]) -> dict[str, Any]:
        """规范化问题字段（兼容 Yourbench 字段）。

        Args:
            item: 原始问题字典

        Returns:
            规范化后的问题字典
        """
        normalized = dict(item)

        # 确保问题模式
        if "question_mode" not in normalized:
            normalized["question_mode"] = "qa"

        # 规范化难度（支持 1-10 到 easy/medium/hard 的转换）
        diff = normalized.get("estimated_difficulty", "medium")
        if isinstance(diff, int):
            if diff <= 3:
                normalized["estimated_difficulty"] = "easy"
            elif diff <= 7:
                normalized["estimated_difficulty"] = "medium"
            else:
                normalized["estimated_difficulty"] = "hard"

        # 规范化引用格式
        citations = normalized.get("citations", [])
        normalized_citations = []
        for cit in citations:
            if isinstance(cit, str):
                normalized_citations.append({
                    "chunk_id": "",
                    "text": cit,
                })
            elif isinstance(cit, dict):
                normalized_citations.append(cit)
        normalized["citations"] = normalized_citations

        # 确保 question_type（Yourbench 字段）
        if "question_type" not in normalized:
            normalized["question_type"] = "factual"

        # 确保 thought_process（Yourbench 字段）
        if "thought_process" not in normalized:
            normalized["thought_process"] = ""

        # 确保 required_capability（能力描述）
        if "required_capability" not in normalized:
            normalized["required_capability"] = ""

        # MCQ 特殊处理：确保使用 choices 字段
        if normalized.get("question_mode") == "multiple_choice":
            # 优先使用 choices，其次使用 options
            choices = normalized.get("choices", [])
            if not choices:
                options = normalized.get("options", [])
                if options:
                    # 格式化为 (A) text 格式
                    normalized["choices"] = [
                        f"({chr(65 + i)}) {c}" for i, c in enumerate(options[:4])
                    ]
                    # 移除旧的 options 字段
                    normalized.pop("options", None)

            # 确保答案是单个字母
            answer = normalized.get("answer") or normalized.get("correct_answer")
            if isinstance(answer, int):
                normalized["answer"] = chr(65 + answer)
            elif isinstance(answer, str):
                answer_upper = answer.upper()
                # 如果是数字字符串，转换为字母
                if answer_upper.isdigit():
                    answer_upper = chr(65 + int(answer_upper))
                # 只保留单个字母
                if answer_upper in ["A", "B", "C", "D", "E", "F"]:
                    normalized["answer"] = answer_upper
                else:
                    # 可能是完整答案文本，保留原样
                    normalized["answer"] = answer_upper

            # 移除旧字段
            normalized.pop("correct_answer", None)

        return normalized


def parse_llm_response(response_text: str) -> list[dict[str, Any]]:
    """解析 LLM 响应为问题列表。

    Args:
        response_text: LLM 响应文本

    Returns:
        问题列表
    """
    import json
    import re

    # 尝试直接解析为 JSON
    try:
        data = json.loads(response_text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 可能包装在某个键下
            for key in ["questions", "items", "results", "data"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 数组
    json_array_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
    if json_array_match:
        try:
            return json.loads(json_array_match.group())
        except json.JSONDecodeError:
            pass

    # 尝试提取多个 JSON 对象
    json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    if json_objects:
        items = []
        for obj_str in json_objects:
            try:
                items.append(json.loads(obj_str))
            except json.JSONDecodeError:
                continue
        return items

    return []


def parse_json_response(text: str) -> dict | list | None:
    """从 LLM 响应文本中提取 JSON（对象或数组）。

    统一的 JSON 提取函数，供各 agent 复用。
    verify_agent 用于提取评分 JSON 对象，
    qa_agent 用于提取题目 JSON 数组。

    Args:
        text: LLM 响应文本

    Returns:
        解析后的 dict 或 list，无法解析时返回 None

    Raises:
        ValueError: 无法从文本中解析 JSON
    """
    import json
    import re

    text = text.strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 提取 ```json ... ``` 块
    match = re.search(r'```(?:json)?\s*([\s\S]+?)```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 提取第一个 { ... } 或 [ ... ] 块
    match = re.search(r'\{[\s\S]+\}', text) or re.search(r'\[[\s\S]+\]', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from LLM response: {text[:200]}")