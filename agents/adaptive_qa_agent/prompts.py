from pathlib import Path


class PromptBuilder:
    def __init__(self, language: str = "en"):
        self.language = language
        self._system_prompts = {}
        self._user_templates = {}

    def _load_prompt(self, name: str, kind: str) -> str:
        key = f"{name}_{kind}"
        if key not in self._system_prompts:
            path = Path(__file__).parent.parent.parent / "prompts" / "question_generator" / f"{name}_{kind}_prompt.md"
            if path.exists():
                self._system_prompts[key] = path.read_text(encoding="utf-8")
            else:
                self._system_prompts[key] = ""
        return self._system_prompts[key]

    def build_generation_prompt(
        self,
        topic: str,
        difficulty: str,
        mode: str,
        evidence_text: str,
        document_summary: str,
        language: str,
        constraints: str | None = None,
    ) -> list[dict]:
        prompt_name = "mcq" if mode == "multiple_choice" else "qa"
        system_content = self._load_prompt(prompt_name, "system")

        instructions_parts = [f"Generate {difficulty} difficulty questions about: {topic}"]
        if language and language != "en":
            instructions_parts.append(f"Generate questions in {language}.")
        if constraints == "multi_group":
            instructions_parts.append("Focus on multi-hop reasoning requiring multiple evidence sources.")
        elif constraints == "high_hard_score":
            instructions_parts.append("Focus on challenging questions requiring deep understanding.")

        additional_instructions = " ".join(instructions_parts)

        user_content = f"""<additional_instructions>
{additional_instructions}
</additional_instructions>

<title>
{topic}
</title>

<document_summary>
{document_summary}
</document_summary>

<text_chunk>
{evidence_text}
</text_chunk>"""

        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        return messages
