from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class BaseLlmClient(ABC):

    @abstractmethod
    def complete_json(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, Any]:
        raise NotImplementedError


class JsonContentParser:

    @staticmethod
    def parse(content: str) -> Any:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            candidate = JsonContentParser._extract_balanced_json(cleaned)
            if candidate is None:
                raise ValueError(f"LLM response does not contain JSON: {content}")

            return json.loads(candidate)

    @staticmethod
    def _extract_balanced_json(content: str) -> str | None:
        for opening, closing in (("{", "}"), ("[", "]")):
            start = content.find(opening)
            if start == -1:
                continue

            depth = 0
            in_string = False
            escape = False

            for index in range(start, len(content)):
                char = content[index]
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == opening:
                    depth += 1
                elif char == closing:
                    depth -= 1
                    if depth == 0:
                        return content[start : index + 1]

        return None
