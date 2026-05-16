from __future__ import annotations

import json
import logging
from typing import Any

from analytics_agent.config import AppConfig
from analytics_agent.providers.base import BaseLlmClient, JsonContentParser


logger = logging.getLogger(__name__)


class GeminiApiClient(BaseLlmClient):

    def __init__(self, config: AppConfig, client: Any | None = None) -> None:
        self.config = config
        self._client = client

    def complete_json(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, Any]:
        client = self._get_client()

        logger.info("Gemini API context clear: user_id=%s", self.config.gemini_user_id)
        client.clear_context()

        system_instruction, message_text = self._prepare_prompt(messages)
        logger.info("Gemini API request: model=%s", self.config.gemini_model)
        response = client.generate(
            message_text=message_text,
            model_name=self.config.gemini_model,
            flag_search=False,
            system_instruction=system_instruction,
        )

        content = self._response_to_text(response)
        logger.info("Gemini API response received: model=%s", self.config.gemini_model)

        return self.config.gemini_model, JsonContentParser.parse(content)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from g_api_view import Client
            except ImportError as error:
                raise RuntimeError(
                    "g_api_view is required for LLM_PROVIDER=gemini. "
                    "Install it with: pip install "
                    "git+https://github.com/brdchy/ApiClientG.git@v0.2.1"
                ) from error

            self._client = Client(
                server_url=self.config.gemini_server_url,
                user_id=self.config.gemini_user_id,
                password=self.config.gemini_password,
                auto_register=True,
            )

        return self._client

    def _prepare_prompt(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        system_parts = [
            message["content"]
            for message in messages
            if message.get("role") == "system"
        ]
        transcript_parts = []

        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                continue
            transcript_parts.append(f"{role.upper()}:\n{message.get('content', '')}")

        system_instruction = "\n\n".join(system_parts)
        message_text = (
            "Return only a valid JSON object. Do not wrap it in markdown.\n\n"
            + "\n\n".join(transcript_parts)
        )

        return system_instruction, message_text

    def _response_to_text(self, response: Any) -> str:
        if isinstance(response, dict):
            answer = response.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer

            return json.dumps(response, ensure_ascii=False)

        if isinstance(response, str):
            return response

        return json.dumps(response, ensure_ascii=False)
