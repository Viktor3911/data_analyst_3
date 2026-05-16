from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from analytics_agent.config import AppConfig
from analytics_agent.providers.base import BaseLlmClient, JsonContentParser


logger = logging.getLogger(__name__)


class OpenRouterClient(BaseLlmClient):
    RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
    AUTH_STATUS_CODES = {401, 403}

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def complete_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 3000,
    ) -> tuple[str, Any]:
        last_error: Exception | None = None
        failures: list[str] = []

        for model in self.config.models:

            for json_mode in (True, False):
                mode_label = "json_mode" if json_mode else "plain_mode"

                for attempt in range(1, self.config.max_model_attempts + 1):
                    try:
                        logger.info(
                            "OpenRouter request: model=%s mode=%s attempt=%s",
                            model,
                            mode_label,
                            attempt,
                        )
                        content = self._complete(
                            model,
                            messages,
                            max_tokens=max_tokens,
                            json_mode=json_mode,
                        )
                        logger.info(
                            "OpenRouter response received: model=%s mode=%s attempt=%s",
                            model,
                            mode_label,
                            attempt,
                        )

                        return model, JsonContentParser.parse(content)
                    except Exception as error:
                        last_error = error
                        failure = f"{model}/{mode_label}/attempt_{attempt}: {error}"
                        failures.append(failure)
                        if self._is_auth_error(error):
                            raise RuntimeError(
                                f"OpenRouter authentication failed: {error}"
                            ) from error
                        if not self._is_retryable_error(error) and not isinstance(
                            error,
                            ValueError,
                        ):
                            logger.warning(
                                "Model %s failed with non-retryable error: %s; "
                                "trying next model.",
                                model,
                                error,
                            )

                            break

                        logger.warning(
                            "Model %s failed in %s attempt %s: %s",
                            model,
                            mode_label,
                            attempt,
                            error,
                        )

                logger.info(
                    "Model %s exhausted %s; trying next mode/model.",
                    model,
                    mode_label,
                )

        if last_error is not None:
            details = " | ".join(failures[-8:])

            raise RuntimeError(
                "All configured OpenRouter models failed. "
                f"Recent failures: {details}"
            ) from last_error

        raise RuntimeError("No LLM models configured")

    def _complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        model_kwargs = (
            {"response_format": {"type": "json_object"}}
            if json_mode
            else {}
        )

        llm = ChatOpenAI(
            model=model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=0.1,
            max_tokens=max_tokens,
            timeout=self.config.llm_timeout_seconds,
            default_headers={
                "HTTP-Referer": self.config.referer,
                "X-OpenRouter-Title": self.config.app_title,
            },
            model_kwargs=model_kwargs,
        )

        response = llm.invoke(self._to_langchain_messages(messages))

        return self._content_to_text(response)

    def _is_retryable_error(self, error: Exception) -> bool:
        status_code = self._status_code(error)
        if status_code is not None:
            return status_code in self.RETRYABLE_STATUS_CODES

        message = str(error).lower()
        retryable_markers = (
            "timeout",
            "timed out",
            "rate",
            "temporarily",
            "overloaded",
            "connection",
            "json",
        )

        return any(marker in message for marker in retryable_markers)

    def _is_auth_error(self, error: Exception) -> bool:
        status_code = self._status_code(error)
        if status_code is not None:
            return status_code in self.AUTH_STATUS_CODES

        message = str(error).lower()

        return (
            "401" in message
            or "403" in message
            or "unauthorized" in message
            or "forbidden" in message
        )

    def _status_code(self, error: Exception) -> int | None:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(error, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status

        return None

    def _to_langchain_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[tuple[str, str]]:
        role_map = {
            "assistant": "ai",
            "user": "human",
            "system": "system"
        }

        return [
            (role_map.get(message["role"], "human"), message["content"])
            for message in messages
        ]

    def _content_to_text(self, response: BaseMessage) -> str:
        if isinstance(response, AIMessage):
            return response.text

        content = response.content
        if isinstance(content, str):
            return content

        return json.dumps(content, ensure_ascii=False)
