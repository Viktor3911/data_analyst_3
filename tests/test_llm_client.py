from analytics_agent.config import AppConfig
from analytics_agent.providers.gemini import GeminiApiClient
from analytics_agent.providers.openrouter import OpenRouterClient


class FakeOpenRouterClient(OpenRouterClient):
    def __init__(self, config: AppConfig, responses: list[str]) -> None:
        super().__init__(config)
        self.responses = responses
        self.calls: list[tuple[str, bool]] = []

    def _complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        self.calls.append((model, json_mode))
        response = self.responses.pop(0)
        if response == "raise_429":
            error = RuntimeError("Error code: 429")
            setattr(error, "status_code", 429)
            raise error
        return response


def test_complete_json_retries_modes_and_models() -> None:
    config = AppConfig(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="test",
        models=["bad/model", "good/model"],
        referer="test",
        app_title="test",
        max_model_attempts=1,
    )

    client = FakeOpenRouterClient(
        config,
        ["", "not json", "raise_429", '{"type":"final","report_markdown":"ok"}'],
    )

    model, payload = client.complete_json([{"role": "user", "content": "test"}])

    assert model == "good/model"
    assert payload["report_markdown"] == "ok"
    assert client.calls == [
        ("bad/model", True),
        ("bad/model", False),
        ("good/model", True),
        ("good/model", False),
    ]


class FakeGeminiClient:
    def __init__(self) -> None:
        self.clear_calls = 0
        self.generate_calls = []

    def clear_context(self):
        self.clear_calls += 1
        return {"ok": True}

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return {"answer": '{"type":"final","report_markdown":"готово"}'}


def test_gemini_client_clears_context_and_parses_answer() -> None:
    config = AppConfig(
        provider="gemini",
        base_url="https://openrouter.ai/api/v1",
        api_key="",
        models=[],
        referer="test",
        app_title="test",
        gemini_server_url="https://g-assistant-api.ru",
        gemini_user_id=3838,
        gemini_password="password",
        gemini_model="Gemini 3.1 Flash Lite Preview",
    )
    fake_client = FakeGeminiClient()
    client = GeminiApiClient(config, client=fake_client)

    model, payload = client.complete_json(
        [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "analyze"},
        ]
    )

    assert model == "Gemini 3.1 Flash Lite Preview"
    assert payload["report_markdown"] == "готово"
    assert fake_client.clear_calls == 1
    assert fake_client.generate_calls[0]["system_instruction"] == "system rules"
    assert "USER:" in fake_client.generate_calls[0]["message_text"]