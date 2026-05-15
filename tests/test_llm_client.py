from analytics_agent.config import AppConfig
from analytics_agent.llm_client import OpenRouterClient


class FakeOpenRouterClient(OpenRouterClient):
    def __init__(self, config: AppConfig, responses: list[str]) -> None:
        super().__init__(config)
        self.responses = responses
        self.calls: list[tuple[str, bool]] = []

    def _complete(self, model: str, messages: list[dict[str, str]], max_tokens: int, json_mode: bool) -> str:
        self.calls.append((model, json_mode))
        response = self.responses.pop(0)
        if response == "raise_429":
            error = RuntimeError("Error code: 429")
            setattr(error, "status_code", 429)
            raise error
        return response


def test_complete_json_retries_modes_and_models() -> None:
    config = AppConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key="test",
        models=["bad/model", "good/model"],
        referer="test",
        app_title="test",
        max_model_attempts=1,
    )
    client = FakeOpenRouterClient(config, ["", "not json", "raise_429", '{"type":"final","report_markdown":"ok"}'])

    model, payload = client.complete_json([{"role": "user", "content": "test"}])

    assert model == "good/model"
    assert payload["report_markdown"] == "ok"
    assert client.calls == [("bad/model", True), ("bad/model", False), ("good/model", True), ("good/model", False)]