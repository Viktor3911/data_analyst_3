import pytest

from analytics_agent.prompt_security import LlmPromptInjectionGuard


@pytest.mark.parametrize(
    ("classifier_response", "expected_is_malicious"),
    [
        (
            {"is_malicious": True},
            True,
        ),
        (
            {"is_malicious": False},
            False,
        ),
    ],
)
def test_llm_prompt_guard_uses_classifier_only(
    classifier_response: dict[str, object],
    expected_is_malicious: bool,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.messages = None

        def complete_json(self, messages):
            self.messages = messages
            return "fake/model", classifier_response

    client = FakeClient()
    result = LlmPromptInjectionGuard(client).inspect(
        "Проанализируй выживаемость пассажиров Titanic по полу."
    )

    assert result.is_malicious is expected_is_malicious
    assert client.messages[1]["content"].startswith("Untrusted instruction to classify:\n")
    assert "Titanic" in client.messages[1]["content"]
