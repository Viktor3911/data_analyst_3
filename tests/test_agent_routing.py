from pathlib import Path

import pytest

from analytics_agent.agent import AgentStep, DataAnalysisAgent, UnsafeInstructionError
from analytics_agent.prompt_security import PromptSafetyAssessment


def _agent() -> DataAnalysisAgent:
    return DataAnalysisAgent.__new__(DataAnalysisAgent)


def _assessment(*, is_malicious: bool) -> PromptSafetyAssessment:
    return PromptSafetyAssessment(is_malicious=is_malicious)


def _step(*, stderr: str = "", artifacts: list[Path] | None = None) -> AgentStep:
    return AgentStep(
        number=1,
        model="test-model",
        reason="проверка",
        code="print('ok')",
        stdout="ok",
        stderr=stderr,
        artifacts=artifacts or [],
    )


@pytest.mark.parametrize(
    ("state", "expected_route"),
    [
        (
            {
                "action": {
                    "type": "final",
                    "report_markdown": "Survival analysis report",
                    "metrics": ["Total rows: 891"],
                    "insights": ["Women survived more often"],
                    "limitations": ["Missing age values"],
                },
                "steps": [_step(artifacts=[Path("chart.png")])],
                "repair_attempts": 0,
            },
            "repair",
        ),
        (
            {
                "action": {
                    "type": "final",
                    "report_markdown": "Отчет готов",
                    "metrics": ["Всего строк: 891"],
                    "insights": ["Выживаемость выше у женщин"],
                    "limitations": ["Есть пропуски возраста"],
                },
                "steps": [_step(stderr="Traceback")],
                "repair_attempts": 0,
            },
            "repair",
        ),
        (
            {
                "action": {
                    "type": "final",
                    "report_markdown": "Отчет готов",
                    "metrics": ["Всего строк: 891"],
                    "insights": ["Выживаемость выше у женщин"],
                    "limitations": ["Есть пропуски возраста"],
                },
                "steps": [],
                "repair_attempts": 0,
            },
            "repair",
        ),
        (
            {
                "action": {
                    "type": "final",
                    "report_markdown": "Отчет по выживаемости готов",
                    "metrics": ["Всего строк: 891"],
                    "insights": ["Выживаемость выше у женщин"],
                    "limitations": ["Есть пропуски возраста"],
                },
                "steps": [_step(artifacts=[Path("chart.png")])],
                "repair_attempts": 0,
            },
            "done",
        ),
    ],
)
def test_route_after_llm_for_final_reports(
    state: dict[str, object],
    expected_route: str,
) -> None:
    agent = _agent()
    assert agent._route_after_llm(state) == expected_route


def test_malicious_instruction_blocks_agent_graph(tmp_path: Path) -> None:
    agent = _agent()
    agent._progress = lambda message: None

    class FakeGuard:
        def inspect(self, instruction: str) -> PromptSafetyAssessment:
            return _assessment(is_malicious=True)

    def fail_build_graph() -> None:
        raise AssertionError("graph should not be built for malicious prompts")

    agent.llm_guard = FakeGuard()
    agent._build_graph = fail_build_graph

    with pytest.raises(UnsafeInstructionError):
        agent.analyze(tmp_path / "data.csv", None, "плохой промпт")
