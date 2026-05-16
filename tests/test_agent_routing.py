from pathlib import Path

from analytics_agent.agent import AgentStep, DataAnalysisAgent


def _agent() -> DataAnalysisAgent:
    return DataAnalysisAgent.__new__(DataAnalysisAgent)


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


def test_final_report_with_english_fields_is_repaired(tmp_path: Path) -> None:
    agent = _agent()
    artifact = tmp_path / "chart.png"

    route = agent._route_after_llm(
        {
            "action": {
                "type": "final",
                "report_markdown": "Survival analysis report",
                "metrics": ["Total rows: 891"],
                "insights": ["Women survived more often"],
                "limitations": ["Missing age values"],
            },
            "steps": [_step(artifacts=[artifact])],
            "repair_attempts": 0,
        }
    )

    assert route == "repair"


def test_final_report_after_tool_error_is_repaired(tmp_path: Path) -> None:
    agent = _agent()

    route = agent._route_after_llm(
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
        }
    )

    assert route == "repair"


def test_final_report_without_tool_step_is_repaired() -> None:
    agent = _agent()

    route = agent._route_after_llm(
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
        }
    )

    assert route == "repair"


def test_final_report_with_russian_text_and_artifact_is_done(tmp_path: Path) -> None:
    agent = _agent()
    artifact = tmp_path / "chart.png"

    route = agent._route_after_llm(
        {
            "action": {
                "type": "final",
                "report_markdown": "Отчет по выживаемости готов",
                "metrics": ["Всего строк: 891"],
                "insights": ["Выживаемость выше у женщин"],
                "limitations": ["Есть пропуски возраста"],
            },
            "steps": [_step(artifacts=[artifact])],
            "repair_attempts": 0,
        }
    )

    assert route == "done"
