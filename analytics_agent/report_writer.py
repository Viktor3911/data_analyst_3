from pathlib import Path

from analytics_agent.agent import AnalysisReport


class MarkdownReportWriter:

    def write(self, report: AnalysisReport, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Отчет LLM-аналитика",
            "",
            f"Модель: `{report.model}`",
            "",
            report.report_markdown.strip(),
            "",
        ]

        self._append_list(lines, "Ключевые метрики", report.metrics)
        self._append_list(lines, "Инсайты", report.insights)
        self._append_list(lines, "Ограничения", report.limitations)

        if report.artifacts:
            lines.append("## Артефакты")
            for artifact in report.artifacts:
                lines.append(f"- {artifact.name}")
            lines.append("")

        output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _append_list(
        self,
        lines: list[str],
        title: str,
        items: list[str],
    ) -> None:
        if not items:
            return

        lines.append(f"## {title}")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
