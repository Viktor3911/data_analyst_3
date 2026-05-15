from pathlib import Path

from analytics_agent.agent import AnalysisReport


class MarkdownReportWriter:
    def write(self, report: AnalysisReport, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Отчет LLM-аналитика", "", f"Модель: `{report.model}`", "", report.report_markdown.strip(), ""]

        if report.artifacts:
            lines.append("## Артефакты")
            for artifact in report.artifacts:
                lines.append(f"- {artifact.name}")
            lines.append("")

        if report.security_warnings:
            lines.append("## Предупреждения защиты")
            for warning in report.security_warnings:
                lines.append(f"- {warning}")
            lines.append("")

        output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
