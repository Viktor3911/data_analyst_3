import logging
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from analytics_agent.agent import (
    AnalysisReport,
    DataAnalysisAgent,
    UnsafeInstructionError,
)
from analytics_agent.code_sandbox import CodeSandbox
from analytics_agent.config import AppConfig
from analytics_agent.data_loader import DatasetProfile, DatasetProfiler, DatasetStorage
from analytics_agent.llm_client import create_llm_client
from analytics_agent.report_writer import MarkdownReportWriter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StreamlitDataAnalystApp:

    def run(self) -> None:
        self._configure_page()

        self._render_header()
        uploaded_file, instruction = self._render_inputs()

        if st.button(
            "Запустить анализ",
            type="primary",
            disabled=uploaded_file is None,
        ):
            self._handle_analysis(uploaded_file, instruction)

        if "report" in st.session_state:
            self._render_report(
                st.session_state["report"],
                st.session_state["profile"],
            )

    def _configure_page(self) -> None:
        st.set_page_config(page_title="LLM Data Analyst", page_icon="📊", layout="wide")

    def _render_header(self) -> None:
        st.title("LLM Data Analyst")
        st.caption("Загрузите датасет и запустите анализ.")

    def _render_inputs(self) -> tuple[Any, str]:
        uploaded_file = st.file_uploader(
            "Датасет",
            type=["csv", "xlsx", "xls", "json", "txt"],
        )

        instruction = st.text_area(
            "Инструкция или контекст",
            placeholder="Например: найди факторы, связанные с выручкой.",
            height=140,
        )

        return uploaded_file, instruction

    def _handle_analysis(self, uploaded_file: Any, instruction: str) -> None:
        self._clear_analysis_state()
        try:
            config = AppConfig.from_environment()
            run_dir = Path(tempfile.mkdtemp(prefix="llm_data_agent_"))
            dataset_path = DatasetStorage(run_dir / "input").save(
                uploaded_file.name,
                uploaded_file.getvalue(),
            )
            profile = DatasetProfiler().profile(dataset_path)


            report, output_path = self._run_agent(
                config,
                run_dir,
                dataset_path,
                profile,
                instruction,
                uploaded_file.name,
            )
            st.session_state["report"] = report
            st.session_state["profile"] = profile
            st.session_state["output_path"] = output_path
        except UnsafeInstructionError as error:
            st.error(str(error))
        except Exception as error:
            st.error(str(error))

    def _clear_analysis_state(self) -> None:
        for key in ("report", "profile", "output_path"):
            st.session_state.pop(key, None)

    def _run_agent(
        self,
        config: AppConfig,
        run_dir: Path,
        dataset_path: Path,
        profile: DatasetProfile,
        instruction: str,
        uploaded_file_name: str,
    ) -> tuple[AnalysisReport, Path]:
        sandbox = CodeSandbox(
            run_dir=run_dir,
            timeout_seconds=config.code_timeout_seconds,
        )
        llm_client = create_llm_client(config)

        with st.status(
            "Агент анализирует данные...",
            expanded=True,
        ) as status:
            status.write(f"LLM-провайдер: {config.provider}")
            status.write("Файл сохранён во временную рабочую папку.")
            status.write("Профиль датасета построен, агент запускает вычисления.")
            logger.info(
                "Analysis started: file=%s rows=%s columns=%s",
                uploaded_file_name,
                profile.rows,
                profile.columns,
            )

            agent = DataAnalysisAgent(
                config=config,
                client=llm_client,
                sandbox=sandbox,
                progress_callback=status.write,
            )

            report = agent.analyze(dataset_path, profile, instruction)

            output_path = run_dir / "report.md"
            MarkdownReportWriter().write(report, output_path)
            status.write(f"Отчет сохранен: {output_path.name}")

            status.update(label="Анализ готов", state="complete")
            logger.info(
                "Analysis finished: model=%s steps=%s artifacts=%s",
                report.model,
                len(report.steps),
                len(report.artifacts),
            )

            return report, output_path

    def _render_report(
        self,
        report: AnalysisReport,
        profile: DatasetProfile,
    ) -> None:
        self._render_profile(profile, report)

        st.subheader("Отчет")

        st.markdown(report.report_markdown)

        self._render_list("Ключевые метрики", report.metrics)
        self._render_list("Инсайты", report.insights)
        self._render_artifacts(report)
        self._render_list("Ограничения", report.limitations)

        self._render_agent_steps(report)

        report_text = Path(st.session_state["output_path"]).read_text(encoding="utf-8")
        st.download_button(
            "Скачать Markdown-отчет",
            report_text,
            file_name="llm_data_report.md",
        )

    def _render_profile(self, profile: DatasetProfile, report: AnalysisReport) -> None:
        st.subheader("Профиль датасета")
        st.write(f"Строк: {profile.rows}")
        st.write(f"Колонок: {profile.columns}")
        st.write(f"Модель: {report.model}")

    def _render_list(self, title: str, items: list[str]) -> None:
        if not items:
            return

        st.subheader(title)

        for item in items:
            st.write(f"- {item}")

    def _render_artifacts(self, report: AnalysisReport) -> None:
        if not report.artifacts:
            return

        st.subheader("Графики")

        for artifact in report.artifacts:
            if artifact.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.image(str(artifact), caption=artifact.name)
            else:
                st.write(artifact.name)

    def _render_agent_steps(self, report: AnalysisReport) -> None:
        st.subheader("Шаги агента")

        if not report.steps:
            st.caption("Шаги анализа появятся здесь после запуска.")
            return

        for step in report.steps:
            with st.expander(f"Шаг {step.number}: {step.reason}"):
                st.code(step.code, language="python")
                if step.stdout:
                    st.text("stdout")
                    st.code(step.stdout)
                if step.stderr:
                    st.text("stderr")
                    st.code(step.stderr)


def run_app() -> None:
    StreamlitDataAnalystApp().run()
