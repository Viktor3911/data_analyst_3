import logging
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from analytics_agent.agent import AnalysisReport, DataAnalysisAgent
from analytics_agent.code_sandbox import CodeSandbox
from analytics_agent.config import AppConfig
from analytics_agent.data_loader import DatasetProfile, DatasetProfiler, DatasetStorage
from analytics_agent.llm_client import create_llm_client
from analytics_agent.prompt_security import PromptInjectionGuard
from analytics_agent.report_writer import MarkdownReportWriter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StreamlitDataAnalystApp:

    def run(self) -> None:
        self._configure_page()
        uploaded_file, instruction = self._render_inputs()
        show_steps = self._render_sidebar()

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
                show_steps,
            )

    def _configure_page(self) -> None:
        st.set_page_config(page_title="LLM Data Analyst", page_icon="📊", layout="wide")

        st.title("LLM Data Analyst")

        st.caption(
            "Агент загружает датасет, вызывает Python-интерпретатор и "
            "формирует аналитический отчет."
        )

    def _render_inputs(self) -> tuple[Any, str]:
        uploaded_file = st.file_uploader(
            "Датасет",
            type=["csv", "xlsx", "xls", "json", "txt"],
        )
        instruction = st.text_area(
            "Инструкция или контекст",
            placeholder=(
                "Например: найди факторы, связанные с выручкой, проверь "
                "выбросы и покажи ключевые сегменты."
            ),
            height=120,
        )

        return uploaded_file, instruction

    def _render_sidebar(self) -> bool:
        with st.sidebar:
            st.header("Настройки")
            st.write("Интерпретатор: локальный Python tool")

            return st.checkbox("Показывать шаги агента", value=True)

    def _handle_analysis(self, uploaded_file: Any, instruction: str) -> None:
        try:
            config = AppConfig.from_environment()

            run_dir = Path(tempfile.mkdtemp(prefix="llm_data_agent_"))
            dataset_path = DatasetStorage(run_dir / "input").save(
                uploaded_file.name,
                uploaded_file.getvalue(),
            )
            profile = DatasetProfiler().profile(dataset_path)

            self._show_local_prompt_warning(instruction)

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
        except Exception as error:
            st.error(str(error))

    def _show_local_prompt_warning(self, instruction: str) -> None:
        local_prompt_check = PromptInjectionGuard().inspect(instruction)
        if local_prompt_check.warnings:
            st.warning(
                "Обнаружены подозрительные управляющие фразы в инструкции. "
                "Они будут нейтрализованы перед анализом."
            )

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
            "Агент анализирует данные через Python tool...",
            expanded=True,
        ) as status:
            status.write(f"Провайдер LLM: {config.provider}")
            status.write("Файл сохранен во временную рабочую папку.")
            status.write(
                "Профиль датасета построен приложением, полный анализ "
                "выполняет агент."
            )
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
        show_steps: bool,
    ) -> None:
        self._render_profile(profile, report)

        st.subheader("Отчет")

        st.markdown(report.report_markdown)

        self._render_list("Ключевые метрики", report.metrics)
        self._render_list("Инсайты", report.insights)
        self._render_artifacts(report)
        self._render_list("Ограничения", report.limitations)
        self._render_security_warnings(report)

        if show_steps:
            self._render_agent_steps(report)

        report_text = Path(st.session_state["output_path"]).read_text(encoding="utf-8")
        st.download_button(
            "Скачать Markdown-отчет",
            report_text,
            file_name="llm_data_report.md",
        )

    def _render_profile(self, profile: DatasetProfile, report: AnalysisReport) -> None:
        st.subheader("Профиль датасета")

        c1, c2, c3 = st.columns(3)
        c1.metric("Строк", profile.rows)
        c2.metric("Колонок", profile.columns)
        c3.metric("Модель", report.model)

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

    def _render_security_warnings(self, report: AnalysisReport) -> None:
        if report.security_warnings:
            st.warning(
                "Обнаружены подозрительные управляющие фразы в инструкции. "
                "Они были нейтрализованы."
            )

    def _render_agent_steps(self, report: AnalysisReport) -> None:
        st.subheader("Шаги агента")

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
