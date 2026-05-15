import logging
import tempfile
from pathlib import Path

import streamlit as st

from analytics_agent.agent import DataAnalysisAgent
from analytics_agent.code_sandbox import CodeSandbox
from analytics_agent.config import AppConfig
from analytics_agent.data_loader import DatasetProfiler, DatasetStorage
from analytics_agent.llm_client import OpenRouterClient
from analytics_agent.prompt_security import PromptInjectionGuard
from analytics_agent.report_writer import MarkdownReportWriter


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def run_app() -> None:
    st.set_page_config(page_title="LLM Data Analyst", page_icon="📊", layout="wide")
    st.title("LLM Data Analyst")
    st.caption("Агент загружает датасет, вызывает Python-интерпретатор и формирует аналитический отчет.")

    uploaded_file = st.file_uploader("Датасет", type=["csv", "xlsx", "xls", "json", "parquet"])
    instruction = st.text_area(
        "Инструкция или контекст",
        placeholder="Например: найди факторы, связанные с выручкой, проверь выбросы и покажи ключевые сегменты.",
        height=120,
    )

    with st.sidebar:
        st.header("Настройки")
        st.write("Провайдер: OpenRouter")
        st.write("Интерпретатор: локальный Python tool")
        show_steps = st.checkbox("Показывать шаги агента", value=True)

    if st.button("Запустить анализ", type="primary", disabled=uploaded_file is None):
        try:
            config = AppConfig.from_environment()
            run_dir = Path(tempfile.mkdtemp(prefix="llm_data_agent_"))
            dataset_path = DatasetStorage(run_dir / "input").save(uploaded_file.name, uploaded_file.getvalue())
            profile = DatasetProfiler().profile(dataset_path)
            local_prompt_check = PromptInjectionGuard().inspect(instruction)
            if local_prompt_check.warnings:
                st.warning("Обнаружены подозрительные управляющие фразы в инструкции. Они будут нейтрализованы перед анализом.")

            sandbox = CodeSandbox(run_dir=run_dir, timeout_seconds=config.code_timeout_seconds)

            with st.status("Агент анализирует данные через Python tool...", expanded=True) as status:
                st.write("Файл сохранен во временную рабочую папку.")
                st.write("Профиль датасета построен приложением, полный анализ выполняет агент.")
                logger.info("Analysis started: file=%s rows=%s columns=%s", uploaded_file.name, profile.rows, profile.columns)

                def progress(message: str) -> None:
                    status.write(message)

                agent = DataAnalysisAgent(
                    config=config,
                    client=OpenRouterClient(config),
                    sandbox=sandbox,
                    progress_callback=progress,
                )
                report = agent.analyze(dataset_path, profile, instruction)
                output_path = run_dir / "report.md"
                MarkdownReportWriter().write(report, output_path)
                status.write(f"Отчет сохранен: {output_path.name}")
                status.update(label="Анализ готов", state="complete")
                logger.info("Analysis finished: model=%s steps=%s artifacts=%s", report.model, len(report.steps), len(report.artifacts))

            st.session_state["report"] = report
            st.session_state["profile"] = profile
            st.session_state["output_path"] = output_path
        except Exception as error:
            st.error(str(error))

    if "report" in st.session_state:
        report = st.session_state["report"]
        profile = st.session_state["profile"]

        st.subheader("Профиль датасета")
        c1, c2, c3 = st.columns(3)
        c1.metric("Строк", profile.rows)
        c2.metric("Колонок", profile.columns)
        c3.metric("Модель", report.model)

        st.subheader("Отчет")
        st.markdown(report.report_markdown)

        if report.metrics:
            st.subheader("Ключевые метрики")
            for metric in report.metrics:
                st.write(f"- {metric}")

        if report.insights:
            st.subheader("Инсайты")
            for insight in report.insights:
                st.write(f"- {insight}")

        if report.artifacts:
            st.subheader("Графики")
            for artifact in report.artifacts:
                if artifact.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    st.image(str(artifact), caption=artifact.name)
                else:
                    st.write(artifact.name)

        if report.limitations:
            st.subheader("Ограничения")
            for limitation in report.limitations:
                st.write(f"- {limitation}")

        if report.security_warnings:
            st.warning("Обнаружены подозрительные управляющие фразы в инструкции. Они были нейтрализованы.")

        if show_steps:
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

        report_text = Path(st.session_state["output_path"]).read_text(encoding="utf-8")
        st.download_button("Скачать Markdown-отчет", report_text, file_name="llm_data_report.md")
