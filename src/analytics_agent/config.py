import os
from dataclasses import dataclass
from pathlib import Path

from analytics_agent.env import DotEnvLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = "openrouter/free,deepseek/deepseek-v4-flash:free,google/gemma-4-26b-a4b-it:free,nvidia/nemotron-3-super-120b-a12b:free,minimax/minimax-m2.5:free,openai/gpt-oss-120b:free"
OPENROUTER_REFERER = "https://github.com/data-analyst-agent-product"
OPENROUTER_APP_TITLE = "LLM Data Analyst Agent"


@dataclass(frozen=True)
class AppConfig:
    base_url: str
    api_key: str
    models: list[str]
    referer: str
    app_title: str
    max_agent_steps: int = 4
    code_timeout_seconds: int = 20
    llm_timeout_seconds: int = 30
    max_model_attempts: int = 1

    @classmethod
    def from_environment(cls) -> "AppConfig":
        DotEnvLoader(PROJECT_ROOT / ".env").load()

        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise ValueError("LLM_API_KEY is required. Add it to .env or set it as an environment variable.")

        return cls(
            base_url=DEFAULT_BASE_URL,
            api_key=api_key,
            models=parse_model_list(os.getenv("LLM_MODELS", DEFAULT_MODELS)),
            referer=OPENROUTER_REFERER,
            app_title=OPENROUTER_APP_TITLE,
            llm_timeout_seconds=parse_positive_int(os.getenv("LLM_TIMEOUT_SECONDS", "30"), default=30),
            max_model_attempts=parse_positive_int(os.getenv("LLM_MAX_MODEL_ATTEMPTS", "1"), default=1),
        )


def parse_model_list(raw_models: str) -> list[str]:
    return list(dict.fromkeys(model.strip() for model in raw_models.split(",") if model.strip()))


def parse_positive_int(raw_value: str, default: int) -> int:
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default
