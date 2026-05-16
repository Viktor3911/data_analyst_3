from analytics_agent.config import AppConfig
from analytics_agent.providers.base import BaseLlmClient
from analytics_agent.providers.gemini import GeminiApiClient
from analytics_agent.providers.openrouter import OpenRouterClient


def create_llm_client(config: AppConfig) -> BaseLlmClient:
    if config.provider == "gemini":
        return GeminiApiClient(config)
    if config.provider == "openrouter":
        return OpenRouterClient(config)

    raise ValueError(f"Unsupported LLM_PROVIDER: {config.provider}")
