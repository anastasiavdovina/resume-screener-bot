"""Конфигурация приложения, читаемая из переменных окружения / .env.

Дефолт `llm_provider="mock"` означает, что проект запускается и тестируется
сразу после `git clone`, без каких-либо секретов.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "local", "mock"]


class Settings(BaseSettings):
    """Типизированная конфигурация. Имена полей сопоставляются с ENV без учёта регистра
    (например, поле ``telegram_bot_token`` читается из ``TELEGRAM_BOT_TOKEN``)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Доступы (опциональны: для разработки/тестов на mock они не нужны)
    telegram_bot_token: str | None = None
    anthropic_api_key: str | None = None

    # Выбор и параметры LLM
    llm_provider: LLMProvider = "mock"
    llm_model: str = "claude-haiku-4-5"
    local_model: str = "llama3.1"
    local_base_url: str = "http://localhost:11434/v1"

    # Поведение
    request_timeout: int = Field(default=30, ge=1)
    max_input_chars: int = Field(default=15000, ge=1)


@lru_cache
def get_settings() -> Settings:
    """Singleton-доступ к конфигурации (кэшируется)."""
    return Settings()
