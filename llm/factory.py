"""Выбор LLM-клиента по конфигурации (`LLM_PROVIDER`).

Единственная точка, где конкретный провайдер связывается с настройками. Тяжёлые
клиенты импортируются лениво, чтобы `mock`-режим не тянул SDK.
"""

from __future__ import annotations

from config.settings import Settings
from llm.base import LLMClient
from llm.mock_client import MockLLMClient


def get_client(settings: Settings) -> LLMClient:
    """Создать LLM-клиент согласно `settings.llm_provider`."""
    provider = settings.llm_provider
    if provider == "mock":
        return MockLLMClient()
    if provider == "anthropic":
        from llm.anthropic_client import AnthropicClient

        return AnthropicClient(
            settings.anthropic_api_key or "",
            model=settings.llm_model,
            timeout=settings.request_timeout,
        )
    if provider == "local":
        from llm.local_client import LocalClient

        return LocalClient(
            model=settings.local_model,
            base_url=settings.local_base_url,
            timeout=settings.request_timeout,
        )
    raise ValueError(f"неизвестный LLM_PROVIDER: {provider!r}")
