"""Тесты фабрики LLM-клиентов: выбор по LLM_PROVIDER, обработка отсутствия ключа."""

import pytest

from config.settings import Settings
from llm.anthropic_client import AnthropicClient
from llm.base import LLMError
from llm.factory import get_client
from llm.local_client import LocalClient
from llm.mock_client import MockLLMClient


def test_factory_returns_mock():
    assert isinstance(get_client(Settings(llm_provider="mock")), MockLLMClient)


def test_factory_anthropic_requires_key():
    settings = Settings(llm_provider="anthropic", anthropic_api_key=None)
    with pytest.raises(LLMError):
        get_client(settings)


def test_factory_returns_anthropic_with_key():
    settings = Settings(
        llm_provider="anthropic", anthropic_api_key="sk-ant-test", llm_model="claude-haiku-4-5"
    )
    assert isinstance(get_client(settings), AnthropicClient)


def test_factory_returns_local():
    assert isinstance(get_client(Settings(llm_provider="local")), LocalClient)


def test_factory_unknown_provider_raises():
    settings = Settings(llm_provider="mock")
    settings.llm_provider = "bogus"  # validate_assignment выключен -> можно подсунуть
    with pytest.raises(ValueError):
        get_client(settings)
