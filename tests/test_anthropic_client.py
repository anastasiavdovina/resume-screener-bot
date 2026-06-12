"""Тесты адаптера Anthropic без сети — SDK-вызов подменяется monkeypatch'ем.

Проверяем извлечение текста из content-блоков, маппинг usage-токенов, применение
prompt caching к системному блоку и заворачивание ошибок SDK в LLMError.
"""

from types import SimpleNamespace

import anthropic
import pytest

from llm.anthropic_client import AnthropicClient
from llm.base import LLMError


def _fake_message(text: str, *, in_tok: int = 12, out_tok: int = 34) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


def test_requires_api_key():
    with pytest.raises(LLMError):
        AnthropicClient("", model="claude-haiku-4-5")


async def test_complete_extracts_text_and_usage_and_caches_system(monkeypatch):
    client = AnthropicClient("sk-ant-test", model="claude-haiku-4-5")
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_message('{"ok": true}')

    monkeypatch.setattr(client._client.messages, "create", fake_create)
    resp = await client.complete("sys-prompt", "user-msg")

    assert resp.text == '{"ok": true}'
    assert resp.input_tokens == 12
    assert resp.output_tokens == 34
    assert resp.latency_s >= 0
    # prompt caching применён к системному блоку
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured["system"][0]["text"] == "sys-prompt"
    assert captured["messages"] == [{"role": "user", "content": "user-msg"}]


async def test_complete_joins_multiple_text_blocks(monkeypatch):
    client = AnthropicClient("sk-ant-test", model="claude-haiku-4-5")
    msg = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="he"),
            SimpleNamespace(type="thinking", text="IGNORED"),  # не text-блок
            SimpleNamespace(type="text", text="llo"),
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )

    async def fake_create(**kwargs):
        return msg

    monkeypatch.setattr(client._client.messages, "create", fake_create)
    resp = await client.complete("s", "u")
    assert resp.text == "hello"


async def test_complete_wraps_sdk_error(monkeypatch):
    client = AnthropicClient("sk-ant-test", model="claude-haiku-4-5")

    async def boom(**kwargs):
        raise anthropic.AnthropicError("API упал")

    monkeypatch.setattr(client._client.messages, "create", boom)
    with pytest.raises(LLMError):
        await client.complete("s", "u")


async def test_complete_none_usage_tokens_coerced_to_zero(monkeypatch):
    client = AnthropicClient("sk-ant-test", model="claude-haiku-4-5")

    async def fake_create(**kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="{}")],
            usage=SimpleNamespace(input_tokens=None, output_tokens=None),
        )

    monkeypatch.setattr(client._client.messages, "create", fake_create)
    resp = await client.complete("s", "u")
    assert resp.input_tokens == 0
    assert resp.output_tokens == 0
