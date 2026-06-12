"""Тесты адаптера локальной модели (Ollama) без сети — SDK подменяется monkeypatch'ем."""

from types import SimpleNamespace

import openai
import pytest

from llm.base import LLMError
from llm.local_client import LocalClient


def _fake_completion(text: str, *, prompt_tok: int = 5, completion_tok: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tok, completion_tokens=completion_tok),
    )


async def test_complete_extracts_text_and_usage(monkeypatch):
    client = LocalClient(model="llama3.1", base_url="http://localhost:11434/v1")
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_completion('{"ok": true}')

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    resp = await client.complete("sys", "user")

    assert resp.text == '{"ok": true}'
    assert resp.input_tokens == 5
    assert resp.output_tokens == 7
    assert resp.latency_s >= 0
    assert captured["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


async def test_complete_handles_none_content(monkeypatch):
    client = LocalClient(model="llama3.1", base_url="http://localhost:11434/v1")

    async def fake_create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    resp = await client.complete("s", "u")
    assert resp.text == ""


async def test_complete_wraps_sdk_error(monkeypatch):
    client = LocalClient(model="llama3.1", base_url="http://localhost:11434/v1")

    async def boom(**kwargs):
        raise openai.OpenAIError("Ollama недоступна")

    monkeypatch.setattr(client._client.chat.completions, "create", boom)
    with pytest.raises(LLMError):
        await client.complete("s", "u")


async def test_complete_none_usage_tokens_coerced_to_zero(monkeypatch):
    client = LocalClient(model="llama3.1", base_url="http://localhost:11434/v1")

    async def fake_create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            usage=SimpleNamespace(prompt_tokens=None, completion_tokens=None),
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    resp = await client.complete("s", "u")
    assert resp.input_tokens == 0
    assert resp.output_tokens == 0


async def test_complete_empty_choices_degrades_to_empty_text(monkeypatch):
    client = LocalClient(model="llama3.1", base_url="http://localhost:11434/v1")

    async def fake_create(**kwargs):
        return SimpleNamespace(
            choices=[], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    resp = await client.complete("s", "u")  # не падает IndexError
    assert resp.text == ""
