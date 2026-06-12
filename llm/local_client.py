"""LLM-клиент к локальной модели через Ollama (OpenAI-совместимый эндпоинт).

Бесплатная альтернатива Anthropic API: митигация риска исчерпания кредитов Console
и удобство офлайн-отладки. Ollama отдаёт ответ и usage в формате OpenAI Chat
Completions, поэтому используем `openai.AsyncOpenAI` с локальным `base_url`.
"""

from __future__ import annotations

import time

import openai

from llm.base import LLMClient, LLMError, LLMResponse


class LocalClient(LLMClient):
    def __init__(
        self, *, model: str, base_url: str, timeout: float = 60, temperature: float = 0.0
    ) -> None:
        self._model = model
        # temperature=0 -> детерминированнее и точнее для задачи извлечения фактов.
        self._temperature = temperature
        # api_key не используется Ollama, но обязателен для конструктора клиента.
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key="ollama", timeout=timeout)

    async def complete(
        self, system_prompt: str, user_msg: str, *, response_schema: dict | None = None
    ) -> LLMResponse:
        # Structured output: при наличии схемы Ollama гарантирует соответствие ей
        # (строка vs массив, enum-значения и т.п.); иначе — просто валидный JSON.
        if response_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": response_schema},
            }
        else:
            response_format = {"type": "json_object"}
        start = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                response_format=response_format,
                temperature=self._temperature,
            )
        except openai.OpenAIError as e:
            raise LLMError(f"ошибка локальной модели (Ollama): {e}") from e
        latency = time.perf_counter() - start
        # Пустой choices — некорректный ответ; деградируем до "" (как anthropic-клиент),
        # дальше parse_model даст ParseError и сработает ретрай (а не голый IndexError).
        text = resp.choices[0].message.content or "" if resp.choices else ""
        usage = resp.usage
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
        )
