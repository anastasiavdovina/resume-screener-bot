"""Реальный LLM-клиент поверх Anthropic API (асинхронный).

Читает usage-токены из ответа (для метрик стоимости) и кеширует системный промпт
через ``cache_control: ephemeral`` (снижает стоимость повторных вызовов с тем же
системным промптом). Все ошибки SDK заворачиваются в доменную :class:`LLMError`.
"""

from __future__ import annotations

import time

import anthropic

from llm.base import LLMClient, LLMError, LLMResponse

DEFAULT_MAX_TOKENS = 1024


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        timeout: float = 30,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY не задан — нельзя создать AnthropicClient")
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)

    async def complete(
        self, system_prompt: str, user_msg: str, *, response_schema: dict | None = None
    ) -> LLMResponse:
        # response_schema не используется: модели Anthropic надёжно следуют JSON по
        # промпту + ретрай в complete_model. Параметр принят для совместимости интерфейса.
        start = time.perf_counter()
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                # system как список блоков с cache_control -> prompt caching
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
        except anthropic.AnthropicError as e:
            raise LLMError(f"ошибка Anthropic API: {e}") from e
        latency = time.perf_counter() - start
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        usage = resp.usage
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_s=latency,
        )
