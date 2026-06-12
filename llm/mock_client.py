"""Детерминированный LLM-клиент для разработки и тестов без ключей.

Позволяет прогонять весь флоу (service, bot, тесты) на `LLM_PROVIDER=mock`, не тратя
кредиты и не завися от сети. Два режима:

* без ``responses`` — авто-роутинг по системному промпту: на промпт оценки
  возвращается готовый `MatchResult`-JSON, иначе — `ResumeFields`-JSON. Благодаря
  этому полный сценарий `analyze` (parse → score) работает на дефолтном моке;
* со списком ``responses`` — отдаёт их по очереди (последний повторяется, когда список
  исчерпан). Удобно для проверки ретраев: ``["мусор", "<валидный JSON>"]``.
"""

from __future__ import annotations

from llm.base import LLMClient, LLMResponse

# Готовые валидные ответы под схемы service.schemas — используются и в тестах.
MOCK_RESUME_JSON = (
    '{"category": "Python Developer", '
    '"skills": ["Python", "FastAPI", "Docker", "PostgreSQL"], '
    '"years_experience": 4, "grade": "middle", '
    '"education": "МГТУ им. Баумана, ПМИ", '
    '"summary": "Backend-разработчик на Python с опытом микросервисов.", '
    '"language": "ru"}'
)

MOCK_MATCH_JSON = (
    '{"score": 78, "verdict": "match", '
    '"strengths": ["Опыт с FastAPI и Docker", "Релевантный стек"], '
    '"gaps": ["Нет опыта с Kubernetes"], '
    '"explanation": "Кандидат закрывает ключевые требования вакансии."}'
)


class MockLLMClient(LLMClient):
    """Возвращает заранее заданные ответы, не обращаясь ни к какой модели.

    Курсор по ``responses`` — общий для инстанса, поэтому seeded-режим рассчитан на
    один поток (один ``complete_model``). Для конкурентных сценариев заводите
    отдельный инстанс на каждый поток.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        resume_text: str = MOCK_RESUME_JSON,
        match_text: str = MOCK_MATCH_JSON,
        input_tokens: int = 10,
        output_tokens: int = 20,
        latency_s: float = 0.0,
    ) -> None:
        self._responses = list(responses) if responses else None
        self._resume_text = resume_text
        self._match_text = match_text
        self._idx = 0
        self._meta = (input_tokens, output_tokens, latency_s)
        # История вызовов (system_prompt, user_msg) — для ассертов в тестах.
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _is_scoring_prompt(system_prompt: str) -> bool:
        """Эвристика: распознаём промпт оценки соответствия по ключевым словам."""
        s = system_prompt.lower()
        return "verdict" in s or "соответстви" in s

    async def complete(
        self, system_prompt: str, user_msg: str, *, response_schema: dict | None = None
    ) -> LLMResponse:
        self.calls.append((system_prompt, user_msg))
        if self._responses:
            idx = min(self._idx, len(self._responses) - 1)  # последний повторяется
            text = self._responses[idx]
            self._idx += 1
        elif self._is_scoring_prompt(system_prompt):
            text = self._match_text
        else:
            text = self._resume_text
        input_tokens, output_tokens, latency_s = self._meta
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency_s,
        )
