"""Оркестрация скрининга: разбор резюме (parse) и оценка соответствия (score).

`Screener` — единственная точка бизнес-логики. Он валидирует ввод, рендерит промпты
и вызывает LLM-адаптер, возвращая провалидированные pydantic-модели. О Telegram и о
конкретном провайдере модели сервис ничего не знает.

Промпты внедряются через `PromptSet` (DI); по умолчанию берутся из YAML-бандла
(`llm/prompt_loader.default_prompt_set`). Сами тексты промптов и их загрузка вынесены
в llm-слой — здесь только оркестрация.
"""

from __future__ import annotations

from llm.base import LLMClient, Usage
from llm.prompt_loader import PromptSet, default_prompt_set
from service.schemas import MatchResult, Report, ResumeFields
from service.validation import validate_input

DEFAULT_MAX_INPUT_CHARS = 15000


class Screener:
    """Оркестратор: parse → score. Зависит только от абстрактного `LLMClient`."""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        prompts: PromptSet | None = None,
    ) -> None:
        self._client = client
        self._max_input_chars = max_input_chars
        self._prompts = prompts if prompts is not None else default_prompt_set()

    async def parse(self, resume_text: str, *, usage: Usage | None = None) -> ResumeFields:
        """Разобрать резюме в структуру. Бросает InputValidationError / ParseError / LLMError."""
        cleaned = validate_input(resume_text, "резюме", self._max_input_chars)
        prompt = self._prompts.parse_resume
        return await self._client.complete_model(
            prompt.system, prompt.render(resume=cleaned), ResumeFields, usage=usage
        )

    async def score(
        self, resume: ResumeFields, vacancy_text: str, *, usage: Usage | None = None
    ) -> MatchResult:
        """Оценить соответствие разобранного резюме вакансии.

        Бросает InputValidationError / ParseError / LLMError.
        """
        cleaned = validate_input(vacancy_text, "вакансия", self._max_input_chars)
        prompt = self._prompts.score_match
        user_msg = prompt.render(resume=resume.model_dump_json(indent=2), vacancy=cleaned)
        return await self._client.complete_model(
            prompt.system, user_msg, MatchResult, usage=usage
        )

    async def analyze(
        self, resume_text: str, vacancy_text: str | None = None, *, usage: Usage | None = None
    ) -> Report:
        """Полный сценарий: разбор резюме и (если задана вакансия) оценка соответствия.

        Оба ввода валидируются ДО любого обращения к LLM — битая вакансия не тратит
        вызов parse. ``usage`` (если передан) аккумулирует токены обоих вызовов.
        Бросает InputValidationError / ParseError / LLMError.
        """
        cleaned_resume = validate_input(resume_text, "резюме", self._max_input_chars)
        cleaned_vacancy = (
            validate_input(vacancy_text, "вакансия", self._max_input_chars)
            if vacancy_text is not None
            else None
        )
        resume = await self.parse(cleaned_resume, usage=usage)
        match = (
            await self.score(resume, cleaned_vacancy, usage=usage)
            if cleaned_vacancy is not None
            else None
        )
        return Report(resume=resume, match=match)
