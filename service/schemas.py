"""Pydantic-схемы — контракт между LLM и остальным кодом.

Если модель вернёт структуру, не соответствующую этим схемам, валидация упадёт
с понятной ошибкой (её ловит слой сервиса). Опциональные поля (``| None``) — это
анти-галлюцинационная мера: «нет данных в тексте» → ``null``, а не выдумка.

Замечание по дизайну: ``category`` — свободная строка (универсальная догадка модели
о профессии), а НЕ enum из 25 категорий Kaggle. Закрытый список меток для метрик
живёт отдельно в eval-слое (``classify.yaml``) и грузится из реального CSV — так код
не привязан к конкретному варианту датасета и не ломается при расхождении строк.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Непустая строка: обрезаем пробелы и требуем непустой результат — так required-поля
# защищены не только от "", но и от пробельного вывода LLM ("  ", "\n"), который иначе
# давал бы внешне валидный, но блёклый отчёт.
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

Grade = Literal["junior", "middle", "senior", "lead"]
Verdict = Literal["match", "maybe", "no_match"]
Language = Literal["ru", "en"]


class ResumeFields(BaseModel):
    """Структурированный разбор резюме, извлечённый LLM."""

    # extra="ignore": модель может добавить лишние поля — это не ошибка, просто игнорируем.
    model_config = ConfigDict(extra="ignore")

    category: NonBlankStr = Field(description="Профессиональная категория (догадка модели)")
    skills: list[str] = Field(default_factory=list, description="Извлечённые навыки")
    years_experience: float | None = Field(
        default=None, ge=0, description="Суммарный опыт в годах; null если не указан"
    )
    grade: Grade | None = Field(default=None, description="Грейд; null если не определить")
    education: str | None = Field(default=None, description="Образование; null если не указано")
    summary: NonBlankStr = Field(description="Краткий разбор резюме, 2-3 предложения")
    language: Language = Field(description="Определённый язык резюме")


class MatchResult(BaseModel):
    """Оценка соответствия резюме вакансии."""

    model_config = ConfigDict(extra="ignore")

    score: int = Field(ge=0, le=100, description="Скор соответствия, 0-100")
    verdict: Verdict = Field(description="Итоговый вердикт")
    strengths: list[str] = Field(default_factory=list, description="Сильные стороны под вакансию")
    gaps: list[str] = Field(default_factory=list, description="Чего не хватает под вакансию")
    explanation: NonBlankStr = Field(description="Обоснование оценки")


class Report(BaseModel):
    """Итоговый ответ сервиса: разбор резюме и (опционально) оценка соответствия."""

    resume: ResumeFields
    match: MatchResult | None = None
