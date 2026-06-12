"""Промпты как версионируемый конфиг: типы `Prompt`/`PromptSet` и загрузчик YAML.

Промпты живут в llm-слое (как мы инструктируем модель), а не в бизнес-логике. Сервис
получает готовый `PromptSet` и не знает, откуда тот взялся — из YAML или собран в коде.
Это закрывает критерий «конфигурация вынесена» (Блок 3): чтобы поменять формулировку,
правится YAML-файл, а не код.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# Плейсхолдеры вида {key} в шаблонах промптов.
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

# Каталог с YAML-промптами (рядом с этим модулем).
PROMPTS_DIR = Path(__file__).parent / "prompts"

_REQUIRED_KEYS = ("name", "version", "system", "user_template")


@dataclass(frozen=True)
class Prompt:
    """Один промпт: метаданные + системная инструкция + шаблон сообщения пользователя."""

    name: str
    version: int
    system: str
    user_template: str

    def render(self, **values: str) -> str:
        """Подставить значения в шаблон за один проход (``re.sub`` по шаблону).

        Сканируется только сам шаблон, подставленный текст не пересканируется. Поэтому
        фигурные скобки в JSON-резюме/вакансии безопасны (в отличие от ``str.format``),
        и плейсхолдер-токен, случайно оказавшийся внутри значения (например литерал
        ``{vacancy}`` в тексте резюме), не затрагивается другой подстановкой.
        Неизвестные плейсхолдеры остаются как есть.
        """

        def _replace(match: re.Match[str]) -> str:
            return values.get(match.group(1), match.group(0))

        return _PLACEHOLDER_RE.sub(_replace, self.user_template)


@dataclass(frozen=True)
class PromptSet:
    """Набор промптов, нужных сервису."""

    parse_resume: Prompt
    score_match: Prompt


def _load_one(path: Path) -> Prompt:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: ожидался YAML-объект")
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"{path.name}: отсутствуют обязательные ключи {missing}")
    return Prompt(
        name=str(data["name"]),
        version=int(data["version"]),
        system=str(data["system"]),
        user_template=str(data["user_template"]),
    )


def load_prompts(directory: Path | None = None) -> PromptSet:
    """Загрузить набор промптов из YAML-файлов каталога (по умолчанию — `PROMPTS_DIR`)."""
    d = directory or PROMPTS_DIR
    return PromptSet(
        parse_resume=_load_one(d / "parse_resume.yaml"),
        score_match=_load_one(d / "score_match.yaml"),
    )


@lru_cache
def default_prompt_set() -> PromptSet:
    """Промпты из бандла (кэшируется): один раз читаем YAML за процесс."""
    return load_prompts()


def load_classify_prompt(directory: Path | None = None) -> Prompt:
    """Загрузить промпт строгой классификации (только для eval-метрик).

    Плейсхолдеры шаблона: ``{categories}`` (закрытый список меток из датасета) и
    ``{resume}``. Не входит в `PromptSet`, т.к. боту не нужен.
    """
    d = directory or PROMPTS_DIR
    return _load_one(d / "classify.yaml")
