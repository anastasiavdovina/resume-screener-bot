"""Валидация и нормализация пользовательского ввода до обращения к LLM.

Дешёвая проверка на входе экономит и токены, и время: пустой или огромный текст
отсекается сразу, не доходя до модели.
"""

from __future__ import annotations

from service.errors import InputValidationError


def validate_input(text: str, field_name: str, max_chars: int) -> str:
    """Проверить и нормализовать текстовый ввод.

    Возвращает текст без краевых пробелов. Бросает :class:`InputValidationError`,
    если ввод пустой (или только пробелы) либо длиннее ``max_chars`` символов.
    """
    if not text or not text.strip():
        raise InputValidationError(f"{field_name}: пустой ввод")
    cleaned = text.strip()
    if len(cleaned) > max_chars:
        raise InputValidationError(
            f"{field_name}: слишком длинный ввод "
            f"({len(cleaned)} > {max_chars} символов)"
        )
    return cleaned
