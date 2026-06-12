"""FSM-состояния диалога бота."""

from __future__ import annotations

from aiogram.fsm.state import State, StatesGroup


class Screening(StatesGroup):
    """Шаги сбора ввода у пользователя."""

    analyze_resume = State()  # /analyze: ждём текст резюме
    match_resume = State()    # /match: ждём текст резюме (шаг 1/2)
    match_vacancy = State()   # /match: ждём текст вакансии (шаг 2/2)
