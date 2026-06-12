"""Обработчики Telegram (aiogram): команды, FSM-диалог, обработка ошибок.

UI-сообщения (приветствие, подсказки, ошибки) — билингва (RU + EN в одном сообщении),
команды бота работают в любом состоянии. Результат анализа форматируется на языке
резюме (см. `bot.formatting`). Любая ошибка сервиса ловится и превращается в сообщение —
бот не падает.

Сервис `Screener` инжектится aiogram'ом из данных диспетчера (`dp["screener"]`).
"""

from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.formatting import format_report
from bot.states import Screening
from llm.base import LLMError, ParseError
from service.errors import InputValidationError
from service.screener import Screener

logger = logging.getLogger(__name__)
router = Router()

MSG_START = (
    "👋 Привет! Я бот-скринер резюме.\n"
    "/analyze — структурный разбор резюме\n"
    "/match — оценка соответствия резюме и вакансии\n"
    "/help — подробнее\n\n"
    "👋 Hi! I'm a resume screening bot.\n"
    "/analyze — structured resume breakdown\n"
    "/match — resume-vacancy match assessment\n"
    "/help — details"
)
MSG_HELP = (
    "Как пользоваться:\n"
    "• /analyze — пришлите текст резюме, получите структурный разбор.\n"
    "• /match — пришлите резюме, затем вакансию, получите оценку соответствия.\n"
    "• /cancel — отменить текущий диалог.\n\n"
    "How to use:\n"
    "• /analyze — send a resume, get a structured breakdown.\n"
    "• /match — send a resume, then a vacancy, get a match assessment.\n"
    "• /cancel — cancel the current dialog."
)
MSG_CANCEL = "Отменено. / Cancelled."
MSG_SEND_RESUME = "Пришлите текст резюме одним сообщением.\nSend the resume text in one message."
MSG_SEND_VACANCY = "Теперь пришлите текст вакансии.\nNow send the vacancy text."
MSG_NEED_TEXT = "Пришлите, пожалуйста, текст.\nPlease send text."
MSG_FALLBACK = "Используйте /analyze или /match.\nUse /analyze or /match."
MSG_BAD_INPUT = "⚠️ Ввод пустой или слишком длинный.\n⚠️ Input is empty or too long."
MSG_PARSE_ERROR = (
    "🤔 Не удалось разобрать ответ модели. Попробуйте ещё раз.\n"
    "🤔 Could not parse the model's response. Please try again."
)
MSG_LLM_ERROR = (
    "⚠️ Сервис модели временно недоступен. Попробуйте позже.\n"
    "⚠️ The model service is temporarily unavailable. Try again later."
)
MSG_UNEXPECTED = "❌ Что-то пошло не так.\n❌ Something went wrong."


async def _respond(
    message: Message, screener: Screener, resume_text: str, vacancy_text: str | None
) -> None:
    """Прогнать анализ и отправить результат либо понятное сообщение об ошибке."""
    try:
        report = await screener.analyze(resume_text, vacancy_text)
    except InputValidationError:
        await message.answer(MSG_BAD_INPUT)
    except ParseError:  # подкласс LLMError — ловим раньше LLMError
        await message.answer(MSG_PARSE_ERROR)
    except LLMError:
        await message.answer(MSG_LLM_ERROR)
    except Exception:
        logger.exception("Неожиданная ошибка при анализе")
        await message.answer(MSG_UNEXPECTED)
    else:
        await message.answer(format_report(report))


# --- Команды (работают в любом состоянии; регистрируются раньше FSM-хендлеров) ---


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(MSG_START)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(MSG_HELP)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(MSG_CANCEL)


@router.message(Command("analyze"))
async def cmd_analyze(message: Message, state: FSMContext) -> None:
    await state.set_state(Screening.analyze_resume)
    await message.answer(MSG_SEND_RESUME)


@router.message(Command("match"))
async def cmd_match(message: Message, state: FSMContext) -> None:
    await state.set_state(Screening.match_resume)
    await message.answer(MSG_SEND_RESUME)


# --- FSM-шаги ---


@router.message(Screening.analyze_resume)
async def on_analyze_resume(message: Message, state: FSMContext, screener: Screener) -> None:
    if not message.text:
        await message.answer(MSG_NEED_TEXT)
        return
    await _respond(message, screener, message.text, None)
    await state.clear()


@router.message(Screening.match_resume)
async def on_match_resume(message: Message, state: FSMContext) -> None:
    if not message.text:
        await message.answer(MSG_NEED_TEXT)
        return
    await state.update_data(resume=message.text)
    await state.set_state(Screening.match_vacancy)
    await message.answer(MSG_SEND_VACANCY)


@router.message(Screening.match_vacancy)
async def on_match_vacancy(message: Message, state: FSMContext, screener: Screener) -> None:
    if not message.text:
        await message.answer(MSG_NEED_TEXT)
        return
    data = await state.get_data()
    await _respond(message, screener, data.get("resume", ""), message.text)
    await state.clear()


@router.message(StateFilter(None))
async def fallback(message: Message) -> None:
    await message.answer(MSG_FALLBACK)
