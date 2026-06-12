"""Тесты хендлеров бота на замоканных Message/FSMContext и реальном Screener(mock)."""

from unittest.mock import AsyncMock, MagicMock

from bot.handlers import (
    MSG_BAD_INPUT,
    MSG_CANCEL,
    MSG_LLM_ERROR,
    MSG_NEED_TEXT,
    MSG_PARSE_ERROR,
    MSG_SEND_VACANCY,
    MSG_UNEXPECTED,
    cmd_analyze,
    cmd_cancel,
    cmd_help,
    cmd_match,
    cmd_start,
    fallback,
    on_analyze_resume,
    on_match_resume,
    on_match_vacancy,
)
from bot.states import Screening
from llm.base import LLMClient, LLMError, LLMResponse
from llm.mock_client import MockLLMClient
from service.screener import Screener


class _RaisingClient(LLMClient):
    """LLM-клиент, всегда падающий заданным исключением — для веток обработки ошибок."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def complete(
        self, system_prompt: str, user_msg: str, *, response_schema: dict | None = None
    ) -> LLMResponse:
        raise self._exc


def make_message(text=None):
    m = MagicMock()
    m.text = text
    m.answer = AsyncMock()
    return m


def make_state(data=None):
    s = MagicMock()
    s.set_state = AsyncMock()
    s.clear = AsyncMock()
    s.update_data = AsyncMock()
    s.get_data = AsyncMock(return_value=data or {})
    return s


def last_answer(message):
    return message.answer.await_args.args[0]


# --- команды --------------------------------------------------------------


async def test_cmd_start_clears_state_and_greets():
    msg, state = make_message("/start"), make_state()
    await cmd_start(msg, state)
    state.clear.assert_awaited_once()
    msg.answer.assert_awaited_once()
    assert "/analyze" in last_answer(msg)


async def test_cmd_help_sends_help():
    msg = make_message("/help")
    await cmd_help(msg)
    assert "/analyze" in last_answer(msg)


async def test_cmd_cancel_clears_state_and_replies():
    msg, state = make_message("/cancel"), make_state()
    await cmd_cancel(msg, state)
    state.clear.assert_awaited_once()
    assert last_answer(msg) == MSG_CANCEL


async def test_cmd_analyze_sets_state_and_prompts():
    msg, state = make_message("/analyze"), make_state()
    await cmd_analyze(msg, state)
    state.set_state.assert_awaited_once_with(Screening.analyze_resume)
    assert "резюме" in last_answer(msg).lower()


async def test_cmd_match_sets_state_and_prompts():
    msg, state = make_message("/match"), make_state()
    await cmd_match(msg, state)
    state.set_state.assert_awaited_once_with(Screening.match_resume)
    assert "резюме" in last_answer(msg).lower()


# --- /analyze flow --------------------------------------------------------


async def test_analyze_resume_happy_path():
    msg, state = make_message("текст резюме кандидата"), make_state()
    await on_analyze_resume(msg, state, Screener(MockLLMClient()))
    out = last_answer(msg)
    assert "Категория: Python Developer" in out
    assert "🎯" not in out  # только разбор, без оценки
    state.clear.assert_awaited_once()


async def test_analyze_resume_non_text_asks_for_text():
    msg, state = make_message(None), make_state()
    await on_analyze_resume(msg, state, Screener(MockLLMClient()))
    assert last_answer(msg) == MSG_NEED_TEXT


# --- /match flow (два шага) ------------------------------------------------


async def test_match_flow_two_steps():
    screener = Screener(MockLLMClient())  # дефолтный мок роутит parse, затем score

    msg1, state = make_message("текст резюме"), make_state()
    await on_match_resume(msg1, state)
    state.update_data.assert_awaited_once_with(resume="текст резюме")
    state.set_state.assert_awaited_with(Screening.match_vacancy)
    assert last_answer(msg1) == MSG_SEND_VACANCY

    msg2, state2 = make_message("текст вакансии"), make_state({"resume": "текст резюме"})
    await on_match_vacancy(msg2, state2, screener)
    out = last_answer(msg2)
    assert "🎯 Соответствие вакансии" in out
    assert "Скор:" in out
    state2.clear.assert_awaited_once()


# --- обработка ошибок ------------------------------------------------------


async def test_bad_input_maps_to_message():
    msg, state = make_message("   "), make_state()  # пробелы -> InputValidationError
    await on_analyze_resume(msg, state, Screener(MockLLMClient()))
    assert last_answer(msg) == MSG_BAD_INPUT


async def test_parse_error_maps_to_message():
    msg, state = make_message("резюме"), make_state()
    screener = Screener(MockLLMClient(["мусор", "снова мусор"]))  # никогда не валиден
    await on_analyze_resume(msg, state, screener)
    assert last_answer(msg) == MSG_PARSE_ERROR


async def test_llm_error_maps_to_message():
    msg, state = make_message("резюме"), make_state()
    await on_analyze_resume(msg, state, Screener(_RaisingClient(LLMError("api down"))))
    assert last_answer(msg) == MSG_LLM_ERROR


async def test_unexpected_error_maps_to_message():
    msg, state = make_message("резюме"), make_state()
    await on_analyze_resume(msg, state, Screener(_RaisingClient(RuntimeError("boom"))))
    assert last_answer(msg) == MSG_UNEXPECTED


async def test_match_vacancy_missing_resume_data_maps_to_bad_input():
    # state без ключа resume (шаги вне порядка) -> "" -> InputValidationError -> BAD_INPUT
    msg, state = make_message("вакансия"), make_state({})
    await on_match_vacancy(msg, state, Screener(MockLLMClient()))
    assert last_answer(msg) == MSG_BAD_INPUT
    state.clear.assert_awaited_once()


async def test_fallback_suggests_commands():
    msg = make_message("просто болтаю")
    await fallback(msg)
    assert "analyze" in last_answer(msg)
