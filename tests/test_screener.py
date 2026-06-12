"""Тесты оркестрации Screener на MockLLMClient (без сети и ключей)."""

import pytest

from llm.base import ParseError, Usage
from llm.mock_client import MOCK_MATCH_JSON, MOCK_RESUME_JSON, MockLLMClient
from service.errors import InputValidationError
from service.schemas import MatchResult, ResumeFields
from service.screener import Screener

RESUME_TEXT = "Иван Петров. Python-разработчик, 4 года, FastAPI, Docker."
VACANCY_TEXT = "Ищем middle Python backend: FastAPI, PostgreSQL, Docker."


# --- parse / score --------------------------------------------------------


async def test_parse_returns_resume_fields():
    screener = Screener(MockLLMClient([MOCK_RESUME_JSON]))
    r = await screener.parse(RESUME_TEXT)
    assert isinstance(r, ResumeFields)
    assert r.category == "Python Developer"


async def test_parse_sends_nonempty_system_prompt():
    client = MockLLMClient([MOCK_RESUME_JSON])
    await Screener(client).parse(RESUME_TEXT)
    system_prompt, user_msg = client.calls[0]
    assert system_prompt.strip()  # промпт реально передан
    assert RESUME_TEXT in user_msg  # текст резюме подставлен в шаблон


async def test_score_returns_match_result():
    screener = Screener(MockLLMClient([MOCK_MATCH_JSON]))
    resume = ResumeFields(category="Python Developer", summary="x", language="ru")
    m = await screener.score(resume, VACANCY_TEXT)
    assert isinstance(m, MatchResult)
    assert m.verdict == "match"


# --- analyze --------------------------------------------------------------


async def test_analyze_resume_only():
    client = MockLLMClient([MOCK_RESUME_JSON])
    report = await Screener(client).analyze(RESUME_TEXT)  # vacancy=None
    assert report.match is None
    assert report.resume.category == "Python Developer"
    assert len(client.calls) == 1  # только parse


async def test_analyze_with_vacancy():
    client = MockLLMClient([MOCK_RESUME_JSON, MOCK_MATCH_JSON])
    report = await Screener(client).analyze(RESUME_TEXT, VACANCY_TEXT)
    assert report.match is not None
    assert report.match.score == 78
    assert len(client.calls) == 2  # parse + score


async def test_analyze_accumulates_usage():
    client = MockLLMClient([MOCK_RESUME_JSON, MOCK_MATCH_JSON], input_tokens=10, output_tokens=20)
    usage = Usage()
    await Screener(client).analyze(RESUME_TEXT, VACANCY_TEXT, usage=usage)
    assert usage.calls == 2
    assert usage.input_tokens == 20  # 10 * 2 вызова
    assert usage.output_tokens == 40  # 20 * 2 вызова


# --- валидация ввода ------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
async def test_parse_empty_input_rejected_without_llm_call(bad):
    client = MockLLMClient([MOCK_RESUME_JSON])
    with pytest.raises(InputValidationError):
        await Screener(client).parse(bad)
    assert len(client.calls) == 0  # до модели не дошли


async def test_parse_too_long_input_rejected():
    client = MockLLMClient([MOCK_RESUME_JSON])
    screener = Screener(client, max_input_chars=10)
    with pytest.raises(InputValidationError):
        await screener.parse("a" * 11)
    assert len(client.calls) == 0


async def test_score_empty_vacancy_rejected():
    client = MockLLMClient([MOCK_MATCH_JSON])
    resume = ResumeFields(category="HR", summary="x", language="ru")
    with pytest.raises(InputValidationError):
        await Screener(client).score(resume, "   ")
    assert len(client.calls) == 0


# --- проброс ошибок модели ------------------------------------------------


async def test_parse_propagates_parse_error_after_retries():
    client = MockLLMClient(["мусор", "снова мусор"])  # никогда не валиден
    with pytest.raises(ParseError):
        await Screener(client).parse(RESUME_TEXT)


async def test_score_propagates_parse_error_after_retries():
    client = MockLLMClient(["мусор", "снова мусор"])
    resume = ResumeFields(category="HR", summary="x", language="ru")
    with pytest.raises(ParseError):
        await Screener(client).score(resume, VACANCY_TEXT)


async def test_score_too_long_vacancy_rejected():
    client = MockLLMClient([MOCK_MATCH_JSON])
    resume = ResumeFields(category="HR", summary="x", language="ru")
    with pytest.raises(InputValidationError):
        await Screener(client, max_input_chars=10).score(resume, "a" * 11)
    assert len(client.calls) == 0


# --- граница длины и валидация вакансии до вызова LLM в analyze ------------


async def test_parse_input_exactly_at_limit_passes():
    client = MockLLMClient([MOCK_RESUME_JSON])
    r = await Screener(client, max_input_chars=10).parse("a" * 10)  # ровно на границе
    assert r.category == "Python Developer"
    assert len(client.calls) == 1


@pytest.mark.parametrize("bad_vacancy", ["   ", "a" * 50])
async def test_analyze_validates_vacancy_before_any_llm_call(bad_vacancy):
    # битая вакансия отвергается ДО parse -> ни одного обращения к модели
    client = MockLLMClient([MOCK_RESUME_JSON, MOCK_MATCH_JSON])
    with pytest.raises(InputValidationError):
        await Screener(client, max_input_chars=30).analyze("Python dev", bad_vacancy)
    assert len(client.calls) == 0


# --- интеграция с дефолтным моком из фабрики ------------------------------


async def test_analyze_on_default_mock_routes_parse_and_score():
    """Дефолтный мок (как из фабрики в mock-режиме) обслуживает ОБА шага analyze:
    parse-промпт -> ResumeFields, score-промпт -> MatchResult."""
    from config.settings import Settings
    from llm.factory import get_client

    client = get_client(Settings(llm_provider="mock"))
    report = await Screener(client).analyze(RESUME_TEXT, VACANCY_TEXT)
    assert report.resume.category == "Python Developer"
    assert report.match is not None
    assert report.match.verdict == "match"
