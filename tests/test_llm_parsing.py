"""Тесты LLM-обвязки: извлечение JSON, валидация по схеме, ретрай, учёт токенов.

Сеть и ключи не нужны — используется MockLLMClient.
"""

import pytest

from llm.base import ParseError, Usage, extract_json, parse_model
from llm.mock_client import MOCK_MATCH_JSON, MOCK_RESUME_JSON, MockLLMClient
from service.schemas import MatchResult, ResumeFields

# --- extract_json ---------------------------------------------------------


def test_extract_json_strips_json_fence():
    raw = '```json\n{"a": 1}\n```'
    assert extract_json(raw) == '{"a": 1}'


def test_extract_json_strips_bare_fence():
    raw = '```\n{"a": 1}\n```'
    assert extract_json(raw) == '{"a": 1}'


def test_extract_json_pulls_object_out_of_prose():
    raw = 'Конечно! Вот результат: {"a": 1, "b": 2} — надеюсь, помог.'
    assert extract_json(raw) == '{"a": 1, "b": 2}'


def test_extract_json_passthrough_plain():
    assert extract_json('  {"a": 1}  ') == '{"a": 1}'


# --- parse_model ----------------------------------------------------------


def test_parse_model_valid():
    r = parse_model(MOCK_RESUME_JSON, ResumeFields)
    assert isinstance(r, ResumeFields)
    assert r.category == "Python Developer"


def test_parse_model_valid_with_fence():
    r = parse_model(f"```json\n{MOCK_RESUME_JSON}\n```", ResumeFields)
    assert r.grade == "middle"


def test_parse_model_invalid_json_raises_parse_error():
    with pytest.raises(ParseError):
        parse_model("это не json", ResumeFields)


def test_parse_model_schema_mismatch_raises_parse_error():
    # синтаксически валидный JSON, но не подходит под схему (score вне диапазона)
    with pytest.raises(ParseError):
        parse_model('{"score": 500, "verdict": "match", "explanation": "x"}', MatchResult)


# --- complete_model: успех, ретрай, исчерпание ----------------------------


async def test_complete_model_success_first_try():
    client = MockLLMClient([MOCK_RESUME_JSON])
    r = await client.complete_model("sys", "user", ResumeFields)
    assert r.category == "Python Developer"
    assert len(client.calls) == 1


async def test_complete_model_retries_then_succeeds():
    client = MockLLMClient(["совершенно не json", MOCK_RESUME_JSON])
    r = await client.complete_model("sys", "user", ResumeFields, max_retries=1)
    assert isinstance(r, ResumeFields)
    assert len(client.calls) == 2
    # на втором вызове в промпт добавлена корректирующая инструкция
    _, retry_user_msg = client.calls[1]
    assert "СТРОГО валидный JSON" in retry_user_msg


async def test_complete_model_exhausts_retries_and_raises():
    client = MockLLMClient(["мусор", "снова мусор"])
    with pytest.raises(ParseError):
        await client.complete_model("sys", "user", ResumeFields, max_retries=1)
    assert len(client.calls) == 2  # 1 попытка + 1 ретрай


async def test_complete_model_zero_retries():
    client = MockLLMClient(["мусор", MOCK_RESUME_JSON])
    with pytest.raises(ParseError):
        await client.complete_model("sys", "user", ResumeFields, max_retries=0)
    assert len(client.calls) == 1


# --- учёт токенов ---------------------------------------------------------


async def test_usage_accumulates_across_retry():
    client = MockLLMClient(["мусор", MOCK_RESUME_JSON], input_tokens=10, output_tokens=20)
    usage = Usage()
    await client.complete_model("sys", "user", ResumeFields, usage=usage, max_retries=1)
    assert usage.calls == 2
    assert usage.input_tokens == 20  # 10 * 2 вызова
    assert usage.output_tokens == 40


# --- mock по умолчанию ----------------------------------------------------


async def test_mock_default_response_is_valid_resume():
    client = MockLLMClient()  # без responses, не score-промпт -> разбор резюме
    r = await client.complete_model("sys", "user", ResumeFields)
    assert r.language == "ru"


async def test_mock_default_routes_scoring_prompt_to_match():
    # системный промпт со словом "verdict" -> дефолтный мок отдаёт MatchResult-JSON
    client = MockLLMClient()
    m = await client.complete_model("... verdict ...", "u", MatchResult)
    assert m.verdict == "match"


def test_mock_routing_matches_real_yaml_prompts():
    # пришпиливаем эвристику роутинга к реальным YAML-промптам: если их переформулируют
    # так, что роутинг ломается, этот тест упадёт громко (а не молча в рантайме).
    from llm.prompt_loader import default_prompt_set

    ps = default_prompt_set()
    assert MockLLMClient._is_scoring_prompt(ps.score_match.system) is True
    assert MockLLMClient._is_scoring_prompt(ps.parse_resume.system) is False


# --- extract_json: устойчивость к болтливым моделям ------------------------


def test_extract_json_ignores_trailing_prose_with_braces():
    raw = '{"a": 1, "b": 2}\n\nНадеюсь, формат {ключ: значение} понятен!'
    assert extract_json(raw) == '{"a": 1, "b": 2}'


def test_extract_json_respects_braces_inside_strings():
    raw = '{"a": "x}y", "b": 2}'
    assert extract_json(raw) == '{"a": "x}y", "b": 2}'


def test_extract_json_returns_first_object_when_multiple():
    assert extract_json('{"a": 1} {"b": 2}') == '{"a": 1}'


def test_extract_json_handles_nested_object():
    raw = 'prefix {"a": {"b": 1}} suffix'
    assert extract_json(raw) == '{"a": {"b": 1}}'


# --- parse_model: положительная проверка второй mock-фикстуры --------------


def test_parse_model_match_valid():
    m = parse_model(MOCK_MATCH_JSON, MatchResult)
    assert m.verdict == "match"
    assert m.score == 78


# --- complete_model: ретрай на несоответствии схеме (не только битый JSON) --


async def test_complete_model_retries_on_schema_mismatch():
    bad = '{"score": 500, "verdict": "match", "explanation": "x"}'  # score вне диапазона
    client = MockLLMClient([bad, MOCK_MATCH_JSON])
    r = await client.complete_model("sys", "user", MatchResult, max_retries=1)
    assert r.score == 78
    assert len(client.calls) == 2


async def test_complete_model_negative_max_retries_raises_value_error():
    client = MockLLMClient([MOCK_RESUME_JSON])
    with pytest.raises(ValueError):
        await client.complete_model("sys", "user", ResumeFields, max_retries=-1)


# --- учёт токенов и latency на успешном пути -------------------------------


async def test_usage_success_path_counts_one_call_and_latency():
    client = MockLLMClient([MOCK_RESUME_JSON], input_tokens=7, output_tokens=11, latency_s=0.5)
    usage = Usage()
    await client.complete_model("sys", "user", ResumeFields, usage=usage)
    assert usage.calls == 1
    assert usage.input_tokens == 7
    assert usage.output_tokens == 11
    assert usage.latency_s == 0.5
