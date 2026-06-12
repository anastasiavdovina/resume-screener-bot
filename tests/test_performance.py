"""Тесты замеров производительности на mock-клиентах (без сети)."""

import pytest

from eval.latency_cost import (
    Price,
    cost_usd,
    measure,
    percentile,
    summarize_latencies,
)
from eval.load_test import load_test
from llm.base import LLMClient, LLMError, LLMResponse
from llm.mock_client import MockLLMClient
from service.screener import Screener

RESUME = "Python backend, FastAPI, Docker, 4 года"
VACANCY = "Ищем middle Python: FastAPI, Docker, PostgreSQL"


class _RaisingClient(LLMClient):
    async def complete(
        self, system_prompt: str, user_msg: str, *, response_schema: dict | None = None
    ) -> LLMResponse:
        raise LLMError("api down")


# --- percentile -----------------------------------------------------------


def test_percentile_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)


def test_percentile_single_value():
    assert percentile([7.0], 95) == 7.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


# --- cost -----------------------------------------------------------------


def test_cost_usd_known_model():
    pricing = {"m": Price(2.0, 10.0)}  # $2/$10 за 1M
    assert cost_usd(1_000_000, 1_000_000, "m", pricing) == pytest.approx(12.0)
    assert cost_usd(500_000, 0, "m", pricing) == pytest.approx(1.0)


def test_cost_usd_unknown_model_uses_fallback():
    # неизвестная модель -> ненулевой fallback-прайс, не падаем
    assert cost_usd(1_000_000, 0, "no-such-model") > 0


# --- summarize ------------------------------------------------------------


def test_summarize_latencies_keys():
    s = summarize_latencies([0.1, 0.2, 0.3])
    assert set(s) >= {"n", "mean", "p50", "p95", "max"}
    assert s["n"] == 3
    assert s["max"] == 0.3


# --- measure (на mock) ----------------------------------------------------


async def test_measure_runs_and_accumulates_usage():
    from llm.base import Usage

    client = MockLLMClient(input_tokens=10, output_tokens=20)
    usage = Usage()
    lat = await measure(Screener(client), RESUME, VACANCY, runs=3, usage=usage)
    assert len(lat) == 3
    assert all(x >= 0 for x in lat)
    assert usage.calls == 6  # 2 вызова (parse+score) * 3 запуска
    assert len(client.calls) == 6


# --- load_test (на mock) --------------------------------------------------


async def test_load_test_no_errors_on_mock():
    client = MockLLMClient()
    r = await load_test(Screener(client), RESUME, VACANCY, concurrency=4)
    assert r["n"] == 4
    assert r["errors"] == 0
    assert r["error_rate"] == 0.0
    assert len(client.calls) == 8  # 2 * 4
    assert r["p95"] >= 0


async def test_load_test_counts_errors():
    r = await load_test(Screener(_RaisingClient()), RESUME, VACANCY, concurrency=3)
    assert r["errors"] == 3
    assert r["error_rate"] == 1.0


async def test_load_test_rejects_nonpositive_concurrency():
    with pytest.raises(ValueError):
        await load_test(Screener(MockLLMClient()), RESUME, VACANCY, concurrency=0)
