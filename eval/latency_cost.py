"""Замер latency и стоимости запроса к боту, плюс поведение на длинных входах.

Latency меряется как wall-clock одного полного `analyze` (parse + score). Стоимость
считается из usage-токенов по прайсу модели. На `mock`-провайдере latency ~0 и стоимость
условна — реальные цифры получаются с `--provider anthropic`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from config.settings import get_settings
from llm.base import Usage
from llm.factory import get_client
from service.screener import Screener

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    """Цена в долларах за 1 млн токенов."""

    input_per_mtok: float
    output_per_mtok: float


# Иллюстративные цены ($/1M токенов). СВЕРЬТЕ с актуальным прайсом Anthropic Console.
DEFAULT_PRICING: dict[str, Price] = {
    "claude-haiku-4-5": Price(1.0, 5.0),
    "claude-sonnet-4-6": Price(3.0, 15.0),
}
_FALLBACK_PRICE = Price(1.0, 5.0)


def cost_usd(
    input_tokens: int,
    output_tokens: int,
    model: str,
    pricing: dict[str, Price] | None = None,
) -> float:
    """Стоимость в долларах по числу токенов и прайсу модели."""
    table = pricing if pricing is not None else DEFAULT_PRICING
    price = table.get(model, _FALLBACK_PRICE)
    return (
        input_tokens / 1_000_000 * price.input_per_mtok
        + output_tokens / 1_000_000 * price.output_per_mtok
    )


def percentile(values: list[float], p: float) -> float:
    """p-й перцентиль с линейной интерполяцией (как numpy.percentile)."""
    if not values:
        raise ValueError("пустой список значений")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100) * (len(s) - 1)
    low = int(rank)
    high = min(low + 1, len(s) - 1)
    return s[low] + (s[high] - s[low]) * (rank - low)


def summarize_latencies(latencies: list[float]) -> dict[str, float | int]:
    """Сводка по latency: mean / p50 / p95 / max (секунды)."""
    return {
        "n": len(latencies),
        "mean": mean(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "max": max(latencies),
    }


async def measure(
    screener: Screener,
    resume: str,
    vacancy: str | None,
    runs: int,
    *,
    usage: Usage | None = None,
) -> list[float]:
    """Прогнать `analyze` `runs` раз, вернуть список wall-clock latency (сек)."""
    latencies: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        await screener.analyze(resume, vacancy, usage=usage)
        latencies.append(time.perf_counter() - start)
    return latencies


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Latency и стоимость одного запроса")
    p.add_argument("--resume", default="data/sample_resume.txt")
    p.add_argument("--vacancy", default="data/sample_vacancy.txt")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--provider", choices=["anthropic", "local", "mock"], default=None)
    args = p.parse_args()

    settings = get_settings()
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": args.provider})
    screener = Screener(get_client(settings), max_input_chars=settings.max_input_chars)
    resume, vacancy = _read(args.resume), _read(args.vacancy)

    usage = Usage()
    latencies = await measure(screener, resume, vacancy, args.runs, usage=usage)
    summary = summarize_latencies(latencies)
    print(f"\nLatency по {args.runs} запросам (сек):")
    print(f"  mean={summary['mean']:.2f}  p50={summary['p50']:.2f}  p95={summary['p95']:.2f}")

    per_req_in = usage.input_tokens / args.runs
    per_req_out = usage.output_tokens / args.runs
    total_cost = cost_usd(usage.input_tokens, usage.output_tokens, settings.llm_model)
    print(f"Токены на запрос: in≈{per_req_in:.0f} out≈{per_req_out:.0f}")
    print(f"Стоимость: всего=${total_cost:.4f}  на запрос≈${total_cost / args.runs:.4f}")
    print(f"  (прайс модели {settings.llm_model}; сверьте с Anthropic Console)")

    # Поведение на длинном входе: дублируем резюме почти до лимита.
    factor = max(1, settings.max_input_chars // max(1, len(resume)) - 1)
    long_resume = ((resume + "\n") * factor)[: settings.max_input_chars]
    long_lat = await measure(screener, long_resume, vacancy, 1)
    print(f"\nДлинный вход ({len(long_resume)} симв.): latency={long_lat[0]:.2f} сек")


if __name__ == "__main__":
    asyncio.run(main())
