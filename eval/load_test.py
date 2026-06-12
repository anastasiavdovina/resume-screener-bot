"""Базовый нагрузочный тест: N параллельных запросов через asyncio.gather.

Фиксирует error rate и деградацию latency под нагрузкой. На `mock` запросы почти
мгновенны (проверяется сама механика конкуренции); реальная деградация видна с
`--provider anthropic` (упор в rate limits провайдера).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

from config.settings import get_settings
from eval.latency_cost import percentile
from llm.factory import get_client
from service.screener import Screener

logger = logging.getLogger(__name__)


async def load_test(
    screener: Screener, resume: str, vacancy: str | None, concurrency: int
) -> dict[str, float | int]:
    """Запустить `concurrency` запросов одновременно. Вернуть сводку с error rate."""
    if concurrency < 1:
        raise ValueError("concurrency должно быть >= 1")

    async def one() -> tuple[float, Exception | None]:
        start = time.perf_counter()
        try:
            await screener.analyze(resume, vacancy)
            return time.perf_counter() - start, None
        except Exception as e:  # нагрузочный тест не должен падать на ошибке одного запроса
            return time.perf_counter() - start, e

    wall_start = time.perf_counter()
    results = await asyncio.gather(*(one() for _ in range(concurrency)))
    wall = time.perf_counter() - wall_start

    latencies = [latency for latency, _ in results]
    errors = [err for _, err in results if err is not None]
    return {
        "n": concurrency,
        "errors": len(errors),
        "error_rate": len(errors) / concurrency,
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "wall_s": wall,
        "throughput_rps": concurrency / wall if wall > 0 else float("inf"),
    }


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Нагрузочный тест: N параллельных запросов")
    p.add_argument("--resume", default="data/sample_resume.txt")
    p.add_argument("--vacancy", default="data/sample_vacancy.txt")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--provider", choices=["anthropic", "local", "mock"], default=None)
    args = p.parse_args()

    settings = get_settings()
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": args.provider})
    screener = Screener(get_client(settings), max_input_chars=settings.max_input_chars)
    resume = Path(args.resume).read_text(encoding="utf-8")
    vacancy = Path(args.vacancy).read_text(encoding="utf-8")

    r = await load_test(screener, resume, vacancy, args.concurrency)
    print(f"\nНагрузка: {r['n']} параллельных запросов")
    print(f"  ошибок={r['errors']} ({r['error_rate'] * 100:.0f}%)")
    print(f"  latency p50={r['p50']:.2f}с p95={r['p95']:.2f}с")
    print(f"  wall={r['wall_s']:.2f}с throughput≈{r['throughput_rps']:.1f} запр/с")


if __name__ == "__main__":
    asyncio.run(main())
