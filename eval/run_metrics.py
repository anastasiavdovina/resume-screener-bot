"""Прогон метрик: baseline (TF-IDF+LogReg) и, опционально, LLM-классификация.

Запуск:
    # только baseline на полном тесте (без ключа):
    python -m eval.run_metrics --csv data/UpdatedResumeDataSet.csv --limit 0
    # baseline + LLM (нужен ключ/провайдер):
    python -m eval.run_metrics --csv data/UpdatedResumeDataSet.csv --provider anthropic --limit 80

LLM гоняется на ПОДВЫБОРКЕ теста (--limit) ради экономии кредитов. Чтобы сравнение было
apples-to-apples, baseline дополнительно считается на ТОЙ ЖЕ подвыборке (рядом с полным
тестом). Confusion matrix сохраняются в docs/.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from config.settings import get_settings
from eval.baseline import predict, train_baseline
from eval.classify import classify_resume
from eval.metrics import compute_metrics, save_confusion_matrix
from eval.prepare_data import Split, load_dataset, make_split
from llm.base import LLMError, Usage
from llm.factory import get_client
from llm.prompt_loader import Prompt, load_classify_prompt

logger = logging.getLogger(__name__)
DOCS_DIR = Path("docs")


def evaluate(gold: list[str], preds: list[str], out_dir: Path, cm_name: str, title: str) -> dict:
    """Посчитать метрики (macro-F1 по реальным классам gold) и сохранить confusion matrix."""
    save_confusion_matrix(
        gold, preds, sorted(set(gold) | set(preds)), out_dir / cm_name, title=title
    )
    return compute_metrics(gold, preds)


def run_baseline(split: Split, out_dir: Path):
    """Обучить baseline и оценить на ПОЛНОМ тесте. Возвращает (модель, метрики)."""
    model = train_baseline(split.train_texts, split.train_labels)
    preds = predict(model, split.test_texts)
    metrics = evaluate(
        split.test_labels, preds, out_dir, "cm_baseline.png", "Baseline (полный тест)"
    )
    return model, metrics


async def classify_all(
    client, prompt: Prompt, categories: list[str], texts: list[str]
) -> tuple[list[str], Usage]:
    """LLM-классификация списка резюме. Возвращает (предсказания, расход токенов)."""
    usage = Usage()
    preds: list[str] = []
    for i, text in enumerate(texts, 1):
        try:
            preds.append(await classify_resume(client, prompt, categories, text, usage=usage))
        except LLMError as e:  # один сбойный пример не валит весь прогон
            logger.warning("classify failed on #%d: %s", i, e)
            preds.append("<error>")
    return preds, usage


def _fmt(name: str, metrics: dict, n: int) -> str:
    return f"{name} (n={n}): accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Метрики baseline vs LLM на датасете резюме")
    p.add_argument("--csv", required=True, help="путь к CSV датасета")
    p.add_argument("--resume-col", default=None)
    p.add_argument("--category-col", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=50, help="резюме для LLM (0 = только baseline)")
    p.add_argument("--provider", choices=["anthropic", "local", "mock"], default=None)
    return p.parse_args()


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    texts, labels = load_dataset(
        args.csv, resume_col=args.resume_col, category_col=args.category_col
    )
    split = make_split(texts, labels, test_size=args.test_size, seed=args.seed)
    DOCS_DIR.mkdir(exist_ok=True)
    logger.info(
        "train=%d test=%d classes=%d",
        len(split.train_texts), len(split.test_texts), len(split.labels),
    )

    model, base_full = run_baseline(split, DOCS_DIR)
    print("\n" + _fmt("Baseline, полный тест", base_full, len(split.test_texts)))

    if args.limit == 0:
        print("\nLLM пропущен (--limit 0). Confusion matrix → docs/cm_baseline.png")
        return

    settings = get_settings()
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": args.provider})
    if settings.llm_provider == "mock":
        logger.warning(
            "LLM_PROVIDER=mock: метрики LLM не осмысленны (мок не классифицирует); "
            "укажите --provider anthropic"
        )

    n = min(args.limit, len(split.test_texts))
    sub_texts, sub_gold = split.test_texts[:n], split.test_labels[:n]
    logger.info("LLM-подвыборка: n=%d, классов в ней=%d", n, len(set(sub_gold)))

    # baseline на ТОЙ ЖЕ подвыборке -> честное сравнение на одном множестве
    base_sub = evaluate(
        sub_gold, predict(model, sub_texts), DOCS_DIR, "cm_baseline_sub.png", "Baseline, подвыборка"
    )
    client = get_client(settings)
    llm_preds, usage = await classify_all(client, load_classify_prompt(), split.labels, sub_texts)
    llm = evaluate(sub_gold, llm_preds, DOCS_DIR, "cm_llm.png", "LLM, подвыборка")

    print("\nСравнение на одной подвыборке (apples-to-apples):")
    print("  " + _fmt("Baseline", base_sub, n))
    print("  " + _fmt("LLM     ", llm, n))
    print(
        f"  LLM токены: in={usage.input_tokens} "
        f"out={usage.output_tokens} вызовов={usage.calls}"
    )
    print("\nConfusion matrix → docs/cm_baseline.png, docs/cm_baseline_sub.png, docs/cm_llm.png")


if __name__ == "__main__":
    asyncio.run(main())
