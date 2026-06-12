"""Метрики качества классификации: accuracy, macro-F1, confusion matrix.

Чистые функции над списками меток — не зависят ни от LLM, ни от датасета, поэтому
переиспользуются и для baseline, и для LLM, и легко тестируются.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-бэкенд: рендерим в файл без GUI
import matplotlib.pyplot as plt  # noqa: E402  (после matplotlib.use)
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: list[str], y_pred: list[str], *, labels: list[str] | None = None
) -> dict[str, float]:
    """Accuracy и macro-F1.

    macro-F1 усредняется по ``labels`` (по умолчанию — по реальным классам из
    ``y_true``). Это важно для честности сравнения: предсказания вне списка
    (галлюцинации, ``<error>``) НЕ создают фантомных классов с нулевым F1, которые
    иначе асимметрично занижали бы macro-F1 у LLM. Такие предсказания всё равно
    штрафуются как ошибка (потеря recall у истинного класса) и снижают accuracy.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true и y_pred должны быть одной длины")
    if not y_true:
        raise ValueError("пустые метки")
    macro_labels = labels if labels is not None else sorted(set(y_true))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=macro_labels, average="macro", zero_division=0)
        ),
    }


def save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    path: Path,
    *,
    title: str = "Confusion matrix",
) -> Path:
    """Сохранить confusion matrix как PNG и вернуть путь."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    display.plot(ax=ax, xticks_rotation="vertical", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
