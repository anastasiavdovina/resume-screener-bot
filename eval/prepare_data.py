"""Подготовка датасета: загрузка CSV резюме и стратифицированное train/test разбиение.

Поддерживает оба распространённых Kaggle-датасета резюме: имена колонок резюме/категории
определяются автоматически (или задаются явно).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

# Возможные имена колонок в разных Kaggle-датасетах резюме.
_RESUME_COLS = ("Resume", "Resume_str", "resume", "resume_str", "text")
_CATEGORY_COLS = ("Category", "category", "label")


@dataclass(frozen=True)
class Split:
    """Результат разбиения: тексты и метки train/test."""

    train_texts: list[str]
    train_labels: list[str]
    test_texts: list[str]
    test_labels: list[str]

    @property
    def labels(self) -> list[str]:
        """Отсортированный список уникальных меток (для confusion matrix / classify)."""
        return sorted(set(self.train_labels) | set(self.test_labels))


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"колонка {explicit!r} не найдена; есть: {list(df.columns)}")
        return explicit
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"не нашёл колонку из {candidates}; есть: {list(df.columns)}")


def load_dataset(
    csv_path: str,
    *,
    resume_col: str | None = None,
    category_col: str | None = None,
) -> tuple[list[str], list[str]]:
    """Прочитать CSV, вернуть (тексты резюме, метки категорий). Пустые строки отброшены."""
    df = pd.read_csv(csv_path)
    r_col = _pick_column(df, _RESUME_COLS, resume_col)
    c_col = _pick_column(df, _CATEGORY_COLS, category_col)
    df = df[[r_col, c_col]].dropna()
    texts = [str(t).strip() for t in df[r_col]]
    labels = [str(c).strip() for c in df[c_col]]
    keep = [(t, c) for t, c in zip(texts, labels, strict=True) if t and c]
    if not keep:
        raise ValueError("после очистки не осталось примеров")
    texts, labels = map(list, zip(*keep, strict=True))
    return texts, labels


def make_split(
    texts: list[str],
    labels: list[str],
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> Split:
    """Стратифицированное разбиение train/test (доли классов сохраняются)."""
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    return Split(list(x_train), list(y_train), list(x_test), list(y_test))
