"""Baseline без LLM: TF-IDF + логистическая регрессия.

Нужен для честного сравнения «простая модель vs LLM» по тем же метрикам — это и есть
обоснование выбора подхода через цифры, а не «по умолчанию взяли нейросеть».
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline() -> Pipeline:
    """TF-IDF (uni+bi-граммы) → логистическая регрессия."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_baseline(texts: list[str], labels: list[str]) -> Pipeline:
    """Обучить baseline на размеченных резюме."""
    model = build_baseline()
    model.fit(texts, labels)
    return model


def predict(model: Pipeline, texts: list[str]) -> list[str]:
    """Предсказать категории для списка резюме (как обычные str, не numpy.str_)."""
    return [str(p) for p in model.predict(texts)]
