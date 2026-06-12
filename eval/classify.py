"""LLM-классификация резюме в закрытый список категорий (для метрик).

Использует промпт `classify.yaml`: список допустимых меток берётся из реального датасета
и инжектится в промпт. Предсказание нормализуется к точной метке из списка (без учёта
регистра), иначе остаётся как есть и честно засчитывается как ошибка.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm.base import LLMClient, Usage
from llm.prompt_loader import Prompt


class ClassifyResult(BaseModel):
    category: str = Field(min_length=1)


def normalize_label(predicted: str, categories: list[str]) -> str:
    """Сопоставить предсказание точной метке из списка (без учёта регистра/пробелов)."""
    pred = predicted.strip()
    lowered = {c.lower(): c for c in categories}
    return lowered.get(pred.lower(), pred)


async def classify_resume(
    client: LLMClient,
    prompt: Prompt,
    categories: list[str],
    resume_text: str,
    *,
    usage: Usage | None = None,
) -> str:
    """Классифицировать одно резюме; вернуть метку из ``categories`` (по возможности)."""
    user_msg = prompt.render(categories="\n".join(f"- {c}" for c in categories), resume=resume_text)
    result = await client.complete_model(prompt.system, user_msg, ClassifyResult, usage=usage)
    return normalize_label(result.category, categories)
