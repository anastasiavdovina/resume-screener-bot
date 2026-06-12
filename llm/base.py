"""Абстракция над LLM-провайдером + утилиты парсинга/валидации JSON-ответа.

Слой `llm` ничего не знает о Telegram и бизнес-логике: он умеет только отправить
системный промпт + сообщение пользователя и вернуть текст (плюс токены/latency для
метрик). Конкретные провайдеры (anthropic / ollama / mock) реализуют один метод
``complete``; общая обвязка (извлечение JSON, валидация по pydantic-схеме, один ретрай
при невалидном ответе) живёт здесь и переиспользуется всеми провайдерами.

Интерфейс асинхронный, потому что бот на aiogram асинхронный: блокирующий вызов LLM
внутри хендлера завесил бы весь event loop. Это же позволяет дёшево сделать
нагрузочный тест через ``asyncio.gather`` (этап 7).
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# ```json ... ``` или просто ``` ... ``` (нежадно, с переносами строк)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# --- Доменные исключения LLM-слоя -----------------------------------------


class LLMError(Exception):
    """Базовая ошибка обращения к модели (сеть, таймаут, API)."""


class ParseError(LLMError):
    """Ответ модели не удалось извлечь / распарсить / провалидировать по схеме."""


# --- Контейнеры ответа и учёта -------------------------------------------


@dataclass
class LLMResponse:
    """Сырой ответ модели + метаданные для метрик стоимости/latency."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class Usage:
    """Аккумулятор расхода по запросу (с учётом ретраев) — для оценки стоимости."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    def add(self, resp: LLMResponse) -> None:
        self.calls += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.latency_s += resp.latency_s


# --- Извлечение и валидация JSON ------------------------------------------


def _first_balanced_object(text: str) -> str | None:
    """Первый сбалансированный объект ``{...}`` верхнего уровня.

    Считает глубину скобок и пропускает их внутри строковых литералов (учитывает
    экранирование). Поэтому устойчив и к скобкам в значениях, и к болтливой прозе,
    которую модель может дописать после JSON. Возвращает None, если объект не найден
    или не закрыт.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> str:
    """Достаёт JSON-объект из ответа модели.

    Снимает markdown-обёртку ```json ... ``` (если есть), затем выделяет первый
    сбалансированный объект ``{...}``. Это устойчиво к болтливым моделям, которые
    дописывают пояснение после JSON, и к фигурным скобкам внутри строковых значений.
    Сам не парсит. Если объект не найден — возвращает исходный текст (дальше упадёт
    в ParseError и сработает ретрай).
    """
    text = text.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    obj = _first_balanced_object(text)
    return obj if obj is not None else text


def parse_model[T: BaseModel](text: str, model: type[T]) -> T:
    """Извлекает JSON из ответа и валидирует его по pydantic-схеме.

    Бросает :class:`ParseError` и на синтаксически невалидном JSON, и на
    несоответствии схеме — вызывающий код одинаково реагирует ретраем.
    """
    payload = extract_json(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ParseError(f"ответ модели — невалидный JSON: {e}") from e
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ParseError(f"JSON не соответствует схеме {model.__name__}: {e}") from e


# --- Абстрактный клиент ----------------------------------------------------


class LLMClient(ABC):
    """Интерфейс провайдера модели. Подклассы реализуют только ``complete``."""

    @abstractmethod
    async def complete(
        self, system_prompt: str, user_msg: str, *, response_schema: dict | None = None
    ) -> LLMResponse:
        """Отправить промпт и вернуть сырой ответ модели.

        ``response_schema`` (JSON Schema ожидаемого ответа) — подсказка для провайдеров
        со structured output (Ollama использует её, чтобы гарантировать соответствие
        схеме). Провайдеры без поддержки могут игнорировать.
        """
        raise NotImplementedError

    async def complete_model[T: BaseModel](
        self,
        system_prompt: str,
        user_msg: str,
        model: type[T],
        *,
        usage: Usage | None = None,
        max_retries: int = 1,
    ) -> T:
        """Вызвать модель и вернуть ответ, провалидированный по ``model``.

        При невалидном ответе делает до ``max_retries`` повторов, добавляя в промпт
        текст ошибки и требование вернуть строго валидный JSON. Если все попытки
        провалились — пробрасывает последнюю :class:`ParseError`.
        """
        if max_retries < 0:
            raise ValueError("max_retries не может быть отрицательным")
        msg = user_msg
        last_err: ParseError | None = None
        schema = model.model_json_schema()
        for attempt in range(max_retries + 1):
            resp = await self.complete(system_prompt, msg, response_schema=schema)
            if usage is not None:
                usage.add(resp)
            try:
                return parse_model(resp.text, model)
            except ParseError as e:
                last_err = e
                logger.warning(
                    "LLM-ответ не прошёл валидацию (попытка %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                msg = (
                    f"{user_msg}\n\n"
                    f"Твой предыдущий ответ не прошёл валидацию: {e}\n"
                    f"Верни СТРОГО валидный JSON по требуемой схеме, "
                    f"без markdown-обёртки и без каких-либо пояснений."
                )
        # Сюда попадаем, только если все попытки кинули ParseError -> last_err задан.
        # Явная проверка (а не assert) — чтобы корректно работать и под `python -O`.
        if last_err is None:  # недостижимо при max_retries >= 0
            raise LLMError("complete_model: не выполнено ни одной попытки")
        raise last_err
