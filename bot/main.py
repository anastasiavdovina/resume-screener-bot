"""Точка входа бота: сборка зависимостей и запуск polling.

Запуск: ``python -m bot.main`` (нужен TELEGRAM_BOT_TOKEN в .env). LLM-провайдер
выбирается по LLM_PROVIDER: на дефолтном `mock` бот работает без ключей Anthropic.
"""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher

from bot.handlers import router
from config.settings import get_settings
from llm.factory import get_client
from service.screener import Screener


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан в .env — нельзя запустить бота")

    screener = Screener(get_client(settings), max_input_chars=settings.max_input_chars)

    bot = Bot(settings.telegram_bot_token)
    dp = Dispatcher()
    dp["screener"] = screener  # инжектится в хендлеры по имени параметра
    dp.include_router(router)

    logging.getLogger(__name__).info("Бот запущен, провайдер=%s", settings.llm_provider)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
