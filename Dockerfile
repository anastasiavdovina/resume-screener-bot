# Образ только для бота: runtime-зависимости (без eval/dev). Запуск: python -m bot.main
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Зависимости отдельным слоем — кэшируется, пока requirements.txt не менялся
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения (eval/, tests/ в образ не нужны)
COPY config/ config/
COPY llm/ llm/
COPY service/ service/
COPY bot/ bot/

# Непривилегированный пользователь
RUN useradd --create-home appuser
USER appuser

# Токены и LLM_PROVIDER передаются через окружение (docker compose env_file: .env)
CMD ["python", "-m", "bot.main"]
