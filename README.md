# Resume Screener Bot

Телеграм-бот для первичного скрининга резюме. Принимает текст резюме и (если нужно)
текст вакансии, возвращает структурированный разбор резюме и оценку соответствия
вакансии с объяснением. ML-часть — это вызовы LLM, поведение задаётся системными
промптами.


## Что делает

- `/analyze` — разбирает резюме: категория, навыки, опыт, грейд, образование, краткое описание.
- `/match` — сравнивает резюме с вакансией: скор 0–100, вердикт, сильные стороны, пробелы, объяснение.
- Отвечает на языке резюме (русский или английский).

## Запуск

Нужен Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

По умолчанию провайдер модели — `mock` (заглушка с фиксированными ответами), поэтому
проект запускается и тесты проходят без ключей и без интернета:

```bash
pytest
```

Чтобы поднять самого бота, нужен токен от [@BotFather](https://t.me/BotFather) и
выбранная модель. Копируем конфиг и заполняем его:

```bash
cp .env.example .env
```

Дальше один из трёх вариантов модели.

**Локальная модель через Ollama (бесплатно).** Ставим [Ollama](https://ollama.com) и тянем модель:

```bash
ollama pull llama3.1
```

В `.env`: `LLM_PROVIDER=local`, `LOCAL_MODEL=llama3.1`. Запуск:

```bash
python -m bot.main
```

**Anthropic API.** В `.env`: `LLM_PROVIDER=anthropic` и `ANTHROPIC_API_KEY=sk-ant-...`.
Запуск тот же (`python -m bot.main`).

**Docker.** Заполненный `.env` + одна команда:

```bash
docker compose up --build
```

## Команды бота

`/start`, `/help`, `/analyze`, `/match`, `/cancel`.

## Переменные окружения

| Переменная | Что задаёт | По умолчанию |
|---|---|---|
| `LLM_PROVIDER` | `mock` / `anthropic` / `local` | `mock` |
| `TELEGRAM_BOT_TOKEN` | токен бота от @BotFather | — |
| `ANTHROPIC_API_KEY` | ключ Anthropic (для `anthropic`) | — |
| `LLM_MODEL` | модель Anthropic | `claude-haiku-4-5` |
| `LOCAL_MODEL` | модель Ollama (для `local`) | `llama3.1` |
| `LOCAL_BASE_URL` | адрес Ollama | `http://localhost:11434/v1` |
| `REQUEST_TIMEOUT` | таймаут запроса к модели, сек | `30` |
| `MAX_INPUT_CHARS` | лимит длины ввода | `15000` |

## Метрики

Качество классификации резюме по категории сравнивается у простого baseline
(TF-IDF + логистическая регрессия) и у LLM — по accuracy и macro-F1, плюс confusion
matrix. Датасет — открытый Kaggle Resume Dataset (`UpdatedResumeDataSet.csv`, 25 категорий),
кладётся в `data/`.

```bash
pip install -r requirements-eval.txt
# только baseline, без ключа:
python -m eval.run_metrics --csv data/UpdatedResumeDataSet.csv --limit 0
# baseline + LLM на подвыборке:
python -m eval.run_metrics --csv data/UpdatedResumeDataSet.csv --provider local --limit 50
```

Результаты прогона и интерпретация — в [docs/results.md](docs/results.md).

## Тесты

```bash
pytest
```

## Структура проекта

- `bot/` — телеграм-бот на aiogram: команды, FSM-диалог, форматирование ответа
- `service/` — бизнес-логика: разбор → оценка, схемы данных, валидация, ошибки
- `llm/` — клиенты моделей (mock / anthropic / ollama), фабрика, промпты в YAML
- `config/` — настройки из переменных окружения
- `eval/` — baseline, метрики, замеры latency и нагрузки
- `tests/` — unit-тесты
