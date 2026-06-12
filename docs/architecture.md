# Архитектура — Resume Screener Bot

Документ для Блока 3 (архитектура): слои и их ответственность, поток данных,
заменяемость модели, узкое место, масштабирование и риски.

## Слои и ответственность

Проект разделён на слои с чёткой ответственностью — это исключает «монолитный скрипт»
и обеспечивает заменяемость модели. Зависимости направлены вниз: `bot → service → llm`;
`config` — общий нижний уровень; `eval` использует те же `service`/`llm` офлайн.

| Слой | Пакет | Ответственность | Технологии |
|---|---|---|---|
| Интерфейс | `bot/` | приём сообщений, FSM-диалог, валидация, форматирование ответа (RU/EN) | aiogram |
| Бизнес-логика | `service/` | оркестрация parse→score, pydantic-схемы, валидация ввода, доменные ошибки | Python, pydantic |
| Адаптер модели | `llm/` | абстракция `LLMClient`, провайдеры, парсинг/валидация JSON + ретрай, промпты | anthropic / openai (Ollama) SDK |
| Конфигурация | `config/` | чтение настроек из окружения | pydantic-settings |
| Оценка | `eval/` | baseline, метрики, latency/стоимость, нагрузочный тест | scikit-learn, matplotlib |

Доменные ошибки разнесены по слоям: `LLMError`/`ParseError` определены в `llm/` (нижний
слой владеет своими ошибками), `ScreenerError`/`InputValidationError` — в `service/`.

## Контейнерная диаграмма (C4-Container, уровень контейнеров)

```mermaid
flowchart LR
    User([HR / кандидат]):::ext
    TG([Telegram]):::ext

    User <-->|резюме + вакансия| TG

    subgraph App[Resume Screener Bot]
        direction TB
        Bot["bot/<br/>aiogram: команды, FSM,<br/>форматирование"]
        Service["service/<br/>Screener: parse → score,<br/>валидация, схемы"]
        Adapter["llm/<br/>LLMClient + промпты (YAML)<br/>+ парсинг JSON + ретрай"]
        Bot --> Service --> Adapter
    end

    TG <--> Bot

    Adapter -->|complete| Provider{LLM_PROVIDER}
    Provider -->|anthropic| Anthropic[["Anthropic API"]]:::ext
    Provider -->|local| Ollama[["Ollama (локально)"]]:::ext
    Provider -->|mock| Mock[["MockLLMClient<br/>(dev/тесты)"]]

    Eval["eval/<br/>baseline, метрики,<br/>latency, нагрузка"]:::offline -. офлайн .-> Adapter

    classDef ext fill:#eef,stroke:#88a;
    classDef offline fill:#efe,stroke:#8a8;
```

## Поток данных (сценарий /match)

```mermaid
sequenceDiagram
    actor U as Пользователь
    participant B as bot (handlers/FSM)
    participant S as service (Screener)
    participant L as llm (адаптер)
    U->>B: /match
    B->>U: пришлите резюме
    U->>B: текст резюме
    B->>U: пришлите вакансию
    U->>B: текст вакансии
    B->>S: analyze(resume, vacancy)
    S->>S: валидация ввода (до LLM)
    S->>L: parse_resume → JSON
    L-->>S: ResumeFields (валидировано pydantic, ретрай при сбое)
    S->>L: score_match (резюме + вакансия) → JSON
    L-->>S: MatchResult
    S-->>B: Report
    B->>U: форматированный результат (язык резюме)
```

Ошибки на любом шаге (`InputValidationError`, `ParseError`, `LLMError`) ловятся ботом и
превращаются в понятное сообщение — бот не падает.

## Заменяемость модели

Провайдер скрыт за абстрактным `LLMClient` (метод `complete`). Фабрика `llm/factory.py`
выбирает реализацию по `LLM_PROVIDER` (`anthropic` | `local` | `mock`), модель — по
`LLM_MODEL`. Это даёт:

- разработку и тесты **без ключей** (`mock`);
- страховку от исчерпания кредитов Console (`local` → Ollama);
- смену модели/провайдера одной переменной окружения, без правок логики.

Промпты вынесены в `llm/prompts/*.yaml` (версионируются) — формулировки меняются без кода.

## Узкое место и масштабирование

- **Узкое место:** сетевые вызовы LLM (latency ~секунды) и rate limits провайдера, а не CPU.
  Интерфейс асинхронный (aiogram + async `LLMClient`), поэтому один процесс обрабатывает
  много диалогов конкурентно, не блокируя event loop.
- **Как масштабировать:**
  - горизонтально — несколько реплик бота (stateless; FSM-хранилище вынести в Redis);
  - очередь задач (например, при пакетном ранжировании резюме) с воркерами;
  - кэширование/`prompt caching` системного промпта (уже включено в anthropic-клиенте) —
    снижает стоимость и latency повторов;
  - батч-режим Anthropic для офлайн-прогона метрик на большом датасете.

## Риски

- **Исчерпание кредитов Console** → митигация: `local`-провайдер (Ollama) в том же интерфейсе.
- **Галлюцинации LLM** (выдуманные навыки) → строгий промпт «только на основе текста»,
  валидация JSON по pydantic-схемам, опциональные поля → `null` вместо выдумки, один ретрай.
- **Предвзятость модели** при оценке кандидатов → этическое ограничение: бот помогает и
  структурирует первичный этап, финальное решение принимает человек.
