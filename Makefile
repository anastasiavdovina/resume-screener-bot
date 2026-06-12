PY := .venv/bin/python

.PHONY: help install install-eval test lint run eval clean

help:
	@echo "install      - установить runtime + dev зависимости в .venv"
	@echo "install-eval - доустановить зависимости для блока метрик"
	@echo "test         - запустить pytest"
	@echo "lint         - проверить код через ruff"
	@echo "run          - запустить бота (bot.main)"
	@echo "eval         - baseline-метрики (нужен data/UpdatedResumeDataSet.csv; LLM — см. README)"

install:
	$(PY) -m pip install -r requirements-dev.txt

install-eval:
	$(PY) -m pip install -r requirements-eval.txt

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

run:
	$(PY) -m bot.main

eval:
	$(PY) -m eval.run_metrics --csv data/UpdatedResumeDataSet.csv --limit 0

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
