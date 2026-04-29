.PHONY: help install web web-dev web-prod frontend-dev frontend-build \
        docker-build docker-up docker-up-ollama docker-down test cli

PYTHON ?= python
HOST ?= 0.0.0.0
PORT ?= 28473

help:
	@echo "Boardroom — common targets"
	@echo ""
	@echo "  install            Install Python deps (CLI + web) into the active venv"
	@echo "  cli                Run the Typer CLI (e.g. make cli ARGS='train --help')"
	@echo "  test               Run the test suite"
	@echo ""
	@echo "  frontend-dev       Vite dev server (polling watch, ulimit bump; proxy /api -> $(PORT))"
	@echo "  frontend-build     Build the React app into webapp/frontend/dist"
	@echo "  web                Run uvicorn with the built frontend at /"
	@echo "  web-dev            Run uvicorn with --reload (frontend served via Vite)"
	@echo ""
	@echo "  docker-build       Build the boardroom-web image"
	@echo "  docker-up          Run webapp + tensorboard with GPU"
	@echo "  docker-up-ollama   Same as docker-up plus the Ollama service"
	@echo "  docker-down        Stop everything"

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-web.txt

cli:
	$(PYTHON) -m trading_system $(ARGS)

test:
	$(PYTHON) -m pytest tests/

frontend-dev:
	cd webapp/frontend && (ulimit -n 65536 2>/dev/null || ulimit -n 32768 2>/dev/null || true) && npm run dev

frontend-build:
	cd webapp/frontend && npm install --no-audit --no-fund && npm run build

web: frontend-build
	$(PYTHON) -m uvicorn webapp.backend.main:app --host $(HOST) --port $(PORT)

web-dev:
	BOARDROOM_RELOAD=1 $(PYTHON) -m uvicorn webapp.backend.main:app --reload --host $(HOST) --port $(PORT)

web-prod: frontend-build
	$(PYTHON) -m uvicorn webapp.backend.main:app --host 0.0.0.0 --port $(PORT)

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d --build

docker-up-ollama:
	docker compose -f docker/docker-compose.yml --profile ollama up -d --build

docker-down:
	docker compose -f docker/docker-compose.yml --profile ollama down
