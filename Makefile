.PHONY: install test test-architecture test-integration test-unit test-golden test-e2e test-journey test-answer-quality audit lint format typecheck dev serve web golden golden-check clean stale-imports help

PY := backend/.venv/bin/python

# backend/.env sets DATABASE_URL to host.docker.internal, which resolves only
# inside a container; a native run needs the host name. Derived here rather than
# left to the caller, because forgetting it fails every request with "could not
# translate host name". An exported DATABASE_URL wins.
DB_URL := $(shell [ -n "$$DATABASE_URL" ] && printf '%s' "$$DATABASE_URL" || sed -n 's/^DATABASE_URL=//p' backend/.env 2>/dev/null | tr -d '"' | sed 's/host\.docker\.internal/localhost/')

# The API refuses to start without a real JWT_SECRET_KEY unless the run is
# explicitly a development one. `make dev` and `make serve` are that: a random
# secret is generated once under STATE_DIR and reused across reloads. Tests use
# a per-process secret. A deployment sets JWT_SECRET_KEY (32+ characters).
APP_ENV ?= development

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install:  ## sync backend/.venv from the lockfile, then install frontend dependencies
	cd backend && uv sync --locked --extra dev --python 3.11
	$(PY) backend/scripts/setup_nltk.py
	cd frontend && npm ci

test: stale-imports test-architecture test-unit test-integration test-golden test-e2e test-journey  ## everything

test-architecture:  ## the one-way dependency and slot wiring; no stack required
	$(PY) -m pytest test/architecture

test-integration:  ## the real routers over HTTP with fake slots; no stack required
	$(PY) -m pytest test/pipeline

test-unit:  ## backend unit tests; no stack required
	cd backend && ../$(PY) -m pytest tests/unit
	cd frontend && npx vitest run

test-golden:  ## complete offline compatibility suite, including real PDF replay
	cd backend && ../$(PY) -m pytest tests/unit/test_golden.py -m "slow or not slow"

test-e2e:  ## end-to-end; needs Postgres, Elasticsearch and the API on :8000
	$(PY) -m pytest test/backend
	cd frontend && node ../test/frontend/ui.e2e.cjs
	cd frontend && node ../test/frontend/features.ui.cjs

test-journey:  ## structural: clean account -> document -> parse -> grounded answers
	$(PY) -m pytest test/journey/test_journey_e2e.py
	cd frontend && node ../test/journey/journey.ui.cjs

test-answer-quality:  ## golden-set pass rates: a measurement of the provider, not a gate
	$(PY) -m pytest test/journey/test_answer_quality_e2e.py

lint:  ## ruff over first-party code (vendor/ is excluded in pyproject)
	cd backend && ../$(PY) -m ruff check src tests ../test

format:  ## ruff autofix
	cd backend && ../$(PY) -m ruff check --fix src tests ../test

audit:  ## audit locked Python and npm dependencies
	cd backend && uv export --locked --extra dev --no-emit-project --no-annotate --no-header | ../$(PY) -m pip_audit --strict -r /dev/stdin --disable-pip --require-hashes --ignore-vuln PYSEC-2026-3740
	cd frontend && npm audit --audit-level=moderate

typecheck:  ## mypy over first-party code
	cd backend && ../$(PY) -m mypy src

dev:  ## the whole app: API on :8000 and frontend on :5181, ctrl-c stops both
	@docker ps --format '{{.Names}}' | grep -q . || echo "note: Postgres and Elasticsearch do not appear to be running -- docker compose up -d es01 postgres"
	@APP_ENV="$(APP_ENV)" DATABASE_URL="$(DB_URL)" $(PY) -m uvicorn visionagent.api.main:app --reload --reload-dir backend/src --host 0.0.0.0 --port 8000 & api=$$!; \
	  (cd frontend && npm run dev) & web=$$!; \
	  trap 'kill $$api $$web 2>/dev/null' EXIT INT TERM; \
	  until curl -sf -o /dev/null http://localhost:5181/ 2>/dev/null; do sleep 1; done; \
	  printf '\n  ────────────────────────────────\n   frontend  http://localhost:5181\n   api       http://localhost:8000/docs\n  ────────────────────────────────\n   ctrl-c stops both\n\n'; \
	  wait

serve:  ## run the API alone; works from any directory, no CWD assumption
	APP_ENV="$(APP_ENV)" DATABASE_URL="$(DB_URL)" $(PY) -m uvicorn visionagent.api.main:app --reload --reload-dir backend/src --host 0.0.0.0 --port 8000

web:  ## run the frontend dev server on :5181
	cd frontend && npm run dev

golden:  ## atomically update the reviewed offline compatibility baseline
	$(PY) backend/scripts/record_golden.py

golden-check:  ## replay the complete offline baseline; no stack or credentials
	$(PY) backend/scripts/record_golden.py --check

clean:
	find . -type d -name __pycache__ -not -path "./node_modules/*" -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache backend/src/*.egg-info

stale-imports:  ## fail if any pre-restructure package path survives anywhere
	@bad=$$(grep -rn "visionagent\.\(intent\|planner\|executer\|rerank\|evaluator\|answer\|parsers\|vectorstore\|tools\|utils\.auth_utils\|utils\.database\|pipeline\.sse\|service\.\(llm\|embedding\|websearch\)\)\b" \
	          --include="*.py" --include="*.toml" --include="*.md" --include="*.cfg" \
	          backend test docs 2>/dev/null | grep -v __pycache__ | grep -v "visionagent\.service\.\(intent\|planner\|executer\|rerank\|evaluator\|answer\|parsers\|vectorstore\)"); \
	  if [ -n "$$bad" ]; then echo "$$bad"; echo; echo "stale import paths above"; exit 1; fi; \
	  echo "no stale import paths"
