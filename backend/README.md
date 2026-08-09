# vrf-chat backend

FastAPI + Celery backend for the VRF/VRV technical chatbot. See
`Documentation/system-design/` (repo root) for architecture and API
contracts, and `Documentation/project-milestones/01-phase-0-foundation.md`
for the current implementation task list.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) — all dependency management, virtualenv
  and script execution goes through `uv`, not `pip`/`poetry` directly.
- Docker (via WSL only — see repo root `CLAUDE.md` §5) for Postgres, Redis,
  Qdrant, MinIO, etc.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in real values; never commit .env
```

## Running

```bash
uv run uvicorn app.main:app --reload
```

`GET /api/v1/health` should respond `{"status": "ok"}`.

## Database migrations (Alembic)

All schema changes — including seed/data migrations (roles, scopes, admin
bootstrap) — go through Alembic. Never call `Base.metadata.create_all()`
against a real database.

```bash
uv run alembic upgrade head        # apply all migrations
uv run alembic downgrade base      # roll back everything (dev/testing only)
uv run alembic revision --autogenerate -m "description"   # new migration
```

The connection URL is built from `DB_ENGINE`/`DB_HOST`/... in `.env` (see
`app/db/engine.py` and `alembic/env.py`) — `alembic.ini`'s `sqlalchemy.url` is
intentionally left blank.

## Tests

```bash
uv run pytest
```

Coverage is enforced at 100% (`--cov-fail-under=100`, see `pyproject.toml`).

## Lint / type-check

```bash
uv run ruff check .
uv run mypy app
```
