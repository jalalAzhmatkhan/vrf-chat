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

### Auth/RBAC seed data + admin bootstrap

`alembic upgrade head` also runs a **data migration**
(`184c1548d41a_seed_auth_rbac_roles_scopes_and_.py`) that seeds the 2 default
roles + 10 scopes + `role_scopes` mapping (`admin` gets every scope, `user`
gets `chat:read`/`chat:write`/`documents:read`) per
`Documentation/system-design/08-authentication-rbac.md` §4, and idempotently
bootstraps the first admin account from `FIRST_SUPERUSER_EMAIL`/
`FIRST_SUPERUSER_PASSWORD` (skipped if unset, or if a user with that email
already exists — safe to re-run). This replaces the standalone bootstrap
script originally sketched as a separate CLI tool.

**Convention for future scopes**: any migration that adds a new row to
`scopes` MUST also insert the corresponding `role_scopes` row for `admin` in
the *same* migration (see `08-authentication-rbac.md` §3.4 — admin access is
enforced via explicit, auditable `role_scopes` rows, never a hardcoded
`if role == "admin"` bypass).

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

## Docker (via WSL only — see repo root `CLAUDE.md` §5)

`docker`/`docker compose` must be run from a WSL shell, never Windows-native
PowerShell/CMD. From WSL, at the `vrf-chat/` repo root (not `backend/`):

```bash
docker compose up --build                 # backend-api, backend-worker, redis, postgres, qdrant, minio, frontend
docker compose --profile gpu up --build   # + backend-worker-gpu (GPU ingestion queue)
docker compose --profile kg up --build    # + neo4j (Fase 3, off by default)
```

All host-published ports are overridable via env vars if they collide with
other projects on a shared dev machine, e.g.
`POSTGRES_HOST_PORT=15432 docker compose up` — see comments at the top of
`../docker-compose.yml`.

`backend/Dockerfile` is multi-stage/multi-target (`api`/`worker`/
`worker-gpu`), all installed via `uv` (never bare pip).
