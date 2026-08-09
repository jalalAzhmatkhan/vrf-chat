"""Celery application factory + instance.

Deliberately minimal at this stage (B0.5, docker-compose skeleton): no real
tasks are registered yet — those arrive with `app/ingestion/`,
`app/evaluation/` in later phases. This module exists now so
`backend-worker`/`backend-worker-gpu` (see `Documentation/system-design/
01-architecture-overview.md` §6) have something real to boot in Docker
Compose without error, with the queue split (`default` vs `gpu`) already
wired per `04-provider-abstractions.md`/`01-architecture-overview.md`
DIRECT MESSAGE to Backend Engineer: GPU-bound ingestion jobs must never
queue behind/in front of lightweight jobs.
"""

from __future__ import annotations

from celery import Celery

from app.core.config import get_settings


def create_celery_app() -> Celery:
    settings = get_settings()

    app = Celery("vrf_chat", broker=settings.redis_url, backend=settings.redis_url)
    app.conf.update(
        task_default_queue=settings.CELERY_TASK_DEFAULT_QUEUE,
        task_track_started=True,
        worker_prefetch_multiplier=1,
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = create_celery_app()
