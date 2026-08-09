from celery import Celery

from app.workers.celery_app import celery_app, create_celery_app


def test_celery_app_is_celery_instance() -> None:
    assert isinstance(celery_app, Celery)
    assert celery_app.main == "vrf_chat"


def test_create_celery_app_configures_default_queue() -> None:
    app = create_celery_app()

    assert app.conf.task_default_queue == "default"
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.timezone == "UTC"
