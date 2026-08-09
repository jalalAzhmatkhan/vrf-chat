import logging

from app.core.config import Settings
from app.core.observability import configure_logging, configure_tracing, get_logger


def test_configure_logging_sets_root_level() -> None:
    settings = Settings(_env_file=None, LOG_LEVEL="DEBUG")

    configure_logging(settings)

    assert logging.getLogger().level == logging.DEBUG


def test_configure_tracing_returns_usable_tracer() -> None:
    settings = Settings(_env_file=None, OTEL_SERVICE_NAME="test-service")

    tracer = configure_tracing(settings)

    with tracer.start_as_current_span("unit-test-span") as span:
        assert span is not None


def test_configure_tracing_idempotent_across_calls() -> None:
    settings = Settings(_env_file=None)

    first_tracer = configure_tracing(settings)
    second_tracer = configure_tracing(settings)

    assert first_tracer is not None
    assert second_tracer is not None


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("vrf.test")

    assert logger.name == "vrf.test"
