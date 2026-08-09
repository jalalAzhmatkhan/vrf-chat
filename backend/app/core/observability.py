"""Structured logging + OpenTelemetry-compatible tracing skeleton.

Design notes (see `Documentation/system-design/01-architecture-overview.md`
§5): a team without dedicated DevOps still needs structured logs and
end-to-end tracing from day one, since the async ingestion pipeline is easy
to turn into a black box otherwise. This module is intentionally vendor
neutral (plain OTel SDK, not Logfire) so it can be exported to any OTel
compatible backend later purely via `OTEL_EXPORTER_OTLP_ENDPOINT`.
"""

from __future__ import annotations

import logging
from logging.config import dictConfig

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from app.core.config import Settings

_TRACING_CONFIGURED = False


def configure_logging(settings: Settings) -> None:
    """Configure structured (JSON) logging for the whole process.

    Every log record includes at least: timestamp, level, logger name, and
    message. Call sites that need extra structured fields (e.g. ingestion
    stage tracking: `document_hash`, `stage`, `status`, `duration_ms`) should
    pass them via the `extra=` kwarg of the stdlib `logging` calls.
    """
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.json.JsonFormatter",
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                },
            },
            "root": {
                "handlers": ["console"],
                "level": settings.LOG_LEVEL,
            },
        }
    )


def configure_tracing(settings: Settings) -> trace.Tracer:
    """Configure an OTel `TracerProvider` and return the app-wide tracer.

    If `OTEL_EXPORTER_OTLP_ENDPOINT` is not set, spans are still created (so
    tracing code paths are exercised) but exported to a `ConsoleSpanExporter`
    only — this keeps the tracing skeleton functional without requiring an
    OTel collector to be running in dev.
    """
    global _TRACING_CONFIGURED

    resource = Resource.create({SERVICE_NAME: settings.OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    if not _TRACING_CONFIGURED:
        trace.set_tracer_provider(provider)
        _TRACING_CONFIGURED = True

    return trace.get_tracer(settings.OTEL_SERVICE_NAME)


def get_logger(name: str) -> logging.Logger:
    """Thin wrapper so call sites don't need to import stdlib logging directly."""
    return logging.getLogger(name)
