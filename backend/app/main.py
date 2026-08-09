"""FastAPI application factory.

Kept as a factory function (`create_app`) rather than a bare module-level
`app = FastAPI()` so tests can construct isolated app instances with
overridden settings/dependencies.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_v1_router
from app.core.config import Settings, get_settings
from app.core.observability import configure_logging, configure_tracing


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application instance."""
    settings = settings or get_settings()

    configure_logging(settings)
    configure_tracing(settings)

    app = FastAPI(
        title=settings.APP_NAME,
        debug=settings.DEBUG,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_app()
