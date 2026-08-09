"""FastAPI application factory.

Kept as a factory function (`create_app`) rather than a bare module-level
`app = FastAPI()` so tests can construct isolated app instances with
overridden settings/dependencies.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_v1_router
from app.api.v1.internal_storage import router as internal_storage_router
from app.core.config import Settings, get_settings
from app.core.observability import configure_logging, configure_tracing
from app.llm_providers.factory import validate_llm_providers_or_raise
from app.storage.factory import validate_object_storage_or_raise


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application instance.

    Fail-fast provider validation runs here (not lazily on first request) —
    see `Documentation/system-design/04-provider-abstractions.md` Bagian A/B:
    an invalid `CHAT_LLM_*`/`EVAL_JUDGE_LLM_*`/`OBJECT_STORAGE_*` combination
    must prevent the app from starting, with a clear error, rather than fail
    silently later.
    """
    settings = settings or get_settings()

    configure_logging(settings)
    configure_tracing(settings)
    validate_llm_providers_or_raise(settings)
    validate_object_storage_or_raise(settings)

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
    app.include_router(internal_storage_router)

    # Make routes that depend on `get_settings` (e.g. the internal local
    # storage route) see the *same* settings instance this app was built
    # with, even when a custom `settings` was passed in (tests, multiple
    # app instances in-process) rather than the process-wide cached one.
    app.dependency_overrides[get_settings] = lambda: settings

    # Make routes that depend on `get_settings` see the *same* settings
    # instance this app was built with, even when a custom `settings` was
    # passed in (tests, multiple app instances in-process) rather than the
    # process-wide cached one — otherwise e.g. JWT verification in
    # app/auth/security.py would silently use real-env settings instead of
    # the test's, causing hard-to-diagnose signature mismatches.
    app.dependency_overrides[get_settings] = lambda: settings

    return app


app = create_app()
