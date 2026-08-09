"""Liveness/readiness endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Simple liveness probe — does not check downstream dependencies.

    Deeper readiness checks (DB, Qdrant, object storage, Redis reachability)
    can be added in a future `/health/ready` endpoint once those modules
    exist; kept minimal for B0.1 scaffold per
    `Documentation/project-milestones/01-phase-0-foundation.md`.
    """
    return {"status": "ok"}
