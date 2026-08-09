"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.v1 import auth, health

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(auth.router)
