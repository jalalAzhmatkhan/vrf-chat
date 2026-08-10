"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.v1 import admin_rbac, auth, documents, health

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(auth.router)
api_v1_router.include_router(admin_rbac.router)
api_v1_router.include_router(documents.router)
