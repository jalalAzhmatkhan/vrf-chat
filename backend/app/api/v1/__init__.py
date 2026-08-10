"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.v1 import admin_rbac, auth, chat, conversations, documents, elements, health

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(auth.router)
api_v1_router.include_router(admin_rbac.router)
api_v1_router.include_router(documents.router)
api_v1_router.include_router(elements.router)
api_v1_router.include_router(chat.router)
api_v1_router.include_router(conversations.router)
