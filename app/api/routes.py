"""
Main API routes aggregator.
Combines all API routers into a single router.
"""
from fastapi import APIRouter
from app.api.chat import router as chat_router
import config

# Create main router
router = APIRouter()

# Include all sub-routers
router.include_router(chat_router, prefix=config.settings.api_prefix)
