"""
Main FastAPI application entry point.
Personal AI Assistant - Chat Interface
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import config
from app.api.chat import router as chat_router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if config.settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.settings.app_name,
    version=config.settings.app_version,
    description="A personal AI assistant with conversation memory and extensible tool support",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allows frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix=config.settings.api_prefix)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": config.settings.app_name,
        "version": config.settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "provider": config.settings.llm_provider
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {config.settings.app_name} on {config.settings.host}:{config.settings.port}")
    uvicorn.run(
        "main:app",
        host=config.settings.host,
        port=config.settings.port,
        reload=config.settings.debug
    )
