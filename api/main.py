from contextlib import asynccontextmanager
from fastapi import FastAPI

from .config import settings
from .routers import health_router, recommendation_router
from .vespa_client import get_vespa_client


# ---------------------------------------------------------
# Lifespan Context Manager
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events.
    Initializes the Vespa client connection to ensure availability.
    """
    get_vespa_client()

    yield


# ---------------------------------------------------------
# FastAPI Application Configuration
# ---------------------------------------------------------
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Router Registration
# ---------------------------------------------------------
app.include_router(health_router)
app.include_router(recommendation_router)


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------
@app.get("/")
def root():
    """
    Root endpoint to verify service status and get metadata.
    """
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs_url": "/docs",
    }
