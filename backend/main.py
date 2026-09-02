"""Main FastAPI application for SATQuery AI."""

from fastapi import FastAPI
from backend.routes import health, analyze

app = FastAPI(
    title="SATQuery AI API",
    description="Evidence-first agentic geospatial intelligence system.",
    version="1.0.0"
)

# Include routers
app.include_router(health.router)
app.include_router(analyze.router, prefix="/api/v1")

# Mount frontend prototype UI
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
