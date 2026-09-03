"""Main FastAPI application for SATQuery AI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import health, analyze, upload

app = FastAPI(
    title="SATQuery AI API",
    description="Evidence-first agentic geospatial intelligence system.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(analyze.router, prefix="/api/v1")
app.include_router(upload.router, prefix="/api/v1")

# Mount frontend prototype UI
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="Frontend", html=True), name="frontend")
