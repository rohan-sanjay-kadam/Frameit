"""
app.py — Frameit MVP
====================
FastAPI application factory. Synchronous pipeline — no Celery, no Redis.
Run: uvicorn app:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import get_config
from api.routes import upload, generate, download, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    Path(cfg.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    cfg = get_config()

    app = FastAPI(
        title="Frameit MVP",
        description="AI-powered Instagram collage generator",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(upload.router,   prefix="/api/v1", tags=["upload"])
    app.include_router(generate.router, prefix="/api/v1", tags=["generate"])
    app.include_router(download.router, prefix="/api/v1", tags=["download"])

    app.mount(
        "/output",
        StaticFiles(directory=cfg.output_dir),
        name="output",
    )

    return app


app = create_app()