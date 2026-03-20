"""
api/routes/health.py
====================
GET /health — liveness probe.
Returns 200 immediately. Used by docker-compose healthcheck and any future
load balancer or k8s readiness probe.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")
