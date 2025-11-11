"""
Orchestration Service - Main Application
Manages vLLM deployments on Kubernetes with GPU allocation
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.core.config import settings
from src.core.database import init_db, close_db
from src.core.kubernetes_client import init_k8s_client
from src.api import internal, admin, public
from src.services.health_monitor import HealthMonitor
from src.services.resource_tracker import ResourceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
health_monitor: Optional[HealthMonitor] = None
resource_tracker: Optional[ResourceTracker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global health_monitor, resource_tracker

    logger.info("Starting Orchestration Service...")

    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized")

        # Initialize Kubernetes client
        init_k8s_client()
        logger.info("Kubernetes client initialized")

        # Start background services
        health_monitor = HealthMonitor()
        resource_tracker = ResourceTracker()

        # Start monitoring tasks
        asyncio.create_task(health_monitor.start_monitoring())
        asyncio.create_task(resource_tracker.start_tracking())

        # Start deployment status monitoring worker
        from src.services.deployment_manager import DeploymentManager
        deployment_manager = DeploymentManager()
        asyncio.create_task(deployment_manager.deployment_status_worker())

        logger.info("Background services started")
        logger.info(f"Orchestration Service running on port {settings.PORT}")

        yield

    finally:
        logger.info("Shutting down Orchestration Service...")

        # Stop background services
        if health_monitor:
            await health_monitor.stop_monitoring()
        if resource_tracker:
            await resource_tracker.stop_tracking()

        # Close database connection
        await close_db()

        logger.info("Orchestration Service stopped")


# Create FastAPI application
app = FastAPI(
    title="AI Platform - Orchestration Service",
    description="Manages vLLM deployments and GPU resources on Kubernetes",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(internal.router, prefix="/internal", tags=["Internal API"])
app.include_router(admin.router, prefix="/admin/v1", tags=["Admin API"])
app.include_router(public.router, prefix="/api/v1", tags=["Public API"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Orchestration Service",
        "version": "1.0.0",
        "status": "running",
        "port": settings.PORT
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": "orchestration-service",
        "database": "connected",
        "kubernetes": "connected"
    }

    try:
        # Check database connection
        from src.core.database import check_db_health
        db_healthy = await check_db_health()
        health_status["database"] = "connected" if db_healthy else "disconnected"

        # Check Kubernetes connection
        from src.core.kubernetes_client import check_k8s_health
        k8s_healthy = check_k8s_health()
        health_status["kubernetes"] = "connected" if k8s_healthy else "disconnected"

        if not db_healthy or not k8s_healthy:
            health_status["status"] = "degraded"

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)

    return health_status


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8003"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )