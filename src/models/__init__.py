"""
Models module for Orchestration Service
"""

from src.core.database import Base
from src.models.db_models import (
    GPUNode, PodAllocation, Deployment, ScalingEvent,
    DeploymentStatus, PodStatus, NodeStatus
)
from src.models.schemas import (
    CreateDeploymentRequest, DeploymentResponse, DeploymentStatusResponse,
    ResourceAvailabilityResponse, GPUNodeResponse, GPUUtilizationResponse,
    ErrorResponse, SuccessResponse
)

__all__ = [
    # SQLAlchemy Base
    "Base",

    # Database models
    "GPUNode", "PodAllocation", "Deployment", "ScalingEvent",
    "DeploymentStatus", "PodStatus", "NodeStatus",

    # API schemas
    "CreateDeploymentRequest", "DeploymentResponse", "DeploymentStatusResponse",
    "ResourceAvailabilityResponse", "GPUNodeResponse", "GPUUtilizationResponse",
    "ErrorResponse", "SuccessResponse"
]