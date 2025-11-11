"""
Database models for Orchestration Service
"""

from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, JSON,
    ForeignKey, UniqueConstraint, Index, Text, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from src.core.database import Base
from src.core.config import settings


class DeploymentStatus(str, Enum):
    """Deployment status enum"""
    PENDING = "pending"
    CREATING = "creating"
    RUNNING = "running"
    SCALING = "scaling"
    UPDATING = "updating"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    FAILED = "failed"
    ERROR = "error"


class PodStatus(str, Enum):
    """Pod status enum"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class NodeStatus(str, Enum):
    """Node status enum"""
    READY = "ready"
    NOT_READY = "not_ready"
    DRAINING = "draining"
    DRAINED = "drained"
    MAINTENANCE = "maintenance"
    DELETED = "deleted"


class GPUNode(Base):
    """GPU node information"""
    __tablename__ = "gpu_nodes"
    __table_args__ = {"schema": settings.DB_SCHEMA}

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String(255), unique=True, nullable=False, index=True)
    node_name = Column(String(255), nullable=False)
    node_ip = Column(String(45), nullable=False)

    # GPU information
    gpu_type = Column(String(100), nullable=False)
    gpu_count = Column(Integer, nullable=False)
    gpu_memory_gb = Column(Integer, nullable=False)

    # Resource capacity
    cpu_cores = Column(Integer, nullable=False)
    memory_gb = Column(Integer, nullable=False)
    storage_gb = Column(Integer, nullable=False)

    # Current usage
    gpus_allocated = Column(Integer, default=0, nullable=False)
    cpu_allocated = Column(Float, default=0.0, nullable=False)
    memory_allocated_gb = Column(Float, default=0.0, nullable=False)

    # Status
    status = Column(String(50), default=NodeStatus.READY, nullable=False)
    is_schedulable = Column(Boolean, default=True, nullable=False)

    # Metadata
    labels = Column(JSON, default=dict)
    annotations = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_heartbeat = Column(DateTime, server_default=func.now())

    # Relationships
    pod_allocations = relationship("PodAllocation", back_populates="gpu_node", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<GPUNode {self.node_name} ({self.gpu_type} x{self.gpu_count})>"


class PodAllocation(Base):
    """Pod to GPU allocation mapping"""
    __tablename__ = "pod_allocations"
    __table_args__ = (
        UniqueConstraint('pod_name', 'namespace', name='_pod_namespace_uc'),
        {"schema": settings.DB_SCHEMA}
    )

    id = Column(Integer, primary_key=True, index=True)
    allocation_id = Column(String(255), unique=True, nullable=False, index=True)

    # Deployment information
    deployment_id = Column(String(255), ForeignKey(f'{settings.DB_SCHEMA}.deployments.deployment_id', ondelete='CASCADE'), nullable=False, index=True)
    deployment_name = Column(String(255), nullable=False)
    model_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Pod information
    pod_name = Column(String(255), nullable=False)
    pod_uid = Column(String(255), unique=True, nullable=True)
    namespace = Column(String(255), nullable=False, default=settings.K8S_NAMESPACE)
    pod_ip = Column(String(45), nullable=True)

    # Node allocation
    node_id = Column(String(255), ForeignKey(f"{settings.DB_SCHEMA}.gpu_nodes.node_id"), nullable=False)
    gpu_count = Column(Integer, nullable=False, default=1)
    gpu_indices = Column(JSON, nullable=True)  # List of GPU indices on the node

    # Resource allocation
    cpu_request = Column(Float, nullable=False)
    cpu_limit = Column(Float, nullable=False)
    memory_request_gb = Column(Float, nullable=False)
    memory_limit_gb = Column(Float, nullable=False)

    # Service information
    service_name = Column(String(255), nullable=True)
    service_port = Column(Integer, nullable=True)
    node_port = Column(Integer, nullable=True)

    # Status
    status = Column(String(50), default=PodStatus.PENDING, nullable=False)
    is_ready = Column(Boolean, default=False, nullable=False)
    restart_count = Column(Integer, default=0, nullable=False)

    # Configuration
    container_image = Column(String(500), nullable=False)
    environment_vars = Column(JSON, default=dict)
    model_config = Column(JSON, default=dict)

    # Metrics
    gpu_utilization_percent = Column(Float, nullable=True)
    memory_utilization_percent = Column(Float, nullable=True)
    cpu_utilization_percent = Column(Float, nullable=True)
    request_count = Column(BigInteger, default=0)
    error_count = Column(BigInteger, default=0)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    started_at = Column(DateTime, nullable=True)
    terminated_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    gpu_node = relationship("GPUNode", back_populates="pod_allocations")

    def __repr__(self):
        return f"<PodAllocation {self.pod_name} on {self.node_id}>"


class Deployment(Base):
    """vLLM deployment information"""
    __tablename__ = "deployments"
    __table_args__ = {"schema": settings.DB_SCHEMA}

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(String(255), unique=True, nullable=False, index=True)
    deployment_name = Column(String(255), nullable=False)

    # Tenant and model information
    # Note: Stored as VARCHAR(255) in database containing UUID strings
    tenant_id = Column(String(255), nullable=False, index=True)
    model_id = Column(String(255), nullable=False, index=True)
    model_name = Column(String(255), nullable=False)
    engine_type = Column(String(50), default="vllm", nullable=False)  # Inference engine: vllm, tgi, tensorrt, etc.

    # Deployment configuration
    replicas = Column(Integer, default=1, nullable=False)

    # Replica tracking
    desired_replicas = Column(Integer, default=1, nullable=False)  # Target state
    current_replicas = Column(Integer, default=0, nullable=False)  # Actual running state

    # Resource requirements
    gpu_per_replica = Column(Integer, default=1, nullable=False)
    cpu_request = Column(Float, nullable=False)
    cpu_limit = Column(Float, nullable=False)
    memory_request_gb = Column(Float, nullable=False)
    memory_limit_gb = Column(Float, nullable=False)

    # Model configuration
    model_path = Column(String(500), nullable=False)
    model_params = Column(JSON, default=dict)
    vllm_config = Column(JSON, default=dict)

    # Status
    status = Column(String(50), default=DeploymentStatus.PENDING, nullable=False)
    ready_replicas = Column(Integer, default=0, nullable=False)
    available_replicas = Column(Integer, default=0, nullable=False)

    # Endpoints
    internal_endpoint = Column(String(500), nullable=True)
    external_endpoint = Column(String(500), nullable=True)

    # Metadata
    labels = Column(JSON, default=dict)
    annotations = Column(JSON, default=dict)

    # Health tracking
    error_message = Column(Text, nullable=True)
    last_health_check = Column(DateTime, nullable=True)
    health_check_failures = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    deployed_at = Column(DateTime, nullable=True)
    terminated_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<Deployment {self.deployment_name} ({self.status})>"


# Create indexes for better query performance
Index('idx_pod_allocations_tenant', PodAllocation.tenant_id)
Index('idx_pod_allocations_model', PodAllocation.model_id)
Index('idx_pod_allocations_status', PodAllocation.status)
Index('idx_deployments_tenant', Deployment.tenant_id)
Index('idx_deployments_status', Deployment.status)