"""
Pydantic schemas for API requests and responses
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum


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


# Request schemas
class CreateDeploymentRequest(BaseModel):
    """Request to create a new vLLM deployment"""
    model_config = ConfigDict(protected_namespaces=())

    tenant_id: str = Field(..., description="Tenant ID")
    model_id: str = Field(..., description="Model ID")
    model_name: str = Field(..., description="Model name")
    model_path: str = Field(..., description="Path to model weights")

    # Deployment configuration
    replicas: int = Field(default=1, ge=1, le=10, description="Number of replicas")
    gpu_per_replica: int = Field(default=1, ge=1, le=32, description="GPUs per replica")

    # Resource configuration
    cpu_request: Optional[str] = Field(default="4", description="CPU request")
    cpu_limit: Optional[str] = Field(default="8", description="CPU limit")
    memory_request: Optional[str] = Field(default="32Gi", description="Memory request")
    memory_limit: Optional[str] = Field(default="64Gi", description="Memory limit")

    # Model configuration
    model_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model parameters")
    vllm_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="vLLM configuration")
    environment_vars: Optional[Dict[str, str]] = Field(default_factory=dict, description="Environment variables")

    # Metadata
    labels: Optional[Dict[str, str]] = Field(default_factory=dict, description="Kubernetes labels")
    annotations: Optional[Dict[str, str]] = Field(default_factory=dict, description="Kubernetes annotations")


class UpdateDeploymentRequest(BaseModel):
    """Request to update a deployment"""
    model_config = ConfigDict(protected_namespaces=())

    cpu_request: Optional[str] = Field(None, description="CPU request")
    cpu_limit: Optional[str] = Field(None, description="CPU limit")
    memory_request: Optional[str] = Field(None, description="Memory request")
    memory_limit: Optional[str] = Field(None, description="Memory limit")

    model_params: Optional[Dict[str, Any]] = Field(None, description="Model parameters")
    vllm_config: Optional[Dict[str, Any]] = Field(None, description="vLLM configuration")
    environment_vars: Optional[Dict[str, str]] = Field(None, description="Environment variables")

    labels: Optional[Dict[str, str]] = Field(None, description="Kubernetes labels")
    annotations: Optional[Dict[str, str]] = Field(None, description="Kubernetes annotations")


# Response schemas
class DeploymentResponse(BaseModel):
    """Deployment information response"""
    model_config = ConfigDict(protected_namespaces=())

    deployment_id: str
    deployment_name: str
    tenant_id: str
    model_id: str
    model_name: str

    status: DeploymentStatus
    replicas: int
    ready_replicas: int
    available_replicas: int
    desired_replicas: Optional[int] = None  # Target number of replicas
    current_replicas: Optional[int] = None  # Current running replicas

    gpu_per_replica: int
    total_gpus: int

    internal_endpoint: Optional[str]
    external_endpoint: Optional[str]

    created_at: datetime
    updated_at: datetime
    deployed_at: Optional[datetime]

    # Configuration
    vllm_config: Optional[Dict[str, Any]] = None

    # Optional message field for async operations
    message: Optional[str] = None


class DeploymentStatusResponse(BaseModel):
    """Deployment status response"""
    deployment_id: str
    status: DeploymentStatus
    replicas: int
    ready_replicas: int
    available_replicas: int
    desired_replicas: Optional[int] = None  # Target number of replicas
    current_replicas: Optional[int] = None  # Current running replicas

    pods: List[Dict[str, Any]]
    conditions: List[Dict[str, Any]]

    message: Optional[str]
    last_updated: datetime


class PodAllocationResponse(BaseModel):
    """Pod allocation information"""
    model_config = ConfigDict(protected_namespaces=())

    allocation_id: str
    pod_name: str
    pod_uid: Optional[str]
    namespace: str

    deployment_id: str
    model_id: str
    tenant_id: str

    node_id: str
    node_name: Optional[str]
    pod_ip: Optional[str]

    status: PodStatus
    is_ready: bool

    gpu_count: int
    cpu_request: float
    memory_request_gb: float

    service_port: Optional[int]
    node_port: Optional[int]

    created_at: datetime
    started_at: Optional[datetime]


class ResourceAvailabilityResponse(BaseModel):
    """Resource availability response"""
    total_gpus: int
    available_gpus: int
    allocated_gpus: int

    total_nodes: int
    ready_nodes: int

    nodes: List[Dict[str, Any]]

    can_schedule: bool
    max_deployable_gpus: int


class GPUNodeResponse(BaseModel):
    """GPU node information"""
    node_id: str
    node_name: str
    node_ip: str

    gpu_type: str
    gpu_count: int
    gpu_memory_gb: int
    gpus_allocated: int
    gpus_available: int

    cpu_cores: int
    memory_gb: int

    status: NodeStatus
    is_schedulable: bool

    pod_count: int
    pods: Optional[List[str]]

    created_at: datetime
    updated_at: datetime
    last_heartbeat: datetime


class GPUUtilizationResponse(BaseModel):
    """GPU utilization metrics"""
    timestamp: datetime

    cluster_metrics: Dict[str, Any]
    node_metrics: List[Dict[str, Any]]
    deployment_metrics: List[Dict[str, Any]]

    summary: Dict[str, float]


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: Optional[str]
    details: Optional[Dict[str, Any]]
    status_code: int


class SuccessResponse(BaseModel):
    """Success response"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None