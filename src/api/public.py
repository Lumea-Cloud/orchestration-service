"""
Public API endpoints for model deployment
Used by external clients and Model Registry Service
"""

import uuid
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Header
from sqlalchemy import select, and_
from pydantic import BaseModel, Field, ConfigDict

from src.core.database import get_db, AsyncSession
from src.models.db_models import Deployment
from src.models.schemas import (
    CreateDeploymentRequest,
    DeploymentResponse,
    DeploymentStatusResponse,
    SuccessResponse,
    UpdateDeploymentRequest
)
from src.services.deployment_manager import DeploymentManager
from src.services.resource_allocator import ResourceAllocator
from src.services.model_registry_client import ModelRegistryClient

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
deployment_manager = DeploymentManager()
resource_allocator = ResourceAllocator()
model_registry_client = ModelRegistryClient()


class SimpleDeploymentRequest(BaseModel):
    """Simplified deployment request from Model Registry"""
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(..., description="Model ID from registry")
    model_name: Optional[str] = Field(None, description="Model name (fetched from registry if not provided)")
    model_path: Optional[str] = Field(None, description="Path to model files (fetched from registry if not provided)")
    tenant_id: Optional[str] = Field(None, description="Tenant ID (can be provided in body or X-Tenant-ID header)")
    replicas: int = Field(default=1, ge=1, le=10)
    gpu_type: Optional[str] = Field(default="L40S", description="GPU type")
    gpu_count: Optional[int] = Field(None, description="GPUs per replica (auto-detected if not specified)")
    huggingface_token: Optional[str] = Field(None, description="HuggingFace API token for accessing private models")
    # Additional model information from Model Registry
    model_requirements: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model resource requirements")
    model_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model metadata")
    model_capabilities: Optional[List[str]] = Field(default_factory=list, description="Model capabilities")
    model_type: Optional[str] = Field(None, description="Model type")
    model_family: Optional[str] = Field(None, description="Model family")
    model_size: Optional[str] = Field(None, description="Model size")
    model_version: Optional[str] = Field(None, description="Model version")
    model_pricing: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model pricing information")


@router.post("/deployments", response_model=DeploymentResponse, status_code=202)
async def create_deployment(
    request: SimpleDeploymentRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    tenant_id_header: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Create a new model deployment (async)

    This endpoint initiates a new vLLM deployment on Kubernetes with automatic GPU allocation.
    Returns immediately with 202 Accepted. Use GET /deployments/{deployment_id} to check status.
    """
    try:
        # Get tenant_id from request body or header (header takes precedence as it comes from Gateway)
        tenant_id = tenant_id_header or request.tenant_id
        if not tenant_id:
            raise HTTPException(
                status_code=400,
                detail="Tenant ID is required (via X-Tenant-ID header or in request body)"
            )

        # Use the resolved tenant_id for all operations
        request.tenant_id = tenant_id

        logger.info(f"Received deployment request for model {request.model_id} from tenant {tenant_id}")

        # Check if model_name and model_path are provided, if not fetch from Model Registry
        if not request.model_name or not request.model_path:
            logger.info(f"Fetching model details from Model Registry for model {request.model_id}")

            # Fetch model details from Model Registry using the public API endpoint
            model_details = await model_registry_client.get_model(request.model_id)

            if not model_details:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {request.model_id} not found in Model Registry"
                )

            # Check if model has been soft deleted
            if model_details.get("deleted_at") is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {request.model_id} has been deleted and cannot be deployed"
                )

            # Extract model information from the registry response
            model_name = model_details.get("name")
            if not model_name:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model {request.model_id} has no name in registry"
                )

            # Determine the model path (HuggingFace ID)
            # Priority: huggingface_id > source_url > path field
            model_path = None

            # Check for huggingface_id field (top level or in metadata)
            if model_details.get("huggingface_id"):
                model_path = model_details["huggingface_id"]
            elif model_details.get("metadata", {}).get("huggingface_id"):
                model_path = model_details["metadata"]["huggingface_id"]
            # Check for source_url at top level (for Hugging Face models)
            elif model_details.get("source_url"):
                source_url = model_details["source_url"]
                if "huggingface.co/" in source_url:
                    # Extract the model ID from URL like https://huggingface.co/google/gemma-2-27b-it
                    model_path = source_url.split("huggingface.co/")[-1].strip("/")
                else:
                    model_path = source_url
            # Check in metadata for source_url
            elif model_details.get("metadata", {}).get("source_url"):
                source_url = model_details["metadata"]["source_url"]
                if "huggingface.co/" in source_url:
                    # Extract the model ID from URL
                    model_path = source_url.split("huggingface.co/")[-1].strip("/")
                else:
                    model_path = source_url
            # Check if there's a direct path field (should be full HF ID, not just name)
            elif model_details.get("path"):
                model_path = model_details["path"]
            elif model_details.get("metadata", {}).get("path"):
                model_path = model_details["metadata"]["path"]

            # If still no path, log warning and raise error
            # We should NOT fall back to just the model name as it won't work with vLLM
            if not model_path:
                logger.error(
                    f"Could not determine HuggingFace ID for model {request.model_id}. "
                    f"Model Registry should provide huggingface_id or source_url. "
                    f"Available fields: {list(model_details.keys())}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Model {request.model_id} missing huggingface_id or source_url in registry"
                )

            # Update request with fetched values
            request.model_name = model_name
            request.model_path = model_path

            # Also update other optional fields if not provided
            if not request.model_requirements and model_details.get("requirements"):
                request.model_requirements = model_details["requirements"]
            if not request.model_metadata and model_details.get("metadata"):
                request.model_metadata = model_details["metadata"]
            if not request.model_capabilities and model_details.get("capabilities"):
                request.model_capabilities = model_details["capabilities"]
            if not request.model_type and model_details.get("type"):
                request.model_type = model_details["type"]
            if not request.model_family and model_details.get("family"):
                request.model_family = model_details["family"]
            if not request.model_size and model_details.get("size"):
                request.model_size = model_details["size"]
            if not request.model_version and model_details.get("version"):
                request.model_version = model_details["version"]
            if not request.model_pricing and model_details.get("pricing"):
                request.model_pricing = model_details["pricing"]

            logger.info(f"Fetched model details: name={model_name}, path={model_path}")

        # Use model details from request (either provided or fetched)
        model_info = {
            "model_id": request.model_id,
            "model_name": request.model_name,
            "path": request.model_path,
            "requirements": request.model_requirements,
            "metadata": request.model_metadata,
            "capabilities": request.model_capabilities,
            "type": request.model_type,
            "family": request.model_family,
            "size": request.model_size,
            "version": request.model_version,
            "pricing": request.model_pricing,
            "size_gb": request.model_requirements.get("memory_gb", 50) if request.model_requirements else 50
        }

        # Use model requirements if available, otherwise auto-detect based on model name
        if request.model_requirements and request.model_requirements.get("gpu_count"):
            # Use requirements from model registry
            gpu_per_replica = request.gpu_count or request.model_requirements.get("gpu_count", 1)
            memory_request = f"{request.model_requirements.get('memory_gb', 32)}Gi"
            memory_limit = f"{int(request.model_requirements.get('memory_gb', 32) * 1.5)}Gi"
            cpu_request = str(request.model_requirements.get("cpu_cores", 4))
            cpu_limit = str(int(request.model_requirements.get("cpu_cores", 4) * 2))
        else:
            # Auto-detect GPU requirements based on model name
            model_name_lower = request.model_name.lower()
            if "27b" in model_name_lower or "30b" in model_name_lower:
                gpu_per_replica = request.gpu_count or 2
                memory_request = "64Gi"
                memory_limit = "96Gi"
                cpu_request = "8"
                cpu_limit = "16"
            elif "13b" in model_name_lower or "14b" in model_name_lower:
                gpu_per_replica = request.gpu_count or 1
                memory_request = "32Gi"
                memory_limit = "48Gi"
                cpu_request = "4"
                cpu_limit = "8"
            elif "7b" in model_name_lower or "8b" in model_name_lower:
                gpu_per_replica = request.gpu_count or 1
                memory_request = "16Gi"
                memory_limit = "24Gi"
                cpu_request = "2"
                cpu_limit = "4"
            else:
                # Default for large models
                gpu_per_replica = request.gpu_count or 1
                memory_request = "32Gi"
                memory_limit = "48Gi"
                cpu_request = "4"
                cpu_limit = "8"

        # Generate deployment ID
        deployment_id = f"vllm-{request.model_id}-{uuid.uuid4().hex[:8]}"
        deployment_name = f"vllm-{request.model_id.replace('/', '-').replace('_', '-').lower()}"

        # Check resource availability
        availability = await resource_allocator.check_availability(
            gpu_count=gpu_per_replica * request.replicas
        )

        if not availability["can_allocate"]:
            raise HTTPException(
                status_code=503,
                detail=f"Insufficient GPU resources. Required: {gpu_per_replica * request.replicas}, Available: {availability['available_gpus']}"
            )

        # Create deployment record
        deployment = Deployment(
            deployment_id=deployment_id,
            deployment_name=deployment_name,
            tenant_id=tenant_id,
            model_id=request.model_id,
            model_name=request.model_name,
            replicas=request.replicas,
            gpu_per_replica=gpu_per_replica,
            cpu_request=float(cpu_request),
            cpu_limit=float(cpu_limit),
            memory_request_gb=float(memory_request.rstrip("Gi")),
            memory_limit_gb=float(memory_limit.rstrip("Gi")),
            model_path=request.model_path,
            model_params={
                "gpu_type": request.gpu_type,
                "auto_detected": True
            },
            vllm_config={
                "tensor_parallel_size": gpu_per_replica,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096
            },
            status="creating",
            desired_replicas=request.replicas,  # Set desired replicas
            current_replicas=0,  # Initially 0 until pods are created
            labels={
                "tenant-id": tenant_id,
                "model-id": request.model_id,
                "managed-by": "ai-platform"
            }
        )

        db.add(deployment)
        await db.commit()
        await db.refresh(deployment)

        # Schedule background deployment
        background_tasks.add_task(
            deployment_manager.deploy_async,
            deployment_id=deployment_id,
            deployment_config={
                "deployment_name": deployment_name,
                "model_id": request.model_id,
                "model_path": request.model_path,
                "replicas": request.replicas,
                "gpu_count": gpu_per_replica,
                "cpu_request": cpu_request,
                "cpu_limit": cpu_limit,
                "memory_request": memory_request,
                "memory_limit": memory_limit,
                "env_vars": {
                    "MODEL_ID": request.model_id,
                    "MODEL_NAME": request.model_name,
                    "TENSOR_PARALLEL_SIZE": str(gpu_per_replica)
                },
                "labels": {
                    "tenant-id": tenant_id,
                    "model-id": request.model_id,
                    "managed-by": "ai-platform"
                },
                "model_info": model_info,
                "huggingface_token": request.huggingface_token  # Pass HF token to deployment config
            }
        )

        logger.info(f"Created deployment {deployment_id} for model {request.model_id}")

        # Return response with async pattern indication
        response = DeploymentResponse(
            deployment_id=deployment.deployment_id,
            deployment_name=deployment.deployment_name,
            tenant_id=deployment.tenant_id,
            model_id=deployment.model_id,
            model_name=deployment.model_name,
            status="creating",  # Always return "creating" for async deployment
            replicas=deployment.replicas,
            ready_replicas=0,
            available_replicas=0,
            desired_replicas=deployment.replicas,  # Set to initial replica count
            current_replicas=0,  # No replicas running yet
            gpu_per_replica=deployment.gpu_per_replica,
            total_gpus=deployment.gpu_per_replica * deployment.replicas,
            internal_endpoint=None,
            external_endpoint=None,
            created_at=deployment.created_at,
            updated_at=deployment.updated_at,
            deployed_at=None,
            message=f"Deployment is being created. Use GET /api/v1/deployments/{deployment.deployment_id} to check status"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments", response_model=List[DeploymentResponse])
async def list_deployments(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    List model deployments
    """
    try:
        query = select(Deployment)

        conditions = []
        if tenant_id:
            conditions.append(Deployment.tenant_id == tenant_id)
        if model_id:
            conditions.append(Deployment.model_id == model_id)
        if status:
            conditions.append(Deployment.status == status)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.limit(limit).offset(offset)

        result = await db.execute(query)
        deployments = result.scalars().all()

        return [
            DeploymentResponse(
                deployment_id=d.deployment_id,
                deployment_name=d.deployment_name,
                tenant_id=d.tenant_id,
                model_id=d.model_id,
                model_name=d.model_name,
                status=d.status,
                replicas=d.replicas,
                ready_replicas=d.ready_replicas,
                available_replicas=d.available_replicas,
                desired_replicas=d.desired_replicas or 1,
                current_replicas=d.current_replicas or 0,
                gpu_per_replica=d.gpu_per_replica,
                total_gpus=d.gpu_per_replica * d.replicas,
                internal_endpoint=d.internal_endpoint,
                external_endpoint=d.external_endpoint,
                created_at=d.created_at,
                updated_at=d.updated_at,
                deployed_at=d.deployed_at
            ) for d in deployments
        ]

    except Exception as e:
        logger.error(f"Failed to list deployments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get deployment details
    """
    try:
        result = await db.execute(
            select(Deployment).where(Deployment.deployment_id == deployment_id)
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")

        return DeploymentResponse(
            deployment_id=deployment.deployment_id,
            deployment_name=deployment.deployment_name,
            tenant_id=deployment.tenant_id,
            model_id=deployment.model_id,
            model_name=deployment.model_name,
            status=deployment.status,
            replicas=deployment.replicas,
            ready_replicas=deployment.ready_replicas,
            available_replicas=deployment.available_replicas,
            desired_replicas=deployment.desired_replicas or 1,
            current_replicas=deployment.current_replicas or 0,
            gpu_per_replica=deployment.gpu_per_replica,
            total_gpus=deployment.gpu_per_replica * deployment.replicas,
            internal_endpoint=deployment.internal_endpoint,
            external_endpoint=deployment.external_endpoint,
            created_at=deployment.created_at,
            updated_at=deployment.updated_at,
            deployed_at=deployment.deployed_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments/{deployment_id}/status", response_model=DeploymentStatusResponse)
async def get_deployment_status(
    deployment_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get deployment status
    """
    try:
        result = await db.execute(
            select(Deployment).where(Deployment.deployment_id == deployment_id)
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")

        # Get Kubernetes status if available
        from src.core import kubernetes_client as k8s
        k8s_status = k8s.get_deployment_status(deployment.deployment_name)

        return DeploymentStatusResponse(
            deployment_id=deployment.deployment_id,
            status=deployment.status,
            replicas=deployment.replicas,
            ready_replicas=k8s_status.get("ready_replicas", 0) if k8s_status else deployment.ready_replicas,
            available_replicas=k8s_status.get("available_replicas", 0) if k8s_status else deployment.available_replicas,
            desired_replicas=getattr(deployment, 'desired_replicas', deployment.replicas),
            current_replicas=k8s_status.get("ready_replicas", 0) if k8s_status else deployment.ready_replicas,
            pods=[],
            conditions=k8s_status.get("conditions", []) if k8s_status else [],
            message=deployment.error_message,
            last_updated=deployment.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/deployments/{deployment_id}", response_model=DeploymentResponse)
async def update_deployment(
    deployment_id: str,
    request: UpdateDeploymentRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id_header: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Update a deployment configuration with tenant validation.

    Args:
        deployment_id: Deployment ID to update
        request: Update request with new configuration
        db: Database session
        tenant_id_header: Tenant ID from header

    Returns:
        Updated deployment information

    Raises:
        HTTPException: If deployment not found, tenant mismatch, or update fails
    """
    try:
        # Get deployment from database
        result = await db.execute(
            select(Deployment).where(Deployment.deployment_id == deployment_id)
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise HTTPException(
                status_code=404,
                detail=f"Deployment {deployment_id} not found"
            )

        # Validate tenant access
        if tenant_id_header and deployment.tenant_id != tenant_id_header:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Deployment belongs to a different tenant."
            )

        # Check if deployment is in a state that allows updates
        if deployment.status not in ["running", "error"]:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot update deployment in status {deployment.status}"
            )

        # Build update configuration
        update_config = {}
        if request.cpu_request is not None:
            update_config["cpu_request"] = request.cpu_request
        if request.cpu_limit is not None:
            update_config["cpu_limit"] = request.cpu_limit
        if request.memory_request is not None:
            update_config["memory_request"] = request.memory_request
        if request.memory_limit is not None:
            update_config["memory_limit"] = request.memory_limit
        if request.model_params is not None:
            update_config["model_params"] = request.model_params
        if request.vllm_config is not None:
            update_config["vllm_config"] = request.vllm_config
        if request.environment_vars is not None:
            update_config["environment_vars"] = request.environment_vars
        if request.labels is not None:
            update_config["labels"] = request.labels
        if request.annotations is not None:
            update_config["annotations"] = request.annotations

        # Apply update
        success = await deployment_manager.update_deployment(
            deployment_id=deployment_id,
            update_config=update_config
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update deployment"
            )

        # Refresh deployment from database
        await db.refresh(deployment)

        # Return updated deployment
        return DeploymentResponse(
            deployment_id=deployment.deployment_id,
            deployment_name=deployment.deployment_name,
            tenant_id=deployment.tenant_id,
            model_id=deployment.model_id,
            model_name=deployment.model_name,
            status=deployment.status,
            replicas=deployment.replicas,
            ready_replicas=deployment.ready_replicas,
            available_replicas=deployment.available_replicas,
            desired_replicas=deployment.desired_replicas,
            current_replicas=deployment.current_replicas,
            gpu_per_replica=deployment.gpu_per_replica,
            total_gpus=deployment.gpu_per_replica * deployment.replicas,
            internal_endpoint=deployment.internal_endpoint,
            external_endpoint=deployment.external_endpoint,
            created_at=deployment.created_at,
            updated_at=deployment.updated_at,
            deployed_at=deployment.deployed_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/deployments/{deployment_id}", response_model=SuccessResponse)
async def delete_deployment(
    deployment_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a model deployment
    """
    try:
        result = await db.execute(
            select(Deployment).where(Deployment.deployment_id == deployment_id)
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")

        if deployment.status == "terminated":
            raise HTTPException(
                status_code=400,
                detail="Deployment is already terminated"
            )

        # Store deployment name for K8s cleanup
        deployment_name = deployment.deployment_name

        # Update status to terminating
        deployment.status = "terminating"
        await db.commit()

        # Schedule deletion
        background_tasks.add_task(
            deployment_manager.delete_async,
            deployment_id=deployment_id,
            deployment_name=deployment_name
        )

        logger.info(f"Initiated deletion of deployment {deployment_id}")

        return SuccessResponse(
            success=True,
            message=f"Deployment {deployment_id} deletion initiated"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))