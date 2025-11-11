"""
Internal API endpoints for deployment management
Used by other services in the AI platform
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from sqlalchemy import select, update, and_

from src.core.database import get_db, AsyncSession
from src.core import kubernetes_client as k8s
from src.core.config import settings
from src.models.db_models import Deployment, PodAllocation, GPUNode
from src.models.schemas import (
    CreateDeploymentRequest,
    DeploymentResponse,
    DeploymentStatusResponse,
    ResourceAvailabilityResponse,
    ErrorResponse,
    SuccessResponse
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


@router.post("/deployments/create", response_model=DeploymentResponse)
async def create_deployment(
    request: CreateDeploymentRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new vLLM deployment

    This endpoint creates a new vLLM deployment on Kubernetes with GPU allocation.
    The deployment process is asynchronous and runs in the background.
    """
    try:
        # Fetch model details from Model Registry
        model_info = await model_registry_client.get_model(request.model_id)
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_id} not found in registry"
            )

        # Use model info to populate deployment details
        model_name = model_info.get("model_name", request.model_name)

        # Determine the model path (HuggingFace ID)
        # Priority: huggingface_id > source_url > path field > request.model_path
        model_path = None

        # Check for huggingface_id field (top level or in metadata)
        if model_info.get("huggingface_id"):
            model_path = model_info["huggingface_id"]
        elif model_info.get("metadata", {}).get("huggingface_id"):
            model_path = model_info["metadata"]["huggingface_id"]
        # Check for source_url at top level (for Hugging Face models)
        elif model_info.get("source_url"):
            source_url = model_info["source_url"]
            if "huggingface.co/" in source_url:
                # Extract the model ID from URL like https://huggingface.co/google/gemma-2-27b-it
                model_path = source_url.split("huggingface.co/")[-1].strip("/")
            else:
                model_path = source_url
        # Check in metadata for source_url
        elif model_info.get("metadata", {}).get("source_url"):
            source_url = model_info["metadata"]["source_url"]
            if "huggingface.co/" in source_url:
                # Extract the model ID from URL
                model_path = source_url.split("huggingface.co/")[-1].strip("/")
            else:
                model_path = source_url
        # Check if there's a direct path field (should be full HF ID, not just name)
        elif model_info.get("path"):
            model_path = model_info["path"]
        elif model_info.get("metadata", {}).get("path"):
            model_path = model_info["metadata"]["path"]
        # Fall back to request model_path if provided
        elif request.model_path:
            model_path = request.model_path

        # If still no path, log warning and raise error
        if not model_path:
            logger.error(
                f"Could not determine HuggingFace ID for model {request.model_id}. "
                f"Model Registry should provide huggingface_id or source_url. "
                f"Available fields: {list(model_info.keys())}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Model {request.model_id} missing huggingface_id or source_url in registry"
            )

        model_size = model_info.get("size_gb", 0)

        # Determine GPU requirements based on model size (27B model needs 2 GPUs)
        if "27b" in model_name.lower() or model_size > 50:
            gpu_per_replica = 2
            memory_request = "64Gi"
            memory_limit = "96Gi"
        else:
            gpu_per_replica = request.gpu_per_replica
            memory_request = request.memory_request
            memory_limit = request.memory_limit

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

        # Create deployment record in database
        deployment = Deployment(
            deployment_id=deployment_id,
            deployment_name=deployment_name,
            tenant_id=request.tenant_id,
            model_id=request.model_id,
            model_name=model_name,
            replicas=request.replicas,
            gpu_per_replica=gpu_per_replica,
            cpu_request=float(request.cpu_request.rstrip("m")) / 1000 if request.cpu_request.endswith("m") else float(request.cpu_request),
            cpu_limit=float(request.cpu_limit.rstrip("m")) / 1000 if request.cpu_limit.endswith("m") else float(request.cpu_limit),
            memory_request_gb=float(memory_request.rstrip("Gi")),
            memory_limit_gb=float(memory_limit.rstrip("Gi")),
            model_path=model_path,
            model_params=request.model_params,
            vllm_config=request.vllm_config,
            status="creating",
            labels=request.labels,
            annotations=request.annotations
        )

        db.add(deployment)
        await db.commit()
        await db.refresh(deployment)

        # Schedule background deployment with model registry update
        background_tasks.add_task(
            deployment_manager.deploy_async,
            deployment_id=deployment_id,
            deployment_config={
                "deployment_name": deployment_name,
                "model_id": request.model_id,
                "model_path": model_path,
                "replicas": request.replicas,
                "gpu_count": gpu_per_replica,
                "cpu_request": request.cpu_request,
                "cpu_limit": request.cpu_limit,
                "memory_request": memory_request,
                "memory_limit": memory_limit,
                "env_vars": request.environment_vars,
                "labels": {
                    **request.labels,
                    "tenant-id": request.tenant_id,
                    "model-id": request.model_id
                },
                "model_info": model_info  # Pass model info for status updates
            }
        )

        logger.info(f"Created deployment {deployment_id} for tenant {request.tenant_id}")

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
            desired_replicas=getattr(deployment, 'desired_replicas', deployment.replicas),
            current_replicas=getattr(deployment, 'current_replicas', deployment.ready_replicas),
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
        logger.error(f"Failed to create deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/deployments/{deployment_id}", response_model=SuccessResponse)
async def delete_deployment(
    deployment_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a vLLM deployment

    This endpoint deletes a vLLM deployment and releases allocated resources.
    The deletion process is asynchronous.
    """
    try:
        # Get deployment from database
        result = await db.execute(
            select(Deployment).where(Deployment.deployment_id == deployment_id)
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")

        if deployment.status == "terminated":
            return SuccessResponse(
                success=True,
                message="Deployment already terminated"
            )

        # Update status to terminating
        deployment.status = "terminating"
        await db.commit()

        # Schedule background deletion
        background_tasks.add_task(
            deployment_manager.delete_async,
            deployment_id=deployment_id,
            deployment_name=deployment.deployment_name
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


@router.get("/deployments/{deployment_id}/status", response_model=DeploymentStatusResponse)
async def get_deployment_status(
    deployment_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get deployment status

    Returns the current status of a deployment including pod information.
    """
    try:
        # Get deployment from database
        result = await db.execute(
            select(Deployment).where(Deployment.deployment_id == deployment_id)
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")

        # Get Kubernetes deployment status
        k8s_status = k8s.get_deployment_status(deployment.deployment_name)

        # Get pod information
        pods = k8s.get_pods_for_deployment(deployment.deployment_name)

        # Get pod allocations from database
        result = await db.execute(
            select(PodAllocation).where(PodAllocation.deployment_id == deployment_id)
        )
        allocations = result.scalars().all()

        # Combine information
        pod_info = []
        for pod in pods:
            allocation = next((a for a in allocations if a.pod_name == pod["name"]), None)
            pod_info.append({
                "name": pod["name"],
                "node": pod.get("node_name"),
                "status": pod.get("phase"),
                "ready": all(c.get("ready", False) for c in pod.get("containers", [])),
                "gpu_count": allocation.gpu_count if allocation else 0,
                "pod_ip": pod.get("pod_ip"),
                "restart_count": sum(c.get("restart_count", 0) for c in pod.get("containers", []))
            })

        return DeploymentStatusResponse(
            deployment_id=deployment.deployment_id,
            status=deployment.status,
            replicas=deployment.replicas,
            ready_replicas=k8s_status.get("ready_replicas", 0) if k8s_status else 0,
            available_replicas=k8s_status.get("available_replicas", 0) if k8s_status else 0,
            desired_replicas=getattr(deployment, 'desired_replicas', deployment.replicas),
            current_replicas=k8s_status.get("ready_replicas", 0) if k8s_status else 0,
            pods=pod_info,
            conditions=k8s_status.get("conditions", []) if k8s_status else [],
            message=None,
            last_updated=deployment.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resources/available", response_model=ResourceAvailabilityResponse)
async def get_available_resources(
    db: AsyncSession = Depends(get_db)
):
    """
    Get available GPU resources

    Returns information about available GPU resources in the cluster.
    """
    try:
        # Get GPU usage from Kubernetes
        gpu_usage = k8s.get_gpu_usage()

        # Get node information from database
        result = await db.execute(select(GPUNode))
        nodes = result.scalars().all()

        # Prepare node information
        node_info = []
        ready_nodes = 0
        max_deployable_gpus = 0

        for node in nodes:
            available = node.gpu_count - node.gpus_allocated
            if node.status == "ready" and node.is_schedulable:
                ready_nodes += 1
                max_deployable_gpus = max(max_deployable_gpus, available)

            node_info.append({
                "node_id": node.node_id,
                "node_name": node.node_name,
                "gpu_type": node.gpu_type,
                "total_gpus": node.gpu_count,
                "allocated_gpus": node.gpus_allocated,
                "available_gpus": available,
                "status": node.status,
                "is_schedulable": node.is_schedulable
            })

        return ResourceAvailabilityResponse(
            total_gpus=gpu_usage.get("total_gpus", 0),
            available_gpus=gpu_usage.get("available_gpus", 0),
            allocated_gpus=gpu_usage.get("allocated_gpus", 0),
            total_nodes=len(nodes),
            ready_nodes=ready_nodes,
            nodes=node_info,
            can_schedule=gpu_usage.get("available_gpus", 0) > 0,
            max_deployable_gpus=max_deployable_gpus
        )

    except Exception as e:
        logger.error(f"Failed to get available resources: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments", response_model=List[DeploymentResponse])
async def list_deployments(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db)
):
    """
    List deployments

    Returns a list of deployments with optional filtering.
    """
    try:
        # Build query
        query = select(Deployment)

        # Apply filters
        conditions = []
        if tenant_id:
            conditions.append(Deployment.tenant_id == tenant_id)
        if model_id:
            conditions.append(Deployment.model_id == model_id)
        if status:
            conditions.append(Deployment.status == status)

        if conditions:
            query = query.where(and_(*conditions))

        # Apply pagination
        query = query.limit(limit).offset(offset)

        # Execute query
        result = await db.execute(query)
        deployments = result.scalars().all()

        # Convert to response models
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
                desired_replicas=getattr(d, 'desired_replicas', d.replicas),
                current_replicas=getattr(d, 'current_replicas', d.ready_replicas),
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


@router.post("/deployments/{deployment_id}/restart", response_model=SuccessResponse)
async def restart_deployment(
    deployment_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Restart a deployment

    This endpoint restarts all pods in a deployment.
    """
    try:
        # Get deployment from database
        result = await db.execute(
            select(Deployment).where(Deployment.deployment_id == deployment_id)
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise HTTPException(status_code=404, detail="Deployment not found")

        if deployment.status not in ["running", "error"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot restart deployment in status: {deployment.status}"
            )

        # Schedule restart
        background_tasks.add_task(
            deployment_manager.restart_deployment,
            deployment_id=deployment_id,
            deployment_name=deployment.deployment_name
        )

        logger.info(f"Initiated restart of deployment {deployment_id}")

        return SuccessResponse(
            success=True,
            message=f"Deployment {deployment_id} restart initiated"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments/status")
async def get_deployment_status_for_model(
    model_id: str = Query(..., description="Model catalog ID"),
    tenant_id: str = Query(..., description="Tenant ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get deployment status for a specific model.

    This endpoint is used by Model Registry Service to enrich model responses
    with deployment status information.

    Args:
        model_id: The model catalog ID (UUID)
        tenant_id: The tenant ID to filter deployments

    Returns:
        Dict with deployment status and deployment_id if model is deployed
    """
    try:
        # Validate UUID format
        try:
            uuid.UUID(model_id)
            uuid.UUID(tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid UUID format: {e}")

        # Find active deployment for this model and tenant
        # Note: Database stores these as VARCHAR but model defines them as UUID
        # We need to compare as strings
        result = await db.execute(
            select(Deployment).where(
                and_(
                    Deployment.model_id == model_id,
                    Deployment.tenant_id == tenant_id,
                    Deployment.status.in_(["pending", "creating", "running", "scaling", "updating"])
                )
            ).order_by(Deployment.created_at.desc())
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            # No active deployment found
            return {"status": "not_deployed", "deployment_id": None}

        # Map deployment status to a simpler status for users
        status_mapping = {
            "pending": "deploying",
            "creating": "deploying",
            "running": "running",
            "scaling": "running",
            "updating": "running"
        }

        return {
            "status": status_mapping.get(deployment.status, deployment.status),
            "deployment_id": deployment.deployment_id,
            "deployment_name": deployment.deployment_name,
            "replicas": deployment.replicas,
            "ready_replicas": deployment.ready_replicas,
            "endpoint": deployment.internal_endpoint or deployment.external_endpoint,
            "gpu_count": deployment.gpu_per_replica * deployment.replicas if deployment.gpu_per_replica else None
        }

    except Exception as e:
        logger.error(f"Failed to get deployment status for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/endpoint")
async def get_model_endpoint(
    model_name: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get vLLM endpoint URL for a model

    This endpoint is used by the Gateway Service to route sync inference requests
    directly to vLLM pods without going through the Worker Service.

    Args:
        model_name: The model name (can be model_id or model_name from registry)

    Returns:
        endpoint_url: The vLLM service endpoint URL
        model_id: The actual model ID
        deployment_id: The deployment ID
    """
    try:
        # Try to find deployment by model_id first, then by model_name
        # Check if model_name is a valid UUID, if so, search by both model_id and model_name
        # Otherwise, only search by model_name
        try:
            model_uuid = uuid.UUID(model_name)
            # Valid UUID - search by both model_id and model_name
            result = await db.execute(
                select(Deployment).where(
                    and_(
                        (Deployment.model_id == str(model_uuid)) | (Deployment.model_name == model_name),
                        Deployment.status.in_(["running", "ready"])
                    )
                ).order_by(Deployment.created_at.desc())
            )
        except (ValueError, AttributeError):
            # Not a valid UUID - only search by model_name
            result = await db.execute(
                select(Deployment).where(
                    and_(
                        Deployment.model_name == model_name,
                        Deployment.status.in_(["running", "ready"])
                    )
                ).order_by(Deployment.created_at.desc())
            )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise HTTPException(
                status_code=404,
                detail=f"No active deployment found for model: {model_name}"
            )

        # Check if deployment has an endpoint
        if not deployment.internal_endpoint and not deployment.external_endpoint:
            raise HTTPException(
                status_code=503,
                detail=f"Deployment {deployment.deployment_id} does not have an endpoint yet. Status: {deployment.status}"
            )

        # Prefer external endpoint for development (when Gateway runs outside cluster)
        # In production, this should be internal_endpoint
        endpoint_url = deployment.external_endpoint or deployment.internal_endpoint

        # Fetch huggingface_id from Model Registry
        huggingface_id = None
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                registry_response = await client.get(
                    f"{settings.MODEL_REGISTRY_URL}/internal/models/{deployment.model_id}",
                    headers={"X-Tenant-ID": str(deployment.tenant_id)},  # Use deployment's tenant_id
                    timeout=5.0
                )
                if registry_response.status_code == 200:
                    model_data = registry_response.json()
                    huggingface_id = model_data.get("huggingface_id")
                else:
                    logger.warning(f"Model Registry returned status {registry_response.status_code}: {registry_response.text}")
        except Exception as e:
            logger.warning(f"Failed to fetch huggingface_id from Model Registry: {e}")

        # Fall back to deployment.model_path if huggingface_id is not available
        if not huggingface_id and deployment.model_path:
            huggingface_id = deployment.model_path

        # Fetch model visibility (is_public) from Model Registry
        is_public = False
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                registry_response = await client.get(
                    f"{settings.MODEL_REGISTRY_URL}/internal/models/{deployment.model_id}",
                    headers={"X-Tenant-ID": str(deployment.tenant_id)},
                    timeout=5.0
                )
                if registry_response.status_code == 200:
                    model_data = registry_response.json()
                    is_public = model_data.get("is_public", False)
        except Exception as e:
            logger.warning(f"Failed to fetch model visibility from Model Registry: {e}")

        logger.info(f"Resolved model {model_name} to endpoint {endpoint_url} (deployment: {deployment.deployment_id}, huggingface_id: {huggingface_id}, tenant_id: {deployment.tenant_id}, is_public: {is_public})")

        return {
            "endpoint_url": endpoint_url,
            "model_id": deployment.model_id,
            "model_name": deployment.model_name,
            "huggingface_id": huggingface_id,
            "deployment_id": deployment.deployment_id,
            "tenant_id": str(deployment.tenant_id),  # Include tenant_id for validation
            "is_public": is_public,  # Include model visibility
            "status": deployment.status
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))