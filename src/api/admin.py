"""
Admin API endpoints for infrastructure management
Used for administrative operations and monitoring
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from sqlalchemy import select, update, and_, func

from src.core.database import get_db, AsyncSession
from src.core import kubernetes_client as k8s
from src.models.db_models import (
    Deployment, PodAllocation, GPUNode,
    NodeStatus, DeploymentStatus
)
from src.models.schemas import (
    GPUNodeResponse,
    GPUUtilizationResponse,
    SuccessResponse,
    ErrorResponse,
    DeploymentResponse,
    CreateDeploymentRequest,
    UpdateDeploymentRequest
)
from src.services.deployment_manager import DeploymentManager
from src.services.resource_tracker import ResourceTracker
from src.services.node_manager import NodeManager
from src.services.gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
deployment_manager = DeploymentManager()
resource_tracker = ResourceTracker()
node_manager = NodeManager()
gpu_monitor = GPUMonitor()


@router.get("/deployments", response_model=List[DeploymentResponse])
async def list_deployments(
    status: Optional[DeploymentStatus] = Query(None, description="Filter by deployment status"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db)
):
    """
    List all deployments.

    Args:
        status: Optional filter by deployment status
        tenant_id: Optional filter by tenant ID
        model_id: Optional filter by model ID
        limit: Maximum number of results to return
        offset: Offset for pagination
        db: Database session

    Returns:
        List of deployment information

    Raises:
        HTTPException: If database query fails
    """
    try:
        # Build query
        query = select(Deployment).order_by(Deployment.created_at.desc())

        # Apply filters
        conditions = []
        if status:
            conditions.append(Deployment.status == status)
        if tenant_id:
            conditions.append(Deployment.tenant_id == tenant_id)
        if model_id:
            conditions.append(Deployment.model_id == model_id)

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
                desired_replicas=d.desired_replicas,
                current_replicas=d.current_replicas,
                gpu_per_replica=d.gpu_per_replica,
                total_gpus=d.replicas * d.gpu_per_replica,
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
    Get a specific deployment by ID.

    Args:
        deployment_id: Deployment ID to retrieve
        db: Database session

    Returns:
        Deployment information

    Raises:
        HTTPException: If deployment not found or database query fails
    """
    try:
        # Query deployment
        result = await db.execute(
            select(Deployment)
            .where(Deployment.deployment_id == deployment_id)
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise HTTPException(
                status_code=404,
                detail=f"Deployment {deployment_id} not found"
            )

        # Convert to response model
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
            total_gpus=deployment.replicas * deployment.gpu_per_replica,
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


@router.post("/deployments", response_model=DeploymentResponse, status_code=202)
async def create_deployment(
    request: CreateDeploymentRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new deployment (admin can create for any tenant).

    Args:
        request: Deployment creation request
        background_tasks: Background tasks for async deployment
        db: Database session

    Returns:
        Deployment information with status 202 Accepted

    Raises:
        HTTPException: If creation fails or resources unavailable
    """
    try:
        logger.info(f"Admin creating deployment for model {request.model_id} for tenant {request.tenant_id}")

        # Validate model ownership/access via Model Registry
        from src.services.model_registry_client import ModelRegistryClient
        model_client = ModelRegistryClient()

        try:
            model_info = await model_client.get_model(request.model_id, request.tenant_id)

            # Check if tenant has access to this model (either owner or public model)
            if model_info.get("tenant_id") and model_info["tenant_id"] != request.tenant_id:
                if not model_info.get("is_public", False):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Tenant {request.tenant_id} does not have access to model {request.model_id}"
                    )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to validate model access: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_id} not found or access denied"
            )

        # Generate deployment ID and name
        deployment_id = f"vllm-{request.model_id}-{uuid.uuid4().hex[:8]}"
        deployment_name = f"vllm-{request.model_id.replace('/', '-').replace('_', '-').lower()}"

        # Check resource availability
        from src.services.resource_allocator import ResourceAllocator
        resource_allocator = ResourceAllocator()

        availability = await resource_allocator.check_availability(
            gpu_count=request.gpu_per_replica * request.replicas
        )

        if not availability["can_allocate"]:
            raise HTTPException(
                status_code=503,
                detail=f"Insufficient GPU resources. Required: {request.gpu_per_replica * request.replicas}, Available: {availability['available_gpus']}"
            )

        # Parse memory values
        memory_request_gb = float(request.memory_request.rstrip("Gi"))
        memory_limit_gb = float(request.memory_limit.rstrip("Gi"))

        # Create deployment record
        deployment = Deployment(
            deployment_id=deployment_id,
            deployment_name=deployment_name,
            tenant_id=request.tenant_id,
            model_id=request.model_id,
            model_name=request.model_name,
            model_path=request.model_path,
            replicas=request.replicas,
            gpu_per_replica=request.gpu_per_replica,
            cpu_request=float(request.cpu_request),
            cpu_limit=float(request.cpu_limit),
            memory_request_gb=memory_request_gb,
            memory_limit_gb=memory_limit_gb,
            model_params=request.model_params or {},
            vllm_config=request.vllm_config or {
                "tensor_parallel_size": request.gpu_per_replica,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096
            },
            status="creating",
            desired_replicas=request.replicas,
            current_replicas=0,
            labels=request.labels or {
                "tenant-id": request.tenant_id,
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
                "gpu_count": request.gpu_per_replica,
                "cpu_request": request.cpu_request,
                "cpu_limit": request.cpu_limit,
                "memory_request": request.memory_request,
                "memory_limit": request.memory_limit,
                "env_vars": request.environment_vars or {
                    "MODEL_ID": request.model_id,
                    "MODEL_NAME": request.model_name,
                    "TENSOR_PARALLEL_SIZE": str(request.gpu_per_replica)
                },
                "labels": request.labels or {
                    "tenant-id": request.tenant_id,
                    "model-id": request.model_id,
                    "managed-by": "ai-platform"
                },
                "vllm_config": deployment.vllm_config,
                "model_info": {
                    "model_id": request.model_id,
                    "model_name": request.model_name,
                    "path": request.model_path,
                    "size_gb": 50  # Default size
                }
            }
        )

        logger.info(f"Created deployment {deployment_id} for model {request.model_id}")

        # Return response
        return DeploymentResponse(
            deployment_id=deployment.deployment_id,
            deployment_name=deployment.deployment_name,
            tenant_id=deployment.tenant_id,
            model_id=deployment.model_id,
            model_name=deployment.model_name,
            status="creating",
            replicas=deployment.replicas,
            ready_replicas=0,
            available_replicas=0,
            desired_replicas=deployment.replicas,
            current_replicas=0,
            gpu_per_replica=deployment.gpu_per_replica,
            total_gpus=deployment.gpu_per_replica * deployment.replicas,
            internal_endpoint=None,
            external_endpoint=None,
            created_at=deployment.created_at,
            updated_at=deployment.updated_at,
            deployed_at=None,
            message=f"Deployment is being created. Use GET /admin/deployments/{deployment.deployment_id} to check status"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/deployments/{deployment_id}", response_model=DeploymentResponse)
async def update_deployment(
    deployment_id: str,
    request: UpdateDeploymentRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Update a deployment configuration.

    Args:
        deployment_id: Deployment ID to update
        request: Update request with new configuration
        db: Database session

    Returns:
        Updated deployment information

    Raises:
        HTTPException: If deployment not found or update fails
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


async def _execute_deployment_deletion(deployment_id: str, deployment_name: str):
    """
    Wrapper function to ensure proper async execution of deployment deletion.

    Args:
        deployment_id: Deployment ID to delete
        deployment_name: Kubernetes deployment name
    """
    try:
        await deployment_manager.delete_async(deployment_id, deployment_name)
    except Exception as e:
        logger.error(f"Error in deployment deletion wrapper: {e}", exc_info=True)


@router.delete("/deployments/{deployment_id}", response_model=SuccessResponse)
async def delete_deployment(
    deployment_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a deployment.

    Args:
        deployment_id: Deployment ID to delete
        background_tasks: Background tasks for async deletion
        db: Database session

    Returns:
        Success response

    Raises:
        HTTPException: If deployment not found or deletion fails
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

        if deployment.status == "terminated":
            raise HTTPException(
                status_code=400,
                detail=f"Deployment {deployment_id} is already terminated"
            )

        # Store deployment name before committing (to avoid detached instance issues)
        deployment_name = deployment.deployment_name

        # Update status to terminating and commit before background task
        deployment.status = "terminating"
        deployment.updated_at = datetime.utcnow()
        await db.commit()

        # Schedule background deletion using wrapper function
        # This ensures proper async execution
        background_tasks.add_task(
            _execute_deployment_deletion,
            deployment_id=deployment_id,
            deployment_name=deployment_name
        )

        logger.info(f"Admin initiated deletion of deployment {deployment_id}")

        return SuccessResponse(
            success=True,
            message=f"Deployment {deployment_id} deletion initiated"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/infrastructure/nodes", response_model=List[GPUNodeResponse])
async def list_gpu_nodes(
    status: Optional[NodeStatus] = Query(None, description="Filter by node status"),
    gpu_type: Optional[str] = Query(None, description="Filter by GPU type"),
    db: AsyncSession = Depends(get_db)
):
    """
    List GPU nodes in the cluster

    Returns detailed information about all GPU nodes including their current status and allocation.
    """
    try:
        # Build query
        query = select(GPUNode)

        # Apply filters
        conditions = []
        # Always exclude deleted nodes
        conditions.append(GPUNode.status != "deleted")
        if status:
            conditions.append(GPUNode.status == status)
        if gpu_type:
            conditions.append(GPUNode.gpu_type == gpu_type)

        if conditions:
            query = query.where(and_(*conditions))

        # Execute query
        result = await db.execute(query)
        nodes = result.scalars().all()

        # Get current pod allocations for each node
        node_responses = []
        for node in nodes:
            # Get RUNNING pods on this node (exclude terminated pods)
            result = await db.execute(
                select(PodAllocation.pod_name)
                .where(
                    PodAllocation.node_id == node.node_id,
                    PodAllocation.status == "running"
                )
            )
            pods = [row[0] for row in result.fetchall()]

            node_responses.append(
                GPUNodeResponse(
                    node_id=node.node_id,
                    node_name=node.node_name,
                    node_ip=node.node_ip,
                    gpu_type=node.gpu_type,
                    gpu_count=node.gpu_count,
                    gpu_memory_gb=node.gpu_memory_gb,
                    gpus_allocated=node.gpus_allocated,
                    gpus_available=node.gpu_count - node.gpus_allocated,
                    cpu_cores=node.cpu_cores,
                    memory_gb=node.memory_gb,
                    status=node.status,
                    is_schedulable=node.is_schedulable,
                    pod_count=len(pods),
                    pods=pods if pods else None,
                    created_at=node.created_at,
                    updated_at=node.updated_at,
                    last_heartbeat=node.last_heartbeat
                )
            )

        return node_responses

    except Exception as e:
        logger.error(f"Failed to list GPU nodes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/infrastructure/gpu-utilization", response_model=GPUUtilizationResponse)
async def get_gpu_utilization(
    db: AsyncSession = Depends(get_db)
):
    """
    Get GPU allocation and utilization metrics

    Returns current GPU allocation status across the cluster. Note:
    - Node-level metrics show GPU/memory ALLOCATION (how many resources are reserved)
    - Deployment-level metrics show actual compute UTILIZATION when available
    - GPU allocation != GPU compute utilization (a GPU can be allocated but idle)
    """
    try:
        # Get cluster-wide metrics
        gpu_usage = k8s.get_gpu_usage()

        # Get node-level metrics
        result = await db.execute(select(GPUNode))
        nodes = result.scalars().all()

        node_metrics = []
        total_gpu_util = 0
        total_memory_util = 0
        node_count = 0

        for node in nodes:
            if node.status == "ready":
                # Get RUNNING pod allocations on this node (exclude terminated pods)
                result = await db.execute(
                    select(PodAllocation)
                    .where(
                        PodAllocation.node_id == node.node_id,
                        PodAllocation.status == "running"
                    )
                )
                allocations = result.scalars().all()

                # Calculate utilization
                gpu_util = (node.gpus_allocated / node.gpu_count) * 100 if node.gpu_count > 0 else 0
                memory_util = (node.memory_allocated_gb / node.memory_gb) * 100 if node.memory_gb > 0 else 0

                node_metrics.append({
                    "node_id": node.node_id,
                    "node_name": node.node_name,
                    "gpu_allocation_percent": gpu_util,  # Renamed to clarify it's allocation, not compute usage
                    "memory_allocation_percent": memory_util,  # Renamed to clarify it's allocation
                    "cpu_allocation_percent": (node.cpu_allocated / node.cpu_cores) * 100 if node.cpu_cores > 0 else 0,  # Renamed to clarify
                    "gpus_allocated": node.gpus_allocated,
                    "gpus_total": node.gpu_count,
                    "pod_count": len(allocations)
                })

                total_gpu_util += gpu_util
                total_memory_util += memory_util
                node_count += 1

        # Get deployment-level metrics
        result = await db.execute(
            select(Deployment)
            .where(Deployment.status == DeploymentStatus.RUNNING)
        )
        deployments = result.scalars().all()

        deployment_metrics = []
        for deployment in deployments:
            # Get RUNNING pod allocations for this deployment (exclude terminated pods)
            result = await db.execute(
                select(PodAllocation)
                .where(
                    PodAllocation.deployment_id == deployment.deployment_id,
                    PodAllocation.status == "running"
                )
            )
            allocations = result.scalars().all()

            # Calculate average metrics
            avg_gpu_util = 0
            avg_memory_util = 0
            avg_cpu_util = 0
            total_requests = 0
            total_errors = 0

            for alloc in allocations:
                if alloc.gpu_utilization_percent:
                    avg_gpu_util += alloc.gpu_utilization_percent
                if alloc.memory_utilization_percent:
                    avg_memory_util += alloc.memory_utilization_percent
                if alloc.cpu_utilization_percent:
                    avg_cpu_util += alloc.cpu_utilization_percent
                total_requests += alloc.request_count or 0
                total_errors += alloc.error_count or 0

            if allocations:
                avg_gpu_util /= len(allocations)
                avg_memory_util /= len(allocations)
                avg_cpu_util /= len(allocations)

            deployment_metrics.append({
                "deployment_id": deployment.deployment_id,
                "model_id": deployment.model_id,
                "tenant_id": deployment.tenant_id,
                "replicas": deployment.replicas,
                "gpu_compute_utilization_percent": avg_gpu_util,  # This is actual compute utilization from metrics
                "memory_utilization_percent": avg_memory_util,
                "cpu_utilization_percent": avg_cpu_util,
                "request_count": total_requests,
                "error_count": total_errors,
                "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0
            })

        # Calculate summary
        summary = {
            "cluster_gpu_allocation_percent": (total_gpu_util / node_count) if node_count > 0 else 0,  # Clarify it's allocation
            "cluster_memory_allocation_percent": (total_memory_util / node_count) if node_count > 0 else 0,  # Clarify it's allocation
            "total_gpus": gpu_usage.get("total_gpus", 0),
            "allocated_gpus": gpu_usage.get("allocated_gpus", 0),
            "available_gpus": gpu_usage.get("available_gpus", 0),
            "active_deployments": len(deployments),
            "total_running_pods": sum(n["pod_count"] for n in node_metrics)  # Use the corrected pod counts
        }

        return GPUUtilizationResponse(
            timestamp=datetime.utcnow(),
            cluster_metrics=gpu_usage,
            node_metrics=node_metrics,
            deployment_metrics=deployment_metrics,
            summary=summary
        )

    except Exception as e:
        logger.error(f"Failed to get GPU utilization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/infrastructure/health", response_model=Dict[str, Any])
async def get_infrastructure_health(
    db: AsyncSession = Depends(get_db)
):
    """
    Get infrastructure health status

    Returns overall health status of the orchestration infrastructure.
    """
    try:
        # Check Kubernetes health
        k8s_healthy = k8s.check_k8s_health()

        # Check database health
        from src.core.database import check_db_health
        db_healthy = await check_db_health()

        # Get node health
        result = await db.execute(select(GPUNode))
        nodes = result.scalars().all()

        total_nodes = len(nodes)
        healthy_nodes = sum(1 for n in nodes if n.status == "ready")

        # Get deployment health
        result = await db.execute(select(Deployment))
        deployments = result.scalars().all()

        total_deployments = len(deployments)
        running_deployments = sum(1 for d in deployments if d.status == "running")
        failed_deployments = sum(1 for d in deployments if d.status in ["failed", "error"])

        # Get storage configuration info
        from src.core.storage import get_storage_info
        storage_info = get_storage_info()

        # Calculate overall health
        issues = []
        if not k8s_healthy:
            issues.append("Kubernetes connection unhealthy")
        if not db_healthy:
            issues.append("Database connection unhealthy")
        if healthy_nodes < total_nodes:
            issues.append(f"{total_nodes - healthy_nodes} nodes are not ready")
        if failed_deployments > 0:
            issues.append(f"{failed_deployments} deployments have failed")

        overall_status = "healthy" if not issues else "degraded" if len(issues) < 2 else "unhealthy"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow(),
            "components": {
                "kubernetes": "healthy" if k8s_healthy else "unhealthy",
                "database": "healthy" if db_healthy else "unhealthy",
                "nodes": f"{healthy_nodes}/{total_nodes} healthy",
                "deployments": f"{running_deployments}/{total_deployments} running",
                "storage": storage_info
            },
            "metrics": {
                "total_nodes": total_nodes,
                "healthy_nodes": healthy_nodes,
                "total_deployments": total_deployments,
                "running_deployments": running_deployments,
                "failed_deployments": failed_deployments
            },
            "issues": issues if issues else None
        }

    except Exception as e:
        logger.error(f"Failed to get infrastructure health: {e}", exc_info=True)
        return {
            "status": "error",
            "timestamp": datetime.utcnow(),
            "error": str(e)
        }


@router.get("/gpu/cluster-status")
async def get_gpu_cluster_status():
    """
    Get real-time GPU cluster status with node and allocation details

    Returns:
        GPU cluster status with:
        - Summary by GPU type (total, allocated, available %)
        - Per-node details with individual GPU breakdown
        - Deployment mapping to GPUs
    """
    try:
        status = await gpu_monitor.get_cluster_status()
        return status

    except Exception as e:
        logger.error(f"Failed to get GPU cluster status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve GPU cluster status: {str(e)}"
        )