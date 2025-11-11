"""
Resource Allocator Service
Manages GPU resource allocation and tracking
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from sqlalchemy import select, update, and_

from src.core.database import get_db_context
from src.core import kubernetes_client as k8s
from src.models.db_models import GPUNode, PodAllocation, NodeStatus

logger = logging.getLogger(__name__)


def detect_gpu_type(node_labels: Dict[str, str]) -> str:
    """
    Detect GPU type from node labels

    Args:
        node_labels: Kubernetes node labels

    Returns:
        GPU type string (e.g., "nvidia-l40s", "nvidia-h200")
    """
    # NVIDIA Operator label (preferred)
    if "nvidia.com/gpu.product" in node_labels:
        product = node_labels["nvidia.com/gpu.product"]
        # Tesla-L40S → nvidia-l40s, Tesla-H200 → nvidia-h200
        gpu_type = product.lower().replace("tesla-", "nvidia-").replace("_", "-")
        return gpu_type

    # Manual label (fallback)
    if "gpu-type" in node_labels:
        return node_labels["gpu-type"]

    # Unknown GPU
    return "nvidia-unknown"


class ResourceAllocator:
    """Manages GPU resource allocation across the cluster"""

    async def check_availability(self, gpu_count: int) -> Dict[str, Any]:
        """
        Check if requested GPU resources are available

        Args:
            gpu_count: Number of GPUs requested

        Returns:
            Dictionary with availability information
        """
        try:
            # Get current GPU usage from Kubernetes
            gpu_usage = k8s.get_gpu_usage()

            # Get node information from database
            async with get_db_context() as db:
                result = await db.execute(
                    select(GPUNode)
                    .where(
                        and_(
                            GPUNode.status == NodeStatus.READY,
                            GPUNode.is_schedulable == True
                        )
                    )
                )
                nodes = result.scalars().all()

            # Calculate availability
            available_gpus = gpu_usage.get("available_gpus", 0)
            can_allocate = available_gpus >= gpu_count

            # Find best node for allocation
            best_node = None
            if can_allocate:
                for node in nodes:
                    node_available = node.gpu_count - node.gpus_allocated
                    if node_available >= gpu_count:
                        if not best_node or node_available < (best_node.gpu_count - best_node.gpus_allocated):
                            best_node = node

            return {
                "can_allocate": can_allocate,
                "requested_gpus": gpu_count,
                "available_gpus": available_gpus,
                "total_gpus": gpu_usage.get("total_gpus", 0),
                "best_node": best_node.node_id if best_node else None,
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "available_gpus": node.gpu_count - node.gpus_allocated
                    } for node in nodes
                ]
            }

        except Exception as e:
            logger.error(f"Failed to check resource availability: {e}", exc_info=True)
            return {
                "can_allocate": False,
                "error": str(e)
            }

    async def allocate_resources(
        self,
        deployment_id: str,
        gpu_count: int,
        preferred_node: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Allocate GPU resources for a deployment

        Args:
            deployment_id: Deployment ID
            gpu_count: Number of GPUs to allocate
            preferred_node: Preferred node ID for allocation

        Returns:
            Allocation details or None if allocation failed
        """
        try:
            async with get_db_context() as db:
                # Get available nodes
                query = select(GPUNode).where(
                    and_(
                        GPUNode.status == NodeStatus.READY,
                        GPUNode.is_schedulable == True
                    )
                )

                if preferred_node:
                    query = query.where(GPUNode.node_id == preferred_node)

                result = await db.execute(query)
                nodes = result.scalars().all()

                # Find node with enough resources
                allocated_node = None
                for node in nodes:
                    available = node.gpu_count - node.gpus_allocated
                    if available >= gpu_count:
                        allocated_node = node
                        break

                if not allocated_node:
                    logger.error(f"No node available with {gpu_count} GPUs")
                    return None

                # Update node allocation
                allocated_node.gpus_allocated += gpu_count
                allocated_node.updated_at = datetime.utcnow()

                await db.commit()

                logger.info(f"Allocated {gpu_count} GPUs on node {allocated_node.node_id} for deployment {deployment_id}")

                return {
                    "node_id": allocated_node.node_id,
                    "node_name": allocated_node.node_name,
                    "gpu_count": gpu_count,
                    "gpu_type": allocated_node.gpu_type
                }

        except Exception as e:
            logger.error(f"Failed to allocate resources: {e}", exc_info=True)
            return None

    async def release_resources(self, deployment_id: str):
        """
        Release GPU resources allocated to a deployment

        Args:
            deployment_id: Deployment ID
        """
        try:
            async with get_db_context() as db:
                # Get pod allocations for this deployment
                result = await db.execute(
                    select(PodAllocation)
                    .where(PodAllocation.deployment_id == deployment_id)
                )
                allocations = result.scalars().all()

                # Group by node and calculate total GPUs to release
                nodes_to_update = {}
                for allocation in allocations:
                    if allocation.node_id not in nodes_to_update:
                        nodes_to_update[allocation.node_id] = 0
                    nodes_to_update[allocation.node_id] += allocation.gpu_count

                # Update node allocations
                for node_id, gpu_count in nodes_to_update.items():
                    result = await db.execute(
                        select(GPUNode).where(GPUNode.node_id == node_id)
                    )
                    node = result.scalar_one_or_none()

                    if node:
                        node.gpus_allocated = max(0, node.gpus_allocated - gpu_count)
                        node.updated_at = datetime.utcnow()

                await db.commit()

                logger.info(f"Released resources for deployment {deployment_id}")

        except Exception as e:
            logger.error(f"Failed to release resources: {e}", exc_info=True)

    async def update_node_resources(self):
        """Update node resource information from Kubernetes"""
        try:
            # Get node information from Kubernetes
            k8s_nodes = k8s.get_node_info()

            async with get_db_context() as db:
                for k8s_node in k8s_nodes:
                    # Check if node exists in database
                    result = await db.execute(
                        select(GPUNode).where(GPUNode.node_name == k8s_node["name"])
                    )
                    db_node = result.scalar_one_or_none()

                    if db_node:
                        # Update existing node
                        db_node.node_id = k8s_node["uid"]
                        db_node.gpu_type = detect_gpu_type(k8s_node["labels"])
                        db_node.gpu_count = k8s_node["gpu_count"]
                        db_node.cpu_cores = int(k8s_node["cpu_capacity"])
                        db_node.memory_gb = int(k8s_node["memory_capacity"].rstrip("Ki")) // (1024 * 1024)
                        db_node.status = NodeStatus.READY if k8s_node["is_ready"] else NodeStatus.NOT_READY
                        db_node.is_schedulable = k8s_node["is_schedulable"]
                        db_node.labels = k8s_node["labels"]
                        db_node.last_heartbeat = datetime.utcnow()
                    else:
                        # Create new node
                        node_ip = next(
                            (addr["address"] for addr in k8s_node["addresses"] if addr["type"] == "InternalIP"),
                            "unknown"
                        )

                        gpu_type = detect_gpu_type(k8s_node["labels"])

                        # Determine GPU memory based on type
                        gpu_memory_map = {
                            "nvidia-h200": 141,
                            "nvidia-h100": 80,
                            "nvidia-a100": 80,
                            "nvidia-l40s": 48,
                            "nvidia-l40": 48,
                            "nvidia-a10": 24,
                            "nvidia-unknown": 48  # Conservative default
                        }
                        gpu_memory_gb = gpu_memory_map.get(gpu_type, 48)

                        new_node = GPUNode(
                            node_id=k8s_node["uid"],
                            node_name=k8s_node["name"],
                            node_ip=node_ip,
                            gpu_type=gpu_type,
                            gpu_count=k8s_node["gpu_count"],
                            gpu_memory_gb=gpu_memory_gb,
                            cpu_cores=int(k8s_node["cpu_capacity"]),
                            memory_gb=int(k8s_node["memory_capacity"].rstrip("Ki")) // (1024 * 1024),
                            storage_gb=0,
                            status=NodeStatus.READY if k8s_node["is_ready"] else NodeStatus.NOT_READY,
                            is_schedulable=k8s_node["is_schedulable"],
                            labels=k8s_node["labels"]
                        )
                        db.add(new_node)

                # Mark nodes as deleted if they no longer exist in K8s
                k8s_node_names = {node["name"] for node in k8s_nodes}
                result = await db.execute(
                    select(GPUNode).where(GPUNode.status != NodeStatus.DELETED)
                )
                all_db_nodes = result.scalars().all()

                for db_node in all_db_nodes:
                    if db_node.node_name not in k8s_node_names:
                        db_node.status = NodeStatus.DELETED
                        db_node.is_schedulable = False
                        db_node.updated_at = datetime.utcnow()
                        logger.info(f"Marked node {db_node.node_name} as deleted (not in K8s)")

                # Update GPU allocations based on running pods
                await self._update_gpu_allocations()

                await db.commit()

                logger.info("Updated node resources from Kubernetes")

        except Exception as e:
            logger.error(f"Failed to update node resources: {e}", exc_info=True)

    async def _update_gpu_allocations(self):
        """Update GPU allocation counts based on running pods"""
        try:
            # Get current GPU usage
            gpu_usage = k8s.get_gpu_usage()

            async with get_db_context() as db:
                for node_info in gpu_usage.get("nodes", []):
                    # Update node allocation
                    await db.execute(
                        update(GPUNode)
                        .where(GPUNode.node_name == node_info["node_name"])
                        .values(
                            gpus_allocated=node_info["allocated"],
                            updated_at=datetime.utcnow()
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to update GPU allocations: {e}", exc_info=True)

    async def get_optimal_node(self, gpu_count: int, cpu_request: float, memory_gb: float) -> Optional[str]:
        """
        Find optimal node for resource allocation

        Args:
            gpu_count: Number of GPUs required
            cpu_request: CPU cores required
            memory_gb: Memory in GB required

        Returns:
            Node ID of optimal node or None
        """
        try:
            async with get_db_context() as db:
                # Get available nodes
                result = await db.execute(
                    select(GPUNode)
                    .where(
                        and_(
                            GPUNode.status == NodeStatus.READY,
                            GPUNode.is_schedulable == True,
                            GPUNode.gpu_count - GPUNode.gpus_allocated >= gpu_count,
                            GPUNode.cpu_cores - GPUNode.cpu_allocated >= cpu_request,
                            GPUNode.memory_gb - GPUNode.memory_allocated_gb >= memory_gb
                        )
                    )
                    .order_by(
                        # Prefer nodes with least available resources (bin packing)
                        (GPUNode.gpu_count - GPUNode.gpus_allocated).asc()
                    )
                )
                nodes = result.scalars().all()

                if nodes:
                    return nodes[0].node_id

                return None

        except Exception as e:
            logger.error(f"Failed to find optimal node: {e}", exc_info=True)
            return None