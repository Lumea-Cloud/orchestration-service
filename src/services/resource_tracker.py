"""
Resource Tracker Service
Tracks and monitors resource usage across the cluster
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import select, update

from src.core.database import get_db_context
from src.core import kubernetes_client as k8s
from src.core.config import settings
from src.models.db_models import GPUNode, PodAllocation
from src.services.resource_allocator import ResourceAllocator

logger = logging.getLogger(__name__)


class ResourceTracker:
    """Tracks resource usage and availability in the cluster"""

    def __init__(self):
        self.tracking = False
        self.track_interval = settings.RESOURCE_TRACK_INTERVAL
        self.resource_allocator = ResourceAllocator()

    async def start_tracking(self):
        """Start resource tracking loop"""
        self.tracking = True
        logger.info("Resource tracking started")

        while self.tracking:
            try:
                await self._update_node_status()
                await self._update_pod_metrics()
                await self._cleanup_terminated_pods()
                await asyncio.sleep(self.track_interval)

            except Exception as e:
                logger.error(f"Resource tracking error: {e}", exc_info=True)
                await asyncio.sleep(self.track_interval)

    async def stop_tracking(self):
        """Stop resource tracking"""
        self.tracking = False
        logger.info("Resource tracking stopped")

    async def _update_node_status(self):
        """Update node status and resource information"""
        try:
            # Get node information from Kubernetes
            k8s_nodes = k8s.get_node_info()

            async with get_db_context() as db:
                for k8s_node in k8s_nodes:
                    # Find or create node record
                    result = await db.execute(
                        select(GPUNode)
                        .where(GPUNode.node_name == k8s_node["name"])
                    )
                    db_node = result.scalar_one_or_none()

                    if db_node:
                        # Update existing node
                        db_node.node_id = k8s_node["uid"]
                        db_node.is_ready = k8s_node["is_ready"]
                        db_node.is_schedulable = k8s_node["is_schedulable"]
                        db_node.status = "ready" if k8s_node["is_ready"] else "not_ready"
                        db_node.last_heartbeat = datetime.utcnow()

                        # Update resource capacity
                        db_node.gpu_count = int(k8s_node["gpu_count"])
                        db_node.cpu_cores = int(k8s_node["cpu_capacity"])

                        # Parse memory (convert from Ki to GB)
                        memory_ki = int(k8s_node["memory_capacity"].rstrip("Ki"))
                        db_node.memory_gb = memory_ki // (1024 * 1024)

                        # Update labels
                        db_node.labels = k8s_node["labels"]

                    else:
                        # Create new node record
                        node_ip = next(
                            (addr["address"] for addr in k8s_node["addresses"]
                             if addr["type"] == "InternalIP"),
                            "unknown"
                        )

                        memory_ki = int(k8s_node["memory_capacity"].rstrip("Ki"))
                        memory_gb = memory_ki // (1024 * 1024)

                        new_node = GPUNode(
                            node_id=k8s_node["uid"],
                            node_name=k8s_node["name"],
                            node_ip=node_ip,
                            gpu_type=settings.GPU_TYPE,
                            gpu_count=int(k8s_node["gpu_count"]),
                            gpu_memory_gb=settings.GPU_MEMORY_GB,
                            cpu_cores=int(k8s_node["cpu_capacity"]),
                            memory_gb=memory_gb,
                            storage_gb=0,  # Not tracked yet
                            status="ready" if k8s_node["is_ready"] else "not_ready",
                            is_schedulable=k8s_node["is_schedulable"],
                            labels=k8s_node["labels"]
                        )
                        db.add(new_node)
                        logger.info(f"Added new GPU node: {new_node.node_name}")

                # Update GPU allocations
                await self._update_gpu_allocations()

                await db.commit()

        except Exception as e:
            logger.error(f"Failed to update node status: {e}", exc_info=True)

    async def _update_gpu_allocations(self):
        """Update GPU allocation counts based on running pods"""
        try:
            # Get current GPU usage from Kubernetes
            gpu_usage = k8s.get_gpu_usage()

            async with get_db_context() as db:
                # Reset all allocations
                await db.execute(
                    update(GPUNode)
                    .values(
                        gpus_allocated=0,
                        cpu_allocated=0.0,
                        memory_allocated_gb=0.0
                    )
                )

                # Update based on actual usage
                for node_usage in gpu_usage.get("nodes", []):
                    result = await db.execute(
                        select(GPUNode)
                        .where(GPUNode.node_name == node_usage["node_name"])
                    )
                    node = result.scalar_one_or_none()

                    if node:
                        node.gpus_allocated = node_usage["allocated"]

                        # Calculate CPU and memory allocation from pods
                        result = await db.execute(
                            select(PodAllocation)
                            .where(
                                PodAllocation.node_id == node.node_id,
                                PodAllocation.status == "running"  # Only count running pods for allocation
                            )
                        )
                        allocations = result.scalars().all()

                        cpu_allocated = sum(a.cpu_request for a in allocations)
                        memory_allocated = sum(a.memory_request_gb for a in allocations)

                        node.cpu_allocated = cpu_allocated
                        node.memory_allocated_gb = memory_allocated

                await db.commit()

        except Exception as e:
            logger.error(f"Failed to update GPU allocations: {e}", exc_info=True)

    async def _update_pod_metrics(self):
        """Update pod resource utilization metrics"""
        try:
            async with get_db_context() as db:
                # Get running pod allocations
                result = await db.execute(
                    select(PodAllocation)
                    .where(PodAllocation.status == "running")
                )
                allocations = result.scalars().all()

                for allocation in allocations:
                    try:
                        # Get pod metrics from Kubernetes metrics API
                        metrics = await self._get_pod_metrics(
                            allocation.pod_name,
                            allocation.namespace
                        )

                        if metrics:
                            # Update utilization metrics
                            allocation.cpu_utilization_percent = metrics.get("cpu_percent", 0)
                            allocation.memory_utilization_percent = metrics.get("memory_percent", 0)
                            allocation.gpu_utilization_percent = metrics.get("gpu_percent", 0)

                            # These would normally come from the model service
                            # For now, we'll use placeholder values
                            allocation.request_count += 0
                            allocation.error_count += 0

                            allocation.updated_at = datetime.utcnow()

                    except Exception as e:
                        logger.debug(f"Failed to get metrics for pod {allocation.pod_name}: {e}")

                await db.commit()

        except Exception as e:
            logger.error(f"Failed to update pod metrics: {e}", exc_info=True)

    async def _get_pod_metrics(self, pod_name: str, namespace: str) -> Optional[Dict[str, float]]:
        """
        Get resource utilization metrics for a pod

        Args:
            pod_name: Pod name
            namespace: Pod namespace

        Returns:
            Dictionary with utilization percentages
        """
        try:
            from kubernetes import client

            # This would normally use the metrics API
            # For now, return simulated metrics
            import random

            # In production, you would use:
            # custom_api = client.CustomObjectsApi()
            # metrics = custom_api.get_namespaced_custom_object(
            #     group="metrics.k8s.io",
            #     version="v1beta1",
            #     namespace=namespace,
            #     plural="pods",
            #     name=pod_name
            # )

            # Simulated metrics for demo
            return {
                "cpu_percent": random.uniform(20, 80),
                "memory_percent": random.uniform(30, 70),
                "gpu_percent": random.uniform(40, 90)
            }

        except Exception as e:
            logger.debug(f"Failed to get pod metrics: {e}")
            return None

    async def _cleanup_terminated_pods(self):
        """Clean up terminated pod allocations"""
        try:
            async with get_db_context() as db:
                # Get pod allocations that might be terminated
                result = await db.execute(
                    select(PodAllocation)
                    .where(
                        PodAllocation.status.in_(["running", "pending"]),
                        PodAllocation.updated_at < datetime.utcnow() - timedelta(minutes=5)
                    )
                )
                allocations = result.scalars().all()

                for allocation in allocations:
                    # Check if pod still exists
                    try:
                        from kubernetes import client
                        v1 = client.CoreV1Api()
                        pod = v1.read_namespaced_pod(
                            name=allocation.pod_name,
                            namespace=allocation.namespace
                        )
                    except Exception:
                        # Pod no longer exists
                        allocation.status = "terminated"
                        allocation.terminated_at = datetime.utcnow()
                        logger.info(f"Marked pod {allocation.pod_name} as terminated")

                await db.commit()

        except Exception as e:
            logger.error(f"Failed to cleanup terminated pods: {e}", exc_info=True)

    async def get_cluster_metrics(self) -> Dict[str, Any]:
        """
        Get current cluster resource metrics

        Returns:
            Dictionary with cluster metrics
        """
        try:
            # Get GPU usage
            gpu_usage = k8s.get_gpu_usage()

            async with get_db_context() as db:
                # Get node metrics
                result = await db.execute(select(GPUNode))
                nodes = result.scalars().all()

                total_cpu = sum(n.cpu_cores for n in nodes)
                total_memory = sum(n.memory_gb for n in nodes)
                allocated_cpu = sum(n.cpu_allocated for n in nodes)
                allocated_memory = sum(n.memory_allocated_gb for n in nodes)

                # Get deployment metrics
                result = await db.execute(
                    select(PodAllocation)
                    .where(PodAllocation.status == "running")
                )
                running_pods = result.scalars().all()

                return {
                    "timestamp": datetime.utcnow(),
                    "gpu": {
                        "total": gpu_usage.get("total_gpus", 0),
                        "allocated": gpu_usage.get("allocated_gpus", 0),
                        "available": gpu_usage.get("available_gpus", 0),
                        "utilization_percent": (
                            gpu_usage.get("allocated_gpus", 0) / gpu_usage.get("total_gpus", 1) * 100
                        )
                    },
                    "cpu": {
                        "total_cores": total_cpu,
                        "allocated_cores": allocated_cpu,
                        "available_cores": total_cpu - allocated_cpu,
                        "utilization_percent": (allocated_cpu / total_cpu * 100) if total_cpu > 0 else 0
                    },
                    "memory": {
                        "total_gb": total_memory,
                        "allocated_gb": allocated_memory,
                        "available_gb": total_memory - allocated_memory,
                        "utilization_percent": (allocated_memory / total_memory * 100) if total_memory > 0 else 0
                    },
                    "nodes": {
                        "total": len(nodes),
                        "ready": sum(1 for n in nodes if n.status == "ready"),
                        "schedulable": sum(1 for n in nodes if n.is_schedulable)
                    },
                    "pods": {
                        "running": len(running_pods),
                        "total_requests": sum(p.request_count for p in running_pods),
                        "total_errors": sum(p.error_count for p in running_pods)
                    }
                }

        except Exception as e:
            logger.error(f"Failed to get cluster metrics: {e}", exc_info=True)
            return {}


from datetime import timedelta