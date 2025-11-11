"""
GPU Monitoring Service
Provides real-time GPU cluster status with hybrid DB+K8s data
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

from sqlalchemy import select, and_

from src.core.database import get_db_context
from src.core import kubernetes_client as k8s
from src.models.db_models import GPUNode, Deployment, NodeStatus

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Real-time GPU monitoring with hybrid DB+K8s data"""

    async def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get comprehensive GPU cluster status

        Returns:
            Dict with cluster-wide GPU summary and per-node details
        """
        try:
            # Get real-time GPU usage from K8s
            gpu_usage = k8s.get_gpu_usage()
            k8s_node_usage = {node["node_name"]: node for node in gpu_usage.get("nodes", [])}

            # Get node info from DB (for GPU type, memory, etc.)
            async with get_db_context() as db:
                result = await db.execute(
                    select(GPUNode).where(
                        and_(
                            GPUNode.status != NodeStatus.DELETED,
                            GPUNode.gpu_count > 0
                        )
                    )
                )
                db_nodes = result.scalars().all()

                # Get active deployments for GPU breakdown
                result = await db.execute(
                    select(Deployment).where(Deployment.status == "running")
                )
                deployments = result.scalars().all()

            summary_by_type = defaultdict(lambda: {
                "total_gpus": 0,
                "allocated_gpus": 0,
                "available_gpus": 0,
                "available_capacity_percent": 0.0,
                "vram_per_gpu_gb": 0,
                "total_vram_gb": 0.0,
                "allocated_vram_gb": 0.0,
                "available_vram_gb": 0.0,
                "vram_allocation_percent": 0.0  # Renamed to clarify it's allocation
            })

            # Build per-node details
            nodes = []
            for db_node in db_nodes:
                # Get real-time allocation from K8s
                k8s_usage = k8s_node_usage.get(db_node.node_name, {})
                allocated_gpus = k8s_usage.get("allocated", 0)
                total_gpus = k8s_usage.get("total", db_node.gpu_count)

                # Find deployments on this node
                node_deployments = self._get_node_deployments(
                    db_node.node_name,
                    deployments,
                    allocated_gpus
                )

                gpu_memory_gb = db_node.gpu_memory_gb
                total_vram_gb = total_gpus * gpu_memory_gb

                allocated_vram_gb = 0.0
                for dep in node_deployments:
                    dep_vram = dep["_vram_fraction"] * gpu_memory_gb * dep["gpu_count"]
                    allocated_vram_gb += dep_vram
                    dep["vram_allocated_gb"] = round(dep_vram, 2)
                    del dep["_vram_fraction"]

                available_vram_gb = total_vram_gb - allocated_vram_gb
                vram_utilization = (allocated_vram_gb / total_vram_gb * 100) if total_vram_gb > 0 else 0.0

                gpus = self._build_gpu_breakdown(
                    total_gpus,
                    allocated_gpus,
                    node_deployments,
                    gpu_memory_gb
                )

                node_data = {
                    "node_id": db_node.node_id,
                    "node_name": db_node.node_name,
                    "gpu_type": db_node.gpu_type,
                    "total_gpus": total_gpus,
                    "allocated_gpus": allocated_gpus,
                    "available_gpus": total_gpus - allocated_gpus,
                    "available_capacity_percent": round(
                        ((total_gpus - allocated_gpus) / total_gpus * 100) if total_gpus > 0 else 0, 1
                    ),
                    "status": str(db_node.status),
                    "is_schedulable": db_node.is_schedulable,
                    "gpu_memory_gb": db_node.gpu_memory_gb,
                    "vram_total_gb": round(total_vram_gb, 2),
                    "vram_allocated_gb": round(allocated_vram_gb, 2),
                    "vram_available_gb": round(available_vram_gb, 2),
                    "vram_allocation_percent": round(vram_utilization, 1),  # Renamed to clarify it's allocation
                    "deployments": node_deployments,
                    "gpus": gpus
                }

                nodes.append(node_data)

                gpu_type = db_node.gpu_type
                summary_by_type[gpu_type]["total_gpus"] += total_gpus
                summary_by_type[gpu_type]["allocated_gpus"] += allocated_gpus
                summary_by_type[gpu_type]["vram_per_gpu_gb"] = db_node.gpu_memory_gb
                summary_by_type[gpu_type]["total_vram_gb"] += node_data["vram_total_gb"]
                summary_by_type[gpu_type]["allocated_vram_gb"] += node_data["vram_allocated_gb"]

            for gpu_type, stats in summary_by_type.items():
                stats["available_gpus"] = stats["total_gpus"] - stats["allocated_gpus"]
                stats["available_vram_gb"] = round(stats["total_vram_gb"] - stats["allocated_vram_gb"], 2)

                if stats["total_gpus"] > 0:
                    stats["available_capacity_percent"] = round(
                        (stats["available_gpus"] / stats["total_gpus"]) * 100, 1
                    )

                if stats["total_vram_gb"] > 0:
                    stats["vram_allocation_percent"] = round(
                        (stats["allocated_vram_gb"] / stats["total_vram_gb"]) * 100, 1
                    )

                stats["total_vram_gb"] = round(stats["total_vram_gb"], 2)
                stats["allocated_vram_gb"] = round(stats["allocated_vram_gb"], 2)

            cluster_total_vram = sum(node["vram_total_gb"] for node in nodes)
            cluster_allocated_vram = sum(node["vram_allocated_gb"] for node in nodes)
            cluster_available_vram = cluster_total_vram - cluster_allocated_vram
            cluster_vram_allocation = (
                (cluster_allocated_vram / cluster_total_vram * 100) if cluster_total_vram > 0 else 0.0
            )

            return {
                "summary_by_gpu_type": dict(summary_by_type),
                "cluster_total": {
                    "total_gpus": gpu_usage.get("total_gpus", 0),
                    "allocated_gpus": gpu_usage.get("allocated_gpus", 0),
                    "available_gpus": gpu_usage.get("available_gpus", 0),
                    "available_capacity_percent": round(
                        (gpu_usage.get("available_gpus", 0) / gpu_usage.get("total_gpus", 1)) * 100, 1
                    ) if gpu_usage.get("total_gpus", 0) > 0 else 0,
                    "total_vram_gb": round(cluster_total_vram, 2),
                    "allocated_vram_gb": round(cluster_allocated_vram, 2),
                    "available_vram_gb": round(cluster_available_vram, 2),
                    "vram_allocation_percent": round(cluster_vram_allocation, 1)
                },
                "nodes": nodes
            }

        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}", exc_info=True)
            return {
                "summary_by_gpu_type": {},
                "cluster_total": {
                    "total_gpus": 0,
                    "allocated_gpus": 0,
                    "available_gpus": 0,
                    "available_capacity_percent": 0.0
                },
                "nodes": [],
                "error": str(e)
            }

    def _get_node_deployments(
        self,
        node_name: str,
        all_deployments: List[Deployment],
        gpu_count: int
    ) -> List[Dict[str, Any]]:
        """Get deployments running on this node"""
        try:
            # Get pods from K8s for this node
            pods = k8s.get_pods_for_node(node_name)

            node_deployments = []
            for pod in pods:
                # Find matching deployment
                for deployment in all_deployments:
                    if deployment.deployment_name in pod.get("name", ""):
                        pod_gpu_count = 0
                        for container in pod.get("containers", []):
                            gpu_req = container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0)
                            pod_gpu_count += int(gpu_req)

                        # Extract GPU memory fraction from vllm_config
                        from src.adapters.inference_engines import get_engine_adapter

                        memory_fraction = 1.0  # Default: full GPU
                        vllm_config = deployment.vllm_config or {}

                        if vllm_config:
                            engine_type = getattr(deployment, 'engine_type', 'vllm')
                            adapter = get_engine_adapter(engine_type)
                            if adapter:
                                memory_fraction = adapter.get_gpu_memory_fraction(vllm_config)

                        node_deployments.append({
                            "deployment_name": deployment.deployment_name,
                            "model_id": deployment.model_id,
                            "model_name": deployment.model_name,
                            "gpu_count": pod_gpu_count,
                            "memory_fraction": memory_fraction,
                            "status": deployment.status,
                            "_vram_fraction": memory_fraction
                        })
                        break

            return node_deployments

        except Exception as e:
            logger.warning(f"Failed to get node deployments for {node_name}: {e}")
            return []

    def _build_gpu_breakdown(
        self,
        total_gpus: int,
        allocated_gpus: int,
        deployments: List[Dict[str, Any]],
        gpu_memory_gb: float = 48.0
    ) -> List[Dict[str, Any]]:
        """
        Build individual GPU breakdown with per-GPU VRAM allocation
        Since K8s doesn't track individual GPU usage, we simulate logical distribution

        Args:
            total_gpus: Total number of GPUs on node
            allocated_gpus: Number of allocated GPUs
            deployments: List of deployments with vram_allocated_gb
            gpu_memory_gb: Memory per GPU in GB
        """
        gpus = []

        # Create a mutable copy of deployment GPU counts to avoid modifying original
        deployment_gpu_remaining = [dep["gpu_count"] for dep in deployments]

        # Distribute deployments across GPUs
        deployment_idx = 0
        for i in range(total_gpus):
            if i < allocated_gpus and deployment_idx < len(deployments):
                deployment = deployments[deployment_idx]

                # Calculate per-GPU VRAM for this deployment
                # For tensor parallel deployments, VRAM is distributed evenly across GPUs
                dep_gpu_count = deployment["gpu_count"]
                dep_total_vram = deployment.get("vram_allocated_gb", 0)
                vram_per_gpu = dep_total_vram / dep_gpu_count if dep_gpu_count > 0 else 0

                gpu_data = {
                    "gpu_index": i,
                    "status": "allocated",
                    "allocation_status": "allocated",  # GPU is allocated to a deployment
                    "total_vram_gb": round(gpu_memory_gb, 2),
                    "allocated_vram_gb": round(vram_per_gpu, 2),
                    "available_vram_gb": round(gpu_memory_gb - vram_per_gpu, 2),
                    "vram_allocation_percent": round((vram_per_gpu / gpu_memory_gb * 100) if gpu_memory_gb > 0 else 0, 1),
                    "deployment_name": deployment["deployment_name"],
                    "model_name": deployment.get("model_name"),
                    "deployments": [{
                        "deployment_name": deployment["deployment_name"],
                        "model_name": deployment.get("model_name"),
                        "vram_allocated_gb": round(vram_per_gpu, 2)
                    }]
                }

                # Move to next deployment if this one's GPU count is exhausted
                deployment_gpu_remaining[deployment_idx] -= 1
                if deployment_gpu_remaining[deployment_idx] <= 0:
                    deployment_idx += 1

            else:
                # Available GPU
                gpu_data = {
                    "gpu_index": i,
                    "status": "available",
                    "allocation_status": "available",  # GPU is available for allocation
                    "total_vram_gb": round(gpu_memory_gb, 2),
                    "allocated_vram_gb": 0.0,
                    "available_vram_gb": round(gpu_memory_gb, 2),
                    "vram_allocation_percent": 0.0,
                    "deployment_name": None,
                    "model_name": None,
                    "deployments": []
                }

            gpus.append(gpu_data)

        return gpus
