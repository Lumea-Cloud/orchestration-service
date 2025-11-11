"""
Node Manager Service
Manages GPU node operations like draining and uncordoning
"""

import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from kubernetes import client
from kubernetes.client.rest import ApiException
from sqlalchemy import select, update

from src.core.database import get_db_context
from src.core import kubernetes_client as k8s
from src.core.config import settings
from src.models.db_models import GPUNode, PodAllocation, NodeStatus

logger = logging.getLogger(__name__)


class NodeManager:
    """Manages GPU node operations"""

    def __init__(self):
        self.v1_client = None

    def _ensure_client(self):
        """Ensure Kubernetes client is initialized"""
        if not self.v1_client:
            self.v1_client = client.CoreV1Api()

    async def drain_node(
        self,
        node_id: str,
        node_name: str,
        grace_period: int = 300,
        force: bool = False
    ):
        """
        Drain a node by evicting all pods

        Args:
            node_id: Node ID in database
            node_name: Kubernetes node name
            grace_period: Grace period for pod termination in seconds
            force: Force eviction even if pods are not ready to terminate
        """
        try:
            self._ensure_client()
            logger.info(f"Starting node drain for {node_name}")

            # Mark node as unschedulable (cordon)
            self.cordon_node(node_name)

            # Get pods on the node
            pods = await self._get_pods_on_node(node_name)

            if pods:
                logger.info(f"Found {len(pods)} pods on node {node_name}")

                # Evict pods
                for pod in pods:
                    if await self._should_evict_pod(pod, force):
                        await self._evict_pod(
                            pod_name=pod["name"],
                            namespace=pod["namespace"],
                            grace_period=grace_period
                        )

                # Wait for pods to terminate
                await self._wait_for_node_drain(node_name, timeout=grace_period + 60)

            # Update node status in database
            async with get_db_context() as db:
                await db.execute(
                    update(GPUNode)
                    .where(GPUNode.node_id == node_id)
                    .values(
                        status=NodeStatus.DRAINED,
                        is_schedulable=False,
                        updated_at=datetime.utcnow()
                    )
                )
                await db.commit()

            logger.info(f"Successfully drained node {node_name}")

        except Exception as e:
            logger.error(f"Failed to drain node {node_name}: {e}", exc_info=True)

            # Update node status to indicate drain failure
            async with get_db_context() as db:
                await db.execute(
                    update(GPUNode)
                    .where(GPUNode.node_id == node_id)
                    .values(
                        status=NodeStatus.READY,
                        updated_at=datetime.utcnow()
                    )
                )
                await db.commit()

    def cordon_node(self, node_name: str):
        """
        Mark node as unschedulable

        Args:
            node_name: Kubernetes node name
        """
        try:
            self._ensure_client()

            # Get node
            node = self.v1_client.read_node(name=node_name)

            # Mark as unschedulable
            node.spec.unschedulable = True

            # Update node
            self.v1_client.patch_node(name=node_name, body=node)

            logger.info(f"Cordoned node {node_name}")

        except ApiException as e:
            logger.error(f"Failed to cordon node {node_name}: {e}")
            raise

    def uncordon_node(self, node_name: str):
        """
        Mark node as schedulable

        Args:
            node_name: Kubernetes node name
        """
        try:
            self._ensure_client()

            # Get node
            node = self.v1_client.read_node(name=node_name)

            # Mark as schedulable
            node.spec.unschedulable = False

            # Update node
            self.v1_client.patch_node(name=node_name, body=node)

            logger.info(f"Uncordoned node {node_name}")

        except ApiException as e:
            logger.error(f"Failed to uncordon node {node_name}: {e}")
            raise

    async def _get_pods_on_node(self, node_name: str) -> List[Dict[str, Any]]:
        """
        Get all pods running on a node

        Args:
            node_name: Kubernetes node name

        Returns:
            List of pod information
        """
        try:
            self._ensure_client()

            # Get pods on node
            field_selector = f"spec.nodeName={node_name}"
            pods = self.v1_client.list_pod_for_all_namespaces(
                field_selector=field_selector
            )

            pod_list = []
            for pod in pods.items:
                # Skip system pods
                if pod.metadata.namespace in ["kube-system", "kube-public", "kube-node-lease"]:
                    continue

                pod_list.append({
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "uid": pod.metadata.uid,
                    "phase": pod.status.phase,
                    "deletion_timestamp": pod.metadata.deletion_timestamp
                })

            return pod_list

        except Exception as e:
            logger.error(f"Failed to get pods on node {node_name}: {e}")
            return []

    async def _should_evict_pod(self, pod: Dict[str, Any], force: bool) -> bool:
        """
        Determine if a pod should be evicted

        Args:
            pod: Pod information
            force: Force eviction

        Returns:
            True if pod should be evicted
        """
        # Already being deleted
        if pod.get("deletion_timestamp"):
            return False

        # Pod already completed
        if pod.get("phase") in ["Succeeded", "Failed"]:
            return False

        # Check if pod belongs to our deployments
        async with get_db_context() as db:
            result = await db.execute(
                select(PodAllocation)
                .where(
                    PodAllocation.pod_name == pod["name"],
                    PodAllocation.namespace == pod["namespace"]
                )
            )
            allocation = result.scalar_one_or_none()

            if allocation:
                # Our pod - evict if force or not critical
                return force or allocation.status != "running"

        # Not our pod - only evict if force
        return force

    async def _evict_pod(
        self,
        pod_name: str,
        namespace: str,
        grace_period: int = 300
    ):
        """
        Evict a pod from a node

        Args:
            pod_name: Pod name
            namespace: Pod namespace
            grace_period: Grace period for termination
        """
        try:
            self._ensure_client()

            # Create eviction
            eviction = client.V1Eviction(
                metadata=client.V1ObjectMeta(
                    name=pod_name,
                    namespace=namespace
                ),
                delete_options=client.V1DeleteOptions(
                    grace_period_seconds=grace_period
                )
            )

            # Evict pod
            self.v1_client.create_namespaced_pod_eviction(
                name=pod_name,
                namespace=namespace,
                body=eviction
            )

            logger.info(f"Evicted pod {pod_name} from namespace {namespace}")

            # Update pod allocation in database
            async with get_db_context() as db:
                await db.execute(
                    update(PodAllocation)
                    .where(
                        PodAllocation.pod_name == pod_name,
                        PodAllocation.namespace == namespace
                    )
                    .values(
                        status="terminated",
                        terminated_at=datetime.utcnow()
                    )
                )
                await db.commit()

        except ApiException as e:
            if e.status != 404:  # Ignore if pod already gone
                logger.error(f"Failed to evict pod {pod_name}: {e}")

    async def _wait_for_node_drain(self, node_name: str, timeout: int = 600):
        """
        Wait for all pods to be evicted from node

        Args:
            node_name: Kubernetes node name
            timeout: Timeout in seconds
        """
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            pods = await self._get_pods_on_node(node_name)

            # Filter out terminating pods
            active_pods = [
                p for p in pods
                if p.get("phase") not in ["Succeeded", "Failed"] and
                not p.get("deletion_timestamp")
            ]

            if not active_pods:
                logger.info(f"Node {node_name} successfully drained")
                return

            logger.info(f"Waiting for {len(active_pods)} pods to terminate on node {node_name}")
            await asyncio.sleep(10)

        logger.warning(f"Timeout waiting for node {node_name} to drain")

    async def get_node_status(self, node_name: str) -> Dict[str, Any]:
        """
        Get detailed node status

        Args:
            node_name: Kubernetes node name

        Returns:
            Node status information
        """
        try:
            self._ensure_client()

            # Get node from Kubernetes
            node = self.v1_client.read_node(name=node_name)

            # Get node from database
            async with get_db_context() as db:
                result = await db.execute(
                    select(GPUNode)
                    .where(GPUNode.node_name == node_name)
                )
                db_node = result.scalar_one_or_none()

            # Get pods on node
            pods = await self._get_pods_on_node(node_name)

            # Calculate resource usage
            gpu_capacity = int(node.status.capacity.get("nvidia.com/gpu", "0"))
            cpu_capacity = node.status.capacity.get("cpu", "0")
            memory_capacity = node.status.capacity.get("memory", "0")

            return {
                "name": node_name,
                "uid": node.metadata.uid,
                "schedulable": not node.spec.unschedulable,
                "ready": any(
                    c.type == "Ready" and c.status == "True"
                    for c in node.status.conditions
                ),
                "resources": {
                    "gpu": {
                        "capacity": gpu_capacity,
                        "allocated": db_node.gpus_allocated if db_node else 0,
                        "available": gpu_capacity - (db_node.gpus_allocated if db_node else 0)
                    },
                    "cpu": {
                        "capacity": cpu_capacity,
                        "allocated": db_node.cpu_allocated if db_node else 0
                    },
                    "memory": {
                        "capacity": memory_capacity,
                        "allocated_gb": db_node.memory_allocated_gb if db_node else 0
                    }
                },
                "pods": {
                    "total": len(pods),
                    "running": sum(1 for p in pods if p.get("phase") == "Running"),
                    "pending": sum(1 for p in pods if p.get("phase") == "Pending")
                },
                "conditions": [
                    {
                        "type": c.type,
                        "status": c.status,
                        "reason": c.reason,
                        "message": c.message
                    } for c in node.status.conditions
                ],
                "labels": node.metadata.labels,
                "taints": [
                    {
                        "key": t.key,
                        "value": t.value,
                        "effect": t.effect
                    } for t in (node.spec.taints or [])
                ]
            }

        except Exception as e:
            logger.error(f"Failed to get node status: {e}", exc_info=True)
            return {}