"""
Health Monitor Service
Monitors health of deployments and pods
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

from sqlalchemy import select, update, and_

from src.core.database import get_db_context
from src.core import kubernetes_client as k8s
from src.core.config import settings
from src.models.db_models import Deployment, PodAllocation, DeploymentStatus, PodStatus

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitors health of vLLM deployments and pods"""

    def __init__(self):
        self.monitoring = False
        self.check_interval = settings.HEALTH_CHECK_INTERVAL

    async def start_monitoring(self):
        """Start health monitoring loop"""
        self.monitoring = True
        logger.info("Health monitoring started")

        while self.monitoring:
            try:
                await self._check_deployments_health()
                await self._check_pods_health()
                await self._check_stuck_deployments()
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Health monitoring error: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)

    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring = False
        logger.info("Health monitoring stopped")

    async def _check_deployments_health(self):
        """Check health of all active deployments"""
        try:
            async with get_db_context() as db:
                # Get active deployments
                result = await db.execute(
                    select(Deployment)
                    .where(
                        Deployment.status.in_([
                            DeploymentStatus.RUNNING,
                            DeploymentStatus.SCALING,
                            DeploymentStatus.CREATING
                        ])
                    )
                )
                deployments = result.scalars().all()

                for deployment in deployments:
                    # Get Kubernetes deployment status
                    k8s_status = k8s.get_deployment_status(deployment.deployment_name)

                    if k8s_status:
                        # Update deployment metrics
                        deployment.ready_replicas = k8s_status.get("ready_replicas", 0)
                        deployment.available_replicas = k8s_status.get("available_replicas", 0)
                        deployment.updated_at = datetime.utcnow()

                        # Check for issues
                        if deployment.status == DeploymentStatus.RUNNING:
                            if deployment.ready_replicas < deployment.replicas:
                                logger.warning(
                                    f"Deployment {deployment.deployment_id} degraded: "
                                    f"{deployment.ready_replicas}/{deployment.replicas} replicas ready"
                                )

                                # Check if deployment has been degraded for too long
                                if await self._is_deployment_degraded_too_long(deployment):
                                    deployment.status = DeploymentStatus.ERROR
                                    logger.error(
                                        f"Deployment {deployment.deployment_id} marked as error due to prolonged degradation"
                                    )

                        elif deployment.status == DeploymentStatus.CREATING:
                            # Check for creation timeout
                            if deployment.created_at:
                                elapsed = (datetime.utcnow() - deployment.created_at).total_seconds()
                                if elapsed > settings.DEPLOYMENT_TIMEOUT:
                                    deployment.status = DeploymentStatus.FAILED
                                    logger.error(
                                        f"Deployment {deployment.deployment_id} failed: creation timeout"
                                    )

                            # Check if deployment is ready
                            if deployment.ready_replicas == deployment.replicas:
                                deployment.status = DeploymentStatus.RUNNING
                                deployment.deployed_at = datetime.utcnow()
                                logger.info(f"Deployment {deployment.deployment_id} is now running")

                    else:
                        # Deployment not found in Kubernetes
                        if deployment.status != DeploymentStatus.TERMINATING:
                            logger.error(
                                f"Deployment {deployment.deployment_id} not found in Kubernetes"
                            )
                            deployment.status = DeploymentStatus.ERROR

                await db.commit()

        except Exception as e:
            logger.error(f"Failed to check deployments health: {e}", exc_info=True)

    async def _check_pods_health(self):
        """Check health of all running pods"""
        try:
            async with get_db_context() as db:
                # Get active pod allocations
                result = await db.execute(
                    select(PodAllocation)
                    .where(
                        PodAllocation.status.in_([
                            PodStatus.RUNNING,
                            PodStatus.PENDING
                        ])
                    )
                )
                allocations = result.scalars().all()

                for allocation in allocations:
                    # Get pod status from Kubernetes
                    try:
                        from kubernetes import client
                        v1 = client.CoreV1Api()
                        pod = v1.read_namespaced_pod(
                            name=allocation.pod_name,
                            namespace=allocation.namespace
                        )

                        # Update pod status
                        allocation.pod_ip = pod.status.pod_ip
                        allocation.status = self._map_pod_phase(pod.status.phase)
                        allocation.updated_at = datetime.utcnow()

                        # Check container status
                        if pod.status.container_statuses:
                            container_status = pod.status.container_statuses[0]
                            allocation.is_ready = container_status.ready
                            allocation.restart_count = container_status.restart_count

                            # Check for excessive restarts
                            if allocation.restart_count > 5:
                                logger.warning(
                                    f"Pod {allocation.pod_name} has restarted {allocation.restart_count} times"
                                )

                        # Update start time
                        if pod.status.start_time and not allocation.started_at:
                            allocation.started_at = pod.status.start_time

                    except Exception as e:
                        # Pod not found or error accessing
                        if "NotFound" in str(e):
                            allocation.status = PodStatus.FAILED
                            allocation.terminated_at = datetime.utcnow()
                            logger.info(f"Pod {allocation.pod_name} no longer exists")
                        else:
                            logger.error(f"Failed to check pod {allocation.pod_name}: {e}")

                await db.commit()

        except Exception as e:
            logger.error(f"Failed to check pods health: {e}", exc_info=True)

    async def _check_stuck_deployments(self):
        """Check for deployments stuck in transitional states"""
        try:
            async with get_db_context() as db:
                # Check for stuck creating deployments
                cutoff_time = datetime.utcnow() - timedelta(seconds=settings.DEPLOYMENT_TIMEOUT)

                result = await db.execute(
                    select(Deployment)
                    .where(
                        and_(
                            Deployment.status == DeploymentStatus.CREATING,
                            Deployment.created_at < cutoff_time
                        )
                    )
                )
                stuck_deployments = result.scalars().all()

                for deployment in stuck_deployments:
                    logger.error(f"Deployment {deployment.deployment_id} stuck in CREATING state")
                    deployment.status = DeploymentStatus.FAILED

                # Check for stuck scaling deployments
                scaling_cutoff = datetime.utcnow() - timedelta(seconds=300)  # 5 minutes

                result = await db.execute(
                    select(Deployment)
                    .where(
                        and_(
                            Deployment.status == DeploymentStatus.SCALING,
                            Deployment.updated_at < scaling_cutoff
                        )
                    )
                )
                stuck_scaling = result.scalars().all()

                for deployment in stuck_scaling:
                    logger.warning(f"Deployment {deployment.deployment_id} stuck in SCALING state")
                    deployment.status = DeploymentStatus.RUNNING

                await db.commit()

        except Exception as e:
            logger.error(f"Failed to check stuck deployments: {e}", exc_info=True)

    async def _is_deployment_degraded_too_long(self, deployment: Deployment) -> bool:
        """Check if deployment has been degraded for too long"""
        try:
            # Check recent pod allocation updates
            async with get_db_context() as db:
                result = await db.execute(
                    select(PodAllocation)
                    .where(
                        and_(
                            PodAllocation.deployment_id == deployment.deployment_id,
                            PodAllocation.status != PodStatus.RUNNING
                        )
                    )
                )
                unhealthy_pods = result.scalars().all()

                # If any pod has been unhealthy for more than 10 minutes
                cutoff_time = datetime.utcnow() - timedelta(minutes=10)
                for pod in unhealthy_pods:
                    if pod.updated_at < cutoff_time:
                        return True

                return False

        except Exception as e:
            logger.error(f"Failed to check deployment degradation: {e}")
            return False

    def _map_pod_phase(self, phase: str) -> str:
        """Map Kubernetes pod phase to our status"""
        phase_map = {
            "Pending": PodStatus.PENDING,
            "Running": PodStatus.RUNNING,
            "Succeeded": PodStatus.SUCCEEDED,
            "Failed": PodStatus.FAILED,
            "Unknown": PodStatus.UNKNOWN
        }
        return phase_map.get(phase, PodStatus.UNKNOWN)

    async def check_deployment_health(self, deployment_id: str) -> Dict[str, Any]:
        """
        Check health of a specific deployment

        Args:
            deployment_id: Deployment ID

        Returns:
            Health status dictionary
        """
        try:
            async with get_db_context() as db:
                # Get deployment
                result = await db.execute(
                    select(Deployment)
                    .where(Deployment.deployment_id == deployment_id)
                )
                deployment = result.scalar_one_or_none()

                if not deployment:
                    return {"healthy": False, "error": "Deployment not found"}

                # Get Kubernetes status
                k8s_status = k8s.get_deployment_status(deployment.deployment_name)

                # Get pod allocations
                result = await db.execute(
                    select(PodAllocation)
                    .where(PodAllocation.deployment_id == deployment_id)
                )
                allocations = result.scalars().all()

                # Calculate health metrics
                total_pods = len(allocations)
                ready_pods = sum(1 for a in allocations if a.is_ready)
                running_pods = sum(1 for a in allocations if a.status == PodStatus.RUNNING)

                # Determine overall health
                healthy = (
                    deployment.status == DeploymentStatus.RUNNING and
                    ready_pods == deployment.replicas and
                    running_pods == deployment.replicas
                )

                return {
                    "healthy": healthy,
                    "deployment_status": deployment.status,
                    "total_pods": total_pods,
                    "ready_pods": ready_pods,
                    "running_pods": running_pods,
                    "target_replicas": deployment.replicas,
                    "k8s_status": k8s_status
                }

        except Exception as e:
            logger.error(f"Failed to check deployment health: {e}", exc_info=True)
            return {"healthy": False, "error": str(e)}