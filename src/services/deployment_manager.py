"""
Deployment Manager Service
Handles vLLM deployment lifecycle management
"""

import uuid
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import select, update

from src.core.database import get_db_context
from src.core import kubernetes_client as k8s
from src.models.db_models import Deployment, PodAllocation, GPUNode
from src.services.resource_allocator import ResourceAllocator
from src.services.model_registry_client import ModelRegistryClient

logger = logging.getLogger(__name__)


class DeploymentManager:
    """Manages vLLM deployments on Kubernetes"""

    def __init__(self):
        self.resource_allocator = ResourceAllocator()
        self.model_registry_client = ModelRegistryClient()

    def _calculate_probe_timeouts(self, model_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate dynamic probe timeouts based on model size and loading profile

        Args:
            model_info: Model metadata from registry

        Returns:
            Probe configuration dict with startup, liveness, and readiness probe settings
        """
        # Default conservative values for when metadata is missing
        default_startup_failure_threshold = 60  # 60 * 10s = 10 minutes

        if not model_info:
            logger.warning("No model info provided, using default probe timeouts")
            return {
                "startup_failure_threshold": default_startup_failure_threshold,
                "startup_period_seconds": 10,
                "startup_initial_delay": 30
            }

        # Extract model size estimation
        # Priority: 1) estimated_size_gb from metadata, 2) parse from size_params, 3) default
        estimated_size_gb = model_info.get("estimated_size_gb")

        if not estimated_size_gb:
            # Try to extract from size_params (e.g., "7B" -> ~14GB for FP16)
            size_params = model_info.get("size_params", "")
            if "B" in size_params:
                try:
                    # Extract number before 'B' (e.g., "7B" -> 7, "1.5B" -> 1.5)
                    params_billions = float(size_params.replace("B", "").strip())
                    # Rough estimation: FP16 is ~2 bytes per param
                    estimated_size_gb = params_billions * 2
                except ValueError:
                    estimated_size_gb = 15  # Default to 15GB if parsing fails
            else:
                estimated_size_gb = 15  # Default

        # Extract loading profile or use defaults
        loading_profile = model_info.get("loading_profile", {})
        download_time_per_gb = loading_profile.get("download_time_per_gb", 15)  # 15s per GB
        loading_time_base = loading_profile.get("loading_time_base", 60)  # 60s base loading
        compile_time = loading_profile.get("compile_time", 45)  # 45s torch compile

        # Calculate total estimated time
        download_time = estimated_size_gb * download_time_per_gb
        total_estimated_time = download_time + loading_time_base + compile_time

        # Add 30% safety margin
        total_with_margin = int(total_estimated_time * 1.3)

        # Calculate failure threshold (period is 10s)
        startup_failure_threshold = max(10, (total_with_margin // 10) + 1)

        logger.info(
            f"Calculated probe timeouts for model (size: {estimated_size_gb}GB): "
            f"download={download_time}s, loading={loading_time_base}s, compile={compile_time}s, "
            f"total_with_margin={total_with_margin}s, failure_threshold={startup_failure_threshold}"
        )

        return {
            "startup_failure_threshold": startup_failure_threshold,
            "startup_period_seconds": 10,
            "startup_initial_delay": 30
        }

    async def _sync_model_status(
        self,
        model_id: str,
        deployment_status: str,
        deployment_info: Dict[str, Any],
        tenant_id: Optional[str] = None
    ):
        """
        Helper method to sync model catalog status with deployment lifecycle

        Status mapping:
        - running → deployed (model is deployed and available)
        - terminated → available (model is available for new deployments)
        - failed → failed (deployment failed)
        - deploying → deploying (deployment in progress)

        This method handles errors gracefully and won't fail the deployment process
        """
        try:
            await self.model_registry_client.update_model_status(
                model_id=model_id,
                status=deployment_status,
                deployment_info=deployment_info,
                tenant_id=tenant_id
            )
        except Exception as e:
            logger.warning(
                f"Failed to sync model status to registry for model {model_id}: {e}. "
                f"Continuing without failing deployment process."
            )

    async def deploy_async(self, deployment_id: str, deployment_config: Dict[str, Any]):
        """
        Deploy vLLM asynchronously

        Args:
            deployment_id: Deployment ID
            deployment_config: Deployment configuration
        """
        try:
            logger.info(f"Starting deployment {deployment_id}")

            # Update Model Registry - deployment starting
            if deployment_config.get("model_info"):
                tenant_id = deployment_config.get("labels", {}).get("tenant-id")
                await self._sync_model_status(
                    model_id=deployment_config["model_id"],
                    deployment_status="deploying",
                    deployment_info={
                        "deployment_id": deployment_id,
                        "status": "creating"
                    },
                    tenant_id=tenant_id
                )

            # Create HuggingFace token secret if provided
            huggingface_token = deployment_config.get("huggingface_token")
            use_hf_secret = False
            if huggingface_token:
                from src.core.secrets import create_huggingface_secret
                try:
                    secret = create_huggingface_secret(
                        deployment_id=deployment_id,
                        huggingface_token=huggingface_token
                    )
                    if secret:
                        use_hf_secret = True
                        logger.info(f"Created HuggingFace token secret for deployment {deployment_id}")
                except Exception as e:
                    logger.error(f"Failed to create HuggingFace token secret: {e}")
                    # Continue deployment without the secret

            # Calculate dynamic probe timeouts based on model size
            probe_config = self._calculate_probe_timeouts(deployment_config.get("model_info"))

            # Create Kubernetes deployment
            k8s_deployment = k8s.create_vllm_deployment(
                deployment_name=deployment_config["deployment_name"],
                model_id=deployment_config["model_id"],
                model_path=deployment_config["model_path"],
                replicas=deployment_config["replicas"],
                gpu_count=deployment_config["gpu_count"],
                cpu_request=deployment_config["cpu_request"],
                cpu_limit=deployment_config["cpu_limit"],
                memory_request=deployment_config["memory_request"],
                memory_limit=deployment_config["memory_limit"],
                env_vars=deployment_config.get("env_vars", {}),
                labels=deployment_config.get("labels", {}),
                probe_config=probe_config,
                deployment_id=deployment_id,
                use_hf_secret=use_hf_secret,
                vllm_config=deployment_config.get("vllm_config", {})
            )

            # Create service
            service = k8s.create_vllm_service(
                service_name=f"{deployment_config['deployment_name']}-service",
                deployment_name=deployment_config["deployment_name"],
                labels=deployment_config.get("labels", {})
            )

            # Wait for pods to be ready
            await self._wait_for_deployment_ready(
                deployment_name=deployment_config["deployment_name"],
                timeout=300
            )

            # Update deployment status in database
            async with get_db_context() as db:
                result = await db.execute(
                    select(Deployment).where(Deployment.deployment_id == deployment_id)
                )
                deployment = result.scalar_one()

                # Set endpoints
                node_port = service.spec.ports[0].node_port if service.spec.ports else None
                deployment.internal_endpoint = f"{service.metadata.name}.{service.metadata.namespace}:8000"

                # Get proper external hostname instead of localhost
                if node_port:
                    external_host = k8s.get_external_hostname()
                    deployment.external_endpoint = f"http://{external_host}:{node_port}"
                else:
                    deployment.external_endpoint = None

                deployment.deployed_at = datetime.utcnow()
                deployment.status = "deploying"  # Set to deploying, will be updated after health checks

                # Update replica tracking
                deployment.desired_replicas = deployment_config.get("replicas", 1)
                deployment.current_replicas = deployment_config.get("replicas", 1)
                deployment.ready_replicas = deployment_config.get("replicas", 1)
                deployment.available_replicas = deployment_config.get("replicas", 1)

                await db.commit()

                # Wait for pods to be ready (15 minute timeout)
                logger.info(f"Waiting for pods to be ready for deployment {deployment_id}")
                pods_ready = await self.wait_for_pods_ready(
                    deployment_id=deployment_id,
                    deployment_name=deployment_config["deployment_name"],
                    timeout=900
                )

                if not pods_ready:
                    deployment.status = "failed"
                    deployment.error_message = "Pods failed to become ready within 15 minutes"

                    # Get pod logs for debugging
                    logs = await self.get_pod_logs_for_deployment(
                        deployment_config["deployment_name"],
                        tail_lines=50
                    )
                    logger.error(f"Deployment {deployment_id} failed. Pod logs:\n{logs}")

                    await db.commit()
                    return  # Exit early, deployment failed

                # Pods are ready - Kubernetes has already verified health via readiness probes
                # No need for additional HTTP health checks during deployment
                logger.info(f"Pods are ready for deployment {deployment_id}, marking as running")

                # All checks passed - deployment is running!
                deployment.status = "running"
                deployment.last_health_check = datetime.utcnow()
                deployment.health_check_failures = 0
                logger.info(f"Deployment {deployment_id} is now running and healthy")

                await db.commit()

                # Create pod allocation records
                await self._create_pod_allocations(
                    deployment_id=deployment_id,
                    deployment_name=deployment_config["deployment_name"],
                    deployment_config=deployment_config,
                    service_name=service.metadata.name,
                    node_port=node_port
                )

            # Update Model Registry - deployment successful (status: running → catalog: deployed)
            if deployment_config.get("model_info"):
                tenant_id = deployment_config.get("labels", {}).get("tenant-id")

                # Get proper external hostname for registry update
                external_endpoint = None
                if node_port:
                    external_host = k8s.get_external_hostname()
                    external_endpoint = f"http://{external_host}:{node_port}"

                await self._sync_model_status(
                    model_id=deployment_config["model_id"],
                    deployment_status="running",  # This will be mapped to "deployed" in catalog
                    deployment_info={
                        "deployment_id": deployment_id,
                        "status": "running",
                        "internal_endpoint": f"{service.metadata.name}.{service.metadata.namespace}:8000",
                        "external_endpoint": external_endpoint,
                        "replicas": deployment_config["replicas"],
                        "gpu_count": deployment_config["gpu_count"]
                    },
                    tenant_id=tenant_id
                )

            logger.info(f"Successfully deployed {deployment_id}")

        except Exception as e:
            logger.error(f"Failed to deploy {deployment_id}: {e}", exc_info=True)

            # Clean up HuggingFace token secret if it was created
            if deployment_config.get("huggingface_token"):
                from src.core.secrets import delete_huggingface_secret
                try:
                    delete_huggingface_secret(deployment_id=deployment_id)
                    logger.info(f"Cleaned up HuggingFace token secret for failed deployment {deployment_id}")
                except Exception as secret_error:
                    logger.warning(f"Failed to clean up HuggingFace token secret: {secret_error}")

            # Update status to failed
            async with get_db_context() as db:
                await db.execute(
                    update(Deployment)
                    .where(Deployment.deployment_id == deployment_id)
                    .values(status="failed")
                )
                await db.commit()

            # Update Model Registry - deployment failed
            if deployment_config.get("model_info"):
                tenant_id = deployment_config.get("labels", {}).get("tenant-id")
                await self._sync_model_status(
                    model_id=deployment_config["model_id"],
                    deployment_status="failed",
                    deployment_info={
                        "deployment_id": deployment_id,
                        "status": "failed",
                        "error": str(e)
                    },
                    tenant_id=tenant_id
                )

    async def delete_async(self, deployment_id: str, deployment_name: str):
        """
        Delete deployment asynchronously

        Args:
            deployment_id: Deployment ID
            deployment_name: Kubernetes deployment name
        """
        try:
            logger.info(f"Starting async deletion of deployment {deployment_id} ({deployment_name})")

            # Get deployment info before deletion (for model registry update)
            model_id = None
            tenant_id = None
            async with get_db_context() as db:
                result = await db.execute(
                    select(Deployment).where(Deployment.deployment_id == deployment_id)
                )
                deployment = result.scalar_one_or_none()
                if deployment:
                    model_id = deployment.model_id
                    tenant_id = deployment.tenant_id
                    logger.info(f"Found deployment {deployment_id} with status: {deployment.status}")
                else:
                    logger.warning(f"Deployment {deployment_id} not found in database during delete_async")

            # Delete Kubernetes resources
            try:
                k8s.delete_deployment(deployment_name)
                logger.info(f"Deleted Kubernetes deployment: {deployment_name}")
            except Exception as e:
                logger.warning(f"Failed to delete Kubernetes deployment {deployment_name}: {e}")

            try:
                k8s.delete_service(f"{deployment_name}-service")
                logger.info(f"Deleted Kubernetes service: {deployment_name}-service")
            except Exception as e:
                logger.warning(f"Failed to delete Kubernetes service {deployment_name}-service: {e}")

            # Delete HuggingFace token secret if it exists
            from src.core.secrets import delete_huggingface_secret
            try:
                if delete_huggingface_secret(deployment_id=deployment_id):
                    logger.info(f"Deleted HuggingFace token secret for deployment {deployment_id}")
            except Exception as e:
                logger.warning(f"Failed to delete HuggingFace token secret: {e}")

            # Update database
            async with get_db_context() as db:
                logger.info(f"Updating database for deployment {deployment_id} to terminated status")

                # Update deployment status and reset replica counts
                result = await db.execute(
                    update(Deployment)
                    .where(Deployment.deployment_id == deployment_id)
                    .values(
                        status="terminated",
                        terminated_at=datetime.utcnow(),
                        ready_replicas=0,
                        available_replicas=0,
                        current_replicas=0,  # Reset current replicas
                        desired_replicas=0   # Reset desired replicas
                    )
                )
                deployment_rows_updated = result.rowcount
                logger.info(f"Updated {deployment_rows_updated} deployment rows to terminated status")

                # Clean up pod allocations - mark as terminated but don't delete records for audit
                result = await db.execute(
                    update(PodAllocation)
                    .where(PodAllocation.deployment_id == deployment_id)
                    .values(
                        status="terminated",
                        terminated_at=datetime.utcnow(),
                        is_ready=False
                    )
                )
                pod_rows_updated = result.rowcount
                logger.info(f"Updated {pod_rows_updated} pod allocation rows to terminated status")

                # Clean up scaling events - mark as terminated but don't delete records for audit
                # TODO: ScalingEvent feature removed, re-implement if autoscaling is needed
                # result = await db.execute(
                #     update(ScalingEvent)
                #     .where(ScalingEvent.deployment_id == deployment_id)
                #     .values(
                #         status="terminated",
                #         completed_at=datetime.utcnow()
                #     )
                # )
                # scaling_rows_updated = result.rowcount
                # logger.info(f"Updated {scaling_rows_updated} scaling event rows to terminated status")

                await db.commit()
                logger.info(f"Successfully committed database updates for deployment {deployment_id} deletion")

            # Release resources
            await self.resource_allocator.release_resources(deployment_id)

            # Update Model Registry - deployment terminated (status: terminated → catalog: available)
            if model_id:
                await self._sync_model_status(
                    model_id=model_id,
                    deployment_status="terminated",  # This will be mapped to "available" in catalog
                    deployment_info={
                        "deployment_id": deployment_id,
                        "status": "terminated",
                        "terminated_at": datetime.utcnow().isoformat()
                    },
                    tenant_id=tenant_id
                )

            logger.info(f"Successfully deleted deployment {deployment_id}")

        except Exception as e:
            logger.error(f"Failed to delete deployment {deployment_id}: {e}", exc_info=True)

            # Try to update deployment status to error
            try:
                async with get_db_context() as db:
                    await db.execute(
                        update(Deployment)
                        .where(Deployment.deployment_id == deployment_id)
                        .values(
                            status="error",
                            error_message=f"Failed to delete: {str(e)}"
                        )
                    )
                    await db.commit()
                    logger.info(f"Updated deployment {deployment_id} status to error after deletion failure")
            except Exception as update_error:
                logger.error(f"Failed to update deployment status to error: {update_error}")

    async def update_deployment(
        self,
        deployment_id: str,
        update_config: Dict[str, Any]
    ):
        """
        Update deployment configuration.

        Args:
            deployment_id: Deployment ID
            update_config: Configuration updates

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            logger.info(f"Updating deployment {deployment_id} with config: {update_config}")

            # Get deployment from database
            async with get_db_context() as db:
                result = await db.execute(
                    select(Deployment).where(Deployment.deployment_id == deployment_id)
                )
                deployment = result.scalar_one_or_none()

                if not deployment:
                    logger.error(f"Deployment {deployment_id} not found")
                    return False

                # Check if deployment is in a state that allows updates
                if deployment.status not in ["running", "error"]:
                    logger.warning(f"Cannot update deployment {deployment_id} in status {deployment.status}")
                    return False

                # Update database fields if provided
                update_values = {"updated_at": datetime.utcnow()}

                if "cpu_request" in update_config:
                    update_values["cpu_request"] = float(update_config["cpu_request"])
                if "cpu_limit" in update_config:
                    update_values["cpu_limit"] = float(update_config["cpu_limit"])
                if "memory_request" in update_config:
                    memory_str = update_config["memory_request"].rstrip("Gi")
                    update_values["memory_request_gb"] = float(memory_str)
                if "memory_limit" in update_config:
                    memory_str = update_config["memory_limit"].rstrip("Gi")
                    update_values["memory_limit_gb"] = float(memory_str)
                if "model_params" in update_config:
                    update_values["model_params"] = update_config["model_params"]
                if "vllm_config" in update_config:
                    update_values["vllm_config"] = update_config["vllm_config"]
                if "labels" in update_config:
                    update_values["labels"] = update_config["labels"]

                # Update deployment status to updating
                update_values["status"] = "updating"

                await db.execute(
                    update(Deployment)
                    .where(Deployment.deployment_id == deployment_id)
                    .values(**update_values)
                )
                await db.commit()

                # Apply updates to Kubernetes deployment if running
                if deployment.deployment_name:
                    # Build Kubernetes patch
                    k8s_patch = {}

                    # Update resource limits if changed
                    if any(key in update_config for key in ["cpu_request", "cpu_limit", "memory_request", "memory_limit"]):
                        resources = {
                            "requests": {},
                            "limits": {}
                        }
                        if "cpu_request" in update_config:
                            resources["requests"]["cpu"] = update_config["cpu_request"]
                        if "memory_request" in update_config:
                            resources["requests"]["memory"] = update_config["memory_request"]
                        if "cpu_limit" in update_config:
                            resources["limits"]["cpu"] = update_config["cpu_limit"]
                        if "memory_limit" in update_config:
                            resources["limits"]["memory"] = update_config["memory_limit"]

                        k8s_patch["resources"] = resources

                    # Update environment variables if provided
                    if "environment_vars" in update_config:
                        k8s_patch["env_vars"] = update_config["environment_vars"]

                    # Update labels if provided
                    if "labels" in update_config:
                        k8s_patch["labels"] = update_config["labels"]

                    # Apply patch to Kubernetes
                    if k8s_patch:
                        success = k8s.update_deployment(deployment.deployment_name, k8s_patch)
                        if not success:
                            logger.error(f"Failed to update Kubernetes deployment {deployment.deployment_name}")
                            # Revert status
                            await db.execute(
                                update(Deployment)
                                .where(Deployment.deployment_id == deployment_id)
                                .values(status="error", error_message="Failed to update Kubernetes deployment")
                            )
                            await db.commit()
                            return False

                # Set status back to running after successful update
                await db.execute(
                    update(Deployment)
                    .where(Deployment.deployment_id == deployment_id)
                    .values(status="running")
                )
                await db.commit()

                # Update Model Registry
                await self._sync_model_status(
                    model_id=deployment.model_id,
                    deployment_status="running",
                    deployment_info={
                        "deployment_id": deployment_id,
                        "status": "running",
                        "updated_at": datetime.utcnow().isoformat()
                    },
                    tenant_id=deployment.tenant_id
                )

                logger.info(f"Successfully updated deployment {deployment_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to update deployment {deployment_id}: {e}", exc_info=True)
            # Try to revert status on error
            try:
                async with get_db_context() as db:
                    await db.execute(
                        update(Deployment)
                        .where(Deployment.deployment_id == deployment_id)
                        .values(status="error", error_message=str(e))
                    )
                    await db.commit()
            except Exception as revert_error:
                logger.error(f"Failed to revert deployment status: {revert_error}")
            return False

    async def scale_deployment(
        self,
        deployment_id: str,
        deployment_name: str,
        target_replicas: int,
        event_id: Optional[str] = None
    ):
        """
        Scale deployment to target replicas

        Args:
            deployment_id: Deployment ID
            deployment_name: Kubernetes deployment name
            target_replicas: Target number of replicas
            event_id: Scaling event ID
        """
        try:
            logger.info(f"Scaling deployment {deployment_id} to {target_replicas} replicas")

            # Scale Kubernetes deployment
            success = k8s.scale_deployment(deployment_name, target_replicas)

            if success:
                # Wait for scaling to complete (with timeout)
                scaled = await self._wait_for_scaling(deployment_name, target_replicas)

                # Get model_id and tenant_id for registry update
                model_id = None
                tenant_id = None
                async with get_db_context() as db:
                    result = await db.execute(
                        select(Deployment).where(Deployment.deployment_id == deployment_id)
                    )
                    deployment = result.scalar_one_or_none()
                    if deployment:
                        model_id = deployment.model_id
                        tenant_id = deployment.tenant_id

                # Update database
                async with get_db_context() as db:
                    # Update deployment
                    values_to_update = {
                        "replicas": target_replicas,
                        "desired_replicas": target_replicas,  # Update desired replicas
                        "status": "running" if scaled else "scaling",
                        "updated_at": datetime.utcnow()
                    }

                    # If scaling completed successfully, update current replicas
                    if scaled:
                        values_to_update["current_replicas"] = target_replicas
                        values_to_update["ready_replicas"] = target_replicas
                        values_to_update["available_replicas"] = target_replicas

                    await db.execute(
                        update(Deployment)
                        .where(Deployment.deployment_id == deployment_id)
                        .values(**values_to_update)
                    )

                    # Update scaling event
                    # TODO: ScalingEvent feature removed, re-implement if autoscaling is needed
                    # if event_id:
                    #     await db.execute(
                    #         update(ScalingEvent)
                    #         .where(ScalingEvent.event_id == event_id)
                    #         .values(
                    #             status="success" if scaled else "failed",
                    #             completed_at=datetime.utcnow() if scaled else None,
                    #             error_message=None if scaled else "Scaling timeout"
                    #         )
                    #     )

                    await db.commit()

                # Update Model Registry if scaling completed successfully
                if scaled and model_id:
                    await self._sync_model_status(
                        model_id=model_id,
                        deployment_status="running",  # Deployment is still running after scaling
                        deployment_info={
                            "deployment_id": deployment_id,
                            "status": "running",
                            "replicas": target_replicas,
                            "scaled_at": datetime.utcnow().isoformat()
                        },
                        tenant_id=tenant_id
                    )

                logger.info(f"Successfully scaled deployment {deployment_id}")

            else:
                # Update scaling event as failed
                # TODO: ScalingEvent feature removed, re-implement if autoscaling is needed
                # if event_id:
                #     async with get_db_context() as db:
                #         await db.execute(
                #             update(ScalingEvent)
                #             .where(ScalingEvent.event_id == event_id)
                #             .values(
                #                 status="failed",
                #                 error_message="Failed to scale Kubernetes deployment"
                #             )
                #         )
                #         await db.commit()

                logger.error(f"Failed to scale Kubernetes deployment {deployment_name}")

        except Exception as e:
            logger.error(f"Failed to scale deployment {deployment_id}: {e}", exc_info=True)

            # TODO: ScalingEvent feature removed, re-implement if autoscaling is needed
            # if event_id:
            #     async with get_db_context() as db:
            #         await db.execute(
            #             update(ScalingEvent)
            #             .where(ScalingEvent.event_id == event_id)
            #             .values(
            #                 status="failed",
            #                 error_message=str(e)
            #             )
            #         )
            #         await db.commit()

    async def restart_deployment(self, deployment_id: str, deployment_name: str):
        """
        Restart deployment by rolling all pods

        Args:
            deployment_id: Deployment ID
            deployment_name: Kubernetes deployment name
        """
        try:
            logger.info(f"Restarting deployment {deployment_id}")

            # Trigger rolling restart by updating deployment
            from kubernetes import client
            apps_v1 = client.AppsV1Api()

            # Add annotation to trigger restart
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=k8s.settings.K8S_NAMESPACE
            )

            if deployment.spec.template.metadata.annotations is None:
                deployment.spec.template.metadata.annotations = {}

            deployment.spec.template.metadata.annotations["kubectl.kubernetes.io/restartedAt"] = datetime.utcnow().isoformat()

            apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=k8s.settings.K8S_NAMESPACE,
                body=deployment
            )

            logger.info(f"Triggered rolling restart for deployment {deployment_id}")

        except Exception as e:
            logger.error(f"Failed to restart deployment {deployment_id}: {e}", exc_info=True)

    async def _get_node_id_from_name(self, db, node_name: str) -> Optional[str]:
        """
        Lookup node_id (UUID) from node_name

        Args:
            db: Database session
            node_name: Kubernetes node name (e.g., 'k3s-gpu-new')

        Returns:
            node_id (UUID string) or None if not found
        """
        try:
            result = await db.execute(
                select(GPUNode.node_id).where(GPUNode.node_name == node_name)
            )
            node_id = result.scalar_one_or_none()

            if not node_id:
                logger.warning(f"Node with name '{node_name}' not found in gpu_nodes table")

            return node_id
        except Exception as e:
            logger.error(f"Error looking up node_id for node_name '{node_name}': {e}", exc_info=True)
            return None

    async def _create_pod_allocations(
        self,
        deployment_id: str,
        deployment_name: str,
        deployment_config: Dict[str, Any],
        service_name: str,
        node_port: Optional[int]
    ):
        """Create pod allocation records for deployment"""
        try:
            # Wait for pods to be created
            await asyncio.sleep(5)

            # Get pods for deployment
            pods = k8s.get_pods_for_deployment(deployment_name)

            async with get_db_context() as db:
                for pod in pods:
                    # Lookup node_id UUID from node_name
                    node_name = pod.get("node_name", "unknown")
                    node_id = await self._get_node_id_from_name(db, node_name)

                    # If node lookup failed, skip this pod allocation to avoid FK constraint violation
                    if not node_id:
                        logger.error(
                            f"Skipping pod allocation for {pod['name']}: node_id not found for node_name '{node_name}'"
                        )
                        continue

                    allocation = PodAllocation(
                        allocation_id=f"alloc-{uuid.uuid4().hex[:8]}",
                        deployment_id=deployment_id,
                        deployment_name=deployment_name,
                        model_id=deployment_config["model_id"],
                        tenant_id=deployment_config["labels"].get("tenant-id", ""),
                        pod_name=pod["name"],
                        pod_uid=pod.get("uid"),
                        namespace=k8s.settings.K8S_NAMESPACE,
                        pod_ip=pod.get("pod_ip"),
                        node_id=node_id,
                        gpu_count=deployment_config["gpu_count"],
                        cpu_request=float(deployment_config["cpu_request"].rstrip("m")) / 1000 if deployment_config["cpu_request"].endswith("m") else float(deployment_config["cpu_request"]),
                        cpu_limit=float(deployment_config["cpu_limit"].rstrip("m")) / 1000 if deployment_config["cpu_limit"].endswith("m") else float(deployment_config["cpu_limit"]),
                        memory_request_gb=float(deployment_config["memory_request"].rstrip("Gi")),
                        memory_limit_gb=float(deployment_config["memory_limit"].rstrip("Gi")),
                        service_name=service_name,
                        service_port=8000,
                        node_port=node_port,
                        status="running" if pod.get("phase") == "Running" else "pending",
                        is_ready=all(c.get("ready", False) for c in pod.get("containers", [])),
                        container_image=k8s.settings.VLLM_IMAGE,
                        environment_vars=deployment_config.get("env_vars", {}),
                        model_config=deployment_config
                    )
                    db.add(allocation)

                await db.commit()

        except Exception as e:
            logger.error(f"Failed to create pod allocations: {e}", exc_info=True)

    async def _wait_for_deployment_ready(self, deployment_name: str, timeout: int = 300) -> bool:
        """
        Wait for deployment to be ready

        Args:
            deployment_name: Kubernetes deployment name
            timeout: Timeout in seconds

        Returns:
            True if deployment is ready, False if timeout
        """
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            status = k8s.get_deployment_status(deployment_name)

            if status:
                ready_replicas = status.get("ready_replicas", 0)
                replicas = status.get("replicas", 1)

                if ready_replicas == replicas and ready_replicas > 0:
                    logger.info(f"Deployment {deployment_name} is ready with {ready_replicas} replicas")
                    return True

            await asyncio.sleep(10)

        logger.warning(f"Deployment {deployment_name} not ready after {timeout} seconds")
        return False

    async def _wait_for_scaling(self, deployment_name: str, target_replicas: int, timeout: int = 300) -> bool:
        """
        Wait for deployment to scale to target replicas

        Args:
            deployment_name: Kubernetes deployment name
            target_replicas: Target number of replicas
            timeout: Timeout in seconds

        Returns:
            True if scaling completed, False if timeout
        """
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            status = k8s.get_deployment_status(deployment_name)

            if status and status.get("ready_replicas") == target_replicas:
                return True

            await asyncio.sleep(10)

        return False

    async def wait_for_pods_ready(
        self,
        deployment_id: str,
        deployment_name: str,
        timeout: int = 900
    ) -> bool:
        """
        Wait for all pods of a deployment to be in Ready state.

        Args:
            deployment_id: Deployment ID
            deployment_name: Kubernetes deployment name
            timeout: Maximum wait time in seconds (default: 15 minutes)

        Returns:
            bool: True if all pods are ready, False if timeout or failure
        """
        start_time = datetime.utcnow()
        logger.info(f"Waiting for pods to be ready for deployment {deployment_id} (timeout: {timeout}s)")

        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            try:
                status = k8s.get_pod_status(deployment_name)

                if status["ready"]:
                    logger.info(f"All pods ready for deployment {deployment_id}")
                    return True

                # Log progress
                logger.info(
                    f"Deployment {deployment_id}: {status['ready_pods']}/{status['total_pods']} pods ready"
                )

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error checking pod status for {deployment_id}: {e}")
                await asyncio.sleep(10)

        logger.error(f"Timeout waiting for pods for deployment {deployment_id}")
        return False

    async def check_vllm_health(self, endpoint: str, timeout: int = 60) -> bool:
        """
        Check vLLM health endpoint.

        Args:
            endpoint: vLLM service endpoint (e.g., "service:8000" or "http://service:8000")
            timeout: Request timeout in seconds

        Returns:
            bool: True if health check passes
        """
        # Add http:// prefix if not present
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"http://{endpoint}"

        health_url = f"{endpoint}/health"
        logger.info(f"Checking vLLM health at {health_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status == 200:
                        logger.info(f"vLLM health check passed for {endpoint}")
                        return True
                    else:
                        logger.error(f"vLLM health check failed: HTTP {response.status}")
                        return False

        except asyncio.TimeoutError:
            logger.error(f"vLLM health check timeout for {endpoint}")
            return False
        except Exception as e:
            logger.error(f"vLLM health check error for {endpoint}: {e}")
            return False

    async def verify_model_loaded(self, endpoint: str, model_path: str) -> bool:
        """
        Verify that the model is loaded in vLLM via /v1/models endpoint.

        Args:
            endpoint: vLLM service endpoint
            model_path: Expected model path/name

        Returns:
            bool: True if model is loaded
        """
        models_url = f"{endpoint}/v1/models"
        logger.info(f"Verifying model loaded at {models_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("data", [])

                        # Check if our model is in the list
                        for model in models:
                            if model.get("id") == model_path or model_path in model.get("id", ""):
                                logger.info(f"Model {model_path} is loaded")
                                return True

                        logger.error(f"Model {model_path} not found in loaded models: {models}")
                        return False
                    else:
                        logger.error(f"Failed to check loaded models: HTTP {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error verifying model loaded: {e}")
            return False

    async def get_pod_logs_for_deployment(
        self,
        deployment_name: str,
        tail_lines: int = 100
    ) -> str:
        """
        Get logs from pods of a deployment for debugging.

        Args:
            deployment_name: Kubernetes deployment name
            tail_lines: Number of log lines to retrieve

        Returns:
            str: Combined logs from all pods
        """
        try:
            # Get pod status to find pod names
            status = k8s.get_pod_status(deployment_name)

            if not status["pod_statuses"]:
                return "No pods found"

            logs = []
            for pod_status in status["pod_statuses"]:
                pod_name = pod_status["name"]
                pod_logs = k8s.get_pod_logs(pod_name, tail_lines=tail_lines)
                logs.append(f"=== Logs from {pod_name} ===\n{pod_logs}\n")

            return "\n".join(logs)

        except Exception as e:
            logger.error(f"Error getting pod logs for {deployment_name}: {e}")
            return f"Error retrieving logs: {str(e)}"

    async def deployment_status_worker(self):
        """
        Background worker that periodically syncs deployment statuses with Kubernetes.
        Check interval is configurable via DEPLOYMENT_STATUS_CHECK_INTERVAL (default: 60 seconds).
        """
        from ..core.config import settings

        # Get check interval from settings (default to 60 seconds)
        check_interval = getattr(settings, 'DEPLOYMENT_STATUS_CHECK_INTERVAL', 60)

        logger.info(f"Deployment status worker started (check interval: {check_interval}s)")

        while True:
            try:
                # Run initial check immediately, then wait for interval
                await self.sync_all_deployment_statuses()
                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error in deployment status worker: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def sync_all_deployment_statuses(self):
        """
        Sync status of all active deployments with Kubernetes.
        Updates database with current pod status and health information.
        """
        logger.info("Starting deployment status sync")

        async with get_db_context() as db:
            # Get all deployments that might have pods, including:
            # - Active ones: running, deploying, creating
            # - Failed ones: to recover if pod actually running
            result = await db.execute(
                select(Deployment).where(
                    Deployment.status.in_(["running", "deploying", "creating", "failed"])
                )
            )
            deployments = result.scalars().all()

            logger.info(f"Syncing status for {len(deployments)} deployments")

            for deployment in deployments:
                try:
                    # Get pod status from Kubernetes
                    pod_status = k8s.get_pod_status(deployment.deployment_name)

                    if pod_status["ready"]:
                        from ..core.config import settings
                        endpoint = deployment.external_endpoint if settings.K8S_MODE == "out-of-cluster" else deployment.internal_endpoint

                        if endpoint:
                            health_ok = await self.check_vllm_health(endpoint, timeout=10)

                            if health_ok:
                                deployment.status = "running"
                                deployment.last_health_check = datetime.utcnow()
                                deployment.health_check_failures = 0
                                deployment.error_message = None
                            else:
                                deployment.health_check_failures += 1
                                if deployment.health_check_failures >= 3:
                                    deployment.status = "failed"
                                    deployment.error_message = "vLLM health check failed 3 times"
                    else:
                        running_pods = pod_status.get("running_pods", 0)
                        failed_pods = pod_status.get("failed_pods", 0)

                        if running_pods == 0 and failed_pods > 0:
                            deployment.status = "failed"
                            deployment.error_message = f"{failed_pods} pods failed, no running pods"
                            logger.error(f"Deployment {deployment.deployment_id} has no running pods")

                    await db.commit()

                except Exception as e:
                    logger.error(f"Error syncing status for deployment {deployment.deployment_id}: {e}")
                    continue

        logger.info("Deployment status sync completed")