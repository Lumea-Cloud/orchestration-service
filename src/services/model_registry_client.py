"""
Model Registry Client
Handles communication with the Model Registry service
"""

import logging
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime

from src.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistryClient:
    """Client for interacting with Model Registry service"""

    def __init__(self):
        self.base_url = settings.MODEL_REGISTRY_URL
        self.timeout = httpx.Timeout(30.0)

    async def get_model(self, model_id: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get model details from Model Registry

        Args:
            model_id: Model ID
            tenant_id: Optional tenant ID for access validation

        Returns:
            Model details or None if not found
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Use the public API endpoint which is properly exposed
                headers = {}
                if tenant_id:
                    headers["X-Tenant-ID"] = tenant_id

                response = await client.get(
                    f"{self.base_url}/api/v1/models/{model_id}",
                    headers=headers
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    logger.warning(f"Model {model_id} not found in registry")
                    return None
                else:
                    logger.error(f"Error fetching model: {response.status_code} - {response.text}")
                    return None

        except httpx.ConnectError:
            logger.error(f"Failed to connect to Model Registry at {self.base_url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching model {model_id}: {e}")
            return None

    async def update_model_status(
        self,
        model_id: str,
        status: str,
        deployment_info: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Update model catalog status in Model Registry based on deployment lifecycle

        This method calls the internal API endpoint to sync model catalog status
        with deployment status changes.

        Args:
            model_id: Model ID
            status: Deployment status (creating, running, failed, terminated, etc.)
            deployment_info: Additional deployment information
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            True if update was successful
        """
        try:
            # Map deployment status to catalog status
            catalog_status_map = {
                "running": "deployed",
                "terminated": "available",
                "failed": "failed",
                "deploying": "deploying"
            }

            # Use internal API endpoint for status sync
            payload = {
                "deployment_status": catalog_status_map.get(status, status)
            }

            if deployment_info:
                payload["deployment_info"] = deployment_info

            headers = {
                "X-Service-Name": "orchestration-service",
                "X-Internal-Request": "true"
            }

            # Add tenant ID header if provided
            if tenant_id:
                headers["X-Tenant-ID"] = tenant_id

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Call internal endpoint that handles status mapping
                response = await client.patch(
                    f"{self.base_url}/internal/models/{model_id}/status",
                    json=payload,
                    headers=headers
                )

                if response.status_code == 200:
                    logger.info(
                        f"Updated model {model_id} catalog status to '{catalog_status_map.get(status, status)}' "
                        f"based on deployment status '{status}'"
                    )
                    return True
                elif response.status_code == 404:
                    logger.warning(f"Model {model_id} not found in registry")
                    return False
                else:
                    logger.error(
                        f"Failed to update model status: {response.status_code} - {response.text}"
                    )
                    return False

        except httpx.ConnectError:
            logger.error(f"Failed to connect to Model Registry at {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"Error updating model {model_id} status: {e}, will continue without failing")
            # Don't fail the deployment process due to registry update error
            return False

    async def list_models(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List models from Model Registry

        Args:
            tenant_id: Filter by tenant ID
            status: Filter by status
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of models
        """
        try:
            params = {
                "limit": limit,
                "offset": offset
            }

            if tenant_id:
                params["tenant_id"] = tenant_id
            if status:
                params["status"] = status

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/models",
                    params=params
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to list models: {response.status_code} - {response.text}")
                    return []

        except httpx.ConnectError:
            logger.error(f"Failed to connect to Model Registry at {self.base_url}")
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def get_model_by_name(
        self,
        model_name: str,
        tenant_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get model by name from Model Registry

        Args:
            model_name: Model name
            tenant_id: Tenant ID (optional)

        Returns:
            Model details or None if not found
        """
        try:
            params = {"model_name": model_name}
            if tenant_id:
                params["tenant_id"] = tenant_id

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/models",
                    params=params
                )

                if response.status_code == 200:
                    models = response.json()
                    if models:
                        return models[0]  # Return first match
                    return None
                else:
                    logger.error(f"Failed to search model: {response.status_code} - {response.text}")
                    return None

        except httpx.ConnectError:
            logger.error(f"Failed to connect to Model Registry at {self.base_url}")
            return None
        except Exception as e:
            logger.error(f"Error searching model {model_name}: {e}")
            return None

    async def register_deployment(
        self,
        model_id: str,
        deployment_id: str,
        deployment_info: Dict[str, Any]
    ) -> bool:
        """
        Register a deployment with the Model Registry

        Args:
            model_id: Model ID
            deployment_id: Deployment ID
            deployment_info: Deployment information (endpoints, resources, etc.)

        Returns:
            True if registration was successful
        """
        try:
            payload = {
                "deployment_id": deployment_id,
                "deployment_info": deployment_info,
                "created_at": datetime.utcnow().isoformat()
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/deployments",
                    json={
                        "model_id": model_id,
                        **payload
                    }
                )

                if response.status_code in [200, 201]:
                    logger.info(f"Registered deployment {deployment_id} for model {model_id}")
                    return True
                else:
                    logger.error(f"Failed to register deployment: {response.status_code} - {response.text}")
                    return False

        except httpx.ConnectError:
            logger.error(f"Failed to connect to Model Registry at {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"Error registering deployment for model {model_id}: {e}")
            return False