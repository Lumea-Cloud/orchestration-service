"""
Kubernetes secrets management for sensitive data
Handles creation and deletion of secrets for deployments
"""

import logging
from typing import Optional, Dict, Any

from kubernetes import client
from kubernetes.client.rest import ApiException

from src.core.config import settings

logger = logging.getLogger(__name__)


def create_huggingface_secret(
    deployment_id: str,
    huggingface_token: str,
    namespace: str = None
) -> Optional[client.V1Secret]:
    """
    Create a Kubernetes secret for HuggingFace token.

    Args:
        deployment_id: Unique deployment identifier
        huggingface_token: HuggingFace API token
        namespace: Kubernetes namespace (defaults to settings.K8S_NAMESPACE)

    Returns:
        Created V1Secret object or None if creation fails

    Raises:
        ApiException: If Kubernetes API call fails
    """
    if not huggingface_token:
        logger.warning(f"No HuggingFace token provided for deployment {deployment_id}")
        return None

    namespace = namespace or settings.K8S_NAMESPACE
    secret_name = f"hf-token-{deployment_id}"

    try:
        # Import v1_client from kubernetes_client module
        from src.core.kubernetes_client import v1_client

        if not v1_client:
            logger.warning("Kubernetes client not initialized, skipping secret creation")
            return None

        # Create secret object
        secret = client.V1Secret(
            api_version="v1",
            kind="Secret",
            metadata=client.V1ObjectMeta(
                name=secret_name,
                namespace=namespace,
                labels={
                    "deployment-id": deployment_id,
                    "managed-by": "orchestration-service",
                    "type": "huggingface-token"
                }
            ),
            type="Opaque",
            string_data={
                "HF_TOKEN": huggingface_token,
                "HUGGING_FACE_HUB_TOKEN": huggingface_token
            }
        )

        # Create the secret in Kubernetes
        created_secret = v1_client.create_namespaced_secret(
            namespace=namespace,
            body=secret
        )

        logger.info(f"Created HuggingFace token secret '{secret_name}' for deployment {deployment_id}")
        return created_secret

    except ApiException as e:
        if e.status == 409:
            logger.info(f"Secret '{secret_name}' already exists for deployment {deployment_id}")
            # Try to read existing secret
            try:
                existing_secret = v1_client.read_namespaced_secret(
                    name=secret_name,
                    namespace=namespace
                )
                return existing_secret
            except ApiException:
                return None
        else:
            logger.error(f"Failed to create secret '{secret_name}': {e}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error creating secret '{secret_name}': {e}")
        raise


def delete_huggingface_secret(
    deployment_id: str,
    namespace: str = None
) -> bool:
    """
    Delete a HuggingFace token secret for a deployment.

    Args:
        deployment_id: Unique deployment identifier
        namespace: Kubernetes namespace (defaults to settings.K8S_NAMESPACE)

    Returns:
        True if deletion successful or secret doesn't exist, False otherwise
    """
    namespace = namespace or settings.K8S_NAMESPACE
    secret_name = f"hf-token-{deployment_id}"

    try:
        # Import v1_client from kubernetes_client module
        from src.core.kubernetes_client import v1_client

        if not v1_client:
            logger.warning("Kubernetes client not initialized, skipping secret deletion")
            return True

        # Delete the secret
        v1_client.delete_namespaced_secret(
            name=secret_name,
            namespace=namespace,
            body=client.V1DeleteOptions()
        )

        logger.info(f"Deleted HuggingFace token secret '{secret_name}' for deployment {deployment_id}")
        return True

    except ApiException as e:
        if e.status == 404:
            logger.info(f"Secret '{secret_name}' not found, nothing to delete")
            return True
        else:
            logger.error(f"Failed to delete secret '{secret_name}': {e}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error deleting secret '{secret_name}': {e}")
        return False


def check_secret_exists(
    deployment_id: str,
    namespace: str = None
) -> bool:
    """
    Check if a HuggingFace token secret exists for a deployment.

    Args:
        deployment_id: Unique deployment identifier
        namespace: Kubernetes namespace (defaults to settings.K8S_NAMESPACE)

    Returns:
        True if secret exists, False otherwise
    """
    namespace = namespace or settings.K8S_NAMESPACE
    secret_name = f"hf-token-{deployment_id}"

    try:
        # Import v1_client from kubernetes_client module
        from src.core.kubernetes_client import v1_client

        if not v1_client:
            logger.warning("Kubernetes client not initialized")
            return False

        v1_client.read_namespaced_secret(
            name=secret_name,
            namespace=namespace
        )
        return True

    except ApiException as e:
        if e.status == 404:
            return False
        else:
            logger.error(f"Error checking secret '{secret_name}': {e}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error checking secret '{secret_name}': {e}")
        return False


def create_secret_env_vars(deployment_id: str) -> list:
    """
    Create environment variable references to a HuggingFace token secret.

    Args:
        deployment_id: Unique deployment identifier

    Returns:
        List of V1EnvVar objects referencing the secret
    """
    secret_name = f"hf-token-{deployment_id}"

    env_vars = [
        client.V1EnvVar(
            name="HF_TOKEN",
            value_from=client.V1EnvVarSource(
                secret_key_ref=client.V1SecretKeySelector(
                    name=secret_name,
                    key="HF_TOKEN",
                    optional=False
                )
            )
        ),
        client.V1EnvVar(
            name="HUGGING_FACE_HUB_TOKEN",
            value_from=client.V1EnvVarSource(
                secret_key_ref=client.V1SecretKeySelector(
                    name=secret_name,
                    key="HUGGING_FACE_HUB_TOKEN",
                    optional=False
                )
            )
        )
    ]

    return env_vars