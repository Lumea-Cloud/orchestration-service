"""Storage volume management for model deployments"""

from typing import Optional, Tuple
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)


def create_model_volume(deployment_name: str) -> Tuple[client.V1Volume, client.V1VolumeMount]:
    """
    Create appropriate volume and volume mount based on storage mode

    Args:
        deployment_name: Name of the deployment

    Returns:
        Tuple of (V1Volume, V1VolumeMount)

    Raises:
        ValueError: If storage mode is invalid or required configuration is missing
    """
    # Create volume mount (same for all storage modes)
    mount = client.V1VolumeMount(
        name="model-storage",
        mount_path=settings.MODEL_MOUNT_PATH,
        read_only=False
    )

    if settings.STORAGE_MODE == "nfs":
        logger.info(f"Creating NFS volume for {deployment_name}: {settings.NFS_SERVER}:{settings.NFS_PATH}")
        volume = client.V1Volume(
            name="model-storage",
            nfs=client.V1NFSVolumeSource(
                server=settings.NFS_SERVER,
                path=settings.NFS_PATH,
                read_only=False
            )
        )

    elif settings.STORAGE_MODE == "pvc":
        if not settings.PVC_NAME:
            raise ValueError("PVC_NAME must be set when STORAGE_MODE is 'pvc'")

        logger.info(f"Creating PVC volume for {deployment_name}: {settings.PVC_NAME}")
        volume = client.V1Volume(
            name="model-storage",
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                claim_name=settings.PVC_NAME,
                read_only=False
            )
        )

    elif settings.STORAGE_MODE == "csi":
        # For CSI mode, we create a PVC per deployment (or shared)
        pvc_name = f"{deployment_name}-models-pvc"
        logger.info(f"Creating dynamic PVC volume for {deployment_name}: {pvc_name}")

        volume = client.V1Volume(
            name="model-storage",
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                claim_name=pvc_name,
                read_only=False
            )
        )

    else:
        raise ValueError(f"Invalid STORAGE_MODE: {settings.STORAGE_MODE}. Must be 'nfs', 'csi', or 'pvc'")

    logger.debug(f"Created volume configuration for {deployment_name} using {settings.STORAGE_MODE} mode")
    return volume, mount


def create_pvc_for_deployment(deployment_name: str, namespace: str) -> Optional[client.V1PersistentVolumeClaim]:
    """
    Create a PersistentVolumeClaim for CSI mode

    Args:
        deployment_name: Name of the deployment
        namespace: Kubernetes namespace

    Returns:
        V1PersistentVolumeClaim object for CSI mode, None for other modes
    """
    if settings.STORAGE_MODE != "csi":
        logger.debug(f"Skipping PVC creation for {deployment_name} - storage mode is {settings.STORAGE_MODE}")
        return None

    pvc_name = f"{deployment_name}-models-pvc"

    # Create PVC specification
    pvc = client.V1PersistentVolumeClaim(
        api_version="v1",
        kind="PersistentVolumeClaim",
        metadata=client.V1ObjectMeta(
            name=pvc_name,
            namespace=namespace,
            labels={
                "app": deployment_name,
                "managed-by": "orchestration-service",
                "component": "model-storage",
                "storage-mode": "csi"
            }
        ),
        spec=client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteMany"],
            storage_class_name=settings.STORAGE_CLASS,
            resources=client.V1ResourceRequirements(
                requests={"storage": settings.PVC_SIZE}
            )
        )
    )

    logger.info(f"Created PVC spec: {pvc_name} with StorageClass: {settings.STORAGE_CLASS}, Size: {settings.PVC_SIZE}")
    return pvc


def delete_pvc_for_deployment(deployment_name: str, namespace: str) -> bool:
    """
    Delete the PersistentVolumeClaim for a deployment (CSI mode only)

    Args:
        deployment_name: Name of the deployment
        namespace: Kubernetes namespace

    Returns:
        True if PVC was deleted successfully or not in CSI mode, False on error
    """
    if settings.STORAGE_MODE != "csi":
        logger.debug(f"Skipping PVC deletion for {deployment_name} - storage mode is {settings.STORAGE_MODE}")
        return True

    from src.core.kubernetes_client import v1_client

    pvc_name = f"{deployment_name}-models-pvc"

    try:
        # Check if PVC exists
        v1_client.read_namespaced_persistent_volume_claim(
            name=pvc_name,
            namespace=namespace
        )

        # Delete the PVC
        v1_client.delete_namespaced_persistent_volume_claim(
            name=pvc_name,
            namespace=namespace,
            body=client.V1DeleteOptions()
        )
        logger.info(f"Deleted PVC: {pvc_name} in namespace {namespace}")
        return True

    except ApiException as e:
        if e.status == 404:
            logger.warning(f"PVC {pvc_name} not found in namespace {namespace} - skipping deletion")
            return True
        else:
            logger.error(f"Failed to delete PVC {pvc_name}: {e}")
            return False


def validate_storage_configuration() -> bool:
    """
    Validate that the storage configuration is properly set

    Returns:
        True if configuration is valid, raises ValueError otherwise
    """
    mode = settings.STORAGE_MODE.lower()

    if mode not in ["nfs", "csi", "pvc"]:
        raise ValueError(f"Invalid STORAGE_MODE: {mode}. Must be 'nfs', 'csi', or 'pvc'")

    if mode == "nfs":
        if not settings.NFS_SERVER or not settings.NFS_PATH:
            raise ValueError("NFS_SERVER and NFS_PATH must be set when STORAGE_MODE is 'nfs'")
        logger.info(f"Storage configured for NFS mode: {settings.NFS_SERVER}:{settings.NFS_PATH}")

    elif mode == "pvc":
        if not settings.PVC_NAME:
            raise ValueError("PVC_NAME must be set when STORAGE_MODE is 'pvc'")
        logger.info(f"Storage configured for existing PVC mode: {settings.PVC_NAME}")

    elif mode == "csi":
        if not settings.STORAGE_CLASS:
            raise ValueError("STORAGE_CLASS must be set when STORAGE_MODE is 'csi'")
        if not settings.PVC_SIZE:
            raise ValueError("PVC_SIZE must be set when STORAGE_MODE is 'csi'")
        logger.info(f"Storage configured for CSI mode: StorageClass={settings.STORAGE_CLASS}, Size={settings.PVC_SIZE}")

    if not settings.MODEL_MOUNT_PATH:
        raise ValueError("MODEL_MOUNT_PATH must be set")

    return True


def get_storage_info() -> dict:
    """
    Get current storage configuration information

    Returns:
        Dictionary with storage configuration details
    """
    info = {
        "mode": settings.STORAGE_MODE,
        "mount_path": settings.MODEL_MOUNT_PATH,
    }

    if settings.STORAGE_MODE == "nfs":
        info.update({
            "nfs_server": settings.NFS_SERVER,
            "nfs_path": settings.NFS_PATH
        })
    elif settings.STORAGE_MODE == "pvc":
        info.update({
            "pvc_name": settings.PVC_NAME
        })
    elif settings.STORAGE_MODE == "csi":
        info.update({
            "storage_class": settings.STORAGE_CLASS,
            "pvc_size": settings.PVC_SIZE
        })

    return info