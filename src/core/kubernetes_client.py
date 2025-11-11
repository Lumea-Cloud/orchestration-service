"""
Kubernetes client configuration and utilities
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from kubernetes import client, config
from kubernetes.client import Configuration
from kubernetes.client.rest import ApiException

from src.core.config import settings
from src.core.storage import (
    create_model_volume,
    create_pvc_for_deployment,
    delete_pvc_for_deployment,
    validate_storage_configuration
)

logger = logging.getLogger(__name__)

# Global Kubernetes API clients
v1_client: Optional[client.CoreV1Api] = None
apps_v1_client: Optional[client.AppsV1Api] = None
batch_v1_client: Optional[client.BatchV1Api] = None
custom_objects_client: Optional[client.CustomObjectsApi] = None


def init_k8s_client():
    """Initialize Kubernetes client with dual-mode support (in-cluster and out-of-cluster)"""
    global v1_client, apps_v1_client, batch_v1_client, custom_objects_client

    try:
        if settings.K8S_MODE == "in-cluster":
            # Production mode: running inside Kubernetes pod
            logger.info("Loading in-cluster Kubernetes configuration")
            config.load_incluster_config()
        else:
            # Development mode: running outside cluster with Service Account authentication
            logger.info("Loading out-of-cluster Kubernetes configuration with Service Account")

            # Validate required configuration
            if not settings.K8S_API_SERVER or not settings.K8S_TOKEN:
                raise ValueError("K8S_API_SERVER and K8S_TOKEN are required for out-of-cluster mode")

            # Create configuration object
            configuration = Configuration()
            configuration.host = settings.K8S_API_SERVER
            configuration.api_key["authorization"] = f"Bearer {settings.K8S_TOKEN}"

            # Set CA certificate if provided
            if settings.K8S_CA_CERT:
                # Check if it's a file path or actual certificate content
                if os.path.exists(settings.K8S_CA_CERT):
                    configuration.ssl_ca_cert = settings.K8S_CA_CERT
                    logger.info(f"Using CA certificate from file: {settings.K8S_CA_CERT}")
                else:
                    logger.warning("K8S_CA_CERT path does not exist, treating as certificate content")
                    # Write cert to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.crt', delete=False) as f:
                        f.write(settings.K8S_CA_CERT)
                        configuration.ssl_ca_cert = f.name
                        logger.info(f"CA certificate written to temporary file: {f.name}")
            else:
                configuration.verify_ssl = False
                logger.warning("K8S_CA_CERT not provided, SSL verification disabled (not recommended for production)")

            # Set as default configuration
            Configuration.set_default(configuration)

        # Create API clients
        global v1_client, apps_v1_client, batch_v1_client, custom_objects_client
        v1_client = client.CoreV1Api()
        apps_v1_client = client.AppsV1Api()
        batch_v1_client = client.BatchV1Api()
        custom_objects_client = client.CustomObjectsApi()

        # Test connection by listing namespaces
        try:
            namespaces = v1_client.list_namespace(limit=1)
            logger.info(f"Connected to Kubernetes cluster successfully")
        except Exception as e:
            logger.error(f"Could not verify cluster connection: {e}")
            raise

        # Create namespace if it doesn't exist
        create_namespace_if_not_exists()

        # Validate storage configuration
        try:
            validate_storage_configuration()
            logger.info(f"Storage configuration validated: mode={settings.STORAGE_MODE}")
        except ValueError as e:
            logger.error(f"Invalid storage configuration: {e}")
            raise

        logger.info(f"Kubernetes client initialized in {settings.K8S_MODE} mode")
        logger.info(f"Using namespace: {settings.K8S_NAMESPACE}")

    except Exception as e:
        logger.error(f"Failed to initialize Kubernetes client: {e}")
        logger.warning("Running in mock mode - Kubernetes operations will be simulated")
        # Set clients to None to indicate mock mode
        v1_client = None
        apps_v1_client = None
        batch_v1_client = None
        custom_objects_client = None


def create_namespace_if_not_exists():
    """Create namespace if it doesn't exist"""
    try:
        v1_client.read_namespace(name=settings.K8S_NAMESPACE)
        logger.info(f"Namespace {settings.K8S_NAMESPACE} already exists")
    except ApiException as e:
        if e.status == 404:
            # Create namespace
            namespace = client.V1Namespace(
                metadata=client.V1ObjectMeta(
                    name=settings.K8S_NAMESPACE,
                    labels={
                        "name": settings.K8S_NAMESPACE,
                        "managed-by": "orchestration-service"
                    }
                )
            )
            v1_client.create_namespace(body=namespace)
            logger.info(f"Created namespace: {settings.K8S_NAMESPACE}")
        else:
            raise


def check_k8s_health() -> bool:
    """Check Kubernetes connection health"""
    if not v1_client:
        return False

    try:
        # Try to list namespaces as a health check
        v1_client.list_namespace(limit=1)
        return True
    except Exception as e:
        logger.error(f"Kubernetes health check failed: {e}")
        return False


def get_node_info() -> List[Dict[str, Any]]:
    """Get information about GPU nodes in the cluster"""
    nodes_info = []

    try:
        nodes = v1_client.list_node()

        for node in nodes.items:
            # Check if node has GPUs
            gpu_capacity = node.status.capacity.get("nvidia.com/gpu", "0")
            if int(gpu_capacity) > 0:
                node_info = {
                    "name": node.metadata.name,
                    "uid": node.metadata.uid,
                    "labels": node.metadata.labels,
                    "gpu_count": int(gpu_capacity),
                    "cpu_capacity": node.status.capacity.get("cpu", "0"),
                    "memory_capacity": node.status.capacity.get("memory", "0"),
                    "allocatable_gpus": int(node.status.allocatable.get("nvidia.com/gpu", "0")),
                    "allocatable_cpu": node.status.allocatable.get("cpu", "0"),
                    "allocatable_memory": node.status.allocatable.get("memory", "0"),
                    "conditions": [
                        {
                            "type": c.type,
                            "status": c.status,
                            "reason": c.reason
                        } for c in node.status.conditions
                    ],
                    "addresses": [
                        {
                            "type": addr.type,
                            "address": addr.address
                        } for addr in node.status.addresses
                    ]
                }

                # Determine node status
                ready_condition = next(
                    (c for c in node.status.conditions if c.type == "Ready"),
                    None
                )
                node_info["is_ready"] = ready_condition and ready_condition.status == "True"
                node_info["is_schedulable"] = not node.spec.unschedulable

                nodes_info.append(node_info)

    except Exception as e:
        logger.error(f"Failed to get node info: {e}")

    return nodes_info


def get_external_hostname() -> str:
    """
    Get the external hostname for accessing NodePort services.

    Priority:
    1. EXTERNAL_HOSTNAME environment variable
    2. K8S_EXTERNAL_HOST environment variable
    3. First node's external IP from cluster
    4. First node's internal IP from cluster
    5. Fallback to localhost

    Returns:
        External hostname/IP for NodePort access
    """
    # Check environment variables first
    if settings.EXTERNAL_HOSTNAME:
        logger.info(f"Using EXTERNAL_HOSTNAME from config: {settings.EXTERNAL_HOSTNAME}")
        return settings.EXTERNAL_HOSTNAME

    if settings.K8S_EXTERNAL_HOST:
        logger.info(f"Using K8S_EXTERNAL_HOST from config: {settings.K8S_EXTERNAL_HOST}")
        return settings.K8S_EXTERNAL_HOST

    # Try to get node IP from cluster
    try:
        nodes = v1_client.list_node(limit=1)
        if nodes.items:
            node = nodes.items[0]

            # Look for external IP first
            for addr in node.status.addresses:
                if addr.type == "ExternalIP" and addr.address:
                    logger.info(f"Using node ExternalIP: {addr.address}")
                    return addr.address

            # Fall back to internal IP
            for addr in node.status.addresses:
                if addr.type == "InternalIP" and addr.address:
                    logger.info(f"Using node InternalIP: {addr.address}")
                    return addr.address

    except Exception as e:
        logger.warning(f"Failed to get node IP from cluster: {e}")

    # Final fallback
    logger.warning("No external hostname configured and could not determine from cluster, using localhost")
    return "localhost"


def get_gpu_usage() -> Dict[str, Any]:
    """Get current GPU usage in the cluster"""
    usage = {
        "total_gpus": 0,
        "allocated_gpus": 0,
        "available_gpus": 0,
        "nodes": []
    }

    try:
        # Get node information
        nodes = v1_client.list_node()

        for node in nodes.items:
            gpu_capacity = int(node.status.capacity.get("nvidia.com/gpu", "0"))
            if gpu_capacity > 0:
                # Count allocated GPUs by checking pods on this node
                field_selector = f"spec.nodeName={node.metadata.name}"
                pods = v1_client.list_pod_for_all_namespaces(
                    field_selector=field_selector
                )

                allocated = 0
                for pod in pods.items:
                    # Only count Running pods (ignore Failed/Pending/Unknown)
                    if pod.status.phase != "Running":
                        continue

                    for container in pod.spec.containers:
                        if container.resources and container.resources.requests:
                            gpu_request = container.resources.requests.get("nvidia.com/gpu", "0")
                            allocated += int(gpu_request)

                node_usage = {
                    "node_name": node.metadata.name,
                    "total": gpu_capacity,
                    "allocated": allocated,
                    "available": gpu_capacity - allocated
                }

                usage["nodes"].append(node_usage)
                usage["total_gpus"] += gpu_capacity
                usage["allocated_gpus"] += allocated

        usage["available_gpus"] = usage["total_gpus"] - usage["allocated_gpus"]

    except Exception as e:
        logger.error(f"Failed to get GPU usage: {e}")

    return usage


def create_vllm_deployment(
    deployment_name: str,
    model_id: str,
    model_path: str,
    replicas: int = 1,
    gpu_count: int = 1,
    cpu_request: str = "4",
    cpu_limit: str = "8",
    memory_request: str = "32Gi",
    memory_limit: str = "64Gi",
    env_vars: Dict[str, str] = None,
    labels: Dict[str, str] = None,
    probe_config: Optional[Dict[str, Any]] = None,
    deployment_id: Optional[str] = None,
    use_hf_secret: bool = False,
    vllm_config: Optional[Dict[str, Any]] = None
) -> client.V1Deployment:
    """Create a vLLM deployment"""

    # Default environment variables
    default_env = {
        "MODEL": model_path,
        "PORT": str(settings.VLLM_DEFAULT_PORT),
        "HOST": "0.0.0.0",
        "CUDA_VISIBLE_DEVICES": "0" if gpu_count == 1 else ",".join(str(i) for i in range(gpu_count)),
        "HF_HOME": "/models/.cache",
        "TRANSFORMERS_CACHE": "/models/.cache",
        "HF_HUB_CACHE": "/models/.cache"
    }

    if env_vars:
        default_env.update(env_vars)

    # Convert env dict to list
    env_list = [
        client.V1EnvVar(name=k, value=str(v))
        for k, v in default_env.items()
    ]

    # Add HF_TOKEN if provided in env_vars (for gated models)
    # This ensures HuggingFace can authenticate for private/gated model access
    if env_vars and 'HF_TOKEN' in env_vars:
        logger.info(f"HF_TOKEN configured for deployment {deployment_name}")

    # Add HuggingFace token environment variables from secret if requested
    if use_hf_secret and deployment_id:
        from src.core.secrets import create_secret_env_vars
        secret_env_vars = create_secret_env_vars(deployment_id)
        env_list.extend(secret_env_vars)

    # Default labels
    default_labels = {
        "app": deployment_name,
        "model-id": model_id,
        "managed-by": "orchestration-service",
        "component": "vllm"
    }

    if labels:
        default_labels.update(labels)

    # Create volume and volume mount using storage helper
    volume, volume_mount = create_model_volume(deployment_name)

    # Build container args using vLLM adapter
    from src.adapters.inference_engines import get_engine_adapter

    base_args = [
        "--host", "0.0.0.0",
        "--port", str(settings.VLLM_DEFAULT_PORT),
        "--model", "$(MODEL)"
    ]

    # Use adapter to build complete args from vllm_config
    adapter = get_engine_adapter("vllm")
    if adapter and vllm_config:
        container_args = adapter.build_container_args(vllm_config, base_args)
    else:
        # Fallback: use base args with trust-remote-code
        container_args = base_args + ["--trust-remote-code"]

    # Container specification with volume mount for models
    container_spec = {
        "name": "vllm",
        "image": settings.VLLM_IMAGE,
        "args": container_args,
        "ports": [client.V1ContainerPort(container_port=settings.VLLM_DEFAULT_PORT)],
        "env": env_list,
        "resources": client.V1ResourceRequirements(
            requests={
                "cpu": cpu_request,
                "memory": memory_request,
                "nvidia.com/gpu": str(gpu_count)
            },
            limits={
                "cpu": cpu_limit,
                "memory": memory_limit,
                "nvidia.com/gpu": str(gpu_count)
            }
        ),
        "volume_mounts": [volume_mount],
        "liveness_probe": client.V1Probe(
            http_get=client.V1HTTPGetAction(
                path="/health",
                port=settings.VLLM_DEFAULT_PORT
            ),
            initial_delay_seconds=600,  # 10 min - allow time for large model downloads
            period_seconds=30,
            timeout_seconds=10,
            failure_threshold=5  # More tolerant for model loading
        ),
        "readiness_probe": client.V1Probe(
            http_get=client.V1HTTPGetAction(
                path="/health",
                port=settings.VLLM_DEFAULT_PORT
            ),
            initial_delay_seconds=300,  # 5 min - allow time for model download
            period_seconds=10,
            timeout_seconds=10,  # Increased from 5s
            failure_threshold=5  # More tolerant for model loading
        )
    }

    # Add startup probe if probe_config is provided
    if probe_config:
        container_spec["startup_probe"] = client.V1Probe(
            http_get=client.V1HTTPGetAction(
                path="/health",
                port=settings.VLLM_DEFAULT_PORT
            ),
            initial_delay_seconds=probe_config.get("startup_initial_delay", 30),
            period_seconds=probe_config.get("startup_period_seconds", 10),
            timeout_seconds=5,
            failure_threshold=probe_config.get("startup_failure_threshold", 60)
        )

    container = client.V1Container(**container_spec)

    # Pod template with storage volume
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=default_labels),
        spec=client.V1PodSpec(
            containers=[container],
            volumes=[volume],
            runtime_class_name=settings.VLLM_RUNTIME_CLASS,
            restart_policy="Always",
            # Add tolerations for GPU nodes if needed
            tolerations=[
                client.V1Toleration(
                    key="nvidia.com/gpu",
                    operator="Exists",
                    effect="NoSchedule"
                )
            ]
            # Node affinity removed - GPU resource request is sufficient for scheduling
        )
    )

    # Deployment specification
    spec = client.V1DeploymentSpec(
        replicas=replicas,
        selector=client.V1LabelSelector(match_labels={"app": deployment_name}),
        template=template,
        strategy=client.V1DeploymentStrategy(
            type="RollingUpdate",
            rolling_update=client.V1RollingUpdateDeployment(
                max_surge=1,
                max_unavailable=0
            )
        )
    )

    # Create PVC if in CSI mode
    if settings.STORAGE_MODE == "csi":
        pvc = create_pvc_for_deployment(deployment_name, settings.K8S_NAMESPACE)
        if pvc:
            try:
                v1_client.create_namespaced_persistent_volume_claim(
                    body=pvc,
                    namespace=settings.K8S_NAMESPACE
                )
                logger.info(f"Created PVC for deployment {deployment_name}")
            except ApiException as e:
                if e.status == 409:
                    logger.info(f"PVC for deployment {deployment_name} already exists")
                else:
                    logger.error(f"Failed to create PVC for deployment {deployment_name}: {e}")
                    raise

    # Create deployment
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(
            name=deployment_name,
            namespace=settings.K8S_NAMESPACE,
            labels=default_labels
        ),
        spec=spec
    )

    return apps_v1_client.create_namespaced_deployment(
        body=deployment,
        namespace=settings.K8S_NAMESPACE
    )


def create_vllm_service(
    service_name: str,
    deployment_name: str,
    port: int = None,
    labels: Dict[str, str] = None
) -> client.V1Service:
    """Create a service for vLLM deployment"""

    # Allocate a node port if not specified
    if port is None:
        import random
        port = random.randint(settings.NODE_PORT_MIN, settings.NODE_PORT_MAX)

    # Default labels
    default_labels = {
        "app": deployment_name,
        "managed-by": "orchestration-service",
        "component": "vllm-service"
    }

    if labels:
        default_labels.update(labels)

    # Service specification
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(
            name=service_name,
            namespace=settings.K8S_NAMESPACE,
            labels=default_labels
        ),
        spec=client.V1ServiceSpec(
            type=settings.SERVICE_TYPE,
            selector={"app": deployment_name},
            ports=[
                client.V1ServicePort(
                    port=settings.VLLM_DEFAULT_PORT,
                    target_port=settings.VLLM_DEFAULT_PORT,
                    node_port=port if settings.SERVICE_TYPE == "NodePort" else None,
                    protocol="TCP"
                )
            ]
        )
    )

    return v1_client.create_namespaced_service(
        body=service,
        namespace=settings.K8S_NAMESPACE
    )


def delete_deployment(deployment_name: str) -> bool:
    """Delete a deployment and associated resources"""
    try:
        # Delete the deployment
        apps_v1_client.delete_namespaced_deployment(
            name=deployment_name,
            namespace=settings.K8S_NAMESPACE,
            body=client.V1DeleteOptions()
        )
        logger.info(f"Deleted deployment: {deployment_name}")

        # Delete associated PVC if in CSI mode
        delete_pvc_for_deployment(deployment_name, settings.K8S_NAMESPACE)

        return True
    except ApiException as e:
        logger.error(f"Failed to delete deployment {deployment_name}: {e}")
        return False


def delete_service(service_name: str) -> bool:
    """Delete a service"""
    try:
        v1_client.delete_namespaced_service(
            name=service_name,
            namespace=settings.K8S_NAMESPACE,
            body=client.V1DeleteOptions()
        )
        logger.info(f"Deleted service: {service_name}")
        return True
    except ApiException as e:
        logger.error(f"Failed to delete service {service_name}: {e}")
        return False


def get_deployment_status(deployment_name: str) -> Optional[Dict[str, Any]]:
    """Get deployment status"""
    try:
        deployment = apps_v1_client.read_namespaced_deployment(
            name=deployment_name,
            namespace=settings.K8S_NAMESPACE
        )

        return {
            "name": deployment.metadata.name,
            "namespace": deployment.metadata.namespace,
            "replicas": deployment.spec.replicas,
            "ready_replicas": deployment.status.ready_replicas or 0,
            "available_replicas": deployment.status.available_replicas or 0,
            "unavailable_replicas": deployment.status.unavailable_replicas or 0,
            "conditions": [
                {
                    "type": c.type,
                    "status": c.status,
                    "reason": c.reason,
                    "message": c.message
                } for c in (deployment.status.conditions or [])
            ]
        }
    except ApiException as e:
        logger.error(f"Failed to get deployment status: {e}")
        return None


def scale_deployment(deployment_name: str, replicas: int) -> bool:
    """Scale a deployment"""
    try:
        # Get current deployment
        deployment = apps_v1_client.read_namespaced_deployment(
            name=deployment_name,
            namespace=settings.K8S_NAMESPACE
        )

        # Update replicas
        deployment.spec.replicas = replicas

        # Apply the change
        apps_v1_client.patch_namespaced_deployment(
            name=deployment_name,
            namespace=settings.K8S_NAMESPACE,
            body=deployment
        )

        logger.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
        return True

    except ApiException as e:
        logger.error(f"Failed to scale deployment: {e}")
        return False


def update_deployment(deployment_name: str, patch: Dict[str, Any]) -> bool:
    """
    Update a deployment with new configuration.

    Args:
        deployment_name: Name of the deployment to update
        patch: Dictionary containing updates (resources, env_vars, labels, etc.)

    Returns:
        bool: True if update successful, False otherwise
    """
    try:
        # Get current deployment
        deployment = apps_v1_client.read_namespaced_deployment(
            name=deployment_name,
            namespace=settings.K8S_NAMESPACE
        )

        # Apply resource updates if provided
        if "resources" in patch:
            container = deployment.spec.template.spec.containers[0]

            # Update resource requests
            if "requests" in patch["resources"]:
                if not container.resources:
                    container.resources = client.V1ResourceRequirements()
                if not container.resources.requests:
                    container.resources.requests = {}

                for key, value in patch["resources"]["requests"].items():
                    container.resources.requests[key] = value

            # Update resource limits
            if "limits" in patch["resources"]:
                if not container.resources:
                    container.resources = client.V1ResourceRequirements()
                if not container.resources.limits:
                    container.resources.limits = {}

                for key, value in patch["resources"]["limits"].items():
                    container.resources.limits[key] = value

        # Apply environment variable updates if provided
        if "env_vars" in patch:
            container = deployment.spec.template.spec.containers[0]

            # Convert existing env vars to dict for easier manipulation
            existing_env = {env.name: env for env in (container.env or [])}

            # Update or add new env vars
            for key, value in patch["env_vars"].items():
                if key in existing_env:
                    existing_env[key].value = str(value)
                else:
                    existing_env[key] = client.V1EnvVar(name=key, value=str(value))

            # Convert back to list
            container.env = list(existing_env.values())

        # Apply label updates if provided
        if "labels" in patch:
            if not deployment.metadata.labels:
                deployment.metadata.labels = {}
            deployment.metadata.labels.update(patch["labels"])

            # Also update pod template labels
            if not deployment.spec.template.metadata.labels:
                deployment.spec.template.metadata.labels = {}
            deployment.spec.template.metadata.labels.update(patch["labels"])

        # Apply annotation updates if provided
        if "annotations" in patch:
            if not deployment.metadata.annotations:
                deployment.metadata.annotations = {}
            deployment.metadata.annotations.update(patch["annotations"])

            # Also update pod template annotations
            if not deployment.spec.template.metadata.annotations:
                deployment.spec.template.metadata.annotations = {}
            deployment.spec.template.metadata.annotations.update(patch["annotations"])

        # Apply the changes
        apps_v1_client.patch_namespaced_deployment(
            name=deployment_name,
            namespace=settings.K8S_NAMESPACE,
            body=deployment
        )

        logger.info(f"Updated deployment {deployment_name} with patch: {patch}")
        return True

    except ApiException as e:
        logger.error(f"Failed to update deployment {deployment_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating deployment {deployment_name}: {e}")
        return False


def get_pods_for_deployment(deployment_name: str) -> List[Dict[str, Any]]:
    """Get pods for a specific deployment"""
    pods_info = []

    try:
        # List pods with label selector
        pods = v1_client.list_namespaced_pod(
            namespace=settings.K8S_NAMESPACE,
            label_selector=f"app={deployment_name}"
        )

        for pod in pods.items:
            pod_info = {
                "name": pod.metadata.name,
                "uid": pod.metadata.uid,
                "node_name": pod.spec.node_name,
                "pod_ip": pod.status.pod_ip,
                "phase": pod.status.phase,
                "conditions": [
                    {
                        "type": c.type,
                        "status": c.status,
                        "reason": c.reason
                    } for c in (pod.status.conditions or [])
                ],
                "containers": []
            }

            # Get container statuses
            for container_status in (pod.status.container_statuses or []):
                container_info = {
                    "name": container_status.name,
                    "ready": container_status.ready,
                    "restart_count": container_status.restart_count,
                    "state": None
                }

                if container_status.state:
                    if container_status.state.running:
                        container_info["state"] = "running"
                    elif container_status.state.waiting:
                        container_info["state"] = f"waiting: {container_status.state.waiting.reason}"
                    elif container_status.state.terminated:
                        container_info["state"] = f"terminated: {container_status.state.terminated.reason}"

                pod_info["containers"].append(container_info)

            pods_info.append(pod_info)

    except Exception as e:
        logger.error(f"Failed to get pods for deployment: {e}")

    return pods_info


def get_pods_for_node(node_name: str) -> List[Dict[str, Any]]:
    """
    Get all running pods on a specific node

    Args:
        node_name: Kubernetes node name

    Returns:
        List of pod information dictionaries
    """
    pods_info = []

    try:
        field_selector = f"spec.nodeName={node_name},status.phase=Running"
        pods = v1_client.list_pod_for_all_namespaces(field_selector=field_selector)

        for pod in pods.items:
            containers_info = []
            for container_spec in pod.spec.containers:
                container_info = {
                    "name": container_spec.name,
                    "resources": {
                        "requests": dict(container_spec.resources.requests) if container_spec.resources and container_spec.resources.requests else {},
                        "limits": dict(container_spec.resources.limits) if container_spec.resources and container_spec.resources.limits else {}
                    }
                }
                containers_info.append(container_info)

            pod_info = {
                "name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "node_name": pod.spec.node_name,
                "phase": pod.status.phase,
                "containers": containers_info
            }
            pods_info.append(pod_info)

    except Exception as e:
        logger.error(f"Failed to get pods for node {node_name}: {e}")

    return pods_info


def get_pod_logs(
    pod_name: str,
    namespace: str = None,
    tail_lines: int = 100,
    container_name: str = None
) -> str:
    """
    Get logs from a specific pod.

    Args:
        pod_name: Name of the pod
        namespace: Kubernetes namespace (default: settings.K8S_NAMESPACE)
        tail_lines: Number of recent log lines to retrieve
        container_name: Specific container name (optional, uses first container if not specified)

    Returns:
        str: Pod logs

    Raises:
        Exception: If logs cannot be retrieved
    """
    global k8s_client

    if not namespace:
        namespace = settings.K8S_NAMESPACE

    try:
        # Get pod logs
        logs = k8s_client.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            tail_lines=tail_lines,
            container=container_name
        )
        return logs

    except Exception as e:
        logger.error(f"Failed to get logs for pod {pod_name}: {e}")
        return f"Error retrieving logs: {str(e)}"


def get_pod_status(deployment_name: str, namespace: str = None) -> Dict[str, Any]:
    """
    Get detailed status of all pods for a deployment.

    Args:
        deployment_name: Name of the deployment
        namespace: Kubernetes namespace (default: settings.K8S_NAMESPACE)

    Returns:
        dict: Status information with keys:
            - ready: bool - True if all pods are ready
            - total_pods: int - Total number of pods
            - ready_pods: int - Number of ready pods
            - pod_statuses: list - List of individual pod statuses
    """
    global v1_client

    if not namespace:
        namespace = settings.K8S_NAMESPACE

    status = {
        "ready": False,
        "total_pods": 0,
        "ready_pods": 0,
        "pod_statuses": []
    }

    try:
        # Get all pods for this deployment
        pods = v1_client.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app={deployment_name}"
        )

        status["total_pods"] = len(pods.items)
        running_pods = 0
        failed_pods = 0

        for pod in pods.items:
            pod_ready = False
            if pod.status.conditions:
                for condition in pod.status.conditions:
                    if condition.type == "Ready" and condition.status == "True":
                        pod_ready = True
                        status["ready_pods"] += 1
                        break

            if pod.status.phase == "Running":
                running_pods += 1
            elif pod.status.phase == "Failed":
                failed_pods += 1

            pod_status = {
                "name": pod.metadata.name,
                "ready": pod_ready,
                "phase": pod.status.phase,
                "conditions": [
                    {
                        "type": c.type,
                        "status": c.status,
                        "reason": c.reason if c.reason else None
                    } for c in (pod.status.conditions or [])
                ]
            }
            status["pod_statuses"].append(pod_status)

        status["ready"] = status["ready_pods"] > 0
        status["running_pods"] = running_pods
        status["failed_pods"] = failed_pods

    except Exception as e:
        logger.error(f"Failed to get pod status for deployment {deployment_name}: {e}")

    return status