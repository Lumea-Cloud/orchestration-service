"""
Configuration settings for Orchestration Service
"""

import os
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # Service configuration
    SERVICE_NAME: str = "orchestration-service"
    PORT: int = Field(default=8003, env="PORT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # Database configuration
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PORT: int = Field(default=5432, env="DB_PORT")
    DB_NAME: str = Field(default="ai_platform", env="DB_NAME")
    DB_USER: str = Field(default="ai_platform", env="DB_USER")
    DB_PASSWORD: str = Field(default="ai_platform_dev_2024", env="DB_PASSWORD")
    DB_SCHEMA: str = Field(default="orchestration", env="DB_SCHEMA")

    # Database connection pool settings
    DB_POOL_SIZE: int = Field(default=20, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=10, env="DB_MAX_OVERFLOW")
    DB_POOL_TIMEOUT: int = Field(default=30, env="DB_POOL_TIMEOUT")

    # Kubernetes configuration
    # Authentication mode: "in-cluster" for production, "out-of-cluster" for development
    K8S_MODE: str = Field(default="out-of-cluster", env="K8S_MODE")
    K8S_NAMESPACE: str = Field(default="ai-platform", env="K8S_NAMESPACE")

    # Out-of-cluster configuration (Service Account based)
    K8S_API_SERVER: Optional[str] = Field(default=None, env="K8S_API_SERVER")
    K8S_TOKEN: Optional[str] = Field(default=None, env="K8S_TOKEN")
    K8S_CA_CERT: Optional[str] = Field(default=None, env="K8S_CA_CERT")

    # vLLM configuration
    VLLM_IMAGE: str = Field(default="vllm/vllm-openai:latest", env="VLLM_IMAGE")
    VLLM_RUNTIME_CLASS: str = Field(default="nvidia", env="VLLM_RUNTIME_CLASS")
    VLLM_DEFAULT_GPU_MEMORY: int = Field(default=48, env="VLLM_DEFAULT_GPU_MEMORY")  # GB
    VLLM_DEFAULT_PORT: int = Field(default=8000, env="VLLM_DEFAULT_PORT")

    # GPU configuration
    GPU_TYPE: str = Field(default="nvidia-l40s", env="GPU_TYPE")
    GPU_MEMORY_GB: int = Field(default=48, env="GPU_MEMORY_GB")
    MAX_GPUS_PER_NODE: int = Field(default=2, env="MAX_GPUS_PER_NODE")

    # Resource limits
    DEFAULT_CPU_REQUEST: str = Field(default="4", env="DEFAULT_CPU_REQUEST")
    DEFAULT_CPU_LIMIT: str = Field(default="8", env="DEFAULT_CPU_LIMIT")
    DEFAULT_MEMORY_REQUEST: str = Field(default="32Gi", env="DEFAULT_MEMORY_REQUEST")
    DEFAULT_MEMORY_LIMIT: str = Field(default="64Gi", env="DEFAULT_MEMORY_LIMIT")

    # Deployment configuration
    DEPLOYMENT_TIMEOUT: int = Field(default=300, env="DEPLOYMENT_TIMEOUT")  # seconds
    POD_STARTUP_TIMEOUT: int = Field(default=600, env="POD_STARTUP_TIMEOUT")  # seconds
    HEALTH_CHECK_INTERVAL: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")  # seconds
    RESOURCE_TRACK_INTERVAL: int = Field(default=60, env="RESOURCE_TRACK_INTERVAL")  # seconds
    DEPLOYMENT_STATUS_CHECK_INTERVAL: int = Field(default=60, env="DEPLOYMENT_STATUS_CHECK_INTERVAL")  # seconds

    # Service mesh configuration
    SERVICE_TYPE: str = Field(default="NodePort", env="SERVICE_TYPE")
    NODE_PORT_MIN: int = Field(default=30000, env="NODE_PORT_MIN")
    NODE_PORT_MAX: int = Field(default=32767, env="NODE_PORT_MAX")

    # External access configuration
    EXTERNAL_HOSTNAME: Optional[str] = Field(default=None, env="EXTERNAL_HOSTNAME")
    K8S_EXTERNAL_HOST: Optional[str] = Field(default=None, env="K8S_EXTERNAL_HOST")

    # Monitoring configuration
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    ENABLE_TRACING: bool = Field(default=False, env="ENABLE_TRACING")

    # Internal service URLs
    AUTH_SERVICE_URL: str = Field(
        default="http://localhost:8001",
        env="AUTH_SERVICE_URL"
    )
    GATEWAY_SERVICE_URL: str = Field(
        default="http://localhost:8000",
        env="GATEWAY_SERVICE_URL"
    )
    MODEL_REGISTRY_URL: str = Field(
        default="http://localhost:8002",
        env="MODEL_REGISTRY_URL"
    )

    # Storage Configuration
    STORAGE_MODE: str = Field(default="nfs", env="STORAGE_MODE")  # nfs, csi, or pvc
    NFS_SERVER: str = Field(default="nfs-server.local", env="NFS_SERVER")
    NFS_PATH: str = Field(default="/models", env="NFS_PATH")
    MODEL_MOUNT_PATH: str = Field(default="/models", env="MODEL_MOUNT_PATH")

    # CSI/PVC Configuration
    STORAGE_CLASS: Optional[str] = Field(default="ai-platform-models", env="STORAGE_CLASS")
    PVC_NAME: Optional[str] = Field(default=None, env="PVC_NAME")  # For existing PVC mode
    PVC_SIZE: str = Field(default="1Ti", env="PVC_SIZE")  # For CSI mode

    @property
    def database_url(self) -> str:
        """Construct database URL"""
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def sync_database_url(self) -> str:
        """Construct synchronous database URL for Alembic"""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "allow"
    }


# Create settings instance
settings = Settings()