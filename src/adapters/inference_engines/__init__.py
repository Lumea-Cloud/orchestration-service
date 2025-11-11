"""
Inference Engine Adapters

Provides abstraction layer for different inference engines (vLLM, TGI, TensorRT, etc.)
Each adapter normalizes engine-specific configurations into a common interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseInferenceEngine(ABC):
    """
    Base interface for inference engine adapters.

    Each inference engine (vLLM, TGI, TensorRT, etc.) has different configuration
    formats and capabilities. This interface provides a normalized way to extract
    GPU resource information regardless of the engine type.
    """

    @abstractmethod
    def get_gpu_memory_fraction(self, config: Dict[str, Any]) -> float:
        """
        Extract GPU memory fraction from engine-specific config.

        Args:
            config: Engine-specific configuration dictionary

        Returns:
            GPU memory fraction (0.0-1.0). Returns 1.0 if not specified.
        """
        pass

    @abstractmethod
    def build_container_args(self, config: Dict[str, Any], base_args: list) -> list:
        """
        Build container args from engine-specific config.

        Args:
            config: Engine-specific configuration dictionary
            base_args: Base arguments (host, port, model)

        Returns:
            Complete list of container arguments
        """
        pass


class VLLMEngine(BaseInferenceEngine):
    """
    vLLM inference engine adapter.

    vLLM configuration format:
    {
        "gpu_memory_utilization": 0.9,  # 0.0-1.0
        "max_model_len": 4096,
        "tensor_parallel_size": 1,
        "dtype": "auto",
        ...
    }
    """

    def get_gpu_memory_fraction(self, config: Dict[str, Any]) -> float:
        """
        Extract GPU memory utilization from vLLM config.

        Args:
            config: vLLM configuration dictionary

        Returns:
            GPU memory fraction (default: 0.9, vLLM's default)
        """
        return config.get("gpu_memory_utilization", 0.9)

    def build_container_args(self, config: Dict[str, Any], base_args: list) -> list:
        """
        Build vLLM container arguments from config.

        Args:
            config: vLLM configuration dictionary
            base_args: Base arguments (--host, --port, --model)

        Returns:
            Complete list of vLLM arguments
        """
        # Start with base args
        args = base_args.copy()

        # Add vLLM-specific arguments from config
        if "gpu_memory_utilization" in config:
            args.extend(["--gpu-memory-utilization", str(config["gpu_memory_utilization"])])

        if "max_model_len" in config:
            args.extend(["--max-model-len", str(config["max_model_len"])])

        if "tensor_parallel_size" in config:
            args.extend(["--tensor-parallel-size", str(config["tensor_parallel_size"])])

        if "dtype" in config:
            args.extend(["--dtype", config["dtype"]])

        if "quantization" in config:
            args.extend(["--quantization", config["quantization"]])

        if "trust_remote_code" in config and config["trust_remote_code"]:
            args.append("--trust-remote-code")

        return args


# Engine registry
INFERENCE_ENGINES = {
    "vllm": VLLMEngine(),
    # Future engines:
    # "tgi": TGIEngine(),
    # "tensorrt": TensorRTEngine(),
}


def get_engine_adapter(engine_type: str) -> Optional[BaseInferenceEngine]:
    """
    Get inference engine adapter by type.

    Args:
        engine_type: Engine type (vllm, tgi, tensorrt, etc.)

    Returns:
        Engine adapter instance or None if not found
    """
    return INFERENCE_ENGINES.get(engine_type.lower())
