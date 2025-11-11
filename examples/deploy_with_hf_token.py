#!/usr/bin/env python3
"""
Example: Deploy a model with HuggingFace token for accessing private models

This example demonstrates how to deploy a private HuggingFace model
using a HuggingFace API token for authentication.
"""

import requests
import json
import os
import time
from typing import Optional


def deploy_model_with_hf_token(
    base_url: str,
    model_id: str,
    huggingface_token: str,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    tenant_id: Optional[str] = None,
    replicas: int = 1
) -> dict:
    """
    Deploy a model with HuggingFace token support.

    Args:
        base_url: Base URL of the orchestration service
        model_id: Model ID from registry
        huggingface_token: HuggingFace API token for accessing private models
        model_name: Optional model name
        model_path: Optional path to model (HuggingFace ID)
        tenant_id: Optional tenant ID
        replicas: Number of replicas to deploy

    Returns:
        Deployment response dictionary
    """
    endpoint = f"{base_url}/api/v1/deployments"

    # Prepare request payload
    payload = {
        "model_id": model_id,
        "replicas": replicas,
        "huggingface_token": huggingface_token  # Include HF token
    }

    # Add optional fields if provided
    if model_name:
        payload["model_name"] = model_name
    if model_path:
        payload["model_path"] = model_path
    if tenant_id:
        payload["tenant_id"] = tenant_id

    # Set headers
    headers = {
        "Content-Type": "application/json"
    }
    if tenant_id:
        headers["X-Tenant-ID"] = tenant_id

    # Send deployment request
    response = requests.post(endpoint, json=payload, headers=headers)

    if response.status_code == 202:
        print(f"✅ Deployment initiated successfully!")
        deployment_data = response.json()
        print(f"Deployment ID: {deployment_data['deployment_id']}")
        print(f"Status: {deployment_data['status']}")
        return deployment_data
    else:
        print(f"❌ Failed to deploy model: {response.status_code}")
        print(f"Error: {response.text}")
        return None


def check_deployment_status(base_url: str, deployment_id: str) -> dict:
    """
    Check the status of a deployment.

    Args:
        base_url: Base URL of the orchestration service
        deployment_id: Deployment ID to check

    Returns:
        Deployment status dictionary
    """
    endpoint = f"{base_url}/api/v1/deployments/{deployment_id}/status"

    response = requests.get(endpoint)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to get deployment status: {response.status_code}")
        return None


def delete_deployment(base_url: str, deployment_id: str) -> bool:
    """
    Delete a deployment.

    Args:
        base_url: Base URL of the orchestration service
        deployment_id: Deployment ID to delete

    Returns:
        True if deletion successful, False otherwise
    """
    endpoint = f"{base_url}/api/v1/deployments/{deployment_id}"

    response = requests.delete(endpoint)

    if response.status_code == 200:
        print(f"✅ Deployment {deployment_id} deleted successfully!")
        return True
    else:
        print(f"❌ Failed to delete deployment: {response.status_code}")
        return False


def main():
    """Main example function."""
    # Configuration
    BASE_URL = os.getenv("ORCHESTRATION_SERVICE_URL", "http://localhost:8003")
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Get token from environment

    if not HF_TOKEN:
        print("⚠️  Warning: HUGGINGFACE_TOKEN not set in environment variables")
        print("   Private models will not be accessible without a token")
        print("   Set it with: export HUGGINGFACE_TOKEN='hf_your_token_here'")
        print()

    # Example 1: Deploy a private model with HuggingFace token
    print("=" * 60)
    print("Example 1: Deploy Private Model with HuggingFace Token")
    print("=" * 60)

    # Deploy a private model (example using a hypothetical private model)
    deployment = deploy_model_with_hf_token(
        base_url=BASE_URL,
        model_id="private-llama-model",
        model_name="Private Llama 7B",
        model_path="myorg/private-llama-7b",  # Private HuggingFace model ID
        huggingface_token=HF_TOKEN or "hf_example_token_123",
        tenant_id="tenant-123",
        replicas=1
    )

    if deployment:
        deployment_id = deployment["deployment_id"]

        # Wait for deployment to be ready
        print("\n⏳ Waiting for deployment to be ready...")
        max_attempts = 30
        for i in range(max_attempts):
            time.sleep(10)
            status = check_deployment_status(BASE_URL, deployment_id)
            if status:
                print(f"   Status: {status['status']} (Ready: {status.get('ready_replicas', 0)}/{status.get('replicas', 1)})")
                if status['status'] == 'running' and status.get('ready_replicas', 0) > 0:
                    print("\n✅ Deployment is ready!")
                    print(f"   Internal endpoint: {deployment.get('internal_endpoint')}")
                    print(f"   External endpoint: {deployment.get('external_endpoint')}")
                    break
                elif status['status'] == 'failed':
                    print("\n❌ Deployment failed!")
                    break
            if i == max_attempts - 1:
                print("\n⏱️  Deployment timed out")

    # Example 2: Deploy a public model without HuggingFace token
    print("\n" + "=" * 60)
    print("Example 2: Deploy Public Model (No Token Required)")
    print("=" * 60)

    public_deployment = deploy_model_with_hf_token(
        base_url=BASE_URL,
        model_id="llama2-7b",
        model_name="Llama 2 7B",
        model_path="meta-llama/Llama-2-7b-hf",  # Public model
        huggingface_token=None,  # No token needed for public models
        tenant_id="tenant-123",
        replicas=1
    )

    if public_deployment:
        print(f"\n✅ Public model deployed: {public_deployment['deployment_id']}")

    # Cleanup example (optional)
    if deployment:
        print("\n" + "=" * 60)
        print("Cleanup: Delete Deployments")
        print("=" * 60)
        input("\nPress Enter to delete the deployments...")

        delete_deployment(BASE_URL, deployment["deployment_id"])
        if public_deployment:
            delete_deployment(BASE_URL, public_deployment["deployment_id"])


if __name__ == "__main__":
    main()