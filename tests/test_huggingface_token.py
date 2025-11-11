"""
Tests for HuggingFace token support in Kubernetes deployments
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from kubernetes import client

from src.core.secrets import (
    create_huggingface_secret,
    delete_huggingface_secret,
    check_secret_exists,
    create_secret_env_vars
)
from src.api.public import SimpleDeploymentRequest


class TestHuggingFaceSecrets:
    """Test HuggingFace token secret management."""

    @patch('src.core.kubernetes_client.v1_client')
    def test_create_huggingface_secret(self, mock_v1_client):
        """
        Test creating a HuggingFace token secret.
        """
        # Setup mock
        mock_v1_client.create_namespaced_secret.return_value = MagicMock(
            metadata=MagicMock(name="hf-token-test-deployment")
        )

        # Test secret creation
        deployment_id = "test-deployment"
        token = "hf_test_token_123"

        result = create_huggingface_secret(deployment_id, token)

        # Verify the secret was created
        assert result is not None
        mock_v1_client.create_namespaced_secret.assert_called_once()

        # Verify secret structure
        call_args = mock_v1_client.create_namespaced_secret.call_args
        secret = call_args.kwargs['body']

        assert secret.metadata.name == f"hf-token-{deployment_id}"
        assert secret.string_data['HF_TOKEN'] == token
        assert secret.string_data['HUGGING_FACE_HUB_TOKEN'] == token
        assert secret.metadata.labels['deployment-id'] == deployment_id

    @patch('src.core.kubernetes_client.v1_client')
    def test_create_huggingface_secret_no_token(self, mock_v1_client):
        """
        Test that no secret is created when token is None or empty.
        """
        # Test with None
        result = create_huggingface_secret("test-deployment", None)
        assert result is None
        mock_v1_client.create_namespaced_secret.assert_not_called()

        # Test with empty string
        result = create_huggingface_secret("test-deployment", "")
        assert result is None
        mock_v1_client.create_namespaced_secret.assert_not_called()

    @patch('src.core.kubernetes_client.v1_client')
    def test_delete_huggingface_secret(self, mock_v1_client):
        """
        Test deleting a HuggingFace token secret.
        """
        deployment_id = "test-deployment"

        result = delete_huggingface_secret(deployment_id)

        assert result is True
        mock_v1_client.delete_namespaced_secret.assert_called_once()

        call_args = mock_v1_client.delete_namespaced_secret.call_args
        assert call_args.kwargs['name'] == f"hf-token-{deployment_id}"

    @patch('src.core.kubernetes_client.v1_client')
    def test_delete_nonexistent_secret(self, mock_v1_client):
        """
        Test deleting a secret that doesn't exist returns True.
        """
        from kubernetes.client.rest import ApiException

        mock_v1_client.delete_namespaced_secret.side_effect = ApiException(status=404)

        result = delete_huggingface_secret("nonexistent-deployment")

        assert result is True  # Should return True even if secret doesn't exist

    @patch('src.core.kubernetes_client.v1_client')
    def test_check_secret_exists(self, mock_v1_client):
        """
        Test checking if a secret exists.
        """
        deployment_id = "test-deployment"

        # Test when secret exists
        mock_v1_client.read_namespaced_secret.return_value = MagicMock()
        assert check_secret_exists(deployment_id) is True

        # Test when secret doesn't exist
        from kubernetes.client.rest import ApiException
        mock_v1_client.read_namespaced_secret.side_effect = ApiException(status=404)
        assert check_secret_exists(deployment_id) is False

    def test_create_secret_env_vars(self):
        """
        Test creating environment variable references to a secret.
        """
        deployment_id = "test-deployment"
        secret_name = f"hf-token-{deployment_id}"

        env_vars = create_secret_env_vars(deployment_id)

        assert len(env_vars) == 2

        # Check HF_TOKEN env var
        hf_token_var = env_vars[0]
        assert hf_token_var.name == "HF_TOKEN"
        assert hf_token_var.value_from.secret_key_ref.name == secret_name
        assert hf_token_var.value_from.secret_key_ref.key == "HF_TOKEN"
        assert hf_token_var.value_from.secret_key_ref.optional is False

        # Check HUGGING_FACE_HUB_TOKEN env var
        hf_hub_token_var = env_vars[1]
        assert hf_hub_token_var.name == "HUGGING_FACE_HUB_TOKEN"
        assert hf_hub_token_var.value_from.secret_key_ref.name == secret_name
        assert hf_hub_token_var.value_from.secret_key_ref.key == "HUGGING_FACE_HUB_TOKEN"
        assert hf_hub_token_var.value_from.secret_key_ref.optional is False


class TestDeploymentRequestWithToken:
    """Test SimpleDeploymentRequest with HuggingFace token."""

    def test_deployment_request_with_token(self):
        """
        Test that SimpleDeploymentRequest accepts huggingface_token.
        """
        request = SimpleDeploymentRequest(
            model_id="test-model",
            model_name="Test Model",
            model_path="test/model",
            huggingface_token="hf_test_token_123"
        )

        assert request.huggingface_token == "hf_test_token_123"
        assert request.model_id == "test-model"

    def test_deployment_request_without_token(self):
        """
        Test that huggingface_token is optional.
        """
        request = SimpleDeploymentRequest(
            model_id="test-model",
            model_name="Test Model",
            model_path="test/model"
        )

        assert request.huggingface_token is None


class TestKubernetesDeploymentWithSecret:
    """Test Kubernetes deployment creation with HuggingFace secret."""

    @patch('src.core.kubernetes_client.apps_v1_client')
    @patch('src.core.kubernetes_client.v1_client')
    def test_create_deployment_with_secret(self, mock_v1_client, mock_apps_v1_client):
        """
        Test creating a deployment with HuggingFace token secret mounted.
        """
        from src.core.kubernetes_client import create_vllm_deployment

        # Mock the deployment creation
        mock_apps_v1_client.create_namespaced_deployment.return_value = MagicMock()

        # Create deployment with HF secret
        deployment = create_vllm_deployment(
            deployment_name="test-deployment",
            model_id="test-model",
            model_path="test/model",
            deployment_id="dep-123",
            use_hf_secret=True
        )

        # Verify deployment was created
        mock_apps_v1_client.create_namespaced_deployment.assert_called_once()

        # Get the deployment spec
        call_args = mock_apps_v1_client.create_namespaced_deployment.call_args
        deployment_spec = call_args.kwargs['body']

        # Check that the container has environment variables referencing the secret
        container = deployment_spec.spec.template.spec.containers[0]
        env_names = [env.name for env in container.env]

        # Check for secret-referenced env vars
        secret_env_vars = [env for env in container.env if env.value_from is not None]
        secret_env_names = [env.name for env in secret_env_vars]

        assert "HF_TOKEN" in secret_env_names
        assert "HUGGING_FACE_HUB_TOKEN" in secret_env_names

    @patch('src.core.kubernetes_client.apps_v1_client')
    @patch('src.core.kubernetes_client.v1_client')
    def test_create_deployment_without_secret(self, mock_v1_client, mock_apps_v1_client):
        """
        Test creating a deployment without HuggingFace token secret.
        """
        from src.core.kubernetes_client import create_vllm_deployment

        # Mock the deployment creation
        mock_apps_v1_client.create_namespaced_deployment.return_value = MagicMock()

        # Create deployment without HF secret
        deployment = create_vllm_deployment(
            deployment_name="test-deployment",
            model_id="test-model",
            model_path="test/model",
            deployment_id="dep-123",
            use_hf_secret=False
        )

        # Verify deployment was created
        mock_apps_v1_client.create_namespaced_deployment.assert_called_once()

        # Get the deployment spec
        call_args = mock_apps_v1_client.create_namespaced_deployment.call_args
        deployment_spec = call_args.kwargs['body']

        # Check that no environment variables reference secrets
        container = deployment_spec.spec.template.spec.containers[0]
        secret_env_vars = [env for env in container.env if env.value_from is not None]

        # Should not have HF token env vars from secrets
        secret_env_names = [env.name for env in secret_env_vars]
        assert "HF_TOKEN" not in secret_env_names
        assert "HUGGING_FACE_HUB_TOKEN" not in secret_env_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])