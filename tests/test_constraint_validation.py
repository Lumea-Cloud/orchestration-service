"""
Test file for constraint validation in orchestration service.
Tests database constraints, foreign key relationships, and business logic validation.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from src.models.db_models import (
    Deployment, PodAllocation, ScalingEvent, GPUNode,
    DeploymentStatus, PodStatus, NodeStatus
)
from src.services.deployment_manager import DeploymentManager
from src.services.model_registry_client import ModelRegistryClient


@pytest.fixture
async def test_db():
    """
    Create a test database session.

    Returns:
        AsyncSession: Test database session
    """
    # Use in-memory SQLite for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=True
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def deployment_manager():
    """
    Create a DeploymentManager instance with mocked dependencies.

    Returns:
        DeploymentManager: Deployment manager instance
    """
    manager = DeploymentManager()
    manager.model_registry_client = AsyncMock(spec=ModelRegistryClient)
    return manager


class TestDatabaseConstraints:
    """Test database constraint enforcement"""

    @pytest.mark.asyncio
    async def test_deployment_model_id_must_be_uuid(self, test_db):
        """
        Test that deployment model_id must be a valid UUID.
        """
        # Try to create deployment with invalid model_id (string instead of UUID)
        deployment = Deployment(
            deployment_id="dep-test123",
            deployment_name="test-deployment",
            tenant_id=uuid.uuid4(),
            model_id="invalid-string-id",  # This should fail
            model_name="test-model"
        )

        test_db.add(deployment)

        with pytest.raises(IntegrityError) as exc_info:
            await test_db.commit()

        assert "model_id" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_pod_allocation_foreign_key_constraint(self, test_db):
        """
        Test that PodAllocation requires valid deployment_id foreign key.
        """
        # Try to create pod allocation without corresponding deployment
        pod_allocation = PodAllocation(
            allocation_id="alloc-test123",
            deployment_id="non-existent-deployment",  # This should fail
            deployment_name="test-deployment",
            model_id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            pod_name="test-pod",
            node_id="node-123",
            node_name="test-node",
            status=PodStatus.PENDING
        )

        test_db.add(pod_allocation)

        with pytest.raises(IntegrityError) as exc_info:
            await test_db.commit()

        assert "foreign key" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_scaling_event_foreign_key_constraint(self, test_db):
        """
        Test that ScalingEvent requires valid deployment_id foreign key.
        """
        # Try to create scaling event without corresponding deployment
        scaling_event = ScalingEvent(
            event_id="event-test123",
            deployment_id="non-existent-deployment",  # This should fail
            tenant_id=uuid.uuid4(),
            scaling_type="manual",
            from_replicas=1,
            to_replicas=3,
            status="pending",
            created_at=datetime.utcnow()
        )

        test_db.add(scaling_event)

        with pytest.raises(IntegrityError) as exc_info:
            await test_db.commit()

        assert "foreign key" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cascade_delete_protection(self, test_db):
        """
        Test that ON DELETE RESTRICT prevents cascading deletes.
        """
        # Create a deployment
        deployment = Deployment(
            deployment_id="dep-cascade-test",
            deployment_name="cascade-test",
            tenant_id=uuid.uuid4(),
            model_id=uuid.uuid4(),
            model_name="test-model",
            status=DeploymentStatus.RUNNING
        )
        test_db.add(deployment)
        await test_db.commit()

        # Create pod allocation referencing the deployment
        pod_allocation = PodAllocation(
            allocation_id="alloc-cascade-test",
            deployment_id="dep-cascade-test",
            deployment_name="cascade-test",
            model_id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            pod_name="test-pod",
            node_id="node-123",
            node_name="test-node",
            status=PodStatus.RUNNING
        )
        test_db.add(pod_allocation)
        await test_db.commit()

        # Try to delete the deployment - should fail due to RESTRICT
        await test_db.delete(deployment)

        with pytest.raises(IntegrityError) as exc_info:
            await test_db.commit()

        # The delete should be restricted
        assert "restrict" in str(exc_info.value).lower() or "constraint" in str(exc_info.value).lower()


class TestBusinessLogicValidation:
    """Test business logic validation in deployment creation"""

    @pytest.mark.asyncio
    async def test_deployment_creation_with_deleted_model(self, deployment_manager):
        """
        Test that deployment creation fails when model is soft-deleted.
        """
        # Mock model registry to return a deleted model
        deleted_model = {
            "id": str(uuid.uuid4()),
            "name": "deleted-model",
            "deleted_at": "2024-01-01T00:00:00Z",  # Model is soft-deleted
            "huggingface_id": "meta-llama/Llama-2-7b-hf"
        }

        deployment_manager.model_registry_client.get_model.return_value = deleted_model

        # Attempt to deploy the deleted model
        deployment_config = {
            "model_id": deleted_model["id"],
            "model_name": None,  # Will trigger fetch from registry
            "model_path": None,  # Will trigger fetch from registry
            "tenant_id": str(uuid.uuid4()),
            "replicas": 1
        }

        # This should raise an exception about deleted model
        with pytest.raises(ValueError) as exc_info:
            await deployment_manager.validate_deployment_request(deployment_config)

        assert "deleted" in str(exc_info.value).lower()
        assert "cannot be deployed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_deployment_creation_with_valid_model(self, deployment_manager):
        """
        Test that deployment creation succeeds with valid model.
        """
        # Mock model registry to return a valid model
        valid_model = {
            "id": str(uuid.uuid4()),
            "name": "valid-model",
            "deleted_at": None,  # Model is NOT deleted
            "huggingface_id": "meta-llama/Llama-2-7b-hf",
            "estimated_size_gb": 14,
            "requirements": {
                "gpu_memory_gb": 16
            }
        }

        deployment_manager.model_registry_client.get_model.return_value = valid_model

        # Deploy the valid model
        deployment_config = {
            "model_id": valid_model["id"],
            "model_name": valid_model["name"],
            "model_path": valid_model["huggingface_id"],
            "tenant_id": str(uuid.uuid4()),
            "replicas": 1,
            "model_info": valid_model
        }

        # This should succeed without exceptions
        deployment_id = f"dep-{uuid.uuid4().hex[:8]}"

        with patch('src.core.kubernetes_client.create_vllm_deployment') as mock_k8s:
            mock_k8s.return_value = MagicMock()

            # Should not raise any exceptions
            await deployment_manager.deploy_async(deployment_id, deployment_config)

    @pytest.mark.asyncio
    async def test_deployment_deletion_cleanup(self, deployment_manager, test_db):
        """
        Test that deployment deletion properly cleans up related records.
        """
        tenant_id = uuid.uuid4()
        model_id = uuid.uuid4()
        deployment_id = "dep-cleanup-test"

        # Create a deployment
        deployment = Deployment(
            deployment_id=deployment_id,
            deployment_name="cleanup-test",
            tenant_id=tenant_id,
            model_id=model_id,
            model_name="test-model",
            status=DeploymentStatus.RUNNING,
            current_replicas=2,
            desired_replicas=2
        )
        test_db.add(deployment)
        await test_db.commit()

        # Create related pod allocations
        for i in range(2):
            pod_allocation = PodAllocation(
                allocation_id=f"alloc-cleanup-{i}",
                deployment_id=deployment_id,
                deployment_name="cleanup-test",
                model_id=model_id,
                tenant_id=tenant_id,
                pod_name=f"test-pod-{i}",
                node_id=f"node-{i}",
                node_name=f"test-node-{i}",
                status=PodStatus.RUNNING,
                is_ready=True
            )
            test_db.add(pod_allocation)

        # Create related scaling events
        scaling_event = ScalingEvent(
            event_id="event-cleanup-test",
            deployment_id=deployment_id,
            tenant_id=tenant_id,
            scaling_type="manual",
            from_replicas=1,
            to_replicas=2,
            status="completed",
            created_at=datetime.utcnow()
        )
        test_db.add(scaling_event)
        await test_db.commit()

        # Mock Kubernetes operations
        with patch('src.core.kubernetes_client.delete_deployment') as mock_delete_deployment:
            with patch('src.core.kubernetes_client.delete_service') as mock_delete_service:
                mock_delete_deployment.return_value = True
                mock_delete_service.return_value = True

                # Delete the deployment
                await deployment_manager.delete_async(deployment_id, "cleanup-test")

        # Verify deployment status is terminated
        result = await test_db.execute(
            select(Deployment).where(Deployment.deployment_id == deployment_id)
        )
        deployment = result.scalar_one()
        assert deployment.status == "terminated"
        assert deployment.current_replicas == 0
        assert deployment.desired_replicas == 0
        assert deployment.terminated_at is not None

        # Verify pod allocations are marked as terminated
        result = await test_db.execute(
            select(PodAllocation).where(PodAllocation.deployment_id == deployment_id)
        )
        pod_allocations = result.scalars().all()
        for pod_alloc in pod_allocations:
            assert pod_alloc.status == "terminated"
            assert pod_alloc.is_ready is False
            assert pod_alloc.terminated_at is not None

        # Verify scaling events are marked as terminated
        result = await test_db.execute(
            select(ScalingEvent).where(ScalingEvent.deployment_id == deployment_id)
        )
        scaling_events = result.scalars().all()
        for event in scaling_events:
            assert event.status == "terminated"
            assert event.completed_at is not None


class TestUUIDValidation:
    """Test UUID type validation and conversion"""

    @pytest.mark.asyncio
    async def test_uuid_string_conversion(self, test_db):
        """
        Test that UUID strings are properly converted to UUID objects.
        """
        tenant_id = uuid.uuid4()
        model_id = uuid.uuid4()

        # Create deployment with UUID strings
        deployment = Deployment(
            deployment_id="dep-uuid-test",
            deployment_name="uuid-test",
            tenant_id=str(tenant_id),  # String representation
            model_id=str(model_id),  # String representation
            model_name="test-model",
            status=DeploymentStatus.PENDING
        )

        test_db.add(deployment)
        await test_db.commit()

        # Retrieve and verify UUIDs are properly stored
        result = await test_db.execute(
            select(Deployment).where(Deployment.deployment_id == "dep-uuid-test")
        )
        retrieved = result.scalar_one()

        assert isinstance(retrieved.tenant_id, uuid.UUID)
        assert isinstance(retrieved.model_id, uuid.UUID)
        assert retrieved.tenant_id == tenant_id
        assert retrieved.model_id == model_id

    @pytest.mark.asyncio
    async def test_invalid_uuid_format_rejection(self, test_db):
        """
        Test that invalid UUID formats are rejected.
        """
        # Try to create deployment with invalid UUID format
        deployment = Deployment(
            deployment_id="dep-invalid-uuid",
            deployment_name="invalid-uuid-test",
            tenant_id="not-a-valid-uuid",  # Invalid UUID format
            model_id=uuid.uuid4(),
            model_name="test-model",
            status=DeploymentStatus.PENDING
        )

        test_db.add(deployment)

        with pytest.raises((IntegrityError, ValueError)) as exc_info:
            await test_db.commit()

        # Should fail due to invalid UUID format
        assert "uuid" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])