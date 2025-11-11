"""Initial orchestration schema

Revision ID: 001_initial_orchestration
Revises:
Create Date: 2025-11-12 00:00:00.000000

Creates initial orchestration schema with:
- orchestration.gpu_nodes (GPU node inventory)
- orchestration.pod_allocations (Pod to GPU mapping)
- orchestration.deployments (vLLM deployments)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial_orchestration'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create initial orchestration schema and tables.
    """
    # Create orchestration schema if not exists
    op.execute("CREATE SCHEMA IF NOT EXISTS orchestration")

    # Create gpu_nodes table
    op.create_table(
        'gpu_nodes',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('node_id', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('node_name', sa.String(255), nullable=False),
        sa.Column('node_ip', sa.String(45), nullable=False),
        sa.Column('gpu_type', sa.String(100), nullable=False),
        sa.Column('gpu_count', sa.Integer(), nullable=False),
        sa.Column('gpu_memory_gb', sa.Integer(), nullable=False),
        sa.Column('cpu_cores', sa.Integer(), nullable=False),
        sa.Column('memory_gb', sa.Integer(), nullable=False),
        sa.Column('storage_gb', sa.Integer(), nullable=False),
        sa.Column('gpus_allocated', sa.Integer(), default=0, nullable=False),
        sa.Column('cpu_allocated', sa.Float(), default=0.0, nullable=False),
        sa.Column('memory_allocated_gb', sa.Float(), default=0.0, nullable=False),
        sa.Column('status', sa.String(50), default='ready', nullable=False),
        sa.Column('is_schedulable', sa.Boolean(), default=True, nullable=False),
        sa.Column('labels', sa.JSON(), default=dict),
        sa.Column('annotations', sa.JSON(), default=dict),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('last_heartbeat', sa.DateTime(), server_default=sa.func.now()),
        schema='orchestration'
    )

    # Create deployments table
    op.create_table(
        'deployments',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('deployment_id', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('deployment_name', sa.String(255), nullable=False),
        sa.Column('tenant_id', sa.String(255), nullable=False, index=True),
        sa.Column('model_id', sa.String(255), nullable=False, index=True),
        sa.Column('model_name', sa.String(255), nullable=False),
        sa.Column('engine_type', sa.String(50), default='vllm', nullable=False),
        sa.Column('replicas', sa.Integer(), default=1, nullable=False),
        sa.Column('desired_replicas', sa.Integer(), default=1, nullable=False),
        sa.Column('current_replicas', sa.Integer(), default=0, nullable=False),
        sa.Column('gpu_per_replica', sa.Integer(), default=1, nullable=False),
        sa.Column('cpu_request', sa.Float(), nullable=False),
        sa.Column('cpu_limit', sa.Float(), nullable=False),
        sa.Column('memory_request_gb', sa.Float(), nullable=False),
        sa.Column('memory_limit_gb', sa.Float(), nullable=False),
        sa.Column('model_path', sa.String(500), nullable=False),
        sa.Column('model_params', sa.JSON(), default=dict),
        sa.Column('vllm_config', sa.JSON(), default=dict),
        sa.Column('status', sa.String(50), default='pending', nullable=False),
        sa.Column('ready_replicas', sa.Integer(), default=0, nullable=False),
        sa.Column('available_replicas', sa.Integer(), default=0, nullable=False),
        sa.Column('internal_endpoint', sa.String(500), nullable=True),
        sa.Column('external_endpoint', sa.String(500), nullable=True),
        sa.Column('labels', sa.JSON(), default=dict),
        sa.Column('annotations', sa.JSON(), default=dict),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('last_health_check', sa.DateTime(), nullable=True),
        sa.Column('health_check_failures', sa.Integer(), default=0, nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('deployed_at', sa.DateTime(), nullable=True),
        sa.Column('terminated_at', sa.DateTime(), nullable=True),
        schema='orchestration'
    )

    # Create pod_allocations table
    op.create_table(
        'pod_allocations',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('allocation_id', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('deployment_id', sa.String(255), nullable=False, index=True),
        sa.Column('deployment_name', sa.String(255), nullable=False),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('pod_name', sa.String(255), nullable=False),
        sa.Column('pod_uid', sa.String(255), unique=True, nullable=True),
        sa.Column('namespace', sa.String(255), nullable=False, default='ai-platform'),
        sa.Column('pod_ip', sa.String(45), nullable=True),
        sa.Column('node_id', sa.String(255), nullable=False),
        sa.Column('gpu_count', sa.Integer(), nullable=False, default=1),
        sa.Column('gpu_indices', sa.JSON(), nullable=True),
        sa.Column('cpu_request', sa.Float(), nullable=False),
        sa.Column('cpu_limit', sa.Float(), nullable=False),
        sa.Column('memory_request_gb', sa.Float(), nullable=False),
        sa.Column('memory_limit_gb', sa.Float(), nullable=False),
        sa.Column('service_name', sa.String(255), nullable=True),
        sa.Column('service_port', sa.Integer(), nullable=True),
        sa.Column('node_port', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(50), default='pending', nullable=False),
        sa.Column('is_ready', sa.Boolean(), default=False, nullable=False),
        sa.Column('restart_count', sa.Integer(), default=0, nullable=False),
        sa.Column('container_image', sa.String(500), nullable=False),
        sa.Column('environment_vars', sa.JSON(), default=dict),
        sa.Column('model_config', sa.JSON(), default=dict),
        sa.Column('gpu_utilization_percent', sa.Float(), nullable=True),
        sa.Column('memory_utilization_percent', sa.Float(), nullable=True),
        sa.Column('cpu_utilization_percent', sa.Float(), nullable=True),
        sa.Column('request_count', sa.BigInteger(), default=0),
        sa.Column('error_count', sa.BigInteger(), default=0),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('terminated_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('pod_name', 'namespace', name='_pod_namespace_uc'),
        sa.ForeignKeyConstraint(['node_id'], ['orchestration.gpu_nodes.node_id']),
        sa.ForeignKeyConstraint(['deployment_id'], ['orchestration.deployments.deployment_id'], ondelete='CASCADE'),
        schema='orchestration'
    )

    # Create indexes
    op.create_index('idx_pod_allocations_tenant', 'pod_allocations', ['tenant_id'], schema='orchestration')
    op.create_index('idx_pod_allocations_model', 'pod_allocations', ['model_id'], schema='orchestration')
    op.create_index('idx_pod_allocations_status', 'pod_allocations', ['status'], schema='orchestration')
    op.create_index('idx_deployments_tenant', 'deployments', ['tenant_id'], schema='orchestration')
    op.create_index('idx_deployments_status', 'deployments', ['status'], schema='orchestration')


def downgrade() -> None:
    """
    Drop orchestration schema and all tables.
    """
    op.drop_table('pod_allocations', schema='orchestration')
    op.drop_table('deployments', schema='orchestration')
    op.drop_table('gpu_nodes', schema='orchestration')
    op.execute("DROP SCHEMA IF EXISTS orchestration CASCADE")
