"""Initial orchestration schema (existing tables)

Revision ID: 001
Revises:
Create Date: 2025-01-19 15:00:00.000000

This migration represents the existing database schema for orchestration-service.
It will be applied using 'alembic stamp head' without actually creating tables.

The existing migrations in migrations/*.sql have already been applied:
- 001_add_replica_columns.sql (added min_replicas, max_replicas, etc.)
- 002_add_health_tracking_fields.sql (added health check fields)

Tables in orchestration schema:
- orchestration.gpu_nodes (GPU node inventory)
- orchestration.pod_allocations (Pod to GPU mapping)
- orchestration.deployments (vLLM deployments)
- orchestration.deployment_events (Deployment lifecycle events)

This migration is a PLACEHOLDER - the actual schema is defined in SQLAlchemy models.
Use 'alembic stamp 001' to mark the current state without applying changes.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    This is a placeholder migration for existing schema.

    The actual tables already exist in the database and were created
    through previous SQL migrations in the migrations/ directory.

    To mark the database as migrated without applying changes:
        alembic stamp 001

    Future schema changes should be done through Alembic migrations.
    """
    pass


def downgrade() -> None:
    """
    Downgrade is not supported for the initial migration.

    This would require dropping all orchestration tables,
    which should only be done manually with full understanding
    of the consequences.
    """
    pass
