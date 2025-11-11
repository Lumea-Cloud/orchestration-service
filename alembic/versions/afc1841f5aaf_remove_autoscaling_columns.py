"""remove_autoscaling_columns

Revision ID: afc1841f5aaf
Revises: 001
Create Date: 2025-10-31 00:55:54.680408

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'afc1841f5aaf'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Remove autoscaling columns from deployments table"""
    op.drop_column('deployments', 'min_replicas', schema='orchestration')
    op.drop_column('deployments', 'max_replicas', schema='orchestration')
    op.drop_column('deployments', 'target_gpu_utilization', schema='orchestration')


def downgrade() -> None:
    """Restore autoscaling columns to deployments table"""
    op.add_column('deployments', sa.Column('min_replicas', sa.Integer(), nullable=False, server_default='1'), schema='orchestration')
    op.add_column('deployments', sa.Column('max_replicas', sa.Integer(), nullable=False, server_default='5'), schema='orchestration')
    op.add_column('deployments', sa.Column('target_gpu_utilization', sa.Float(), server_default='80.0'), schema='orchestration')
