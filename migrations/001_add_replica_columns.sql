-- Migration: Add replica tracking columns to deployments table
-- Date: 2025-10-05
-- Purpose: Add desired_replicas and current_replicas columns to track deployment replica state

-- Add desired_replicas column (represents the target state)
ALTER TABLE orchestration.deployments
ADD COLUMN IF NOT EXISTS desired_replicas INTEGER DEFAULT 1;

-- Add current_replicas column (represents the actual running state)
ALTER TABLE orchestration.deployments
ADD COLUMN IF NOT EXISTS current_replicas INTEGER DEFAULT 0;

-- Update existing deployments to set proper values
-- Set desired_replicas from existing replicas column
UPDATE orchestration.deployments
SET desired_replicas = replicas
WHERE desired_replicas IS NULL;

-- Set current_replicas from ready_replicas column
UPDATE orchestration.deployments
SET current_replicas = ready_replicas
WHERE current_replicas IS NULL;

-- Add indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_deployments_desired_replicas
ON orchestration.deployments(desired_replicas);

CREATE INDEX IF NOT EXISTS idx_deployments_current_replicas
ON orchestration.deployments(current_replicas);

-- Add comment for documentation
COMMENT ON COLUMN orchestration.deployments.desired_replicas IS 'Target number of replicas for the deployment';
COMMENT ON COLUMN orchestration.deployments.current_replicas IS 'Current number of running replicas';