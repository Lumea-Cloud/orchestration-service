-- Migration: Add health tracking fields to deployments table
-- Date: 2025-10-07
-- Purpose: Track deployment health status, errors, and health check results

-- Add error_message column to store error details
ALTER TABLE orchestration.deployments
ADD COLUMN IF NOT EXISTS error_message TEXT;

-- Add last_health_check column to track when last health check was performed
ALTER TABLE orchestration.deployments
ADD COLUMN IF NOT EXISTS last_health_check TIMESTAMP;

-- Add health_check_failures column to track consecutive failure count
ALTER TABLE orchestration.deployments
ADD COLUMN IF NOT EXISTS health_check_failures INTEGER DEFAULT 0;

-- Add index for querying deployments with failures
CREATE INDEX IF NOT EXISTS idx_deployments_health_failures
ON orchestration.deployments(health_check_failures);

-- Add comments for documentation
COMMENT ON COLUMN orchestration.deployments.error_message IS 'Last error message from deployment or health check';
COMMENT ON COLUMN orchestration.deployments.last_health_check IS 'Timestamp of last health check';
COMMENT ON COLUMN orchestration.deployments.health_check_failures IS 'Consecutive health check failure count';
