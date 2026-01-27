-- Fix for Alert Status Constraint (Hibernate Enum mapping issue)
-- We drop the constraint to allow new values (RESOLVED, ACTIVE) without recreating the table.
ALTER TABLE alerts DROP CONSTRAINT IF EXISTS alerts_status_check;

-- Add report_id column if it doesn't exist (for ReportEntity relationship)
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS report_id VARCHAR(36);
