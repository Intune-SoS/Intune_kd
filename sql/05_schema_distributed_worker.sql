-- ============================================================================
-- Distributed Worker Tracking Columns
-- ============================================================================
-- Adds columns for coordinating distributed inference across multiple machines
-- Prevents duplicate processing and tracks which worker completed each record
--
-- Safe to run multiple times (uses ADD COLUMN IF NOT EXISTS)
-- ============================================================================

-- Worker currently processing this record (collision avoidance)
ALTER TABLE modelcomp_50k 
ADD COLUMN IF NOT EXISTS processing_worker VARCHAR(50) DEFAULT NULL;

-- Timestamp when processing started
ALTER TABLE modelcomp_50k 
ADD COLUMN IF NOT EXISTS processing_at TIMESTAMP DEFAULT NULL;

-- Worker that completed the generation
ALTER TABLE modelcomp_50k 
ADD COLUMN IF NOT EXISTS tuned_worker VARCHAR(50) DEFAULT NULL;

-- Timestamp when tuned output was generated
ALTER TABLE modelcomp_50k 
ADD COLUMN IF NOT EXISTS tuned_at TIMESTAMP DEFAULT NULL;

-- Index for finding unprocessed records (collision avoidance query)
CREATE INDEX IF NOT EXISTS idx_modelcomp_50k_processing_worker 
ON modelcomp_50k(processing_worker) 
WHERE processing_worker IS NULL;

-- Index for tracking output by worker
CREATE INDEX IF NOT EXISTS idx_modelcomp_50k_tuned_worker 
ON modelcomp_50k(tuned_worker) 
WHERE tuned_worker IS NOT NULL;

-- Composite index for common distributed query (status + checkpoint + processing)
CREATE INDEX IF NOT EXISTS idx_modelcomp_50k_distributed_query
ON modelcomp_50k(checkpoint, status, processing_worker);

-- Comments
COMMENT ON COLUMN modelcomp_50k.processing_worker IS 
'Worker ID currently processing this record (e.g., mac-1, rtx-1) - for collision avoidance';

COMMENT ON COLUMN modelcomp_50k.processing_at IS 
'Timestamp when worker started processing (for detecting stale locks)';

COMMENT ON COLUMN modelcomp_50k.tuned_worker IS 
'Worker ID that completed tuned output generation';

COMMENT ON COLUMN modelcomp_50k.tuned_at IS 
'Timestamp when tuned output was completed';
