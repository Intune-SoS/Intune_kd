-- ============================================================================
-- CLEANUP LEGACY COLUMNS - Remove Unused Columns from modelcomp_50k
-- ============================================================================
-- This script removes 33 legacy columns that are no longer used.
-- 
-- The new incremental learning architecture uses:
--   - Separate ROWS per checkpoint (filtered by `checkpoint` column)
--   - `student_output_tuned` for tuned output (per row)
--   - `score_tuned` for tuned score (per row)
--
-- The OLD architecture stored everything in one row with:
--   - student_output_ckpt1, student_output_ckpt2, ... (10 columns)
--   - score_ckpt1, score_ckpt2, ... (10 columns)
--   - latency_ckpt1, latency_ckpt2, ... (10 columns)
--
-- ⚠️  WARNING: This will PERMANENTLY DELETE data in these columns!
-- ⚠️  Make sure to backup your data before running this script.
-- ============================================================================

-- ============================================================================
-- STEP 0: BACKUP RECOMMENDATION
-- ============================================================================
-- Before running this script, create a backup:
-- 
-- Option 1: Export to CSV from Supabase Dashboard
-- Option 2: Create a backup table:
/*
CREATE TABLE modelcomp_50k_backup_legacy AS 
SELECT * FROM modelcomp_50k;
*/

-- ============================================================================
-- STEP 1: VERIFY COLUMNS TO BE DELETED (Run this first to check)
-- ============================================================================
-- This query shows which columns exist and will be deleted

SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'modelcomp_50k' 
AND column_name IN (
    -- Per-checkpoint student outputs (LEGACY)
    'student_output_ckpt1', 'student_output_ckpt2', 'student_output_ckpt3',
    'student_output_ckpt4', 'student_output_ckpt5', 'student_output_ckpt6',
    'student_output_ckpt7', 'student_output_ckpt8', 'student_output_ckpt9',
    'student_output_ckpt10',
    -- Per-checkpoint scores (LEGACY)
    'score_ckpt1', 'score_ckpt2', 'score_ckpt3', 'score_ckpt4', 'score_ckpt5',
    'score_ckpt6', 'score_ckpt7', 'score_ckpt8', 'score_ckpt9', 'score_ckpt10',
    -- Per-checkpoint latencies (LEGACY)
    'latency_ckpt1', 'latency_ckpt2', 'latency_ckpt3', 'latency_ckpt4', 'latency_ckpt5',
    'latency_ckpt6', 'latency_ckpt7', 'latency_ckpt8', 'latency_ckpt9', 'latency_ckpt10',
    -- Batch columns (LEGACY)
    'student_output_batch', 'latency_batch', 'score_batch'
)
ORDER BY column_name;

-- ============================================================================
-- STEP 2: CHECK DATA IN LEGACY COLUMNS (Optional - see if there's data to lose)
-- ============================================================================
-- Run this to see if any legacy columns have data:
/*
SELECT 
    COUNT(*) FILTER (WHERE student_output_ckpt1 IS NOT NULL) as has_ckpt1,
    COUNT(*) FILTER (WHERE student_output_ckpt2 IS NOT NULL) as has_ckpt2,
    COUNT(*) FILTER (WHERE student_output_ckpt3 IS NOT NULL) as has_ckpt3,
    COUNT(*) FILTER (WHERE score_ckpt1 IS NOT NULL) as has_score_ckpt1,
    COUNT(*) FILTER (WHERE student_output_batch IS NOT NULL) as has_batch
FROM modelcomp_50k;
*/

-- ============================================================================
-- STEP 3: DROP LEGACY COLUMNS
-- ============================================================================
-- These columns are from the old architecture and are no longer used.

-- Drop per-checkpoint student output columns (10 columns)
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS student_output_ckpt1;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS student_output_ckpt2;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS student_output_ckpt3;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS student_output_ckpt4;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS student_output_ckpt5;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS student_output_ckpt6;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS student_output_ckpt7;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS student_output_ckpt8;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS student_output_ckpt9;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS student_output_ckpt10;

-- Drop per-checkpoint score columns (10 columns)
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS score_ckpt1;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS score_ckpt2;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS score_ckpt3;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS score_ckpt4;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS score_ckpt5;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS score_ckpt6;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS score_ckpt7;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS score_ckpt8;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS score_ckpt9;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS score_ckpt10;

-- Drop per-checkpoint latency columns (10 columns)
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS latency_ckpt1;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS latency_ckpt2;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS latency_ckpt3;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS latency_ckpt4;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS latency_ckpt5;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS latency_ckpt6;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS latency_ckpt7;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS latency_ckpt8;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS latency_ckpt9;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS latency_ckpt10;

-- Drop batch columns (3 columns)
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS student_output_batch;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS latency_batch;
ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS score_batch;

-- ============================================================================
-- STEP 4: OPTIONAL - Remove other potentially unused columns
-- ============================================================================
-- Uncomment these if you also want to remove:

-- worker_id: Legacy worker tracking (uncomment if not needed)
-- ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS worker_id;

-- overall_score: Duplicate of 'score' column (uncomment if redundant)
-- ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS overall_score;

-- generation_latency: Old latency column (uncomment if using latency_tuned only)
-- ALTER TABLE modelcomp_50k DROP COLUMN IF EXISTS generation_latency;

-- ============================================================================
-- STEP 5: VERIFY CLEANUP
-- ============================================================================
-- Run this to see the final schema:

SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'modelcomp_50k'
ORDER BY ordinal_position;

-- ============================================================================
-- STEP 6: COUNT REMAINING COLUMNS
-- ============================================================================

SELECT COUNT(*) as total_columns
FROM information_schema.columns
WHERE table_name = 'modelcomp_50k';

-- ============================================================================
-- SUMMARY OF CHANGES
-- ============================================================================
/*
COLUMNS REMOVED (33 total):
──────────────────────────────────────────────────────────────────────

LEGACY PER-CHECKPOINT OUTPUTS (10 columns):
  ❌ student_output_ckpt1    ❌ student_output_ckpt6
  ❌ student_output_ckpt2    ❌ student_output_ckpt7
  ❌ student_output_ckpt3    ❌ student_output_ckpt8
  ❌ student_output_ckpt4    ❌ student_output_ckpt9
  ❌ student_output_ckpt5    ❌ student_output_ckpt10

LEGACY PER-CHECKPOINT SCORES (10 columns):
  ❌ score_ckpt1    ❌ score_ckpt6
  ❌ score_ckpt2    ❌ score_ckpt7
  ❌ score_ckpt3    ❌ score_ckpt8
  ❌ score_ckpt4    ❌ score_ckpt9
  ❌ score_ckpt5    ❌ score_ckpt10

LEGACY PER-CHECKPOINT LATENCIES (10 columns):
  ❌ latency_ckpt1    ❌ latency_ckpt6
  ❌ latency_ckpt2    ❌ latency_ckpt7
  ❌ latency_ckpt3    ❌ latency_ckpt8
  ❌ latency_ckpt4    ❌ latency_ckpt9
  ❌ latency_ckpt5    ❌ latency_ckpt10

LEGACY BATCH COLUMNS (3 columns):
  ❌ student_output_batch
  ❌ latency_batch
  ❌ score_batch

──────────────────────────────────────────────────────────────────────

COLUMNS RETAINED (Essential for pipeline):
──────────────────────────────────────────────────────────────────────

CORE DATA:
  ✅ id                      Primary key
  ✅ input                   Instruction/prompt
  ✅ context                 Optional context
  ✅ sevenb                  Teacher output
  ✅ student_output          Base student output
  ✅ checkpoint              Checkpoint number (1-10)
  ✅ created_at              Record creation time
  ✅ updated_at              Last update time

WORKFLOW:
  ✅ status                  Pipeline status tracking

TUNED OUTPUT:
  ✅ student_output_tuned    Finetuned model output
  ✅ latency_tuned           Finetuned generation latency

BASE METRICS (before finetuning):
  ✅ score                   Overall score
  ✅ structured_correctness  Format accuracy
  ✅ task_success            Task completion
  ✅ instruction_following   Instruction adherence
  ✅ coverage                Information completeness
  ✅ faithfulness            Factual accuracy
  ✅ hallucination           Hallucination detection
  ✅ context_grounding       Context usage
  ✅ conciseness             Response length
  ✅ rouge1                  ROUGE-1 score
  ✅ rougel                  ROUGE-L score
  ✅ bleu                    BLEU score

TUNED METRICS (after finetuning):
  ✅ score_tuned
  ✅ structured_correctness_tuned
  ✅ task_success_tuned
  ✅ instruction_following_tuned
  ✅ coverage_tuned
  ✅ faithfulness_tuned
  ✅ hallucination_tuned
  ✅ context_grounding_tuned
  ✅ conciseness_tuned
  ✅ rouge1_tuned
  ✅ rougel_tuned
  ✅ bleu_tuned

IMPROVEMENT:
  ✅ improvement             Score delta

OPTIONAL:
  ✅ task_label              Task type for evaluation
  ✅ generation_latency      Base generation latency (may remove)
  ✅ worker_id               Worker tracking (may remove)
  ✅ overall_score           Duplicate of score (may remove)

*/
-- ============================================================================
