-- ============================================================================
-- Step 08: New Evaluation Matrix Columns for modelComp
-- ============================================================================
-- Run this in Supabase SQL Editor
--
-- This script:
-- 1. Drops old metric columns (answer_relevancy, contextual_*, etc.)
-- 2. Creates new columns based on the 7-metric evaluation matrix
-- ============================================================================

-- ============================================================================
-- PART 1: DROP OLD ALPACA/OSS METRIC COLUMNS
-- ============================================================================
-- (These are the old columns that were created in add_modelcomp_metrics.sql)

-- Drop Alpaca old metrics
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS alpaca_answer_relevancy;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS alpaca_contextual_precision;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS alpaca_contextual_recall;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS alpaca_contextual_relevancy;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS alpaca_faithfulness;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS alpaca_toxicity;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS alpaca_hallucination_rate;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS alpaca_overall;

-- Drop OSS-20B old metrics
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS oss20b_answer_relevancy;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS oss20b_contextual_precision;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS oss20b_contextual_recall;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS oss20b_contextual_relevancy;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS oss20b_faithfulness;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS oss20b_toxicity;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS oss20b_hallucination_rate;
ALTER TABLE "modelComp" DROP COLUMN IF EXISTS oss20b_overall;

-- ============================================================================
-- PART 2: CREATE NEW EVALUATION MATRIX COLUMNS
-- ============================================================================
-- All metrics stored as INT (value * 10000 for 4 decimal precision)
-- 
-- Metrics (7 core + 1 overall = 8 per model):
-- 1. struct_correct    - Structured Correctness (JSON/code validity)
-- 2. task_success      - Task Success Score (heuristic-based)
-- 3. instr_follow      - Instruction Following Score
-- 4. coverage          - Coverage/Completeness vs reference
-- 5. faithfulness      - Faithfulness to context/reference
-- 6. hallucination     - Hallucination Rate (lower is better)
-- 7. ctx_grounding     - Context Grounding Ratio
-- 8. overall           - Overall Score (composite)

-- ALPACA MODEL METRICS
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_struct_correct INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_task_success INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_instr_follow INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_coverage INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_faithfulness INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_hallucination INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_ctx_grounding INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS alpaca_overall INT DEFAULT 0;

-- OSS-20B MODEL METRICS
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss_struct_correct INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss_task_success INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss_instr_follow INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss_coverage INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss_faithfulness INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss_hallucination INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss_ctx_grounding INT DEFAULT 0;
ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS oss_overall INT DEFAULT 0;

-- ============================================================================
-- PART 3: ADD EVALUATION STATUS COLUMN
-- ============================================================================
-- Track which records have been evaluated with new metrics

ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS eval_status VARCHAR(20) DEFAULT 'pending';

-- Values:
-- 'pending'  - Not yet evaluated
-- 'done'     - Evaluation complete
-- 'error'    - Evaluation failed

-- ============================================================================
-- PART 4: ADD WINNER COLUMN
-- ============================================================================
-- Store which model won for this record

ALTER TABLE "modelComp" ADD COLUMN IF NOT EXISTS eval_winner VARCHAR(20);

-- Values:
-- 'alpaca'   - Alpaca scored higher
-- 'oss'      - OSS-20B scored higher
-- 'tie'      - Equal scores

-- ============================================================================
-- PART 5: VERIFY COLUMNS
-- ============================================================================

SELECT column_name, data_type, column_default
FROM information_schema.columns 
WHERE table_name = 'modelComp' 
  AND column_name LIKE '%alpaca%' OR column_name LIKE '%oss%' 
  OR column_name = 'eval_status' OR column_name = 'eval_winner'
ORDER BY column_name;

-- ============================================================================
-- PART 6: CREATE DECIMAL VIEW FOR EASIER QUERYING
-- ============================================================================

CREATE OR REPLACE VIEW modelcomp_eval_view AS
SELECT 
    id,
    input,
    label,
    
    -- Alpaca metrics (converted back to decimal)
    alpaca_struct_correct / 10000.0 AS alpaca_struct_correct_dec,
    alpaca_task_success / 10000.0 AS alpaca_task_success_dec,
    alpaca_instr_follow / 10000.0 AS alpaca_instr_follow_dec,
    alpaca_coverage / 10000.0 AS alpaca_coverage_dec,
    alpaca_faithfulness / 10000.0 AS alpaca_faithfulness_dec,
    alpaca_hallucination / 10000.0 AS alpaca_hallucination_dec,
    alpaca_ctx_grounding / 10000.0 AS alpaca_ctx_grounding_dec,
    alpaca_overall / 10000.0 AS alpaca_overall_dec,
    
    -- OSS metrics (converted back to decimal)
    oss_struct_correct / 10000.0 AS oss_struct_correct_dec,
    oss_task_success / 10000.0 AS oss_task_success_dec,
    oss_instr_follow / 10000.0 AS oss_instr_follow_dec,
    oss_coverage / 10000.0 AS oss_coverage_dec,
    oss_faithfulness / 10000.0 AS oss_faithfulness_dec,
    oss_hallucination / 10000.0 AS oss_hallucination_dec,
    oss_ctx_grounding / 10000.0 AS oss_ctx_grounding_dec,
    oss_overall / 10000.0 AS oss_overall_dec,
    
    -- Winner
    eval_winner,
    eval_status,
    
    -- Difference (alpaca - oss)
    (alpaca_overall - oss_overall) / 10000.0 AS overall_difference
    
FROM "modelComp"
WHERE eval_status = 'done';

-- ============================================================================
-- PART 7: CREATE SUMMARY VIEW BY CATEGORY
-- ============================================================================

CREATE OR REPLACE VIEW modelcomp_category_summary AS
SELECT 
    label,
    COUNT(*) AS total_records,
    SUM(CASE WHEN eval_winner = 'alpaca' THEN 1 ELSE 0 END) AS alpaca_wins,
    SUM(CASE WHEN eval_winner = 'oss' THEN 1 ELSE 0 END) AS oss_wins,
    SUM(CASE WHEN eval_winner = 'tie' THEN 1 ELSE 0 END) AS ties,
    
    -- Average scores
    AVG(alpaca_overall / 10000.0) AS alpaca_avg_overall,
    AVG(oss_overall / 10000.0) AS oss_avg_overall,
    
    AVG(alpaca_task_success / 10000.0) AS alpaca_avg_task_success,
    AVG(oss_task_success / 10000.0) AS oss_avg_task_success,
    
    AVG(alpaca_faithfulness / 10000.0) AS alpaca_avg_faithfulness,
    AVG(oss_faithfulness / 10000.0) AS oss_avg_faithfulness,
    
    AVG(alpaca_hallucination / 10000.0) AS alpaca_avg_hallucination,
    AVG(oss_hallucination / 10000.0) AS oss_avg_hallucination
    
FROM "modelComp"
WHERE eval_status = 'done' AND label IS NOT NULL
GROUP BY label
ORDER BY label;

-- ============================================================================
-- QUERIES TO CHECK RESULTS
-- ============================================================================

-- Overall winner count:
-- SELECT eval_winner, COUNT(*) FROM "modelComp" WHERE eval_status = 'done' GROUP BY eval_winner;

-- Per-category summary:
-- SELECT * FROM modelcomp_category_summary;

-- Top 10 biggest alpaca wins:
-- SELECT id, label, alpaca_overall/10000.0, oss_overall/10000.0, (alpaca_overall-oss_overall)/10000.0 as diff 
-- FROM "modelComp" WHERE eval_status = 'done' ORDER BY diff DESC LIMIT 10;

-- Top 10 biggest OSS wins:
-- SELECT id, label, alpaca_overall/10000.0, oss_overall/10000.0, (alpaca_overall-oss_overall)/10000.0 as diff 
-- FROM "modelComp" WHERE eval_status = 'done' ORDER BY diff ASC LIMIT 10;
