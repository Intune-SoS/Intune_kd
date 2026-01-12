-- ============================================================================
-- Step 10: Create 50K Table for Incremental Learning
-- ============================================================================
-- Run this in Supabase SQL Editor to create the new table
-- NOTE: PostgreSQL is case-insensitive, using lowercase for consistency

-- Main table for 50K incremental learning experiment
CREATE TABLE IF NOT EXISTS modelcomp_50k (
    id BIGSERIAL PRIMARY KEY,
    
    -- Input data
    input TEXT NOT NULL,
    context TEXT,
    
    -- Teacher output (Alpaca - the winner)
    sevenb TEXT,                        -- Alpaca teacher output
    
    -- Student output (generated per stage)
    student_output TEXT,                -- Generated student output
    generation_latency FLOAT,           -- Time to generate (seconds)
    
    -- Stage tracking (renamed from checkpoint for clarity)
    checkpoint INT,                     -- Which 5K stage (1-10)
    
    -- Evaluation metrics (INT8 format: multiply by 10000)
    structured_correctness INT,
    task_success INT,
    instruction_following INT,
    coverage INT,
    faithfulness INT,
    hallucination INT,
    context_grounding INT,
    overall_score INT,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_modelcomp_50k_checkpoint 
    ON modelcomp_50k(checkpoint);

CREATE INDEX IF NOT EXISTS idx_modelcomp_50k_pending 
    ON modelcomp_50k(student_output) 
    WHERE student_output IS NULL;

CREATE INDEX IF NOT EXISTS idx_modelcomp_50k_created 
    ON modelcomp_50k(created_at);

-- ============================================================================
-- Incremental Learning Results Table
-- ============================================================================
-- Stores checkpoint evaluation results for learning curve analysis

CREATE TABLE IF NOT EXISTS incremental_learning_results (
    id SERIAL PRIMARY KEY,
    
    -- Checkpoint info
    checkpoint INT NOT NULL,
    data_size INT NOT NULL,
    
    -- Training metrics
    train_loss FLOAT,
    eval_loss FLOAT,
    train_time_seconds FLOAT,
    
    -- Evaluation metrics (averaged)
    structured_correctness FLOAT,
    task_success FLOAT,
    instruction_following FLOAT,
    coverage FLOAT,
    faithfulness FLOAT,
    hallucination FLOAT,
    context_grounding FLOAT,
    overall_score FLOAT,
    
    -- Metadata
    eval_samples INT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(checkpoint)
);

-- ============================================================================
-- View: Learning Curve Summary
-- ============================================================================
-- Easy view for plotting learning curves

CREATE OR REPLACE VIEW learning_curve AS
SELECT 
    checkpoint,
    data_size,
    faithfulness,
    hallucination,
    coverage,
    context_grounding,
    overall_score,
    train_loss,
    eval_loss,
    
    -- Calculate improvement from previous checkpoint
    faithfulness - LAG(faithfulness) OVER (ORDER BY checkpoint) AS faith_improvement,
    LAG(hallucination) OVER (ORDER BY checkpoint) - hallucination AS halluc_improvement,
    overall_score - LAG(overall_score) OVER (ORDER BY checkpoint) AS overall_improvement
    
FROM incremental_learning_results
ORDER BY checkpoint;

-- ============================================================================
-- Function: Update timestamp trigger
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_modelcomp_50k_updated_at
    BEFORE UPDATE ON modelcomp_50k
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- Sample Queries
-- ============================================================================

-- Get stage statistics
-- SELECT 
--     checkpoint as stage,
--     COUNT(*) as records,
--     AVG(faithfulness::float / 10000) as avg_faithfulness,
--     AVG(hallucination::float / 10000) as avg_hallucination,
--     AVG(overall_score::float / 10000) as avg_overall
-- FROM modelcomp_50k
-- WHERE checkpoint IS NOT NULL
-- GROUP BY checkpoint
-- ORDER BY checkpoint;

-- Get learning curve data
-- SELECT * FROM learning_curve;

-- Check pending records
-- SELECT checkpoint as stage, COUNT(*) as pending
-- FROM modelcomp_50k
-- WHERE student_output IS NULL
-- GROUP BY checkpoint;
