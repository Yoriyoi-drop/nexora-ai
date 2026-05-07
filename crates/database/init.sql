-- ==================== Nexora-AI PostgreSQL Schema ====================
-- Layer 4: Relational Database for User Management, Billing, Analytics

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";  -- For storing small embeddings if needed
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- For text similarity search

-- ==================== User Management ====================

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    api_key VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    tier_id INTEGER REFERENCES subscription_tiers(id) ON DELETE SET NULL
);

-- Subscription tiers
CREATE TABLE subscription_tiers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    monthly_token_quota BIGINT NOT NULL,
    daily_request_limit INTEGER NOT NULL,
    max_concurrent_requests INTEGER NOT NULL,
    features JSONB DEFAULT '{}',
    price_monthly DECIMAL(10, 2) NOT NULL
);

-- Insert default tiers
INSERT INTO subscription_tiers (name, monthly_token_quota, daily_request_limit, max_concurrent_requests, features, price_monthly) VALUES
('free', 100000, 100, 5, '{"models": ["nexora-1b"], "context_length": 4096}'::jsonb, 0.00),
('pro', 10000000, 1000, 20, '{"models": ["nexora-1b", "nexora-7b"], "context_length": 8192, "priority": true}'::jsonb, 29.99),
('enterprise', 1000000000, 10000, 100, '{"models": ["nexora-1b", "nexora-7b", "nexora-13b"], "context_length": 32768, "priority": true, "dedicated": true}'::jsonb, 299.99);

-- API keys (multiple per user)
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    scopes JSONB DEFAULT '[]'::jsonb
);

-- ==================== Billing and Usage ====================

-- Monthly usage tracking
CREATE TABLE monthly_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    tokens_used BIGINT DEFAULT 0,
    requests_made INTEGER DEFAULT 0,
    total_cost DECIMAL(10, 2) DEFAULT 0.00,
    billing_status VARCHAR(50) DEFAULT 'pending',  -- pending, paid, overdue
    UNIQUE(user_id, year, month)
);

-- Usage logs (detailed per request)
CREATE TABLE usage_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    model VARCHAR(50) NOT NULL,
    tokens_input INTEGER NOT NULL,
    tokens_output INTEGER NOT NULL,
    tokens_total INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,  -- success, error, rate_limited
    error_message TEXT,
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    endpoint VARCHAR(100) NOT NULL
);

-- Rate limiting buckets
CREATE TABLE rate_limits (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    request_count INTEGER DEFAULT 0,
    token_count BIGINT DEFAULT 0,
    PRIMARY KEY (user_id, window_start)
);

-- ==================== Evaluation and Benchmarking ====================

-- Checkpoints
CREATE TABLE checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    checkpoint_name VARCHAR(255) UNIQUE NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    step_number INTEGER NOT NULL,
    training_loss DECIMAL(10, 6),
    file_path VARCHAR(512) NOT NULL,
    file_size_bytes BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_deployed BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}'
);

-- Evaluation results
CREATE TABLE evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    checkpoint_id UUID NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    benchmark_name VARCHAR(100) NOT NULL,
    score DECIMAL(10, 6) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,  -- perplexity, accuracy, f1, etc.
    dataset_name VARCHAR(100) NOT NULL,
    evaluation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    additional_metrics JSONB DEFAULT '{}',
    notes TEXT
);

-- ==================== A/B Testing ====================

-- A/B test experiments
CREATE TABLE ab_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    start_date TIMESTAMP WITH TIME ZONE NOT NULL,
    end_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'planned',  -- planned, running, completed, cancelled
    control_checkpoint_id UUID REFERENCES checkpoints(id) ON DELETE SET NULL,
    treatment_checkpoint_id UUID REFERENCES checkpoints(id) ON DELETE SET NULL,
    traffic_split DECIMAL(3, 2) DEFAULT 0.5,  -- 0.0 to 1.0
    success_metric VARCHAR(100) NOT NULL,
    target_sample_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- A/B test assignments
CREATE TABLE ab_assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES ab_experiments(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    group_assignment VARCHAR(20) NOT NULL,  -- control, treatment
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_id, user_id)
);

-- A/B test results
CREATE TABLE ab_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES ab_experiments(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    group_assignment VARCHAR(20) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10, 6) NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ==================== Conversation History ====================

-- Conversations
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    model VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_archived BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}'
);

-- Messages
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- user, assistant, system
    content TEXT NOT NULL,
    tokens INTEGER,
    latency_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- ==================== Analytics ====================

-- Daily aggregated metrics
CREATE TABLE daily_metrics (
    date DATE NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20, 6) NOT NULL,
    dimensions JSONB DEFAULT '{}',
    PRIMARY KEY (date, metric_name, dimensions)
);

-- Model performance tracking
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(50) NOT NULL,
    checkpoint_id UUID REFERENCES checkpoints(id) ON DELETE SET NULL,
    date DATE NOT NULL,
    total_requests INTEGER DEFAULT 0,
    total_tokens BIGINT DEFAULT 0,
    avg_latency_ms DECIMAL(10, 2),
    error_rate DECIMAL(5, 4),
    p50_latency_ms DECIMAL(10, 2),
    p95_latency_ms DECIMAL(10, 2),
    p99_latency_ms DECIMAL(10, 2),
    UNIQUE(model_name, checkpoint_id, date)
);

-- ==================== Indexes for Performance ====================

-- User management indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_tier ON users(tier_id);
CREATE INDEX idx_api_keys_user ON api_keys(user_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- Usage indexes
CREATE INDEX idx_usage_logs_user ON usage_logs(user_id);
CREATE INDEX idx_usage_logs_timestamp ON usage_logs(request_timestamp);
CREATE INDEX idx_usage_logs_user_timestamp ON usage_logs(user_id, request_timestamp);
CREATE INDEX idx_monthly_usage_user ON monthly_usage(user_id);
CREATE INDEX idx_monthly_usage_date ON monthly_usage(year, month);

-- Evaluation indexes
CREATE INDEX idx_evaluations_checkpoint ON evaluations(checkpoint_id);
CREATE INDEX idx_evaluations_benchmark ON evaluations(benchmark_name);
CREATE INDEX idx_checkpoints_model ON checkpoints(model_name);
CREATE INDEX idx_checkpoints_deployed ON checkpoints(is_deployed);

-- A/B testing indexes
CREATE INDEX idx_ab_experiments_status ON ab_experiments(status);
CREATE INDEX idx_ab_assignments_experiment ON ab_assignments(experiment_id);
CREATE INDEX idx_ab_assignments_user ON ab_assignments(user_id);
CREATE INDEX idx_ab_results_experiment ON ab_results(experiment_id);

-- Conversation indexes
CREATE INDEX idx_conversations_user ON conversations(user_id);
CREATE INDEX idx_conversations_created ON conversations(created_at);
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);

-- Analytics indexes
CREATE INDEX idx_daily_metrics_date ON daily_metrics(date);
CREATE INDEX idx_daily_metrics_name ON daily_metrics(metric_name);
CREATE INDEX idx_model_performance_date ON model_performance(date);
CREATE INDEX idx_model_performance_model ON model_performance(model_name);

-- ==================== Triggers for Auto-Update ====================

-- Update updated_at timestamp on users
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==================== Views for Common Queries ====================

-- User with tier info
CREATE VIEW user_details AS
SELECT 
    u.id,
    u.email,
    u.username,
    u.created_at,
    u.is_active,
    u.is_verified,
    t.name as tier_name,
    t.monthly_token_quota,
    t.daily_request_limit,
    t.max_concurrent_requests,
    t.features
FROM users u
LEFT JOIN subscription_tiers t ON u.tier_id = t.id;

-- Current month usage per user
CREATE VIEW current_month_usage AS
SELECT 
    u.id as user_id,
    u.email,
    COALESCE(mu.tokens_used, 0) as tokens_used,
    COALESCE(mu.requests_made, 0) as requests_made,
    t.monthly_token_quota,
    CASE 
        WHEN t.monthly_token_quota IS NULL THEN 0
        ELSE ROUND((COALESCE(mu.tokens_used, 0)::FLOAT / t.monthly_token_quota) * 100, 2)
    END as quota_percentage
FROM users u
LEFT JOIN subscription_tiers t ON u.tier_id = t.id
LEFT JOIN monthly_usage mu ON u.id = mu.user_id 
    AND mu.year = EXTRACT(YEAR FROM CURRENT_DATE)
    AND mu.month = EXTRACT(MONTH FROM CURRENT_DATE);

-- Evaluation summary per checkpoint
CREATE VIEW checkpoint_eval_summary AS
SELECT 
    c.id as checkpoint_id,
    c.checkpoint_name,
    c.model_name,
    c.step_number,
    c.training_loss,
    COUNT(e.id) as num_evaluations,
    AVG(e.score) as avg_score,
    MIN(e.score) as min_score,
    MAX(e.score) as max_score
FROM checkpoints c
LEFT JOIN evaluations e ON c.id = e.checkpoint_id
GROUP BY c.id, c.checkpoint_name, c.model_name, c.step_number, c.training_loss;

-- ==================== Functions for Common Operations ====================

-- Function to check and update rate limit
CREATE OR REPLACE FUNCTION check_rate_limit(
    p_user_id UUID,
    p_tokens INTEGER
) RETURNS BOOLEAN AS $$
DECLARE
    v_limit INTEGER;
    v_used INTEGER;
    v_window_start TIMESTAMP WITH TIME ZONE;
BEGIN
    -- Get user's daily request limit
    SELECT t.daily_request_limit INTO v_limit
    FROM users u
    JOIN subscription_tiers t ON u.tier_id = t.id
    WHERE u.id = p_user_id;
    
    -- Calculate current window (start of day)
    v_window_start := DATE_TRUNC('day', CURRENT_TIMESTAMP);
    
    -- Get current usage
    SELECT COALESCE(request_count, 0) INTO v_used
    FROM rate_limits
    WHERE user_id = p_user_id AND window_start = v_window_start;
    
    -- Check if limit exceeded
    IF v_used >= v_limit THEN
        RETURN FALSE;
    END IF;
    
    -- Update or insert rate limit record
    INSERT INTO rate_limits (user_id, window_start, request_count, token_count)
    VALUES (p_user_id, v_window_start, 1, p_tokens)
    ON CONFLICT (user_id, window_start)
    DO UPDATE SET 
        request_count = rate_limits.request_count + 1,
        token_count = rate_limits.token_count + p_tokens;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to log usage
CREATE OR REPLACE FUNCTION log_usage(
    p_user_id UUID,
    p_api_key_id UUID,
    p_model VARCHAR,
    p_tokens_input INTEGER,
    p_tokens_output INTEGER,
    p_latency_ms INTEGER,
    p_status VARCHAR,
    p_error_message TEXT,
    p_endpoint VARCHAR
) RETURNS VOID AS $$
BEGIN
    INSERT INTO usage_logs (
        user_id, api_key_id, model, tokens_input, tokens_output,
        tokens_total, latency_ms, status, error_message, endpoint
    ) VALUES (
        p_user_id, p_api_key_id, p_model, p_tokens_input, p_tokens_output,
        p_tokens_input + p_tokens_output, p_latency_ms, p_status, p_error_message, p_endpoint
    );
    
    -- Update monthly usage
    INSERT INTO monthly_usage (user_id, year, month, tokens_used, requests_made)
    VALUES (
        p_user_id, 
        EXTRACT(YEAR FROM CURRENT_DATE)::INTEGER,
        EXTRACT(MONTH FROM CURRENT_DATE)::INTEGER,
        p_tokens_input + p_tokens_output,
        1
    )
    ON CONFLICT (user_id, year, month)
    DO UPDATE SET 
        tokens_used = monthly_usage.tokens_used + (p_tokens_input + p_tokens_output),
        requests_made = monthly_usage.requests_made + 1;
END;
$$ LANGUAGE plpgsql;
