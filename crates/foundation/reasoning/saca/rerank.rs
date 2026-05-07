//! Mathematical Reranking Engine
//! 
//! Phase 6 of SACA: Score and rank candidates using mathematical metrics
//! Implements multi-metric scoring with objective selection

use crate::saca::{types::*, config::*, error::*};
use nexora_core::async_executor::AsyncTaskExecutor;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Mathematical Reranking engine
pub struct RerankEngine {
    config: RerankConfig,
    executor: Arc<AsyncTaskExecutor>,
    metric_calculators: Vec<Arc<dyn MetricCalculator>>,
    normalizer: Arc<dyn ScoreNormalizer>,
}

impl RerankEngine {
    /// Create new Rerank engine
    pub fn new(config: RerankConfig) -> SACAResult<Self> {
        let executor = Arc::new(AsyncTaskExecutor::new(nexora_core::async_executor::ExecutorConfig::default()));
        
        let metric_calculators: Vec<Arc<dyn MetricCalculator>> = vec![
            Arc::new(CorrectnessCalculator::new()),
            Arc::new(PerformanceCalculator::new()),
            Arc::new(ReadabilityCalculator::new()),
            Arc::new(MaintainabilityCalculator::new()),
            Arc::new(TestCoverageCalculator::new()),
            Arc::new(DocumentationCalculator::new()),
        ];
        
        let normalizer: Arc<dyn ScoreNormalizer> = match config.normalization_method {
            NormalizationMethod::MinMax => Arc::new(MinMaxNormalizer::new()),
            NormalizationMethod::ZScore => Arc::new(ZScoreNormalizer::new()),
            NormalizationMethod::RobustScaling => Arc::new(RobustNormalizer::new()),
            NormalizationMethod::None => Arc::new(NoNormalizer::new()),
        };
        
        info!("Rerank Engine initialized with {:?} normalization", config.normalization_method);
        
        Ok(Self {
            config,
            executor,
            metric_calculators,
            normalizer,
        })
    }
    
    /// Rerank executed candidates and select the best solution
    pub async fn rerank(
        &self,
        executed_candidates: Vec<SACAExecutionResult>,
        context: &RepositoryContext,
    ) -> SACAResult<SACASolution> {
        debug!("Starting mathematical reranking for {} candidates", executed_candidates.len());
        
        if executed_candidates.is_empty() {
            return Err(SACAError::RerankError("No candidates to rerank".to_string()));
        }
        
        // Calculate metrics for all candidates
        let mut scored_candidates = Vec::new();
        for execution_result in &executed_candidates {
            let scored_candidate = self.calculate_candidate_score(execution_result, context).await?;
            scored_candidates.push(scored_candidate);
        }
        
        // Normalize scores if configured
        let normalized_candidates = self.normalize_scores(scored_candidates).await?;
        
        // Apply weighting and calculate final scores
        let final_candidates = self.apply_weighting(normalized_candidates).await?;
        
        // Sort by final score
        let mut sorted_candidates = final_candidates;
        sorted_candidates.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Select the best candidate
        let best_candidate = sorted_candidates.into_iter().next()
            .ok_or_else(|| SACAError::RerankError("No candidates after sorting".to_string()))?;
        
        // Create final solution
        let solution = self.create_solution(best_candidate, &executed_candidates, context).await?;
        
        info!("Reranking completed. Best solution score: {:.3}", solution.quality_score);
        Ok(solution)
    }
    
    /// Calculate comprehensive score for a single candidate
    async fn calculate_candidate_score(
        &self,
        execution_result: &SACAExecutionResult,
        context: &RepositoryContext,
    ) -> SACAResult<ScoredCandidate> {
        let mut metrics = std::collections::HashMap::new();
        
        // Calculate all metrics
        for calculator in &self.metric_calculators {
            let metric_name = calculator.metric_name();
            let score = calculator.calculate(execution_result, context)?;
            metrics.insert(metric_name, score);
        }
        
        Ok(ScoredCandidate {
            candidate_id: execution_result.candidate_id,
            execution_result: execution_result.clone(),
            metrics,
            final_score: 0.0, // Will be calculated after weighting
        })
    }
    
    /// Normalize scores across all candidates
    async fn normalize_scores(&self, candidates: Vec<ScoredCandidate>) -> SACAResult<Vec<ScoredCandidate>> {
        if candidates.is_empty() {
            return Ok(candidates);
        }
        
        // Collect all scores for each metric
        let mut all_metric_scores = std::collections::HashMap::new();
        
        for candidate in &candidates {
            for (metric_name, score) in &candidate.metrics {
                all_metric_scores.entry(metric_name.clone())
                    .or_insert_with(Vec::new)
                    .push(*score);
            }
        }
        
        // Normalize each metric
        let mut normalized_candidates = candidates;
        
        for candidate in &mut normalized_candidates {
            for (metric_name, score) in &mut candidate.metrics {
                if let Some(all_scores) = all_metric_scores.get(metric_name) {
                    *score = self.normalizer.normalize(*score, all_scores)?;
                }
            }
        }
        
        Ok(normalized_candidates)
    }
    
    /// Apply weighting to calculate final scores
    async fn apply_weighting(&self, candidates: Vec<ScoredCandidate>) -> SACAResult<Vec<ScoredCandidate>> {
        let mut weighted_candidates = candidates;
        
        for candidate in &mut weighted_candidates {
            let mut final_score = 0.0;
            
            // Apply weights from configuration
            final_score += candidate.metrics.get("correctness").unwrap_or(&0.0) * self.config.criteria.correctness_weight;
            final_score += candidate.metrics.get("performance").unwrap_or(&0.0) * self.config.criteria.performance_weight;
            final_score += candidate.metrics.get("readability").unwrap_or(&0.0) * self.config.criteria.readability_weight;
            final_score += candidate.metrics.get("maintainability").unwrap_or(&0.0) * self.config.criteria.maintainability_weight;
            final_score += candidate.metrics.get("test_coverage").unwrap_or(&0.0) * self.config.criteria.test_coverage_weight;
            final_score += candidate.metrics.get("documentation").unwrap_or(&0.0) * self.config.criteria.documentation_weight;
            
            candidate.final_score = final_score;
        }
        
        Ok(weighted_candidates)
    }
    
    /// Create final solution from best candidate
    async fn create_solution(
        &self,
        best_candidate: ScoredCandidate,
        all_executions: &[SACAExecutionResult],
        context: &RepositoryContext,
    ) -> SACAResult<SACASolution> {
        // Find the execution result for the best candidate
        let execution_result = all_executions
            .iter()
            .find(|r| r.candidate_id == best_candidate.candidate_id)
            .ok_or_else(|| SACAError::RerankError("Execution result not found".to_string()))?;
        
        // Calculate test coverage
        let test_coverage = self.calculate_test_coverage(execution_result).await;
        
        // Determine performance grade
        let performance_grade = self.determine_performance_grade(&best_candidate).await;
        
        // Create solved modules (simplified for now)
        let solved_modules = vec![SolvedModule {
            module: Module {
                id: "main".to_string(),
                name: "MainModule".to_string(),
                description: "Main implementation module".to_string(),
                inputs: vec![],
                outputs: vec![],
                dependencies: vec![],
                complexity: ModuleComplexity::Medium,
                estimated_lines: 100,
            },
            implementation: "// Final implementation".to_string(),
            executed_candidates: Some(vec![execution_result.clone()]),
            quality_metrics: ModuleQualityMetrics {
                correctness: *best_candidate.metrics.get("correctness").unwrap_or(&0.0),
                efficiency: *best_candidate.metrics.get("performance").unwrap_or(&0.0),
                readability: *best_candidate.metrics.get("readability").unwrap_or(&0.0),
                maintainability: *best_candidate.metrics.get("maintainability").unwrap_or(&0.0),
                test_coverage,
                documentation_score: *best_candidate.metrics.get("documentation").unwrap_or(&0.0),
            },
        }];
        
        Ok(SACASolution {
            session_id: uuid::Uuid::new_v4(),
            modules: solved_modules,
            quality_score: best_candidate.final_score,
            total_iterations: 1,
            total_feedback_loops: 0,
            execution_time: chrono::Duration::milliseconds(execution_result.execution_time_ms as i64),
            final_code: "// Final generated code".to_string(),
            test_coverage,
            performance_grade,
        })
    }
    
    /// Calculate test coverage percentage
    async fn calculate_test_coverage(&self, execution_result: &SACAExecutionResult) -> f32 {
        if execution_result.test_results.is_empty() {
            return 0.0;
        }
        
        let passed_tests = execution_result.test_results.iter().filter(|t| t.passed).count();
        let total_tests = execution_result.test_results.len();
        
        passed_tests as f32 / total_tests as f32
    }
    
    /// Determine performance grade based on metrics
    async fn determine_performance_grade(&self, candidate: &ScoredCandidate) -> PerformanceGrade {
        let score = candidate.final_score;
        
        if score >= 0.9 {
            PerformanceGrade::Excellent
        } else if score >= 0.75 {
            PerformanceGrade::Good
        } else if score >= 0.6 {
            PerformanceGrade::Average
        } else {
            PerformanceGrade::Poor
        }
    }
}

/// Scored candidate with metrics
#[derive(Debug, Clone)]
struct ScoredCandidate {
    candidate_id: uuid::Uuid,
    execution_result: SACAExecutionResult,
    metrics: std::collections::HashMap<String, f32>,
    final_score: f32,
}

/// Trait for metric calculators
trait MetricCalculator: Send + Sync {
    fn metric_name(&self) -> String;
    fn calculate(&self, execution_result: &SACAExecutionResult, context: &RepositoryContext) -> SACAResult<f32>;
}

/// Correctness metric calculator
struct CorrectnessCalculator {
    _private: (),
}

impl CorrectnessCalculator {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl MetricCalculator for CorrectnessCalculator {
    fn metric_name(&self) -> String {
        "correctness".to_string()
    }
    
    fn calculate(&self, execution_result: &SACAExecutionResult, _context: &RepositoryContext) -> SACAResult<f32> {
        let mut score = 0.0;
        
        // Base score for successful execution
        if execution_result.success {
            score += 0.5;
        }
        
        // Test results contribution
        if !execution_result.test_results.is_empty() {
            let passed_tests = execution_result.test_results.iter().filter(|t| t.passed).count();
            let total_tests = execution_result.test_results.len();
            score += (passed_tests as f32 / total_tests as f32) * 0.5;
        }
        
        Ok(score)
    }
}

/// Performance metric calculator
struct PerformanceCalculator {
    _private: (),
}

impl PerformanceCalculator {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl MetricCalculator for PerformanceCalculator {
    fn metric_name(&self) -> String {
        "performance".to_string()
    }
    
    fn calculate(&self, execution_result: &SACAExecutionResult, _context: &RepositoryContext) -> SACAResult<f32> {
        let mut score = 0.5; // Base score
        
        // Execution time scoring (lower is better)
        let time_score = if execution_result.execution_time_ms < 100 {
            1.0
        } else if execution_result.execution_time_ms < 1000 {
            0.8
        } else if execution_result.execution_time_ms < 5000 {
            0.6
        } else {
            0.4
        };
        
        // Memory usage scoring (lower is better)
        let memory_score = if execution_result.memory_usage_mb < 10.0 {
            1.0
        } else if execution_result.memory_usage_mb < 50.0 {
            0.8
        } else if execution_result.memory_usage_mb < 100.0 {
            0.6
        } else {
            0.4
        };
        
        score = (time_score + memory_score) / 2.0;
        Ok(score)
    }
}

/// Readability metric calculator
struct ReadabilityCalculator {
    _private: (),
}

impl ReadabilityCalculator {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl MetricCalculator for ReadabilityCalculator {
    fn metric_name(&self) -> String {
        "readability".to_string()
    }
    
    fn calculate(&self, execution_result: &SACAExecutionResult, _context: &RepositoryContext) -> SACAResult<f32> {
        // Simulate readability analysis
        // In a real implementation, this would analyze code complexity, naming, etc.
        let mut score: f32 = 0.7; // Base score
        
        // Penalize many error logs (indicates poor code quality)
        let error_penalty = (execution_result.error_logs.len() as f32 * 0.1).min(0.3);
        score -= error_penalty;
        
        Ok(score.max(0.0))
    }
}

/// Maintainability metric calculator
struct MaintainabilityCalculator {
    _private: (),
}

impl MaintainabilityCalculator {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl MetricCalculator for MaintainabilityCalculator {
    fn metric_name(&self) -> String {
        "maintainability".to_string()
    }
    
    fn calculate(&self, execution_result: &SACAExecutionResult, _context: &RepositoryContext) -> SACAResult<f32> {
        // Simulate maintainability analysis
        // In a real implementation, this would analyze modularity, coupling, etc.
        let mut score = 0.6; // Base score
        
        // Good performance metrics suggest efficient implementation
        if execution_result.performance_metrics.cpu_cycles > 0 {
            score += 0.1;
        }
        
        Ok((score as f64).min(1.0) as f32)
    }
}

/// Test coverage metric calculator
struct TestCoverageCalculator {
    _private: (),
}

impl TestCoverageCalculator {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl MetricCalculator for TestCoverageCalculator {
    fn metric_name(&self) -> String {
        "test_coverage".to_string()
    }
    
    fn calculate(&self, execution_result: &SACAExecutionResult, _context: &RepositoryContext) -> SACAResult<f32> {
        if execution_result.test_results.is_empty() {
            return Ok(0.0);
        }
        
        let passed_tests = execution_result.test_results.iter().filter(|t| t.passed).count();
        let total_tests = execution_result.test_results.len();
        
        Ok(passed_tests as f32 / total_tests as f32)
    }
}

/// Documentation metric calculator
struct DocumentationCalculator {
    _private: (),
}

impl DocumentationCalculator {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl MetricCalculator for DocumentationCalculator {
    fn metric_name(&self) -> String {
        "documentation".to_string()
    }
    
    fn calculate(&self, _execution_result: &SACAExecutionResult, _context: &RepositoryContext) -> SACAResult<f32> {
        // Simulate documentation analysis
        // In a real implementation, this would analyze comments, docs, etc.
        Ok(0.5) // Base score
    }
}

/// Trait for score normalizers
trait ScoreNormalizer: Send + Sync {
    fn normalize(&self, score: f32, all_scores: &[f32]) -> SACAResult<f32>;
}

/// Min-Max normalizer
struct MinMaxNormalizer {
    _private: (),
}

impl MinMaxNormalizer {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl ScoreNormalizer for MinMaxNormalizer {
    fn normalize(&self, score: f32, all_scores: &[f32]) -> SACAResult<f32> {
        if all_scores.is_empty() {
            return Ok(score);
        }
        
        let min_score = all_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_score = all_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_score == min_score {
            return Ok(0.5); // All scores are the same
        }
        
        Ok((score - min_score) / (max_score - min_score))
    }
}

/// Z-Score normalizer
struct ZScoreNormalizer {
    _private: (),
}

impl ZScoreNormalizer {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl ScoreNormalizer for ZScoreNormalizer {
    fn normalize(&self, score: f32, all_scores: &[f32]) -> SACAResult<f32> {
        if all_scores.is_empty() {
            return Ok(score);
        }
        
        let mean = all_scores.iter().sum::<f32>() / all_scores.len() as f32;
        let variance = all_scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / all_scores.len() as f32;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return Ok(0.5); // All scores are the same
        }
        
        let z_score = (score - mean) / std_dev;
        // Convert z-score to 0-1 range using sigmoid function
        Ok(1.0 / (1.0 + (-z_score).exp()))
    }
}

/// Robust scaling normalizer
struct RobustNormalizer {
    _private: (),
}

impl RobustNormalizer {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl ScoreNormalizer for RobustNormalizer {
    fn normalize(&self, score: f32, all_scores: &[f32]) -> SACAResult<f32> {
        if all_scores.is_empty() {
            return Ok(score);
        }
        
        let mut sorted_scores = all_scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let q1 = sorted_scores[sorted_scores.len() / 4];
        let q3 = sorted_scores[sorted_scores.len() * 3 / 4];
        let iqr = q3 - q1;
        
        if iqr == 0.0 {
            return Ok(0.5); // All scores are the same
        }
        
        let robust_score = (score - q1) / iqr;
        // Clamp to reasonable range and normalize to 0-1
        Ok((robust_score + 2.0).max(0.0).min(4.0) / 4.0)
    }
}

/// No normalizer (pass-through)
struct NoNormalizer {
    _private: (),
}

impl NoNormalizer {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl ScoreNormalizer for NoNormalizer {
    fn normalize(&self, score: f32, _all_scores: &[f32]) -> SACAResult<f32> {
        Ok(score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_rerank_engine() {
        let config = RerankConfig::default();
        let engine = RerankEngine::new(config).unwrap();
        
        let execution_result = SACAExecutionResult {
            candidate_id: uuid::Uuid::new_v4(),
            success: true,
            execution_time_ms: 100,
            memory_usage_mb: 10.0,
            error_logs: vec![],
            test_results: vec![
                TestResult {
                    test_name: "test1".to_string(),
                    passed: true,
                    execution_time_ms: 10,
                    error_message: None,
                },
            ],
            performance_metrics: PerformanceMetrics {
                time_complexity: "O(n)".to_string(),
                space_complexity: "O(n)".to_string(),
                cpu_cycles: 1000,
                cache_misses: 10,
                instructions: 500,
            },
        };
        
        let context = RepositoryContext::default();
        let solution = engine.rerank(vec![execution_result], &context).await.unwrap();
        
        assert!(solution.quality_score >= 0.0);
        assert!(solution.quality_score <= 1.0);
    }
    
    #[tokio::test]
    async fn test_correctness_calculator() {
        let calculator = CorrectnessCalculator::new();
        
        let execution_result = SACAExecutionResult {
            candidate_id: uuid::Uuid::new_v4(),
            success: true,
            execution_time_ms: 100,
            memory_usage_mb: 10.0,
            error_logs: vec![],
            test_results: vec![
                TestResult {
                    test_name: "test1".to_string(),
                    passed: true,
                    execution_time_ms: 10,
                    error_message: None,
                },
                TestResult {
                    test_name: "test2".to_string(),
                    passed: false,
                    execution_time_ms: 15,
                    error_message: Some("Failed".to_string()),
                },
            ],
            performance_metrics: PerformanceMetrics::default(),
        };
        
        let context = RepositoryContext::default();
        let score = calculator.calculate(&execution_result, &context).unwrap();
        
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }
    
    #[tokio::test]
    async fn test_minmax_normalizer() {
        let normalizer = MinMaxNormalizer::new();
        
        let scores = vec![0.2, 0.5, 0.8, 1.0];
        let normalized = normalizer.normalize(0.5, &scores).unwrap();
        
        assert_eq!(normalized, 0.375); // (0.5 - 0.2) / (1.0 - 0.2)
    }
}
