//! Mathematical Reranking Engine
//! 
//! Phase 6 of SACA: Score and rank candidates using mathematical metrics
//! Implements multi-metric scoring with objective selection

use super::{types::*, config::*, error::*};
use crate::oracle::verifiers::performance::PerformanceThresholds;
use nexora_core::async_executor::AsyncTaskExecutor;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use rayon::prelude::*;

/// Mathematical Reranking engine
pub struct RerankEngine {
    config: RerankConfig,
    _executor: Arc<AsyncTaskExecutor>,
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
            _executor: executor,
            metric_calculators,
            normalizer,
        })
    }
    
    /// Rerank executed candidates and select the best solution with enhanced error handling
    pub async fn rerank(
        &self,
        executed_candidates: Vec<SACAExecutionResult>,
        context: &RepositoryContext,
    ) -> SACAResult<SACASolution> {
        debug!("Starting mathematical reranking for {} candidates", executed_candidates.len());
        
        // Enhanced input validation
        if executed_candidates.is_empty() {
            return Err(SACAError::RerankError(
                "No candidates to rerank. At least one execution result is required.".to_string()
            ));
        }
        
        if executed_candidates.len() > 1000 {
            warn!("Large number of candidates ({}). Consider filtering first.", executed_candidates.len());
        }
        
        // Validate context
        if context.repository_path.as_ref().map_or(true, |path| path.is_empty()) {
            warn!("Empty repository path in context");
        }
        
        // Calculate metrics for all candidates in parallel with error handling
        let scored_candidates = self.calculate_candidate_scores_with_fallback(&executed_candidates, context).await?;
        
        if scored_candidates.is_empty() {
            return Err(SACAError::RerankError(
                "All candidates failed scoring. Check execution results for errors.".to_string()
            ));
        }
        
        // Normalize scores if configured
        let normalized_candidates = self.normalize_scores_safe(scored_candidates).await?;
        
        // Apply weighting and calculate final scores
        let final_candidates = self.apply_weighting_robust(normalized_candidates).await?;
        
        // Sort by final score with stable sorting for reproducibility
        let mut sorted_candidates = final_candidates;
        sorted_candidates.sort_by(|a, b| {
            b.final_score.partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Select the best candidate with validation
        let best_candidate = sorted_candidates.into_iter().next()
            .ok_or_else(|| SACAError::RerankError(
                "No candidates after sorting. This should not happen.".to_string()
            ))?;
        
        // Validate best candidate
        if best_candidate.final_score < 0.0 || best_candidate.final_score > 1.0 {
            warn!("Best candidate has invalid score: {}", best_candidate.final_score);
        }
        
        // Create final solution with error handling
        let solution = self.create_solution_safe(best_candidate, &executed_candidates, context).await?;
        
        info!("Reranking completed successfully. Best solution score: {:.3}", solution.quality_score);
        Ok(solution)
    }
    
    /// Calculate scores with fallback error handling
    async fn calculate_candidate_scores_with_fallback(
        &self,
        executed_candidates: &[SACAExecutionResult],
        context: &RepositoryContext,
    ) -> SACAResult<Vec<ScoredCandidate>> {
        let mut successful_candidates = Vec::new();
        let mut failed_count = 0;
        
        // Process candidates sequentially for async compatibility
        let mut results = Vec::new();
        for (idx, execution_result) in executed_candidates.iter().enumerate() {
            match self.calculate_candidate_score(execution_result, context).await {
                Ok(candidate) => results.push(Some(candidate)),
                Err(e) => {
                    warn!("Failed to score candidate {}: {}", idx, e);
                    failed_count += 1;
                    results.push(None);
                }
            }
        }
        
        // Filter out failed candidates
        for candidate in results.into_iter().flatten() {
            successful_candidates.push(candidate);
        }
        
        if failed_count > 0 {
            warn!("Failed to score {} out of {} candidates", failed_count, executed_candidates.len());
        }
        
        if successful_candidates.is_empty() {
            return Err(SACAError::RerankError(
                format!("All {} candidates failed scoring", executed_candidates.len())
            ));
        }
        
        Ok(successful_candidates)
    }
    
    /// Safe normalization with error handling
    async fn normalize_scores_safe(&self, candidates: Vec<ScoredCandidate>) -> SACAResult<Vec<ScoredCandidate>> {
        if candidates.is_empty() {
            return Ok(candidates);
        }
        
        // Check for edge cases
        let mut all_metric_scores = std::collections::HashMap::new();
        
        for candidate in &candidates {
            for (metric_name, score) in &candidate.metrics {
                if !score.is_finite() {
                    warn!("Non-finite score detected for metric {}: {}", metric_name, score);
                    continue;
                }
                all_metric_scores.entry(metric_name.clone())
                    .or_insert_with(Vec::new)
                    .push(*score);
            }
        }
        
        if all_metric_scores.is_empty() {
            warn!("No valid metrics found for normalization");
            return Ok(candidates);
        }
        
        // Normalize each metric safely
        let mut normalized_candidates = candidates;
        
        for candidate in &mut normalized_candidates {
            for (metric_name, score) in &mut candidate.metrics {
                if let Some(all_scores) = all_metric_scores.get(metric_name) {
                    match self.normalizer.normalize(*score, all_scores) {
                        Ok(normalized_score) => {
                            if normalized_score.is_finite() {
                                *score = normalized_score;
                            } else {
                                warn!("Non-finite normalized score for metric {}: {}", metric_name, normalized_score);
                                *score = 0.5; // Fallback to neutral score
                            }
                        }
                        Err(e) => {
                            warn!("Failed to normalize metric {}: {}", metric_name, e);
                            *score = 0.5; // Fallback to neutral score
                        }
                    }
                }
            }
        }
        
        Ok(normalized_candidates)
    }
    
    /// Robust weighting application with validation
    async fn apply_weighting_robust(&self, candidates: Vec<ScoredCandidate>) -> SACAResult<Vec<ScoredCandidate>> {
        let mut weighted_candidates = candidates;
        
        // Validate configuration weights
        let total_weight = self.config.criteria.correctness_weight +
                          self.config.criteria.performance_weight +
                          self.config.criteria.readability_weight +
                          self.config.criteria.maintainability_weight +
                          self.config.criteria.test_coverage_weight +
                          self.config.criteria.documentation_weight;
        
        if total_weight <= 0.0 {
            return Err(SACAError::RerankError(
                "Invalid configuration: total weight must be positive".to_string()
            ));
        }
        
        for candidate in &mut weighted_candidates {
            let mut final_score = 0.0;
            
            // Apply weights with validation
            final_score += candidate.metrics.get("correctness").unwrap_or(&0.0) * self.config.criteria.correctness_weight;
            final_score += candidate.metrics.get("performance").unwrap_or(&0.0) * self.config.criteria.performance_weight;
            final_score += candidate.metrics.get("readability").unwrap_or(&0.0) * self.config.criteria.readability_weight;
            final_score += candidate.metrics.get("maintainability").unwrap_or(&0.0) * self.config.criteria.maintainability_weight;
            final_score += candidate.metrics.get("test_coverage").unwrap_or(&0.0) * self.config.criteria.test_coverage_weight;
            final_score += candidate.metrics.get("documentation").unwrap_or(&0.0) * self.config.criteria.documentation_weight;
            
            // Normalize by total weight
            final_score /= total_weight;
            
            // Clamp to valid range
            candidate.final_score = final_score.clamp(0.0, 1.0);
        }
        
        Ok(weighted_candidates)
    }
    
    /// Safe solution creation with comprehensive error handling
    async fn create_solution_safe(
        &self,
        best_candidate: ScoredCandidate,
        all_executions: &[SACAExecutionResult],
        context: &RepositoryContext,
    ) -> SACAResult<SACASolution> {
        // Find the execution result for the best candidate
        let execution_result = all_executions
            .iter()
            .find(|r| r.candidate_id == best_candidate.candidate_id)
            .ok_or_else(|| SACAError::RerankError(
                format!("Execution result not found for candidate {}", best_candidate.candidate_id)
            ))?;
        
        // Validate execution result
        if !execution_result.success {
            warn!("Best candidate was not successful");
        }
        
        // Calculate test coverage safely
        let test_coverage = self.calculate_test_coverage_safe(execution_result);
        
        // Determine performance grade with validation
        let performance_grade = self.determine_performance_grade_safe(&best_candidate);
        
        // Create solved modules with validation
        let solved_modules = vec![SolvedModule {
            module: Module {
                id: "main".to_string(),
                name: "MainModule".to_string(),
                description: "Main implementation module".to_string(),
                inputs: vec![],
                outputs: vec![],
                dependencies: vec![],
                complexity: ModuleComplexity::Medium,
                estimated_lines: execution_result.code_lines.unwrap_or(100) as u32,
            },
            implementation: execution_result.generated_code.clone()
                .unwrap_or_else(|| "// Final implementation".to_string()),
            executed_candidates: Some(vec![execution_result.clone()]),
            quality_metrics: ModuleQualityMetrics {
                correctness: best_candidate.metrics.get("correctness").copied().unwrap_or(0.0),
                efficiency: best_candidate.metrics.get("performance").copied().unwrap_or(0.0),
                readability: best_candidate.metrics.get("readability").copied().unwrap_or(0.0),
                maintainability: best_candidate.metrics.get("maintainability").copied().unwrap_or(0.0),
                test_coverage,
                documentation_score: best_candidate.metrics.get("documentation").copied().unwrap_or(0.0),
            },
        }];
        
        Ok(SACASolution {
            session_id: uuid::Uuid::new_v4(),
            modules: solved_modules,
            quality_score: best_candidate.final_score,
            total_iterations: 1,
            total_feedback_loops: 0,
            execution_time: chrono::Duration::milliseconds(execution_result.execution_time_ms as i64),
            final_code: execution_result.generated_code.clone()
                .unwrap_or_else(|| "// Final generated code".to_string()),
            test_coverage,
            performance_grade,
        })
    }
    
    /// Safe test coverage calculation
    fn calculate_test_coverage_safe(&self, _execution_result: &SACAExecutionResult) -> f32 {
        if _execution_result.test_results.is_empty() {
            return 0.0;
        }
        
        let passed_tests = _execution_result.test_results.iter().filter(|t| t.passed).count();
        let total_tests = _execution_result.test_results.len();
        
        let coverage = passed_tests as f32 / total_tests as f32;
        coverage.clamp(0.0, 1.0)
    }
    
    /// Safe performance grade determination
    fn determine_performance_grade_safe(&self, candidate: &ScoredCandidate) -> PerformanceGrade {
        let score = candidate.final_score;
        
        if !score.is_finite() {
            warn!("Non-finite score for performance grade: {}", score);
            return PerformanceGrade::Poor;
        }
        
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
            _execution_result: execution_result.clone(),
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
    async fn calculate_test_coverage(&self, _execution_result: &SACAExecutionResult) -> f32 {
        if _execution_result.test_results.is_empty() {
            return 0.0;
        }
        
        let passed_tests = _execution_result.test_results.iter().filter(|t| t.passed).count();
        let total_tests = _execution_result.test_results.len();
        
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
    _execution_result: SACAExecutionResult,
    metrics: std::collections::HashMap<String, f32>,
    final_score: f32,
}

/// Trait for metric calculators
trait MetricCalculator: Send + Sync {
    fn metric_name(&self) -> String;
    fn calculate(&self, _execution_result: &SACAExecutionResult, context: &RepositoryContext) -> SACAResult<f32>;
}

/// Base metric calculator to reduce code duplication
struct BaseMetricCalculator {
    _private: (),
}

impl BaseMetricCalculator {
    fn new() -> Self {
        Self { _private: () }
    }
    
    /// Helper method to calculate test pass rate
    fn calculate_pass_rate(test_results: &[TestResult]) -> f32 {
        if test_results.is_empty() {
            return 0.0;
        }
        
        let passed_tests = test_results.iter().filter(|t| t.passed).count();
        passed_tests as f32 / test_results.len() as f32
    }
}

/// Correctness metric calculator
struct CorrectnessCalculator {
    _base: BaseMetricCalculator,
}

impl CorrectnessCalculator {
    fn new() -> Self {
        Self { _base: BaseMetricCalculator::new() }
    }
}

impl MetricCalculator for CorrectnessCalculator {
    fn metric_name(&self) -> String {
        "correctness".to_string()
    }
    
    fn calculate(&self, _execution_result: &SACAExecutionResult, _context: &RepositoryContext) -> SACAResult<f32> {
        if !_execution_result.success {
            return Ok(0.0);
        }
        
        if _execution_result.test_results.is_empty() {
            return Ok(0.5); // Neutral score if no tests
        }
        
        Ok(BaseMetricCalculator::calculate_pass_rate(&_execution_result.test_results))
    }
}

/// Performance metric calculator with configurable thresholds
struct PerformanceCalculator {
    _base: BaseMetricCalculator,
    thresholds: PerformanceThresholds,
}

impl PerformanceCalculator {
    fn new() -> Self {
        Self { 
            _base: BaseMetricCalculator::new(),
            thresholds: PerformanceThresholds::default(),
        }
    }
    
    fn _with_thresholds(thresholds: PerformanceThresholds) -> Self {
        Self { 
            _base: BaseMetricCalculator::new(),
            thresholds,
        }
    }
}

impl MetricCalculator for PerformanceCalculator {
    fn metric_name(&self) -> String {
        "performance".to_string()
    }
    
    fn calculate(&self, _execution_result: &SACAExecutionResult, _context: &RepositoryContext) -> SACAResult<f32> {
        // Score based on execution time (lower is better)
        let time_score = self.thresholds.get_time_score(_execution_result.execution_time_ms);
        
        // Memory usage score
        let memory_score = self.thresholds.get_memory_score(_execution_result.memory_usage_mb as f32);
        
        Ok((time_score + memory_score) / 2.0)
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
    
    fn calculate(&self, _execution_result: &SACAExecutionResult, _context: &RepositoryContext) -> SACAResult<f32> {
        // Simple readability scoring based on error logs
        // Fewer errors = better readability
        let error_count = _execution_result.error_logs.len();
        let readability_score = if error_count == 0 {
            1.0
        } else if error_count <= 2 {
            0.8
        } else if error_count <= 5 {
            0.6
        } else if error_count <= 10 {
            0.4
        } else {
            0.2
        };
        
        Ok(readability_score)
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
    
    fn calculate(&self, _execution_result: &SACAExecutionResult, _context: &RepositoryContext) -> SACAResult<f32> {
        // Score based on performance metrics complexity
        let metrics = &_execution_result.performance_metrics;
        
        // Simple complexity scoring
        let time_complexity_score = match metrics.time_complexity.as_str() {
            "O(1)" => 1.0,
            "O(log n)" => 0.9,
            "O(n)" => 0.8,
            "O(n log n)" => 0.7,
            "O(n²)" => 0.5,
            "O(n³)" => 0.3,
            _ => 0.2,
        };
        
        let space_complexity_score = match metrics.space_complexity.as_str() {
            "O(1)" => 1.0,
            "O(log n)" => 0.9,
            "O(n)" => 0.8,
            "O(n log n)" => 0.7,
            "O(n²)" => 0.5,
            "O(n³)" => 0.3,
            _ => 0.2,
        };
        
        Ok((time_complexity_score + space_complexity_score) / 2.0)
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
    
    fn calculate(&self, _execution_result: &SACAExecutionResult, _context: &RepositoryContext) -> SACAResult<f32> {
        // Score based on number and quality of tests
        let test_count = _execution_result.test_results.len();
        
        if test_count == 0 {
            return Ok(0.0);
        }
        
        let passed_tests = _execution_result.test_results.iter()
            .filter(|t| t.passed)
            .count();
        
        let pass_rate = passed_tests as f32 / test_count as f32;
        
        // Bonus for having more tests
        let quantity_bonus = if test_count >= 10 {
            1.0
        } else if test_count >= 5 {
            0.8
        } else if test_count >= 3 {
            0.6
        } else {
            0.4
        };
        
        Ok(pass_rate * quantity_bonus)
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
        // Simple documentation scoring based on error message quality
        // Well-documented code should have meaningful error messages
        let has_meaningful_errors = _execution_result.error_logs.iter()
            .any(|error| error.len() > 10); // Errors with some detail
        
        let doc_score = if _execution_result.error_logs.is_empty() {
            0.8 // Good, but could have more documentation
        } else if has_meaningful_errors {
            0.6 // Some documentation present
        } else {
            0.3 // Poor documentation
        };
        
        Ok(doc_score)
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
    async fn test_rerank_engine() -> Result<(), anyhow::Error> {
        let config = RerankConfig::default();
        let engine = RerankEngine::new(config)
            .map_err(|e| anyhow::anyhow!("Failed to create rerank engine: {}", e))?;
        
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
            code_lines: Some(50),
            generated_code: Some("test code".to_string()),
        };
        
        let context = RepositoryContext::default();
        let solution = engine.rerank(vec![execution_result], &context).await
            .map_err(|e| anyhow::anyhow!("Reranking failed: {}", e))?;
        
        assert!(solution.quality_score >= 0.0);
        assert!(solution.quality_score <= 1.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_correctness_calculator() -> Result<(), anyhow::Error> {
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
            code_lines: Some(60),
            generated_code: Some("test code 2".to_string()),
        };
        
        let context = RepositoryContext::default();
        let score = calculator.calculate(&execution_result, &context)
            .map_err(|e| anyhow::anyhow!("Score calculation failed: {}", e))?;
        
        assert!(score >= 0.0);
        assert!(score <= 1.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_minmax_normalizer() -> Result<(), anyhow::Error> {
        let normalizer = MinMaxNormalizer::new();
        
        let scores = vec![0.2, 0.5, 0.8, 1.0];
        let normalized = normalizer.normalize(0.5, &scores)
            .map_err(|e| anyhow::anyhow!("Normalization failed: {}", e))?;
        
        assert_eq!(normalized, 0.375); // (0.5 - 0.2) / (1.0 - 0.2)
        
        Ok(())
    }
}
