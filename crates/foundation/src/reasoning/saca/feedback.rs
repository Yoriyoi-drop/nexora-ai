//! Adaptive Feedback Loop System
//! 
//! Implements closed-loop feedback for SACA phases
//! Each phase provides signals to other phases for continuous improvement

use super::{types::*, config::*, error::*};
use super::config::FeedbackConfig;
use nexora_core::async_executor::AsyncTaskExecutor;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Feedback system for SACA
pub struct FeedbackSystem {
    config: FeedbackConfig,
    executor: Arc<AsyncTaskExecutor>,
    feedback_history: Arc<RwLock<std::collections::HashMap<uuid::Uuid, Vec<SACAFeedback>>>>,
    pattern_analyzer: Arc<PatternAnalyzer>,
}

impl FeedbackSystem {
    /// Create new Feedback system
    pub fn new(config: FeedbackConfig) -> SACAResult<Self> {
        let executor = Arc::new(AsyncTaskExecutor::new(nexora_core::async_executor::ExecutorConfig::default()));
        let pattern_analyzer = Arc::new(PatternAnalyzer::new());
        
        info!("Feedback System initialized with max {} loops", config.max_loops);
        
        Ok(Self {
            config,
            executor,
            feedback_history: Arc::new(RwLock::new(std::collections::HashMap::new())),
            pattern_analyzer,
        })
    }
    
    /// Generate feedback for solution improvement
    pub async fn generate_feedback(
        &self,
        solution: &SACASolution,
        context: &RepositoryContext,
    ) -> SACAResult<SACAFeedback> {
        debug!("Generating feedback for solution improvement");
        
        // Analyze solution quality
        let quality_analysis = self.analyze_solution_quality(solution).await?;
        
        // Identify issues
        let issues_identified = self.identify_issues(solution, &quality_analysis).await?;
        
        // Generate improvement suggestions
        let improvement_suggestions = self.generate_improvements(solution, &issues_identified).await?;
        
        // Determine new constraints
        let new_constraints = self.derive_constraints(solution, &issues_identified).await?;
        
        // Update requirements
        let updated_requirements = self.update_requirements(solution, &improvement_suggestions).await?;
        
        // Calculate confidence score
        let confidence_score = self.calculate_confidence(solution, &quality_analysis).await?;
        
        let feedback = SACAFeedback {
            feedback_id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            issues_identified,
            improvement_suggestions,
            new_constraints,
            updated_requirements,
            confidence_score,
        };
        
        // Store feedback in history
        self.store_feedback(solution.session_id, feedback.clone()).await?;
        
        info!("Feedback generated with confidence score: {:.3}", confidence_score);
        Ok(feedback)
    }
    
    /// Analyze solution quality
    async fn analyze_solution_quality(&self, solution: &SACASolution) -> SACAResult<QualityAnalysis> {
        let mut analysis = QualityAnalysis {
            correctness_gaps: Vec::new(),
            performance_issues: Vec::new(),
            readability_problems: Vec::new(),
            maintainability_concerns: Vec::new(),
            test_coverage_gaps: Vec::new(),
            documentation_deficits: Vec::new(),
        };
        
        // Analyze each module
        for module in &solution.modules {
            // Correctness analysis
            if module.quality_metrics.correctness < 0.8 {
                analysis.correctness_gaps.push(format!(
                    "Module {} has low correctness score: {:.2}",
                    module.module.name, module.quality_metrics.correctness
                ));
            }
            
            // Performance analysis
            if module.quality_metrics.efficiency < 0.7 {
                analysis.performance_issues.push(format!(
                    "Module {} has efficiency issues: {:.2}",
                    module.module.name, module.quality_metrics.efficiency
                ));
            }
            
            // Readability analysis
            if module.quality_metrics.readability < 0.6 {
                analysis.readability_problems.push(format!(
                    "Module {} has readability problems: {:.2}",
                    module.module.name, module.quality_metrics.readability
                ));
            }
            
            // Maintainability analysis
            if module.quality_metrics.maintainability < 0.7 {
                analysis.maintainability_concerns.push(format!(
                    "Module {} has maintainability concerns: {:.2}",
                    module.module.name, module.quality_metrics.maintainability
                ));
            }
            
            // Test coverage analysis
            if module.quality_metrics.test_coverage < 0.8 {
                analysis.test_coverage_gaps.push(format!(
                    "Module {} has insufficient test coverage: {:.2}",
                    module.module.name, module.quality_metrics.test_coverage
                ));
            }
            
            // Documentation analysis
            if module.quality_metrics.documentation_score < 0.6 {
                analysis.documentation_deficits.push(format!(
                    "Module {} lacks documentation: {:.2}",
                    module.module.name, module.quality_metrics.documentation_score
                ));
            }
        }
        
        Ok(analysis)
    }
    
    /// Identify specific issues in the solution
    async fn identify_issues(
        &self,
        solution: &SACASolution,
        quality_analysis: &QualityAnalysis,
    ) -> SACAResult<Vec<String>> {
        let mut issues = Vec::new();
        
        // Add quality analysis issues
        issues.extend(quality_analysis.correctness_gaps.clone());
        issues.extend(quality_analysis.performance_issues.clone());
        issues.extend(quality_analysis.readability_problems.clone());
        issues.extend(quality_analysis.maintainability_concerns.clone());
        issues.extend(quality_analysis.test_coverage_gaps.clone());
        issues.extend(quality_analysis.documentation_deficits.clone());
        
        // Add general solution issues
        if solution.quality_score < self.config.quality_threshold {
            issues.push(format!(
                "Overall solution quality {:.3} below threshold {:.3}",
                solution.quality_score, self.config.quality_threshold
            ));
        }
        
        if solution.total_feedback_loops >= self.config.max_loops {
            issues.push(format!(
                "Maximum feedback loops ({}) reached",
                self.config.max_loops
            ));
        }
        
        // Analyze execution results
        for module in &solution.modules {
            if let Some(executed_candidates) = &module.executed_candidates {
                if let Some(execution_result) = executed_candidates.first() {
                    if !execution_result.success {
                        issues.push(format!(
                            "Module {} execution failed: {}",
                            module.module.name,
                            execution_result.error_logs.join("; ")
                        ));
                    }
                    
                    if execution_result.execution_time_ms > 5000 {
                        issues.push(format!(
                            "Module {} has high execution time: {}ms",
                            module.module.name, execution_result.execution_time_ms
                        ));
                    }
                }
            } else {
                issues.push(format!(
                    "Module {} has no execution results",
                    module.module.name
                ));
            }
        }
        
        Ok(issues)
    }
    
    /// Generate improvement suggestions
    async fn generate_improvements(
        &self,
        solution: &SACASolution,
        issues: &[String],
    ) -> SACAResult<Vec<String>> {
        let mut suggestions = Vec::new();
        
        for issue in issues {
            if issue.contains("correctness") {
                suggestions.push("Implement additional test cases and edge case handling".to_string());
                suggestions.push("Review and fix logical errors in the implementation".to_string());
            } else if issue.contains("performance") {
                suggestions.push("Optimize algorithms and data structures".to_string());
                suggestions.push("Consider caching and memoization techniques".to_string());
            } else if issue.contains("readability") {
                suggestions.push("Improve code formatting and naming conventions".to_string());
                suggestions.push("Add meaningful comments and documentation".to_string());
            } else if issue.contains("maintainability") {
                suggestions.push("Refactor into smaller, more focused functions".to_string());
                suggestions.push("Reduce coupling between modules".to_string());
            } else if issue.contains("test coverage") {
                suggestions.push("Add comprehensive unit tests".to_string());
                suggestions.push("Include integration and edge case tests".to_string());
            } else if issue.contains("documentation") {
                suggestions.push("Add API documentation and usage examples".to_string());
                suggestions.push("Document algorithm complexity and trade-offs".to_string());
            } else if issue.contains("execution failed") {
                suggestions.push("Debug and fix runtime errors".to_string());
                suggestions.push("Add proper error handling and validation".to_string());
            } else if issue.contains("execution time") {
                suggestions.push("Profile and optimize performance bottlenecks".to_string());
                suggestions.push("Consider parallel processing where applicable".to_string());
            }
        }
        
        // Remove duplicates
        suggestions.sort();
        suggestions.dedup();
        
        Ok(suggestions)
    }
    
    /// Derive new constraints from issues
    async fn derive_constraints(
        &self,
        solution: &SACASolution,
        issues: &[String],
    ) -> SACAResult<Vec<String>> {
        let mut constraints = Vec::new();
        
        // Performance constraints
        if issues.iter().any(|i| i.contains("performance") || i.contains("execution time")) {
            constraints.push("Execution time must be under 1 second".to_string());
            constraints.push("Memory usage must be under 100MB".to_string());
        }
        
        // Quality constraints
        if issues.iter().any(|i| i.contains("correctness") || i.contains("test")) {
            constraints.push("All test cases must pass".to_string());
            constraints.push("Test coverage must be at least 90%".to_string());
        }
        
        // Code quality constraints
        if issues.iter().any(|i| i.contains("readability") || i.contains("maintainability")) {
            constraints.push("Code must follow style guidelines".to_string());
            constraints.push("Functions must be under 50 lines".to_string());
        }
        
        // Documentation constraints
        if issues.iter().any(|i| i.contains("documentation")) {
            constraints.push("All public functions must have documentation".to_string());
            constraints.push("Complex algorithms must have inline comments".to_string());
        }
        
        Ok(constraints)
    }
    
    /// Update requirements based on suggestions
    async fn update_requirements(
        &self,
        solution: &SACASolution,
        suggestions: &[String],
    ) -> SACAResult<Vec<String>> {
        let mut requirements = Vec::new();
        
        for suggestion in suggestions {
            if suggestion.contains("test") {
                requirements.push("Implement comprehensive testing strategy".to_string());
            } else if suggestion.contains("optimize") {
                requirements.push("Achieve optimal performance characteristics".to_string());
            } else if suggestion.contains("document") {
                requirements.push("Provide complete documentation".to_string());
            } else if suggestion.contains("refactor") {
                requirements.push("Maintain clean, modular architecture".to_string());
            } else if suggestion.contains("debug") {
                requirements.push("Ensure robust error handling".to_string());
            }
        }
        
        // Add general improvement requirements
        if solution.quality_score < self.config.quality_threshold {
            requirements.push(format!(
                "Achieve quality score of at least {:.3}",
                self.config.quality_threshold
            ));
        }
        
        // Remove duplicates
        requirements.sort();
        requirements.dedup();
        
        Ok(requirements)
    }
    
    /// Calculate confidence score for feedback
    async fn calculate_confidence(
        &self,
        solution: &SACASolution,
        quality_analysis: &QualityAnalysis,
    ) -> SACAResult<f32> {
        let mut confidence = 0.8; // Base confidence
        
        // Reduce confidence based on number of issues
        let total_issues = quality_analysis.correctness_gaps.len()
            + quality_analysis.performance_issues.len()
            + quality_analysis.readability_problems.len()
            + quality_analysis.maintainability_concerns.len()
            + quality_analysis.test_coverage_gaps.len()
            + quality_analysis.documentation_deficits.len();
        
        confidence -= (total_issues as f32 * 0.05).min(0.5);
        
        // Increase confidence if solution is close to threshold
        if solution.quality_score > self.config.quality_threshold - 0.1 {
            confidence += 0.1;
        }
        
        // Reduce confidence if many feedback loops have been used
        confidence -= (solution.total_feedback_loops as f32 * 0.02).min(0.2);
        
        Ok(confidence.max(0.1).min(1.0))
    }
    
    /// Store feedback in history
    async fn store_feedback(&self, session_id: uuid::Uuid, feedback: SACAFeedback) -> SACAResult<()> {
        let mut history = self.feedback_history.write().await;
        history.entry(session_id).or_insert_with(Vec::new).push(feedback);
        Ok(())
    }
    
    /// Get feedback history for a session
    pub async fn get_feedback_history(&self, session_id: uuid::Uuid) -> Vec<SACAFeedback> {
        self.feedback_history.read().await
            .get(&session_id)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Clear feedback history
    pub async fn clear_history(&self) {
        self.feedback_history.write().await.clear();
        info!("Feedback history cleared");
    }
}

/// Quality analysis result
#[derive(Debug, Clone)]
struct QualityAnalysis {
    correctness_gaps: Vec<String>,
    performance_issues: Vec<String>,
    readability_problems: Vec<String>,
    maintainability_concerns: Vec<String>,
    test_coverage_gaps: Vec<String>,
    documentation_deficits: Vec<String>,
}

/// Pattern analyzer for feedback
struct PatternAnalyzer {
    _private: (),
}

impl PatternAnalyzer {
    fn new() -> Self {
        Self { _private: () }
    }
    
    // In a real implementation, this would analyze patterns in feedback
    // to identify recurring issues and improvement opportunities
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_feedback_generation() -> anyhow::Result<()> {
        let config = FeedbackConfig::default();
        let system = FeedbackSystem::new(config)
            .map_err(|e| anyhow::anyhow!("Failed to create feedback system: {}", e))?;
        
        let solution = SACASolution {
            session_id: uuid::Uuid::new_v4(),
            modules: vec![],
            quality_score: 0.7,
            total_iterations: 1,
            total_feedback_loops: 0,
            execution_time: chrono::Duration::milliseconds(100),
            final_code: "test code".to_string(),
            test_coverage: 0.8,
            performance_grade: PerformanceGrade::Average,
        };
        
        let context = RepositoryContext::default();
        let feedback = system.generate_feedback(&solution, &context).await
            .map_err(|e| anyhow::anyhow!("Feedback generation failed: {}", e))?;
        
        assert!(!feedback.issues_identified.is_empty() || feedback.confidence_score > 0.8);
        assert!(feedback.confidence_score >= 0.0);
        assert!(feedback.confidence_score <= 1.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_quality_analysis() -> anyhow::Result<()> {
        let system = FeedbackSystem::new(FeedbackConfig::default())
            .map_err(|e| anyhow::anyhow!("Failed to create feedback system: {}", e))?;
        
        let solution = SACASolution {
            session_id: uuid::Uuid::new_v4(),
            modules: vec![SolvedModule {
                module: Module {
                    id: "test".to_string(),
                    name: "TestModule".to_string(),
                    description: "Test".to_string(),
                    inputs: vec![],
                    outputs: vec![],
                    dependencies: vec![],
                    complexity: ModuleComplexity::Medium,
                    estimated_lines: 100,
                },
                implementation: "test".to_string(),
                executed_candidates: Some(vec![SACAExecutionResult {
                    candidate_id: uuid::Uuid::new_v4(),
                    success: true,
                    execution_time_ms: 100,
                    memory_usage_mb: 10.0,
                    error_logs: vec![],
                    test_results: vec![],
                    performance_metrics: PerformanceMetrics::default(),
                    code_lines: Some(50),
                    generated_code: Some("test code".to_string()),
                }]),
                quality_metrics: ModuleQualityMetrics {
                    correctness: 0.5,
                    efficiency: 0.6,
                    readability: 0.4,
                    maintainability: 0.7,
                    test_coverage: 0.3,
                    documentation_score: 0.2,
                },
            }],
            quality_score: 0.5,
            total_iterations: 1,
            total_feedback_loops: 0,
            execution_time: chrono::Duration::milliseconds(100),
            final_code: "test".to_string(),
            test_coverage: 0.3,
            performance_grade: PerformanceGrade::Poor,
        };
        
        let analysis = system.analyze_solution_quality(&solution).await
            .map_err(|e| anyhow::anyhow!("Quality analysis failed: {}", e))?;
        
        assert!(!analysis.correctness_gaps.is_empty());
        assert!(!analysis.test_coverage_gaps.is_empty());
        assert!(!analysis.documentation_deficits.is_empty());
        
        Ok(())
    }
}
