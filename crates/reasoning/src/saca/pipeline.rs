//! SACA Pipeline Orchestration
//! 
//! Manages the complete 6-phase pipeline execution
//! Handles phase transitions, error recovery, and progress tracking

use super::{types::*, config::*, error::*, prelude::*};
use nexora_core::async_executor::AsyncTaskExecutor;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use chrono::Utc;

/// Pipeline orchestrator for SACA
pub struct SACAPipeline {
    config: SACAConfig,
    _executor: Arc<AsyncTaskExecutor>,
    
    // Phase engines
    cot_engine: Arc<CoTEngine>,
    decompose_engine: Arc<DecomposeEngine>,
    context_engine: Arc<ContextEngine>,
    sampling_engine: Arc<SamplingEngine>,
    execute_engine: Arc<ExecuteEngine>,
    rerank_engine: Arc<RerankEngine>,
    
    // Feedback system
    feedback_system: Arc<FeedbackSystem>,
    
    // Pipeline state
    current_session: Arc<RwLock<Option<PipelineSession>>>,
    phase_metrics: Arc<RwLock<std::collections::HashMap<SACAPhase, PhaseMetrics>>>,
}

impl SACAPipeline {
    /// Create new pipeline instance
    pub async fn new(config: SACAConfig) -> SACAResult<Self> {
        let executor = Arc::new(AsyncTaskExecutor::new(nexora_core::async_executor::ExecutorConfig::default()));
        
        // Initialize phase engines
        let cot_engine = Arc::new(CoTEngine::new(config.cot_config.clone())?);
        let decompose_engine = Arc::new(DecomposeEngine::new(config.decompose_config.clone())?);
        let context_engine = Arc::new(ContextEngine::new(config.context_config.clone())?);
        let sampling_engine = Arc::new(SamplingEngine::new(config.sampling_config.clone())?);
        let execute_engine = Arc::new(ExecuteEngine::new(config.execute_config.clone())?);
        let rerank_engine = Arc::new(RerankEngine::new(config.rerank_config.clone())?);
        
        // Initialize feedback system
        let feedback_system = Arc::new(FeedbackSystem::new(config.feedback_config.clone())?);
        
        info!("SACA Pipeline initialized with 6 phases");
        
        Ok(Self {
            config,
            _executor: executor,
            cot_engine,
            decompose_engine,
            context_engine,
            sampling_engine,
            execute_engine,
            rerank_engine,
            feedback_system,
            current_session: Arc::new(RwLock::new(None)),
            phase_metrics: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }
    
    /// Execute complete SACA pipeline
    pub async fn execute(&self, task: CodingTask) -> SACAResult<SACASolution> {
        let session_id = uuid::Uuid::new_v4();
        let session = PipelineSession {
            id: session_id,
            task: task.clone(),
            start_time: Utc::now(),
            current_phase: SACAPhase::Think,
            iterations: 0,
            feedback_loops: 0,
            phase_history: Vec::new(),
        };
        
        *self.current_session.write().await = Some(session.clone());
        
        info!("Starting SACA pipeline for task: {} [Session: {}]", task.description, session_id);
        
        // Execute pipeline with feedback loops
        let solution = self.execute_with_feedback_loops(task, session).await?;
        
        // Update final metrics
        self.update_final_metrics(&solution).await?;
        
        info!("SACA pipeline completed. Quality: {:.3}, Iterations: {}, Feedback loops: {}", 
              solution.quality_score, solution.total_iterations, solution.total_feedback_loops);
        
        Ok(solution)
    }
    
    /// Execute pipeline with adaptive feedback loops
    async fn execute_with_feedback_loops(&self, mut task: CodingTask, mut session: PipelineSession) -> SACAResult<SACASolution> {
        loop {
            session.iterations += 1;
            debug!("Starting pipeline iteration {}", session.iterations);
            
            // Execute the 6-phase pipeline
            let solution = self.execute_single_iteration(&task, &mut session).await?;
            
            // Check if solution meets quality threshold
            if solution.quality_score >= self.config.quality_threshold {
                info!("Solution meets quality threshold {:.3}", solution.quality_score);
                return Ok(solution);
            }
            
            // Check if we've reached max feedback loops
            if session.feedback_loops >= self.config.max_feedback_loops {
                warn!("Max feedback loops reached. Returning best solution");
                return Ok(solution);
            }
            
            // Generate feedback for improvement
            let context = RepositoryContext::default(); // Would be populated from context phase
            let feedback = self.feedback_system.generate_feedback(&solution, &context).await?;
            
            // Apply feedback to task
            task = self.apply_feedback_to_task(task, &feedback).await?;
            
            session.feedback_loops += 1;
            debug!("Starting feedback loop {}/{}", session.feedback_loops, self.config.max_feedback_loops);
        }
    }
    
    /// Execute single pipeline iteration
    async fn execute_single_iteration(&self, task: &CodingTask, session: &mut PipelineSession) -> SACAResult<SACASolution> {
        let mut pipeline_data = PipelineData::new();
        
        // Phase 1: Chain-of-Thought Reasoning
        session.current_phase = SACAPhase::Think;
        let cot_result = self.execute_phase_with_metrics(
            SACAPhase::Think,
            "Chain-of-Thought Reasoning",
            self.cot_engine.reason(task)
        ).await?;
        pipeline_data.cot_result = Some(cot_result);
        session.phase_history.push(session.current_phase);
        
        // Phase 2: Modular Decomposition
        session.current_phase = SACAPhase::Decompose;
        let modules = self.execute_phase_with_metrics(
            SACAPhase::Decompose,
            "Modular Decomposition",
            self.decompose_engine.decompose(pipeline_data.cot_result.as_ref().ok_or_else(|| anyhow::anyhow!("CoT result not available"))?)
        ).await?;
        pipeline_data.modules = Some(modules);
        session.phase_history.push(session.current_phase);
        
        // Phase 3: Repository-Level Context
        session.current_phase = SACAPhase::Context;
        let context = self.execute_phase_with_metrics(
            SACAPhase::Context,
            "Repository Context Analysis",
            self.context_engine.analyze(pipeline_data.modules.as_ref().ok_or_else(|| anyhow::anyhow!("Modules not available"))?, task)
        ).await?;
        pipeline_data.context = Some(context);
        session.phase_history.push(session.current_phase);
        
        // Phase 4: Large-Scale Sampling
        session.current_phase = SACAPhase::Sample;
        let candidates = self.execute_phase_with_metrics(
            SACAPhase::Sample,
            "Large-Scale Sampling",
            self.sampling_engine.sample(
                pipeline_data.modules.as_ref().ok_or_else(|| anyhow::anyhow!("Modules not available"))?,
                pipeline_data.context.as_ref().ok_or_else(|| anyhow::anyhow!("Context not available"))?,
                pipeline_data.cot_result.as_ref().ok_or_else(|| anyhow::anyhow!("CoT result not available"))?
            )
        ).await?;
        pipeline_data.candidates = Some(candidates);
        session.phase_history.push(session.current_phase);
        
        // Phase 5: Execute-Fail-Fix Loop
        session.current_phase = SACAPhase::Execute;
        let executed_candidates = self.execute_phase_with_metrics(
            SACAPhase::Execute,
            "Execute-Fail-Fix Loop",
            self.execute_engine.execute_all(pipeline_data.candidates.as_ref().ok_or_else(|| anyhow::anyhow!("Candidates not available"))?.clone(), pipeline_data.context.as_ref().ok_or_else(|| anyhow::anyhow!("Context not available"))?)
        ).await?;
        pipeline_data.executed_candidates = Some(executed_candidates);
        session.phase_history.push(session.current_phase);
        
        // Phase 6: Mathematical Reranking
        session.current_phase = SACAPhase::Optimize;
        let solution = self.execute_phase_with_metrics(
            SACAPhase::Optimize,
            "Mathematical Reranking",
            self.rerank_engine.rerank(pipeline_data.executed_candidates.as_ref().ok_or_else(|| anyhow::anyhow!("Executed candidates not available"))?.clone(), pipeline_data.context.as_ref().ok_or_else(|| anyhow::anyhow!("Context not available"))?)
        ).await?;
        session.phase_history.push(session.current_phase);
        
        // Update session data
        let mut solution = solution;
        solution.session_id = session.id;
        solution.total_iterations = session.iterations;
        solution.total_feedback_loops = session.feedback_loops;
        solution.execution_time = Utc::now() - session.start_time;
        
        Ok(solution)
    }
    
    /// Execute a phase with metrics collection
    async fn execute_phase_with_metrics<T>(
        &self,
        phase: SACAPhase,
        phase_name: &str,
        future: impl std::future::Future<Output = SACAResult<T>>,
    ) -> SACAResult<T> {
        let start_time = std::time::Instant::now();
        
        debug!("Starting phase: {}", phase_name);
        
        let result = future.await;
        
        let execution_time = start_time.elapsed();
        
        // Update phase metrics
        let success = result.is_ok();
        self.update_phase_metrics(phase, execution_time, success).await;
        
        match &result {
            Ok(_) => {
                debug!("Phase {} completed in {:?}", phase_name, execution_time);
            },
            Err(e) => {
                warn!("Phase {} failed in {:?}: {}", phase_name, execution_time, e);
            }
        }
        
        result
    }
    
    /// Update phase metrics
    async fn update_phase_metrics(&self, phase: SACAPhase, execution_time: std::time::Duration, success: bool) {
        let mut metrics = self.phase_metrics.write().await;
        let phase_metrics = metrics.entry(phase).or_insert_with(PhaseMetrics::default);
        
        phase_metrics.average_time_ms = (phase_metrics.average_time_ms + execution_time.as_millis() as f64) / 2.0;
        phase_metrics.success_rate = if success {
            phase_metrics.success_rate * 0.9 + 0.1 // Exponential moving average
        } else {
            phase_metrics.success_rate * 0.9 // Decrease on failure
        };
        phase_metrics.average_attempts += 1.0;
    }
    
    /// Apply feedback to improve the task
    async fn apply_feedback_to_task(&self, mut task: CodingTask, feedback: &SACAFeedback) -> SACAResult<CodingTask> {
        // Incorporate feedback insights
        task.description = format!(
            "{}\n\nFeedback for improvement:\n{}",
            task.description,
            feedback.improvement_suggestions.join("\n")
        );
        
        // Add new constraints
        task.constraints.extend(feedback.new_constraints.clone());
        
        // Update requirements
        task.requirements.extend(feedback.updated_requirements.clone());
        
        Ok(task)
    }
    
    /// Update final metrics after pipeline completion
    async fn update_final_metrics(&self, solution: &SACASolution) -> SACAResult<()> {
        // Update global metrics based on solution performance
        let metrics = self.phase_metrics.read().await;
        
        info!("Pipeline metrics:");
        for (phase, phase_metrics) in metrics.iter() {
            info!("  {:?}: avg_time={:.1}ms, success_rate={:.3}, avg_attempts={:.1}",
                  phase, phase_metrics.average_time_ms, phase_metrics.success_rate, phase_metrics.average_attempts);
        }
        
        Ok(())
    }
    
    /// Get current session information
    pub async fn get_current_session(&self) -> Option<PipelineSession> {
        self.current_session.read().await.clone()
    }
    
    /// Get phase metrics
    pub async fn get_phase_metrics(&self) -> std::collections::HashMap<SACAPhase, PhaseMetrics> {
        self.phase_metrics.read().await.clone()
    }
    
    /// Reset metrics
    pub async fn reset_metrics(&self) {
        self.phase_metrics.write().await.clear();
        info!("Pipeline metrics reset");
    }
}

/// Pipeline session tracking
#[derive(Debug, Clone)]
pub struct PipelineSession {
    pub id: uuid::Uuid,
    pub task: CodingTask,
    pub start_time: chrono::DateTime<Utc>,
    pub current_phase: SACAPhase,
    pub iterations: u32,
    pub feedback_loops: u32,
    pub phase_history: Vec<SACAPhase>,
}

/// Pipeline data passed between phases
#[derive(Debug, Clone, Default)]
struct PipelineData {
    cot_result: Option<CoTResult>,
    modules: Option<Vec<Module>>,
    context: Option<RepositoryContext>,
    candidates: Option<Vec<SamplingCandidate>>,
    executed_candidates: Option<Vec<SACAExecutionResult>>,
}

impl PipelineData {
    fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pipeline_creation() -> anyhow::Result<()> {
        let config = SACAConfig::default();
        let pipeline = SACAPipeline::new(config).await;
        assert!(pipeline.is_ok());
        Ok(())
    }
    
    #[tokio::test]
    async fn test_simple_pipeline_execution() -> anyhow::Result<()> {
        let config = SACAConfig::default();
        let pipeline = SACAPipeline::new(config).await
            .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {}", e))?;
        
        let task = CodingTask {
            description: "Create a function that sorts an array".to_string(),
            requirements: vec!["Use efficient algorithm".to_string()],
            constraints: vec![],
            context: None,
        };
        
        let solution = pipeline.execute(task).await
            .map_err(|e| anyhow::anyhow!("Pipeline execution failed: {}", e))?;
        assert!(solution.quality_score >= 0.0);
        assert!(solution.quality_score <= 1.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_phase_metrics() -> anyhow::Result<()> {
        let config = SACAConfig::default();
        let pipeline = SACAPipeline::new(config).await
            .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {}", e))?;
        
        let task = CodingTask {
            description: "Test task".to_string(),
            requirements: vec![],
            constraints: vec![],
            context: None,
        };
        
        let _solution = pipeline.execute(task).await
            .map_err(|e| anyhow::anyhow!("Pipeline execution failed: {}", e))?;
        
        let metrics = pipeline.get_phase_metrics().await;
        assert!(!metrics.is_empty());
        
        for (phase, phase_metrics) in metrics {
            assert!(phase_metrics.average_time_ms >= 0.0);
            assert!(phase_metrics.success_rate >= 0.0);
            assert!(phase_metrics.success_rate <= 1.0);
            assert!(phase_metrics.average_attempts >= 1.0);
        }
        
        Ok(())
    }
}
