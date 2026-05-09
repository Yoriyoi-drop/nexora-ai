//! Parallel Execution Strategy
//! 
//! Implements parallel execution of candidates for improved performance.

use crate::reasoning::saca::{types::*, config::*, error::*};
use super::super::engine::ExecuteEngine;
use std::sync::Arc;

/// Parallel execution strategy
pub struct ParallelExecutionStrategy;

impl ParallelExecutionStrategy {
    pub async fn execute(
        engine: &ExecuteEngine,
        candidates: Vec<SamplingCandidate>,
        context: &RepositoryContext,
    ) -> SACAResult<Vec<SACAExecutionResult>> {
        let tasks: Vec<_> = candidates
            .into_iter()
            .map(|candidate| {
                let engine = engine.clone();
                let context = context.clone();
                async move { engine.execute_candidate_with_fix_loop(candidate, &context).await }
            })
            .collect();
        
        // Execute all tasks in parallel
        let results = futures::future::join_all(tasks).await;
        
        // Collect results, handling any errors
        let mut execution_results = Vec::new();
        for result in results {
            match result {
                Ok(execution_result) => execution_results.push(execution_result),
                Err(e) => {
                    // Create error result for failed execution
                    execution_results.push(SACAExecutionResult {
                        candidate_id: uuid::Uuid::new_v4(), // Will be set properly
                        success: false,
                        execution_time_ms: 0,
                        memory_usage_mb: 0.0,
                        test_results: vec![],
                        error_logs: vec![format!("Parallel execution failed: {}", e)],
                        performance_metrics: Default::default(),
                        code_lines: None,
                        generated_code: None,
                    });
                }
            }
        }
        
        Ok(execution_results)
    }
}
