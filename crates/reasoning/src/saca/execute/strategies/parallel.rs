//! Parallel Execution Strategy
//!
//! Implements parallel execution of candidates for improved performance.

use super::super::engine::ExecuteEngine;
use crate::saca::{error::*, types::*};

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
                let candidate_id = candidate.id;
                let engine = engine.clone();
                let context = context.clone();
                async move {
                    let result = engine
                        .execute_candidate_with_fix_loop(candidate, &context)
                        .await;
                    (candidate_id, result)
                }
            })
            .collect();

        // Execute all tasks in parallel
        let results = futures::future::join_all(tasks).await;

        // Collect results, handling any errors
        let mut execution_results = Vec::new();
        for (candidate_id, result) in results {
            match result {
                Ok(execution_result) => execution_results.push(execution_result),
                Err(e) => {
                    execution_results.push(SACAExecutionResult {
                        candidate_id,
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
