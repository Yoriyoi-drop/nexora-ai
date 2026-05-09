//! Sequential Execution Strategy
//! 
//! Implements sequential execution of candidates one by one.

use crate::reasoning::saca::{types::*, config::*, error::*};
use super::super::engine::ExecuteEngine;

/// Sequential execution strategy
pub struct SequentialExecutionStrategy;

impl SequentialExecutionStrategy {
    pub async fn execute(
        engine: &ExecuteEngine,
        candidates: Vec<SamplingCandidate>,
        context: &RepositoryContext,
    ) -> SACAResult<Vec<SACAExecutionResult>> {
        let mut results = Vec::new();
        
        for candidate in candidates {
            let result = engine.execute_candidate_with_fix_loop(candidate, context).await?;
            results.push(result);
        }
        
        Ok(results)
    }
}
