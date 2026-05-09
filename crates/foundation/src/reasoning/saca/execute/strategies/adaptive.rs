//! Adaptive Execution Strategy
//! 
//! Implements adaptive execution that switches between sequential and parallel based on conditions.

use crate::reasoning::saca::{types::*, config::*, error::*};
use super::super::engine::ExecuteEngine;
use super::{SequentialExecutionStrategy, ParallelExecutionStrategy};

/// Adaptive execution strategy
pub struct AdaptiveExecutionStrategy;

impl AdaptiveExecutionStrategy {
    pub async fn execute(
        engine: &ExecuteEngine,
        candidates: Vec<SamplingCandidate>,
        context: &RepositoryContext,
    ) -> SACAResult<Vec<SACAExecutionResult>> {
        // Analyze candidates to determine optimal execution strategy
        let execution_mode = Self::determine_execution_mode(&candidates, context);
        
        match execution_mode {
            ExecutionMode::Sequential => {
                SequentialExecutionStrategy::execute(engine, candidates, context).await
            },
            ExecutionMode::Parallel => {
                ParallelExecutionStrategy::execute(engine, candidates, context).await
            },
            ExecutionMode::Hybrid => {
                Self::execute_hybrid(engine, candidates, context).await
            },
        }
    }
    
    fn determine_execution_mode(
        candidates: &[SamplingCandidate],
        _context: &RepositoryContext,
    ) -> ExecutionMode {
        let candidate_count = candidates.len();
        
        // Use parallel for many candidates
        if candidate_count > 5 {
            ExecutionMode::Parallel
        } 
        // Use sequential for few candidates
        else if candidate_count <= 2 {
            ExecutionMode::Sequential
        } 
        // Use hybrid for medium number
        else {
            ExecutionMode::Hybrid
        }
    }
    
    async fn execute_hybrid(
        engine: &ExecuteEngine,
        candidates: Vec<SamplingCandidate>,
        context: &RepositoryContext,
    ) -> SACAResult<Vec<SACAExecutionResult>> {
        let mut results = Vec::new();
        let mut candidates = candidates;
        
        // Execute first few candidates sequentially to get baseline
        let sequential_count = candidates.len().min(2);
        let sequential_candidates: Vec<_> = candidates.drain(..sequential_count).collect();
        
        let sequential_results = SequentialExecutionStrategy::execute(
            engine, 
            sequential_candidates, 
            context
        ).await?;
        
        results.extend(sequential_results);
        
        // Execute remaining candidates in parallel
        if !candidates.is_empty() {
            let parallel_results = ParallelExecutionStrategy::execute(
                engine, 
                candidates, 
                context
            ).await?;
            
            results.extend(parallel_results);
        }
        
        Ok(results)
    }
}

enum ExecutionMode {
    Sequential,
    Parallel,
    Hybrid,
}
