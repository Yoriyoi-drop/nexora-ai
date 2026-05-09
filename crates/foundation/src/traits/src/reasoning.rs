//! Reasoning Engine Trait
//!
//! Defines the interface for reasoning implementations (SACA, etc.)

use async_trait::async_trait;
use std::collections::HashMap;
use crate::FoundationResult;

/// Input for reasoning engine
#[derive(Debug, Clone)]
pub struct ReasoningInput {
    pub query: String,
    pub context: Option<String>,
    pub constraints: Vec<String>,
}

/// Output from reasoning engine
#[derive(Debug, Clone)]
pub struct ReasoningOutput {
    pub result: String,
    pub confidence: f32,
    pub reasoning_steps: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Core reasoning engine trait
#[async_trait]
pub trait ReasoningEngine: Send + Sync {
    /// Perform reasoning on the given input
    async fn reason(&self, input: ReasoningInput) -> FoundationResult<ReasoningOutput>;
    
    /// Get engine capabilities
    fn capabilities(&self) -> Vec<String>;
    
    /// Validate if the engine can handle the given input
    fn can_handle(&self, input: &ReasoningInput) -> bool;
    
    /// Get engine name/version
    fn engine_info(&self) -> EngineInfo;
}

/// Engine metadata
#[derive(Debug, Clone)]
pub struct EngineInfo {
    pub name: String,
    pub version: String,
    pub description: String,
}
