//! Reflection Module - Self-reflection and meta-cognition

use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use nexora_foundation_traits::FoundationResult;

/// Reflection result from self-analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionResult {
    pub confidence: f32,
    pub errors_identified: Vec<String>,
    pub improvements_suggested: Vec<String>,
    pub learning_insights: Vec<String>,
    pub metadata: ReflectionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionMetadata {
    pub reflection_type: ReflectionType,
    pub timestamp: i64,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReflectionType {
    Performance,
    Accuracy,
    Efficiency,
    Learning,
    ErrorAnalysis,
}

/// Reflection engine trait
#[async_trait]
pub trait ReflectionEngine: Send + Sync {
    /// Perform self-reflection on recent actions
    async fn reflect(&self, actions: &[Action], context: &str) -> FoundationResult<ReflectionResult>;
    
    /// Generate improvement suggestions
    async fn suggest_improvements(&self, reflection: &ReflectionResult) -> FoundationResult<Vec<String>>;
    
    /// Update internal model based on reflection
    async fn update_model(&self, reflection: &ReflectionResult) -> FoundationResult<()>;
    
    /// Get reflection statistics
    async fn stats(&self) -> FoundationResult<ReflectionStats>;
}

#[derive(Debug, Clone)]
pub struct Action {
    pub action_type: String,
    pub input: String,
    pub output: String,
    pub timestamp: i64,
    pub success: bool,
}

#[derive(Debug, Clone)]
pub struct ReflectionStats {
    pub total_reflections: usize,
    pub avg_confidence: f32,
    pub improvement_rate: f32,
}
