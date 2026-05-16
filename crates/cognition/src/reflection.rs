//! Reflection Module - Self-reflection and meta-cognition

use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use nexora_foundation::FoundationResult;

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
    async fn reflect(&self, actions: &[Action], context: &str) -> FoundationResult<ReflectionResult>;
    async fn suggest_improvements(&self, reflection: &ReflectionResult) -> FoundationResult<Vec<String>>;
    async fn update_model(&self, reflection: &ReflectionResult) -> FoundationResult<()>;
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

/// Default reflection engine implementation
pub struct DefaultReflector {
    history: std::sync::Arc<tokio::sync::RwLock<Vec<ReflectionResult>>>,
}

impl DefaultReflector {
    pub fn new() -> Self {
        Self {
            history: std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new())),
        }
    }
}

impl Default for DefaultReflector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReflectionEngine for DefaultReflector {
    async fn reflect(&self, actions: &[Action], context: &str) -> FoundationResult<ReflectionResult> {
        let mut errors = Vec::new();
        let mut improvements = Vec::new();
        let mut insights = Vec::new();

        for action in actions {
            if !action.success {
                errors.push(format!("{} failed: {}", action.action_type, action.output));
                improvements.push(format!("Retry {} with adjusted parameters", action.action_type));
            }
        }

        if actions.is_empty() {
            insights.push("No actions to reflect on".to_string());
        } else {
            let success_rate = actions.iter().filter(|a| a.success).count() as f32 / actions.len() as f32;
            insights.push(format!("Success rate: {:.1}%", success_rate * 100.0));
            if success_rate < 0.5 {
                improvements.push("Increase validation before action execution".to_string());
            }
        }

        let has_errors = !errors.is_empty();
        let confidence = if actions.is_empty() {
            0.0
        } else {
            actions.iter().filter(|a| a.success).count() as f32 / actions.len() as f32
        };

        let result = ReflectionResult {
            confidence,
            errors_identified: errors,
            improvements_suggested: improvements,
            learning_insights: insights,
            metadata: ReflectionMetadata {
                reflection_type: if has_errors { ReflectionType::ErrorAnalysis } else { ReflectionType::Performance },
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64,
                context: context.to_string(),
            },
        };

        {
            let mut history = self.history.write().await;
            history.push(result.clone());
            if history.len() > 1000 {
                history.remove(0);
            }
        }

        Ok(result)
    }

    async fn suggest_improvements(&self, reflection: &ReflectionResult) -> FoundationResult<Vec<String>> {
        let mut suggestions = Vec::new();

        if reflection.confidence < 0.3 {
            suggestions.push("Consider using a different approach entirely".to_string());
            suggestions.push("Break down the problem into smaller steps".to_string());
        } else if reflection.confidence < 0.7 {
            suggestions.push("Add more validation checkpoints".to_string());
            suggestions.push("Review recent changes for regressions".to_string());
        }

        suggestions.extend(reflection.improvements_suggested.clone());
        suggestions.extend(reflection.learning_insights.clone());

        Ok(suggestions)
    }

    async fn update_model(&self, reflection: &ReflectionResult) -> FoundationResult<()> {
        let mut history = self.history.write().await;
        history.push(reflection.clone());
        if history.len() > 1000 {
            history.remove(0);
        }
        Ok(())
    }

    async fn stats(&self) -> FoundationResult<ReflectionStats> {
        let history = self.history.read().await;
        let total = history.len();
        let avg_confidence = if total == 0 {
            0.0
        } else {
            history.iter().map(|r| r.confidence).sum::<f32>() / total as f32
        };
        let improvement_rate = if total < 2 {
            0.0
        } else {
            let recent = &history[total - total.min(10)..];
            let improvements = recent.iter().filter(|r| r.confidence > 0.7).count();
            improvements as f32 / recent.len() as f32
        };

        Ok(ReflectionStats {
            total_reflections: total,
            avg_confidence,
            improvement_rate,
        })
    }
}
