//! Nexora Cognition - Cognitive layer for AI systems
//!
//! Provides:
//! - planning: Task planning and execution strategies
//! - reflection: Self-reflection and meta-cognition

//! - context: Context management and evolution
//! - reasoning: High-level reasoning capabilities

pub mod planning;
pub mod reflection;
pub mod context;
pub mod reasoning;

pub use planning::*;
pub use reflection::*;
pub use context::*;
pub use reasoning::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_creation() {
        let step = PlanStep {
            id: uuid::Uuid::new_v4(),
            action: "process".into(),
            parameters: serde_json::json!({}),
            dependencies: vec![],
            estimated_duration_ms: 100,
        };
        let plan = Plan {
            id: uuid::Uuid::new_v4(),
            steps: vec![step],
            dependencies: vec![],
            estimated_duration_ms: 100,
        };
        assert_ne!(plan.id, uuid::Uuid::nil());
        assert_eq!(plan.steps.len(), 1);
    }

    #[test]
    fn test_context_window_creation() {
        let entry = ContextEntry {
            id: uuid::Uuid::new_v4(),
            content: "test entry".into(),
            entry_type: ContextType::UserInput,
            importance: 0.8,
                timestamp: chrono::Utc::now().timestamp(),
            embeddings: None,
        };
        let window = ContextWindow {
            id: uuid::Uuid::new_v4(),
            entries: vec![entry],
            metadata: ContextMetadata {
                created_at: chrono::Utc::now().timestamp(),
                updated_at: chrono::Utc::now().timestamp(),
                total_entries: 1,
                tags: vec![],
            },
            max_size: 100,
        };
        assert_eq!(window.entries.len(), 1);
        assert_eq!(window.max_size, 100);
    }

    #[test]
    fn test_reasoning_chain() {
        let step = ReasoningStep {
            step_id: 1,
            premise: "all humans are mortal".into(),
            inference: "Socrates is human".into(),
            confidence: 0.95,
            step_type: ReasoningType::Deductive,
        };
        let chain = ReasoningChain {
            steps: vec![step],
            conclusion: "Socrates is mortal".into(),
            confidence: 0.95,
        };
        assert_eq!(chain.steps.len(), 1);
        assert_eq!(chain.conclusion, "Socrates is mortal");
    }

    #[test]
    fn test_reflection_result() {
        let result = ReflectionResult {
            confidence: 0.85,
            errors_identified: vec![],
            improvements_suggested: vec!["optimize query".into()],
            learning_insights: vec![],
            metadata: ReflectionMetadata {
                reflection_type: ReflectionType::Performance,
            timestamp: chrono::Utc::now().timestamp(),
                context: "test".into(),
            },
        };
        assert_eq!(result.confidence, 0.85);
        assert_eq!(result.improvements_suggested.len(), 1);
    }

    #[test]
    fn test_reflection_type_display() {
        let perf = ReflectionType::Performance;
        assert!(format!("{:?}", perf).contains("Performance"));
    }

    #[test]
    fn test_reasoning_type_variants() {
        let variants = vec![
            ReasoningType::Deductive,
            ReasoningType::Inductive,
            ReasoningType::Abductive,
            ReasoningType::Analogical,
            ReasoningType::Causal,
        ];
        assert_eq!(variants.len(), 5);
    }
}
