//! Model Backend trait for cognition components
//!
//! Allows cognition modules to optionally use an LLM backend
//! instead of heuristic/fallback implementations.

/// Trait for model backends that cognition components can use
/// for actual reasoning instead of heuristic fallbacks.
#[async_trait::async_trait]
pub trait ModelBackend: Send + Sync {
    /// Query the model with a prompt and get a response
    async fn query(&self, prompt: &str, context: &str) -> Result<String, String>;

    /// Check if this backend is available for use
    fn is_available(&self) -> bool {
        backend_enabled()
    }
}

/// Check whether a model backend should be used based on environment.
pub fn backend_enabled() -> bool {
    std::env::var("NEXORA_MODEL_BACKEND").is_ok()
        || std::env::var("NEXORA_LLM_BACKEND").is_ok()
}
