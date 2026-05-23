//! Text generation functionality
//!
//! Provides text generation using the NXR foundation model registry.
//! Delegates all generation to the active model via model registry inference.

use crate::error::{NexoraError, NexoraResult};
use chrono::Utc;
use nexora_foundation::shared::{
    base_model::{InputData, NxrInput, OutputData},
    model_identity::NxrModelId,
    model_registry::NxrModelRegistry,
};
use std::sync::Arc;
use tracing::{debug, info, warn};

use super::types::{GenerationType, PromptAnalysis};

/// Text generation engine
///
/// Routes text generation requests through the NXR foundation model registry,
/// using the active model ID for all inference calls.
#[derive(Debug, Clone)]
pub struct TextGenerator {
    registry: Arc<NxrModelRegistry>,
    active_model_id: NxrModelId,
}

impl TextGenerator {
    pub fn new(registry: Arc<NxrModelRegistry>, active_model_id: NxrModelId) -> Self {
        Self {
            registry,
            active_model_id,
        }
    }

    /// Generate text with sophisticated parameters
    pub async fn generate_text(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> NexoraResult<String> {
        let generation_start = Utc::now();

        info!(
            "🚀 Starting text generation: prompt_len={}, max_tokens={}, temperature={:.2}",
            prompt.len(),
            max_tokens,
            temperature
        );
        debug!("📝 Prompt content: {}", prompt);

        // Comprehensive validation
        if prompt.is_empty() {
            return Err(NexoraError::validation("prompt", "Prompt cannot be empty"));
        }

        if prompt.len() > 5000 {
            return Err(NexoraError::validation(
                "prompt",
                "Prompt too long (max 5000 characters)",
            ));
        }

        if max_tokens == 0 {
            return Err(NexoraError::validation(
                "max_tokens",
                "Max tokens must be greater than 0",
            ));
        }

        if max_tokens > 4096 {
            return Err(NexoraError::validation(
                "max_tokens",
                "Max tokens too large (max 4096)",
            ));
        }

        if !(0.0..=2.0).contains(&temperature) {
            return Err(NexoraError::validation(
                "temperature",
                "Temperature must be between 0.0 and 2.0",
            ));
        }

        // Analyze prompt complexity
        let prompt_analysis = self.analyze_prompt_complexity(prompt);
        debug!("Prompt analysis: {:?}", prompt_analysis);

        // Generate text based on temperature and complexity
        let generated_text = match temperature {
            t if t < 0.3 => {
                self.generate_deterministic_text(prompt, &prompt_analysis)
                    .await?
            }
            t if t < 0.8 => {
                self.generate_balanced_text(prompt, &prompt_analysis)
                    .await?
            }
            t if t < 1.5 => {
                self.generate_creative_text(prompt, &prompt_analysis)
                    .await?
            }
            _ => {
                self.generate_experimental_text(prompt, &prompt_analysis)
                    .await?
            }
        };

        // Post-process and validate generated text
        let processed_text = self.post_process_generated_text(&generated_text, max_tokens)?;

        let generation_time = (Utc::now() - generation_start).num_milliseconds();
        info!(
            "Text generation completed in {}ms, output length: {} chars",
            generation_time,
            processed_text.len()
        );

        Ok(processed_text)
    }

    /// Analyze prompt complexity for generation strategy
    fn analyze_prompt_complexity(&self, prompt: &str) -> PromptAnalysis {
        let word_count = prompt.split_whitespace().count();
        let sentence_count = prompt
            .split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .count();
        let question_count = prompt.matches('?').count();
        let code_blocks = prompt.matches("```").count() / 2;

        let complexity_score = (word_count as f64 * 0.1
            + sentence_count as f64 * 0.2
            + question_count as f64 * 0.3
            + code_blocks as f64 * 0.4)
            .min(100.0);

        let generation_type = if code_blocks > 0 {
            GenerationType::Code
        } else if question_count > 0 {
            GenerationType::Question
        } else if word_count > 50 {
            GenerationType::LongForm
        } else {
            GenerationType::Short
        };

        PromptAnalysis {
            word_count,
            sentence_count,
            question_count,
            code_blocks,
            complexity_score,
            generation_type,
        }
    }

    /// Generate text via foundation model inference through the model registry.
    /// Routes to the active model (default: Omnis) with configured parameters.
    async fn model_generate(&self, prompt: &str, max_tokens: usize, temperature: f32) -> NexoraResult<String> {
        let model = self
            .registry
            .get_model(&self.active_model_id)
            .await
            .map_err(|e| {
                NexoraError::model(format!(
                    "Model {} not available for text generation: {}",
                    self.active_model_id, e
                ))
            })?;

        let input = NxrInput {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            data: InputData::Text(prompt.to_string()),
            parameters: [
                ("max_tokens".to_string(), serde_json::json!(max_tokens)),
                ("temperature".to_string(), serde_json::json!(temperature)),
            ]
            .into(),
            metadata: std::collections::HashMap::new(),
        };

        let output = model
            .infer(&input)
            .await
            .map_err(|e| NexoraError::model(format!("Model inference failed: {}", e)))?;

        let text = match output.data {
            OutputData::Text(t) => t,
            _ => format!("{:?}", output.data),
        };

        Ok(text)
    }

    /// Generate deterministic text (low temperature)
    /// Routes to foundation model with temperature < 0.3 for focused, predictable output.
    async fn generate_deterministic_text(
        &self,
        prompt: &str,
        _analysis: &PromptAnalysis,
    ) -> NexoraResult<String> {
        self.model_generate(prompt, 256, 0.2).await
    }

    /// Generate balanced text (medium temperature)
    /// Routes to foundation model with temperature 0.3-0.8 for creative yet coherent output.
    async fn generate_balanced_text(
        &self,
        prompt: &str,
        _analysis: &PromptAnalysis,
    ) -> NexoraResult<String> {
        self.model_generate(prompt, 256, 0.7).await
    }

    /// Generate creative text (high temperature)
    /// Routes to foundation model with temperature 0.8-1.5 for diverse, exploratory output.
    async fn generate_creative_text(
        &self,
        prompt: &str,
        _analysis: &PromptAnalysis,
    ) -> NexoraResult<String> {
        self.model_generate(prompt, 512, 1.2).await
    }

    /// Generate experimental text (very high temperature)
    /// Routes to foundation model with temperature > 1.5 for highly diverse output.
    async fn generate_experimental_text(
        &self,
        prompt: &str,
        _analysis: &PromptAnalysis,
    ) -> NexoraResult<String> {
        self.model_generate(prompt, 1024, 1.8).await
    }

    /// Post-process generated text
    fn post_process_generated_text(&self, text: &str, max_tokens: usize) -> NexoraResult<String> {
        // Basic token count estimation (rough approximation)
        let estimated_tokens = text.len() / 4; // Rough estimate: 1 token ≈ 4 characters

        if estimated_tokens > max_tokens {
            // Truncate to approximate token limit
            let char_limit = max_tokens * 4;
            if text.len() > char_limit {
                let truncated: String = text.chars().take(char_limit).collect();
                return Ok(format!("{}...[truncated]", truncated));
            }
        }

        // Ensure text doesn't end abruptly
        let processed = if !text.ends_with('.') && !text.ends_with('!') && !text.ends_with('?') {
            format!("{}.", text)
        } else {
            text.to_string()
        };

        Ok(processed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexora_foundation::shared::model_registry::global_registry;

    fn test_generator() -> TextGenerator {
        let registry = global_registry();
        TextGenerator::new(registry, NxrModelId::Omnis)
    }

    /// Tests validation logic only (no model inference needed — these pass regardless of registry state).
    /// The "Hello world" success case requires initialized models; #[ignore] that path.
    #[tokio::test]
    async fn test_generate_text_validation() {
        let generator = test_generator();

        // Test empty prompt
        let result = generator.generate_text("", 100, 0.7).await;
        assert!(result.is_err());

        // Test prompt too long
        let long_prompt = "a".repeat(5001);
        let result = generator.generate_text(&long_prompt, 100, 0.7).await;
        assert!(result.is_err());

        // Test invalid max_tokens
        let result = generator.generate_text("test", 0, 0.7).await;
        assert!(result.is_err());

        // Test max_tokens too large
        let result = generator.generate_text("test", 4097, 0.7).await;
        assert!(result.is_err());

        // Test invalid temperature
        let result = generator.generate_text("test", 100, -0.1).await;
        assert!(result.is_err());

        let result = generator.generate_text("test", 100, 2.1).await;
        assert!(result.is_err());
    }

    /// Integration test — requires initialized foundation models via global_registry().
    /// Runs only when models are registered (e.g. via initialize_foundation_models()).
    #[tokio::test]
    #[ignore = "requires initialized foundation models in global_registry()"]
    async fn test_generate_text_success_integration() {
        let generator = test_generator();

        let result = generator.generate_text("Hello world", 100, 0.7).await;
        assert!(result.is_ok());

        let text = result.unwrap();
        assert!(!text.is_empty());
        assert!(!text.contains("TODO:"), "Should not return placeholder TODO text");
    }

    #[tokio::test]
    async fn test_analyze_prompt_complexity() {
        let generator = test_generator();

        // Test simple text
        let analysis = generator.analyze_prompt_complexity("Hello world");
        assert_eq!(analysis.word_count, 2);
        assert_eq!(analysis.generation_type, GenerationType::Short);

        // Test question
        let analysis = generator.analyze_prompt_complexity("What is Rust?");
        assert_eq!(analysis.question_count, 1);
        assert_eq!(analysis.generation_type, GenerationType::Question);

        // Test code
        let analysis = generator.analyze_prompt_complexity("```rust\nfn test() {}\n```");
        assert_eq!(analysis.code_blocks, 1);
        assert_eq!(analysis.generation_type, GenerationType::Code);

        // Test long text
        let long_text = "word ".repeat(60);
        let analysis = generator.analyze_prompt_complexity(&long_text);
        assert_eq!(analysis.generation_type, GenerationType::LongForm);
    }

    #[test]
    fn test_post_process_generated_text() {
        let generator = test_generator();

        // Test text within token limit
        let result = generator.post_process_generated_text("Hello world", 100);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello world.");

        // Test text exceeding token limit
        let long_text = "a".repeat(500);
        let result = generator.post_process_generated_text(&long_text, 50);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("truncated"));

        // Test text already ending with punctuation
        let result = generator.post_process_generated_text("Hello world!", 100);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello world!");
    }
}
