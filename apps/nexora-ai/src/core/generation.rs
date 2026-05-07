//! Text generation functionality

use anyhow::Result;
use tracing::{debug, info};
use chrono::Utc;

use super::types::{PromptAnalysis, GenerationType};

/// Text generation engine
#[derive(Debug, Clone)]
pub struct TextGenerator;

impl TextGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate text with sophisticated parameters
    pub async fn generate_text(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String> {
        let generation_start = Utc::now();
        
        info!("Generating text with sophisticated parameters: prompt='{}', max_tokens={}, temperature={}", 
              prompt, max_tokens, temperature);
        
        // Comprehensive validation
        if prompt.is_empty() {
            return Err(anyhow::anyhow!("Prompt cannot be empty"));
        }
        
        if prompt.len() > 5000 {
            return Err(anyhow::anyhow!("Prompt too long (max 5000 characters)"));
        }
        
        if max_tokens == 0 {
            return Err(anyhow::anyhow!("Max tokens must be greater than 0"));
        }
        
        if max_tokens > 4096 {
            return Err(anyhow::anyhow!("Max tokens too large (max 4096)"));
        }
        
        if !(0.0..=2.0).contains(&temperature) {
            return Err(anyhow::anyhow!("Temperature must be between 0.0 and 2.0"));
        }
        
        // Analyze prompt complexity
        let prompt_analysis = self.analyze_prompt_complexity(prompt);
        debug!("Prompt analysis: {:?}", prompt_analysis);
        
        // Generate text based on temperature and complexity
        let generated_text = match temperature {
            t if t < 0.3 => self.generate_deterministic_text(prompt, &prompt_analysis).await?,
            t if t < 0.8 => self.generate_balanced_text(prompt, &prompt_analysis).await?,
            t if t < 1.5 => self.generate_creative_text(prompt, &prompt_analysis).await?,
            _ => self.generate_experimental_text(prompt, &prompt_analysis).await?,
        };
        
        // Post-process and validate generated text
        let processed_text = self.post_process_generated_text(&generated_text, max_tokens)?;
        
        let generation_time = (Utc::now() - generation_start).num_milliseconds();
        info!("Text generation completed in {}ms, output length: {} chars", 
              generation_time, processed_text.len());
        
        Ok(processed_text)
    }
    
    /// Analyze prompt complexity for generation strategy
    fn analyze_prompt_complexity(&self, prompt: &str) -> PromptAnalysis {
        let word_count = prompt.split_whitespace().count();
        let sentence_count = prompt.split(&['.', '!', '?'][..]).filter(|s| !s.trim().is_empty()).count();
        let question_count = prompt.matches('?').count();
        let code_blocks = prompt.matches("```").count() / 2;
        
        let complexity_score = (word_count as f64 * 0.1 + 
                              sentence_count as f64 * 0.2 + 
                              question_count as f64 * 0.3 + 
                              code_blocks as f64 * 0.4).min(100.0);
        
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
    
    /// Generate deterministic text (low temperature)
    async fn generate_deterministic_text(&self, prompt: &str, analysis: &PromptAnalysis) -> Result<String> {
        match analysis.generation_type {
            GenerationType::Code => {
                Ok(format!("// Deterministic code generation for:\n{}\n\n// Generated code:\nfn process_{}() {{\n    // Implementation\n}}", 
                          prompt, analysis.word_count))
            },
            GenerationType::Question => {
                Ok(format!("Based on your question about {} words, here's a structured response.", 
                          analysis.word_count))
            },
            _ => {
                Ok(format!("Deterministic response to: {}\n\nThis is a predictable, structured output.", 
                          prompt))
            }
        }
    }
    
    /// Generate balanced text (medium temperature)
    async fn generate_balanced_text(&self, prompt: &str, analysis: &PromptAnalysis) -> Result<String> {
        let creativity_factor = (analysis.complexity_score / 100.0) * 0.5;
        
        match analysis.generation_type {
            GenerationType::Code => {
                Ok(format!("// Balanced code generation\n// Complexity: {:.1}\n{}\n\n// Enhanced implementation:\nfn enhanced_process_{}() {{\n    let result = calculate_with_creativity({});\n    return result;\n}}", 
                          analysis.complexity_score, prompt, analysis.word_count, creativity_factor))
            },
            _ => {
                Ok(format!("Balanced response to your {}-word prompt with creativity factor {:.2}:\n\n{}\n\nThis response combines structure with creative elements.", 
                          analysis.word_count, creativity_factor, prompt))
            }
        }
    }
    
    /// Generate creative text (high temperature)
    async fn generate_creative_text(&self, prompt: &str, analysis: &PromptAnalysis) -> Result<String> {
        let creative_elements = vec![
            "innovative perspective",
            "creative insight", 
            "imaginative approach",
            "artistic interpretation",
            "original thinking"
        ];
        
        let selected_element = creative_elements[analysis.word_count % creative_elements.len()];
        
        Ok(format!("Creative generation with {}: \n\nOriginal prompt: {}\n\nCreative interpretation: This {}-word input inspires {} with a complexity score of {:.1}. The response flows with creative energy while maintaining coherence.", 
                  selected_element, prompt, analysis.word_count, selected_element, analysis.complexity_score))
    }
    
    /// Generate experimental text (very high temperature)
    async fn generate_experimental_text(&self, prompt: &str, analysis: &PromptAnalysis) -> Result<String> {
        let experimental_patterns = vec![
            "quantum-inspired",
            "neural-network-driven",
            "chaos-theory-based",
            "fractal-generated",
            "emergent-behavior"
        ];
        
        let pattern = experimental_patterns[analysis.complexity_score as usize % experimental_patterns.len()];
        
        Ok(format!("🚀 EXPERIMENTAL GENERATION MODE 🚀\n\nPattern: {}\nInput complexity: {:.1}\nPrompt: {}\n\nExperimental output: [{}] This {}-word challenge triggers {} processing with {} complexity. The response emerges from the intersection of deterministic logic and creative chaos, producing something entirely new and unexpected.", 
                  pattern, analysis.complexity_score, prompt, pattern, analysis.word_count, pattern, analysis.complexity_score))
    }
    
    /// Post-process generated text
    fn post_process_generated_text(&self, text: &str, max_tokens: usize) -> Result<String> {
        // Basic token count estimation (rough approximation)
        let estimated_tokens = text.len() / 4; // Rough estimate: 1 token ≈ 4 characters
        
        if estimated_tokens > max_tokens {
            // Truncate to approximate token limit
            let char_limit = max_tokens * 4;
            if text.len() > char_limit {
                let truncated = &text[..char_limit.min(text.len())];
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
