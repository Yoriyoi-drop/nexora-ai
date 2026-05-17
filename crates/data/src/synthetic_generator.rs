//! Synthetic Data Generator - Rust implementation
//! 
//! Generates synthetic training data using various techniques

use anyhow::Result;
use std::collections::HashMap;
use rand::Rng;

use crate::DataEntry;

/// Synthetic data generator with multiple generation strategies
pub struct SyntheticGenerator {
    strategies: Vec<Box<dyn GenerationStrategy>>,
    config: GeneratorConfig,
    _templates: Vec<Template>,
    _vocabulary: HashMap<String, Vec<String>>,
}

/// Generator configuration
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    pub max_length: usize,
    pub min_length: usize,
    pub language: String,
    pub topics: Vec<String>,
    pub styles: Vec<String>,
    pub complexity_levels: Vec<ComplexityLevel>,
}

/// Complexity levels for generated content
#[derive(Debug, Clone)]
pub enum ComplexityLevel {
    Simple,
    Intermediate,
    Advanced,
}

/// Template for structured generation
#[derive(Debug, Clone)]
pub struct Template {
    pub id: String,
    pub pattern: String,
    pub placeholders: Vec<String>,
    pub style: String,
}

/// Trait for generation strategies
pub trait GenerationStrategy: Send + Sync {
    /// Get strategy name
    fn name(&self) -> &str;
    
    /// Generate content based on prompt
    fn generate(&self, prompt: &str, config: &GeneratorConfig) -> Result<String>;
    
    /// Get generation metadata
    fn metadata(&self) -> GenerationMetadata;
    
    /// Configure strategy
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()>;
    
    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
    
    /// Get as Any for mutable downcasting
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Generation metadata
#[derive(Debug, Clone)]
pub struct GenerationMetadata {
    pub strategy: String,
    pub complexity: ComplexityLevel,
    pub topic: String,
    pub style: String,
    pub confidence: f32,
}

/// Template-based generation strategy
#[derive(Debug, Clone)]
pub struct TemplateGenerationStrategy {
    templates: Vec<Template>,
    filler_words: HashMap<String, Vec<String>>,
}

impl TemplateGenerationStrategy {
    pub fn new() -> Self {
        let mut templates = Vec::new();
        let mut filler_words = HashMap::new();
        
        // Add default templates
        templates.push(Template {
            id: "question_answer".to_string(),
            pattern: "Q: {question} A: {answer}".to_string(),
            placeholders: vec!["question".to_string(), "answer".to_string()],
            style: "educational".to_string(),
        });
        
        templates.push(Template {
            id: "definition".to_string(),
            pattern: "{term} is defined as {definition}".to_string(),
            placeholders: vec!["term".to_string(), "definition".to_string()],
            style: "formal".to_string(),
        });
        
        templates.push(Template {
            id: "example".to_string(),
            pattern: "For example, {example}".to_string(),
            placeholders: vec!["example".to_string()],
            style: "casual".to_string(),
        });
        
        // Add filler words
        filler_words.insert("question".to_string(), vec![
            "What is the meaning of life?".to_string(),
            "How does photosynthesis work?".to_string(),
            "Why is the sky blue?".to_string(),
            "What are the benefits of exercise?".to_string(),
            "How do computers process information?".to_string(),
        ]);
        
        filler_words.insert("answer".to_string(), vec![
            "Life has different meanings for different people".to_string(),
            "Photosynthesis converts light energy into chemical energy".to_string(),
            "The sky appears blue due to Rayleigh scattering".to_string(),
            "Exercise improves cardiovascular health and mental well-being".to_string(),
            "Computers use binary logic and algorithms to process data".to_string(),
        ]);
        
        filler_words.insert("term".to_string(), vec![
            "Artificial Intelligence".to_string(),
            "Machine Learning".to_string(),
            "Neural Networks".to_string(),
            "Data Science".to_string(),
            "Algorithm".to_string(),
        ]);
        
        filler_words.insert("definition".to_string(), vec![
            "the simulation of human intelligence in machines".to_string(),
            "a subset of AI that enables systems to learn from data".to_string(),
            "computing systems inspired by biological neural networks".to_string(),
            "the extraction of insights from structured and unstructured data".to_string(),
            "a step-by-step procedure for solving a problem".to_string(),
        ]);
        
        filler_words.insert("example".to_string(), vec![
            "a neural network can learn to recognize images".to_string(),
            "machine learning algorithms can predict customer behavior".to_string(),
            "data scientists use statistical methods to analyze trends".to_string(),
            "AI assistants can understand natural language queries".to_string(),
            "algorithms optimize routes for delivery services".to_string(),
        ]);
        
        Self {
            templates,
            filler_words,
        }
    }
    
    fn fill_template(&self, template: &Template) -> Result<String> {
        let mut result = template.pattern.clone();
        
        for placeholder in &template.placeholders {
            if let Some(fillers) = self.filler_words.get(placeholder) {
                if !fillers.is_empty() {
                    let filler = fillers[rand::thread_rng().gen_range(0..fillers.len())].clone();
                    result = result.replace(&format!("{{{}}}", placeholder), &filler);
                }
            }
        }
        
        Ok(result)
    }
}

impl GenerationStrategy for TemplateGenerationStrategy {
    fn name(&self) -> &str {
        "template_generation"
    }
    
    fn generate(&self, prompt: &str, _config: &GeneratorConfig) -> Result<String> {
        // Simple template selection based on prompt keywords
        let selected_template = if prompt.to_lowercase().contains("question") {
            self.templates.iter().find(|t| t.id == "question_answer")
        } else if prompt.to_lowercase().contains("define") {
            self.templates.iter().find(|t| t.id == "definition")
        } else {
            self.templates.iter().find(|t| t.id == "example")
        };
        
        if let Some(template) = selected_template {
            self.fill_template(template)
        } else {
            // Fallback to random template
            let template = &self.templates[rand::thread_rng().gen_range(0..self.templates.len())];
            self.fill_template(template)
        }
    }
    
    fn metadata(&self) -> GenerationMetadata {
        GenerationMetadata {
            strategy: self.name().to_string(),
            complexity: ComplexityLevel::Intermediate,
            topic: "general".to_string(),
            style: "structured".to_string(),
            confidence: 0.7,
        }
    }
    
    fn configure(&mut self, _config: &HashMap<String, String>) -> Result<()> {
        // Template strategy doesn't need additional configuration
        Ok(())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Markov chain generation strategy
#[derive(Debug, Clone)]
pub struct MarkovChainGenerationStrategy {
    chains: HashMap<String, Vec<String>>,
    order: usize,
}

impl MarkovChainGenerationStrategy {
    pub fn new(order: usize) -> Self {
        Self {
            chains: HashMap::new(),
            order,
        }
    }
    
    pub fn train(&mut self, texts: &[String]) {
        for text in texts {
            self.add_text(text);
        }
    }
    
    fn add_text(&mut self, text: &str) {
        let words: Vec<String> = text.split_whitespace().map(|w| w.to_string()).collect();
        
        for i in 0..words.len().saturating_sub(self.order) {
            let key = words[i..i + self.order].join(" ");
            let next_word = if i + self.order < words.len() {
                words[i + self.order].clone()
            } else {
                "".to_string()
            };
            
            self.chains.entry(key).or_insert_with(Vec::new).push(next_word);
        }
    }
    
    fn generate_text(&self, start_words: &[String], max_length: usize) -> String {
        let mut result = start_words.join(" ");
        let mut current = start_words.join(" ");
        
        for _ in 0..max_length {
            if let Some(next_words) = self.chains.get(&current) {
                if !next_words.is_empty() {
                    let next_word = &next_words[rand::thread_rng().gen_range(0..next_words.len())];
                    if next_word.is_empty() {
                        break;
                    }
                    result.push(' ');
                    result.push_str(next_word);
                    
                    // Update current context
                    let mut words: Vec<String> = result.split_whitespace()
                        .rev()
                        .take(self.order)
                        .map(|s| s.to_string())
                        .collect();
                    words.reverse();
                    current = words.join(" ");
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        result
    }
}

impl GenerationStrategy for MarkovChainGenerationStrategy {
    fn name(&self) -> &str {
        "markov_chain_generation"
    }
    
    fn generate(&self, prompt: &str, config: &GeneratorConfig) -> Result<String> {
        let start_words: Vec<String> = prompt.split_whitespace()
            .take(self.order)
            .map(|w| w.to_string())
            .collect();
        
        let generated = if start_words.len() >= self.order {
            self.generate_text(&start_words, config.max_length / 5) // Rough estimate
        } else {
            // Generate from random start
            let keys: Vec<String> = self.chains.keys().cloned().collect();
            if !keys.is_empty() {
                let random_key = &keys[rand::thread_rng().gen_range(0..keys.len())];
                let start: Vec<String> = random_key.split_whitespace().map(|w| w.to_string()).collect();
                self.generate_text(&start, config.max_length / 5)
            } else {
                "Unable to generate text - no training data".to_string()
            }
        };
        
        Ok(generated)
    }
    
    fn metadata(&self) -> GenerationMetadata {
        GenerationMetadata {
            strategy: self.name().to_string(),
            complexity: ComplexityLevel::Simple,
            topic: "trained".to_string(),
            style: "natural".to_string(),
            confidence: 0.5,
        }
    }
    
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()> {
        if let Some(order) = config.get("order") {
            self.order = order.parse()?;
        }
        Ok(())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Rule-based generation strategy
#[derive(Debug, Clone)]
pub struct RuleBasedGenerationStrategy {
    rules: Vec<GenerationRule>,
}

impl RuleBasedGenerationStrategy {
    pub fn new() -> Self {
        let mut rules = Vec::new();
        
        // Add default rules
        rules.push(GenerationRule {
            pattern: "What is {topic}?".to_string(),
            responses: vec![
                "{topic} is a concept that involves...".to_string(),
                "The term {topic} refers to...".to_string(),
                "{topic} can be understood as...".to_string(),
            ],
            weight: 1.0,
        });
        
        rules.push(GenerationRule {
            pattern: "How does {topic} work?".to_string(),
            responses: vec![
                "{topic} works by...".to_string(),
                "The mechanism of {topic} involves...".to_string(),
                "To understand how {topic} works, consider...".to_string(),
            ],
            weight: 1.0,
        });
        
        Self { rules }
    }
    
    fn find_matching_rule(&self, prompt: &str) -> Option<&GenerationRule> {
        self.rules.iter().find(|rule| {
            // Simple pattern matching
            let pattern = rule.pattern.replace("{topic}", "");
            prompt.to_lowercase().contains(&pattern.to_lowercase())
        })
    }
}

/// Generation rule
#[derive(Debug, Clone)]
pub struct GenerationRule {
    pub pattern: String,
    pub responses: Vec<String>,
    pub weight: f32,
}

impl GenerationStrategy for RuleBasedGenerationStrategy {
    fn name(&self) -> &str {
        "rule_based_generation"
    }
    
    fn generate(&self, prompt: &str, _config: &GeneratorConfig) -> Result<String> {
        if let Some(rule) = self.find_matching_rule(prompt) {
            let response = &rule.responses[rand::thread_rng().gen_range(0..rule.responses.len())];
            
            // Extract topic from prompt (simplified)
            let topic = if let Some(start) = prompt.find('{') {
                if let Some(end) = prompt.find('}') {
                    &prompt[start + 1..end]
                } else {
                    "unknown"
                }
            } else {
                "concept"
            };
            
            Ok(response.replace("{topic}", topic))
        } else {
            // Fallback response
            Ok(format!("Based on your query about {}, I can provide relevant information.", 
                if prompt.len() > 10 { "this topic" } else { prompt }))
        }
    }
    
    fn metadata(&self) -> GenerationMetadata {
        GenerationMetadata {
            strategy: self.name().to_string(),
            complexity: ComplexityLevel::Simple,
            topic: "rule_based".to_string(),
            style: "formal".to_string(),
            confidence: 0.6,
        }
    }
    
    fn configure(&mut self, _config: &HashMap<String, String>) -> Result<()> {
        // Rule-based strategy doesn't need additional configuration
        Ok(())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl SyntheticGenerator {
    /// Create new synthetic generator
    pub fn new(config: GeneratorConfig) -> Self {
        let strategies: Vec<Box<dyn GenerationStrategy>> = vec![
            Box::new(TemplateGenerationStrategy::new()),
            Box::new(MarkovChainGenerationStrategy::new(2)),
            Box::new(RuleBasedGenerationStrategy::new()),
        ];
        
        Self {
            strategies,
            config,
            _templates: Vec::new(),
            _vocabulary: HashMap::new(),
        }
    }
    
    /// Add generation strategy
    pub fn add_strategy(mut self, strategy: Box<dyn GenerationStrategy>) -> Self {
        self.strategies.push(strategy);
        self
    }
    
    /// Generate synthetic data entry
    pub fn generate_entry(&self, prompt: &str, strategy_name: Option<&str>) -> Result<DataEntry> {
        let strategy = if let Some(name) = strategy_name {
            self.strategies.iter().find(|s| s.name() == name)
                .map(|s| s.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Strategy '{}' not found", name))?
        } else {
            // Select strategy based on prompt or randomly
            self.select_strategy(prompt)?
        };
        
        let content = strategy.generate(prompt, &self.config)?;
        let metadata = strategy.metadata();
        
        let mut entry = DataEntry::new(content);
        entry.metadata.insert("generation_strategy".to_string(), metadata.strategy);
        entry.metadata.insert("complexity".to_string(), format!("{:?}", metadata.complexity));
        entry.metadata.insert("topic".to_string(), metadata.topic);
        entry.metadata.insert("style".to_string(), metadata.style);
        entry.metadata.insert("confidence".to_string(), metadata.confidence.to_string());
        
        Ok(entry)
    }
    
    /// Generate batch of synthetic entries
    pub fn generate_batch(&self, prompts: &[String], strategy_name: Option<&str>) -> Result<Vec<DataEntry>> {
        let mut entries = Vec::new();
        
        for prompt in prompts {
            let entry = self.generate_entry(prompt, strategy_name)?;
            entries.push(entry);
        }
        
        Ok(entries)
    }
    
    /// Select appropriate strategy for prompt
    fn select_strategy(&self, prompt: &str) -> Result<&dyn GenerationStrategy> {
        // Simple strategy selection based on prompt characteristics
        if prompt.contains('{') && prompt.contains('}') {
            // Template-like prompt
            self.strategies.iter().find(|s| s.name() == "template_generation")
                .map(|s| s.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Template strategy not available"))
        } else if prompt.len() > 50 {
            // Longer prompt - use Markov chain
            self.strategies.iter().find(|s| s.name() == "markov_chain_generation")
                .map(|s| s.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Markov chain strategy not available"))
        } else {
            // Default to rule-based
            self.strategies.iter().find(|s| s.name() == "rule_based_generation")
                .map(|s| s.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Rule-based strategy not available"))
        }
    }
    
    /// Train Markov chain with existing data
    pub fn train_markov_chain(&mut self, texts: &[String]) -> Result<()> {
        if texts.is_empty() {
            return Err(anyhow::anyhow!("No training texts provided"));
        }
        
        // Find the MarkovChain strategy
        for strategy in &mut self.strategies {
            if let Some(markov_chain) = strategy.as_any_mut().downcast_mut::<MarkovChainGenerationStrategy>() {
                // Process all texts for training
                let mut all_words = Vec::new();
                
                for text in texts {
                    // Tokenize text into words
                    let words: Vec<String> = text
                        .split_whitespace()
                        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_lowercase())
                        .filter(|word| !word.is_empty())
                        .collect();
                    
                    all_words.extend(words);
                }
                
                if all_words.len() < 2 {
                    return Err(anyhow::anyhow!("Insufficient training data: need at least 2 words"));
                }
                
                // Train the Markov chain
                markov_chain.train(&all_words);
                
                tracing::info!(
                    word_count = all_words.len(),
                    text_count = texts.len(),
                    "Markov chain training completed successfully"
                );
                
                return Ok(());
            }
        }
        
        Err(anyhow::anyhow!("MarkovChain strategy not found"))
    }
    
    /// Get available strategies
    pub fn get_strategies(&self) -> Vec<&str> {
        self.strategies.iter().map(|s| s.name()).collect()
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: GeneratorConfig) {
        self.config = config;
    }
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            max_length: 500,
            min_length: 10,
            language: "en".to_string(),
            topics: vec!["general".to_string(), "technical".to_string(), "educational".to_string()],
            styles: vec!["formal".to_string(), "casual".to_string(), "educational".to_string()],
            complexity_levels: vec![ComplexityLevel::Simple, ComplexityLevel::Intermediate],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_generation() {
        let config = GeneratorConfig::default();
        let generator = SyntheticGenerator::new(config);
        
        let entry = generator.generate_entry("What is AI?", None).unwrap();
        assert!(!entry.content.is_empty());
        assert!(entry.metadata.contains_key("generation_strategy"));
    }

    #[test]
    fn test_template_strategy() {
        let strategy = TemplateGenerationStrategy::new();
        let config = GeneratorConfig::default();
        
        let result = strategy.generate("What is machine learning?", &config).unwrap();
        assert!(!result.is_empty());
        assert!(result.contains("Q:") || result.contains("A:"));
    }

    #[test]
    fn test_batch_generation() {
        let config = GeneratorConfig::default();
        let generator = SyntheticGenerator::new(config);
        
        let prompts = vec![
            "What is AI?".to_string(),
            "How does learning work?".to_string(),
            "Define algorithm".to_string(),
        ];
        
        let entries = generator.generate_batch(&prompts, None).unwrap();
        assert_eq!(entries.len(), 3);
        
        for entry in entries {
            assert!(!entry.content.is_empty());
        }
    }
}
