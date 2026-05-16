//! Domain Splitter - Rust implementation
//! 
//! Classifies and splits data by domain for specialized processing

use anyhow::Result;
use std::collections::HashMap;
use regex::Regex;
use serde::{Serialize, Deserialize};

use crate::DataEntry;

/// Domain splitter for classifying data by domain
pub struct DomainSplitter {
    config: SplitterConfig,
    domain_classifiers: Vec<Box<dyn DomainClassifier>>,
    statistics: DomainStatistics,
}

/// Splitter configuration
#[derive(Debug, Clone)]
pub struct SplitterConfig {
    pub min_confidence: f32,
    pub enable_multi_domain: bool,
    pub max_domains_per_entry: usize,
    pub fallback_domain: DomainType,
    pub case_sensitive: bool,
}

/// Domain types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DomainType {
    Code,
    DeepLearningTheory,
    LlmPapers,
    General,
    Unknown,
}

impl DomainType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DomainType::Code => "code",
            DomainType::DeepLearningTheory => "dl_theory",
            DomainType::LlmPapers => "llm_papers",
            DomainType::General => "general",
            DomainType::Unknown => "unknown",
        }
    }
}

/// Domain classification result
#[derive(Debug, Clone)]
pub struct DomainClassification {
    pub domain: DomainType,
    pub confidence: f32,
    pub evidence: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Domain statistics
#[derive(Debug, Clone, Default)]
pub struct DomainStatistics {
    pub total_classified: usize,
    pub domain_counts: HashMap<DomainType, usize>,
    pub average_confidence: HashMap<DomainType, f32>,
    pub multi_domain_count: usize,
}

/// Trait for domain classifiers
pub trait DomainClassifier: Send + Sync {
    /// Get classifier name
    fn name(&self) -> &str;
    
    /// Get target domain
    fn domain(&self) -> DomainType;
    
    /// Classify text
    fn classify(&self, text: &str, config: &SplitterConfig) -> DomainClassification;
    
    /// Configure classifier
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()>;
}

/// Code domain classifier
#[derive(Debug, Clone)]
pub struct CodeClassifier {
    keywords: Vec<String>,
    patterns: Vec<Regex>,
    file_extensions: Vec<String>,
}

impl CodeClassifier {
    pub fn new() -> Result<Self> {
        let keywords = vec![
            "def ".to_string(),
            "function".to_string(),
            "class ".to_string(),
            "import ".to_string(),
            "#include".to_string(),
            "public ".to_string(),
            "private ".to_string(),
            "return ".to_string(),
            "if (".to_string(),
            "for (".to_string(),
            "while (".to_string(),
            "=>".to_string(),
            "lambda".to_string(),
            "var ".to_string(),
            "let ".to_string(),
            "const ".to_string(),
            "print(".to_string(),
            "console.log".to_string(),
            "SELECT".to_string(),
            "INSERT".to_string(),
            "UPDATE".to_string(),
            "DELETE".to_string(),
            "CREATE".to_string(),
            "DROP".to_string(),
        ];
        
        let patterns: Vec<Regex> = vec![
            Regex::new(r"function\s+\w+\s*\(")?,
            Regex::new(r"class\s+\w+\s*\{")?,
            Regex::new(r"def\s+\w+\s*\(")?,
            Regex::new(r"#include\s*<[^>]+>")?,
            Regex::new(r"import\s+\w+")?,
            Regex::new(r"public\s+class\s+\w+")?,
            Regex::new(r"private\s+\w+\s*\(")?,
            Regex::new(r"const\s+\w+\s*=")?,
            Regex::new(r"let\s+\w+\s*=")?,
            Regex::new(r"var\s+\w+\s*=")?,
        ].into_iter().filter_map(|regex| Some(regex)).collect();
        
        let file_extensions = vec![
            ".rs".to_string(),
            ".py".to_string(),
            ".js".to_string(),
            ".ts".to_string(),
            ".java".to_string(),
            ".cpp".to_string(),
            ".c".to_string(),
            ".h".to_string(),
            ".cs".to_string(),
            ".php".to_string(),
            ".rb".to_string(),
            ".go".to_string(),
            ".rs".to_string(),
            ".swift".to_string(),
            ".kt".to_string(),
            ".scala".to_string(),
            ".sql".to_string(),
            ".html".to_string(),
            ".css".to_string(),
            ".xml".to_string(),
            ".json".to_string(),
            ".yaml".to_string(),
            ".yml".to_string(),
            ".toml".to_string(),
            ".sh".to_string(),
            ".bat".to_string(),
            ".ps1".to_string(),
        ];
        
        Ok(Self {
            keywords,
            patterns,
            file_extensions,
        })
    }
    
    fn count_keyword_matches(&self, text: &str) -> usize {
        let text_lower = text.to_lowercase();
        self.keywords.iter()
            .filter(|keyword| text_lower.contains(&**keyword))
            .count()
    }
    
    fn count_pattern_matches(&self, text: &str) -> usize {
        self.patterns.iter()
            .filter(|pattern| pattern.is_match(text))
            .count()
    }
    
    fn check_file_extension(&self, text: &str) -> bool {
        self.file_extensions.iter()
            .any(|ext| text.to_lowercase().contains(ext))
            || text.lines()
                .any(|line| self.file_extensions.iter().any(|ext| line.trim().ends_with(ext)))
    }
}

impl DomainClassifier for CodeClassifier {
    fn name(&self) -> &str {
        "code_classifier"
    }
    
    fn domain(&self) -> DomainType {
        DomainType::Code
    }
    
    fn classify(&self, text: &str, _config: &SplitterConfig) -> DomainClassification {
        let mut evidence = Vec::new();
        let mut score = 0.0f32;
        
        // Check keywords
        let keyword_matches = self.count_keyword_matches(text);
        if keyword_matches > 0 {
            score += keyword_matches as f32 * 0.3;
            evidence.push(format!("Found {} code keywords", keyword_matches));
        }
        
        // Check patterns
        let pattern_matches = self.count_pattern_matches(text);
        if pattern_matches > 0 {
            score += pattern_matches as f32 * 0.4;
            evidence.push(format!("Found {} code patterns", pattern_matches));
        }
        
        // Check file extensions
        if self.check_file_extension(text) {
            score += 0.3;
            evidence.push("Found code file extensions".to_string());
        }
        
        // Normalize score
        let confidence = (score / 2.0).min(1.0);
        
        let mut metadata = HashMap::new();
        metadata.insert("keyword_matches".to_string(), keyword_matches.to_string());
        metadata.insert("pattern_matches".to_string(), pattern_matches.to_string());
        
        DomainClassification {
            domain: DomainType::Code,
            confidence,
            evidence,
            metadata,
        }
    }
    
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()> {
        if let Some(keywords_str) = config.get("keywords") {
            self.keywords = keywords_str.split(',').map(|s| s.trim().to_string()).collect();
        }
        Ok(())
    }
}

/// Deep Learning Theory classifier
#[derive(Debug, Clone)]
pub struct DeepLearningTheoryClassifier {
    keywords: Vec<String>,
    concepts: HashMap<String, Vec<String>>,
}

impl DeepLearningTheoryClassifier {
    pub fn new() -> Self {
        let keywords = vec![
            "neural network".to_string(),
            "backpropagation".to_string(),
            "gradient descent".to_string(),
            "loss function".to_string(),
            "activation function".to_string(),
            "convolution".to_string(),
            "recurrent".to_string(),
            "transformer".to_string(),
            "attention".to_string(),
            "optimization".to_string(),
            "regularization".to_string(),
            "batch normalization".to_string(),
            "dropout".to_string(),
            "training".to_string(),
            "learning rate".to_string(),
            "forward pass".to_string(),
            "backward pass".to_string(),
            "weight initialization".to_string(),
            "bias".to_string(),
            "epoch".to_string(),
            "batch".to_string(),
            "overfitting".to_string(),
            "underfitting".to_string(),
            "cross-validation".to_string(),
            "stochastic gradient descent".to_string(),
            "adam optimizer".to_string(),
            "rmsprop".to_string(),
            "momentum".to_string(),
            "softmax".to_string(),
            "relu".to_string(),
            "sigmoid".to_string(),
            "tanh".to_string(),
            "embedding".to_string(),
            "feature extraction".to_string(),
        ];
        
        let mut concepts = HashMap::new();
        concepts.insert("architectures".to_string(), vec![
            "cnn".to_string(),
            "rnn".to_string(),
            "lstm".to_string(),
            "gru".to_string(),
            "gan".to_string(),
            "vae".to_string(),
            "autoencoder".to_string(),
            "resnet".to_string(),
            "inception".to_string(),
            "vgg".to_string(),
        ]);
        
        concepts.insert("optimizers".to_string(), vec![
            "sgd".to_string(),
            "adam".to_string(),
            "rmsprop".to_string(),
            "adagrad".to_string(),
            "adadelta".to_string(),
            "adamw".to_string(),
            "nadam".to_string(),
        ]);
        
        Self {
            keywords,
            concepts,
        }
    }
    
    fn count_keyword_matches(&self, text: &str) -> usize {
        let text_lower = text.to_lowercase();
        self.keywords.iter()
            .filter(|keyword| text_lower.contains(&**keyword))
            .count()
    }
    
    fn count_concept_matches(&self, text: &str) -> HashMap<String, usize> {
        let text_lower = text.to_lowercase();
        let mut matches = HashMap::new();
        
        for (concept, terms) in &self.concepts {
            let count = terms.iter()
                .filter(|term| text_lower.contains(term.as_str()))
                .count();
            if count > 0 {
                matches.insert(concept.clone(), count);
            }
        }
        
        matches
    }
}

impl DomainClassifier for DeepLearningTheoryClassifier {
    fn name(&self) -> &str {
        "deep_learning_theory_classifier"
    }
    
    fn domain(&self) -> DomainType {
        DomainType::DeepLearningTheory
    }
    
    fn classify(&self, text: &str, _config: &SplitterConfig) -> DomainClassification {
        let mut evidence = Vec::new();
        let mut score = 0.0f32;
        
        // Check keywords
        let keyword_matches = self.count_keyword_matches(text);
        if keyword_matches > 0 {
            score += keyword_matches as f32 * 0.4;
            evidence.push(format!("Found {} DL keywords", keyword_matches));
        }
        
        // Check concepts
        let concept_matches = self.count_concept_matches(text);
        if !concept_matches.is_empty() {
            score += concept_matches.len() as f32 * 0.3;
            for (concept, count) in &concept_matches {
                evidence.push(format!("Found {} {} terms", count, concept));
            }
        }
        
        // Check for mathematical notation
        if text.contains("∂") || text.contains("∇") || text.contains("∑") || text.contains("∏") {
            score += 0.2;
            evidence.push("Found mathematical notation".to_string());
        }
        
        // Check for equations
        if text.contains("=") && (text.contains("∂") || text.contains("∇")) {
            score += 0.1;
            evidence.push("Found mathematical equations".to_string());
        }
        
        // Normalize score
        let confidence = (score / 3.0).min(1.0);
        
        let mut metadata = HashMap::new();
        metadata.insert("keyword_matches".to_string(), keyword_matches.to_string());
        metadata.insert("concept_matches".to_string(), concept_matches.len().to_string());
        
        DomainClassification {
            domain: DomainType::DeepLearningTheory,
            confidence,
            evidence,
            metadata,
        }
    }
    
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()> {
        if let Some(keywords_str) = config.get("keywords") {
            self.keywords = keywords_str.split(',').map(|s| s.trim().to_string()).collect();
        }
        Ok(())
    }
}

/// LLM Papers classifier
#[derive(Debug, Clone)]
pub struct LlmPapersClassifier {
    model_names: Vec<String>,
    paper_terms: Vec<String>,
    research_concepts: Vec<String>,
}

impl LlmPapersClassifier {
    pub fn new() -> Self {
        let model_names = vec![
            "gpt".to_string(),
            "bert".to_string(),
            "transformer".to_string(),
            "t5".to_string(),
            "roberta".to_string(),
            "albert".to_string(),
            "electra".to_string(),
            "deberta".to_string(),
            "llama".to_string(),
            "chatgpt".to_string(),
            "claude".to_string(),
            "palm".to_string(),
            "gemini".to_string(),
            "bard".to_string(),
            "copilot".to_string(),
            "codex".to_string(),
            "davinci".to_string(),
            "curie".to_string(),
            "babbage".to_string(),
            "ada".to_string(),
        ];
        
        let paper_terms = vec![
            "pre-training".to_string(),
            "fine-tuning".to_string(),
            "attention mechanism".to_string(),
            "self-attention".to_string(),
            "multi-head attention".to_string(),
            "positional encoding".to_string(),
            "tokenization".to_string(),
            "subword".to_string(),
            "byte-pair encoding".to_string(),
            "wordpiece".to_string(),
            "sentencepiece".to_string(),
            "masked language modeling".to_string(),
            "next sentence prediction".to_string(),
            "causal language modeling".to_string(),
            "autoregressive".to_string(),
            "bidirectional".to_string(),
            "encoder-decoder".to_string(),
            "encoder-only".to_string(),
            "decoder-only".to_string(),
            "scaling laws".to_string(),
            "emergent abilities".to_string(),
            "in-context learning".to_string(),
            "few-shot learning".to_string(),
            "zero-shot learning".to_string(),
            "chain-of-thought".to_string(),
            "instruction tuning".to_string(),
            "reinforcement learning from human feedback".to_string(),
            "constitutional ai".to_string(),
        ];
        
        let research_concepts = vec![
            "parameters".to_string(),
            "tokens".to_string(),
            "vocabulary".to_string(),
            "context window".to_string(),
            "temperature".to_string(),
            "top-k".to_string(),
            "top-p".to_string(),
            "beam search".to_string(),
            "nucleus sampling".to_string(),
            "perplexity".to_string(),
            "bleu".to_string(),
            "rouge".to_string(),
            "benchmark".to_string(),
            "evaluation".to_string(),
            "dataset".to_string(),
            "corpus".to_string(),
        ];
        
        Self {
            model_names,
            paper_terms,
            research_concepts,
        }
    }
    
    fn count_model_mentions(&self, text: &str) -> usize {
        let text_lower = text.to_lowercase();
        self.model_names.iter()
            .filter(|model| text_lower.contains(&**model))
            .count()
    }
    
    fn count_paper_terms(&self, text: &str) -> usize {
        let text_lower = text.to_lowercase();
        self.paper_terms.iter()
            .filter(|term| text_lower.contains(term.as_str()))
            .count()
    }
    
    fn count_research_concepts(&self, text: &str) -> usize {
        let text_lower = text.to_lowercase();
        self.research_concepts.iter()
            .filter(|concept| text_lower.contains(&**concept))
            .count()
    }
}

impl DomainClassifier for LlmPapersClassifier {
    fn name(&self) -> &str {
        "llm_papers_classifier"
    }
    
    fn domain(&self) -> DomainType {
        DomainType::LlmPapers
    }
    
    fn classify(&self, text: &str, _config: &SplitterConfig) -> DomainClassification {
        let mut evidence = Vec::new();
        let mut score = 0.0f32;
        
        // Check model names
        let model_mentions = self.count_model_mentions(text);
        if model_mentions > 0 {
            score += model_mentions as f32 * 0.4;
            evidence.push(format!("Found {} model mentions", model_mentions));
        }
        
        // Check paper terms
        let paper_terms = self.count_paper_terms(text);
        if paper_terms > 0 {
            score += paper_terms as f32 * 0.3;
            evidence.push(format!("Found {} paper terms", paper_terms));
        }
        
        // Check research concepts
        let research_concepts = self.count_research_concepts(text);
        if research_concepts > 0 {
            score += research_concepts as f32 * 0.2;
            evidence.push(format!("Found {} research concepts", research_concepts));
        }
        
        // Check for academic indicators
        if text.contains("paper") || text.contains("research") || text.contains("study") || 
           text.contains("experiment") || text.contains("results") || text.contains("conclusion") {
            score += 0.1;
            evidence.push("Found academic indicators".to_string());
        }
        
        // Normalize score
        let confidence = (score / 2.0).min(1.0);
        
        let mut metadata = HashMap::new();
        metadata.insert("model_mentions".to_string(), model_mentions.to_string());
        metadata.insert("paper_terms".to_string(), paper_terms.to_string());
        metadata.insert("research_concepts".to_string(), research_concepts.to_string());
        
        DomainClassification {
            domain: DomainType::LlmPapers,
            confidence,
            evidence,
            metadata,
        }
    }
    
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()> {
        if let Some(models_str) = config.get("model_names") {
            self.model_names = models_str.split(',').map(|s| s.trim().to_string()).collect();
        }
        Ok(())
    }
}

/// General domain classifier (fallback)
#[derive(Debug, Clone)]
pub struct GeneralClassifier;

impl GeneralClassifier {
    pub fn new() -> Self {
        Self
    }
}

impl DomainClassifier for GeneralClassifier {
    fn name(&self) -> &str {
        "general_classifier"
    }
    
    fn domain(&self) -> DomainType {
        DomainType::General
    }
    
    fn classify(&self, _text: &str, _config: &SplitterConfig) -> DomainClassification {
        let evidence = vec!["Fallback classification".to_string()];
        
        DomainClassification {
            domain: DomainType::General,
            confidence: 0.3,
            evidence,
            metadata: HashMap::new(),
        }
    }
    
    fn configure(&mut self, _config: &HashMap<String, String>) -> Result<()> {
        Ok(())
    }
}

impl DomainSplitter {
    /// Create new domain splitter
    pub fn new(config: SplitterConfig) -> Result<Self> {
        let domain_classifiers: Vec<Box<dyn DomainClassifier>> = vec![
            Box::new(CodeClassifier::new()?),
            Box::new(DeepLearningTheoryClassifier::new()),
            Box::new(LlmPapersClassifier::new()),
            Box::new(GeneralClassifier::new()),
        ];
        
        Ok(Self {
            config,
            domain_classifiers,
            statistics: DomainStatistics::default(),
        })
    }
    
    /// Add domain classifier
    pub fn add_classifier(mut self, classifier: Box<dyn DomainClassifier>) -> Self {
        self.domain_classifiers.push(classifier);
        self
    }
    
    /// Classify data entry
    pub fn classify_entry(&mut self, entry: &DataEntry) -> Vec<DomainClassification> {
        let text = if self.config.case_sensitive {
            entry.content.clone()
        } else {
            entry.content.to_lowercase()
        };
        
        let mut classifications = Vec::new();
        
        for classifier in &self.domain_classifiers {
            let classification = classifier.classify(&text, &self.config);
            
            if classification.confidence >= self.config.min_confidence {
                classifications.push(classification);
            }
        }
        
        // Sort by confidence
        classifications.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).expect("non-NaN float values"));
        
        // Limit number of domains if multi-domain is disabled
        if !self.config.enable_multi_domain && !classifications.is_empty() {
            classifications.truncate(1);
        } else if self.config.enable_multi_domain {
            classifications.truncate(self.config.max_domains_per_entry);
        }
        
        // Fallback to unknown if no classifications
        if classifications.is_empty() {
            classifications.push(DomainClassification {
                domain: self.config.fallback_domain.clone(),
                confidence: 0.0,
                evidence: vec!["No domain matched".to_string()],
                metadata: HashMap::new(),
            });
        }
        
        // Update statistics
        self.statistics.total_classified += 1;
        for classification in &classifications {
            *self.statistics.domain_counts.entry(classification.domain.clone()).or_insert(0) += 1;
            
            let confidences = self.statistics.average_confidence.entry(classification.domain.clone()).or_insert(0.0);
            *confidences = (*confidences + classification.confidence) / 2.0;
        }
        
        if classifications.len() > 1 {
            self.statistics.multi_domain_count += 1;
        }
        
        classifications
    }
    
    /// Classify batch of entries
    pub fn classify_batch(&mut self, entries: &[DataEntry]) -> Vec<Vec<DomainClassification>> {
        entries.iter().map(|entry| self.classify_entry(entry)).collect()
    }
    
    /// Filter entries by domain
    pub fn filter_by_domain<'a>(&mut self, entries: &'a [DataEntry], domain: DomainType) -> Vec<&'a DataEntry> {
        let mut filtered = Vec::new();
        
        for entry in entries {
            let classifications = self.classify_entry(entry);
            if classifications.iter().any(|c| c.domain == domain) {
                filtered.push(entry);
            }
        }
        
        filtered
    }
    
    /// Get domain distribution
    pub fn get_domain_distribution(&mut self, entries: &[DataEntry]) -> HashMap<DomainType, usize> {
        let mut distribution = HashMap::new();
        
        for entry in entries {
            let classifications = self.classify_entry(entry);
            for classification in classifications {
                *distribution.entry(classification.domain).or_insert(0) += 1;
            }
        }
        
        distribution
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> &DomainStatistics {
        &self.statistics
    }
    
    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = DomainStatistics::default();
    }
    
    /// Get available domains
    pub fn get_available_domains(&self) -> Vec<DomainType> {
        self.domain_classifiers.iter().map(|c| c.domain()).collect()
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: SplitterConfig) {
        self.config = config;
    }
}

impl Default for SplitterConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            enable_multi_domain: false,
            max_domains_per_entry: 2,
            fallback_domain: DomainType::Unknown,
            case_sensitive: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_classification() {
        let config = SplitterConfig::default();
        let mut splitter = DomainSplitter::new(config).unwrap();
        
        let code_entry = DataEntry::new("function hello() { console.log('Hello World'); return true; }".to_string());
        let classifications = splitter.classify_entry(&code_entry);
        
        assert!(!classifications.is_empty());
        assert_eq!(classifications[0].domain, DomainType::Code);
        assert!(classifications[0].confidence > 0.5);
    }

    #[test]
    fn test_dl_theory_classification() {
        let config = SplitterConfig::default();
        let mut splitter = DomainSplitter::new(config).unwrap();
        
        let dl_entry = DataEntry::new("Neural networks use backpropagation and gradient descent for training with loss functions and activation functions.".to_string());
        let classifications = splitter.classify_entry(&dl_entry);
        
        assert!(!classifications.is_empty());
        assert_eq!(classifications[0].domain, DomainType::DeepLearningTheory);
        assert!(classifications[0].confidence > 0.5);
    }

    #[test]
    fn test_llm_papers_classification() {
        let config = SplitterConfig::default();
        let mut splitter = DomainSplitter::new(config).unwrap();
        
        let llm_entry = DataEntry::new("This paper discusses GPT-4 and transformer architectures with attention mechanisms and pre-training strategies for large language models.".to_string());
        let classifications = splitter.classify_entry(&llm_entry);
        
        assert!(!classifications.is_empty());
        assert_eq!(classifications[0].domain, DomainType::LlmPapers);
        assert!(classifications[0].confidence > 0.5);
    }

    #[test]
    fn test_batch_classification() {
        let config = SplitterConfig::default();
        let mut splitter = DomainSplitter::new(config).unwrap();
        
        let entries = vec![
            DataEntry::new("def python_function(): pass".to_string()),
            DataEntry::new("Neural network training with backpropagation".to_string()),
            DataEntry::new("GPT models use transformer architecture".to_string()),
        ];
        
        let classifications = splitter.classify_batch(&entries);
        assert_eq!(classifications.len(), 3);
        
        for classification in classifications {
            assert!(!classification.is_empty());
        }
    }
}
