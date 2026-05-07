//! Token Mixer - Rust implementation
//! 
//! Mixes and combines tokens from different sources for training data

use anyhow::Result;
use std::collections::HashMap;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::DataEntry;

/// Token mixer for creating varied training data
pub struct TokenMixer {
    config: MixerConfig,
    token_sources: HashMap<String, TokenSource>,
    mixing_strategies: Vec<Box<dyn MixingStrategy>>,
}

/// Mixer configuration
#[derive(Debug, Clone)]
pub struct MixerConfig {
    pub max_tokens_per_entry: usize,
    pub min_tokens_per_entry: usize,
    pub preserve_order: bool,
    pub allow_duplicates: bool,
    pub shuffle_probability: f32,
    pub mixing_ratio: f32,
}

/// Token source definition
#[derive(Debug, Clone)]
pub struct TokenSource {
    pub id: String,
    pub name: String,
    pub tokens: Vec<String>,
    pub weights: HashMap<String, f32>,
    pub metadata: HashMap<String, String>,
}

/// Trait for mixing strategies
pub trait MixingStrategy: Send + Sync {
    /// Get strategy name
    fn name(&self) -> &str;
    
    /// Mix tokens from multiple sources
    fn mix(&self, sources: &[&TokenSource], config: &MixerConfig) -> Result<Vec<String>>;
    
    /// Get strategy metadata
    fn metadata(&self) -> MixingMetadata;
    
    /// Configure strategy
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()>;
}

/// Mixing metadata
#[derive(Debug, Clone)]
pub struct MixingMetadata {
    pub strategy: String,
    pub total_sources: usize,
    pub output_tokens: usize,
    pub diversity_score: f32,
}

/// Random mixing strategy
#[derive(Debug, Clone)]
pub struct RandomMixingStrategy {
    seed_distribution: f32,
}

impl RandomMixingStrategy {
    pub fn new(seed_distribution: f32) -> Self {
        Self { seed_distribution }
    }
}

impl MixingStrategy for RandomMixingStrategy {
    fn name(&self) -> &str {
        "random_mixing"
    }
    
    fn mix(&self, sources: &[&TokenSource], config: &MixerConfig) -> Result<Vec<String>> {
        let mut mixed_tokens = Vec::new();
        let mut rng = rand::thread_rng();
        
        // Calculate target number of tokens
        let total_tokens: usize = sources.iter().map(|s| s.tokens.len()).sum();
        let target_tokens = (total_tokens as f32 * config.mixing_ratio) as usize;
        let final_target = target_tokens.clamp(config.min_tokens_per_entry, config.max_tokens_per_entry);
        
        // Randomly select tokens from sources
        while mixed_tokens.len() < final_target {
            let source = sources[rng.gen_range(0..sources.len())];
            
            if source.tokens.is_empty() {
                continue;
            }
            
            let token_index = rng.gen_range(0..source.tokens.len());
            let token = &source.tokens[token_index];
            
            if config.allow_duplicates || !mixed_tokens.contains(token) {
                mixed_tokens.push(token.clone());
            }
        }
        
        // Shuffle if needed
        if !config.preserve_order && rng.gen::<f32>() < config.shuffle_probability {
            mixed_tokens.shuffle(&mut rng);
        }
        
        Ok(mixed_tokens)
    }
    
    fn metadata(&self) -> MixingMetadata {
        MixingMetadata {
            strategy: self.name().to_string(),
            total_sources: 0,
            output_tokens: 0,
            diversity_score: self.seed_distribution,
        }
    }
    
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()> {
        if let Some(seed_dist) = config.get("seed_distribution") {
            self.seed_distribution = seed_dist.parse()?;
        }
        Ok(())
    }
}

/// Weighted mixing strategy
#[derive(Debug, Clone)]
pub struct WeightedMixingStrategy {
    source_weights: HashMap<String, f32>,
}

impl WeightedMixingStrategy {
    pub fn new() -> Self {
        Self {
            source_weights: HashMap::new(),
        }
    }
    
    pub fn set_source_weight(&mut self, source_id: String, weight: f32) {
        self.source_weights.insert(source_id, weight);
    }
    
    fn select_source_by_weight<'a>(&self, sources: &[&'a TokenSource]) -> Option<&'a TokenSource> {
        let mut rng = rand::thread_rng();
        let total_weight: f32 = sources.iter()
            .map(|s| self.source_weights.get(&s.id).unwrap_or(&1.0))
            .sum();
        
        if total_weight <= 0.0 {
            return sources.get(rng.gen_range(0..sources.len())).map(|s| *s);
        }
        
        let mut random = rng.gen::<f32>() * total_weight;
        
        for source in sources {
            let weight = self.source_weights.get(&source.id).unwrap_or(&1.0);
            random -= weight;
            if random <= 0.0 {
                return Some(source);
            }
        }
        
        sources.last().copied()
    }
}

impl MixingStrategy for WeightedMixingStrategy {
    fn name(&self) -> &str {
        "weighted_mixing"
    }
    
    fn mix(&self, sources: &[&TokenSource], config: &MixerConfig) -> Result<Vec<String>> {
        let mut mixed_tokens = Vec::new();
        let mut rng = rand::thread_rng();
        
        // Calculate target number of tokens
        let total_tokens: usize = sources.iter().map(|s| s.tokens.len()).sum();
        let target_tokens = (total_tokens as f32 * config.mixing_ratio) as usize;
        let final_target = target_tokens.clamp(config.min_tokens_per_entry, config.max_tokens_per_entry);
        
        while mixed_tokens.len() < final_target {
            if let Some(source) = self.select_source_by_weight(sources) {
                if source.tokens.is_empty() {
                    continue;
                }
                
                let token_index = rng.gen_range(0..source.tokens.len());
                let token = &source.tokens[token_index];
                
                if config.allow_duplicates || !mixed_tokens.contains(token) {
                    mixed_tokens.push(token.clone());
                }
            }
        }
        
        // Shuffle if needed
        if !config.preserve_order && rng.gen::<f32>() < config.shuffle_probability {
            mixed_tokens.shuffle(&mut rng);
        }
        
        Ok(mixed_tokens)
    }
    
    fn metadata(&self) -> MixingMetadata {
        MixingMetadata {
            strategy: self.name().to_string(),
            total_sources: self.source_weights.len(),
            output_tokens: 0,
            diversity_score: 0.5,
        }
    }
    
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()> {
        for (key, value) in config {
            if key.starts_with("weight_") {
                let source_id = key.strip_prefix("weight_").unwrap_or(key);
                let weight = value.parse::<f32>()?;
                self.source_weights.insert(source_id.to_string(), weight);
            }
        }
        Ok(())
    }
}

/// Sequential mixing strategy
#[derive(Debug, Clone)]
pub struct SequentialMixingStrategy {
    round_robin: bool,
    current_source_index: usize,
}

impl SequentialMixingStrategy {
    pub fn new(round_robin: bool) -> Self {
        Self {
            round_robin,
            current_source_index: 0,
        }
    }
}

impl MixingStrategy for SequentialMixingStrategy {
    fn name(&self) -> &str {
        "sequential_mixing"
    }
    
    fn mix(&self, sources: &[&TokenSource], config: &MixerConfig) -> Result<Vec<String>> {
        let mut mixed_tokens = Vec::new();
        let mut source_index = self.current_source_index;
        
        // Calculate target number of tokens
        let total_tokens: usize = sources.iter().map(|s| s.tokens.len()).sum();
        let target_tokens = (total_tokens as f32 * config.mixing_ratio) as usize;
        let final_target = target_tokens.clamp(config.min_tokens_per_entry, config.max_tokens_per_entry);
        
        while mixed_tokens.len() < final_target {
            if sources.is_empty() {
                break;
            }
            
            let source = &sources[source_index % sources.len()];
            
            if !source.tokens.is_empty() {
                let token_index = (mixed_tokens.len() / sources.len()) % source.tokens.len();
                let token = &source.tokens[token_index];
                
                if config.allow_duplicates || !mixed_tokens.contains(token) {
                    mixed_tokens.push(token.clone());
                }
            }
            
            if self.round_robin {
                source_index += 1;
            }
        }
        
        Ok(mixed_tokens)
    }
    
    fn metadata(&self) -> MixingMetadata {
        MixingMetadata {
            strategy: self.name().to_string(),
            total_sources: 0,
            output_tokens: 0,
            diversity_score: 0.3,
        }
    }
    
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()> {
        if let Some(round_robin) = config.get("round_robin") {
            self.round_robin = round_robin.parse::<bool>()?;
        }
        Ok(())
    }
}

impl TokenMixer {
    /// Create new token mixer
    pub fn new(config: MixerConfig) -> Self {
        let mixing_strategies: Vec<Box<dyn MixingStrategy>> = vec![
            Box::new(RandomMixingStrategy::new(0.5)),
            Box::new(WeightedMixingStrategy::new()),
            Box::new(SequentialMixingStrategy::new(true)),
        ];
        
        Self {
            config,
            token_sources: HashMap::new(),
            mixing_strategies,
        }
    }
    
    /// Add token source
    pub fn add_source(&mut self, source: TokenSource) {
        self.token_sources.insert(source.id.clone(), source);
    }
    
    /// Add mixing strategy
    pub fn add_strategy(mut self, strategy: Box<dyn MixingStrategy>) -> Self {
        self.mixing_strategies.push(strategy);
        self
    }
    
    /// Mix tokens from specified sources
    pub fn mix_tokens(&self, source_ids: &[String], strategy_name: Option<&str>) -> Result<Vec<String>> {
        // Get sources
        let sources: Vec<&TokenSource> = source_ids.iter()
            .filter_map(|id| self.token_sources.get(id))
            .collect();
        
        if sources.is_empty() {
            return Err(anyhow::anyhow!("No valid token sources found"));
        }
        
        // Select strategy
        let strategy = if let Some(name) = strategy_name {
            self.mixing_strategies.iter()
                .find(|s| s.name() == name)
                .ok_or_else(|| anyhow::anyhow!("Strategy '{}' not found", name))?
        } else {
            // Default to first strategy
            &self.mixing_strategies[0]
        };
        
        strategy.mix(&sources, &self.config)
    }
    
    /// Mix tokens for data entry
    pub fn mix_for_entry(&self, source_ids: &[String], strategy_name: Option<&str>) -> Result<DataEntry> {
        let tokens = self.mix_tokens(source_ids, strategy_name)?;
        let content = tokens.join(" ");
        
        let mut entry = DataEntry::new(content);
        entry.metadata.insert("mixing_strategy".to_string(), 
            strategy_name.unwrap_or("default").to_string());
        entry.metadata.insert("source_count".to_string(), source_ids.len().to_string());
        entry.metadata.insert("token_count".to_string(), tokens.len().to_string());
        
        Ok(entry)
    }
    
    /// Create token source from data entries
    pub fn create_source_from_entries(&self, id: String, name: String, entries: &[DataEntry]) -> TokenSource {
        let mut tokens = Vec::new();
        let mut word_counts = HashMap::new();
        
        for entry in entries {
            let words: Vec<String> = entry.content.split_whitespace()
                .map(|w| w.to_lowercase())
                .collect();
            
            for word in words {
                if !word.is_empty() {
                    *word_counts.entry(word.clone()).or_insert(0) += 1;
                    if !tokens.contains(&word) {
                        tokens.push(word);
                    }
                }
            }
        }
        
        // Calculate weights based on frequency
        let total_words: usize = word_counts.values().sum();
        let weights: HashMap<String, f32> = word_counts.iter()
            .map(|(word, count)| (word.clone(), *count as f32 / total_words as f32))
            .collect();
        
        TokenSource {
            id,
            name,
            tokens,
            weights,
            metadata: HashMap::new(),
        }
    }
    
    /// Get available strategies
    pub fn get_strategies(&self) -> Vec<&str> {
        self.mixing_strategies.iter().map(|s| s.name()).collect()
    }
    
    /// Get available sources
    pub fn get_sources(&self) -> Vec<&str> {
        self.token_sources.keys().map(|s| s.as_str()).collect()
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: MixerConfig) {
        self.config = config;
    }
    
    /// Get mixing statistics
    pub fn get_statistics(&self) -> MixerStatistics {
        MixerStatistics {
            total_sources: self.token_sources.len(),
            total_strategies: self.mixing_strategies.len(),
            total_tokens: self.token_sources.values()
                .map(|s| s.tokens.len())
                .sum(),
            config: self.config.clone(),
        }
    }
}

/// Mixer statistics
#[derive(Debug, Clone)]
pub struct MixerStatistics {
    pub total_sources: usize,
    pub total_strategies: usize,
    pub total_tokens: usize,
    pub config: MixerConfig,
}

impl Default for MixerConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_entry: 1000,
            min_tokens_per_entry: 10,
            preserve_order: false,
            allow_duplicates: false,
            shuffle_probability: 0.7,
            mixing_ratio: 0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_mixing() {
        let config = MixerConfig::default();
        let mut mixer = TokenMixer::new(config);
        
        // Add test sources
        let source1 = TokenSource {
            id: "source1".to_string(),
            name: "Test Source 1".to_string(),
            tokens: vec!["hello".to_string(), "world".to_string(), "test".to_string()],
            weights: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        let source2 = TokenSource {
            id: "source2".to_string(),
            name: "Test Source 2".to_string(),
            tokens: vec!["data".to_string(), "processing".to_string(), "mixing".to_string()],
            weights: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        mixer.add_source(source1);
        mixer.add_source(source2);
        
        let mixed = mixer.mix_tokens(&["source1".to_string(), "source2".to_string()], None).unwrap();
        assert!(!mixed.is_empty());
        assert!(mixed.len() <= 1000); // max_tokens_per_entry
    }

    #[test]
    fn test_weighted_mixing() {
        let config = MixerConfig::default();
        let mut mixer = TokenMixer::new(config);
        
        let source1 = TokenSource {
            id: "source1".to_string(),
            name: "Test Source 1".to_string(),
            tokens: vec!["hello".to_string(), "world".to_string()],
            weights: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        let source2 = TokenSource {
            id: "source2".to_string(),
            name: "Test Source 2".to_string(),
            tokens: vec!["data".to_string(), "processing".to_string()],
            weights: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        mixer.add_source(source1);
        mixer.add_source(source2);
        
        let mixed = mixer.mix_tokens(&["source1".to_string(), "source2".to_string()], Some("weighted_mixing")).unwrap();
        assert!(!mixed.is_empty());
    }

    #[test]
    fn test_source_from_entries() {
        let config = MixerConfig::default();
        let mixer = TokenMixer::new(config);
        
        let entries = vec![
            DataEntry::new("Hello world test data".to_string()),
            DataEntry::new("Processing mixing tokens".to_string()),
        ];
        
        let source = mixer.create_source_from_entries(
            "test_source".to_string(),
            "Test Source".to_string(),
            &entries
        );
        
        assert!(!source.tokens.is_empty());
        assert!(source.tokens.contains(&"hello".to_string()));
        assert!(source.tokens.contains(&"world".to_string()));
    }
}
