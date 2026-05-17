//! Nexora Memory - Advanced memory management system
//! 
//! Module ini menyediakan memory management dengan 4 layers untuk Nexora AI system
//!
//! Cognitive Dynamics Extension — mengubah memory system menjadi unified cognitive

#![allow(dead_code, unused_variables)]
//! dynamical system dengan conservation law, phase coherence, attention curvature,
//! meta-learning, dan identity persistence.

pub mod layers;
pub mod episodic;
pub mod cache;
pub mod compression;
pub mod memory;
pub mod memory_model;
pub mod lru_memory;
pub mod optimizer;
pub mod types;
pub mod core;
pub mod conservation;
pub mod coherence;
pub mod curvature;
pub mod meta_learning;
pub mod identity;

pub use layers::{MemoryLayers, MemoryLayer};
pub use episodic::{EpisodicMemory, MemoryEpisode};
pub use cache::{LRUCache, MemoryCache};
pub use compression::{ContextCompressor, CompressedContext};
pub use memory_model::{
    HebbianMemory, MemoryEntry, MemoryType, HebbianMemoryConfig,
    NeuralAttentionMemory, NeuralAttentionMemoryConfig, NeuralMemoryEntry,
};
pub use types::*;
pub use optimizer::{
    MemoryPool, MemoryPoolConfig, MemoryPoolStatistics, MemoryLeakDetector,
    MemoryOptimizer, MemoryOptimizerStatistics, PotentialLeak, LeakStatistics,
};
pub use conservation::EnergyConservation;
pub use coherence::{CoherenceField, EmergenceOperator};
pub use curvature::CurvatureField;
pub use meta_learning::MetaLearningTensor;
pub use identity::IdentityPersistence;

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::debug;

/// Memory management system dengan multi-layer architecture
#[derive(Debug)]
pub struct MemoryManager {
    layers: Arc<RwLock<MemoryLayers>>,
    episodic: Arc<RwLock<EpisodicMemory>>,
    cache: Arc<RwLock<LRUCache<String, String>>>,
    compressor: Arc<RwLock<ContextCompressor>>,
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            layers: Arc::new(RwLock::new(MemoryLayers::new())),
            episodic: Arc::new(RwLock::new(EpisodicMemory::new(1000))),
            cache: Arc::new(RwLock::new(LRUCache::new(100))),
            compressor: Arc::new(RwLock::new(ContextCompressor::new())),
        }
    }
    
    /// Store data ke memory layer yang sesuai
    pub async fn store(&self, layer: MemoryLayer, key: &str, value: &str) -> Result<()> {
        debug!("Storing to {:?}: {} = {}", layer, key, value);
        
        // Store di layer yang ditentukan
        {
            let mut layers = self.layers.write().await;
            layers.store(layer, key, value).await?;
        }
        
        // Jika episodic, simpan juga di episodic memory
        if layer == MemoryLayer::Session {
            let mut episodic = self.episodic.write().await;
            episodic.add_episode(key, value).await?;
        }
        
        // Cache frequently accessed data - use separate scope to avoid deadlock
        if layer == MemoryLayer::Short {
            let cache_key = key.to_string();
            let cache_value = value.to_string();
            let mut cache = self.cache.write().await;
            cache.put(cache_key, cache_value);
        }
        
        Ok(())
    }
    
    /// Retrieve data dari memory layer
    pub async fn retrieve(&self, layer: MemoryLayer, key: &str) -> Result<Option<String>> {
        debug!("Retrieving from {:?}: {}", layer, key);
        
        // Cek cache dulu untuk short memory
        if layer == MemoryLayer::Short {
            let mut cache = self.cache.write().await;
            if let Some(value) = cache.get(&key.to_string()) {
                return Ok(Some(value.clone()));
            }
        }
        
        // Retrieve dari layer yang ditentukan
        let mut layers = self.layers.write().await;
        layers.retrieve(layer, key).await
    }
    
    /// Search di semua memory layers
    pub async fn search(&self, query: &str) -> Result<Vec<MemorySearchResult>> {
        debug!("Searching memory: {}", query);
        
        let mut results = Vec::with_capacity(5);
        
        // Search di setiap layer
        let layers = self.layers.read().await;
        for layer in [MemoryLayer::Short, MemoryLayer::Session, MemoryLayer::Long, MemoryLayer::Knowledge] {
            if let Some(layer_results) = layers.search(layer, query).await? {
                results.extend(layer_results);
            }
        }
        
        // Search di episodic memory
        let episodic = self.episodic.read().await;
        if let Some(episodic_results) = episodic.search(query).await? {
            results.extend(episodic_results);
        }
        
        Ok(results)
    }
    
    /// Compress context untuk storage yang lebih efisien
    pub async fn compress_context(&self, context: &str) -> Result<CompressedContext> {
        // Use shared compressor instance instead of creating new one
        let mut compressor = self.compressor.write().await;
        compressor.compress(context).await
    }
    
    /// Decompress context
    pub async fn decompress_context(&self, compressed: &CompressedContext) -> Result<String> {
        debug!("Decompressing context");
        
        // Use shared compressor instance instead of creating new one
        let compressor = self.compressor.read().await;
        compressor.decompress(compressed).await
    }
    
    /// Get memory statistics
    pub async fn get_stats(&self) -> Result<MemoryStats> {
        let layers = self.layers.read().await;
        let episodic = self.episodic.read().await;
        let cache = self.cache.read().await;
        let compressor = self.compressor.read().await;
        
        Ok(MemoryStats {
            layer_stats: layers.get_stats().await?,
            episodic_count: episodic.count().await,
            cache_size: cache.len(),
            compression_ratio: compressor.get_compression_ratio(),
        })
    }
    
    /// Delete data from memory layer
    pub async fn delete(&self, layer: MemoryLayer, key: &str) -> Result<()> {
        debug!("Deleting from {:?}: {}", layer, key);
        
        let mut layers = self.layers.write().await;
        layers.delete(layer, key).await
    }
    
    /// Store generation data
    pub async fn store_generation(&self, prompt: &str, generated_text: &str) -> Result<()> {
        self.store(MemoryLayer::Session, &format!("generation:{}", prompt), generated_text).await
    }
    
    /// Store code analysis data
    pub async fn store_code_analysis(&self, code: &str, analysis: &str) -> Result<()> {
        self.store(MemoryLayer::Long, &format!("code_analysis:{}", code), analysis).await
    }
    
    /// Store code generation data
    pub async fn store_code_generation(&self, description: &str, generated_code: &str) -> Result<()> {
        self.store(MemoryLayer::Long, &format!("code_gen:{}", description), generated_code).await
    }
    
    /// Store data (alias for store method)
    pub async fn store_data(&self, layer: MemoryLayer, key: &str, value: &str) -> Result<()> {
        self.store(layer, key, value).await
    }
    
    /// Store session data
    pub async fn store_session(&self, key: &str, value: &str) -> Result<()> {
        self.store(MemoryLayer::Session, key, value).await
    }
    
    /// Store long-term data
    pub async fn store_long_term(&self, key: &str, value: &str) -> Result<()> {
        self.store(MemoryLayer::Long, key, value).await
    }
    
    /// Store knowledge data
    pub async fn store_knowledge(&self, key: &str, value: &str) -> Result<()> {
        self.store(MemoryLayer::Knowledge, key, value).await
    }
    
    /// Store interaction data
    pub async fn store_interaction(&self, input: &str, response: &str) -> Result<()> {
        self.store(MemoryLayer::Session, &format!("interaction:{}", input), response).await
    }
    
    /// Clear short-term memory
    pub async fn clear_short_term(&self) -> Result<()> {
        let mut layers = self.layers.write().await;
        layers.clear_layer(MemoryLayer::Short).await
    }
    
    /// Clear session memory
    pub async fn clear_session(&self) -> Result<()> {
        let mut layers = self.layers.write().await;
        layers.clear_layer(MemoryLayer::Session).await
    }
    
    /// Clear long-term memory
    pub async fn clear_long_term(&self) -> Result<()> {
        let mut layers = self.layers.write().await;
        layers.clear_layer(MemoryLayer::Long).await
    }
    
    /// Clear knowledge memory
    pub async fn clear_knowledge(&self) -> Result<()> {
        let mut layers = self.layers.write().await;
        layers.clear_layer(MemoryLayer::Knowledge).await
    }
    
    /// Store short-term data
    pub async fn store_short_term(&self, key: &str, value: &str) -> Result<()> {
        self.store(MemoryLayer::Short, key, value).await
    }
    
    /// Get short-term data
    pub async fn get_short_term_data(&self) -> Result<std::collections::HashMap<String, String>> {
        let layers = self.layers.read().await;
        let entries = layers.get_layer_data(MemoryLayer::Short).await?;
        let mut result = HashMap::new();
        for entry in entries {
            result.insert(entry.key, entry.value);
        }
        Ok(result)
    }
    
    /// Get session data
    pub async fn get_session_data(&self) -> Result<std::collections::HashMap<String, String>> {
        let layers = self.layers.read().await;
        let entries = layers.get_layer_data(MemoryLayer::Session).await?;
        let mut result = HashMap::new();
        for entry in entries {
            result.insert(entry.key, entry.value);
        }
        Ok(result)
    }
    
    /// Get long-term data
    pub async fn get_long_term_data(&self) -> Result<std::collections::HashMap<String, String>> {
        let layers = self.layers.read().await;
        let entries = layers.get_layer_data(MemoryLayer::Long).await?;
        let mut result = HashMap::new();
        for entry in entries {
            result.insert(entry.key, entry.value);
        }
        Ok(result)
    }
    
    /// Get knowledge data
    pub async fn get_knowledge_data(&self) -> Result<std::collections::HashMap<String, String>> {
        let layers = self.layers.read().await;
        let entries = layers.get_layer_data(MemoryLayer::Knowledge).await?;
        let mut result = HashMap::new();
        for entry in entries {
            result.insert(entry.key, entry.value);
        }
        Ok(result)
    }
}

/// Memory search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResult {
    pub layer: MemoryLayer,
    pub key: String,
    pub value: String,
    pub relevance_score: f32,
    pub timestamp: u64,
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub layer_stats: HashMap<MemoryLayer, usize>,
    pub episodic_count: usize,
    pub cache_size: usize,
    pub compression_ratio: f32,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// ============================================================================
/// Cognitive Dynamics — Unified Cognitive Dynamical System
/// ============================================================================
///
/// Final Unified Equation:
///
/// Ψ_I = ∫[w·Φ + ΣT_ijk Φ_i Φ_j Φ_k − γ||Φ||² + R(x,t)] ΛGC_φ dx
///
/// Emergence:
///   Ω = σ(κ(Ψ̇ − Ξ̇)) · C_φ · I_d · K(t)
///
/// Entropy:
///   Ξ = αN + βI + δF
///
/// Komponen:
///   Ψ_I : information field intensity
///   Φ   : cognitive field
///   T   : coupling tensor (3rd order)
///   R   : attention curvature (∇²Φ)
///   ΛGC_φ : Gabor Cognitive field (wavelet window)
///   C_φ : phase coherence
///   I_d : identity persistence
///   K   : meta-learning rate
///   Ξ   : cognitive entropy (αN + βI + δF)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveDynamics {
    pub field_intensity: f64,
    pub entropy: f64,
    pub emergence: f64,
    pub coherence: f64,
    pub identity: f64,
    pub meta_rate: f64,
    pub field_energy: f64,

    // Parameters
    pub w: f64,           // linear coupling weight
    pub gamma: f64,       // ||Φ||² penalty
    pub alpha: f64,       // neural noise coefficient
    pub beta: f64,        // interference coefficient
    pub delta: f64,       // forgetting coefficient
    pub kappa: f64,       // emergence gain

    // Entropy components
    pub neural_entropy: f64,     // αN
    pub interference: f64,       // βI
    pub forgetting: f64,         // δF
}

impl CognitiveDynamics {
    pub fn new() -> Self {
        Self {
            field_intensity: 0.0,
            entropy: 0.0,
            emergence: 0.0,
            coherence: 0.0,
            identity: 0.0,
            meta_rate: 0.01,
            field_energy: 0.0,
            w: 1.0,
            gamma: 0.1,
            alpha: 0.3,
            beta: 0.2,
            delta: 0.1,
            kappa: 1.0,
            neural_entropy: 0.0,
            interference: 0.0,
            forgetting: 0.0,
        }
    }

    /// Compute field intensity:
    /// Ψ_I = ∫[w·Φ + ΣT_ijk Φ_i Φ_j Φ_k − γ||Φ||² + R(x,t)] ΛGC_φ dx
    pub fn compute_field_intensity(
        &mut self,
        field: &[f64],
        curvature: &[f64],
        coupling: f64,
    ) -> f64 {
        let n = field.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let field_norm2: f64 = field.iter().map(|v| v * v).sum();
        let field_norm4: f64 = field.iter().map(|v| v * v * v * v).sum();

        // Linear term: w·Φ
        let linear: f64 = field.iter().map(|v| v * coupling).sum();

        // Cubic coupling: ΣT_ijk Φ_i Φ_j Φ_k — approximated as coupling * ΣΦ_i³
        let cubic: f64 = field.iter().map(|v| coupling * v * v * v).sum();

        // Regularization: −γ||Φ||²
        let regularization = self.gamma * field_norm2;

        // Curvature contribution
        let curvature_sum: f64 = curvature.iter().sum();

        // Gabor Cognitive field (ΛGC_φ) — Gaussian window approximation
        let gabor_window: f64 = (0..field.len())
            .map(|i| {
                let x = i as f64 / n;
                (-x * x * 10.0).exp() * field[i]
            })
            .sum();

        // Ψ_I = sum over field points
        let psi_i = (linear + cubic - regularization + curvature_sum) * gabor_window / n;

        self.field_intensity = psi_i.max(0.0);
        self.field_energy = field_norm2;
        self.field_intensity
    }

    /// Compute entropy: Ξ = αN + βI + δF
    pub fn compute_entropy(
        &mut self,
        neural_noise: f64,
        interference_level: f64,
        forgetting_level: f64,
    ) -> f64 {
        self.neural_entropy = self.alpha * neural_noise;
        self.interference = self.beta * interference_level;
        self.forgetting = self.delta * forgetting_level;
        self.entropy = self.neural_entropy + self.interference + self.forgetting;
        self.entropy
    }

    /// Compute emergence:
    /// Ω = σ(κ(Ψ̇ − Ξ̇)) · C_φ · I_d · K(t)
    pub fn compute_emergence(
        &mut self,
        psi_dot: f64,
        xi_dot: f64,
        coherence: f64,
        identity: f64,
        meta_rate: f64,
    ) -> f64 {
        // Sigmoid gate
        let growth_signal = self.kappa * (psi_dot - xi_dot);
        let gate = 1.0 / (1.0 + (-growth_signal).exp());

        let emergence = gate * coherence * identity * meta_rate;

        self.coherence = coherence;
        self.identity = identity;
        self.meta_rate = meta_rate;
        self.emergence = emergence;
        emergence
    }

    /// Full step: update all dynamics with time delta
    pub fn step(
        &mut self,
        field: &[f64],
        curvature: &[f64],
        coupling: f64,
        psi_dot: f64,
        xi_dot: f64,
        coherence: f64,
        identity: f64,
        meta_rate: f64,
        neural_noise: f64,
        interference_level: f64,
        forgetting_level: f64,
    ) -> CognitiveState {
        let psi_i = self.compute_field_intensity(field, curvature, coupling);
        let entropy = self.compute_entropy(neural_noise, interference_level, forgetting_level);
        let emergence = self.compute_emergence(psi_dot, xi_dot, coherence, identity, meta_rate);

        CognitiveState {
            field_intensity: psi_i,
            entropy,
            emergence,
            coherence: self.coherence,
            identity: self.identity,
            meta_rate: self.meta_rate,
            field_energy: self.field_energy,
            conservation_error: (psi_dot * 0.6 - xi_dot * 0.4).abs(),
        }
    }
}

impl Default for CognitiveDynamics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot dari cognitive dynamics state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveState {
    pub field_intensity: f64,
    pub entropy: f64,
    pub emergence: f64,
    pub coherence: f64,
    pub identity: f64,
    pub meta_rate: f64,
    pub field_energy: f64,
    pub conservation_error: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_memory_manager() {
        let manager = MemoryManager::new();
        
        // Test store and retrieve
        manager.store(MemoryLayer::Short, "test_key", "test_value").await.unwrap();
        let result = manager.retrieve(MemoryLayer::Short, "test_key").await.unwrap();
        assert_eq!(result, Some("test_value".to_string()));
        
        // Test search
        let search_results = manager.search("test").await.unwrap();
        assert!(!search_results.is_empty());
        
        // Test compression
        let context = "This is a test context for compression";
        let compressed = manager.compress_context(context).await.unwrap();
        let decompressed = manager.decompress_context(&compressed).await.unwrap();
        assert_eq!(context, decompressed);
    }
    
    #[tokio::test]
    async fn test_memory_layer_operations() {
        let manager = MemoryManager::new();
        
        // Test storing and retrieving from different layers
        manager.store_short_term("temp_key", "temp_value").await.unwrap();
        manager.store_session("session_key", "session_value").await.unwrap();
        manager.store_long_term("persistent_key", "persistent_value").await.unwrap();
        manager.store_knowledge("knowledge_key", "knowledge_value").await.unwrap();
        
        // Verify data is stored (basic check)
        let short_term_data = manager.get_short_term_data().await.unwrap();
        assert!(!short_term_data.is_empty());
        
        let session_data = manager.get_session_data().await.unwrap();
        assert!(!session_data.is_empty());
        
        let long_term_data = manager.get_long_term_data().await.unwrap();
        assert!(!long_term_data.is_empty());
        
        let knowledge_data = manager.get_knowledge_data().await.unwrap();
        assert!(!knowledge_data.is_empty());
    }
    
    #[tokio::test]
    async fn test_memory_clear_operations() {
        let manager = MemoryManager::new();
        
        // Store some data first
        manager.store_short_term("temp_key", "temp_value").await.unwrap();
        manager.store_session("session_key", "session_value").await.unwrap();
        manager.store_long_term("persistent_key", "persistent_value").await.unwrap();
        manager.store_knowledge("knowledge_key", "knowledge_value").await.unwrap();
        
        // Test clearing operations
        manager.clear_short_term().await.unwrap();
        manager.clear_session().await.unwrap();
        manager.clear_long_term().await.unwrap();
        manager.clear_knowledge().await.unwrap();
        
        // Verify clearing worked (data should be empty)
        let short_term_data = manager.get_short_term_data().await.unwrap();
        assert!(short_term_data.is_empty());
        
        let session_data = manager.get_session_data().await.unwrap();
        assert!(session_data.is_empty());
        
        let long_term_data = manager.get_long_term_data().await.unwrap();
        assert!(long_term_data.is_empty());
        
        let knowledge_data = manager.get_knowledge_data().await.unwrap();
        assert!(knowledge_data.is_empty());
    }
    
    #[tokio::test]
    async fn test_memory_interaction_storage() {
        let manager = MemoryManager::new();
        
        // Test interaction storage
        manager.store_interaction("User: Hello", "Assistant: Hi there!").await.unwrap();
        
        // Verify interaction is stored
        let session_data = manager.get_session_data().await.unwrap();
        assert!(!session_data.is_empty());
    }
    
    #[tokio::test]
    async fn test_memory_search_functionality() {
        let manager = MemoryManager::new();
        
        // Store test data
        manager.store_short_term("test_key_1", "test_value_1").await.unwrap();
        manager.store_session("test_key_2", "test_value_2").await.unwrap();
        manager.store_long_term("test_key_3", "test_value_3").await.unwrap();
        
        // Test search functionality
        let search_results = manager.search("test").await.unwrap();
        assert!(!search_results.is_empty());
        
        // Test specific layer search - filter results for specific layer
        let all_results = manager.search("test").await.unwrap();
        let layer_results: Vec<_> = all_results.iter()
            .filter(|r| matches!(r.layer, MemoryLayer::Short))
            .cloned()
            .collect();
        assert!(!layer_results.is_empty());
    }
    
    #[test]
    fn test_memory_config_default() {
        let config = HebbianMemoryConfig::default();
        assert!(config.capacity > 0);
        assert!(config.decay_rate > 0.0);
        assert!(config.consolidation_gamma > 0.0);
        assert!(config.capacity > 0);
    }
    
    #[test]
    fn test_memory_layer_enum() {
        // Test all memory layers can be created and compared
        let layers = vec![
            MemoryLayer::Short,
            MemoryLayer::Session,
            MemoryLayer::Long,
            MemoryLayer::Knowledge,
        ];
        
        assert_eq!(layers.len(), 4);
        assert!(layers.contains(&MemoryLayer::Short));
        assert!(layers.contains(&MemoryLayer::Session));
        assert!(layers.contains(&MemoryLayer::Long));
        assert!(layers.contains(&MemoryLayer::Knowledge));
    }

    #[test]
    fn test_cognitive_dynamics_new() {
        let cd = CognitiveDynamics::new();
        assert_eq!(cd.field_intensity, 0.0);
        assert_eq!(cd.entropy, 0.0);
        assert_eq!(cd.w, 1.0);
        assert_eq!(cd.kappa, 1.0);
    }

    #[test]
    fn test_cognitive_dynamics_field_intensity() {
        let mut cd = CognitiveDynamics::new();
        let field = vec![1.0, 0.5, 0.0, -0.5, 1.0];
        let curvature = vec![0.1, 0.2, 0.0, -0.1, 0.3];
        let psi_i = cd.compute_field_intensity(&field, &curvature, 0.5);
        assert!(psi_i >= 0.0);
    }

    #[test]
    fn test_cognitive_dynamics_entropy() {
        let mut cd = CognitiveDynamics::new();
        let entropy = cd.compute_entropy(1.0, 0.5, 0.3);
        assert!((entropy - (0.3 * 1.0 + 0.2 * 0.5 + 0.1 * 0.3)).abs() < 1e-10);
        assert_eq!(cd.entropy, entropy);
    }

    #[test]
    fn test_cognitive_dynamics_emergence() {
        let mut cd = CognitiveDynamics::new();
        let emergence = cd.compute_emergence(5.0, 1.0, 0.9, 0.8, 0.5);
        assert!(emergence > 0.0);
        assert!(emergence <= 1.0);
    }

    #[test]
    fn test_cognitive_dynamics_full_step() {
        let mut cd = CognitiveDynamics::new();
        let field = vec![1.0, 0.5, 0.0, -0.5, 1.0];
        let curvature = vec![0.1, 0.0, -0.1, 0.0, 0.2];
        let state = cd.step(
            &field, &curvature, 0.5,
            5.0, 1.0, 0.9, 0.8, 0.5,
            1.0, 0.5, 0.3,
        );
        assert!(state.field_intensity >= 0.0);
        assert!(state.entropy > 0.0);
        assert!(state.emergence > 0.0);
    }
}
