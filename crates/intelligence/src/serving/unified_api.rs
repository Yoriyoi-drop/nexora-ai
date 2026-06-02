//! Unified API for Integrated Nexora Models
//!
//! Provides a single interface for using all integrated models:
//! - SACA (Systematic Adaptive Code Architecture)
//! - ATQS (Adaptive Tensor Quantization & Sparsification)
//! - CAFFEINE (Contrastive-Aware Fusion Framework)
//! - HAS-MoE-FFN (Hybrid Adaptive Structured MoE-FFN) — real top-k gating

// Import from foundation modules
use nexora_foundation::atqs::{compression::CompressionEngine, ATQSConfig};
use nexora_foundation::multimodal::caffeine::{
    types::{MultiModalInputs, TextInput},
    Caffeine, CaffeineConfig,
};
use nexora_foundation::reasoning::{CodingTask, SACAConfig, SACAIntegration, SACASolution};

use parking_lot::Mutex as ParkingMutex;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

pub type ApiResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

// ─── MoE Router & FFN ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
}

#[derive(Debug, Clone)]
pub struct HasMoeFfnConfig {
    pub router_config: RouterConfig,
}

impl HasMoeFfnConfig {
    pub fn medium_model() -> Self {
        Self {
            router_config: RouterConfig {
                hidden_size: 768,
                num_experts: 8,
                top_k: 2,
            },
        }
    }
}

/// Expert routing result: which experts are selected and their gating weights.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Indices of selected experts (length = top_k)
    pub expert_indices: Vec<usize>,
    /// Normalised softmax weights for selected experts (sum ≈ 1.0)
    pub expert_weights: Vec<f32>,
    /// Load statistics: how often each expert is selected (running counts)
    pub load_counters: Vec<usize>,
}

/// ExpertRouter: linear gating network W_g ∈ R^{hidden × num_experts}.
/// Uses top-k softmax gating with auxiliary load-balancing tracking.
pub struct ExpertRouter {
    config: RouterConfig,
    /// Gating weight matrix stored row-major: shape [hidden_size × num_experts]
    gate_weights: Vec<f32>,
    /// Running selection count per expert (for load monitoring)
    load_counters: std::sync::Arc<ParkingMutex<Vec<usize>>>,
}

impl ExpertRouter {
    /// Initialise gating weights with Xavier uniform initialisation.
    pub fn new(config: RouterConfig) -> ApiResult<Self> {
        let n = config.hidden_size * config.num_experts;
        let scale = (6.0_f32 / (config.hidden_size + config.num_experts) as f32).sqrt();
        let gate_weights: Vec<f32> = (0..n)
            .map(|_| (fastrand::f32() * 2.0 - 1.0) * scale)
            .collect();

        info!(
            "ExpertRouter initialised: hidden={}, num_experts={}, top_k={}",
            config.hidden_size, config.num_experts, config.top_k
        );

        Ok(Self {
            load_counters: std::sync::Arc::new(ParkingMutex::new(vec![0usize; config.num_experts])),
            gate_weights,
            config,
        })
    }

    /// Route a hidden-state vector `h` of length `hidden_size` through the
    /// gating network and return the top-k routing decision.
    pub fn route(&self, h: &[f32]) -> ApiResult<RoutingDecision> {
        if h.len() != self.config.hidden_size {
            return Err(format!(
                "ExpertRouter: expected hidden_size={}, got {}",
                self.config.hidden_size,
                h.len()
            )
            .into());
        }

        let num_experts = self.config.num_experts;
        let top_k = self.config.top_k.min(num_experts);

        // Compute gate logits: g = h · W_g  (shape: num_experts)
        let mut logits = vec![0.0f32; num_experts];
        for e in 0..num_experts {
            let col_start = e; // column-major access: W[row, col] at index row*num_experts + col
            let mut dot = 0.0f32;
            for row in 0..self.config.hidden_size {
                dot += h[row] * self.gate_weights[row * num_experts + col_start];
            }
            logits[e] = dot;
        }

        // Softmax over all logits (numerically stable)
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp.max(1e-9)).collect();

        // Select top-k by probability
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

        let expert_indices: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
        let raw_weights: Vec<f32> = top.iter().map(|(_, w)| *w).collect();

        // Re-normalise selected weights so they sum to 1.0
        let weight_sum: f32 = raw_weights.iter().sum();
        let expert_weights: Vec<f32> = raw_weights
            .iter()
            .map(|&w| w / weight_sum.max(1e-9))
            .collect();

        // Update load counters
        let mut counters = self.load_counters.lock();
        for &idx in &expert_indices {
            counters[idx] += 1;
        }
        let load_counters = counters.clone();
        drop(counters);

        debug!(
            "Routed to experts {:?} with weights {:?}",
            expert_indices, expert_weights
        );

        Ok(RoutingDecision {
            expert_indices,
            expert_weights,
            load_counters,
        })
    }

    /// Return a copy of the current load counter vector.
    pub fn load_stats(&self) -> Vec<usize> {
        self.load_counters.lock().clone()
    }
}

/// HasMoeFfn: Mixture-of-Experts FFN layer.
///
/// Each expert is a two-layer FFN: Linear(hidden→4×hidden) → ReLU → Linear(4×hidden→hidden).
/// The router selects top-k experts; their outputs are combined via weighted sum.
pub struct HasMoeFfn {
    router: ExpertRouter,
    /// Expert weights: each expert has W1 (hidden × 4*hidden) and W2 (4*hidden × hidden)
    /// Stored as flat Vec<f32>. Expert e: w1[e] shape [hidden × 4h], w2[e] shape [4h × hidden]
    expert_w1: Vec<Vec<f32>>, // [num_experts][hidden * 4*hidden]
    expert_w2: Vec<Vec<f32>>, // [num_experts][4*hidden * hidden]
    hidden_size: usize,
    ffn_size: usize,
}

impl HasMoeFfn {
    /// Initialise all expert FFN weights with Xavier uniform.
    pub fn new(config: HasMoeFfnConfig) -> ApiResult<Self> {
        let hidden = config.router_config.hidden_size;
        let ffn = hidden * 4;
        let num_experts = config.router_config.num_experts;

        let xavier_w1 = (6.0_f32 / (hidden + ffn) as f32).sqrt();
        let xavier_w2 = (6.0_f32 / (ffn + hidden) as f32).sqrt();

        let expert_w1: Vec<Vec<f32>> = (0..num_experts)
            .map(|_| {
                (0..hidden * ffn)
                    .map(|_| (fastrand::f32() * 2.0 - 1.0) * xavier_w1)
                    .collect()
            })
            .collect();

        let expert_w2: Vec<Vec<f32>> = (0..num_experts)
            .map(|_| {
                (0..ffn * hidden)
                    .map(|_| (fastrand::f32() * 2.0 - 1.0) * xavier_w2)
                    .collect()
            })
            .collect();

        let router = ExpertRouter::new(config.router_config)?;

        info!(
            "HasMoeFfn initialised: hidden={}, ffn={}, num_experts={}",
            hidden, ffn, num_experts
        );

        Ok(Self {
            router,
            expert_w1,
            expert_w2,
            hidden_size: hidden,
            ffn_size: ffn,
        })
    }

    /// Forward pass: route `h` through top-k experts and return the weighted
    /// combined output (same shape as `h`).
    ///
    /// Computation:
    /// 1. Route: get top-k expert indices + weights
    /// 2. For each selected expert e:
    ///    a. mid = W1_e · h  (shape: ffn_size), then ReLU
    ///    b. out_e = W2_e · mid  (shape: hidden_size)
    /// 3. output = Σ weight_e * out_e
    pub fn forward(&self, h: &[f32]) -> ApiResult<Vec<f32>> {
        if h.len() != self.hidden_size {
            return Err(format!(
                "HasMoeFfn forward: input length {} ≠ hidden_size {}",
                h.len(),
                self.hidden_size
            )
            .into());
        }

        let decision = self.router.route(h)?;
        let mut output = vec![0.0f32; self.hidden_size];

        for (&expert_idx, &weight) in decision
            .expert_indices
            .iter()
            .zip(decision.expert_weights.iter())
        {
            let w1 = &self.expert_w1[expert_idx];
            let w2 = &self.expert_w2[expert_idx];

            // Linear 1: mid = W1 · h, shape [ffn_size]
            let mut mid = vec![0.0f32; self.ffn_size];
            for j in 0..self.ffn_size {
                let mut acc = 0.0f32;
                for i in 0..self.hidden_size {
                    acc += w1[i * self.ffn_size + j] * h[i];
                }
                // ReLU activation
                mid[j] = acc.max(0.0);
            }

            // Linear 2: out = W2 · mid, shape [hidden_size]
            let mut expert_out = vec![0.0f32; self.hidden_size];
            for k in 0..self.hidden_size {
                let mut acc = 0.0f32;
                for j in 0..self.ffn_size {
                    acc += w2[j * self.hidden_size + k] * mid[j];
                }
                expert_out[k] = acc;
            }

            // Weighted accumulate
            for (o, &e) in output.iter_mut().zip(expert_out.iter()) {
                *o += weight * e;
            }
        }

        Ok(output)
    }

    /// Return load statistics for monitoring expert utilisation.
    pub fn load_stats(&self) -> Vec<usize> {
        self.router.load_stats()
    }

    /// Return routing efficiency: 1 - normalised max/min load ratio.
    /// 1.0 = perfectly balanced, 0.0 = one expert handles everything.
    pub fn routing_efficiency(&self) -> f32 {
        let stats = self.load_stats();
        let total: usize = stats.iter().sum();
        if total == 0 {
            return 1.0;
        }
        let max = *stats.iter().max().unwrap_or(&1) as f32;
        let ideal = total as f32 / stats.len() as f32;
        (ideal / max).min(1.0)
    }
}

// ─── UnifiedModel (unchanged except MoE wiring) ───────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationMode {
    SACAOnly,
    SACAWithATQS,
    SACAWithCaffeine,
    SACAWithHasMoe,
    FullIntegration,
}

#[derive(Debug, Clone)]
pub struct UnifiedConfig {
    pub saca_config: SACAConfig,
    pub atqs_config: Option<ATQSConfig>,
    pub caffeine_config: Option<CaffeineConfig>,
    pub has_moe_config: Option<HasMoeFfnConfig>,
    pub integration_mode: IntegrationMode,
}

pub struct UnifiedModel {
    config: UnifiedConfig,
    saca_integration: SACAIntegration,
    caffeine_model: Option<Arc<Mutex<Caffeine>>>,
    /// Real HasMoeFfn layer (Some when HasMoe or FullIntegration is active)
    moe_ffn: Option<Arc<Mutex<HasMoeFfn>>>,
}

impl UnifiedModel {
    pub async fn new(config: UnifiedConfig) -> ApiResult<Self> {
        info!(
            "Initialising Unified Model with integration mode: {:?}",
            config.integration_mode
        );

        let mut saca_integration = SACAIntegration::new(config.saca_config.clone()).await?;

        if let Some(atqs_config) = &config.atqs_config {
            match config.integration_mode {
                IntegrationMode::SACAWithATQS | IntegrationMode::FullIntegration => {
                    let compression_engine = Arc::new(
                        CompressionEngine::new(atqs_config.clone())
                            .map_err(|e| format!("Compression engine: {}", e))?,
                    );
                    saca_integration = saca_integration.with_atqs_compression(compression_engine);
                    info!("ATQS compression enabled");
                }
                _ => {}
            }
        }

        let caffeine_model = if let Some(caffeine_config) = &config.caffeine_config {
            match config.integration_mode {
                IntegrationMode::SACAWithCaffeine | IntegrationMode::FullIntegration => {
                    let caffeine = Arc::new(Mutex::new(Caffeine::new(caffeine_config.clone())?));
                    saca_integration = saca_integration.with_caffeine(caffeine.clone());
                    info!("CAFFEINE multimodal processing enabled");
                    Some(caffeine)
                }
                _ => None,
            }
        } else {
            None
        };

        // Wire real HasMoeFfn when requested
        let moe_ffn = if let Some(moe_config) = &config.has_moe_config {
            match config.integration_mode {
                IntegrationMode::SACAWithHasMoe | IntegrationMode::FullIntegration => {
                    let ffn = HasMoeFfn::new(moe_config.clone())?;
                    let ffn_arc = Arc::new(Mutex::new(ffn));

                    // Wire to SACAIntegration via foundation Router
                    let router = Arc::new(nexora_foundation::has_moe_ffn::routing::Router::new(
                        moe_config.router_config.hidden_size,
                        moe_config.router_config.num_experts,
                        moe_config.router_config.top_k,
                    ));
                    saca_integration = saca_integration.with_has_moe_routing(router);
                    info!(
                        "HAS-MoE-FFN enabled: {} experts, top-{}",
                        moe_config.router_config.num_experts, moe_config.router_config.top_k
                    );
                    Some(ffn_arc)
                }
                _ => None,
            }
        } else {
            None
        };

        Ok(Self {
            config,
            saca_integration,
            caffeine_model,
            moe_ffn,
        })
    }

    pub async fn generate_code(&self, task: &CodingTask) -> ApiResult<UnifiedSolution> {
        info!("Starting unified coding task solution");
        let start_time = std::time::Instant::now();

        let enhanced_solution = self
            .saca_integration
            .solve_with_models(task.clone())
            .await?;

        let mut solution = UnifiedSolution {
            base_solution: enhanced_solution.base_solution,
            atqs_compression_applied: enhanced_solution.atqs_compression_applied,
            caffeine_multimodal_enhanced: enhanced_solution.caffeine_multimodal_enhanced,
            has_moe_routing_applied: enhanced_solution.has_moe_routing_applied,
            compression_ratio: enhanced_solution.compression_ratio,
            routing_efficiency: enhanced_solution.routing_efficiency,
            multimodal_features: enhanced_solution.multimodal_features,
            execution_time: start_time.elapsed(),
            integration_mode: self.config.integration_mode.clone(),
            quality_score: 0.0,
        };

        match self.config.integration_mode {
            IntegrationMode::FullIntegration => {
                self.apply_full_integration_processing(&mut solution, task)
                    .await?;
            }
            IntegrationMode::SACAWithCaffeine => {
                let task_text = format!(
                    "Task: {}\nRequirements: {}",
                    task.description,
                    task.requirements.join(", ")
                );
                let multimodal_input = MultiModalInputs {
                    text: Some(TextInput {
                        text: task_text,
                        tokens: None,
                        language: "en".to_string(),
                    }),
                    image: None,
                    audio: None,
                    video: None,
                    context: None,
                };
                let output = self.process_multimodal(&multimodal_input).await?;
                solution.caffeine_multimodal_enhanced = true;
                if let Some(text_output) = &output.text {
                    solution.quality_score =
                        (solution.quality_score + text_output.confidence * 0.05).min(1.0);
                }
            }
            IntegrationMode::SACAWithHasMoe => {
                self.apply_moe_post_processing(&mut solution).await?;
            }
            IntegrationMode::SACAWithATQS => {
                if solution.atqs_compression_applied {
                    solution.quality_score = (solution.quality_score
                        + solution.compression_ratio as f32 * 0.05)
                        .min(1.0);
                }
            }
            IntegrationMode::SACAOnly => {}
        }

        info!(
            "Unified solution completed in {:?}",
            solution.execution_time
        );
        Ok(solution)
    }

    pub async fn process_multimodal(
        &self,
        inputs: &MultiModalInputs,
    ) -> ApiResult<nexora_foundation::multimodal::caffeine::types::MultiModalOutputs> {
        if let Some(caffeine) = &self.caffeine_model {
            let mut guard = caffeine.lock().await;
            guard
                .forward(inputs)
                .await
                .map_err(|e| format!("Caffeine forward failed: {}", e).into())
        } else {
            Err("CAFFEINE model not enabled in this configuration".into())
        }
    }

    async fn apply_full_integration_processing(
        &self,
        solution: &mut UnifiedSolution,
        _task: &CodingTask,
    ) -> ApiResult<()> {
        debug!("Applying full integration processing");

        if solution.atqs_compression_applied && solution.caffeine_multimodal_enhanced {
            solution.quality_score += 0.02;
        }
        if solution.has_moe_routing_applied {
            self.apply_moe_post_processing(solution).await?;
        }

        solution.quality_score = solution.quality_score.min(1.0);
        Ok(())
    }

    /// Apply the real MoE FFN to the solution's multimodal feature vector.
    async fn apply_moe_post_processing(&self, solution: &mut UnifiedSolution) -> ApiResult<()> {
        if let Some(ffn_lock) = &self.moe_ffn {
            let ffn_guard = ffn_lock.lock().await;
            let hidden = ffn_guard.hidden_size;

            // Use multimodal_features as the hidden state; pad or truncate to hidden_size
            let h: Vec<f32> = if solution.multimodal_features.len() >= hidden {
                solution.multimodal_features[..hidden].to_vec()
            } else {
                let mut padded = solution.multimodal_features.clone();
                padded.resize(hidden, 0.0);
                padded
            };

            let out = ffn_guard.forward(&h)?;
            // Store MoE output back as refined feature vector
            solution.multimodal_features = out;
            solution.routing_efficiency = ffn_guard.routing_efficiency();
            solution.has_moe_routing_applied = true;
            solution.quality_score =
                (solution.quality_score + solution.routing_efficiency * 0.03).min(1.0);

            debug!(
                "MoE post-processing done: routing_efficiency={:.3}",
                solution.routing_efficiency
            );
        }
        Ok(())
    }

    pub async fn get_statistics(&self) -> UnifiedStats {
        let integration_stats = self.saca_integration.get_integration_stats();
        UnifiedStats {
            integration_mode: self.config.integration_mode.clone(),
            models_enabled: integration_stats.total_models_enabled,
            atqs_enabled: integration_stats.atqs_enabled,
            caffeine_enabled: integration_stats.caffeine_enabled,
            has_moe_enabled: integration_stats.has_moe_enabled,
            moe_load_stats: match &self.moe_ffn {
                Some(f) => Some(f.lock().await.load_stats()),
                None => None,
            },
        }
    }
}

// ─── Result and stats types ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct UnifiedSolution {
    pub base_solution: SACASolution,
    pub atqs_compression_applied: bool,
    pub caffeine_multimodal_enhanced: bool,
    pub has_moe_routing_applied: bool,
    pub compression_ratio: f64,
    pub routing_efficiency: f32,
    pub multimodal_features: Vec<f32>,
    pub execution_time: std::time::Duration,
    pub integration_mode: IntegrationMode,
    pub quality_score: f32,
}

#[derive(Debug, Clone)]
pub struct UnifiedStats {
    pub integration_mode: IntegrationMode,
    pub models_enabled: u32,
    pub atqs_enabled: bool,
    pub caffeine_enabled: bool,
    pub has_moe_enabled: bool,
    /// Per-expert selection counts; None when MoE is disabled.
    pub moe_load_stats: Option<Vec<usize>>,
}

// ─── Factory ──────────────────────────────────────────────────────────────────

pub struct UnifiedModelFactory;

impl UnifiedModelFactory {
    pub async fn create_basic_coder() -> ApiResult<UnifiedModel> {
        UnifiedModel::new(UnifiedConfig {
            saca_config: SACAConfig::default(),
            atqs_config: None,
            caffeine_config: None,
            has_moe_config: None,
            integration_mode: IntegrationMode::SACAOnly,
        })
        .await
    }

    pub async fn create_compressed_coder() -> ApiResult<UnifiedModel> {
        UnifiedModel::new(UnifiedConfig {
            saca_config: SACAConfig::default(),
            atqs_config: Some(ATQSConfig::default()),
            caffeine_config: None,
            has_moe_config: None,
            integration_mode: IntegrationMode::SACAWithATQS,
        })
        .await
    }

    pub async fn create_multimodal_coder() -> ApiResult<UnifiedModel> {
        UnifiedModel::new(UnifiedConfig {
            saca_config: SACAConfig::default(),
            atqs_config: None,
            caffeine_config: Some(CaffeineConfig::medium_model()),
            has_moe_config: None,
            integration_mode: IntegrationMode::SACAWithCaffeine,
        })
        .await
    }

    pub async fn create_expert_coder() -> ApiResult<UnifiedModel> {
        UnifiedModel::new(UnifiedConfig {
            saca_config: SACAConfig::default(),
            atqs_config: None,
            caffeine_config: None,
            has_moe_config: Some(HasMoeFfnConfig::medium_model()),
            integration_mode: IntegrationMode::SACAWithHasMoe,
        })
        .await
    }

    pub async fn create_full_integration() -> ApiResult<UnifiedModel> {
        UnifiedModel::new(UnifiedConfig {
            saca_config: SACAConfig::default(),
            atqs_config: Some(ATQSConfig::default()),
            caffeine_config: Some(CaffeineConfig::medium_model()),
            has_moe_config: Some(HasMoeFfnConfig::medium_model()),
            integration_mode: IntegrationMode::FullIntegration,
        })
        .await
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_router_forward() {
        let cfg = RouterConfig {
            hidden_size: 16,
            num_experts: 4,
            top_k: 2,
        };
        let router = ExpertRouter::new(cfg).unwrap();
        let h = vec![0.1f32; 16];
        let decision = router.route(&h).unwrap();
        assert_eq!(decision.expert_indices.len(), 2);
        let weight_sum: f32 = decision.expert_weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-5, "weights should sum to 1.0");
    }

    #[test]
    fn test_expert_router_load_tracking() {
        let cfg = RouterConfig {
            hidden_size: 8,
            num_experts: 4,
            top_k: 1,
        };
        let router = ExpertRouter::new(cfg).unwrap();
        let h = vec![1.0f32; 8];
        router.route(&h).unwrap();
        router.route(&h).unwrap();
        let stats = router.load_stats();
        let total: usize = stats.iter().sum();
        assert_eq!(total, 2, "two routing calls should record 2 selections");
    }

    #[test]
    fn test_has_moe_ffn_forward_shape() {
        let cfg = HasMoeFfnConfig {
            router_config: RouterConfig {
                hidden_size: 16,
                num_experts: 4,
                top_k: 2,
            },
        };
        let ffn = HasMoeFfn::new(cfg).unwrap();
        let h = vec![0.5f32; 16];
        let out = ffn.forward(&h).unwrap();
        assert_eq!(out.len(), 16, "output must match hidden_size");
    }

    #[test]
    fn test_has_moe_ffn_routing_efficiency() {
        let cfg = HasMoeFfnConfig {
            router_config: RouterConfig {
                hidden_size: 8,
                num_experts: 4,
                top_k: 2,
            },
        };
        let ffn = HasMoeFfn::new(cfg).unwrap();
        // Before any calls: no load → should return 1.0
        assert_eq!(ffn.routing_efficiency(), 1.0);
        // After forward passes: efficiency should be a valid float
        let h = vec![0.1f32; 8];
        ffn.forward(&h).unwrap();
        ffn.forward(&h).unwrap();
        let eff = ffn.routing_efficiency();
        assert!(eff >= 0.0 && eff <= 1.0);
    }

    #[test]
    fn test_router_wrong_input_size_errors() {
        let cfg = RouterConfig {
            hidden_size: 16,
            num_experts: 4,
            top_k: 2,
        };
        let router = ExpertRouter::new(cfg).unwrap();
        let h = vec![0.0f32; 8]; // wrong size
        assert!(router.route(&h).is_err());
    }

    #[tokio::test]
    async fn test_unified_model_creation() {
        let model = UnifiedModelFactory::create_basic_coder().await;
        assert!(model.is_ok());
    }

    #[tokio::test]
    async fn test_integration_modes_stats() {
        let basic = UnifiedModelFactory::create_basic_coder().await.unwrap();
        let stats = basic.get_statistics().await;
        assert_eq!(stats.models_enabled, 0);
        assert!(stats.moe_load_stats.is_none());
    }
}
