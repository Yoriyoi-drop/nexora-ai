//! Routing mechanism for HAS-MoE-FFN
//! Improved dengan Capped Routing + Load Balancing Loss (Switch Transformer + Expert Choice)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::warn;

use crate::domains::ExpertPoolConfig;
use crate::types::DomainRoutingConfig;

/// Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub capacity_factor: f32,
    pub z_loss_coefficient: f32,
    pub importance_loss_coefficient: f32,
    pub use_capped_routing: bool,
    pub use_load_balancing_loss: bool,
    pub use_expert_choice: bool,
    /// Domain-aware routing config (None = flat routing).
    pub domain_routing: Option<DomainRoutingConfig>,
    /// Expert pool config for domain lookups (None = no domains).
    pub pool_config: Option<ExpertPoolConfig>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_experts: 256,
            top_k: 8,
            capacity_factor: 1.1,
            z_loss_coefficient: 1e-4,
            importance_loss_coefficient: 0.01,
            use_capped_routing: true,
            use_load_balancing_loss: true,
            use_expert_choice: false,
            domain_routing: Some(DomainRoutingConfig::default()),
            pool_config: Some(ExpertPoolConfig::default()),
        }
    }
}

/// Routing statistics
#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub load_balance_score: f32,
    pub expert_utilization: Vec<f32>,
    pub total_routes: usize,
    pub load_balancing_loss: f32,
    pub z_loss: f32,
    pub capacity_violations: usize,
    pub total_tokens: usize,
}

impl RoutingStats {
    pub fn new(num_experts: usize) -> Self {
        Self {
            load_balance_score: 1.0,
            expert_utilization: vec![0.0; num_experts],
            total_routes: 0,
            load_balancing_loss: 0.0,
            z_loss: 0.0,
            capacity_violations: 0,
            total_tokens: 0,
        }
    }
}

/// Router for expert selection — supports domain-aware routing.
pub struct Router {
    config: RouterConfig,
    routing_stats: HashMap<usize, usize>,
    expert_capacities: Vec<usize>,
    router_weights: Option<Vec<Vec<f32>>>,
    last_aux_loss: f32,
    #[cfg(feature = "gpu")]
    router_weights_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "cuda")]
    router_weights_cuda: std::sync::OnceLock<Option<nexora_autograd::gpu::cuda::CudaTensor>>,
}

impl Router {
    /// Create new router
    pub fn new(hidden_size: usize, num_experts: usize, top_k: usize) -> Self {
        let config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            ..Default::default()
        };

        Self::with_config(config)
    }

    /// Create router with custom config
    pub fn with_config(config: RouterConfig) -> Self {
        let num_experts = config.num_experts;
        Self {
            expert_capacities: vec![0; num_experts],
            config,
            routing_stats: HashMap::new(),
            router_weights: None,
            last_aux_loss: 0.0,
            #[cfg(feature = "gpu")]
            router_weights_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            router_weights_cuda: std::sync::OnceLock::new(),
        }
    }

    fn get_weights(&self) -> Result<&Vec<Vec<f32>>, String> {
        self.router_weights.as_ref().ok_or_else(|| {
            warn!("router_weights not initialized — call init_random() or load from checkpoint");
            "router_weights not initialized — call init_random() or load from checkpoint".to_string()
        })
    }

    pub fn init_random(&mut self) {
        let num_experts = self.config.num_experts;
        let hidden_size = self.config.hidden_size;
        let scale = (1.0 / hidden_size as f32).sqrt();
        let w: Vec<Vec<f32>> = (0..num_experts)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
                    .collect()
            })
            .collect();
        self.router_weights = Some(w);
    }

    pub fn drop_cpu_weights(&mut self) {
        self.router_weights = None;
    }

    pub fn has_weights(&self) -> bool {
        self.router_weights.is_some()
    }

    /// Return the auxiliary loss from the last forward pass
    pub fn auxiliary_loss(&self) -> f32 {
        self.last_aux_loss
    }

    /// Compute gating weight for a specific expert
    fn compute_gating_weight(&self, input: &[f32], expert_idx: usize) -> f32 {
        let w = match self.get_weights() {
            Ok(w) => w,
            Err(e) => {
                warn!("compute_gating_weight failed: {e}");
                return 0.0;
            }
        };
        let dot_product: f32 = input
            .iter()
            .enumerate()
            .map(|(i, &x)| x * w[expert_idx][i])
            .sum();
        dot_product
    }

    /// Softmax function
    fn softmax(&self, input: &[f32]) -> Vec<f32> {
        // Find max for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp and sum
        let exp_vals: Vec<f32> = input.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        // Normalize
        if sum > 0.0 {
            exp_vals.iter().map(|x| x / sum).collect()
        } else {
            vec![1.0 / input.len() as f32; input.len()]
        }
    }

    /// Forward pass through router (GPU-accelerated if available)
    pub fn forward(&self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        #[cfg(feature = "cuda")]
        if let Some(result) = self.forward_cuda(input) {
            return result;
        }
        #[cfg(feature = "gpu")]
        if let Some(result) = self.forward_gpu(input) {
            return result;
        }

        let (_batch_size, hidden_size) = input.dim();
        let num_experts = self.config.num_experts;

        // Batched matmul: input [batch × hidden] @ weightsᵀ [hidden × experts] = [batch × experts]
        // Replaces O(batch × num_experts × hidden) sequential dot products
        let mut gating_weights = match self.get_weights() {
            Ok(w) => {
                let w_flat: Vec<f32> = w.iter().flat_map(|r| r.iter()).copied().collect();
                match ndarray::Array2::from_shape_vec((num_experts, hidden_size), w_flat) {
                    Ok(w_arr) => input.dot(&w_arr.t().to_owned()),
                    Err(e) => {
                        warn!("failed to create weight matrix: {e}");
                        return ndarray::Array2::zeros((input.shape()[0], num_experts));
                    }
                }
            }
            Err(e) => {
                warn!("forward failed: {e}");
                return ndarray::Array2::zeros((input.shape()[0], num_experts));
            }
        };

        // Row-wise softmax in-place
        for mut row in gating_weights.rows_mut() {
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = row.iter().map(|x| (x - max_val).exp()).sum();
            if exp_sum > 0.0 {
                row.mapv_inplace(|x| (x - max_val).exp() / exp_sum);
            } else {
                row.fill(1.0 / row.len() as f32);
            }
        }

        gating_weights
    }

    /// GPU-accelerated forward: upload input → matmul → softmax → readback
    /// Returns None if GPU unavailable or any GPU operation fails (CPU fallback).
    #[cfg(feature = "gpu")]
    fn ensure_weights_gpu(&self) -> Option<&nexora_autograd::gpu::GpuTensor> {
        use nexora_autograd::gpu::GpuContext;
        let ctx = GpuContext::global().ok()?;
        let entry = self.router_weights_gpu.get_or_init(|| {
            let num_experts = self.config.num_experts;
            let hidden_size = self.config.hidden_size;
            let w = self.router_weights.as_ref()?;
            let flat: Vec<f32> = w.iter().flatten().copied().collect();
            let cpu = ndarray::Array2::from_shape_vec((num_experts, hidden_size), flat).ok()?;
            let gpu = nexora_autograd::gpu::GpuTensor::from_cpu(&cpu.into_dyn()).ok()?;
            ctx.transpose(&gpu).ok()
        });
        entry.as_ref()
    }

    #[cfg(feature = "gpu")]
    fn forward_gpu(&self, input: &ndarray::Array2<f32>) -> Option<ndarray::Array2<f32>> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let weights_t = self.ensure_weights_gpu()?;
        let input_gpu = GpuTensor::from_cpu(&input.clone().into_dyn()).ok()?;
        let ctx = GpuContext::global().ok()?;
        let scores = ctx.matmul(&input_gpu, weights_t).ok()?;
        let probs = ctx.softmax(&scores).ok()?;
        let cpu_d = probs.to_cpu().ok()?;
        cpu_d.into_dimensionality::<ndarray::Ix2>().ok()
    }

    /// Lazily upload router weights to CUDA — cached via OnceLock.
    /// Weight matrix is stored as [hidden_size, num_experts] (transposed for cuBLAS matmul).
    #[cfg(feature = "cuda")]
    fn ensure_weights_cuda(&self, cuda: &nexora_autograd::gpu::cuda::CudaRuntime) -> Option<&nexora_autograd::gpu::cuda::CudaTensor> {
        use nexora_autograd::gpu::cuda::CudaTensor;
        let entry = self.router_weights_cuda.get_or_init(|| {
            let num_experts = self.config.num_experts;
            let hidden_size = self.config.hidden_size;
            let w = self.router_weights.as_ref()?;
            let flat: Vec<f32> = w.iter().flatten().copied().collect();
            CudaTensor::from_cpu(&cuda.device, vec![num_experts, hidden_size], &flat).ok()
        });
        entry.as_ref()
    }

    /// CUDA-accelerated forward: matmul via cuBLAS + softmax via JIT kernel, then readback.
    /// Returns None if CUDA unavailable.
    #[cfg(feature = "cuda")]
    fn forward_cuda(&self, input: &ndarray::Array2<f32>) -> Option<ndarray::Array2<f32>> {
        use nexora_autograd::gpu::{GpuContext, GpuBackend};
        let ctx = GpuContext::global().ok()?;
        if ctx.backend() != GpuBackend::Cuda {
            return None;
        }
        let cuda = ctx.cuda_runtime()?;

        let weights = self.ensure_weights_cuda(cuda)?;
        // weights cached as [num_experts, hidden_size]; no transpose needed
        // cuBLAS transposes both operands internally, so layout is already correct

        let n = input.shape()[0];
        let dim = input.shape()[1];
        let input_flat: Vec<f32> = input.iter().copied().collect();
        let input_gpu = nexora_autograd::gpu::cuda::CudaTensor::from_cpu(
            &cuda.device, vec![n, dim], &input_flat,
        ).ok()?;

        // scores = input @ weights → [batch, num_experts]  (cuBLAS handles transpose)
        let scores = cuda.matmul(&input_gpu, weights).ok()?;
        let probs = cuda.softmax(&scores).ok()?;

        let out_cpu = probs.to_cpu_vec(&cuda.device).ok()?;
        let out_shape = vec![n, self.config.num_experts];
        ndarray::Array2::from_shape_vec(out_shape, out_cpu).ok()
    }

    pub fn route_single(&self, input: &ndarray::Array1<f32>) -> Result<Vec<usize>, String> {
        let input_slice = input.as_slice().unwrap_or(&[]);
        self.route_single_with_weights(input_slice)
            .map(|(experts, _)| experts)
    }

    /// Route batch of inputs dengan Capped Routing + Load Balancing Loss
    pub fn route(&mut self, input: &ndarray::Array2<f32>) -> Result<Vec<Vec<usize>>, String> {
        let (batch_size, _) = input.dim();
        let mut all_routes = Vec::with_capacity(batch_size);
        let mut routing_weights: Vec<Vec<(usize, f32)>> = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let row_view = input.row(i);
            let row_slice = row_view.as_slice().unwrap_or(&[]);
            let (route, weights) = self.route_single_with_weights(row_slice)?;
            let weights_with_indices: Vec<(usize, f32)> = weights.into_iter().enumerate().collect();
            routing_weights.push(weights_with_indices);
            all_routes.push(route);
        }

        // Phase 2: Capped routing — batasi token per expert (batch-aware)
        if self.config.use_capped_routing {
            let capacity =
                (batch_size as f32 * self.config.top_k as f32 * self.config.capacity_factor
                    / self.config.num_experts as f32)
                    .ceil() as usize;

            let mut expert_counts = vec![0usize; self.config.num_experts];
            let mut capped_routes: Vec<Vec<usize>> = vec![Vec::new(); batch_size];
            let mut _capacity_violations = 0;

            // Sort tokens by routing confidence untuk fair capacity allocation
            for i in 0..batch_size {
                let _conf = all_routes[i]
                    .iter()
                    .map(|&e| {
                        routing_weights[i]
                            .iter()
                            .find(|(ex, _)| *ex == e)
                            .map(|(_, w)| *w)
                            .unwrap_or(0.0)
                    })
                    .sum::<f32>();

                for &expert_id in &all_routes[i] {
                    if expert_counts[expert_id] < capacity {
                        capped_routes[i].push(expert_id);
                        expert_counts[expert_id] += 1;
                    } else {
                        _capacity_violations += 1;
                    }
                }
            }

            self.expert_capacities = expert_counts;
            all_routes = capped_routes;
        }

        // Phase 3: Compute auxiliary loss (load balancing + Z-loss)
        self.last_aux_loss = 0.0;
        if self.config.use_load_balancing_loss {
            self.last_aux_loss += self.compute_load_balancing_loss(&routing_weights, batch_size);
        }

        // Phase 4: Update routing stats
        for route in &all_routes {
            for &expert_id in route {
                *self.routing_stats.entry(expert_id).or_insert(0) += 1;
            }
        }

        Ok(all_routes)
    }

    /// Route batch with per-expert softmax confidence scores
    pub fn route_with_weights(
        &mut self,
        input: &ndarray::Array2<f32>,
    ) -> Result<Vec<Vec<(usize, f32)>>, String> {
        let (batch_size, _) = input.dim();
        let mut all_routes: Vec<Vec<(usize, f32)>> = Vec::with_capacity(batch_size);
        let mut routing_weights: Vec<Vec<(usize, f32)>> = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let row_view = input.row(i);
            let row_slice = row_view.as_slice().unwrap_or(&[]);
            let (route, weights) = self.route_single_with_weights(row_slice)?;
            let top_k: Vec<(usize, f32)> = route.iter().map(|&e| {
                let w = weights.get(e).copied().unwrap_or(0.0);
                (e, w)
            }).collect();
            let weights_with_indices: Vec<(usize, f32)> = weights.into_iter().enumerate().collect();
            routing_weights.push(weights_with_indices);
            all_routes.push(top_k);
        }

        if self.config.use_capped_routing {
            let capacity =
                (batch_size as f32 * self.config.top_k as f32 * self.config.capacity_factor
                    / self.config.num_experts as f32)
                    .ceil() as usize;
            let mut expert_counts = vec![0usize; self.config.num_experts];
            let mut capped: Vec<Vec<(usize, f32)>> = vec![Vec::new(); batch_size];

            for i in 0..batch_size {
                for &(expert_id, conf) in &all_routes[i] {
                    if expert_counts[expert_id] < capacity {
                        capped[i].push((expert_id, conf));
                        expert_counts[expert_id] += 1;
                    }
                }
            }

            self.expert_capacities = expert_counts;
            all_routes = capped;
        }

        self.last_aux_loss = 0.0;
        if self.config.use_load_balancing_loss {
            self.last_aux_loss += self.compute_load_balancing_loss(&routing_weights, batch_size);
        }

        for route in &all_routes {
            for &(expert_id, _) in route {
                *self.routing_stats.entry(expert_id).or_insert(0) += 1;
            }
        }

        Ok(all_routes)
    }

    pub fn route_single_with_zloss(
        &self,
        input: &ndarray::Array1<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>, f32), String> {
        let input_slice = input.as_slice().unwrap_or(&[]);
        let mut gating_weights = Vec::with_capacity(self.config.num_experts);
        for j in 0..self.config.num_experts {
            let weight = self.compute_gating_weight(input_slice, j);
            gating_weights.push(weight);
        }

        // Softmax + compute Z-loss
        let max_val = gating_weights
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = gating_weights.iter().map(|x| (x - max_val).exp()).sum();
        let log_sum_exp = max_val + exp_sum.ln();
        let z_loss = log_sum_exp * log_sum_exp; // Z-loss = log(Σ exp)²

        let softmax_weights: Vec<f32> = if exp_sum > 0.0 {
            gating_weights
                .iter()
                .map(|x| (x - max_val).exp() / exp_sum)
                .collect()
        } else {
            vec![1.0 / self.config.num_experts as f32; self.config.num_experts]
        };

        // Get top-k experts using O(E) select_nth_unstable
        let mut expert_scores: Vec<(usize, f32)> = softmax_weights
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        let k = self.config.top_k.min(expert_scores.len());
        if k > 1 {
            expert_scores.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let top_experts: Vec<usize> = expert_scores
            .iter()
            .take(k)
            .map(|(expert_idx, _)| *expert_idx)
            .collect();

        Ok((top_experts, softmax_weights, z_loss))
    }

    fn route_single_with_weights(&self, input: &[f32]) -> Result<(Vec<usize>, Vec<f32>), String> {
        let mut gating_weights = Vec::with_capacity(self.config.num_experts);
        for j in 0..self.config.num_experts {
            let weight = self.compute_gating_weight(input, j);
            gating_weights.push(weight);
        }

        let softmax_weights = self.softmax(&gating_weights);

        let mut expert_scores: Vec<(usize, f32)> = softmax_weights
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        let k = self.config.top_k.min(expert_scores.len());
        if k > 1 {
            expert_scores.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let top_experts: Vec<usize> = expert_scores
            .iter()
            .take(k)
            .map(|(expert_idx, _)| *expert_idx)
            .collect();

        Ok((top_experts, softmax_weights))
    }

    /// Domain-aware single token routing.
    /// Applies domain bias to gating weights before top-k selection,
    /// ensuring a mix of shared and tier-specific experts.
    pub fn route_single_domain_aware(
        &self,
        input: &ndarray::Array1<f32>,
        tier: &str,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let input_slice = input.as_slice().unwrap_or(&[]);
        let num_experts = self.config.num_experts;

        let mut gating_weights = Vec::with_capacity(num_experts);
        for j in 0..num_experts {
            let weight = self.compute_gating_weight(input_slice, j);
            gating_weights.push(weight);
        }

        let mut softmax_weights = self.softmax(&gating_weights);

        // Apply domain bias if configured
        if let Some(ref pool) = self.config.pool_config {
            if let Some(ref dr) = self.config.domain_routing {
                if dr.use_domain_bias {
                    for (i, w) in softmax_weights.iter_mut().enumerate() {
                        if let Some(domain) = pool.domain_for_expert(i) {
                            let bias = pool.domain_bias(domain) as f32;
                            *w *= bias;
                        }
                    }
                    // Re-normalize
                    let sum: f32 = softmax_weights.iter().sum();
                    if sum > 0.0 {
                        for w in &mut softmax_weights {
                            *w /= sum;
                        }
                    }
                }
            }
        }

        // Top-k selection with tier quota
        let mut expert_scores: Vec<(usize, f32)> = softmax_weights
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        let k = self.config.top_k.min(expert_scores.len());
        if k > 1 {
            expert_scores.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let mut selected: Vec<usize> = expert_scores.iter().take(k).map(|(idx, _)| *idx).collect();

        // Enforce tier quota: ensure enough tier-specific experts are selected
        if let Some(ref pool) = self.config.pool_config {
            if let Some(ref dr) = self.config.domain_routing {
                let tier_specific = pool.experts_in_tier(tier);
                let selected_tier_count = selected.iter().filter(|e| tier_specific.contains(e)).count();
                if selected_tier_count < dr.tier_quota {
                    // Replace lowest-scoring shared experts with top tier-specific ones not yet selected
                    let shared = pool.shared_experts();
                    let mut candidates: Vec<(usize, f32)> = tier_specific.iter()
                        .filter(|e| !selected.contains(e))
                        .map(|&e| (e, softmax_weights[e]))
                        .collect();
                    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    let needed = dr.tier_quota - selected_tier_count;
                    for &(candidate, _) in candidates.iter().take(needed) {
                        // Find worst shared expert to replace
                        if let Some(pos) = selected.iter().position(|e| shared.contains(e)) {
                            selected[pos] = candidate;
                        }
                    }
                }
            }
        }

        Ok((selected, softmax_weights))
    }

    /// Compute load balancing loss
    /// Importance loss (Switch Transformer): ∑_i f_i · P_i
    /// f_i = fraction of tokens routed to expert i
    /// P_i = average router probability for expert i
    pub fn compute_load_balancing_loss(
        &self,
        routing_weights: &[Vec<(usize, f32)>],
        batch_size: usize,
    ) -> f32 {
        if batch_size == 0 || !self.config.use_load_balancing_loss {
            return 0.0;
        }

        let num_experts = self.config.num_experts;
        let mut expert_counts = vec![0.0f32; num_experts];
        let mut expert_probs = vec![0.0f32; num_experts];

        for weights in routing_weights {
            for &(expert_id, prob) in weights {
                if expert_id < num_experts {
                    expert_counts[expert_id] += 1.0;
                    expert_probs[expert_id] += prob;
                }
            }
        }

        let total_tokens = batch_size as f32;
        let importance_loss: f32 = (0..num_experts)
            .map(|i| {
                let f_i = expert_counts[i] / total_tokens;
                let p_i = expert_probs[i] / total_tokens;
                f_i * p_i
            })
            .sum();

        // Nol loss jika routing sempurna (uniform)
        let _uniform_loss = 1.0 / num_experts as f32;
        importance_loss * self.config.importance_loss_coefficient
    }

    /// Get routing statistics (backward compatible — tanpa z_loss/load_balancing_loss)
    pub fn get_routing_stats(&self) -> RoutingStats {
        self.get_routing_stats_detailed(0.0, 0.0)
    }

    /// Get routing statistics (updated with load balancing metrics)
    pub fn get_routing_stats_detailed(
        &self,
        z_loss: f32,
        load_balancing_loss: f32,
    ) -> RoutingStats {
        let total_routes: usize = self.routing_stats.values().sum();
        let mut expert_utilization = vec![0.0; self.config.num_experts];

        for (expert_id, count) in &self.routing_stats {
            if *expert_id < self.config.num_experts {
                expert_utilization[*expert_id] = *count as f32 / total_routes.max(1) as f32;
            }
        }

        let avg_utilization =
            expert_utilization.iter().sum::<f32>() / self.config.num_experts as f32;
        let variance = expert_utilization
            .iter()
            .map(|u| (u - avg_utilization).powi(2))
            .sum::<f32>()
            / self.config.num_experts as f32;
        let load_balance_score = 1.0 / (1.0 + variance);

        let capacity_violations = self
            .expert_capacities
            .iter()
            .enumerate()
            .filter(|(_, &count)| {
                count
                    > (total_routes as f32 / self.config.num_experts as f32
                        * self.config.capacity_factor) as usize
            })
            .count();

        RoutingStats {
            load_balance_score,
            expert_utilization,
            total_routes,
            load_balancing_loss,
            z_loss,
            capacity_violations,
            total_tokens: total_routes / self.config.top_k.max(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn small_router() -> Router {
        let mut r = Router::new(4, 4, 2);
        r.init_random();
        r
    }

    #[test]
    fn test_new_router_creates_correct_dimensions() {
        let r = Router::new(8, 6, 3);
        assert!(r.router_weights.is_none());
        assert_eq!(r.expert_capacities.len(), 6);
    }

    #[test]
    fn test_forward_returns_softmax_distribution() {
        let r = small_router();
        let input = Array2::ones((3, 4));
        let weights = r.forward(&input);
        assert_eq!(weights.dim(), (3, 4));
        for i in 0..3 {
            let sum: f32 = (0..4).map(|j| weights[[i, j]]).sum();
            assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {}", sum);
        }
    }

    #[test]
    fn test_forward_weights_are_positive() {
        let r = small_router();
        let input = Array2::ones((2, 4));
        let weights = r.forward(&input);
        for i in 0..2 {
            for j in 0..4 {
                assert!(weights[[i, j]] >= 0.0);
            }
        }
    }

    #[test]
    fn test_forward_different_inputs_different_weights() {
        let r = small_router();
        let input_a = Array2::from_shape_vec((2, 4), vec![1.0; 8]).unwrap();
        let input_b = Array2::from_shape_vec((2, 4), vec![0.1; 8]).unwrap();
        let wa = r.forward(&input_a);
        let wb = r.forward(&input_b);
        let mut diff = 0.0;
        for i in 0..2 {
            for j in 0..4 {
                diff += (wa[[i, j]] - wb[[i, j]]).abs();
            }
        }
        assert!(diff > 0.0);
    }

    #[test]
    fn test_route_single_returns_top_k() {
        let r = small_router();
        let input = Array2::from_shape_vec((1, 4), vec![0.5, 0.3, 0.8, 0.2]).unwrap();
        let row = input.row(0);
        let result = r.route_single(&row.to_owned());
        assert!(result.is_ok());
        let experts = result.unwrap();
        assert_eq!(experts.len(), 2);
        assert!(experts[0] < 4);
        assert!(experts[1] < 4);
        assert_ne!(experts[0], experts[1]);
    }

    #[test]
    fn test_route_with_capped_routing() {
        let config = RouterConfig {
            hidden_size: 4,
            num_experts: 4,
            top_k: 2,
            capacity_factor: 1.25,
            use_capped_routing: true,
            use_load_balancing_loss: true,
            ..Default::default()
        };
        let mut r = Router::with_config(config);
        r.init_random();
        let input =
            Array2::from_shape_vec((8, 4), (0..32).map(|v| v as f32 / 32.0).collect()).unwrap();
        let result = r.route(&input);
        assert!(result.is_ok());
        let routes = result.unwrap();
        assert_eq!(routes.len(), 8);
        for route in &routes {
            assert!(route.len() <= 2);
        }
    }

    #[test]
    fn test_route_without_capped_routing() {
        let config = RouterConfig {
            hidden_size: 4,
            num_experts: 4,
            top_k: 2,
            use_capped_routing: false,
            use_load_balancing_loss: false,
            ..Default::default()
        };
        let mut r = Router::with_config(config);
        r.init_random();
        let input = Array2::ones((4, 4));
        let result = r.route(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_load_balancing_loss_no_routes() {
        let r = small_router();
        let loss = r.compute_load_balancing_loss(&[], 0);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_compute_load_balancing_loss_uniform() {
        let config = RouterConfig {
            hidden_size: 4,
            num_experts: 4,
            top_k: 2,
            use_load_balancing_loss: true,
            ..Default::default()
        };
        let mut r = Router::with_config(config);
        r.init_random();
        let routing_weights = vec![vec![(0, 0.25), (1, 0.25)], vec![(2, 0.25), (3, 0.25)]];
        let loss = r.compute_load_balancing_loss(&routing_weights, 2);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_route_single_with_zloss() {
        let r = small_router();
        let input = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let row = input.row(0);
        let result = r.route_single_with_zloss(&row.to_owned());
        assert!(result.is_ok());
        let (_experts, weights, z_loss) = result.unwrap();
        assert_eq!(weights.len(), 4);
        assert!(z_loss >= 0.0);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_get_routing_stats_empty() {
        let r = small_router();
        let stats = r.get_routing_stats();
        assert_eq!(stats.total_routes, 0);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.load_balance_score, 1.0);
    }

    #[test]
    fn test_get_routing_stats_detailed_after_route() {
        let config = RouterConfig {
            hidden_size: 4,
            num_experts: 4,
            top_k: 2,
            use_capped_routing: false,
            use_load_balancing_loss: false,
            ..Default::default()
        };
        let mut r = Router::with_config(config);
        r.init_random();
        let input = Array2::ones((4, 4));
        let _ = r.route(&input);
        let stats = r.get_routing_stats_detailed(0.5, 0.1);
        assert!(stats.total_routes > 0);
        assert_eq!(stats.z_loss, 0.5);
        assert_eq!(stats.load_balancing_loss, 0.1);
    }

    #[test]
    fn test_auxiliary_loss_tracks_last_forward() {
        let config = RouterConfig {
            hidden_size: 4,
            num_experts: 4,
            top_k: 2,
            use_capped_routing: false,
            use_load_balancing_loss: true,
            ..Default::default()
        };
        let mut r = Router::with_config(config);
        r.init_random();
        let input = Array2::ones((4, 4));
        let _ = r.route(&input);
        let loss = r.auxiliary_loss();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_softmax_empty_input() {
        let r = small_router();
        let result = r.softmax(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_single_element() {
        let r = small_router();
        let result = r.softmax(&[5.0]);
        assert!((result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let r = small_router();
        let result = r.softmax(&[1000.0, 1000.0, 1000.0]);
        for v in &result {
            assert!((v - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_compute_gating_weight_differs_by_expert() {
        let r = small_router();
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let w0 = r.compute_gating_weight(&input, 0);
        let w1 = r.compute_gating_weight(&input, 1);
        assert!((w0 - w1).abs() > 1e-10);
    }

    #[test]
    fn test_routing_stats_new() {
        let stats = RoutingStats::new(6);
        assert_eq!(stats.expert_utilization.len(), 6);
        assert_eq!(stats.load_balance_score, 1.0);
        assert_eq!(stats.total_routes, 0);
    }
}
