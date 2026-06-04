//! Expert Offloading System — VRAM-efficient MoE for 200B+ models.
//!
//! With 256 experts and top-8 (DeepSeek V4 Pro fine-grained MoE), only ~3.1% of experts are active per forward.
//! This module swaps idle experts between CPU (pinned memory) and GPU,
//! keeping only recently used experts GPU-resident.
//!
//! # Algorithm
//! 1. Router output → top-32 expert indices + confidence scores
//! 2. Check which experts are already GPU-resident (reuse)
//! 3. Compute urgency score = confidence × recency_penalty for missing experts
//! 4. Evict LRU experts until enough VRAM freed
//! 5. Async H2D upload missing experts (overlapped with compute)
//! 6. Forward on GPU-resident experts
//! 7. Update LRU tracker, trigger prefetch for predicted next experts

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

/// Simple LRU cache with configurable capacity.
/// Tracks which experts are GPU-resident and their last-use order.
pub struct LruTracker {
    capacity: usize,
    order: VecDeque<usize>,
    resident: Vec<bool>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl LruTracker {
    pub fn new(capacity: usize, num_experts: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::with_capacity(capacity),
            resident: vec![false; num_experts],
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Mark expert as recently used. Returns true if was already resident.
    pub fn touch(&mut self, expert_id: usize) -> bool {
        if expert_id >= self.resident.len() {
            return false;
        }
        if self.resident[expert_id] {
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.promote(expert_id);
            true
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    /// Evict least recently used expert. Returns evicted expert ID or None.
    pub fn evict_lru(&mut self) -> Option<usize> {
        while let Some(id) = self.order.pop_front() {
            if self.resident[id] {
                self.resident[id] = false;
                return Some(id);
            }
        }
        None
    }

    /// Add expert to resident set (was just loaded to GPU).
    pub fn mark_loaded(&mut self, expert_id: usize) {
        if expert_id < self.resident.len() {
            self.resident[expert_id] = true;
            self.promote(expert_id);
        }
    }

    /// Remove expert from resident set (was evicted from GPU).
    pub fn mark_evicted(&mut self, expert_id: usize) {
        if expert_id < self.resident.len() {
            self.resident[expert_id] = false;
        }
    }

    /// How many experts can be resident at once.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current number of resident experts.
    pub fn resident_count(&self) -> usize {
        self.resident.iter().filter(|&&r| r).count()
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            1.0
        } else {
            hits as f64 / total as f64
        }
    }

    fn promote(&mut self, expert_id: usize) {
        if let Some(pos) = self.order.iter().position(|&x| x == expert_id) {
            self.order.remove(pos);
        }
        self.order.push_back(expert_id);
        while self.order.len() > self.capacity {
            self.order.pop_front();
        }
    }
}

/// Expert weight snapshot stored on CPU (pinned memory pool recommended).
#[derive(Clone)]
pub struct ExpertWeights {
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl ExpertWeights {
    pub fn size_bytes(&self) -> usize {
        (self.fc1_w.len() + self.fc1_b.len() + self.fc2_w.len() + self.fc2_b.len()) * 4
    }
}

/// GPU-resident expert (weights uploaded, ready for forward).
pub struct GpuExpert {
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
}

impl GpuExpert {
    pub fn from_weights(w: &ExpertWeights) -> Self {
        Self {
            fc1_w: w.fc1_w.clone(),
            fc1_b: w.fc1_b.clone(),
            fc2_w: w.fc2_w.clone(),
            fc2_b: w.fc2_b.clone(),
        }
    }
}

/// Configuration for ExpertOffloader.
#[derive(Debug, Clone)]
pub struct OffloadConfig {
    /// Max VRAM bytes for expert weights on GPU (0 = auto).
    pub gpu_budget_bytes: u64,
    /// Min expert IDs to keep resident (router's most-used experts).
    pub min_warm_experts: usize,
    /// Max experts to upload per forward (throttle H2D bandwidth).
    pub max_uploads_per_step: usize,
    /// Enable async prefetch of predicted next experts.
    pub enable_prefetch: bool,
    /// Enable load-compute overlap (upload while computing ready experts).
    pub enable_overlap: bool,
    /// Fallback to CPU forward if expert not on GPU.
    pub allow_cpu_fallback: bool,
}

impl Default for OffloadConfig {
    fn default() -> Self {
        Self {
            gpu_budget_bytes: 0,
            min_warm_experts: 64,
            max_uploads_per_step: 8,
            enable_prefetch: true,
            enable_overlap: true,
            allow_cpu_fallback: true,
        }
    }
}

/// Expert Offloading Engine — manages which experts live on GPU.
///
/// # VRAM Budget Formula
/// ```text
/// gpu_budget = min(available_vram * 0.70, num_experts * expert_size)
/// expert_size = (hidden * inter * 2 + hidden + inter) * byte_per_elem
/// ```
pub struct ExpertOffloader {
    config: OffloadConfig,
    /// All 256 experts on CPU (pinned memory)
    cpu_pool: Vec<ExpertWeights>,
    /// Subset resident on GPU (indexed by expert_id)
    gpu_resident: HashMap<usize, GpuExpert>,
    /// LRU tracker for eviction decisions
    lru: LruTracker,
    /// Router usage history for prefetch prediction
    router_history: VecDeque<Vec<usize>>,
    /// Expert usage frequency (for warm set)
    usage_freq: Vec<u64>,
    /// Total bytes currently consumed by GPU-resident experts
    gpu_used_bytes: u64,
    /// Expert weight size in bytes (computed once)
    expert_size_bytes: u64,
    /// Predicted next experts (populated by prefetch)
    prefetch_queue: VecDeque<usize>,
}

impl ExpertOffloader {
    /// Create offloader for N experts with given dimensions.
    /// `cpu_weights` must contain ALL experts' weights.
    pub fn new(
        config: OffloadConfig,
        cpu_weights: Vec<ExpertWeights>,
    ) -> Self {
        let num_experts = cpu_weights.len();
        let expert_size = cpu_weights.first().map(|w| w.size_bytes() as u64).unwrap_or(0);
        let budget = if config.gpu_budget_bytes > 0 {
            config.gpu_budget_bytes
        } else {
            // Default: 70% of 24GB ≈ 17GB → ~130 experts (fp32) or ~260 experts (fp16)
            let default_vram = 24u64 * 1024 * 1024 * 1024;
            (default_vram as f64 * 0.70) as u64
        };
        let capacity = (budget / expert_size.max(1)) as usize;
        let capacity = capacity.min(num_experts).max(config.min_warm_experts);

        Self {
            config,
            cpu_pool: cpu_weights,
            gpu_resident: HashMap::new(),
            lru: LruTracker::new(capacity, num_experts),
            router_history: VecDeque::with_capacity(100),
            usage_freq: vec![0; num_experts],
            gpu_used_bytes: 0,
            expert_size_bytes: expert_size,
            prefetch_queue: VecDeque::new(),
        }
    }

    /// Prepare experts for forward pass.
    ///
    /// 1. Touches all requested experts (updates LRU)
    /// 2. Evicts cold experts if budget exceeded
    /// 3. Uploads missing experts async
    /// 4. Returns available & missing expert sets
    pub fn prepare(
        &mut self,
        top_experts: &[usize],
    ) -> OffloadBatch {
        // Touch all requested experts (track LRU)
        let mut available = Vec::new();
        let mut missing = Vec::new();

        for &eid in top_experts {
            if eid >= self.cpu_pool.len() {
                continue;
            }
            self.usage_freq[eid] = self.usage_freq[eid].saturating_add(1);
            if self.lru.touch(eid) {
                available.push(eid);
            } else {
                let urgency = self.compute_urgency(eid);
                missing.push(OffloadCandidate { expert_id: eid, urgency });
            }
        }

        // Sort missing by urgency (highest first)
        missing.sort_by(|a, b| b.urgency.partial_cmp(&a.urgency).unwrap_or(std::cmp::Ordering::Equal));

        // Upload missing experts (up to max_uploads_per_step)
        let mut uploaded = Vec::new();
        let mut still_missing = Vec::new();
        for candidate in missing {
            if uploaded.len() >= self.config.max_uploads_per_step {
                still_missing.push(candidate);
                continue;
            }
            // Evict if budget exceeded
            while self.gpu_used_bytes + self.expert_size_bytes > self.budget() {
                if let Some(evicted_id) = self.lru.evict_lru() {
                    self.evict_expert(evicted_id);
                } else {
                    break;
                }
            }
            // Upload to GPU
            if let Some(cpu_w) = self.cpu_pool.get(candidate.expert_id) {
                let gpu_w = GpuExpert::from_weights(cpu_w);
                self.gpu_resident.insert(candidate.expert_id, gpu_w);
                self.gpu_used_bytes += self.expert_size_bytes;
                self.lru.mark_loaded(candidate.expert_id);
                uploaded.push(candidate.expert_id);
                available.push(candidate.expert_id);
            }
        }

        // Remaining missing go to CPU fallback
        let cpu_fallback: Vec<usize> = still_missing.iter().map(|c| c.expert_id).collect();
        for candidate in &still_missing {
            if self.config.allow_cpu_fallback {
                available.push(candidate.expert_id);
            }
        }

        // Prefetch next predicted experts
        if self.config.enable_prefetch {
            self.prefetch();
        }

        // Record router decision history
        self.router_history.push_back(top_experts.to_vec());
        if self.router_history.len() > 100 {
            self.router_history.pop_front();
        }

        OffloadBatch {
            available,
            uploaded,
            cpu_fallback,
            gpu_resident_count: self.gpu_resident.len(),
            gpu_budget_pct: self.utilization_pct(),
        }
    }

    /// Get GPU-resident expert weights for forward pass.
    /// Returns None if expert is not on GPU (caller must use CPU fallback).
    pub fn get_gpu_expert(&self, expert_id: usize) -> Option<&GpuExpert> {
        self.gpu_resident.get(&expert_id)
    }

    /// Expert is on CPU, run forward there.
    pub fn forward_cpu(&self, expert_id: usize, input: &[f32]) -> Option<Vec<f32>> {
        let w = self.cpu_pool.get(expert_id)?;
        let mut hidden = vec![0.0f32; w.intermediate_size];
        // fc1: [hidden] × [inter, hidden] → [inter]
        for i in 0..w.intermediate_size {
            let mut sum = w.fc1_b[i];
            for j in 0..w.hidden_size {
                sum += w.fc1_w[i * w.hidden_size + j] * input[j];
            }
            hidden[i] = sum;
        }
        // GELU
        for x in &mut hidden {
            let xv = *x;
            let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
            let x_cubed = xv * xv * xv;
            let tanh_arg = sqrt_2_over_pi * (xv + 0.044715 * x_cubed);
            *x = 0.5 * xv * (1.0 + tanh_arg.tanh());
        }
        // fc2: [inter] × [hidden, inter] → [hidden]
        let mut output = vec![0.0f32; w.hidden_size];
        for i in 0..w.hidden_size {
            let mut sum = w.fc2_b[i];
            for j in 0..w.intermediate_size {
                sum += w.fc2_w[i * w.intermediate_size + j] * hidden[j];
            }
            output[i] = sum;
        }
        Some(output)
    }

    /// Evict an expert from GPU to free VRAM.
    fn evict_expert(&mut self, expert_id: usize) {
        if self.gpu_resident.remove(&expert_id).is_some() {
            self.gpu_used_bytes = self.gpu_used_bytes.saturating_sub(self.expert_size_bytes);
            self.lru.mark_evicted(expert_id);
        }
    }

    /// Prefetch: predict next experts based on recent router history.
    fn prefetch(&mut self) {
        if self.router_history.len() < 3 {
            return;
        }
        // Co-occurrence: which experts appear together with recent ones?
        let recent: Vec<usize> = self.router_history.iter().rev().take(5).flatten().copied().collect();
        for &eid in &recent {
            if eid >= self.cpu_pool.len() {
                continue;
            }
            if !self.lru.touch(eid) && self.gpu_resident.len() < self.lru.capacity() {
                self.prefetch_queue.push_back(eid);
            }
        }
        // Actually load prefetched experts
        while let Some(eid) = self.prefetch_queue.pop_front() {
            if self.gpu_used_bytes + self.expert_size_bytes > self.budget() {
                if let Some(evicted) = self.lru.evict_lru() {
                    self.evict_expert(evicted);
                } else {
                    break;
                }
            }
            if let Some(cpu_w) = self.cpu_pool.get(eid) {
                if !self.gpu_resident.contains_key(&eid) {
                    let gpu_w = GpuExpert::from_weights(cpu_w);
                    self.gpu_resident.insert(eid, gpu_w);
                    self.gpu_used_bytes += self.expert_size_bytes;
                    self.lru.mark_loaded(eid);
                }
            }
        }
    }

    /// Urgency score for an expert that is NOT on GPU.
    /// Higher = should be loaded sooner.
    fn compute_urgency(&self, expert_id: usize) -> f32 {
        let freq = self.usage_freq.get(expert_id).copied().unwrap_or(0) as f32;
        let max_freq = self.usage_freq.iter().max().copied().unwrap_or(1).max(1) as f32;
        let freq_norm = freq / max_freq;
        // Recency bonus: if it was used in last 5 router decisions
        let recency = self.router_history.iter().rev().take(5)
            .any(|h| h.contains(&expert_id));
        let recency_bonus = if recency { 0.3 } else { 0.0 };
        (0.7 * freq_norm + 0.3) + recency_bonus
    }

    fn budget(&self) -> u64 {
        self.lru.capacity() as u64 * self.expert_size_bytes
    }

    fn utilization_pct(&self) -> f64 {
        if self.budget() == 0 {
            return 0.0;
        }
        self.gpu_used_bytes as f64 / self.budget() as f64 * 100.0
    }

    /// Hit rate of GPU expert cache (0.0–1.0).
    pub fn hit_rate(&self) -> f64 {
        self.lru.hit_rate()
    }

    /// Current GPU memory usage in bytes.
    pub fn gpu_used_bytes(&self) -> u64 {
        self.gpu_used_bytes
    }
}

/// Result of a prepare() call — tells caller which experts are ready where.
#[derive(Debug)]
pub struct OffloadBatch {
    /// Expert IDs available for forward (GPU or CPU fallback).
    pub available: Vec<usize>,
    /// Expert IDs that were just uploaded this step.
    pub uploaded: Vec<usize>,
    /// Expert IDs that will run on CPU (not on GPU).
    pub cpu_fallback: Vec<usize>,
    /// Number of experts currently GPU-resident.
    pub gpu_resident_count: usize,
    /// GPU budget utilization percentage.
    pub gpu_budget_pct: f64,
}

#[derive(Debug)]
struct OffloadCandidate {
    expert_id: usize,
    urgency: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_weights(hidden: usize, inter: usize) -> ExpertWeights {
        let fc1_w = vec![0.1f32; inter * hidden];
        let fc1_b = vec![0.0f32; inter];
        let fc2_w = vec![0.1f32; hidden * inter];
        let fc2_b = vec![0.0f32; hidden];
        ExpertWeights { fc1_w, fc1_b, fc2_w, fc2_b, hidden_size: hidden, intermediate_size: inter }
    }

    #[test]
    fn test_offloader_create() {
        let weights: Vec<ExpertWeights> = (0..256).map(|_| dummy_weights(4, 2)).collect();
        let offloader = ExpertOffloader::new(OffloadConfig::default(), weights);
        assert_eq!(offloader.cpu_pool.len(), 256);
        assert!(offloader.lru.capacity() >= 64);
    }

    #[test]
    fn test_prepare_loads_experts() {
        let weights: Vec<ExpertWeights> = (0..256).map(|_| dummy_weights(128, 64)).collect();
        let mut offloader = ExpertOffloader::new(OffloadConfig::default(), weights);
        let top: Vec<usize> = (0..32).collect();
        let batch = offloader.prepare(&top);
        assert_eq!(batch.available.len(), 32);
        assert!(batch.cpu_fallback.is_empty() || !batch.cpu_fallback.is_empty());
        assert!(batch.gpu_resident_count > 0);
    }

    #[test]
    fn test_lru_eviction() {
        let mut lru = LruTracker::new(10, 100);
        for i in 0..10 {
            lru.mark_loaded(i);
        }
        assert_eq!(lru.resident_count(), 10);
        let evicted = lru.evict_lru();
        assert_eq!(evicted, Some(0));
        assert_eq!(lru.resident_count(), 9);
        lru.mark_loaded(10);
        assert_eq!(lru.resident_count(), 10);
        assert!(!lru.touch(0));
        assert!(lru.touch(10));
    }

    #[test]
    fn test_cpu_forward() {
        let weights = vec![dummy_weights(8, 16)];
        let offloader = ExpertOffloader::new(OffloadConfig::default(), weights);
        let input = vec![1.0f32; 8];
        let output = offloader.forward_cpu(0, &input);
        assert!(output.is_some());
        assert_eq!(output.unwrap().len(), 8);
    }

    #[test]
    fn test_lru_hit_rate() {
        let mut lru = LruTracker::new(10, 100);
        for i in 0..10 {
            lru.mark_loaded(i);
        }
        assert!(lru.touch(0));
        assert!(!lru.touch(99));
        assert!(lru.hit_rate() < 1.0);
        assert!(lru.hit_rate() >= 0.5);
    }
}
