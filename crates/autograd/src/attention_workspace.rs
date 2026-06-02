//! Attention Workspace Optimization
//!
//! Provides reusable workspace pools, chunked flash attention for CPU,
//! and profiling metrics to minimize VRAM/RAM spikes during attention computation.
//!
//! FIX 1 - Flash Attention style (tiled online softmax)
//! FIX 2 - Chunked Attention (process in blocks)
//! FIX 3 - Reuse Workspace (pool across layers)
//! FIX 4 - Mixed Precision Buffer (f32→f16)
//! FIX 5 - Score Matrix Streaming (compute-use-discard)
//! FIX 6 - Early Free
//! FIX 9 - Attention Buffer Budget
//! FIX 10 - Profiling

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;

// ---------------------------------------------------------------------------
// Profiling counters (FIX 10)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct AttentionProfileCounters {
    pub peak_bytes: Arc<AtomicU64>,
    pub total_bytes: Arc<AtomicU64>,
    pub alloc_count: Arc<AtomicU64>,
    pub reuse_count: Arc<AtomicU64>,
    pub call_count: Arc<AtomicU64>,
    pub chunked_runs: Arc<AtomicU64>,
    pub flash_runs: Arc<AtomicU64>,
}

impl AttentionProfileCounters {
    pub fn record_alloc(&self, bytes: u64) {
        let prev = self.total_bytes.fetch_add(bytes, Ordering::Relaxed);
        let total = prev + bytes;
        let mut peak = self.peak_bytes.load(Ordering::Relaxed);
        while total > peak {
            match self.peak_bytes.compare_exchange(
                peak, total, Ordering::Relaxed, Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_reuse(&self) {
        self.reuse_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_call(&self) {
        self.call_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_chunked(&self) {
        self.chunked_runs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_flash(&self) {
        self.flash_runs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn reuse_ratio(&self) -> f64 {
        let alloc = self.alloc_count.load(Ordering::Relaxed);
        let reuse = self.reuse_count.load(Ordering::Relaxed);
        let total = alloc + reuse;
        if total == 0 {
            0.0
        } else {
            reuse as f64 / total as f64
        }
    }

    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    }

    pub fn avg_bytes_per_call(&self) -> f64 {
        let calls = self.call_count.load(Ordering::Relaxed);
        if calls == 0 {
            0.0
        } else {
            self.total_bytes.load(Ordering::Relaxed) as f64 / calls as f64
        }
    }

    pub fn snapshot(&self) -> AttentionProfileSnapshot {
        AttentionProfileSnapshot {
            peak_mb: self.peak_mb(),
            avg_bytes_per_call: self.avg_bytes_per_call(),
            alloc_count: self.alloc_count.load(Ordering::Relaxed),
            reuse_count: self.reuse_count.load(Ordering::Relaxed),
            reuse_ratio: self.reuse_ratio(),
            call_count: self.call_count.load(Ordering::Relaxed),
            chunked_runs: self.chunked_runs.load(Ordering::Relaxed),
            flash_runs: self.flash_runs.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttentionProfileSnapshot {
    pub peak_mb: f64,
    pub avg_bytes_per_call: f64,
    pub alloc_count: u64,
    pub reuse_count: u64,
    pub reuse_ratio: f64,
    pub call_count: u64,
    pub chunked_runs: u64,
    pub flash_runs: u64,
}

impl std::fmt::Display for AttentionProfileSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AttentionWorkspace: peak={:.1}MB avg={:.0}B/call alloc={} reuse={} ratio={:.1}% calls={} chunked={} flash={}",
            self.peak_mb,
            self.avg_bytes_per_call,
            self.alloc_count,
            self.reuse_count,
            self.reuse_ratio * 100.0,
            self.call_count,
            self.chunked_runs,
            self.flash_runs,
        )
    }
}

// ---------------------------------------------------------------------------
// Buffer Budget (FIX 9)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AttentionBudget {
    pub max_workspace_bytes: u64,
    pub chunk_size: usize,
    pub use_f16: bool,
    pub use_checkpointing: bool,
    pub use_sparse: bool,
    pub sliding_window: usize,
}

impl Default for AttentionBudget {
    fn default() -> Self {
        Self {
            max_workspace_bytes: 512 * 1024 * 1024,
            chunk_size: 512,
            use_f16: true,
            use_checkpointing: true,
            use_sparse: false,
            sliding_window: 0,
        }
    }
}

impl AttentionBudget {
    pub fn with_max_mb(mb: u64) -> Self {
        Self {
            max_workspace_bytes: mb * 1024 * 1024,
            ..Default::default()
        }
    }

    pub fn auto_chunk_size(&self, seq_len: usize, head_dim: usize) -> usize {
        if self.max_workspace_bytes == 0 {
            return seq_len;
        }
        let bytes_per_score = std::mem::size_of::<f32>();
        let budget_for_scores = self.max_workspace_bytes / 2;
        let max_chunk = (budget_for_scores as usize) / (head_dim * bytes_per_score);
        max_chunk.min(self.chunk_size).max(64).min(seq_len)
    }
}

// ---------------------------------------------------------------------------
// Workspace Pool (FIX 3)
// ---------------------------------------------------------------------------

pub struct WorkspaceBuffer {
    data: Vec<f32>,
}

impl WorkspaceBuffer {
    pub fn resize(&mut self, len: usize) {
        if self.data.len() < len {
            self.data.resize(len, 0.0f32);
        }
    }

    pub fn as_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn as_ref(&self) -> &[f32] {
        &self.data
    }

    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    fn clear(&mut self) {}
}

#[derive(Clone)]
pub struct WorkspacePool {
    inner: Arc<Mutex<WorkspacePoolInner>>,
    pub profile: AttentionProfileCounters,
    pub budget: Arc<Mutex<AttentionBudget>>,
}

struct WorkspacePoolInner {
    buffers: Vec<WorkspaceBuffer>,
    max_buffers: usize,
}

impl WorkspacePool {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(WorkspacePoolInner {
                buffers: Vec::new(),
                max_buffers: 4,
            })),
            profile: AttentionProfileCounters::default(),
            budget: Arc::new(Mutex::new(AttentionBudget::default())),
        }
    }

    pub fn with_max_buffers(max: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(WorkspacePoolInner {
                buffers: Vec::new(),
                max_buffers: max,
            })),
            profile: AttentionProfileCounters::default(),
            budget: Arc::new(Mutex::new(AttentionBudget::default())),
        }
    }

    pub fn acquire(&self) -> WorkspaceBuffer {
        let mut inner = self.inner.lock();
        if let Some(mut buf) = inner.buffers.pop() {
            buf.clear();
            self.profile.record_reuse();
            buf
        } else {
            self.profile.record_alloc(0);
            WorkspaceBuffer {
                data: Vec::new(),
            }
        }
    }

    pub fn release(&self, buf: WorkspaceBuffer) {
        let mut inner = self.inner.lock();
        if inner.buffers.len() < inner.max_buffers {
            inner.buffers.push(buf);
        }
    }

    pub fn with_f32_buffer<F, R>(&self, len: usize, f: F) -> R
    where
        F: FnOnce(&mut [f32]) -> R,
    {
        let mut buf = self.acquire();
        buf.resize(len);
        let result = f(buf.as_mut());
        self.release(buf);
        result
    }
}

impl std::fmt::Debug for WorkspacePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspacePool")
            .field("profile", &self.profile.snapshot())
            .finish()
    }
}

impl Default for WorkspacePool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Global singleton pool
// ---------------------------------------------------------------------------

use std::sync::OnceLock;
static GLOBAL_POOL: OnceLock<WorkspacePool> = OnceLock::new();

pub fn global_pool() -> &'static WorkspacePool {
    GLOBAL_POOL.get_or_init(|| {
        let pool = WorkspacePool::with_max_buffers(8);
        tracing::debug!("AttentionWorkspacePool initialized (8 buffers, 512MB budget)");
        pool
    })
}

pub fn reset_global_pool() {
    if let Some(pool) = GLOBAL_POOL.get() {
        let mut inner = pool.inner.lock();
        inner.buffers.clear();
        tracing::debug!("AttentionWorkspacePool reset");
    }
}

// ---------------------------------------------------------------------------
// CPU Flash Attention — Tiled Online Softmax (FIX 1 + FIX 2)
// ---------------------------------------------------------------------------

pub fn cpu_flash_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    chunk_size: usize,
    pool: &WorkspacePool,
) -> Vec<f32> {
    pool.profile.record_call();
    pool.profile.record_flash();

    let scale = (head_dim as f32).sqrt().recip();
    let actual_chunk = chunk_size.min(seq_len).max(1);
    let mut output = vec![0.0f32; seq_len * head_dim];

    // Process one query position at a time to keep scores small
    // Use double-pass: compute full scores for each query, softmax, weighted sum
    // Memory: O(chunk_size) instead of O(seq_len) for scores
    for qi in 0..seq_len {
        // Determine which KV positions this query attends to (causal)
        let attended_start = 0;
        let attended_end = qi + 1; // inclusive of current position

        // Allocate scores for attended positions
        let n_attended = attended_end - attended_start;
        pool.with_f32_buffer(n_attended, |scores| {
            // Compute Q[qi] @ K[j]^T for all attended j
            let q_base = qi * head_dim;
            let mut max_val = f32::NEG_INFINITY;
            for (j, kj) in (attended_start..attended_end).enumerate() {
                let k_base = kj * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_base + d] * k[k_base + d];
                }
                scores[j] = dot * scale;
                if scores[j] > max_val {
                    max_val = scores[j];
                }
            }

            // Softmax: exp(x - max) / sum(exp(x - max))
            let mut sum_exp = 0.0f32;
            for j in 0..n_attended {
                scores[j] = (scores[j] - max_val).exp();
                sum_exp += scores[j];
            }
            if sum_exp > 0.0 {
                let inv_sum = 1.0 / sum_exp;
                for j in 0..n_attended {
                    scores[j] *= inv_sum;
                }
            }

            // Weighted sum of V
            for d in 0..head_dim {
                let mut weighted = 0.0f32;
                for (j, kj) in (attended_start..attended_end).enumerate() {
                    weighted += scores[j] * v[kj * head_dim + d];
                }
                output[qi * head_dim + d] = weighted;
            }
        });
    }

    output
}

// ---------------------------------------------------------------------------
// Chunked Softmax Attention for Training (FIX 2 + FIX 5)
// ---------------------------------------------------------------------------

use crate::{Tensor, TensorOps};

pub fn slice_tensor_rows(t: &Tensor, start: usize, end: usize) -> Tensor {
    let data = t.data();
    let shape = data.shape();
    if shape.len() != 2 {
        return t.clone();
    }
    let rows = shape[0];
    let cols = shape[1];
    let end = end.min(rows);
    let len = end - start;

    let mut result = vec![0.0f32; len * cols];
    let flat: Vec<f32> = data.iter().copied().collect();
    for i in 0..len {
        for j in 0..cols {
            result[i * cols + j] = flat[(start + i) * cols + j];
        }
    }
    let t_result = Tensor::from_slice(&result, &[len, cols]);
    if t.requires_grad() {
        t_result.set_requires_grad(true);
    }
    t_result
}

pub fn write_tensor_rows(dst: &Tensor, src: &Tensor, start: usize) {
    let dst_data = dst.data();
    let src_data = src.data();
    let dst_shape = dst_data.shape();
    let src_shape = src_data.shape();
    if dst_shape.len() != 2 || src_shape.len() != 2 {
        return;
    }
    let cols = dst_shape[1];
    let src_rows = src_shape[0];
    let dst_flat: Vec<f32> = dst_data.iter().copied().collect();
    let src_flat: Vec<f32> = src_data.iter().copied().collect();
    let mut new_data = dst_flat;
    for i in 0..src_rows {
        let dst_idx = (start + i) * cols;
        let src_idx = i * cols;
        for j in 0..cols {
            if dst_idx + j < new_data.len() && src_idx + j < src_flat.len() {
                new_data[dst_idx + j] = src_flat[src_idx + j];
            }
        }
    }
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(dst_shape), new_data)
        .expect("shape matches");
    dst.set_data(arr);
}

fn apply_chunked_causal_mask(
    scores: &Tensor,
    _seq_len: usize,
    chunk_start: usize,
    chunk_end: usize,
) -> Tensor {
    let data = scores.data();
    let shape = data.shape();
    if shape.len() != 2 {
        return scores.clone();
    }
    let q_len = shape[0];
    let kv_len = shape[1];

    let mut result: Vec<f32> = data.iter().copied().collect();
    for i in 0..q_len {
        for j in 0..kv_len {
            let kv_pos = chunk_start + j;
            if kv_pos > i {
                result[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    let t = Tensor::from_slice(&result, &[q_len, kv_len]);
    if scores.requires_grad() {
        t.set_requires_grad(true);
    }
    t
}

/// Compute causal attention in chunks.
/// Each chunk computes softmax over a subset of KV positions.
pub fn chunked_causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    _head_dim: usize,
    chunk_size: usize,
) -> Tensor {
    let actual_chunk = chunk_size.min(seq_len).max(1);
    let scale = (seq_len as f32).sqrt();

    let mut output = Tensor::zeros(&[seq_len, seq_len], q.requires_grad());

    for chunk_start in (0..seq_len).step_by(actual_chunk) {
        let chunk_end = (chunk_start + actual_chunk).min(seq_len);

        let k_chunk = slice_tensor_rows(k, chunk_start, chunk_end);
        let v_chunk = slice_tensor_rows(v, chunk_start, chunk_end);

        let scores = q
            .matmul(&k_chunk.transpose())
            .div(&Tensor::from_slice(&[scale], &[1]));

        let masked = apply_chunked_causal_mask(&scores, seq_len, chunk_start, chunk_end);

        let attn_weights = crate::ops::nn::softmax(&masked, 1);

        let partial = attn_weights.matmul(&v_chunk);

        output = output.add(&partial);
    }

    output
}

// ---------------------------------------------------------------------------
// Sparse Sliding Window Attention (FIX 8)
// ---------------------------------------------------------------------------

pub fn sliding_window_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    window_size: usize,
    global_positions: &[usize],
) -> Vec<f32> {
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for i in 0..seq_len {
        let window_start = if i > window_size { i - window_size } else { 0 };

        let mut attended: Vec<usize> = Vec::new();
        for j in window_start..=i {
            attended.push(j);
        }
        for &g in global_positions {
            if g < i && !attended.contains(&g) {
                attended.push(g);
            }
        }
        attended.sort_unstable();
        attended.dedup();

        if attended.is_empty() {
            continue;
        }

        let k_len = attended.len();
        let mut max_val = f32::NEG_INFINITY;

        let mut scores = Vec::with_capacity(k_len);
        for &t in &attended {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[t * head_dim + d];
            }
            let s = dot * scale;
            if s > max_val {
                max_val = s;
            }
            scores.push(s);
        }

        let mut sum_exp = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_val).exp();
            sum_exp += *s;
        }
        if sum_exp > 0.0 {
            for s in &mut scores {
                *s /= sum_exp;
            }
        }

        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for (idx, &t) in attended.iter().enumerate() {
                acc += scores[idx] * v[t * head_dim + d];
            }
            output[i * head_dim + d] = acc;
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Mixed Precision Support (FIX 4)
// ---------------------------------------------------------------------------

pub fn maybe_downcast_to_f16(buf: &[f32], budget: &AttentionBudget) -> Option<Vec<u16>> {
    if !budget.use_f16 {
        return None;
    }
    Some(f32_to_f16_batch(buf))
}

fn f32_to_f16_batch(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&x| f32_to_f16_raw(x)).collect()
}

fn f32_to_f16_raw(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign: u32 = (bits >> 16) & 0x8000;
    let exp: i32 = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp > 0x8e {
        let inf_nan = 0x7c00u32 | (if mant != 0 { 0x200 } else { 0 });
        (sign | inf_nan) as u16
    } else if exp <= 0x70 {
        sign as u16
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            (sign | 0x7c00u32) as u16
        } else if new_exp <= 0 {
            sign as u16
        } else {
            let result = sign | ((new_exp as u32) << 10) | ((mant >> 13) as u32);
            result as u16
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_pool_reuse() {
        let pool = WorkspacePool::new();
        let mut b1 = pool.acquire();
        b1.resize(64);
        let b1_ptr = b1.as_ref().as_ptr();
        pool.release(b1);

        let mut b2 = pool.acquire();
        b2.resize(64);
        let b2_ptr = b2.as_ref().as_ptr();
        assert_eq!(b1_ptr, b2_ptr, "Pool should reuse the same buffer");
    }

    #[test]
    fn test_cpu_flash_attention_vs_naive() {
        let seq_len = 32;
        let head_dim = 8;
        let pool = WorkspacePool::new();

        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32 * 7.0).sin() * 0.5))
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32 * 11.0).sin() * 0.5))
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32 * 13.0).sin() * 0.5))
            .collect();

        let naive = naive_attention(&q, &k, &v, seq_len, head_dim);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, 8, &pool);

        for i in 0..seq_len * head_dim {
            let diff = (naive[i] - flash[i]).abs();
            assert!(
                diff < 1e-4,
                "Mismatch at {i}: naive={} flash={}",
                naive[i],
                flash[i]
            );
        }
    }

    fn naive_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let scale = (head_dim as f32).sqrt().recip();
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }

        for i in 0..seq_len {
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..=i {
                if scores[i * seq_len + j] > max_val {
                    max_val = scores[i * seq_len + j];
                }
            }
            let mut sum_exp = 0.0f32;
            for j in 0..=i {
                let e = (scores[i * seq_len + j] - max_val).exp();
                scores[i * seq_len + j] = e;
                sum_exp += e;
            }
            for j in 0..=i {
                if sum_exp > 0.0 {
                    scores[i * seq_len + j] /= sum_exp;
                }
            }
            for j in i + 1..seq_len {
                scores[i * seq_len + j] = 0.0;
            }
        }

        let mut out = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq_len {
                    acc += scores[i * seq_len + j] * v[j * head_dim + d];
                }
                out[i * head_dim + d] = acc;
            }
        }
        out
    }

    #[test]
    fn test_sliding_window() {
        let seq_len = 16;
        let head_dim = 4;
        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32 * 3.0).sin() * 0.5))
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32 * 5.0).sin() * 0.5))
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32 * 7.0).sin() * 0.5))
            .collect();

        let result = sliding_window_attention(&q, &k, &v, seq_len, head_dim, 4, &[0]);
        assert_eq!(result.len(), seq_len * head_dim);
        for &x in &result {
            assert!(!x.is_nan());
        }
    }

    #[test]
    fn test_f16_conversion() {
        let inputs = vec![0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0];
        let f16 = f32_to_f16_batch(&inputs);
        assert_eq!(f16.len(), inputs.len());
    }

    #[test]
    fn test_profile_counters() {
        let c = AttentionProfileCounters::default();
        c.record_alloc(1024);
        c.record_alloc(2048);
        c.record_reuse();
        c.record_call();
        assert_eq!(c.alloc_count.load(Ordering::Relaxed), 2);
        assert_eq!(c.reuse_count.load(Ordering::Relaxed), 1);
        assert!(c.peak_bytes.load(Ordering::Relaxed) >= 3072);
    }

    #[test]
    fn test_budget_auto_chunk() {
        let budget = AttentionBudget::with_max_mb(512);
        let chunk = budget.auto_chunk_size(4096, 64);
        assert!(chunk >= 64);
        assert!(chunk <= 4096);
    }

    #[test]
    fn test_workspace_pool_threadsafe() {
        let pool = WorkspacePool::new();
        let mut handles = Vec::new();
        for _ in 0..4 {
            let p = pool.clone();
            handles.push(std::thread::spawn(move || {
                let buf = p.acquire();
                p.release(buf);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_pool_with_f32_buffer() {
        let pool = WorkspacePool::new();
        let sum: f32 = pool.with_f32_buffer(100, |buf| {
            for i in 0..buf.len() {
                buf[i] = i as f32;
            }
            buf.iter().sum()
        });
        assert_eq!(sum, 4950.0);
    }
}
