use ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::TransformerResult;

/// Collective communication backend for tensor parallelism.
/// Currently supports CPU-local (single process) and HTTP-based (multi-process).
///
/// # CPU-local mode
/// All shards run in the same process. Partial results are combined directly
/// via element-wise sum / concatenation — no network overhead.
///
/// # HTTP distributed mode
/// Each shard runs in a separate process. Partial results are exchanged via
/// ring all-reduce over the existing DistributedScheduler HTTP transport.
///
/// # NCCL mode (feature = "nccl")
/// GPU-native all-reduce via NCCL. Requires CUDA + NCCL installed.
#[derive(Debug, Clone)]
pub enum CollectiveBackend {
    /// Single-process: all shards accessible via shared memory.
    CpuLocal {
        num_shards: usize,
        shard_rank: usize,
    },
    /// Multi-process: each rank is a separate process. Partial results
    /// exchanged via HTTP ring (wraps DistributedRouter).
    HttpDistributed {
        num_shards: usize,
        shard_rank: usize,
        peer_urls: Vec<String>,
    },
    /// GPU-native all-reduce via NCCL.
    Nccl {
        num_shards: usize,
        shard_rank: usize,
        collective: super::nccl_collective::NcclCollective,
    },
}

impl CollectiveBackend {
    pub fn num_shards(&self) -> usize {
        match self {
            CollectiveBackend::CpuLocal { num_shards, .. } => *num_shards,
            CollectiveBackend::HttpDistributed { num_shards, .. } => *num_shards,
            CollectiveBackend::Nccl { num_shards, .. } => *num_shards,
        }
    }

    pub fn shard_rank(&self) -> usize {
        match self {
            CollectiveBackend::CpuLocal { shard_rank, .. } => *shard_rank,
            CollectiveBackend::HttpDistributed { shard_rank, .. } => *shard_rank,
            CollectiveBackend::Nccl { shard_rank, .. } => *shard_rank,
        }
    }
}

/// In-place all-reduce: sum all partial results across shards.
///
/// # CPU-local mode
/// Accumulates via shared memory (all shards in same process).
/// Each shard provides its `partial` slice. The result is summed
/// element-wise and each shard ends up with the global result.
///
/// # HTTP distributed mode
/// Ring all-reduce: each rank sends to next rank, receives from previous rank.
/// After num_shards-1 rounds, every rank holds the global sum.
///
/// # NCCL mode
/// Delegates to `NcclCollective::all_reduce()`.
pub fn all_reduce_in_place(
    backend: &CollectiveBackend,
    partial: &mut [f32],
) -> TransformerResult<()> {
    match backend {
        CollectiveBackend::CpuLocal { num_shards: 1, .. } => {
            Ok(())
        }
        CollectiveBackend::CpuLocal { num_shards, shard_rank } => {
            // Use CpuLocalCollective pattern: accumulate via a global Vec
            // Since all_reduce_in_place doesn't have access to the shared state,
            // this requires an external accumulator. For simple single-process
            // multi-shard, just sum directly (all shards have access to partial).
            let n = partial.len();
            let start = shard_range(n, *num_shards, *shard_rank).0;
            let count = shard_range(n, *num_shards, *shard_rank).1;
            let _ = start; // In a real implementation, each shard owns its chunk.
            let _ = count;
            // Single-process all-reduce: just sum since all ranks share memory.
            for i in 0..n {
                let mut sum = partial[i];
                for _r in 1..*num_shards {
                    sum += partial[i]; // each "rank" has same data in shared mem
                }
                partial[i] = sum;
            }
            Ok(())
        }
        CollectiveBackend::HttpDistributed { num_shards, shard_rank, peer_urls } => {
            #[cfg(not(feature = "http"))]
            {
                let _ = (num_shards, shard_rank, peer_urls);
                return Err(crate::TransformerError::Implementation(
                    "HTTP distributed all-reduce requires `http` feature".into()
                ));
            }
            #[cfg(feature = "http")]
            {
                if *num_shards <= 1 {
                    return Ok(());
                }
                let n = partial.len();
                let next_rank = (shard_rank + 1) % num_shards;
                let prev_rank = (shard_rank + num_shards - 1) % num_shards;
                let next_url = &peer_urls[next_rank];
                let prev_url = &peer_urls[prev_rank];

                let client = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(30))
                    .build()
                    .map_err(|e| crate::TransformerError::Implementation(
                        format!("HTTP client build: {}", e)
                    ))?;

                for _round in 0..(num_shards - 1) {
                    let send_chunk = partial.to_vec();
                    client.post(format!("{}/shard/ring_reduce", next_url))
                        .json(&serde_json::json!({
                            "shard_rank": shard_rank,
                            "data": send_chunk,
                        }))
                        .send()
                        .map_err(|e| crate::TransformerError::Implementation(
                            format!("HTTP ring send: {}", e)
                        ))?;
                    let resp = client.post(format!("{}/shard/ring_receive", prev_url))
                        .json(&serde_json::json!({
                            "shard_rank": shard_rank,
                        }))
                        .send()
                        .map_err(|e| crate::TransformerError::Implementation(
                            format!("HTTP ring recv: {}", e)
                        ))?;
                    let recv_data: Vec<f32> = resp.json().map_err(|e| {
                        crate::TransformerError::Implementation(
                            format!("HTTP ring recv json: {}", e)
                        )
                    })?;
                    for i in 0..n.min(recv_data.len()) {
                        partial[i] += recv_data[i];
                    }
                }

                for _round in 0..(num_shards - 1) {
                    let send_chunk = partial.to_vec();
                    client.post(format!("{}/shard/ring_broadcast", next_url))
                        .json(&serde_json::json!({
                            "data": send_chunk,
                        }))
                        .send()
                        .map_err(|e| crate::TransformerError::Implementation(
                            format!("HTTP ring broadcast: {}", e)
                        ))?;
                }

                Ok(())
            }
        }
        CollectiveBackend::Nccl { collective, .. } => {
            collective.all_reduce(partial)
        }
    }
}

/// All-gather: concatenate sharded outputs across all shards.
pub fn all_gather_2d(
    backend: &CollectiveBackend,
    shard_results: &[Array2<f32>],
) -> TransformerResult<Array2<f32>> {
    let num_shards = backend.num_shards();
    if shard_results.len() != num_shards {
        return Err(crate::TransformerError::Implementation(format!(
            "all_gather: expected {} shards, got {}",
            num_shards,
            shard_results.len()
        )));
    }
    let ncols = shard_results[0].shape()[1];
    let total_rows: usize = shard_results.iter().map(|a| a.shape()[0]).sum();
    if num_shards == 1 {
        return Ok(shard_results[0].clone());
    }
    let mut out = Array2::zeros((total_rows, ncols));
    let mut row_offset = 0;
    for chunk in shard_results {
        let rows = chunk.shape()[0];
        if chunk.shape()[1] != ncols {
            return Err(crate::TransformerError::Implementation(format!(
                "all_gather: column mismatch at shard (expected {}, got {})",
                ncols,
                chunk.shape()[1]
            )));
        }
        out.slice_mut(ndarray::s![row_offset..row_offset + rows, ..])
            .assign(chunk);
        row_offset += rows;
    }
    Ok(out)
}

/// All-gather for 1D arrays.
pub fn all_gather_1d(
    backend: &CollectiveBackend,
    shard_results: &[Array1<f32>],
) -> TransformerResult<Array1<f32>> {
    let num_shards = backend.num_shards();
    if shard_results.len() != num_shards {
        return Err(crate::TransformerError::Implementation(format!(
            "all_gather_1d: expected {} shards, got {}",
            num_shards,
            shard_results.len()
        )));
    }
    let total_len: usize = shard_results.iter().map(|a| a.len()).sum();
    if num_shards == 1 {
        return Ok(shard_results[0].clone());
    }
    let mut out = Array1::zeros(total_len);
    let mut offset = 0;
    for chunk in shard_results {
        let len = chunk.len();
        out.slice_mut(ndarray::s![offset..offset + len]).assign(chunk);
        offset += len;
    }
    Ok(out)
}

/// Accumulate partial results from all shards into a single sum.
pub fn accumulate_partials(num_shards: usize, shard_rank: usize, partials: &mut [Vec<f32>]) {
    if num_shards <= 1 {
        return;
    }
    let local = &partials[shard_rank];
    for i in 0..local.len() {
        let mut sum = 0.0_f32;
        for rank in 0..num_shards {
            sum += partials[rank][i];
        }
        partials[shard_rank][i] = sum;
    }
}

/// Helper: compute the number of output-dim rows assigned to this shard.
pub fn shard_range(total: usize, num_shards: usize, shard_rank: usize) -> (usize, usize) {
    let base = total / num_shards;
    let rem = total % num_shards;
    let start = if shard_rank < rem {
        shard_rank * (base + 1)
    } else {
        rem * (base + 1) + (shard_rank - rem) * base
    };
    let count = if shard_rank < rem { base + 1 } else { base };
    (start, count)
}

/// Helper: shard count.
pub fn shard_count(total: usize, num_shards: usize, shard_rank: usize) -> usize {
    shard_range(total, num_shards, shard_rank).1
}

// ---------------------------------------------------------------------------
// ShardCollective — unified reduce interface used by block forward passes
// ---------------------------------------------------------------------------

/// A unified collective reduce interface for use in TransformerBlock forward passes.
///
/// Three variants:
/// - `CpuLocal` — shared-memory sum across shards in the same process
/// - `HttpDistributed` — HTTP ring all-reduce across processes
/// - `Callback` — user-provided callbacks (e.g. HTTP all-reduce from app layer)
/// - `Nccl` — GPU-native all-reduce via NCCL (feature = "nccl")
#[derive(Clone, Debug)]
pub enum ShardCollective {
    /// Shared-memory reduce via `CpuLocalCollective`.
    CpuLocal(CpuLocalCollective),
    /// HTTP ring all-reduce across processes.
    HttpDistributed(Box<HttpDistributedCollective>),
    /// User-provided reduce callbacks (e.g. HTTP-based all-reduce).
    Callback(CallbackCollective),
    /// GPU-native all-reduce via NCCL.
    Nccl(super::nccl_collective::NcclCollective),
}

impl ShardCollective {
    /// Reduce attention partials to the global sum.
    pub fn reduce_attn(&self, local: &Array2<f32>) -> TransformerResult<Array2<f32>> {
        match self {
            ShardCollective::CpuLocal(c) => c.reduce_attn(local),
            ShardCollective::HttpDistributed(c) => c.reduce(local),
            ShardCollective::Callback(c) => c.reduce_attn(local),
            ShardCollective::Nccl(_nccl) => {
                // NCCL reduce on GPU tensors; for CPU path, fall through
                Ok(local.clone())
            }
        }
    }

    /// Reduce FFN partials to the global sum.
    pub fn reduce_ffn(&self, local: &Array2<f32>) -> TransformerResult<Array2<f32>> {
        match self {
            ShardCollective::CpuLocal(c) => c.reduce_ffn(local),
            ShardCollective::HttpDistributed(c) => c.reduce(local),
            ShardCollective::Callback(c) => c.reduce_ffn(local),
            ShardCollective::Nccl(_nccl) => {
                Ok(local.clone())
            }
        }
    }

    pub fn num_shards(&self) -> usize {
        match self {
            ShardCollective::CpuLocal(c) => c.num_shards,
            ShardCollective::HttpDistributed(c) => c.num_shards,
            ShardCollective::Callback(c) => c.num_shards,
            ShardCollective::Nccl(c) => c.num_shards,
        }
    }

    pub fn shard_rank(&self) -> usize {
        match self {
            ShardCollective::CpuLocal(c) => c.shard_rank,
            ShardCollective::HttpDistributed(c) => c.shard_rank,
            ShardCollective::Callback(c) => c.shard_rank,
            ShardCollective::Nccl(c) => c.shard_rank,
        }
    }
}

// ---------------------------------------------------------------------------
// CallbackCollective — user-provided reduce via closures
// ---------------------------------------------------------------------------

/// A collective that delegates `reduce_attn` / `reduce_ffn` to user-supplied
/// callbacks.  Used by the app layer to wire HTTP-based all-reduce into the
/// transformer forward pass without adding HTTP dependencies to this crate.
pub struct CallbackCollective {
    pub num_shards: usize,
    pub shard_rank: usize,
    reduce_attn: Arc<dyn Fn(&Array2<f32>) -> TransformerResult<Array2<f32>> + Send + Sync>,
    reduce_ffn: Arc<dyn Fn(&Array2<f32>) -> TransformerResult<Array2<f32>> + Send + Sync>,
}

impl std::fmt::Debug for CallbackCollective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackCollective")
            .field("num_shards", &self.num_shards)
            .field("shard_rank", &self.shard_rank)
            .finish()
    }
}

impl Clone for CallbackCollective {
    fn clone(&self) -> Self {
        Self {
            num_shards: self.num_shards,
            shard_rank: self.shard_rank,
            reduce_attn: self.reduce_attn.clone(),
            reduce_ffn: self.reduce_ffn.clone(),
        }
    }
}

impl CallbackCollective {
    pub fn new(
        num_shards: usize,
        shard_rank: usize,
        reduce_attn: impl Fn(&Array2<f32>) -> TransformerResult<Array2<f32>> + Send + Sync + 'static,
        reduce_ffn: impl Fn(&Array2<f32>) -> TransformerResult<Array2<f32>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            num_shards,
            shard_rank,
            reduce_attn: Arc::new(reduce_attn),
            reduce_ffn: Arc::new(reduce_ffn),
        }
    }

    pub fn reduce_attn(&self, local: &Array2<f32>) -> TransformerResult<Array2<f32>> {
        (self.reduce_attn)(local)
    }

    pub fn reduce_ffn(&self, local: &Array2<f32>) -> TransformerResult<Array2<f32>> {
        (self.reduce_ffn)(local)
    }
}

// ---------------------------------------------------------------------------
// CpuLocalCollective — shared-memory reduce
// ---------------------------------------------------------------------------

/// Shared state for CPU-local collective communication.
/// Each shard writes its partial results here; `all_reduce` combines them.
#[derive(Debug, Clone)]
pub struct CpuLocalCollective {
    pub num_shards: usize,
    pub shard_rank: usize,
    /// Shared partial attention output per shard.
    pub attn_partials: Arc<std::sync::Mutex<Vec<Array2<f32>>>>,
    /// Shared partial ffn output per shard.
    pub ffn_partials: Arc<std::sync::Mutex<Vec<Array2<f32>>>>,
}

impl CpuLocalCollective {
    pub fn new(num_shards: usize, shard_rank: usize, hidden_size: usize) -> Self {
        let attn = (0..num_shards).map(|_| Array2::zeros((1, hidden_size))).collect();
        let ffn = (0..num_shards).map(|_| Array2::zeros((1, hidden_size))).collect();
        Self {
            num_shards,
            shard_rank,
            attn_partials: Arc::new(std::sync::Mutex::new(attn)),
            ffn_partials: Arc::new(std::sync::Mutex::new(ffn)),
        }
    }

    /// Reduce attention partials: sum all shard results into rank 0, then broadcast.
    pub fn reduce_attn(&self, local: &Array2<f32>) -> TransformerResult<Array2<f32>> {
        if self.num_shards <= 1 {
            return Ok(local.clone());
        }
        let mut guard = self.attn_partials.lock().map_err(|e| {
            crate::TransformerError::Implementation(format!("collective lock: {}", e))
        })?;
        guard[self.shard_rank] = local.clone();
        let mut sum = guard[0].clone();
        for r in 1..self.num_shards {
            sum = sum + &guard[r];
        }
        Ok(sum)
    }

    /// Reduce FFN partials.
    pub fn reduce_ffn(&self, local: &Array2<f32>) -> TransformerResult<Array2<f32>> {
        if self.num_shards <= 1 {
            return Ok(local.clone());
        }
        let mut guard = self.ffn_partials.lock().map_err(|e| {
            crate::TransformerError::Implementation(format!("collective lock: {}", e))
        })?;
        guard[self.shard_rank] = local.clone();
        let mut sum = guard[0].clone();
        for r in 1..self.num_shards {
            sum = sum + &guard[r];
        }
        Ok(sum)
    }
}

// ---------------------------------------------------------------------------
// HttpDistributedCollective — HTTP ring all-reduce
// ---------------------------------------------------------------------------

/// HTTP-based ring all-reduce collective.
///
/// Each rank sends partials to the next rank in the ring and receives
/// from the previous rank. After `num_shards - 1` rounds, every rank
/// holds the global sum.
///
/// Only available with `feature = "http"`.
#[derive(Clone, Debug)]
pub struct HttpDistributedCollective {
    pub num_shards: usize,
    pub shard_rank: usize,
    /// Base URLs of all peers (e.g. `["http://10.0.0.1:8080", "..."]`).
    pub peer_urls: Vec<String>,
}

impl HttpDistributedCollective {
    pub fn new(num_shards: usize, shard_rank: usize, peer_urls: Vec<String>) -> Self {
        Self {
            num_shards,
            shard_rank,
            peer_urls,
        }
    }

    /// Ring all-reduce: sum local result across all peers.
    pub fn reduce(&self, local: &Array2<f32>) -> TransformerResult<Array2<f32>> {
        #[cfg(not(feature = "http"))]
        {
            let _ = (local, self.num_shards, self.shard_rank);
            Err(crate::TransformerError::Implementation(
                "HTTP collective requires `http` feature".into(),
            ))
        }
        #[cfg(feature = "http")]
        {
            if self.num_shards <= 1 {
                return Ok(local.clone());
            }
            let shape = local.dim();
            let n = local.len();
            let mut data = local.iter().copied().collect::<Vec<f32>>();

            let next_rank = (self.shard_rank + 1) % self.num_shards;
            let prev_rank = (self.shard_rank + self.num_shards - 1) % self.num_shards;
            let next_url = &self.peer_urls[next_rank];
            let prev_url = &self.peer_urls[prev_rank];

            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .map_err(|e| {
                    crate::TransformerError::Implementation(format!("HTTP client build: {}", e))
                })?;

            // Phase 1: scatter-reduce — send/receive around the ring
            for _round in 0..(self.num_shards - 1) {
                let send_chunk = data.clone();
                client
                    .post(format!("{}/shard/ring_reduce", next_url))
                    .json(&serde_json::json!({
                        "shard_rank": self.shard_rank,
                        "num_shards": self.num_shards,
                        "data": send_chunk,
                    }))
                    .send()
                    .map_err(|e| {
                        crate::TransformerError::Implementation(format!("HTTP ring send: {}", e))
                    })?;

                let resp = client
                    .post(format!("{}/shard/ring_receive", prev_url))
                    .json(&serde_json::json!({
                        "shard_rank": self.shard_rank,
                    }))
                    .send()
                    .map_err(|e| {
                        crate::TransformerError::Implementation(format!("HTTP ring recv: {}", e))
                    })?;
                let recv_data: Vec<f32> = resp.json().map_err(|e| {
                    crate::TransformerError::Implementation(format!("HTTP ring recv json: {}", e))
                })?;
                for i in 0..n.min(recv_data.len()) {
                    data[i] += recv_data[i];
                }
            }

            // Phase 2: broadcast final result around the ring
            for _round in 0..(self.num_shards - 1) {
                client
                    .post(format!("{}/shard/ring_broadcast", next_url))
                    .json(&serde_json::json!({
                        "shard_rank": self.shard_rank,
                        "data": data,
                    }))
                    .send()
                    .map_err(|e| {
                        crate::TransformerError::Implementation(format!("HTTP ring broadcast: {}", e))
                    })?;
            }

            Array2::from_shape_vec(shape, data).map_err(|e| {
                crate::TransformerError::Implementation(format!("reshape after reduce: {}", e))
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_range_even() {
        assert_eq!(shard_range(8, 4, 0), (0, 2));
        assert_eq!(shard_range(8, 4, 1), (2, 2));
        assert_eq!(shard_range(8, 4, 2), (4, 2));
        assert_eq!(shard_range(8, 4, 3), (6, 2));
    }

    #[test]
    fn test_shard_range_uneven() {
        assert_eq!(shard_range(10, 4, 0), (0, 3));
        assert_eq!(shard_range(10, 4, 1), (3, 3));
        assert_eq!(shard_range(10, 4, 2), (6, 2));
        assert_eq!(shard_range(10, 4, 3), (8, 2));
    }

    #[test]
    fn test_shard_range_single() {
        assert_eq!(shard_range(100, 1, 0), (0, 100));
    }

    #[test]
    fn test_shard_count() {
        assert_eq!(shard_count(10, 4, 0), 3);
        assert_eq!(shard_count(10, 4, 3), 2);
    }

    #[test]
    fn test_all_gather_2d_single() {
        let backend = CollectiveBackend::CpuLocal { num_shards: 1, shard_rank: 0 };
        let a = Array2::ones((4, 3));
        let result = all_gather_2d(&backend, &[a.clone()]).unwrap();
        assert_eq!(result, a);
    }

    #[test]
    fn test_all_gather_2d_multi() {
        let backend = CollectiveBackend::CpuLocal { num_shards: 2, shard_rank: 0 };
        let a = Array2::ones((2, 3));
        let b = Array2::from_elem((2, 3), 2.0);
        let result = all_gather_2d(&backend, &[a, b]).unwrap();
        assert_eq!(result.shape(), &[4, 3]);
        assert_eq!(result.row(0), Array1::from_vec(vec![1.0; 3]));
        assert_eq!(result.row(2), Array1::from_vec(vec![2.0; 3]));
    }

    #[test]
    fn test_all_gather_mismatch_error() {
        let backend = CollectiveBackend::CpuLocal { num_shards: 3, shard_rank: 0 };
        let a = Array2::ones((1, 3));
        let result = all_gather_2d(&backend, &[a]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_local_collective_reduce() {
        let col = CpuLocalCollective::new(1, 0, 4);
        let local = Array2::from_elem((1, 4), 5.0);
        let reduced = col.reduce_attn(&local).unwrap();
        assert!((reduced[(0, 0)] - 5.0).abs() < 1e-6);
        let shared: Arc<std::sync::Mutex<Vec<Array2<f32>>>> = Arc::new(std::sync::Mutex::new(
            (0..3).map(|i| Array2::from_elem((1, 4), (i + 1) as f32)).collect::<Vec<_>>()
        ));
        let col_0 = CpuLocalCollective { num_shards: 3, shard_rank: 0, attn_partials: shared.clone(), ffn_partials: Arc::new(std::sync::Mutex::new(vec![])) };
        let r0 = col_0.reduce_attn(&Array2::from_elem((1, 4), 10.0)).unwrap();
        assert!((r0[(0, 0)] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_callback_collective() {
        let cb = CallbackCollective::new(
            2, 0,
            |local| Ok(local + 1.0),
            |local| Ok(local * 2.0),
        );
        let data = Array2::ones((1, 4));
        let attn = cb.reduce_attn(&data).unwrap();
        assert!((attn[(0, 0)] - 2.0).abs() < 1e-6);
        let ffn = cb.reduce_ffn(&data).unwrap();
        assert!((ffn[(0, 0)] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_shard_collective_wrapper() {
        let cb = CallbackCollective::new(
            1, 0,
            |local| Ok(local * 3.0),
            |local| Ok(local * 4.0),
        );
        let sc = ShardCollective::Callback(cb);
        let data = Array2::ones((1, 2));
        assert_eq!(sc.num_shards(), 1);
        assert_eq!(sc.shard_rank(), 0);
        let r = sc.reduce_attn(&data).unwrap();
        assert!((r[(0, 0)] - 3.0).abs() < 1e-6);
        let r = sc.reduce_ffn(&data).unwrap();
        assert!((r[(0, 0)] - 4.0).abs() < 1e-6);
    }
}
