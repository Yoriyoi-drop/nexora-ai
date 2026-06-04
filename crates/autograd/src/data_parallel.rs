//! Data-parallel training utilities.
//!
//! Provides gradient accumulation (`GradientAccumulator`), gradient averaging
//! across workers (`all_reduce_gradients`), multi-worker coordination
//! (`DataParallel`), and GPU-native gradient allreduce (`gpu_allreduce_gradients`).
//!
//! ## GPU support
//!
//! When the `gpu` feature is enabled, [`GradientAccumulator`] stores and
//! accumulates gradients entirely on-device with no CPU round-trip.
//! [`gpu_allreduce_gradients`] averages gradient tensors across model replicas
//! using the GPU Gradient AllReduce WGSL kernel (`GRADIENT_ALLREDUCE_WGSL`),
//! eliminating CPU readback for gradient synchronization.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::ArrayD;

use super::device::Storage;
use super::Tensor;

#[cfg(feature = "gpu")]
use crate::gpu::GpuContext;

// ─── Gradient Accumulation ────────────────────────────────────────────────────

/// Accumulates gradients across multiple micro-batches before each optimizer step.
///
/// Works with both CPU and GPU storage. When gradients live on GPU,
/// accumulation happens entirely on-device with no CPU round-trip.
///
/// Usage:
/// ```ignore
/// let mut acc = GradientAccumulator::new(4);
/// for batch in micro_batches {
///     let loss = model.forward(&batch).sum();
///     loss.backward();
///     acc.accumulate(&model);
///     if acc.ready() {
///         acc.finalize(&model);
///         optimizer.step();
///         optimizer.zero_grad();
///         acc.reset();
///     }
/// }
/// ```
pub struct GradientAccumulator {
    num_micro_batches: usize,
    counter: usize,
    stash: HashMap<usize, Storage>,
}

impl GradientAccumulator {
    pub fn new(num_micro_batches: usize) -> Self {
        assert!(num_micro_batches >= 1, "num_micro_batches must be >= 1");
        Self {
            num_micro_batches,
            counter: 0,
            stash: HashMap::new(),
        }
    }

    pub fn accumulate(&mut self, params: &[Tensor]) {
        self.counter += 1;
        let scale = 1.0 / self.num_micro_batches as f32;

        for (i, p) in params.iter().enumerate() {
            let grad = get_storage_grad(p);
            if let Some(g) = grad {
                let scaled = storage_scale(&g, scale);
                stash_add_inplace(&mut self.stash, i, scaled);
            }
        }
    }

    pub fn ready(&self) -> bool {
        self.counter >= self.num_micro_batches
    }

    pub fn finalize(&self, params: &[Tensor]) {
        for (i, p) in params.iter().enumerate() {
            if let Some(g) = self.stash.get(&i) {
                set_storage_grad(p, g);
            }
        }
    }

    pub fn reset(&mut self) {
        self.counter = 0;
        self.stash.clear();
    }

    pub fn progress(&self) -> f32 {
        self.counter as f32 / self.num_micro_batches as f32
    }

    pub fn counter(&self) -> usize {
        self.counter
    }

    pub fn num_micro_batches(&self) -> usize {
        self.num_micro_batches
    }
}

// ─── Gradient Reduction ───────────────────────────────────────────────────────

/// Average gradients from multiple sources (workers or devices).
/// Mutates the first gradient set in-place to the average.
pub fn all_reduce_gradients(gradients: &mut [Vec<ArrayD<f32>>]) {
    if gradients.is_empty() {
        return;
    }
    let num_workers = gradients.len() as f32;
    if num_workers <= 1.0 {
        return;
    }
    let num_params = gradients[0].len();

    for param_idx in 0..num_params {
        let mut sum = gradients[0][param_idx].clone();
        for worker_idx in 1..gradients.len() {
            if worker_idx < gradients.len() && param_idx < gradients[worker_idx].len() {
                sum = &sum + &gradients[worker_idx][param_idx];
            }
        }
        gradients[0][param_idx] = &sum / num_workers;
    }
}

/// GPU-native gradient allreduce: average gradients across model replicas.
///
/// Uses the GPU Gradient AllReduce WGSL kernel (`GRADIENT_ALLREDUCE_WGSL`)
/// which operates on a flat interleaved buffer `[replica0 | replica1 | ...]`.
/// Eliminates CPU readback for gradient synchronization.
///
/// `grad_tensors`: one gradient tensor per replica for a single parameter.
/// All tensors must be on the same GPU device.
#[cfg(feature = "gpu")]
pub fn gpu_allreduce_gradients(
    ctx: &crate::gpu::GpuContext,
    grad_tensors: &[&crate::gpu::GpuTensor],
) -> Result<(), crate::gpu::GpuError> {
    let n = grad_tensors.len();
    if n <= 1 {
        return Ok(());
    }

    let numel = grad_tensors[0].numel();
    let total_elements = numel * n;

    // Allocate flat buffer: [replica0 | replica1 | ...]
    let flat_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dp_allreduce_flat"),
        size: (total_elements * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dp_allreduce_copy"),
        });
    for (i, g) in grad_tensors.iter().enumerate() {
        let offset = (i * numel * 4) as u64;
        enc.copy_buffer_to_buffer(g.buffer(), 0, &flat_buffer, offset, (numel * 4) as u64);
    }
    ctx.queue.submit(Some(enc.finish()));

    // Allreduce on flat buffer
    let flat_tensor = crate::gpu::GpuTensor {
        shape: vec![total_elements],
        buffer: flat_buffer,
        dtype: crate::gpu::GpuDtype::F32,
        device_id: 0,
    };
    let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dp_allreduce_out"),
        size: (numel * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_tensor = crate::gpu::GpuTensor {
        shape: vec![numel],
        buffer: out_buffer,
        dtype: crate::gpu::GpuDtype::F32,
        device_id: 0,
    };
    ctx.gradient_allreduce(&flat_tensor, &out_tensor, n as u32)?;

    // Copy averaged gradients back to each replica
    let mut cb = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dp_allreduce_copy_back"),
        });
    for g in grad_tensors {
        cb.copy_buffer_to_buffer(out_tensor.buffer(), 0, g.buffer(), 0, (numel * 4) as u64);
    }
    ctx.queue.submit(Some(cb.finish()));

    Ok(())
}

// ─── Data-Parallel Config ─────────────────────────────────────────────────────

/// Configuration for data-parallel training
#[derive(Clone, Debug)]
pub struct DataParallelConfig {
    pub num_workers: usize,
    pub batch_size_per_worker: usize,
    pub gradient_accumulation: bool,
    pub accumulation_steps: usize,
}

impl Default for DataParallelConfig {
    fn default() -> Self {
        Self {
            num_workers: 1,
            batch_size_per_worker: 32,
            gradient_accumulation: false,
            accumulation_steps: 1,
        }
    }
}

impl DataParallelConfig {
    pub fn effective_batch_size(&self) -> usize {
        let per_step = self.num_workers * self.batch_size_per_worker;
        if self.gradient_accumulation {
            per_step * self.accumulation_steps
        } else {
            per_step
        }
    }

    pub fn auto_configure() -> Self {
        let num_cpus = num_cpus::get();
        Self {
            num_workers: num_cpus.max(1),
            batch_size_per_worker: 32,
            gradient_accumulation: true,
            accumulation_steps: 4,
        }
    }
}

// ─── Worker Result ────────────────────────────────────────────────────────────

/// Result from a single data-parallel worker
pub struct WorkerResult {
    pub worker_id: usize,
    pub gradients: Vec<ArrayD<f32>>,
    pub loss: f32,
    pub num_samples: usize,
}

// ─── Data-Parallel Coordinator ────────────────────────────────────────────────

/// Coordinates data-parallel training across multiple workers.
///
/// Each worker processes a shard of the batch independently. Gradients are
/// collected and averaged before the optimizer step.
pub struct DataParallel {
    config: DataParallelConfig,
}

impl DataParallel {
    pub fn new(config: DataParallelConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &DataParallelConfig {
        &self.config
    }

    pub fn shard_indices(&self, total_samples: usize) -> Vec<std::ops::Range<usize>> {
        let per_worker = total_samples / self.config.num_workers;
        let remainder = total_samples % self.config.num_workers;
        let mut ranges = Vec::with_capacity(self.config.num_workers);
        let mut start = 0;
        for i in 0..self.config.num_workers {
            let extra = if i < remainder { 1 } else { 0 };
            let end = start + per_worker + extra;
            ranges.push(start..end);
            start = end;
        }
        ranges
    }

    pub fn reduce_gradients(&self, results: &mut [WorkerResult]) {
        let grads: Vec<&mut Vec<ArrayD<f32>>> =
            results.iter_mut().map(|r| &mut r.gradients).collect();
        let mut grad_refs: Vec<Vec<ArrayD<f32>>> = Vec::new();
        for g in grads {
            grad_refs.push(std::mem::take(g));
        }
        all_reduce_gradients(&mut grad_refs);
        for (i, g) in grad_refs.into_iter().enumerate() {
            results[i].gradients = g;
        }
    }

    pub fn total_loss(&self, results: &[WorkerResult]) -> f32 {
        let total: f32 = results.iter().map(|r| r.loss * r.num_samples as f32).sum();
        let total_samples: usize = results.iter().map(|r| r.num_samples).sum();
        if total_samples > 0 {
            total / total_samples as f32
        } else {
            0.0
        }
    }
}

// ─── Internal helpers: Storage gradient ops ───────────────────────────────────

/// Read gradient as [`Storage`] without forcing a GPU→CPU readback when the
/// gradient already lives on GPU.
fn get_storage_grad(p: &Tensor) -> Option<Storage> {
    #[cfg(feature = "gpu")]
    if let Some(s) = p.grad_storage() {
        return Some(s);
    }
    p.grad().map(Storage::from)
}

/// Scale a [`Storage`] by `scale`. GPU gradients are scaled on-device via
/// [`GpuContext::scale_inplace`]; CPU gradients use ndarray arithmetic.
fn storage_scale(s: &Storage, scale: f32) -> Storage {
    match s {
        Storage::Cpu(arr) => Storage::Cpu(Arc::new(arr.as_ref() * scale)),
        #[cfg(feature = "gpu")]
        Storage::Gpu(t) => {
            let copy = t.clone();
            if let Ok(ctx) = GpuContext::global() {
                let _ = ctx.scale_inplace(&copy, scale);
            }
            Storage::Gpu(copy)
        }
    }
}

/// Add a scaled gradient into the stash. Handles all CPU/GPU cross-product cases.
fn stash_add_inplace(stash: &mut HashMap<usize, Storage>, idx: usize, grad: Storage) {
    use std::collections::hash_map::Entry;

    match stash.entry(idx) {
        Entry::Occupied(mut entry) => {
            let existing = entry.get_mut();
            match (existing, &grad) {
                (Storage::Cpu(ref mut e), Storage::Cpu(g)) => {
                    *e = Arc::new(e.as_ref() + g.as_ref());
                }
                #[cfg(feature = "gpu")]
                (Storage::Cpu(ref mut e), Storage::Gpu(g)) => {
                    if let Ok(g_cpu) = g.to_cpu() {
                        *e = Arc::new(e.as_ref() + &g_cpu);
                    }
                }
                #[cfg(feature = "gpu")]
                (Storage::Gpu(ref mut e), Storage::Cpu(g)) => {
                    if let Ok(ctx) = GpuContext::global() {
                        if let Ok(g_gpu) = crate::gpu::GpuTensor::from_cpu(g) {
                            let _ = ctx.add_inplace(e, &g_gpu);
                        }
                    }
                }
                #[cfg(feature = "gpu")]
                (Storage::Gpu(ref mut e), Storage::Gpu(g)) => {
                    if let Ok(ctx) = GpuContext::global() {
                        let _ = ctx.add_inplace(e, g);
                    }
                }
            }
        }
        Entry::Vacant(entry) => {
            entry.insert(grad);
        }
    }
}

/// Set a gradient on a [`Tensor`] from a [`Storage`] reference.
/// GPU gradients use [`Tensor::set_grad_storage`] to avoid a CPU round-trip.
fn set_storage_grad(p: &Tensor, g: &Storage) {
    match g {
        Storage::Cpu(arr) => p.set_grad(arr.as_ref().clone()),
        #[cfg(feature = "gpu")]
        Storage::Gpu(t) => {
            p.set_grad_storage(Storage::Gpu(t.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::TensorOps;

    #[test]
    fn test_gradient_accumulator_basics() {
        let acc = GradientAccumulator::new(4);
        assert_eq!(acc.num_micro_batches(), 4);
        assert_eq!(acc.counter(), 0);
        assert!(!acc.ready());
        assert!((acc.progress() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator_ready() {
        let mut acc = GradientAccumulator::new(3);
        let p = Tensor::randn(&[4], true);
        for _ in 0..3 {
            acc.accumulate(&[p.clone()]);
        }
        assert!(acc.ready());
        assert!((acc.progress() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator_reset() {
        let mut acc = GradientAccumulator::new(2);
        let p = Tensor::randn(&[4], true);
        acc.accumulate(&[p.clone()]);
        assert!(!acc.ready());
        acc.reset();
        assert_eq!(acc.counter(), 0);
        assert!(!acc.ready());
    }

    #[test]
    fn test_all_reduce_single_worker() {
        let mut grads = vec![vec![
            ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()
        ]];
        all_reduce_gradients(&mut grads);
        assert_eq!(grads[0][0][0], 1.0);
        assert_eq!(grads[0][0][1], 2.0);
        assert_eq!(grads[0][0][2], 3.0);
    }

    #[test]
    fn test_all_reduce_two_workers() {
        let mut grads = vec![
            vec![ArrayD::from_shape_vec(vec![2], vec![2.0, 4.0]).unwrap()],
            vec![ArrayD::from_shape_vec(vec![2], vec![4.0, 6.0]).unwrap()],
        ];
        all_reduce_gradients(&mut grads);
        assert!((grads[0][0][0] - 3.0).abs() < 1e-6);
        assert!((grads[0][0][1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_all_reduce_empty() {
        let mut grads: Vec<Vec<ArrayD<f32>>> = vec![];
        all_reduce_gradients(&mut grads);
    }

    #[test]
    fn test_data_parallel_config_default() {
        let config = DataParallelConfig::default();
        assert_eq!(config.num_workers, 1);
        assert_eq!(config.batch_size_per_worker, 32);
    }

    #[test]
    fn test_data_parallel_config_effective_batch_size() {
        let config = DataParallelConfig {
            num_workers: 4,
            batch_size_per_worker: 16,
            gradient_accumulation: true,
            accumulation_steps: 2,
        };
        assert_eq!(config.effective_batch_size(), 128);
    }

    #[test]
    fn test_shard_indices() {
        let dp = DataParallel::new(DataParallelConfig {
            num_workers: 4,
            ..Default::default()
        });
        let shards = dp.shard_indices(100);
        assert_eq!(shards.len(), 4);
        let total: usize = shards.iter().map(|r| r.len()).sum();
        assert_eq!(total, 100);
        assert_eq!(shards[0].len(), 25);
        assert_eq!(shards[1].len(), 25);
        assert_eq!(shards[2].len(), 25);
        assert_eq!(shards[3].len(), 25);
    }

    #[test]
    fn test_shard_indices_exact() {
        let dp = DataParallel::new(DataParallelConfig {
            num_workers: 3,
            ..Default::default()
        });
        let shards = dp.shard_indices(9);
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0].len(), 3);
        assert_eq!(shards[1].len(), 3);
        assert_eq!(shards[2].len(), 3);
    }

    #[test]
    fn test_worker_result_total_loss() {
        let dp = DataParallel::new(DataParallelConfig::default());
        let results = vec![
            WorkerResult {
                worker_id: 0,
                gradients: vec![],
                loss: 0.5,
                num_samples: 8,
            },
            WorkerResult {
                worker_id: 1,
                gradients: vec![],
                loss: 0.7,
                num_samples: 8,
            },
        ];
        let total = dp.total_loss(&results);
        assert!((total - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulate_backward_integration() {
        let w = Tensor::randn(&[4, 2], true);
        let mut acc = GradientAccumulator::new(3);
        let params = vec![w.clone()];

        for _batch in 0..3 {
            let x = Tensor::randn(&[2, 4], false);
            let y = x.matmul(&w).sum();
            y.backward();
            acc.accumulate(&params);
            if acc.ready() {
                acc.finalize(&params);
                assert!(w.grad().is_some());
                acc.reset();
            }
        }

        assert!(w.grad().is_some());
    }
}
