use std::collections::HashMap;

use ndarray::ArrayD;

use super::Tensor;

// ─── Gradient Accumulation ────────────────────────────────────────────────────

/// Accumulates gradients across multiple micro-batches before each optimizer step.
///
/// Usage:
/// ```ignore
/// let mut acc = GradientAccumulator::new(4);  // 4 micro-batches
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
    /// Number of micro-batches to accumulate
    num_micro_batches: usize,
    /// Current micro-batch counter
    counter: usize,
    /// Stashed accumulated gradients: parameter_id → gradient
    stash: HashMap<usize, ArrayD<f32>>,
}

impl GradientAccumulator {
    /// Create new accumulator that accumulates across `num_micro_batches` steps
    pub fn new(num_micro_batches: usize) -> Self {
        assert!(num_micro_batches >= 1, "num_micro_batches must be >= 1");
        Self {
            num_micro_batches,
            counter: 0,
            stash: HashMap::new(),
        }
    }

    /// Accumulate gradients from current micro-batch.
    /// Call after `loss.backward()` for each micro-batch.
    pub fn accumulate(&mut self, params: &[Tensor]) {
        self.counter += 1;
        let scale = 1.0 / self.num_micro_batches as f32;

        for (i, p) in params.iter().enumerate() {
            if let Some(g) = p.grad() {
                let scaled = &g * scale;
                self.stash
                    .entry(i)
                    .and_modify(|existing| {
                        *existing = &*existing + &scaled;
                    })
                    .or_insert(scaled);
            }
        }
    }

    /// Check if ready for optimizer step
    pub fn ready(&self) -> bool {
        self.counter >= self.num_micro_batches
    }

    /// Finalize: set averaged gradients on model parameters.
    /// Call before `optimizer.step()`.
    pub fn finalize(&self, params: &[Tensor]) {
        for (i, p) in params.iter().enumerate() {
            if let Some(g) = self.stash.get(&i) {
                p.set_grad(g.clone());
            }
        }
    }

    /// Reset accumulator for next cycle
    pub fn reset(&mut self) {
        self.counter = 0;
        self.stash.clear();
    }

    /// Progress (0.0 – 1.0)
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

// ─── Data-Parallel Config ─────────────────────────────────────────────────────

/// Configuration for data-parallel training
#[derive(Clone, Debug)]
pub struct DataParallelConfig {
    /// Number of parallel workers (threads). Default: 1 (no parallelism)
    pub num_workers: usize,
    /// Batch size per worker per step
    pub batch_size_per_worker: usize,
    /// Whether to use gradient accumulation across micro-batches
    pub gradient_accumulation: bool,
    /// Number of micro-batches to accumulate (if gradient_accumulation is true)
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
    /// Effective batch size across all workers
    pub fn effective_batch_size(&self) -> usize {
        let per_step = self.num_workers * self.batch_size_per_worker;
        if self.gradient_accumulation {
            per_step * self.accumulation_steps
        } else {
            per_step
        }
    }

    /// Optimize config based on available CPU cores
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
    /// Per-parameter gradients
    pub gradients: Vec<ArrayD<f32>>,
    /// Average loss for this worker's shard
    pub loss: f32,
    /// Number of samples processed
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

    /// Shard data indices across workers
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

    /// Reduce (average) gradients from all workers into the first worker's gradients
    pub fn reduce_gradients(&self, results: &mut [WorkerResult]) {
        let grads: Vec<&mut Vec<ArrayD<f32>>> = results.iter_mut().map(|r| &mut r.gradients).collect();
        let mut grad_refs: Vec<Vec<ArrayD<f32>>> = Vec::new();
        for g in grads {
            grad_refs.push(std::mem::take(g));
        }
        all_reduce_gradients(&mut grad_refs);
        for (i, g) in grad_refs.into_iter().enumerate() {
            results[i].gradients = g;
        }
    }

    /// Compute total loss across all workers (mean)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Tensor;
    use crate::autograd::TensorOps;

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
        // Simulate accumulation without real gradients
        let p = Tensor::randn(&[4], true);
        for _ in 0..3 {
            // Just increment counter by calling accumulate with empty grads
            // (params have no grads yet since we didn't backward)
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
        let mut grads = vec![vec![ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()]];
        all_reduce_gradients(&mut grads);
        // Single worker: no change
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
        // Average: (2+4)/2=3, (4+6)/2=5
        assert!((grads[0][0][0] - 3.0).abs() < 1e-6);
        assert!((grads[0][0][1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_all_reduce_empty() {
        let mut grads: Vec<Vec<ArrayD<f32>>> = vec![];
        all_reduce_gradients(&mut grads);
        // Should not panic
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
        // First shards get the remainder
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
        // End-to-end: multiple micro-batches with gradient accumulation
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
