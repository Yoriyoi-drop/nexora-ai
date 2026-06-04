use std::time::Instant;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use ndarray::Array1;
use nexora_inference::batching::{
    ContinuousBatchingConfig, ContinuousBatchingEngine, SchedulingPolicy,
};
use nexora_inference::inference_trait::ModelForward;
use nexora_inference::InferenceRequest;
use nexora_transformer::{CpuKVCache, KVCacheProvider};

// ─── Mock Model ─────────────────────────────────────────────────────────────────

struct MockModel {
    vocab_size: usize,
    forward_cost_ns: u64,
}

impl ModelForward for MockModel {
    fn forward(&self, input_ids: &[u32], _kv_cache: &mut dyn KVCacheProvider) -> Array1<f32> {
        if self.forward_cost_ns > 0 {
            let start = Instant::now();
            while start.elapsed().as_nanos() < self.forward_cost_ns as u128 {}
        }
        let mut logits = Array1::zeros(self.vocab_size);
        if let Some(&tid) = input_ids.last() {
            let idx = (tid as usize).min(self.vocab_size - 1);
            logits[idx] = 10.0;
        }
        logits
    }

    fn reset_cache(&self) -> CpuKVCache {
        CpuKVCache::new(0)
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────────

fn make_request(input_tokens: Vec<u32>, max_tokens: u32) -> InferenceRequest {
    InferenceRequest {
        input_tokens,
        max_tokens,
        ..Default::default()
    }
}

/// Run the engine until all sequences complete or max_steps is reached.
/// Returns (steps, total_completed, total_time).
fn run_to_completion(
    engine: &mut ContinuousBatchingEngine<MockModel>,
    max_steps: usize,
) -> (usize, usize, std::time::Duration) {
    let start = Instant::now();
    let mut steps = 0;
    let mut total_completed = 0;
    while engine.active_count() > 0 && steps < max_steps {
        let r = engine.step();
        steps += 1;
        total_completed += r.completed.len();
    }
    (steps, total_completed, start.elapsed())
}

fn run_bench(
    c: &mut Criterion,
    name: &str,
    config: ContinuousBatchingConfig,
    requests: Vec<InferenceRequest>,
    forward_cost_ns: u64,
    max_steps: usize,
) {
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(requests.len() as u64));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(15));

    group.bench_function("run", |b| {
        b.iter_batched(
            || {
                let model = MockModel {
                    vocab_size: 32000,
                    forward_cost_ns,
                };
                let mut engine = ContinuousBatchingEngine::with_config(model, config.clone());
                for req in &requests {
                    engine.add_request(req.clone());
                }
                engine
            },
            |mut engine| {
                run_to_completion(&mut engine, max_steps);
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

// ─── Benchmark Definitions ──────────────────────────────────────────────────────

fn bench_small_batch(c: &mut Criterion) {
    let config = ContinuousBatchingConfig {
        max_batch_size: 8,
        max_total_sequences: 256,
        enable_dynamic_padding: false,
        enable_adaptive_batching: false,
        use_paged_cache: true,
        ..Default::default()
    };
    let requests: Vec<InferenceRequest> = (0..16)
        .map(|i| make_request(vec![i as u32, i as u32 + 1], 20))
        .collect();
    run_bench(c, "batch_16_seq_20tok_8slot", config, requests, 0, 500);
}

fn bench_medium_batch(c: &mut Criterion) {
    let config = ContinuousBatchingConfig {
        max_batch_size: 32,
        max_total_sequences: 1024,
        enable_dynamic_padding: true,
        enable_adaptive_batching: true,
        use_paged_cache: true,
        ..Default::default()
    };
    let requests: Vec<InferenceRequest> = (0..64)
        .map(|i| make_request(vec![i as u32, i as u32 + 1], 40))
        .collect();
    run_bench(c, "batch_64_seq_40tok_32slot", config, requests, 0, 1000);
}

fn bench_large_batch(c: &mut Criterion) {
    let config = ContinuousBatchingConfig {
        max_batch_size: 64,
        max_total_sequences: 4096,
        enable_dynamic_padding: true,
        enable_adaptive_batching: true,
        use_paged_cache: true,
        ..Default::default()
    };
    let requests: Vec<InferenceRequest> = (0..256)
        .map(|i| make_request(vec![i as u32, (i + 1) as u32], 64))
        .collect();
    run_bench(
        c,
        "batch_256_seq_64tok_64slot",
        config,
        requests,
        0,
        2000,
    );
}

fn bench_spike_load(c: &mut Criterion) {
    let config = ContinuousBatchingConfig {
        max_batch_size: 32,
        max_total_sequences: 4096,
        enable_dynamic_padding: true,
        enable_adaptive_batching: true,
        use_paged_cache: true,
        ..Default::default()
    };
    let requests: Vec<InferenceRequest> = (0..200)
        .map(|i| make_request(vec![i as u32, (i + 1) as u32], 10))
        .collect();
    run_bench(
        c,
        "spike_200_seq_10tok",
        config,
        requests,
        0,
        5000,
    );
}

fn bench_mixed_workload(c: &mut Criterion) {
    let config = ContinuousBatchingConfig {
        max_batch_size: 32,
        max_total_sequences: 4096,
        enable_dynamic_padding: true,
        enable_adaptive_batching: true,
        use_paged_cache: true,
        ..Default::default()
    };
    let mut requests = Vec::with_capacity(100);
    for i in 0..80 {
        requests.push(make_request(vec![i as u32, i as u32 + 1], 4));
    }
    for i in 0..20 {
        requests.push(make_request(
            vec![(80 + i) as u32, (80 + i + 1) as u32],
            50,
        ));
    }
    run_bench(
        c,
        "mixed_80short_20long",
        config,
        requests,
        0,
        2000,
    );
}

fn bench_starvation_avoidance(c: &mut Criterion) {
    let config = ContinuousBatchingConfig {
        max_batch_size: 16,
        max_total_sequences: 1024,
        scheduling_policy: SchedulingPolicy::PriorityAging,
        aging_boost_per_ms: 0.01,
        enable_dynamic_padding: true,
        enable_adaptive_batching: true,
        use_paged_cache: true,
        ..Default::default()
    };
    let mut requests = Vec::with_capacity(50);
    for i in 0..40 {
        requests.push(make_request(vec![i as u32, i as u32 + 1], 4));
    }
    for i in 0..10 {
        requests.push(make_request(
            vec![(40 + i) as u32, (40 + i + 1) as u32],
            80,
        ));
    }
    run_bench(
        c,
        "starvation_40short_10long_priorityaging",
        config,
        requests,
        0,
        2000,
    );
}

fn bench_paged_vs_flat_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_vs_flat");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(15));

    let requests: Vec<InferenceRequest> = (0..32)
        .map(|i| make_request(vec![i as u32, i as u32 + 1], 30))
        .collect();

    // Flat cache (old default)
    group.bench_function("flat_cache", |b| {
        b.iter_batched(
            || {
                let model = MockModel {
                    vocab_size: 32000,
                    forward_cost_ns: 0,
                };
                let cfg = ContinuousBatchingConfig {
                    max_batch_size: 16,
                    max_total_sequences: 256,
                    use_paged_cache: false,
                    enable_adaptive_batching: false,
                    ..Default::default()
                };
                let mut engine = ContinuousBatchingEngine::with_config(model, cfg);
                for req in &requests {
                    engine.add_request(req.clone());
                }
                engine
            },
            |mut engine| {
                run_to_completion(&mut engine, 500);
            },
            criterion::BatchSize::LargeInput,
        );
    });

    // Paged cache (new default)
    group.bench_function("paged_cache", |b| {
        b.iter_batched(
            || {
                let model = MockModel {
                    vocab_size: 32000,
                    forward_cost_ns: 0,
                };
                let cfg = ContinuousBatchingConfig {
                    max_batch_size: 16,
                    max_total_sequences: 256,
                    use_paged_cache: true,
                    enable_adaptive_batching: false,
                    ..Default::default()
                };
                let mut engine = ContinuousBatchingEngine::with_config(model, cfg);
                for req in &requests {
                    engine.add_request(req.clone());
                }
                engine
            },
            |mut engine| {
                run_to_completion(&mut engine, 500);
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_small_batch,
    bench_medium_batch,
    bench_large_batch,
    bench_spike_load,
    bench_mixed_workload,
    bench_starvation_avoidance,
    bench_paged_vs_flat_cache,
);
criterion_main!(benches);
