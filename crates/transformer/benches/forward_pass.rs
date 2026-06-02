use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nexora_transformer::{CausalLM, TransformerConfig};

fn bench_generate_short(c: &mut Criterion) {
    let config = TransformerConfig {
        vocab_size: 50257,
        hidden_size: 128,
        num_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        intermediate_size: 512,
        max_seq_len: 512,
        ..Default::default()
    };
    let model = CausalLM::new(config);
    let prompt: Vec<u32> = (0..64).map(|i| (i % 50257) as u32).collect();

    let mut group = c.benchmark_group("causal_lm_generate");
    group.throughput(Throughput::Elements(64));
    group.bench_function("generate_64_prompt_32_tokens", |b| {
        b.iter(|| {
            let (tokens, _cache) = model.generate(&prompt, 32, 0.7, 50);
            tokens
        })
    });
    group.finish();
}

fn bench_forward_single(c: &mut Criterion) {
    let config = TransformerConfig {
        vocab_size: 50257,
        hidden_size: 128,
        num_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        intermediate_size: 512,
        max_seq_len: 512,
        ..Default::default()
    };
    let model = CausalLM::new(config);
    let mut cache = model.reset_cache();

    let mut group = c.benchmark_group("causal_lm_forward");
    group.throughput(Throughput::Elements(1));
    group.bench_function("forward_single_token", |b| {
        b.iter(|| {
            let logits = model.forward(&[42], &mut cache);
            logits
        })
    });
    group.finish();
}

fn bench_generate_long(c: &mut Criterion) {
    let config = TransformerConfig {
        vocab_size: 50257,
        hidden_size: 256,
        num_heads: 8,
        num_kv_heads: 4,
        num_layers: 4,
        intermediate_size: 1024,
        max_seq_len: 1024,
        ..Default::default()
    };
    let model = CausalLM::new(config);
    let prompt: Vec<u32> = (0..256).map(|i| (i % 50257) as u32).collect();

    let mut group = c.benchmark_group("causal_lm_generate_long");
    group.throughput(Throughput::Elements(256));
    group.bench_function("generate_256_prompt_128_tokens", |b| {
        b.iter(|| {
            let (tokens, _cache) = model.generate(&prompt, 128, 0.7, 50);
            tokens
        })
    });
    group.finish();
}

criterion_group!(benches, bench_generate_short, bench_forward_single, bench_generate_long);
criterion_main!(benches);
