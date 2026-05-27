//! Golden Path E2E tests: Tokenizer → Transformer → KV Cache → Sampler → (Streaming) Response.
//!
//! These tests exercise the full inference pipeline end-to-end using a tiny model
//! with random weights. They verify that all components wire together correctly
//! and produce valid output shapes/types — NOT that the output text is meaningful
//! (that requires a trained model).

use std::sync::Arc;
use std::time::Duration;

use nexora_inference::engine::{InferenceConfig, InferenceEngine};
use nexora_inference::sampler::{Sampler, SamplingConfig, SamplingMethod};
use nexora_inference::{GeneratedToken, InferenceRequest};
use nexora_tokenizer::{BpeConfig, BpeTokenizer};
use nexora_transformer::{CausalLM, TransformerConfig};
use tokio::sync::Mutex;

// ── Helpers ─────────────────────────────────────────────────────────────────────

/// Tiny model with KV cache enabled, suitable for inference testing.
fn test_model() -> CausalLM {
    let mut model = CausalLM::new(TransformerConfig {
        vocab_size: 64,
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        max_seq_len: 64,
        intermediate_size: 64,
        rope_theta: 10000.0,
        use_cache: true,
        norm_eps: 1e-6,
        num_experts: 0,
        top_k_experts: 0,
        expert_intermediate_size: 0,
        use_half_precision: true,
    });
    model.set_keep_on_gpu(false);
    model
}

/// Train a tiny BPE tokenizer from a small corpus.
fn test_tokenizer() -> BpeTokenizer {
    let mut tok = BpeTokenizer::new(BpeConfig {
        vocab_size: 64,
        min_frequency: 1,
        ..Default::default()
    });
    tok.train("hello world test inference data model tokenizer golden path e2e")
        .unwrap();
    tok
}

/// Inference config with GPU disabled and small limits for fast test execution.
fn test_config() -> InferenceConfig {
    InferenceConfig {
        use_gpu: false,
        enable_caching: false,
        max_concurrent_requests: 4,
        queue_size_limit: 10,
        enable_streaming: true,
        generation_timeout_seconds: 10,
        ..Default::default()
    }
}

// ── Golden Path 1: Non-streaming via InferenceEngine ────────────────────────────

#[tokio::test]
async fn test_golden_path_non_streaming() {
    let model = Arc::new(test_model());
    let tokenizer = Arc::new(Mutex::new(test_tokenizer()));
    let config = test_config();

    let mut engine = InferenceEngine::with_model(model, Some(tokenizer), config);
    engine.initialize().await.unwrap();

    let request = InferenceRequest::new("hello".to_string()).with_max_tokens(10);

    let mut rx = engine.submit_request(request).await.unwrap();
    let response = tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("timeout waiting for inference response")
        .expect("response channel closed without message");

    assert!(
        response.total_tokens <= 10,
        "response should have at most 10 tokens"
    );
    assert!(
        response.total_tokens > 0,
        "response should have at least 1 token"
    );
    assert!(
        response.finish_reason == nexora_inference::FinishReason::MaxTokens,
        "should finish cleanly"
    );
    assert!(
        response
            .tokens
            .iter()
            .all(|t| (t.token_id as usize) < test_model().config.vocab_size),
        "all token IDs should be within vocab_size"
    );

    engine.shutdown().await.unwrap();
}

// ── Golden Path 2: Streaming via InferenceEngine ────────────────────────────────

#[tokio::test]
async fn test_golden_path_streaming() {
    let model = Arc::new(test_model());
    let tokenizer = Arc::new(Mutex::new(test_tokenizer()));
    let config = test_config();

    let mut engine = InferenceEngine::with_model(model, Some(tokenizer), config);
    engine.initialize().await.unwrap();

    let request = InferenceRequest::new("hello".to_string())
        .with_max_tokens(5)
        .with_streaming(true);

    let mut rx = engine.submit_streaming_request(request).await.unwrap();
    let mut tokens: Vec<Arc<GeneratedToken>> = Vec::new();

    while let Ok(Some(token)) = tokio::time::timeout(Duration::from_secs(10), rx.recv()).await {
        tokens.push(token);
    }

    assert!(
        !tokens.is_empty(),
        "streaming should produce at least 1 token"
    );
    assert!(
        tokens.len() <= 5,
        "streaming should produce at most 5 tokens"
    );

    let _all_text: String = tokens.iter().map(|t| t.token_text.as_ref()).collect();
    let vocab_size = test_model().config.vocab_size;
    assert!(
        tokens.iter().all(|t| (t.token_id as usize) < vocab_size),
        "all streaming token IDs should be within vocab_size"
    );

    engine.shutdown().await.unwrap();
}

// ── Golden Path 3: Manual forward + sample loop (no InferenceEngine) ────────────
// This tests the low-level component wiring: model.forward → sampler.sample.
// Useful as a fast smoke test that doesn't depend on the engine's async scheduler.

#[tokio::test]
async fn test_forward_then_sample_manual() {
    let model = test_model();
    let mut cache = model.reset_cache();
    let mut sampler = Sampler::new(SamplingConfig {
        method: SamplingMethod::Greedy,
        ..Default::default()
    });

    let mut token: u32 = 1; // BOS
    let mut generated = Vec::new();

    for _ in 0..10 {
        let logits = model.forward(&[token], &mut cache).unwrap();
        let idx = sampler.sample(logits.as_slice().unwrap()).unwrap() as u32;
        generated.push(idx);
        token = idx;
        if idx == 0 {
            break;
        }
    }

    assert!(
        !generated.is_empty(),
        "manual loop should generate at least 1 token"
    );
    assert!(
        generated.len() <= 10,
        "manual loop should generate at most 10 tokens"
    );
    assert!(
        generated.iter().all(|&t| (t as usize) < 64),
        "all generated token IDs should be within vocab_size (64)"
    );
}
