use nexora_inference::engine::{InferenceConfig, InferenceEngine};
use nexora_inference::sampler::{Sampler, SamplingConfig, SamplingMethod};
use tokio::runtime::Runtime;

#[test]
fn test_inference_config_defaults() {
    let config = InferenceConfig::default();
    assert_eq!(config.max_concurrent_requests, 32);
    assert_eq!(config.queue_size_limit, 1000);
    assert!(config.enable_caching);
    assert!(config.enable_streaming);
    assert_eq!(config.default_timeout_seconds, 30);

    let custom = InferenceConfig {
        max_concurrent_requests: 5,
        queue_size_limit: 50,
        enable_caching: false,
        ..Default::default()
    };
    assert_eq!(custom.max_concurrent_requests, 5);
    assert_eq!(custom.queue_size_limit, 50);
    assert!(!custom.enable_caching);
}

#[test]
fn test_engine_rejects_uninitialized() {
    let rt = Runtime::new().unwrap();
    let engine = InferenceEngine::new(InferenceConfig::default());
    let request = nexora_inference::InferenceRequest::new("hello".to_string());
    let result = rt.block_on(engine.submit_request(request));
    assert!(
        result.is_err(),
        "Uninitialized engine should reject requests"
    );
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("not initialized") || msg.contains("not ready"),
        "Error should mention initialization, got: {}",
        msg
    );
}

#[test]
fn test_sampler_returns_valid_token_indices() {
    let mut greedy = Sampler::new(SamplingConfig {
        method: SamplingMethod::Greedy,
        temperature: 1.0,
        top_k: 10,
        top_p: 1.0,
        min_prob: 0.0,
        seed: Some(42),
    });

    let logits = vec![0.1, 0.5, 2.0, 0.3, 0.05];
    let token = greedy.sample(&logits).unwrap();
    assert_eq!(token, 2, "Greedy should pick highest logit (index 2)");
    assert!(
        token < logits.len(),
        "Token index should be within vocabulary bounds"
    );

    let mut temp = Sampler::new(SamplingConfig {
        method: SamplingMethod::Temperature,
        temperature: 0.5,
        top_k: 10,
        top_p: 1.0,
        min_prob: 0.0,
        seed: Some(42),
    });
    let token = temp.sample(&logits).unwrap();
    assert!(
        token < logits.len(),
        "Temperature sample should be valid index"
    );

    let mut full = Sampler::new(SamplingConfig {
        method: SamplingMethod::TemperatureTopKTopP,
        temperature: 0.8,
        top_k: 5,
        top_p: 0.9,
        min_prob: 0.0,
        seed: Some(42),
    });
    let token = full.sample(&logits).unwrap();
    assert!(
        token < logits.len(),
        "Full sampler should return valid index"
    );

    let uniform = vec![0.0, 0.0, 0.0, 0.0];
    let token = greedy.sample(&uniform).unwrap();
    assert!(
        token < uniform.len(),
        "Sample from uniform logits should be valid"
    );

    let single = vec![42.0];
    let token = greedy.sample(&single).unwrap();
    assert_eq!(token, 0, "Single-element logits should return index 0");
}
