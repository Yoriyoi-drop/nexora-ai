use std::time::Duration;
use tokio::sync::mpsc;
use uuid::Uuid;

use nexora_inference::batching::{BatchCollector, BatchKey};
use nexora_inference::kv_cache::KVCache;
use nexora_inference::scheduler::RequestScheduler;
use nexora_inference::sampler::{Sampler, SamplingConfig, SamplingMethod};
use nexora_inference::streaming::StreamingEngine;
use nexora_inference::{GeneratedToken, InferenceRequest, InferenceResponse};

// ============================================================
// KV Cache + Sampler Integration
// ============================================================

#[tokio::test]
async fn test_kv_cache_sampler_pipeline() {
    let cache = KVCache::new();
    cache.initialize().await.unwrap();

    let logits: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    cache.insert(b"test_logits_1".to_vec(), logits.clone()).await;

    let cached = cache.get(b"test_logits_1").await;
    assert!(cached.is_some());
    assert_eq!(cached.as_ref().unwrap(), &logits);

    let sampler = Sampler::new(SamplingConfig {
        method: SamplingMethod::Greedy,
        temperature: 1.0,
        top_k: 10,
        top_p: 1.0,
        seed: Some(42),
    });

    let sample = sampler.sample(cached.as_ref().unwrap()).unwrap();
    assert_eq!(sample, 9);

    let stats = cache.get_stats();
    assert!(stats.hits >= 1);
}

#[tokio::test]
async fn test_kv_cache_with_repeated_access() {
    let cache = KVCache::new();
    cache.initialize().await.unwrap();

    let logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    cache.insert(b"repeated_key".to_vec(), logits.clone()).await;

    for _ in 0..5 {
        let result = cache.get(b"repeated_key").await;
        assert!(result.is_some());
        assert_eq!(result.unwrap(), logits);
    }

    cache.clear().await;
    assert!(cache.get(b"repeated_key").await.is_none());
}

// ============================================================
// Batching + Scheduler Integration
// ============================================================

#[tokio::test]
async fn test_scheduler_batching_grouping() {
    let scheduler = RequestScheduler::new().with_max_batch_size(4);

    let req1 = InferenceRequest::new("hello".to_string())
        .with_model("model-a".to_string())
        .with_temperature(0.7)
        .with_top_k(40)
        .with_top_p(0.9);

    let req3 = InferenceRequest::new("different".to_string())
        .with_model("model-b".to_string())
        .with_temperature(0.7)
        .with_top_k(40)
        .with_top_p(0.9);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();
    let (tx3, _rx3) = mpsc::unbounded_channel();

    scheduler.submit_request(req1.request_id, tx1).await.unwrap();
    let req2 = InferenceRequest::new("world".to_string())
        .with_model("model-a".to_string())
        .with_temperature(0.7)
        .with_top_k(40)
        .with_top_p(0.9);
    scheduler.submit_request(req2.request_id, tx2).await.unwrap();
    scheduler.submit_request(req3.request_id, tx3).await.unwrap();

    // Add to batch collector
    for r in &[&req1, &req2, &req3] {
        scheduler.add_to_batch_collector(r).await;
    }

    let batch = scheduler.pop_batch().await;
    assert!(batch.is_some());
    let batch = batch.unwrap();
    assert!(batch.size() >= 1);
    assert!(batch.size() <= 3);

    scheduler.complete_batch(&batch).await.unwrap();
}

#[tokio::test]
async fn test_scheduler_stats_with_batching() {
    let scheduler = RequestScheduler::new().with_max_batch_size(8);

    let req = InferenceRequest::new("test".to_string());
    let (tx, _rx) = mpsc::unbounded_channel();
    scheduler.submit_request(req.request_id, tx).await.unwrap();
    scheduler.add_to_batch_collector(&req).await;

    let stats = scheduler.get_stats().await;
    assert!(stats.total_requests >= 1);
    assert!(stats.pending_batch_requests >= 1);
}

// ============================================================
// Streaming Engine Integration
// ============================================================

#[tokio::test]
async fn test_streaming_e2e() {
    let engine = StreamingEngine::new();
    engine.initialize().await.unwrap();

    let (stream_id, mut rx) = engine.create_stream().await.unwrap();

    let token1 = GeneratedToken::new(1, "hello".to_string(), -0.5, 0);
    let token2 = GeneratedToken::new(2, " world".to_string(), -0.3, 1);

    engine
        .push_tokens(stream_id, vec![token1.clone()], false)
        .await
        .unwrap();
    engine
        .push_tokens(stream_id, vec![token2.clone()], true)
        .await
        .unwrap();

    let received: Vec<GeneratedToken> = {
        let mut tokens = Vec::new();
        while let Some(token) = rx.recv().await {
            tokens.push(token);
        }
        tokens
    };

    assert_eq!(received.len(), 2);
    assert_eq!(received[0].token_id, 1);
    assert_eq!(received[1].token_text, " world");

    let streams = engine.active_stream_count().await;
    assert_eq!(streams, 0);

    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_streaming_multiple_streams() {
    let engine = StreamingEngine::new();
    engine.initialize().await.unwrap();

    let (sid1, mut rx1) = engine.create_stream().await.unwrap();
    let (sid2, mut rx2) = engine.create_stream().await.unwrap();

    let t1 = GeneratedToken::new(10, "alpha".to_string(), -0.1, 0);
    let t2 = GeneratedToken::new(20, "beta".to_string(), -0.2, 0);

    engine.push_tokens(sid1, vec![t1], false).await.unwrap();
    engine.push_tokens(sid2, vec![t2], false).await.unwrap();

    engine.close_stream(sid1).await;
    engine.close_stream(sid2).await;

    let r1 = rx1.recv().await;
    let r2 = rx2.recv().await;
    assert!(r1.is_some());
    assert!(r2.is_some());
    assert_eq!(r1.unwrap().token_id, 10);
    assert_eq!(r2.unwrap().token_id, 20);

    engine.shutdown().await.unwrap();
}

// ============================================================
// BatchCollector Direct Integration
// ============================================================

#[tokio::test]
async fn test_batch_collector_timed_flush() {
    let mut collector = BatchCollector::new(10, 50);

    let req = InferenceRequest::new("timeout-test".to_string())
        .with_model("model-x".to_string());
    let (tx, _rx) = mpsc::unbounded_channel();
    collector.add_request(req, tx);

    assert!(collector.drain_ready().is_empty());

    tokio::time::sleep(Duration::from_millis(60)).await;
    let batches = collector.drain_ready();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].size(), 1);
}

#[tokio::test]
async fn test_batch_collector_max_size_flush() {
    let mut collector = BatchCollector::new(3, 5000);

    for i in 0..3 {
        let req = InferenceRequest::new(format!("req-{}", i))
            .with_model("batch-model".to_string());
        let (tx, _rx) = mpsc::unbounded_channel();
        collector.add_request(req, tx);
    }

    let batches = collector.drain_ready();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].size(), 3);
}

// ============================================================
// Sampler Integration
// ============================================================

#[tokio::test]
async fn test_sampler_steers_tokens() {
    let logits: Vec<f32> = vec![0.0, 0.0, 10.0, 0.0, 0.0];

    let greedy = Sampler::new(SamplingConfig {
        method: SamplingMethod::Greedy,
        temperature: 1.0,
        top_k: 10,
        top_p: 1.0,
        seed: None,
    });
    assert_eq!(greedy.sample(&logits).unwrap(), 2);

    let temp = Sampler::new(SamplingConfig {
        method: SamplingMethod::Temperature,
        temperature: 0.1,
        top_k: 10,
        top_p: 1.0,
        seed: Some(42),
    });
    assert_eq!(temp.sample(&logits).unwrap(), 2);

    let topk = Sampler::new(SamplingConfig {
        method: SamplingMethod::TopK,
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        seed: Some(42),
    });
    assert_eq!(topk.sample(&logits).unwrap(), 2);
}

// ============================================================
// Cross-component: Cache + Stream lifetime
// ============================================================

#[tokio::test]
async fn test_cache_stream_independence() {
    let cache = KVCache::new();
    let stream_engine = StreamingEngine::new();

    cache.initialize().await.unwrap();
    stream_engine.initialize().await.unwrap();

    cache.insert(b"data".to_vec(), vec![1.0, 2.0, 3.0]).await;
    assert!(cache.get(b"data").await.is_some());

    let (sid, _rx) = stream_engine.create_stream().await.unwrap();
    assert!(stream_engine.active_stream_count().await > 0);

    assert!(cache.get(b"data").await.is_some());
    stream_engine.close_stream(sid).await;

    stream_engine.shutdown().await.unwrap();
}

// ============================================================
// KV cache lifecycle integration
// ============================================================

#[tokio::test]
async fn test_cache_ttl_eviction_lifecycle() {
    let cache = KVCache::new().with_ttl(Duration::from_millis(50));
    cache.initialize().await.unwrap();

    cache
        .insert(b"ephemeral".to_vec(), vec![42.0])
        .await;
    assert!(cache.get(b"ephemeral").await.is_some());

    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(cache.get(b"ephemeral").await.is_none());
}

// ============================================================
// Error handling integration: sampler edge cases
// ============================================================

#[tokio::test]
async fn test_sampler_edge_cases() {
    let sampler = Sampler::new(SamplingConfig {
        method: SamplingMethod::Greedy,
        temperature: 1.0,
        top_k: 10,
        top_p: 1.0,
        seed: None,
    });

    // NaN in logits should not panic
    let nan_logits = vec![f32::NAN, 1.0, 2.0];
    let result = sampler.sample(&nan_logits);
    assert!(result.is_ok());

    // All zeros
    let zeros = vec![0.0, 0.0, 0.0];
    let result = sampler.sample(&zeros);
    assert!(result.is_ok());

    // Single element
    let single = vec![5.0];
    let result = sampler.sample(&single);
    assert_eq!(result.unwrap(), 0);
}
