use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use nexora_runtime::*;

// ─── Helpers ────────────────────────────────────────────────────────────

fn make_request(model_id: &str, priority: u8) -> InferenceRequest {
    InferenceRequest {
        model_id: model_id.to_string(),
        inputs: vec![],
        parameters: [("max_tokens".to_string(), serde_json::json!(100))]
            .into_iter().collect(),
        request_id: Some(uuid::Uuid::new_v4().to_string()),
        input_tokens: vec![],
        target_tokens: None,
        priority,
        metadata: [].into_iter().collect(),
    }
}

// ─── KV Cache Stress ────────────────────────────────────────────────────

#[tokio::test]
async fn kv_cache_concurrent_put_get() {
    let cache = Arc::new({
        let c = KVCache::new(1024 * 1024 * 100);
        c.initialize().await.unwrap();
        c
    });

    let counter = Arc::new(AtomicUsize::new(0));
    let n = 100;
    let mut handles = Vec::with_capacity(n);

    for i in 0..n {
        let c = Arc::clone(&cache);
        let cnt = Arc::clone(&counter);
        handles.push(tokio::spawn(async move {
            let key = format!("key-{}", i);
            let entry = CacheEntry::new(
                vec![i as f32; 64],
                vec![i as f32; 64],
                1, 12, 64, 12,
            );
            c.put(key.clone(), entry).await.unwrap();
            let got = c.get(&key).await.unwrap();
            assert!(got.is_some(), "key-{} should exist", i);
            cnt.fetch_add(1, Ordering::SeqCst);
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    assert_eq!(counter.load(Ordering::SeqCst), n);
    Arc::try_unwrap(cache).unwrap_or_else(|_| panic!("refs still held")).shutdown().await.unwrap();
}

#[tokio::test]
async fn kv_cache_concurrent_eviction() {
    let cache = Arc::new({
        let c = KVCache::with_config(CacheConfig {
            max_size_bytes: 1024 * 10,
            eviction_policy: EvictionPolicy::LRU,
            shard_count: 4,
            ..Default::default()
        });
        c.initialize().await.unwrap();
        c
    });

    let n = 200;
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let c = Arc::clone(&cache);
        let key = format!("evict-{}", i);
        let entry = CacheEntry::new(
            vec![1.0; 256], vec![1.0; 256],
            1, 12, 64, 12,
        );
        handles.push(tokio::spawn(async move {
            c.put(key, entry).await.unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let stats = cache.get_stats().await;
    assert!(stats.evictions > 0 || stats.current_size_bytes <= stats.max_size_bytes);
    Arc::try_unwrap(cache).unwrap_or_else(|_| panic!("refs still held")).shutdown().await.unwrap();
}

#[tokio::test]
async fn kv_cache_ttl_expiry() {
    let cache = KVCache::with_config(CacheConfig {
        max_size_bytes: 1024 * 1024,
        ttl_seconds: Some(1),
        ..Default::default()
    });
    cache.initialize().await.unwrap();

    cache.put("ttl-key".to_string(), CacheEntry::new(
        vec![1.0; 64], vec![1.0; 64], 1, 12, 64, 12,
    )).await.unwrap();

    assert!(cache.get("ttl-key").await.unwrap().is_some());

    tokio::time::sleep(Duration::from_secs(2)).await;
    let cleaned = cache.clean_expired().await.unwrap();
    assert!(cleaned > 0, "expected expired entries cleaned");
    assert!(cache.get("ttl-key").await.unwrap().is_none());
}

#[tokio::test]
async fn kv_cache_1000_concurrent_access() {
    let cache = Arc::new({
        let c = KVCache::new(1024 * 1024 * 200);
        c.initialize().await.unwrap();
        c
    });

    let n = 1000;
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let c = Arc::clone(&cache);
        handles.push(tokio::spawn(async move {
            let key = format!("heavy-{}", i % 100);
            if i % 3 == 0 {
                c.put(key, CacheEntry::new(
                    vec![i as f32; 64], vec![i as f32; 64],
                    1, 12, 64, 12,
                )).await.ok();
            } else {
                let _ = c.get(&key).await;
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    Arc::try_unwrap(cache).unwrap_or_else(|_| panic!("refs still held")).shutdown().await.unwrap();
}

// ─── Scheduler Stress ───────────────────────────────────────────────────

#[tokio::test]
async fn scheduler_concurrent_submit() {
    let scheduler = Arc::new({
        let s = RequestScheduler::new(50);
        s.initialize().await.unwrap();
        s
    });

    let n = 200;
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let s = Arc::clone(&scheduler);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let req = make_request("model-a", (i % 10) as u8);
        handles.push(tokio::spawn(async move {
            s.submit_request(req, tx).await.unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let stats = scheduler.get_stats().await;
    assert_eq!(stats.total_requests, n as u64);
    Arc::try_unwrap(scheduler).unwrap_or_else(|_| panic!("refs still held")).shutdown().await.unwrap();
}

#[tokio::test]
async fn scheduler_fifo_ordering() {
    let scheduler = RequestScheduler::new(1);
    scheduler.initialize().await.unwrap();

    let n = 5;
    let mut req_ids = Vec::with_capacity(n);
    for _ in 0..n {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let req = make_request("model-a", 50);
        let id = req.request_id.clone();
        req_ids.push(id);
        scheduler.submit_request(req, tx).await.unwrap();
    }

    // Complete the active one, then check next in FIFO order
    for expected in req_ids.iter() {
        if let Some(next) = scheduler.get_next_request().await.unwrap() {
            assert_eq!(next.request.request_id.as_deref(), expected.as_deref());
            // Start and complete it to allow next
            let uuid = next.request.request_id.as_ref().and_then(|s| uuid::Uuid::parse_str(s).ok()).unwrap();
            scheduler.start_request(uuid).await.unwrap();
            scheduler.complete_request(uuid).await.unwrap();
        } else {
            break;
        }
    }
}

#[tokio::test]
async fn scheduler_stats_track_requests() {
    let scheduler = RequestScheduler::new(100);
    scheduler.initialize().await.unwrap();

    let n = 50;
    for _ in 0..n {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        scheduler.submit_request(make_request("model-a", 50), tx).await.unwrap();
    }

    let stats = scheduler.get_stats().await;
    assert_eq!(stats.total_requests, n as u64);
    assert_eq!(stats.current_active_requests, n);
    assert_eq!(stats.current_queue_length, 0);

    scheduler.shutdown().await.unwrap();
}

#[tokio::test]
async fn scheduler_cancellation_stress() {
    let scheduler = Arc::new({
        let s = RequestScheduler::new(10);
        s.initialize().await.unwrap();
        s
    });

    let n = 100;
    let mut uuids = Vec::with_capacity(n);
    for _i in 0..n {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut req = make_request("model-a", 50);
        let uuid = uuid::Uuid::new_v4();
        req.request_id = Some(uuid.to_string());
        uuids.push(uuid);
        scheduler.submit_request(req, tx).await.unwrap();
    }

    let mut cancelled = 0;
    for uuid in &uuids[..50] {
        if scheduler.cancel_request(*uuid).await.unwrap() {
            cancelled += 1;
        }
    }
    assert!(cancelled > 0);
}

#[tokio::test]
async fn scheduler_shutdown_stress() {
    let scheduler = Arc::new({
        let s = RequestScheduler::new(20);
        s.initialize().await.unwrap();
        s
    });

    let n = 50;
    for _ in 0..n {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        scheduler.submit_request(make_request("model-a", 50), tx).await.unwrap();
    }

    scheduler.shutdown().await.unwrap();
    let stats = scheduler.get_stats().await;
    assert_eq!(stats.current_queue_length, 0);
}

// ─── Streaming Stress ───────────────────────────────────────────────────

#[tokio::test]
async fn streaming_concurrent_streams() {
    let engine = StreamingEngine::new();
    engine.initialize().await.unwrap();

    let n = 50;
    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let engine = engine.clone();
        handles.push(tokio::spawn(async move {
            let req = make_request("model-a", 50);
            let stream = engine.create_stream(&req).await.unwrap();
            let sid = stream.stream_id;

            for j in 0..5 {
                engine.send_token(sid, GeneratedToken {
                    token_id: j, text: format!("token-{}", j),
                    logprob: 0.0, is_special: false,
                }, j == 4).await.unwrap();
            }

            let status = engine.get_stream_status(sid).await.unwrap();
            assert!(status.is_some());
            drop(stream);
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let stats = engine.get_stats().await;
    assert_eq!(stats.total_streams, n as u64);
    engine.shutdown().await.unwrap();
}

// ─── Executor Stress ────────────────────────────────────────────────────

#[tokio::test]
async fn executor_concurrent_tasks() {
    let executor = Arc::new(TaskExecutor::new());
    let counter = Arc::new(AtomicUsize::new(0));

    let n = 100;
    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let exec = Arc::clone(&executor);
        let c = Arc::clone(&counter);
        handles.push(tokio::spawn(async move {
            let task = || {
                let c = Arc::clone(&c);
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            };
            exec.execute_with_retry(task, 2).await.unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    assert_eq!(counter.load(Ordering::SeqCst), n);
}

#[tokio::test]
async fn executor_retry_on_failure() {
    let executor = TaskExecutor::new();
    let attempt = Arc::new(AtomicUsize::new(0));

    let a = Arc::clone(&attempt);
    let result = executor.execute_with_retry(|| {
        let a = Arc::clone(&a);
        async move {
            let prev = a.fetch_add(1, Ordering::SeqCst);
            if prev < 2 {
                Err(anyhow::anyhow!("simulated failure {}", prev))
            } else {
                Ok(())
            }
        }
    }, 3).await;

    assert!(result.is_ok());
    assert_eq!(attempt.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn executor_exhaust_retries() {
    let executor = TaskExecutor::new();
    let result = executor.execute_with_retry(|| async {
        Err(anyhow::anyhow!("always fails"))
    }, 2).await;

    assert!(result.is_err());
}

// ─── Full System Stress ─────────────────────────────────────────────────

#[tokio::test]
async fn runtime_concurrent_task_storm() {
    let executor = Arc::new(TaskExecutor::new());
    let n = 500;
    let counter = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let exec = Arc::clone(&executor);
        let c = Arc::clone(&counter);
        handles.push(tokio::spawn(async move {
            exec.execute_with_retry(|| {
                let c = Arc::clone(&c);
                async move {
                    tokio::time::sleep(Duration::from_micros(10)).await;
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            }, 1).await.ok();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    assert_eq!(counter.load(Ordering::SeqCst), n);
}

#[tokio::test]
async fn runtime_timeout_storm() {
    let executor = Arc::new(TaskExecutor::new());
    let n = 100;

    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let exec = Arc::clone(&executor);
        handles.push(tokio::spawn(async move {
            let result = tokio::time::timeout(
                Duration::from_millis(1),
                exec.execute_with_retry(|| async {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    Ok(())
                }, 0)
            ).await;
            assert!(result.is_err() || result.unwrap().is_err());
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
}
