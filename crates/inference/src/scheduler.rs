use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{watch, Notify, RwLock};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::batching::{Batch, BatchCollector, BatchKey};
use async_trait::async_trait;

/// Deprecated: use `nexora_runtime::scheduler::RequestStatus` instead.
#[derive(Debug, Clone, PartialEq)]
pub enum RequestStatus {
    Queued,
    Processing,
    Completed,
    Failed(String),
    Cancelled,
    Timeout,
}

/// Deprecated: use `nexora_runtime::scheduler::SchedulerStats` instead.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub total_requests: usize,
    pub active_requests: usize,
    pub queued_requests: usize,
    pub completed_requests: usize,
    pub failed_requests: usize,
    pub pending_batches: usize,
    pub pending_batch_requests: usize,
    pub avg_queue_time_ms: f64,
    pub avg_processing_time_ms: f64,
    pub throughput_rps: f64,
}

struct ScheduledRequest {
    _request_id: Uuid,
    response_tx: tokio::sync::mpsc::Sender<crate::InferenceResponse>,
    status: RequestStatus,
    queued_at: Instant,
    processing_started_at: Option<Instant>,
    batch_key: Option<BatchKey>,
}

pub struct RequestScheduler {
    queue: RwLock<VecDeque<Uuid>>,
    requests: Arc<RwLock<HashMap<Uuid, ScheduledRequest>>>,
    max_concurrent: usize,
    max_batch_size: usize,
    max_queue_time_ms: u64,
    active_count: RwLock<usize>,
    batch_collector: Arc<RwLock<BatchCollector>>,
    shutdown: AtomicBool,
    notify: Arc<Notify>,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
    background_handles: tokio::sync::Mutex<Vec<JoinHandle<()>>>,
    total_queue_time_ms: RwLock<f64>,
    total_processing_time_ms: RwLock<f64>,
    scheduler_start: Instant,
    /// Delegate scheduler that handles queue ordering, priority, and strategies.
    inner: Arc<nexora_runtime::scheduler::RequestScheduler>,
}

impl RequestScheduler {
    pub fn new(max_concurrent_requests: usize) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            queue: RwLock::new(VecDeque::new()),
            requests: Arc::new(RwLock::new(HashMap::new())),
            max_concurrent: max_concurrent_requests,
            max_batch_size: 32,
            max_queue_time_ms: 5000,
            active_count: RwLock::new(0),
            batch_collector: Arc::new(RwLock::new(BatchCollector::new(32, 50))),
            shutdown: AtomicBool::new(false),
            notify: Arc::new(Notify::new()),
            shutdown_tx,
            shutdown_rx,
            background_handles: tokio::sync::Mutex::new(Vec::new()),
            total_queue_time_ms: RwLock::new(0.0),
            total_processing_time_ms: RwLock::new(0.0),
            scheduler_start: Instant::now(),
            inner: Arc::new(nexora_runtime::scheduler::RequestScheduler::new(
                max_concurrent_requests,
            )),
        }
    }

    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    pub fn with_max_queue_time(mut self, max_queue_time_ms: u64) -> Self {
        self.max_queue_time_ms = max_queue_time_ms;
        self
    }

    /// Set the scheduling strategy on the inner delegate scheduler.
    pub fn with_strategy(
        mut self,
        strategy: nexora_runtime::scheduler::SchedulingStrategy,
    ) -> Result<Self, anyhow::Error> {
        let inner = Arc::into_inner(self.inner)
            .ok_or_else(|| anyhow::anyhow!("inner Arc has refcount > 1, cannot mutate"))?;
        self.inner = Arc::new(inner.with_strategy(strategy));
        Ok(self)
    }

    pub async fn initialize(&mut self) -> Result<(), anyhow::Error> {
        self.shutdown.store(false, Ordering::Relaxed);
        self.inner.initialize().await?;
        Ok(())
    }

    pub async fn submit_request(
        &self,
        request_id: Uuid,
        response_tx: tokio::sync::mpsc::Sender<crate::InferenceResponse>,
    ) -> Result<(), anyhow::Error> {
        // Register via inner scheduler for strategy-aware queue management.
        let runtime_rx = {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<nexora_runtime::InferenceResponse>(16);
            let response_tx_clone = response_tx.clone();
            let handle = tokio::spawn(async move {
                while let Some(resp) = rx.recv().await {
                    let req_id: uuid::Uuid = resp
                        .request_id
                        .as_deref()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(request_id);
                    let _ = response_tx_clone
                        .send(crate::InferenceResponse {
                            request_id: req_id,
                            inference_time_ms: resp.processing_time_ms,
                            finish_reason: crate::FinishReason::StopSequence,
                            text: String::new(),
                            tokens: Vec::new(),
                            total_tokens: 0,
                            metadata: resp.metadata,
                        })
                        .await;
                }
            });
            let mut handles = self.background_handles.lock().await;
            handles.push(handle);
            tx
        };

        let runtime_req = nexora_runtime::InferenceRequest {
            model_id: String::new(),
            inputs: vec![],
            parameters: std::collections::HashMap::new(),
            request_id: Some(request_id.to_string()),
            input_tokens: vec![],
            target_tokens: None,
            priority: 50,
            metadata: std::collections::HashMap::new(),
        };
        self.inner.submit_request(runtime_req, runtime_rx).await?;

        let mut requests = self.requests.write().await;
        requests.insert(
            request_id,
            ScheduledRequest {
                _request_id: request_id,
                response_tx,
                status: RequestStatus::Queued,
                queued_at: Instant::now(),
                processing_started_at: None,
                batch_key: None,
            },
        );
        let mut queue = self.queue.write().await;
        queue.push_back(request_id);
        Ok(())
    }

    /// Submit a request and register it in the batch collector for potential batching.
    /// Returns the batch collector grouping key.
    pub async fn submit_request_batched(
        &self,
        request: &crate::InferenceRequest,
        response_tx: tokio::sync::mpsc::Sender<crate::InferenceResponse>,
    ) -> Result<BatchKey, anyhow::Error> {
        let mut requests = self.requests.write().await;
        let mut queue = self.queue.write().await;

        let key = BatchKey::from_request(request);
        requests.insert(
            request.request_id,
            ScheduledRequest {
                _request_id: request.request_id,
                response_tx,
                status: RequestStatus::Queued,
                queued_at: Instant::now(),
                processing_started_at: None,
                batch_key: Some(key.clone()),
            },
        );
        queue.push_back(request.request_id);
        Ok(key)
    }

    /// Check all queued requests for timeouts, not just those at the front.
    /// Scans the full queue and times out any request that has exceeded max_queue_time_ms.
    async fn drain_timed_out_requests(&self) {
        let guard = self.requests.read().await;
        let timed_out_ids: Vec<Uuid> = guard
            .iter()
            .filter(|(_, s)| matches!(s.status, RequestStatus::Queued))
            .filter(|(_, s)| s.queued_at.elapsed().as_millis() as u64 > self.max_queue_time_ms)
            .map(|(id, _)| *id)
            .collect();
        drop(guard);

        for req_id in timed_out_ids {
            self.update_status(req_id, RequestStatus::Timeout).await;
            let mut resp = crate::InferenceResponse::new(req_id);
            resp.finish_reason = crate::FinishReason::Timeout;
            let guard = self.requests.read().await;
            if let Some(tx) = guard.get(&req_id).map(|r| r.response_tx.clone()) {
                drop(guard);
                let _ = tx.send(resp).await;
            }
        }
    }

    /// Watchdog that checks for timed-out in-flight requests.
    /// Runs as a background task, checking every second.
    async fn watchdog_check(&self) {
        let guard = self.requests.read().await;
        let timed_out: Vec<Uuid> = guard
            .iter()
            .filter(|(_, s)| matches!(s.status, RequestStatus::Processing))
            .filter(|(_, s)| {
                if let Some(started) = s.processing_started_at {
                    started.elapsed().as_millis() as u64 > self.max_queue_time_ms
                } else {
                    s.queued_at.elapsed().as_millis() as u64 > self.max_queue_time_ms
                }
            })
            .map(|(id, _)| *id)
            .collect();
        drop(guard);

        for req_id in timed_out {
            let mut resp = crate::InferenceResponse::new(req_id);
            resp.finish_reason = crate::FinishReason::Timeout;
            let guard = self.requests.read().await;
            if let Some(tx) = guard.get(&req_id).map(|r| r.response_tx.clone()) {
                drop(guard);
                let _ = tx.send(resp).await;
            }
            let _ = self.complete_request(req_id).await;
        }
    }

    /// Pop the next ready batch from the collector.
    /// Returns None if no batch is ready or max_concurrent is saturated.
    /// Requests that have exceeded max_queue_time_ms are timed out before processing.
    pub async fn pop_batch(&self) -> Option<Batch> {
        let active = *self.active_count.read().await;
        if active >= self.max_concurrent {
            return None;
        }
        let mut collector = self.batch_collector.write().await;
        let batches = collector.drain_ready();
        if batches.is_empty() {
            return None;
        }

        let mut timed_out = Vec::new();

        // Find first batch with non-timed-out requests
        let batch = 'search: {
            for mut b in batches {
                let mut batch_timed_out = Vec::new();
                let guard = self.requests.read().await;
                b.requests.retain(|breq| {
                    if let Some(sreq) = guard.get(&breq.request_id) {
                        let elapsed_ms = sreq.queued_at.elapsed().as_millis() as u64;
                        if elapsed_ms > self.max_queue_time_ms {
                            batch_timed_out.push(breq.request_id);
                            return false;
                        }
                    }
                    true
                });
                drop(guard);
                timed_out.extend(batch_timed_out);

                if !b.requests.is_empty() {
                    break 'search Some(b);
                }
            }
            None
        };

        // Send timeout responses for timed-out requests
        for req_id in &timed_out {
            self.update_status(*req_id, RequestStatus::Timeout).await;
            let mut resp = crate::InferenceResponse::new(*req_id);
            resp.finish_reason = crate::FinishReason::Timeout;
            let guard = self.requests.read().await;
            if let Some(tx) = guard.get(req_id).map(|r| r.response_tx.clone()) {
                drop(guard);
                if let Err(e) = tx.send(resp).await {
                    tracing::warn!("Failed to send response for request {:?}: {}", req_id, e);
                }
            }
        }

        let batch = match batch {
            Some(b) => b,
            None => return None,
        };

        // claim a concurrent slot for the whole batch
        {
            let mut active = self.active_count.write().await;
            *active += 1;
        }
        for breq in &batch.requests {
            self.update_status(breq.request_id, RequestStatus::Processing)
                .await;
            let mut requests = self.requests.write().await;
            if let Some(sreq) = requests.get_mut(&breq.request_id) {
                sreq.processing_started_at = Some(Instant::now());
            }
        }
        Some(batch)
    }

    async fn update_status(&self, request_id: Uuid, status: RequestStatus) {
        let mut requests = self.requests.write().await;
        if let Some(req) = requests.get_mut(&request_id) {
            if status == RequestStatus::Processing && req.processing_started_at.is_none() {
                req.processing_started_at = Some(Instant::now());
            }
            if req.status != status
                && matches!(
                    &status,
                    RequestStatus::Completed | RequestStatus::Failed(_) | RequestStatus::Timeout
                )
            {
                if let Some(started) = req.processing_started_at {
                    let queue_time = started.duration_since(req.queued_at).as_secs_f64() * 1000.0;
                    let processing_time = started.elapsed().as_secs_f64() * 1000.0;
                    *self.total_queue_time_ms.write().await += queue_time;
                    *self.total_processing_time_ms.write().await += processing_time;
                }
            }
            req.status = status;
        }
    }

    pub async fn cancel_request(&self, request_id: Uuid) -> Result<bool, anyhow::Error> {
        // Delegate to inner runtime scheduler for strategy-aware cancellation.
        let inner_result = self.inner.cancel_request(request_id).await?;
        // Also clean up local tracking.
        let mut requests = self.requests.write().await;
        if let Some(req) = requests.get_mut(&request_id) {
            if matches!(
                req.status,
                RequestStatus::Queued | RequestStatus::Processing
            ) {
                req.status = RequestStatus::Cancelled;
                return Ok(true);
            }
        }
        Ok(inner_result)
    }

    pub async fn get_request_status(
        &self,
        request_id: Uuid,
    ) -> Result<Option<RequestStatus>, anyhow::Error> {
        // Check inner scheduler first, fall back to local tracking.
        if let Ok(Some(inner_status)) = self.inner.get_request_status(request_id).await {
            let mapped = match inner_status {
                nexora_runtime::scheduler::RequestStatus::Queued => RequestStatus::Queued,
                nexora_runtime::scheduler::RequestStatus::Processing => RequestStatus::Processing,
                nexora_runtime::scheduler::RequestStatus::Completed => RequestStatus::Completed,
                nexora_runtime::scheduler::RequestStatus::Failed(msg) => RequestStatus::Failed(msg),
                nexora_runtime::scheduler::RequestStatus::Cancelled => RequestStatus::Cancelled,
                nexora_runtime::scheduler::RequestStatus::Timeout => RequestStatus::Timeout,
            };
            return Ok(Some(mapped));
        }
        let requests = self.requests.read().await;
        Ok(requests.get(&request_id).map(|r| r.status.clone()))
    }

    pub async fn shutdown(&self) -> Result<(), anyhow::Error> {
        // Delegate to inner scheduler for strategy-aware shutdown.
        self.inner.shutdown().await?;
        self.shutdown.store(true, Ordering::Relaxed);
        let _ = self.shutdown_tx.send(true);
        let mut requests = self.requests.write().await;
        let mut queue = self.queue.write().await;
        queue.clear();
        requests.retain(|_, req| matches!(req.status, RequestStatus::Processing));
        self.cancel_background_handles().await;
        Ok(())
    }

    /// Cancel all tracked background handles with graceful shutdown.
    pub async fn cancel_background_handles(&self) {
        let handles = {
            let mut h = self.background_handles.lock().await;
            h.drain(..).collect::<Vec<_>>()
        };
        for h in handles {
            let _ = tokio::time::timeout(Duration::from_secs(5), h).await;
        }
    }

    pub async fn get_stats(&self) -> SchedulerStats {
        let requests = self.requests.read().await;
        let batching = self.batch_collector.read().await;
        let inner_stats = self.inner.get_stats().await;
        let mut stats = SchedulerStats {
            total_requests: requests.len(),
            active_requests: 0,
            queued_requests: 0,
            completed_requests: 0,
            failed_requests: 0,
            pending_batches: batching.batch_count(),
            pending_batch_requests: batching.pending_count(),
            avg_queue_time_ms: inner_stats.avg_queue_time_ms,
            avg_processing_time_ms: inner_stats.avg_processing_time_ms,
            throughput_rps: inner_stats.throughput_rps,
        };
        for req in requests.values() {
            match req.status {
                RequestStatus::Queued => stats.queued_requests += 1,
                RequestStatus::Processing => stats.active_requests += 1,
                RequestStatus::Completed => stats.completed_requests += 1,
                RequestStatus::Failed(_) => stats.failed_requests += 1,
                RequestStatus::Cancelled => {}
                RequestStatus::Timeout => {}
            }
        }
        let completed_count = stats.completed_requests.max(1);
        let total_queue = *self.total_queue_time_ms.read().await;
        let total_proc = *self.total_processing_time_ms.read().await;
        stats.avg_queue_time_ms = total_queue / completed_count as f64;
        stats.avg_processing_time_ms = total_proc / completed_count as f64;
        let elapsed = self.scheduler_start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            stats.throughput_rps = completed_count as f64 / elapsed;
        }
        stats
    }

    /// Register a submitted request with the batch collector for batching.
    /// Call this in the request loop after receiving the request from the channel.
    /// Notify the background worker that new requests may be ready
    pub fn notify_background(&self) {
        self.notify.notify_one();
    }

    pub async fn add_to_batch_collector(&self, request: &crate::InferenceRequest) {
        let requests = self.requests.read().await;
        if let Some(req) = requests.get(&request.request_id) {
            if req.status != RequestStatus::Queued {
                return;
            }
            let key = BatchKey::from_request(request);
            let response_tx = req.response_tx.clone();
            drop(requests);
            let mut collector = self.batch_collector.write().await;
            collector.add_request(request.clone(), response_tx);
            // update the stored batch_key
            drop(collector);
            let mut requests = self.requests.write().await;
            if let Some(sreq) = requests.get_mut(&request.request_id) {
                sreq.batch_key = Some(key);
            }
        }
    }

    pub async fn send_response(
        &self,
        request_id: Uuid,
        response: crate::InferenceResponse,
    ) -> Result<(), anyhow::Error> {
        // Clone the sender while holding the lock, then drop the lock
        // before awaiting on `.send()` to avoid holding an async lock
        // across an await point (reduces contention / deadlock risk).
        let maybe_tx = {
            let requests = self.requests.read().await;
            requests.get(&request_id).map(|req| req.response_tx.clone())
        };

        if let Some(tx) = maybe_tx {
            tx.send(response)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to send response: {}", e))?;
        }

        Ok(())
    }

    pub async fn complete_request(&self, request_id: Uuid) -> Result<(), anyhow::Error> {
        // Delegate to inner scheduler for stats tracking.
        let _ = self.inner.complete_request(request_id).await;
        {
            let requests = self.requests.read().await;
            if let Some(sreq) = requests.get(&request_id) {
                let queue_time = sreq.queued_at.elapsed().as_secs_f64() * 1000.0;
                let processing_time = sreq
                    .processing_started_at
                    .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);
                let mut total_q = self.total_queue_time_ms.write().await;
                let mut total_p = self.total_processing_time_ms.write().await;
                *total_q += queue_time;
                *total_p += processing_time;
            }
        }
        self.update_status(request_id, RequestStatus::Completed)
            .await;
        let mut active = self.active_count.write().await;
        if *active > 0 {
            *active -= 1;
        }
        Ok(())
    }

    /// Mark all requests in a batch as completed and release the concurrent slot.
    pub async fn complete_batch(&self, batch: &Batch) -> Result<(), anyhow::Error> {
        for breq in &batch.requests {
            self.update_status(breq.request_id, RequestStatus::Completed)
                .await;
        }
        let mut active = self.active_count.write().await;
        if *active > 0 {
            *active -= 1;
        }
        Ok(())
    }

    /// Complete a batch by batch ID without requiring the full Batch struct.
    /// Verifies the batch exists in the collector before releasing the concurrent slot.
    pub async fn complete_batch_id(&self, batch_id: uuid::Uuid) -> Result<(), anyhow::Error> {
        // Verify the batch exists by checking batch collector state
        let collector = self.batch_collector.read().await;
        if !collector.contains_batch(batch_id) {
            tracing::warn!(
                "complete_batch_id: batch {} not found in collector, slot may already be released",
                batch_id
            );
            // Don't error — idempotent release is safer than double-counting
        }
        drop(collector);

        let mut active = self.active_count.write().await;
        if *active > 0 {
            *active -= 1;
        }
        Ok(())
    }
}

#[async_trait]
impl nexora_runtime::Scheduler for RequestScheduler {
    async fn submit(
        &self,
        request_id: Uuid,
        response_tx: tokio::sync::mpsc::Sender<nexora_runtime::InferenceResponse>,
    ) -> anyhow::Result<()> {
        // Delegate to inner runtime scheduler for strategy-aware submission.
        self.inner.submit(request_id, response_tx).await
    }

    async fn cancel(&self, request_id: Uuid) -> anyhow::Result<bool> {
        self.inner.cancel(request_id).await
    }

    async fn status(
        &self,
        request_id: Uuid,
    ) -> anyhow::Result<Option<nexora_runtime::RequestStatus>> {
        self.inner.status(request_id).await
    }

    async fn stats(&self) -> nexora_runtime::SchedulerStats {
        self.inner.stats().await
    }

    async fn shutdown(&self) -> anyhow::Result<()> {
        self.inner.shutdown().await
    }
}

impl Default for RequestScheduler {
    fn default() -> Self {
        Self::new(16)
    }
}
