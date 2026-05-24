use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{watch, Mutex, Notify, RwLock};
use tokio::task::JoinHandle;
use tracing::debug;
use uuid::Uuid;

use crate::batching::{Batch, BatchCollector, BatchKey};
use async_trait::async_trait;

#[derive(Debug, Clone, PartialEq)]
pub enum RequestStatus {
    Queued,
    Processing,
    Completed,
    Failed(String),
    Cancelled,
    Timeout,
}

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

/// Handler type for processing batches through the inference engine
pub type BatchHandler = Arc<dyn Fn(Batch) -> crate::Result<()> + Send + Sync>;

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
    engine_handler: Option<BatchHandler>,
    total_queue_time_ms: RwLock<f64>,
    total_processing_time_ms: RwLock<f64>,
    scheduler_start: Instant,
}

impl RequestScheduler {
    pub fn new() -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            queue: RwLock::new(VecDeque::new()),
            requests: Arc::new(RwLock::new(HashMap::new())),
            max_concurrent: 4,
            max_batch_size: 8,
            max_queue_time_ms: 5000,
            active_count: RwLock::new(0),
            batch_collector: Arc::new(RwLock::new(BatchCollector::new(8, 50))),
            shutdown: AtomicBool::new(false),
            notify: Arc::new(Notify::new()),
            shutdown_tx,
            shutdown_rx,
            background_handles: tokio::sync::Mutex::new(Vec::new()),
            engine_handler: None,
            total_queue_time_ms: RwLock::new(0.0),
            total_processing_time_ms: RwLock::new(0.0),
            scheduler_start: Instant::now(),
        }
    }

    /// Set the engine handler for batch processing
    pub fn with_engine_handler(mut self, handler: BatchHandler) -> Self {
        self.engine_handler = Some(handler);
        self
    }

    /// Set the engine handler after construction
    pub fn set_engine_handler(&mut self, handler: BatchHandler) {
        self.engine_handler = Some(handler);
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

    pub async fn initialize(&mut self) -> Result<(), anyhow::Error> {
        self.shutdown.store(false, Ordering::Relaxed);
        Ok(())
    }

    pub fn validate(&self) -> Result<(), anyhow::Error> {
        if self.engine_handler.is_none() {
            return Err(anyhow::anyhow!("engine_handler not set — call set_engine_handler before start"));
        }
        Ok(())
    }

    /// Spawn a background worker that periodically polls for ready batches,
    /// dispatches them to the inference engine, and fans out responses.
    /// This ensures both timed-out and ready batches are drained.
    /// Returns the JoinHandle so callers can await/cancel the background worker.
    pub fn start(this: Arc<RwLock<Self>>) -> JoinHandle<()> {
        tokio::spawn(async move {
            {
                let s = this.read().await;
                if s.engine_handler.is_none() {
                    tracing::error!("engine_handler not set — call set_engine_handler before start");
                    return;
                }
            }
            let notify = {
                let s = this.read().await;
                s.notify.clone()
            };
            let mut shutdown_rx = {
                let s = this.read().await;
                s.shutdown_rx.clone()
            };
            let mut last_watchdog = Instant::now();
            loop {
                if this.read().await.shutdown.load(Ordering::Relaxed) {
                    break;
                }

                // Periodic timeout sweep for queued requests (every pop)
                this.read().await.drain_timed_out_requests().await;

                // Periodic watchdog for in-flight requests (every 1s)
                if last_watchdog.elapsed() >= Duration::from_secs(1) {
                    this.read().await.watchdog_check().await;
                    last_watchdog = Instant::now();
                }

                while let Some(batch) = this.read().await.pop_batch().await {
                    debug!("background worker popped ready batch {}", batch.batch_id);
                    let req_ids: Vec<Uuid> = batch.requests.iter().map(|r| r.request_id).collect();
                    let handler = this.read().await.engine_handler.clone();
                    match handler {
                        Some(engine_fn) => {
                            match engine_fn(batch) {
                                Ok(_) => {
                                    for req_id in &req_ids {
                                        if let Err(e) = this.read().await.complete_request(*req_id).await {
                                            tracing::warn!("failed to complete request {}: {}", req_id, e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    for req_id in &req_ids {
                                        this.read().await.update_status(*req_id, RequestStatus::Failed(e.to_string())).await;
                                        let mut resp = crate::InferenceResponse::new(*req_id);
                                        resp.finish_reason = crate::FinishReason::Error(e.to_string());
                                        let requests = this.read().await.requests.clone();
                                        let tx = {
                                            let requests = requests.read().await;
                                            requests.get(req_id).map(|r| r.response_tx.clone())
                                        };
                                        if let Some(tx) = tx {
                                            if let Err(e) = tx.send(resp).await {
                                                tracing::warn!("Failed to send response for request {:?}: {}", req_id, e);
                                            }
                                        }
                                        if let Err(e) = this.read().await.complete_request(*req_id).await {
                                            tracing::warn!("failed to complete errored request: {}", e);
                                        }
                                    }
                                }
                            }
                        }
                        None => {
                            for req_id in &req_ids {
                                this.read().await.update_status(*req_id, RequestStatus::Timeout).await;
                                let mut resp = crate::InferenceResponse::new(*req_id);
                                resp.finish_reason = crate::FinishReason::Timeout;
                                let requests = this.read().await.requests.clone();
                                let tx = {
                                    let requests = requests.read().await;
                                    requests.get(req_id).map(|r| r.response_tx.clone())
                                };
                                if let Some(tx) = tx {
                                    if let Err(e) = tx.send(resp).await {
                                        tracing::warn!("Failed to send response for request {:?}: {}", req_id, e);
                                    }
                                }
                                if let Err(e) = this.read().await.complete_request(*req_id).await {
                                    tracing::warn!("background worker: failed to complete request {}: {}", req_id, e);
                                }
                            }
                        }
                    }
                }
                tokio::select! {
                    biased;
                    _ = shutdown_rx.changed() => {
                        tracing::debug!("background worker received shutdown signal");
                        break;
                    }
                    _ = notify.notified() => {},
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {},
                }
            }
        })
    }

    pub async fn submit_request(
        &self,
        request_id: Uuid,
        response_tx: tokio::sync::mpsc::Sender<crate::InferenceResponse>,
    ) -> Result<(), anyhow::Error> {
        let mut requests = self.requests.write().await;
        let mut queue = self.queue.write().await;

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
                && matches!(&status, RequestStatus::Completed | RequestStatus::Failed(_) | RequestStatus::Timeout)
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
        Ok(false)
    }

    pub async fn get_request_status(
        &self,
        request_id: Uuid,
    ) -> Result<Option<RequestStatus>, anyhow::Error> {
        let requests = self.requests.read().await;
        Ok(requests.get(&request_id).map(|r| r.status.clone()))
    }

    pub async fn shutdown(&self) -> Result<(), anyhow::Error> {
        self.shutdown.store(true, Ordering::Relaxed);
        let _ = self.shutdown_tx.send(true);
        let mut requests = self.requests.write().await;
        let mut queue = self.queue.write().await;
        queue.clear();
        requests.retain(|_, req| matches!(req.status, RequestStatus::Processing));
        // Await background handles for graceful shutdown
        self.cancel_background_handles().await;
        Ok(())
    }

    /// Register a background handle for tracking (called from start() result).
    pub async fn register_background_handle(&self, handle: JoinHandle<()>) {
        let mut handles = self.background_handles.lock().await;
        handles.push(handle);
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
        let mut stats = SchedulerStats {
            total_requests: requests.len(),
            active_requests: 0,
            queued_requests: 0,
            completed_requests: 0,
            failed_requests: 0,
            pending_batches: batching.batch_count(),
            pending_batch_requests: batching.pending_count(),
            avg_queue_time_ms: 0.0,
            avg_processing_time_ms: 0.0,
            throughput_rps: 0.0,
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
        let (tx, mut rx) = tokio::sync::mpsc::channel::<crate::InferenceResponse>(16);
        let handle = tokio::spawn(async move {
            while let Some(resp) = rx.recv().await {
                let runtime_resp = nexora_runtime::InferenceResponse {
                    request_id: Some(resp.request_id.to_string()),
                    model_id: String::new(),
                    outputs: Vec::new(),
                    metadata: resp.metadata,
                    processing_time_ms: resp.inference_time_ms,
                };
                if response_tx.send(runtime_resp).await.is_err() {
                    break;
                }
            }
        });
        let mut handles = self.background_handles.lock().await;
        handles.push(handle);
        self.submit_request(request_id, tx)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    async fn cancel(&self, request_id: Uuid) -> anyhow::Result<bool> {
        self.cancel_request(request_id)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    async fn status(
        &self,
        request_id: Uuid,
    ) -> anyhow::Result<Option<nexora_runtime::RequestStatus>> {
        let status = self
            .get_request_status(request_id)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(status.map(|s| match s {
            RequestStatus::Queued => nexora_runtime::RequestStatus::Queued,
            RequestStatus::Processing => nexora_runtime::RequestStatus::Processing,
            RequestStatus::Completed => nexora_runtime::RequestStatus::Completed,
            RequestStatus::Failed(msg) => nexora_runtime::RequestStatus::Failed(msg),
            RequestStatus::Cancelled => nexora_runtime::RequestStatus::Cancelled,
            RequestStatus::Timeout => nexora_runtime::RequestStatus::Timeout,
        }))
    }

    async fn stats(&self) -> nexora_runtime::SchedulerStats {
        let s = self.get_stats().await;
        nexora_runtime::SchedulerStats {
            total_requests: s.total_requests as u64,
            queued_requests: s.queued_requests as u64,
            processed_requests: s.completed_requests as u64,
            failed_requests: s.failed_requests as u64,
            cancelled_requests: 0,
            current_queue_length: s.queued_requests,
            current_active_requests: s.active_requests,
            avg_queue_time_ms: s.avg_queue_time_ms,
            avg_processing_time_ms: s.avg_processing_time_ms,
            throughput_rps: s.throughput_rps,
            last_updated: chrono::Utc::now(),
        }
    }

    async fn shutdown(&self) -> anyhow::Result<()> {
        RequestScheduler::shutdown(self)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
    }
}

impl Default for RequestScheduler {
    fn default() -> Self {
        Self::new()
    }
}
