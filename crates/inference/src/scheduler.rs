use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{watch, Notify, RwLock};
use tracing::debug;
use uuid::Uuid;

use crate::batching::{Batch, BatchCollector, BatchKey};

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
}

struct ScheduledRequest {
    _request_id: Uuid,
    response_tx: tokio::sync::mpsc::Sender<crate::InferenceResponse>,
    status: RequestStatus,
    _submitted_at: chrono::DateTime<chrono::Utc>,
    batch_key: Option<BatchKey>,
}

pub struct RequestScheduler {
    queue: RwLock<VecDeque<Uuid>>,
    requests: RwLock<HashMap<Uuid, ScheduledRequest>>,
    max_concurrent: usize,
    max_batch_size: usize,
    active_count: RwLock<usize>,
    batch_collector: Arc<RwLock<BatchCollector>>,
    shutdown: AtomicBool,
    notify: Arc<Notify>,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
}

impl RequestScheduler {
    pub fn new() -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            queue: RwLock::new(VecDeque::new()),
            requests: RwLock::new(HashMap::new()),
            max_concurrent: 4,
            max_batch_size: 8,
            active_count: RwLock::new(0),
            batch_collector: Arc::new(RwLock::new(BatchCollector::new(8, 50))),
            shutdown: AtomicBool::new(false),
            notify: Arc::new(Notify::new()),
            shutdown_tx,
            shutdown_rx,
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

    pub async fn initialize(&mut self) -> Result<(), anyhow::Error> {
        self.shutdown.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Spawn a background worker that periodically polls for ready batches
    /// and processes them by fanning out responses to request channels.
    /// This ensures timed-out batches are drained even when the engine's
    /// request loop is busy elsewhere.
    pub fn start(this: Arc<RwLock<Self>>) {
        tokio::spawn(async move {
            let notify = {
                let s = this.read().await;
                s.notify.clone()
            };
            let mut shutdown_rx = {
                let s = this.read().await;
                s.shutdown_rx.clone()
            };
            loop {
                if this.read().await.shutdown.load(Ordering::Relaxed) {
                    break;
                }
                while let Some(batch) = this.read().await.pop_batch().await {
                    debug!("background worker popped ready batch {}", batch.batch_id);
                    // Process each request in the batch by sending a timeout response
                    // so callers don't hang indefinitely.
                    for breq in &batch.requests {
                        this.read().await.update_status(breq.request_id, RequestStatus::Timeout).await;
                        let mut resp = crate::InferenceResponse::new(breq.request_id);
                        resp.finish_reason = crate::FinishReason::Timeout;
                        let guard = this.read().await;
                        let requests = guard.requests.read().await;
                        if let Some(tx) = requests.get(&breq.request_id).map(|r| r.response_tx.clone()) {
                            let _ = tx.send(resp).await;
                        }
                        if let Err(e) = this.read().await.complete_request(breq.request_id).await {
                            tracing::warn!("background worker: failed to complete request {}: {}", breq.request_id, e);
                        }
                    }
                }
                tokio::select! {
                    biased;
                    _ = shutdown_rx.changed() => {},
                    _ = notify.notified() => {},
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {},
                }
            }
        });
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
                _submitted_at: chrono::Utc::now(),
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
                _submitted_at: chrono::Utc::now(),
                batch_key: Some(key.clone()),
            },
        );
        queue.push_back(request.request_id);
        Ok(key)
    }

    /// Pop the next ready batch from the collector.
    /// Returns None if no batch is ready or max_concurrent is saturated.
    pub async fn pop_batch(&self) -> Option<Batch> {
        let active = *self.active_count.read().await;
        if active >= self.max_concurrent {
            return None;
        }
        let mut collector = self.batch_collector.write().await;
        let mut batches = collector.drain_ready();
        if batches.is_empty() {
            return None;
        }
        let batch = batches.remove(0);
        // claim a concurrent slot for the whole batch
        {
            let mut active = self.active_count.write().await;
            *active += 1;
        }
        for breq in &batch.requests {
            self.update_status(breq.request_id, RequestStatus::Processing)
                .await;
        }
        Some(batch)
    }

    async fn update_status(&self, request_id: Uuid, status: RequestStatus) {
        let mut requests = self.requests.write().await;
        if let Some(req) = requests.get_mut(&request_id) {
            req.status = status.clone();
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
        Ok(())
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
    /// Releases the concurrent slot for the batch.
    pub async fn complete_batch_id(&self, _batch_id: uuid::Uuid) -> Result<(), anyhow::Error> {
        let mut active = self.active_count.write().await;
        if *active > 0 {
            *active -= 1;
        }
        Ok(())
    }
}

impl Default for RequestScheduler {
    fn default() -> Self {
        Self::new()
    }
}
