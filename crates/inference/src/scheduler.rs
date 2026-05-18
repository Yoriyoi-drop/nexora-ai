use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
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
}

impl RequestScheduler {
    pub fn new() -> Self {
        Self {
            queue: RwLock::new(VecDeque::new()),
            requests: RwLock::new(HashMap::new()),
            max_concurrent: 4,
            max_batch_size: 8,
            active_count: RwLock::new(0),
            batch_collector: Arc::new(RwLock::new(BatchCollector::new(8, 50))),
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
        Ok(())
    }

    pub async fn submit_request(
        &self,
        request_id: Uuid,
        response_tx: tokio::sync::mpsc::Sender<crate::InferenceResponse>,
    ) -> Result<(), anyhow::Error> {
        let mut requests = self.requests.write().await;
        let mut queue = self.queue.write().await;

        requests.insert(request_id, ScheduledRequest {
            _request_id: request_id,
            response_tx,
            status: RequestStatus::Queued,
            _submitted_at: chrono::Utc::now(),
            batch_key: None,
        });
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
        requests.insert(request.request_id, ScheduledRequest {
            _request_id: request.request_id,
            response_tx,
            status: RequestStatus::Queued,
            _submitted_at: chrono::Utc::now(),
            batch_key: Some(key.clone()),
        });
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
            self.update_status(breq.request_id, RequestStatus::Processing).await;
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
            if matches!(req.status, RequestStatus::Queued | RequestStatus::Processing) {
                req.status = RequestStatus::Cancelled;
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub async fn get_request_status(&self, request_id: Uuid) -> Result<Option<RequestStatus>, anyhow::Error> {
        let requests = self.requests.read().await;
        Ok(requests.get(&request_id).map(|r| r.status.clone()))
    }

    pub async fn shutdown(&self) -> Result<(), anyhow::Error> {
        let mut queue = self.queue.write().await;
        queue.clear();
        let mut requests = self.requests.write().await;
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
                _ => {}
            }
        }
        stats
    }

    /// Register a submitted request with the batch collector for batching.
    /// Call this in the request loop after receiving the request from the channel.
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

    pub async fn send_response(&self, request_id: Uuid, response: crate::InferenceResponse) -> Result<(), anyhow::Error> {
        let requests = self.requests.read().await;
        if let Some(req) = requests.get(&request_id) {
            req.response_tx.send(response).await.map_err(|e| anyhow::anyhow!("Failed to send response: {}", e))?;
        }
        Ok(())
    }

    pub async fn complete_request(&self, request_id: Uuid) -> Result<(), anyhow::Error> {
        self.update_status(request_id, RequestStatus::Completed).await;
        let mut active = self.active_count.write().await;
        if *active > 0 {
            *active -= 1;
        }
        Ok(())
    }

    /// Mark all requests in a batch as completed and release the concurrent slot.
    pub async fn complete_batch(&self, batch: &Batch) -> Result<(), anyhow::Error> {
        for breq in &batch.requests {
            self.update_status(breq.request_id, RequestStatus::Completed).await;
        }
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
