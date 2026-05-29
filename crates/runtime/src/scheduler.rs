use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{InferenceError, InferenceRequest, InferenceResponse, Result, Scheduler};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SchedulingStrategy {
    FIFO,
    Priority,
    SJF,
    RoundRobin,
    Fair,
}

#[derive(Debug, Clone)]
pub struct QueuedRequest {
    pub request: InferenceRequest,
    pub response_tx: mpsc::Sender<InferenceResponse>,
    pub queued_at: DateTime<Utc>,
    pub priority: u8,
    pub estimated_time_ms: u64,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RequestStatus {
    Queued,
    Processing,
    Completed,
    Failed(String),
    Cancelled,
    Timeout,
}

struct SchedulerInner {
    queue: VecDeque<QueuedRequest>,
    active: HashMap<Uuid, RequestInfo>,
    status: HashMap<Uuid, RequestStatus>,
    channels: HashMap<Uuid, mpsc::Sender<InferenceResponse>>,
    stats: SchedulerStats,
    state: SchedulerState,
    round_robin_index: usize,
}

pub struct RequestScheduler {
    max_concurrent_requests: usize,
    max_queue_time_ms: u64,
    strategy: SchedulingStrategy,
    inner: Arc<RwLock<SchedulerInner>>,
}

#[derive(Debug, Clone)]
pub struct RequestInfo {
    pub request_id: Uuid,
    pub started_at: DateTime<Utc>,
    pub processing_time_ms: u64,
    pub priority: u8,
    pub session_id: Option<Uuid>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub total_requests: u64,
    pub queued_requests: u64,
    pub processed_requests: u64,
    pub failed_requests: u64,
    pub cancelled_requests: u64,
    pub current_queue_length: usize,
    pub current_active_requests: usize,
    pub avg_queue_time_ms: f64,
    pub avg_processing_time_ms: f64,
    pub throughput_rps: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerState {
    Uninitialized,
    Initializing,
    Ready,
    ShuttingDown,
    Shutdown,
}

impl RequestScheduler {
    pub fn new(max_concurrent_requests: usize) -> Self {
        Self {
            max_concurrent_requests,
            max_queue_time_ms: 30000,
            strategy: SchedulingStrategy::Priority,
            inner: Arc::new(RwLock::new(SchedulerInner {
                queue: VecDeque::new(),
                active: HashMap::new(),
                status: HashMap::new(),
                channels: HashMap::new(),
                stats: SchedulerStats::default(),
                state: SchedulerState::Uninitialized,
                round_robin_index: 0,
            })),
        }
    }

    pub fn with_max_queue_time(mut self, ms: u64) -> Self {
        self.max_queue_time_ms = ms;
        self
    }

    pub fn with_strategy(mut self, strategy: SchedulingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing request scheduler");
        let mut inner = self.inner.write().await;
        inner.state = SchedulerState::Initializing;
        inner.stats.last_updated = Utc::now();
        inner.state = SchedulerState::Ready;
        info!("Request scheduler initialized successfully");
        Ok(())
    }

    pub async fn submit_request(
        &self,
        request: InferenceRequest,
        response_tx: mpsc::Sender<InferenceResponse>,
    ) -> Result<()> {
        debug!("Submitting request to scheduler: {:?}", request.request_id);

        {
            let inner = self.inner.read().await;
            if inner.state != SchedulerState::Ready {
                return Err(
                    InferenceError::InternalError("Scheduler not ready".to_string()).into(),
                );
            }
        }

        let request_uuid = request
            .request_id
            .as_ref()
            .and_then(|s| match Uuid::parse_str(s) {
                Ok(uuid) => Some(uuid),
                Err(e) => {
                    tracing::warn!("Failed to parse request UUID '{}': {}", s, e);
                    None
                }
            })
            .unwrap_or_else(Uuid::new_v4);

        let priority = self.calculate_priority(&request);
        let estimated_time = self.estimate_processing_time(&request);

        let queued_request = QueuedRequest {
            request,
            response_tx,
            queued_at: Utc::now(),
            priority,
            estimated_time_ms: estimated_time,
            metadata: HashMap::new(),
        };

        {
            let mut inner = self.inner.write().await;
            inner
                .channels
                .insert(request_uuid, queued_request.response_tx.clone());
            inner.status.insert(request_uuid, RequestStatus::Queued);
            Self::insert_into_queue_inner(&mut inner.queue, &self.strategy, queued_request);
            inner.stats.total_requests += 1;
            inner.stats.queued_requests += 1;
            inner.stats.current_queue_length = inner.queue.len();
            inner.stats.last_updated = Utc::now();
        }

        self.process_queue().await?;
        debug!("Request {:?} queued successfully", request_uuid);
        Ok(())
    }

    pub async fn get_next_request(&self) -> Result<Option<QueuedRequest>> {
        debug!("Getting next request from scheduler");

        // Phase 1: read-only check — no exclusive lock needed
        let (strategy, rr_index) = {
            let inner = self.inner.read().await;
            if inner.active.len() >= self.max_concurrent_requests {
                debug!("Maximum concurrent requests reached ({})", inner.active.len());
                return Ok(None);
            }
            if inner.queue.is_empty() {
                return Ok(None);
            }
            (self.strategy, inner.round_robin_index)
        };

        // Phase 2: narrow write window for timeout removal, index update, and dequeue
        let mut inner = self.inner.write().await;

        // Re-check after lock upgrade (another writer may have drained the queue)
        if inner.active.len() >= self.max_concurrent_requests {
            return Ok(None);
        }
        if inner.queue.is_empty() {
            return Ok(None);
        }

        // Check for timed-out requests at the front of the queue
        let now = Utc::now();
        while let Some(front) = inner.queue.front() {
            let elapsed_ms = (now - front.queued_at).num_milliseconds().max(0) as u64;
            if elapsed_ms > self.max_queue_time_ms {
                let timed_out = match inner.queue.pop_front() {
                    Some(item) => item,
                    None => break,
                };
                let rid = timed_out
                    .request
                    .request_id
                    .as_ref()
                    .and_then(|s| Uuid::parse_str(s).ok())
                    .unwrap_or_else(Uuid::new_v4);
                inner.status.insert(rid, RequestStatus::Timeout);
                tracing::warn!("Request {} timed out after {}ms", rid, elapsed_ms);
            } else {
                break;
            }
        }

        if inner.queue.is_empty() {
            return Ok(None);
        }

        // Respect strategy when dequeuing
        let idx = match strategy {
            SchedulingStrategy::FIFO => 0,
            SchedulingStrategy::Priority => 0,
            SchedulingStrategy::SJF => 0,
            SchedulingStrategy::RoundRobin => {
                let queue_len = inner.queue.len().max(1);
                let idx = rr_index % queue_len;
                inner.round_robin_index = (idx + 1) % queue_len;
                idx
            }
            SchedulingStrategy::Fair => {
                let now = Utc::now();
                let mut best_idx = 0;
                let mut best_score = f64::NEG_INFINITY;
                for (i, req) in inner.queue.iter().enumerate() {
                    let group = req.priority / 10 * 10;
                    let wait_ms = (now - req.queued_at).num_milliseconds().max(0) as f64;
                    let age_boost = (wait_ms / 5000.0).min(5.0);
                    let score = (100 - group as i32) as f64 + age_boost;
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                    }
                }
                best_idx
            }
        };

        let queued_request = match inner.queue.remove(idx) {
            Some(r) => r,
            None => {
                tracing::warn!("scheduler: queue became empty between check and dequeue");
                return Ok(None);
            }
        };
        inner.stats.current_queue_length = inner.queue.len();
        Ok(Some(queued_request))
    }

    pub async fn start_request(&self, request_id: Uuid) -> Result<()> {
        debug!("Starting request processing: {}", request_id);

        let mut inner = self.inner.write().await;
        inner
            .status
            .get_mut(&request_id)
            .map(|s| *s = RequestStatus::Processing);

        inner.active.insert(
            request_id,
            RequestInfo {
                request_id,
                started_at: Utc::now(),
                processing_time_ms: 0,
                priority: 0,
                session_id: None,
                metadata: HashMap::new(),
            },
        );

        inner.stats.current_active_requests = inner.active.len();

        debug!("Request {} started processing", request_id);
        Ok(())
    }

    pub async fn complete_request(&self, request_id: Uuid) -> Result<()> {
        debug!("Completing request: {}", request_id);

        let mut inner = self.inner.write().await;
        let processing_time = inner
            .active
            .remove(&request_id)
            .map(|info| (Utc::now() - info.started_at).num_milliseconds().max(0) as u64)
            .unwrap_or_else(|| {
                warn!("Request {} not found in active requests", request_id);
                0
            });

        inner
            .status
            .get_mut(&request_id)
            .map(|s| *s = RequestStatus::Completed);

        inner.stats.processed_requests += 1;
        inner.stats.current_active_requests = inner.active.len();
        if inner.stats.processed_requests > 0 {
            inner.stats.avg_processing_time_ms = (inner.stats.avg_processing_time_ms
                * (inner.stats.processed_requests - 1) as f64
                + processing_time as f64)
                / inner.stats.processed_requests as f64;
        }
        inner.stats.last_updated = Utc::now();
        inner.channels.remove(&request_id);
        drop(inner);

        self.process_queue().await?;
        debug!("Request {} completed successfully", request_id);
        Ok(())
    }

    pub async fn cancel_request(&self, request_id: Uuid) -> Result<bool> {
        debug!("Cancelling request: {}", request_id);

        let mut inner = self.inner.write().await;

        let cancelled = if let Some(pos) = inner.queue.iter().position(|req| {
            req.request
                .request_id
                .as_ref()
                .and_then(|s| match Uuid::parse_str(s) {
                    Ok(uuid) => Some(uuid),
                    Err(e) => {
                        tracing::warn!("Failed to parse request UUID '{}': {}", s, e);
                        None
                    }
                })
                .map(|uuid| uuid == request_id)
                .unwrap_or(false)
        }) {
            inner.queue.remove(pos);
            true
        } else if inner.active.remove(&request_id).is_some() {
            true
        } else {
            false
        };

        if cancelled {
            inner.status.insert(request_id, RequestStatus::Cancelled);
            inner.stats.cancelled_requests += 1;
            inner.stats.current_queue_length = inner.queue.len();
            inner.stats.current_active_requests = inner.active.len();
            inner.channels.remove(&request_id);
            debug!("Request {} cancelled successfully", request_id);
            Ok(true)
        } else {
            debug!("Request {} not found for cancellation", request_id);
            Ok(false)
        }
    }

    pub async fn send_response(&self, request_id: Uuid, response: InferenceResponse) -> Result<()> {
        debug!("Sending response for request: {}", request_id);

        let response_tx = {
            let inner = self.inner.read().await;
            inner.channels.get(&request_id).cloned()
        };

        match response_tx {
            Some(tx) => match tx.try_send(response) {
                Ok(_) => Ok(()),
                Err(mpsc::error::TrySendError::Full(_)) => {
                    warn!("Response channel full for request {}", request_id);
                    Err(InferenceError::InternalError("Response channel full".to_string()).into())
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    Err(InferenceError::InternalError("Response channel closed".to_string()).into())
                }
            },
            None => {
                Err(InferenceError::InternalError("Response channel not found".to_string()).into())
            }
        }
    }

    pub async fn get_request_status(&self, request_id: Uuid) -> Result<Option<RequestStatus>> {
        let inner = self.inner.read().await;
        Ok(inner.status.get(&request_id).cloned())
    }

    pub async fn get_stats(&self) -> SchedulerStats {
        let inner = self.inner.read().await;
        inner.stats.clone()
    }

    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down request scheduler");

        let mut inner = self.inner.write().await;
        inner.state = SchedulerState::ShuttingDown;

        let queued_count = inner.queue.len();
        inner.queue.clear();
        inner.stats.cancelled_requests += queued_count as u64;

        let active_ids: Vec<Uuid> = inner.active.keys().cloned().collect();
        drop(inner);

        for request_id in active_ids {
            if let Err(e) = self.cancel_request(request_id).await {
                warn!(
                    "Failed to cancel request {} during scheduler shutdown: {}",
                    request_id, e
                );
            }
        }

        self.inner.write().await.state = SchedulerState::Shutdown;
        info!("Request scheduler shutdown complete");
        Ok(())
    }

    async fn process_queue(&self) -> Result<()> {
        while let Some(queued_request) = self.get_next_request().await? {
            self.start_request(
                queued_request
                    .request
                    .request_id
                    .as_ref()
                    .and_then(|s| match Uuid::parse_str(s) {
                        Ok(uuid) => Some(uuid),
                        Err(e) => {
                            tracing::warn!("Failed to parse request UUID '{}': {}", s, e);
                            None
                        }
                    })
                    .unwrap_or_else(Uuid::new_v4),
            )
            .await?;
        }
        Ok(())
    }

    fn insert_into_queue_inner(queue: &mut VecDeque<QueuedRequest>, strategy: &SchedulingStrategy, request: QueuedRequest) {
        let target_group = request.priority / 10 * 10;
        let insert_pos = match strategy {
            SchedulingStrategy::FIFO => None,
            SchedulingStrategy::RoundRobin => {
                // RoundRobin: interleave across priority groups for a natural
                // rotating pattern. Insert before the next existing request from
                // the same priority group. Combined with the index-based rotation
                // in get_next_request, this ensures fair round-robin dispatching.
                queue
                    .iter()
                    .position(|existing| existing.priority / 10 * 10 == target_group)
            }
            SchedulingStrategy::Fair => {
                // Fair: fair queuing by priority group. Within each group,
                // requests are FIFO. Place after same-priority requests,
                // before lower-priority groups. The dequeue logic (get_next_request)
                // boosts aged low-priority requests to prevent starvation.
                queue.iter().position(|existing| {
                    let existing_group = existing.priority / 10 * 10;
                    existing_group < target_group
                })
            }
            SchedulingStrategy::Priority => {
                queue.iter().position(|existing| request.priority > existing.priority)
            }
            SchedulingStrategy::SJF => {
                queue.iter().position(|existing| request.estimated_time_ms < existing.estimated_time_ms)
            }
        };
        match insert_pos {
            Some(pos) => queue.insert(pos, request),
            None => queue.push_back(request),
        }
    }

    fn calculate_priority(&self, request: &InferenceRequest) -> u8 {
        let mut priority = 50;
        let max_tokens = request
            .parameters
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(100);
        let streaming = request
            .parameters
            .get("streaming")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let temperature = request
            .parameters
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;

        if max_tokens < 50 {
            priority += 20;
        } else if max_tokens > 500 {
            priority -= 20;
        }
        if streaming {
            priority += 10;
        }
        if temperature < 0.1 {
            priority += 15;
        }
        priority.clamp(0, 100)
    }

    fn estimate_processing_time(&self, request: &InferenceRequest) -> u64 {
        let base_time_per_token = 50;
        let max_tokens = request
            .parameters
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(100);
        let temperature = request
            .parameters
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;
        let temperature_factor = if temperature > 0.0 { 1.5 } else { 1.0 };
        (max_tokens * base_time_per_token * temperature_factor as u64).max(100)
    }
}

#[async_trait]
impl Scheduler for RequestScheduler {
    async fn submit(
        &self,
        request_id: Uuid,
        response_tx: mpsc::Sender<crate::InferenceResponse>,
    ) -> crate::Result<()> {
        let request = InferenceRequest {
            model_id: String::new(),
            inputs: vec![],
            parameters: HashMap::new(),
            request_id: Some(request_id.to_string()),
            input_tokens: vec![],
            target_tokens: None,
            priority: 50,
            metadata: HashMap::new(),
        };
        self.submit_request(request, response_tx).await
    }

    async fn cancel(&self, request_id: Uuid) -> crate::Result<bool> {
        self.cancel_request(request_id).await
    }

    async fn status(&self, request_id: Uuid) -> crate::Result<Option<crate::RequestStatus>> {
        self.get_request_status(request_id).await
    }

    async fn stats(&self) -> crate::SchedulerStats {
        self.get_stats().await
    }

    async fn shutdown(&self) -> crate::Result<()> {
        RequestScheduler::shutdown(self).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn test_request(priority: u8) -> QueuedRequest {
        let (tx, _rx) = tokio::sync::mpsc::channel(1024);
        QueuedRequest {
            request: InferenceRequest {
                model_id: "test".into(),
                inputs: vec![],
                parameters: [("max_tokens".to_string(), serde_json::json!(100))]
                    .into_iter()
                    .collect(),
                request_id: Some(uuid::Uuid::new_v4().to_string()),
                input_tokens: vec![],
                target_tokens: None,
                priority,
                metadata: [].into_iter().collect(),
            },
            response_tx: tx,
            queued_at: Utc::now(),
            priority,
            estimated_time_ms: 100,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_insert_into_queue_fifo() {
        let mut queue = VecDeque::new();

        RequestScheduler::insert_into_queue_inner(&mut queue, &SchedulingStrategy::FIFO, test_request(50));
        RequestScheduler::insert_into_queue_inner(&mut queue, &SchedulingStrategy::FIFO, test_request(50));
        RequestScheduler::insert_into_queue_inner(&mut queue, &SchedulingStrategy::FIFO, test_request(50));

        assert_eq!(queue.len(), 3);
    }

    #[test]
    fn test_insert_into_queue_priority() {
        let mut queue = VecDeque::new();

        RequestScheduler::insert_into_queue_inner(&mut queue, &SchedulingStrategy::Priority, test_request(10));
        RequestScheduler::insert_into_queue_inner(&mut queue, &SchedulingStrategy::Priority, test_request(90));
        RequestScheduler::insert_into_queue_inner(&mut queue, &SchedulingStrategy::Priority, test_request(50));

        assert_eq!(queue.len(), 3);
        assert_eq!(queue[0].priority, 90, "highest priority first");
        assert_eq!(queue[1].priority, 50, "middle priority second");
        assert_eq!(queue[2].priority, 10, "lowest priority last");
    }

    #[test]
    fn test_insert_into_queue_sjf() {
        let mut queue = VecDeque::new();
        let mut r1 = test_request(50);
        r1.estimated_time_ms = 200;
        let mut r2 = test_request(50);
        r2.estimated_time_ms = 50;
        let mut r3 = test_request(50);
        r3.estimated_time_ms = 100;

        RequestScheduler::insert_into_queue_inner(&mut queue, &SchedulingStrategy::SJF, r1);
        RequestScheduler::insert_into_queue_inner(&mut queue, &SchedulingStrategy::SJF, r2);
        RequestScheduler::insert_into_queue_inner(&mut queue, &SchedulingStrategy::SJF, r3);

        assert_eq!(queue[0].estimated_time_ms, 50, "shortest first");
        assert_eq!(queue[1].estimated_time_ms, 100, "middle");
        assert_eq!(queue[2].estimated_time_ms, 200, "longest last");
    }
}
