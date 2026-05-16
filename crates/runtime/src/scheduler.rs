use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;
use tracing::{debug, info, warn};
use chrono::{DateTime, Utc};

use crate::{Result, InferenceError, InferenceRequest, InferenceResponse};

#[derive(Debug, Clone, PartialEq)]
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
    pub response_tx: mpsc::UnboundedSender<InferenceResponse>,
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

pub struct RequestScheduler {
    max_concurrent_requests: usize,
    strategy: SchedulingStrategy,
    request_queue: Arc<RwLock<VecDeque<QueuedRequest>>>,
    active_requests: Arc<RwLock<HashMap<Uuid, RequestInfo>>>,
    request_status: Arc<RwLock<HashMap<Uuid, RequestStatus>>>,
    response_channels: Arc<RwLock<HashMap<Uuid, mpsc::UnboundedSender<InferenceResponse>>>>,
    stats: Arc<RwLock<SchedulerStats>>,
    state: Arc<RwLock<SchedulerState>>,
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
            strategy: SchedulingStrategy::FIFO,
            request_queue: Arc::new(RwLock::new(VecDeque::new())),
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            request_status: Arc::new(RwLock::new(HashMap::new())),
            response_channels: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SchedulerStats::default())),
            state: Arc::new(RwLock::new(SchedulerState::Uninitialized)),
        }
    }

    pub fn with_strategy(mut self, strategy: SchedulingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing request scheduler");
        *self.state.write().await = SchedulerState::Initializing;
        self.stats.write().await.last_updated = Utc::now();
        *self.state.write().await = SchedulerState::Ready;
        info!("Request scheduler initialized successfully");
        Ok(())
    }

    pub async fn submit_request(&self, request: InferenceRequest, response_tx: mpsc::UnboundedSender<InferenceResponse>) -> Result<()> {
        debug!("Submitting request to scheduler: {:?}", request.request_id);

        {
            let state = self.state.read().await;
            if *state != SchedulerState::Ready {
                return Err(InferenceError::InternalError("Scheduler not ready".to_string()).into());
            }
        }

        let request_uuid = request.request_id
            .as_ref()
            .and_then(|s| Uuid::parse_str(s).ok())
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

        self.response_channels.write().await.insert(request_uuid, queued_request.response_tx.clone());
        self.request_status.write().await.insert(request_uuid, RequestStatus::Queued);
        self.insert_into_queue(&mut *self.request_queue.write().await, queued_request);

        {
            let queue_len = self.request_queue.read().await.len();
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
            stats.queued_requests += 1;
            stats.current_queue_length = queue_len;
            stats.last_updated = Utc::now();
        }

        self.process_queue().await?;
        debug!("Request {:?} queued successfully", request_uuid);
        Ok(())
    }

    pub async fn get_next_request(&self) -> Result<Option<QueuedRequest>> {
        debug!("Getting next request from scheduler");

        {
            let active = self.active_requests.read().await;
            if active.len() >= self.max_concurrent_requests {
                debug!("Maximum concurrent requests reached ({})", active.len());
                return Ok(None);
            }
        }

        let mut queue = self.request_queue.write().await;
        if let Some(queued_request) = queue.pop_front() {
            self.stats.write().await.current_queue_length = queue.len();
            Ok(Some(queued_request))
        } else {
            Ok(None)
        }
    }

    pub async fn start_request(&self, request_id: Uuid) -> Result<()> {
        debug!("Starting request processing: {}", request_id);

        self.request_status.write().await.get_mut(&request_id)
            .map(|s| *s = RequestStatus::Processing);

        self.active_requests.write().await.insert(request_id, RequestInfo {
            request_id,
            started_at: Utc::now(),
            processing_time_ms: 0,
            priority: 0,
            session_id: None,
            metadata: HashMap::new(),
        });

        let active_len = self.active_requests.read().await.len();
        self.stats.write().await.current_active_requests = active_len;

        debug!("Request {} started processing", request_id);
        Ok(())
    }

    pub async fn complete_request(&self, request_id: Uuid) -> Result<()> {
        debug!("Completing request: {}", request_id);

        let processing_time = {
            let mut active = self.active_requests.write().await;
            active.remove(&request_id)
                .map(|info| (Utc::now() - info.started_at).num_milliseconds() as u64)
                .unwrap_or_else(|| {
                    warn!("Request {} not found in active requests", request_id);
                    0
                })
        };

        self.request_status.write().await.get_mut(&request_id)
            .map(|s| *s = RequestStatus::Completed);

        let active_len = self.active_requests.read().await.len();
        {
            let mut stats = self.stats.write().await;
            stats.processed_requests += 1;
            stats.current_active_requests = active_len;
            if stats.processed_requests > 0 {
                stats.avg_processing_time_ms = (stats.avg_processing_time_ms * (stats.processed_requests - 1) as f64
                    + processing_time as f64) / stats.processed_requests as f64;
            }
            stats.last_updated = Utc::now();
        }

        self.response_channels.write().await.remove(&request_id);
        self.process_queue().await?;
        debug!("Request {} completed successfully", request_id);
        Ok(())
    }

    pub async fn cancel_request(&self, request_id: Uuid) -> Result<bool> {
        debug!("Cancelling request: {}", request_id);

        let mut cancelled = false;

        {
            let mut queue = self.request_queue.write().await;
            if let Some(pos) = queue.iter().position(|req| {
                req.request.request_id
                    .as_ref()
                    .and_then(|s| Uuid::parse_str(s).ok())
                    .map(|uuid| uuid == request_id)
                    .unwrap_or(false)
            }) {
                queue.remove(pos);
                cancelled = true;
            }
        }

        if !cancelled {
            let mut active = self.active_requests.write().await;
            if active.remove(&request_id).is_some() {
                cancelled = true;
            }
        }

        if cancelled {
            self.request_status.write().await.insert(request_id, RequestStatus::Cancelled);

            let queue_len = self.request_queue.read().await.len();
            let active_len = self.active_requests.read().await.len();
            {
                let mut stats = self.stats.write().await;
                stats.cancelled_requests += 1;
                stats.current_queue_length = queue_len;
                stats.current_active_requests = active_len;
            }

            self.response_channels.write().await.remove(&request_id);
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
            let channels = self.response_channels.read().await;
            channels.get(&request_id).cloned()
        };

        match response_tx {
            Some(tx) => tx.send(response)
                .map_err(|_| InferenceError::InternalError("Response channel closed".to_string()).into()),
            None => Err(InferenceError::InternalError("Response channel not found".to_string()).into()),
        }
    }

    pub async fn get_request_status(&self, request_id: Uuid) -> Result<Option<RequestStatus>> {
        Ok(self.request_status.read().await.get(&request_id).cloned())
    }

    pub async fn get_stats(&self) -> SchedulerStats {
        self.stats.read().await.clone()
    }

    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down request scheduler");

        *self.state.write().await = SchedulerState::ShuttingDown;

        let queued_count = {
            let mut queue = self.request_queue.write().await;
            let count = queue.len();
            queue.clear();
            count
        };

        self.stats.write().await.cancelled_requests += queued_count as u64;

        let active_ids: Vec<Uuid> = {
            let active = self.active_requests.read().await;
            active.keys().cloned().collect()
        };

        for request_id in active_ids {
            let _ = self.cancel_request(request_id).await;
        }

        *self.state.write().await = SchedulerState::Shutdown;
        info!("Request scheduler shutdown complete");
        Ok(())
    }

    async fn process_queue(&self) -> Result<()> {
        while let Some(queued_request) = self.get_next_request().await? {
            self.start_request(
                queued_request.request.request_id
                    .as_ref()
                    .and_then(|s| Uuid::parse_str(s).ok())
                    .unwrap_or_else(Uuid::new_v4)
            ).await?;
        }
        Ok(())
    }

    fn insert_into_queue(&self, queue: &mut VecDeque<QueuedRequest>, request: QueuedRequest) {
        let insert_pos = match self.strategy {
            SchedulingStrategy::FIFO | SchedulingStrategy::RoundRobin | SchedulingStrategy::Fair => None,
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
        let max_tokens = request.parameters.get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(100);
        let streaming = request.parameters.get("streaming")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let temperature = request.parameters.get("temperature")
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
        let max_tokens = request.parameters.get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(100);
        let temperature = request.parameters.get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;
        let temperature_factor = if temperature > 0.0 { 1.5 } else { 1.0 };
        (max_tokens * base_time_per_token * temperature_factor as u64).max(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn test_request(priority: u8) -> QueuedRequest {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        QueuedRequest {
            request: InferenceRequest {
                model_id: "test".into(),
                inputs: vec![],
                parameters: [("max_tokens".to_string(), serde_json::json!(100))]
                    .into_iter().collect(),
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
        let scheduler = RequestScheduler::new(10);
        let mut queue = VecDeque::new();

        scheduler.insert_into_queue(&mut queue, test_request(50));
        scheduler.insert_into_queue(&mut queue, test_request(50));
        scheduler.insert_into_queue(&mut queue, test_request(50));

        assert_eq!(queue.len(), 3);
    }

    #[test]
    fn test_insert_into_queue_priority() {
        let scheduler = RequestScheduler::new(10)
            .with_strategy(SchedulingStrategy::Priority);
        let mut queue = VecDeque::new();

        scheduler.insert_into_queue(&mut queue, test_request(10));
        scheduler.insert_into_queue(&mut queue, test_request(90));
        scheduler.insert_into_queue(&mut queue, test_request(50));

        assert_eq!(queue.len(), 3);
        assert_eq!(queue[0].priority, 90, "highest priority first");
        assert_eq!(queue[1].priority, 50, "middle priority second");
        assert_eq!(queue[2].priority, 10, "lowest priority last");
    }

    #[test]
    fn test_insert_into_queue_sjf() {
        let scheduler = RequestScheduler::new(10)
            .with_strategy(SchedulingStrategy::SJF);

        let mut queue = VecDeque::new();
        let mut r1 = test_request(50); r1.estimated_time_ms = 200;
        let mut r2 = test_request(50); r2.estimated_time_ms = 50;
        let mut r3 = test_request(50); r3.estimated_time_ms = 100;

        scheduler.insert_into_queue(&mut queue, r1);
        scheduler.insert_into_queue(&mut queue, r2);
        scheduler.insert_into_queue(&mut queue, r3);

        assert_eq!(queue[0].estimated_time_ms, 50, "shortest first");
        assert_eq!(queue[1].estimated_time_ms, 100, "middle");
        assert_eq!(queue[2].estimated_time_ms, 200, "longest last");
    }
}
