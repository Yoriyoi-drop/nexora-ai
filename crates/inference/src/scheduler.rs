use std::collections::{HashMap, VecDeque};
use uuid::Uuid;
use tokio::sync::RwLock;

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
}

struct ScheduledRequest {
    request_id: Uuid,
    response_tx: tokio::sync::mpsc::UnboundedSender<crate::InferenceResponse>,
    status: RequestStatus,
    submitted_at: chrono::DateTime<chrono::Utc>,
}

pub struct RequestScheduler {
    queue: RwLock<VecDeque<Uuid>>,
    requests: RwLock<HashMap<Uuid, ScheduledRequest>>,
    max_concurrent: usize,
    active_count: RwLock<usize>,
}

impl RequestScheduler {
    pub fn new() -> Self {
        Self {
            queue: RwLock::new(VecDeque::new()),
            requests: RwLock::new(HashMap::new()),
            max_concurrent: 4,
            active_count: RwLock::new(0),
        }
    }

    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    pub async fn initialize(&mut self) -> Result<(), anyhow::Error> {
        Ok(())
    }

    pub async fn submit_request(
        &self,
        request_id: Uuid,
        response_tx: tokio::sync::mpsc::UnboundedSender<crate::InferenceResponse>,
    ) -> Result<(), anyhow::Error> {
        let mut requests = self.requests.write().await;
        let mut queue = self.queue.write().await;

        requests.insert(request_id, ScheduledRequest {
            request_id,
            response_tx,
            status: RequestStatus::Queued,
            submitted_at: chrono::Utc::now(),
        });
        queue.push_back(request_id);

        self.try_process_next().await;
        Ok(())
    }

    async fn try_process_next(&self) {
        let active = *self.active_count.read().await;
        if active >= self.max_concurrent {
            return;
        }

        let mut queue = self.queue.write().await;
        if let Some(request_id) = queue.pop_front() {
            let requests = self.requests.read().await;
            if let Some(req) = requests.get(&request_id) {
                if req.status == RequestStatus::Queued {
                    drop(requests);
                    drop(queue);
                    let mut active = self.active_count.write().await;
                    *active += 1;
                    self.update_status(request_id, RequestStatus::Processing).await;
                }
            }
        }
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
        let mut stats = SchedulerStats {
            total_requests: requests.len(),
            active_requests: 0,
            queued_requests: 0,
            completed_requests: 0,
            failed_requests: 0,
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

    pub async fn send_response(&self, request_id: Uuid, response: crate::InferenceResponse) -> Result<(), anyhow::Error> {
        let requests = self.requests.read().await;
        if let Some(req) = requests.get(&request_id) {
            req.response_tx.send(response).map_err(|e| anyhow::anyhow!("Failed to send response: {}", e))?;
        }
        Ok(())
    }

    pub async fn complete_request(&self, request_id: Uuid) -> Result<(), anyhow::Error> {
        self.update_status(request_id, RequestStatus::Completed).await;
        let mut active = self.active_count.write().await;
        if *active > 0 {
            *active -= 1;
        }
        drop(active);
        self.try_process_next().await;
        Ok(())
    }
}

impl Default for RequestScheduler {
    fn default() -> Self {
        Self::new()
    }
}
