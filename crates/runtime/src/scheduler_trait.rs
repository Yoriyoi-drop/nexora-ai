use async_trait::async_trait;
use uuid::Uuid;

/// Shared scheduler interface for both inference and runtime schedulers
#[async_trait]
pub trait Scheduler: Send + Sync {
    /// Submit a request and get back a response channel
    async fn submit(
        &self,
        request_id: Uuid,
        response_tx: tokio::sync::mpsc::Sender<crate::InferenceResponse>,
    ) -> crate::Result<()>;

    /// Cancel a queued or in-flight request
    async fn cancel(&self, request_id: Uuid) -> crate::Result<bool>;

    /// Get the status of a request
    async fn status(&self, request_id: Uuid) -> crate::Result<Option<crate::RequestStatus>>;

    /// Get scheduler statistics
    async fn stats(&self) -> crate::SchedulerStats;

    /// Graceful shutdown
    async fn shutdown(&self) -> crate::Result<()>;
}
