//! Scheduler Backend Trait
//!
//! Defines the interface for scheduling implementations

use async_trait::async_trait;
use uuid::Uuid;
use std::time::Duration;
use crate::FoundationResult;

/// Task to be scheduled
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    pub id: Uuid,
    pub priority: TaskPriority,
    pub payload: Vec<u8>,
    pub metadata: TaskMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

#[derive(Debug, Clone)]
pub struct TaskMetadata {
    pub timeout: Option<Duration>,
    pub retries: u32,
    pub dependencies: Vec<Uuid>,
}

/// Scheduling result
#[derive(Debug, Clone)]
pub struct ScheduleResult {
    pub task_id: Uuid,
    pub scheduled_at: i64,
    pub estimated_completion: i64,
}

/// Core scheduler backend trait
#[async_trait]
pub trait SchedulerBackend: Send + Sync {
    /// Schedule a task
    async fn schedule(&self, task: ScheduledTask) -> FoundationResult<ScheduleResult>;
    
    /// Cancel a scheduled task
    async fn cancel(&self, task_id: Uuid) -> FoundationResult<bool>;
    
    /// Get task status
    async fn status(&self, task_id: Uuid) -> FoundationResult<TaskStatus>;
    
    /// List pending tasks
    async fn list_pending(&self, limit: usize) -> FoundationResult<Vec<ScheduledTask>>;
    
    /// Get scheduler statistics
    async fn stats(&self) -> FoundationResult<SchedulerStats>;
}

#[derive(Debug, Clone)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub pending_count: usize,
    pub running_count: usize,
    pub completed_count: usize,
    pub failed_count: usize,
}
