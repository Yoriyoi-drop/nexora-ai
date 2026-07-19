use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type EventId = Uuid;
pub type TopicName = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventPriority {
    Low,
    Normal,
    High,
    Critical,
}

impl Default for EventPriority {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub topic: TopicName,
    pub priority: EventPriority,
    pub timestamp: DateTime<Utc>,
    pub ttl_ms: Option<u64>,
    pub source: String,
    pub payload: serde_json::Value,
}

impl Event {
    pub fn new(topic: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            topic: topic.into(),
            priority: EventPriority::Normal,
            timestamp: Utc::now(),
            ttl_ms: None,
            source: String::new(),
            payload,
        }
    }

    pub fn with_priority(mut self, p: EventPriority) -> Self {
        self.priority = p;
        self
    }

    pub fn with_ttl(mut self, ms: u64) -> Self {
        self.ttl_ms = Some(ms);
        self
    }

    pub fn with_source(mut self, src: impl Into<String>) -> Self {
        self.source = src.into();
        self
    }

    pub fn is_expired(&self) -> bool {
        self.ttl_ms.map_or(false, |ttl| {
            let elapsed = Utc::now() - self.timestamp;
            elapsed.num_milliseconds() as u64 > ttl
        })
    }
}

pub mod topics {
    pub const SCHEDULER_TASK_SUBMITTED: &str = "scheduler.task.submitted";
    pub const SCHEDULER_TASK_COMPLETED: &str = "scheduler.task.completed";
    pub const SCHEDULER_TASK_FAILED: &str = "scheduler.task.failed";
    pub const SCHEDULER_TASK_CANCELLED: &str = "scheduler.task.cancelled";
    pub const GPU_KERNEL_QUEUED: &str = "gpu.kernel.queued";
    pub const GPU_KERNEL_COMPLETED: &str = "gpu.kernel.completed";
    pub const GPU_OOM: &str = "gpu.oom";
    pub const MEMORY_POOL_LOW: &str = "memory.pool.low";
    pub const MEMORY_POOL_CRITICAL: &str = "memory.pool.critical";
    pub const MEMORY_POOL_RELEASED: &str = "memory.pool.released";
    pub const AGENT_SPAWNED: &str = "agent.spawned";
    pub const AGENT_STOPPED: &str = "agent.stopped";
    pub const AGENT_SCALED: &str = "agent.scaled";
    pub const AGENT_HEALTH_CHANGE: &str = "agent.health.change";
    pub const CACHE_HIT: &str = "cache.hit";
    pub const CACHE_MISS: &str = "cache.miss";
    pub const CACHE_EVICTION: &str = "cache.eviction";
    pub const COST_OPTIMIZER_ROUTE: &str = "cost.optimizer.route";
    pub const COST_OPTIMIZER_SAVED: &str = "cost.optimizer.saved";
    pub const SYSTEM_STARTUP: &str = "system.startup";
    pub const SYSTEM_SHUTDOWN: &str = "system.shutdown";
    pub const SYSTEM_HEALTH: &str = "system.health";
    pub const SYSTEM_ERROR: &str = "system.error";
    pub const OBSERVABILITY_METRIC: &str = "observability.metric";
    pub const OBSERVABILITY_ALERT: &str = "observability.alert";
    pub const ALL: &str = "*";
}
