//! Worker agent data types.
//!
//! Single responsibility: define all data structures used by the worker agent
//! (config, work items, status enums, execution results).

use serde_json::Value;
use uuid::Uuid;

use crate::planner_agent::PlanStep;

/// Configuration for a worker agent.
#[derive(Debug, Clone)]
pub struct WorkerAgentConfig {
    pub max_concurrent_work: usize,
    pub default_timeout_seconds: u64,
    pub max_retries: u32,
    pub retry_delay_ms: u64,
    pub enable_persistence: bool,
}

impl Default for WorkerAgentConfig {
    fn default() -> Self {
        Self {
            max_concurrent_work: 50,
            default_timeout_seconds: 60,
            max_retries: 3,
            retry_delay_ms: 1000,
            enable_persistence: true,
        }
    }
}

/// Represents a single unit of work assigned to a worker agent.
#[derive(Debug, Clone)]
pub struct WorkItem {
    pub work_id: Uuid,
    pub plan_id: Uuid,
    pub step: PlanStep,
    pub status: WorkStatus,
    pub retry_count: u32,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub output: Option<Value>,
    pub error: Option<String>,
}

/// Status of a work item.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum WorkStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

/// Result of executing a single plan step.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepExecutionResult {
    pub step_id: Uuid,
    pub plan_id: Uuid,
    pub success: bool,
    pub output: Value,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Build a prompt string with pre-allocated capacity to reduce heap churn.
pub fn build_prompt(prefix: &str, description: &str) -> String {
    let cap = prefix.len() + description.len() + 4;
    let mut s = String::with_capacity(cap);
    s.push_str(prefix);
    s.push_str(description);
    s
}
