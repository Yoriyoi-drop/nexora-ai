//! Worker agent persistence — store/restore work items to/from memory layers.
//!
//! Single responsibility: manage durable state for worker agent work items
//! (save on create/complete/retry/fail, restore orphaned work on startup).

use serde_json::json;
use tracing::info;
use uuid::Uuid;

use crate::planner_agent::{PlanStep, StepStatus, StepType};

use super::types::{WorkItem, WorkStatus};
use super::WorkerAgent;

impl WorkerAgent {
    /// Persist a new work item to session-level memory.
    pub(crate) async fn persist_work_item(&self, work: &WorkItem) {
        let store = match &self.memory_store {
            Some(s) => s,
            None => return,
        };
        let key = format!("worker:work:{}", work.work_id);
        let val = serde_json::to_string(&json!({
            "work_id": work.work_id,
            "plan_id": work.plan_id,
            "step_id": work.step.step_id,
            "description": work.step.description,
            "step_type": format!("{:?}", work.step.step_type),
            "status": format!("{:?}", work.status),
            "retry_count": work.retry_count,
        }))
        .unwrap_or_default();
        let _ = store
            .lock()
            .await
            .store(nexora_memory::MemoryLayer::Session, &key, &val)
            .await;
    }

    /// Persist a completed work item to long-term memory.
    pub(crate) async fn persist_work_completed(&self, work_id: &Uuid, output: &serde_json::Value, elapsed: u64) {
        let store = match &self.memory_store {
            Some(s) => s,
            None => return,
        };
        let key = format!("worker:completed:{}", work_id);
        let val = serde_json::to_string(&json!({
            "work_id": work_id,
            "output": output,
            "duration_ms": elapsed,
        }))
        .unwrap_or_default();
        let _ = store
            .lock()
            .await
            .store(nexora_memory::MemoryLayer::Long, &key, &val)
            .await;
    }

    /// Persist a retry event to session-level memory.
    pub(crate) async fn persist_work_retry(&self, work_id: &Uuid, error: &str, attempt: u32) {
        let store = match &self.memory_store {
            Some(s) => s,
            None => return,
        };
        let key = format!("worker:retry:{}", work_id);
        let val = serde_json::to_string(&json!({
            "work_id": work_id,
            "attempt": attempt,
            "error": error,
        }))
        .unwrap_or_default();
        let _ = store
            .lock()
            .await
            .store(nexora_memory::MemoryLayer::Session, &key, &val)
            .await;
    }

    /// Persist a failed work item to long-term memory.
    pub(crate) async fn persist_work_failed(&self, work_id: &Uuid, error: &str) {
        let store = match &self.memory_store {
            Some(s) => s,
            None => return,
        };
        let key = format!("worker:failed:{}", work_id);
        let val = serde_json::to_string(&json!({
            "work_id": work_id,
            "error": error,
        }))
        .unwrap_or_default();
        let _ = store
            .lock()
            .await
            .store(nexora_memory::MemoryLayer::Long, &key, &val)
            .await;
    }

    /// Restore in-flight work items from session memory on startup.
    pub(crate) async fn restore_in_flight_work(&self) -> Vec<WorkItem> {
        let store = match &self.memory_store {
            Some(s) => s,
            None => return Vec::new(),
        };
        let restore_key = "worker:in_flight";
        let mut store_guard = store.lock().await;
        let mut wm = self.active_work.lock().await;
        if let Ok(Some(val)) = store_guard
            .retrieve(nexora_memory::MemoryLayer::Session, restore_key)
            .await
        {
            if let Ok(ids) = serde_json::from_str::<Vec<String>>(&val) {
                for id_str in ids {
                    if let Ok(wid) = Uuid::parse_str(&id_str) {
                        let key = format!("worker:work:{}", wid);
                        if let Ok(Some(entry)) = store_guard
                            .retrieve(nexora_memory::MemoryLayer::Session, &key)
                            .await
                        {
                            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&entry) {
                                if let Some(step_id) = json_val
                                    .get("step_id")
                                    .and_then(|v| v.as_str())
                                    .and_then(|s| Uuid::parse_str(s).ok())
                                {
                                    let plan_id = json_val
                                        .get("plan_id")
                                        .and_then(|v| v.as_str())
                                        .and_then(|s| Uuid::parse_str(s).ok())
                                        .unwrap_or_default();
                                    let desc = json_val
                                        .get("description")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("restored");
                                    let work = WorkItem {
                                        work_id: wid,
                                        plan_id,
                                        step: PlanStep {
                                            step_id,
                                            description: desc.to_string(),
                                            step_type: StepType::Processing,
                                            dependencies: Vec::new(),
                                            required_capabilities: Vec::new(),
                                            estimated_duration_seconds: 60,
                                            status: StepStatus::Pending,
                                            result: None,
                                            error_message: None,
                                        },
                                        status: WorkStatus::Pending,
                                        retry_count: 0,
                                        started_at: chrono::Utc::now(),
                                        last_updated: chrono::Utc::now(),
                                        output: None,
                                        error: None,
                                    };
                                    info!("Restored in-flight work item {}", wid);
                                    wm.insert(wid, work);
                                }
                            }
                        }
                    }
                }
            }
        }
        drop(store_guard);
        wm.values().cloned().collect()
    }
}
