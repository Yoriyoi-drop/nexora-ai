//! Worker Agent — executes individual plan steps via inference engine.
//!
//! Single responsibility: manage the worker agent lifecycle and coordinate
//! step execution (create work items, run with retry/timeout, track results).
//! Inference and persistence details are delegated to submodules.
//!
//! Submodules:
//! - `types`: data types (config, work item, status, results)
//! - `processing`: step-type routing and inference calls
//! - `persistence`: work item save/restore to memory layers

pub mod persistence;
pub mod processing;
pub mod types;

// ── Re-exports for backward compatibility ──────────────────────────────────
pub use types::{build_prompt, StepExecutionResult, WorkItem, WorkStatus, WorkerAgentConfig};

use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{
    Agent, AgentConfig, AgentContext, AgentError, AgentMessage, AgentResponse, AgentStats,
    AgentStatus, Result,
};
use crate::inference_agent::InferenceEngine;
use crate::planner_agent::{PlanStep, StepType};

// ── WorkerAgent ────────────────────────────────────────────────────────────

/// Executes plan steps by routing them through the inference engine.
///
/// Owns:
/// - Work queue (`active_work` map with concurrency cap)
/// - Retry/timout logic for failing steps
/// - Optional inference engine + memory store for persistence
pub struct WorkerAgent {
    id: Uuid,
    name: String,
    status: AgentStatus,
    active_work: Arc<tokio::sync::Mutex<HashMap<Uuid, types::WorkItem>>>,
    stats: tokio::sync::Mutex<AgentStats>,
    config: types::WorkerAgentConfig,
    memory_store: Option<Arc<tokio::sync::Mutex<nexora_memory::MemoryLayers>>>,
    inference_engine: Option<Arc<dyn InferenceEngine>>,
}

impl WorkerAgent {
    pub fn new(config: types::WorkerAgentConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "WorkerAgent".to_string(),
            status: AgentStatus::Initializing,
            active_work: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            stats: tokio::sync::Mutex::new(AgentStats::default()),
            config,
            memory_store: None,
            inference_engine: None,
        }
    }

    pub fn with_memory_store(
        mut self,
        store: Arc<tokio::sync::Mutex<nexora_memory::MemoryLayers>>,
    ) -> Self {
        self.memory_store = Some(store);
        self
    }

    pub fn with_inference_engine(mut self, engine: Arc<dyn InferenceEngine>) -> Self {
        self.inference_engine = Some(engine);
        self
    }

    /// Create a new work item and add it to the active queue.
    pub async fn execute_step(&self, plan_id: Uuid, step: PlanStep) -> Result<types::WorkItem> {
        let work_id = Uuid::new_v4();
        let now = chrono::Utc::now();

        let work = types::WorkItem {
            work_id,
            plan_id,
            step: step.clone(),
            status: types::WorkStatus::Pending,
            retry_count: 0,
            started_at: now,
            last_updated: now,
            output: None,
            error: None,
        };

        {
            let mut wm = self.active_work.lock().await;
            if wm.len() >= self.config.max_concurrent_work {
                return Err(AgentError::ProcessingError {
                    operation: "execute_step".to_string(),
                    reason: "Max concurrent work items reached".to_string(),
                });
            }
            wm.insert(work_id, work.clone());
        }

        if self.config.enable_persistence {
            self.persist_work_item(&work).await;
        }

        debug!(
            "Work item {} created for plan {} step {}",
            work_id, plan_id, step.step_id
        );
        Ok(work)
    }

    /// Execute a work item with retry/timeout loop.
    pub async fn run_step(&self, work_id: Uuid) -> Result<types::StepExecutionResult> {
        let start = std::time::Instant::now();
        loop {
            let work = {
                let mut wm = self.active_work.lock().await;
                let w = wm.get_mut(&work_id).ok_or_else(|| {
                    AgentError::ProcessingError {
                        operation: "run_step".to_string(),
                        reason: format!("Work item {} not found", work_id),
                    }
                })?;
                w.status = types::WorkStatus::Running;
                w.last_updated = chrono::Utc::now();
                w.clone()
            };

            let result = tokio::time::timeout(
                Duration::from_secs(self.config.default_timeout_seconds),
                self.process_step_inner(&work),
            )
            .await;

            let elapsed = start.elapsed().as_millis() as u64;

            match result {
                Ok(Ok(output)) => {
                    let mut wm = self.active_work.lock().await;
                    if let Some(w) = wm.get_mut(&work_id) {
                        w.status = types::WorkStatus::Completed;
                        w.output = Some(output.clone());
                        w.last_updated = chrono::Utc::now();
                    }
                    if self.config.enable_persistence {
                        self.persist_work_completed(&work_id, &output, elapsed)
                            .await;
                    }
                    self.stats.lock().await.messages_processed += 1;
                    return Ok(types::StepExecutionResult {
                        step_id: work.step.step_id,
                        plan_id: work.plan_id,
                        success: true,
                        output,
                        error: None,
                        duration_ms: elapsed,
                    });
                }
                Ok(Err(_)) | Err(_) => {
                    let error = match &result {
                        Ok(Err(e)) => e.to_string(),
                        _ => "Step timed out".to_string(),
                    };
                    return self
                        .handle_step_failure(&work, work_id, error, elapsed)
                        .await;
                }
            }
        }
    }

    /// Handle a step failure — retry or mark as failed.
    async fn handle_step_failure(
        &self,
        work: &types::WorkItem,
        work_id: Uuid,
        error: String,
        elapsed: u64,
    ) -> Result<types::StepExecutionResult> {
        let mut wm = self.active_work.lock().await;
        if let Some(w) = wm.get_mut(&work_id) {
            w.error = Some(error.clone());
            w.last_updated = chrono::Utc::now();

            if w.retry_count < self.config.max_retries {
                w.retry_count += 1;
                w.status = types::WorkStatus::Pending;
                w.last_updated = chrono::Utc::now();
                warn!(
                    "Work {} failed (attempt {}/{}), will retry: {}",
                    work_id, w.retry_count, self.config.max_retries, error
                );
                if self.config.enable_persistence {
                    self.persist_work_retry(&work_id, &error, w.retry_count)
                        .await;
                }
                drop(wm);
                tokio::time::sleep(Duration::from_millis(self.config.retry_delay_ms)).await;
                return self.run_step(work_id).await;
            } else {
                w.status = types::WorkStatus::Failed(error.clone());
                if self.config.enable_persistence {
                    self.persist_work_failed(&work_id, &error).await;
                }
            }
        }
        self.stats.lock().await.errors += 1;
        Ok(types::StepExecutionResult {
            step_id: work.step.step_id,
            plan_id: work.plan_id,
            success: false,
            output: json!(null),
            error: Some(error),
            duration_ms: elapsed,
        })
    }

    pub async fn get_work(&self, work_id: Uuid) -> Result<Option<types::WorkItem>> {
        let wm = self.active_work.lock().await;
        Ok(wm.get(&work_id).cloned())
    }

    pub async fn list_active_work(&self) -> Vec<types::WorkItem> {
        let wm = self.active_work.lock().await;
        wm.values().cloned().collect()
    }

    pub async fn cancel_work(&self, work_id: Uuid) -> Result<()> {
        let mut wm = self.active_work.lock().await;
        if let Some(w) = wm.get_mut(&work_id) {
            w.status = types::WorkStatus::Cancelled;
            w.last_updated = chrono::Utc::now();
            info!("Work {} cancelled", work_id);
            Ok(())
        } else {
            Err(AgentError::ProcessingError {
                operation: "cancel_work".to_string(),
                reason: format!("Work item {} not found", work_id),
            })
        }
    }
}

// ── Agent trait implementation ─────────────────────────────────────────────

#[async_trait]
impl Agent for WorkerAgent {
    fn id(&self) -> Uuid {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn agent_type(&self) -> &str {
        "worker"
    }

    fn status(&self) -> AgentStatus {
        self.status.clone()
    }

    async fn initialize(&mut self, _config: AgentConfig) -> Result<()> {
        info!("Initializing WorkerAgent");
        if self.config.enable_persistence {
            self.restore_in_flight_work().await;
        }
        self.status = AgentStatus::Ready;
        Ok(())
    }

    async fn receive(&mut self, message: AgentMessage) -> Result<()> {
        debug!("WorkerAgent received message: {}", message.message_type);
        Ok(())
    }

    async fn process(&mut self, context: AgentContext) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();

        let action = context
            .parameters
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("status");

        let result = match action {
            "execute_step" => {
                let plan_id_str = context
                    .parameters
                    .get("plan_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError {
                        operation: "validate".to_string(),
                        reason: "plan_id required".to_string(),
                    })?;
                let step_id_str = context
                    .parameters
                    .get("step_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError {
                        operation: "validate".to_string(),
                        reason: "step_id required".to_string(),
                    })?;
                let description = context
                    .parameters
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Execute step");
                let step_type_str = context
                    .parameters
                    .get("step_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Processing");

                let plan_id = Uuid::parse_str(plan_id_str).map_err(|_| {
                    AgentError::ProcessingError {
                        operation: "parse".to_string(),
                        reason: "Invalid plan_id".to_string(),
                    }
                })?;
                let step_id = Uuid::parse_str(step_id_str).map_err(|_| {
                    AgentError::ProcessingError {
                        operation: "parse".to_string(),
                        reason: "Invalid step_id".to_string(),
                    }
                })?;

                let step_type = match step_type_str {
                    "DataCollection" => StepType::DataCollection,
                    "Analysis" => StepType::Analysis,
                    "Processing" => StepType::Processing,
                    "Generation" => StepType::Generation,
                    "Validation" => StepType::Validation,
                    "Communication" => StepType::Communication,
                    "Decision" => StepType::Decision,
                    _ => StepType::Custom(step_type_str.to_string()),
                };

                let step = PlanStep {
                    step_id,
                    description: description.to_string(),
                    step_type,
                    dependencies: Vec::new(),
                    required_capabilities: Vec::new(),
                    estimated_duration_seconds: 60,
                    status: crate::planner_agent::StepStatus::Pending,
                    result: None,
                    error_message: None,
                };

                let work = self.execute_step(plan_id, step).await?;
                let result = self.run_step(work.work_id).await?;

                json!({
                    "action": "execute_step",
                    "work_id": work.work_id,
                    "step_id": step_id,
                    "plan_id": plan_id,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                    "duration_ms": result.duration_ms,
                })
            }

            "get_work" => {
                let work_id_str = context
                    .parameters
                    .get("work_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError {
                        operation: "validate".to_string(),
                        reason: "work_id required".to_string(),
                    })?;
                let work_id = Uuid::parse_str(work_id_str).map_err(|_| {
                    AgentError::ProcessingError {
                        operation: "parse".to_string(),
                        reason: "Invalid work_id".to_string(),
                    }
                })?;
                let work = self.get_work(work_id).await?;

                match work {
                    Some(w) => json!({
                        "work_id": w.work_id,
                        "plan_id": w.plan_id,
                        "step_id": w.step.step_id,
                        "status": format!("{:?}", w.status),
                        "output": w.output,
                        "error": w.error,
                    }),
                    None => json!({"error": "Work item not found"}),
                }
            }

            "list_work" => {
                let work = self.list_active_work().await;
                let items: Vec<Value> = work
                    .iter()
                    .map(|w| {
                        json!({
                            "work_id": w.work_id,
                            "plan_id": w.plan_id,
                            "step_id": w.step.step_id,
                            "status": format!("{:?}", w.status),
                        })
                    })
                    .collect();
                json!({"work_items": items, "count": items.len()})
            }

            "cancel_work" => {
                let work_id_str = context
                    .parameters
                    .get("work_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError {
                        operation: "validate".to_string(),
                        reason: "work_id required".to_string(),
                    })?;
                let work_id = Uuid::parse_str(work_id_str).map_err(|_| {
                    AgentError::ProcessingError {
                        operation: "parse".to_string(),
                        reason: "Invalid work_id".to_string(),
                    }
                })?;
                self.cancel_work(work_id).await?;
                json!({"action": "cancel_work", "work_id": work_id, "status": "cancelled"})
            }

            _ => {
                return Err(AgentError::ProcessingError {
                    operation: "execute_action".to_string(),
                    reason: format!("Unknown action: {}", action),
                });
            }
        };

        let processing_time = start_time.elapsed().as_millis() as u64;
        {
            let mut s = self.stats.lock().await;
            s.messages_processed += 1;
            s.avg_processing_time_ms = (s.avg_processing_time_ms * (s.messages_processed - 1) as f64
                + processing_time as f64)
                / s.messages_processed as f64;
            s.last_activity = chrono::Utc::now();
        }

        Ok(AgentResponse::success(
            context.session_id,
            result,
            processing_time,
        ))
    }

    async fn respond(&mut self, _response: AgentResponse) -> Result<()> {
        debug!("WorkerAgent sending response");
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down WorkerAgent");
        let work_ids: Vec<Uuid> = {
            let wm = self.active_work.lock().await;
            wm.keys().cloned().collect()
        };
        for wid in work_ids {
            if let Err(e) = self.cancel_work(wid).await {
                warn!("Failed to cancel work {}: {}", wid, e);
            }
        }
        if self.config.enable_persistence {
            let store = match &self.memory_store {
                Some(s) => s,
                None => {
                    self.status = AgentStatus::Shutdown;
                    return Ok(());
                }
            };
            let wm = self.active_work.lock().await;
            let ids: Vec<String> = wm.keys().map(|k| k.to_string()).collect();
            let _ = store
                .lock()
                .await
                .store(
                    nexora_memory::MemoryLayer::Session,
                    "worker:in_flight",
                    &serde_json::to_string(&ids).unwrap_or_default(),
                )
                .await;
        }
        self.status = AgentStatus::Shutdown;
        Ok(())
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    fn get_stats(&self) -> AgentStats {
        self.stats
            .try_lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    fn get_config(&self) -> AgentConfig {
        AgentConfig {
            agent_id: "worker_agent".to_string(),
            agent_type: "worker".to_string(),
            max_concurrent_tasks: self.config.max_concurrent_work,
            timeout_seconds: self.config.default_timeout_seconds,
        }
    }
}

impl From<types::WorkerAgentConfig> for AgentConfig {
    fn from(config: types::WorkerAgentConfig) -> Self {
        AgentConfig {
            agent_id: "worker_agent".to_string(),
            agent_type: "worker".to_string(),
            max_concurrent_tasks: config.max_concurrent_work,
            timeout_seconds: config.default_timeout_seconds,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner_agent::{StepStatus, StepType};

    #[test]
    fn test_worker_agent_config_default() {
        let config = types::WorkerAgentConfig::default();
        assert_eq!(config.max_concurrent_work, 50);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_delay_ms, 1000);
        assert!(config.enable_persistence);
    }

    #[test]
    fn test_work_item_creation() {
        let now = chrono::Utc::now();
        let step = PlanStep {
            step_id: Uuid::new_v4(),
            description: "Test step".to_string(),
            step_type: StepType::Analysis,
            dependencies: Vec::new(),
            required_capabilities: vec!["analysis".to_string()],
            estimated_duration_seconds: 30,
            status: StepStatus::Pending,
            result: None,
            error_message: None,
        };
        let work = types::WorkItem {
            work_id: Uuid::new_v4(),
            plan_id: Uuid::new_v4(),
            step,
            status: types::WorkStatus::Pending,
            retry_count: 0,
            started_at: now,
            last_updated: now,
            output: None,
            error: None,
        };
        assert_eq!(work.status, types::WorkStatus::Pending);
        assert!(work.output.is_none());
        assert_eq!(work.retry_count, 0);
    }

    #[test]
    fn test_work_status_variants() {
        assert!(matches!(types::WorkStatus::Pending, types::WorkStatus::Pending));
        assert!(matches!(types::WorkStatus::Running, types::WorkStatus::Running));
        assert!(matches!(types::WorkStatus::Completed, types::WorkStatus::Completed));
        assert!(matches!(types::WorkStatus::Failed("x".into()), types::WorkStatus::Failed(_)));
        assert!(matches!(types::WorkStatus::Cancelled, types::WorkStatus::Cancelled));
    }

    #[test]
    fn test_step_execution_result() {
        let result = types::StepExecutionResult {
            step_id: Uuid::new_v4(),
            plan_id: Uuid::new_v4(),
            success: true,
            output: json!({"result": "ok"}),
            error: None,
            duration_ms: 100,
        };
        assert!(result.success);
        assert!(result.error.is_none());
        assert_eq!(result.duration_ms, 100);
    }

    #[test]
    fn test_worker_agent_new() {
        let agent = WorkerAgent::new(types::WorkerAgentConfig::default());
        assert_eq!(agent.name(), "WorkerAgent");
        assert_eq!(agent.agent_type(), "worker");
        assert_eq!(agent.status(), AgentStatus::Initializing);
    }

    #[tokio::test]
    async fn test_worker_process_step_types() {
        let agent = WorkerAgent::new(types::WorkerAgentConfig::default());
        let now = chrono::Utc::now();

        let types_list = vec![
            StepType::Analysis,
            StepType::Processing,
            StepType::Validation,
            StepType::DataCollection,
            StepType::Generation,
            StepType::Communication,
            StepType::Decision,
            StepType::Custom("test".to_string()),
        ];

        for st in types_list {
            let step = PlanStep {
                step_id: Uuid::new_v4(),
                description: format!("Test {:?}", st),
                step_type: st,
                dependencies: Vec::new(),
                required_capabilities: Vec::new(),
                estimated_duration_seconds: 10,
                status: StepStatus::Pending,
                result: None,
                error_message: None,
            };
            let work = types::WorkItem {
                work_id: Uuid::new_v4(),
                plan_id: Uuid::new_v4(),
                step,
                status: types::WorkStatus::Pending,
                retry_count: 0,
                started_at: now,
                last_updated: now,
                output: None,
                error: None,
            };
            let work_id = work.work_id;
            agent.active_work.lock().await.insert(work_id, work);
            let result = agent.run_step(work_id).await;
            assert!(result.is_ok(), "Step type should succeed");
            let r = result.unwrap();
            assert!(r.success, "Step should succeed");
        }
    }

    #[tokio::test]
    async fn test_worker_max_concurrent() {
        let config = types::WorkerAgentConfig {
            max_concurrent_work: 1,
            ..Default::default()
        };
        let agent = WorkerAgent::new(config);
        let now = chrono::Utc::now();

        let step1 = PlanStep {
            step_id: Uuid::new_v4(),
            description: "step1".to_string(),
            step_type: StepType::Processing,
            dependencies: Vec::new(),
            required_capabilities: Vec::new(),
            estimated_duration_seconds: 10,
            status: StepStatus::Pending,
            result: None,
            error_message: None,
        };
        let step2 = PlanStep {
            step_id: Uuid::new_v4(),
            description: "step2".to_string(),
            step_type: StepType::Processing,
            dependencies: Vec::new(),
            required_capabilities: Vec::new(),
            estimated_duration_seconds: 10,
            status: StepStatus::Pending,
            result: None,
            error_message: None,
        };

        assert!(agent.execute_step(Uuid::new_v4(), step1).await.is_ok());
        assert!(agent.execute_step(Uuid::new_v4(), step2).await.is_err());
    }
}
