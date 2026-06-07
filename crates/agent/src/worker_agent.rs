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

/// Helper: pre-allocate prompt string with known capacity to reduce heap churn.
fn build_prompt(prefix: &str, description: &str) -> String {
    let cap = prefix.len() + description.len() + 4;
    let mut s = String::with_capacity(cap);
    s.push_str(prefix);
    s.push_str(description);
    s
}

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

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum WorkStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepExecutionResult {
    pub step_id: Uuid,
    pub plan_id: Uuid,
    pub success: bool,
    pub output: Value,
    pub error: Option<String>,
    pub duration_ms: u64,
}

pub struct WorkerAgent {
    id: Uuid,
    name: String,
    status: AgentStatus,
    active_work: Arc<tokio::sync::Mutex<HashMap<Uuid, WorkItem>>>,
    stats: tokio::sync::Mutex<AgentStats>,
    config: WorkerAgentConfig,
    memory_store: Option<Arc<tokio::sync::Mutex<nexora_memory::MemoryLayers>>>,
    inference_engine: Option<Arc<dyn InferenceEngine>>,
}

impl WorkerAgent {
    pub fn new(config: WorkerAgentConfig) -> Self {
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

    pub fn with_memory_store(mut self, store: Arc<tokio::sync::Mutex<nexora_memory::MemoryLayers>>) -> Self {
        self.memory_store = Some(store);
        self
    }

    pub fn with_inference_engine(mut self, engine: Arc<dyn InferenceEngine>) -> Self {
        self.inference_engine = Some(engine);
        self
    }

    pub async fn execute_step(
        &self,
        plan_id: Uuid,
        step: PlanStep,
    ) -> Result<WorkItem> {
        let work_id = Uuid::new_v4();
        let now = chrono::Utc::now();

        let work = WorkItem {
            work_id,
            plan_id,
            step: step.clone(),
            status: WorkStatus::Pending,
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

        debug!("Work item {} created for plan {} step {}", work_id, plan_id, step.step_id);
        Ok(work)
    }

    pub async fn run_step(&self, work_id: Uuid) -> Result<StepExecutionResult> {
        let start = std::time::Instant::now();
        loop {
            let work = {
                let mut wm = self.active_work.lock().await;
                let w = wm.get_mut(&work_id).ok_or_else(|| AgentError::ProcessingError {
                    operation: "run_step".to_string(),
                    reason: format!("Work item {} not found", work_id),
                })?;
                w.status = WorkStatus::Running;
                w.last_updated = chrono::Utc::now();
                w.clone()
            };

            let result = tokio::time::timeout(
                Duration::from_secs(self.config.default_timeout_seconds),
                self.process_step_inner(&work),
            ).await;

            let elapsed = start.elapsed().as_millis() as u64;

            match result {
                Ok(Ok(output)) => {
                    let mut wm = self.active_work.lock().await;
                    if let Some(w) = wm.get_mut(&work_id) {
                        w.status = WorkStatus::Completed;
                        w.output = Some(output.clone());
                        w.last_updated = chrono::Utc::now();
                    }
                    if self.config.enable_persistence {
                        self.persist_work_completed(&work_id, &output, elapsed).await;
                    }
                    self.stats.lock().await.messages_processed += 1;
                    return Ok(StepExecutionResult {
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

                    let mut wm = self.active_work.lock().await;
                    if let Some(w) = wm.get_mut(&work_id) {
                        w.error = Some(error.clone());
                        w.last_updated = chrono::Utc::now();

                        if w.retry_count < self.config.max_retries {
                            w.retry_count += 1;
                            w.status = WorkStatus::Pending;
                            w.last_updated = chrono::Utc::now();
                            warn!(
                                "Work {} failed (attempt {}/{}), will retry: {}",
                                work_id, w.retry_count, self.config.max_retries, error
                            );
                            if self.config.enable_persistence {
                                self.persist_work_retry(&work_id, &error, w.retry_count).await;
                            }
                            drop(wm);
                            tokio::time::sleep(Duration::from_millis(self.config.retry_delay_ms)).await;
                            continue;
                        } else {
                            w.status = WorkStatus::Failed(error.clone());
                            if self.config.enable_persistence {
                                self.persist_work_failed(&work_id, &error).await;
                            }
                        }
                    }
                    self.stats.lock().await.errors += 1;
                    return Ok(StepExecutionResult {
                        step_id: work.step.step_id,
                        plan_id: work.plan_id,
                        success: false,
                        output: json!(null),
                        error: Some(error),
                        duration_ms: elapsed,
                    });
                }
            }
        }
    }

    /// Call inference engine or return fallback message when engine unavailable.
    async fn infer_or_fallback(
        engine: Option<&Arc<dyn InferenceEngine>>,
        operation: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<String> {
        match engine {
            Some(eng) => eng
                .generate_tokens(Uuid::new_v4(), prompt, max_tokens)
                .await
                .map_err(|e| AgentError::ProcessingError {
                    operation: operation.into(),
                    reason: e.to_string(),
                }),
            None => Ok(format!("[no engine] {} step deferred: engine unavailable", operation)),
        }
    }

    #[tracing::instrument(skip(self, work), fields(step_type = ?work.step.step_type, description = %work.step.description))]
    async fn process_step_inner(&self, work: &WorkItem) -> Result<Value> {
        let engine = self.inference_engine.as_ref();

        let desc = &work.step.description;
        match work.step.step_type {
            StepType::Generation => {
                let prompt = build_prompt("Generate content for: ", desc);
                let result = Self::infer_or_fallback(engine, "generate", &prompt, 512).await?;
                Ok(json!({
                    "type": "generation",
                    "description": desc,
                    "result": result,
                }))
            }
            StepType::Analysis => {
                let prompt = build_prompt("Analyze the following: ", desc);
                let result = Self::infer_or_fallback(engine, "analysis", &prompt, 256).await?;
                Ok(json!({
                    "type": "analysis",
                    "description": desc,
                    "result": result,
                }))
            }
            StepType::Processing => {
                let prompt = build_prompt("Process the following: ", desc);
                let result = Self::infer_or_fallback(engine, "processing", &prompt, 256).await?;
                Ok(json!({
                    "type": "processing",
                    "description": desc,
                    "result": result,
                }))
            }
            StepType::Validation => {
                let prompt = {
                    let pfx = "Validate the following and respond with 'valid' or 'invalid': ";
                    let cap = pfx.len() + desc.len() + 4;
                    let mut s = String::with_capacity(cap);
                    s.push_str(pfx);
                    s.push_str(desc);
                    s
                };
                let result = Self::infer_or_fallback(engine, "validation", &prompt, 128).await?;
                let valid = result.to_lowercase().contains("valid");
                Ok(json!({
                    "type": "validation",
                    "description": desc,
                    "result": result,
                    "valid": valid,
                }))
            }
            StepType::DataCollection => {
                let prompt = build_prompt("Collect data about: ", desc);
                let result = Self::infer_or_fallback(engine, "data_collection", &prompt, 256).await?;
                Ok(json!({
                    "type": "data_collection",
                    "description": desc,
                    "result": result,
                }))
            }
            StepType::Communication => {
                let prompt = build_prompt("Compose a communication about: ", desc);
                let result = Self::infer_or_fallback(engine, "communication", &prompt, 256).await?;
                Ok(json!({
                    "type": "communication",
                    "description": desc,
                    "result": result,
                }))
            }
            StepType::Decision => {
                let prompt = {
                    let pfx = "Make a decision about: ";
                    let sfx = ". Respond with approved or rejected.";
                    let cap = pfx.len() + desc.len() + sfx.len() + 4;
                    let mut s = String::with_capacity(cap);
                    s.push_str(pfx);
                    s.push_str(desc);
                    s.push_str(sfx);
                    s
                };
                let result = Self::infer_or_fallback(engine, "decision", &prompt, 64).await?;
                let decision = if result.to_lowercase().contains("approve") { "approved" } else { "rejected" };
                Ok(json!({
                    "type": "decision",
                    "description": desc,
                    "decision": decision,
                    "reasoning": result,
                }))
            }
            StepType::Custom(ref label) => {
                let prompt = {
                    let pfx = "Execute custom step '";
                    let mid = "': ";
                    let cap = pfx.len() + label.len() + mid.len() + desc.len() + 4;
                    let mut s = String::with_capacity(cap);
                    s.push_str(pfx);
                    s.push_str(label);
                    s.push_str(mid);
                    s.push_str(desc);
                    s
                };
                let result = Self::infer_or_fallback(engine, "custom", &prompt, 256).await?;
                Ok(json!({
                    "type": "custom",
                    "label": label,
                    "description": desc,
                    "result": result,
                }))
            }
        }
    }

    async fn handle_step_failure(
        &self,
        work: &WorkItem,
        work_id: Uuid,
        error: String,
        elapsed: u64,
    ) -> Result<StepExecutionResult> {
        let mut wm = self.active_work.lock().await;
        if let Some(w) = wm.get_mut(&work_id) {
            w.error = Some(error.clone());
            w.last_updated = chrono::Utc::now();

            if w.retry_count < self.config.max_retries {
                w.retry_count += 1;
                w.status = WorkStatus::Pending;
                w.last_updated = chrono::Utc::now();
                warn!(
                    "Work {} failed (attempt {}/{}), will retry: {}",
                    work_id, w.retry_count, self.config.max_retries, error
                );
                if self.config.enable_persistence {
                    self.persist_work_retry(&work_id, &error, w.retry_count).await;
                }
                drop(wm);
                tokio::time::sleep(Duration::from_millis(self.config.retry_delay_ms)).await;
                return self.run_step(work_id).await;
            } else {
                w.status = WorkStatus::Failed(error.clone());
                if self.config.enable_persistence {
                    self.persist_work_failed(&work_id, &error).await;
                }
            }
        }
        self.stats.lock().await.errors += 1;
        Ok(StepExecutionResult {
            step_id: work.step.step_id,
            plan_id: work.plan_id,
            success: false,
            output: json!(null),
            error: Some(error),
            duration_ms: elapsed,
        })
    }

    pub async fn get_work(&self, work_id: Uuid) -> Result<Option<WorkItem>> {
        let wm = self.active_work.lock().await;
        Ok(wm.get(&work_id).cloned())
    }

    pub async fn list_active_work(&self) -> Vec<WorkItem> {
        let wm = self.active_work.lock().await;
        wm.values().cloned().collect()
    }

    pub async fn cancel_work(&self, work_id: Uuid) -> Result<()> {
        let mut wm = self.active_work.lock().await;
        if let Some(w) = wm.get_mut(&work_id) {
            w.status = WorkStatus::Cancelled;
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

    async fn persist_work_item(&self, work: &WorkItem) {
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
        let _ = store.lock().await.store(nexora_memory::MemoryLayer::Session, &key, &val).await;
    }

    async fn persist_work_completed(&self, work_id: &Uuid, output: &Value, elapsed: u64) {
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
        let _ = store.lock().await.store(nexora_memory::MemoryLayer::Long, &key, &val).await;
    }

    async fn persist_work_retry(&self, work_id: &Uuid, error: &str, attempt: u32) {
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
        let _ = store.lock().await.store(nexora_memory::MemoryLayer::Session, &key, &val).await;
    }

    async fn persist_work_failed(&self, work_id: &Uuid, error: &str) {
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
        let _ = store.lock().await.store(nexora_memory::MemoryLayer::Long, &key, &val).await;
    }

    pub async fn restore_in_flight_work(&self) -> Vec<WorkItem> {
        let store = match &self.memory_store {
            Some(s) => s,
            None => return Vec::new(),
        };
        let restore_key = "worker:in_flight";
        let mut store_guard = store.lock().await;
        let mut wm = self.active_work.lock().await;
        if let Ok(Some(val)) = store_guard.retrieve(nexora_memory::MemoryLayer::Session, restore_key).await {
            if let Ok(ids) = serde_json::from_str::<Vec<String>>(&val) {
                for id_str in ids {
                    if let Ok(wid) = Uuid::parse_str(&id_str) {
                        let key = format!("worker:work:{}", wid);
                        if let Ok(Some(entry)) = store_guard.retrieve(nexora_memory::MemoryLayer::Session, &key).await {
                            if let Ok(json_val) = serde_json::from_str::<Value>(&entry) {
                                if let Some(step_id) = json_val.get("step_id").and_then(|v| v.as_str()).and_then(|s| Uuid::parse_str(s).ok()) {
                                    let plan_id = json_val.get("plan_id").and_then(|v| v.as_str()).and_then(|s| Uuid::parse_str(s).ok()).unwrap_or_default();
                                    let desc = json_val.get("description").and_then(|v| v.as_str()).unwrap_or("restored");
                                    let work = WorkItem {
                                        work_id: wid,
                                        plan_id,
                                        step: crate::planner_agent::PlanStep {
                                            step_id,
                                            description: desc.to_string(),
                                            step_type: crate::planner_agent::StepType::Processing,
                                            dependencies: Vec::new(),
                                            required_capabilities: Vec::new(),
                                            estimated_duration_seconds: 60,
                                            status: crate::planner_agent::StepStatus::Pending,
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

                let plan_id = Uuid::parse_str(plan_id_str).map_err(|_| AgentError::ProcessingError {
                    operation: "parse".to_string(),
                    reason: "Invalid plan_id".to_string(),
                })?;
                let step_id = Uuid::parse_str(step_id_str).map_err(|_| AgentError::ProcessingError {
                    operation: "parse".to_string(),
                    reason: "Invalid step_id".to_string(),
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
                let work_id = Uuid::parse_str(work_id_str).map_err(|_| AgentError::ProcessingError {
                    operation: "parse".to_string(),
                    reason: "Invalid work_id".to_string(),
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
                let items: Vec<Value> = work.iter().map(|w| json!({
                    "work_id": w.work_id,
                    "plan_id": w.plan_id,
                    "step_id": w.step.step_id,
                    "status": format!("{:?}", w.status),
                })).collect();
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
                let work_id = Uuid::parse_str(work_id_str).map_err(|_| AgentError::ProcessingError {
                    operation: "parse".to_string(),
                    reason: "Invalid work_id".to_string(),
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
            s.avg_processing_time_ms = (s.avg_processing_time_ms
                * (s.messages_processed - 1) as f64
                + processing_time as f64)
                / s.messages_processed as f64;
            s.last_activity = chrono::Utc::now();
        }

        Ok(AgentResponse::success(context.session_id, result, processing_time))
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
        self.stats.blocking_lock().clone()
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

impl From<WorkerAgentConfig> for AgentConfig {
    fn from(config: WorkerAgentConfig) -> Self {
        AgentConfig {
            agent_id: "worker_agent".to_string(),
            agent_type: "worker".to_string(),
            max_concurrent_tasks: config.max_concurrent_work,
            timeout_seconds: config.default_timeout_seconds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_agent_config_default() {
        let config = WorkerAgentConfig::default();
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
            status: crate::planner_agent::StepStatus::Pending,
            result: None,
            error_message: None,
        };
        let work = WorkItem {
            work_id: Uuid::new_v4(),
            plan_id: Uuid::new_v4(),
            step,
            status: WorkStatus::Pending,
            retry_count: 0,
            started_at: now,
            last_updated: now,
            output: None,
            error: None,
        };
        assert_eq!(work.status, WorkStatus::Pending);
        assert!(work.output.is_none());
        assert_eq!(work.retry_count, 0);
    }

    #[test]
    fn test_work_status_variants() {
        assert!(matches!(WorkStatus::Pending, WorkStatus::Pending));
        assert!(matches!(WorkStatus::Running, WorkStatus::Running));
        assert!(matches!(WorkStatus::Completed, WorkStatus::Completed));
        assert!(matches!(WorkStatus::Failed("x".into()), WorkStatus::Failed(_)));
        assert!(matches!(WorkStatus::Cancelled, WorkStatus::Cancelled));
    }

    #[test]
    fn test_step_execution_result() {
        let result = StepExecutionResult {
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
        let agent = WorkerAgent::new(WorkerAgentConfig::default());
        assert_eq!(agent.name(), "WorkerAgent");
        assert_eq!(agent.agent_type(), "worker");
        assert_eq!(agent.status(), AgentStatus::Initializing);
    }

    #[tokio::test]
    async fn test_worker_process_step_types() {
        let agent = WorkerAgent::new(WorkerAgentConfig::default());
        let now = chrono::Utc::now();

        let types = vec![
            StepType::Analysis,
            StepType::Processing,
            StepType::Validation,
            StepType::DataCollection,
            StepType::Generation,
            StepType::Communication,
            StepType::Decision,
            StepType::Custom("test".to_string()),
        ];

        for st in types {
            let step = PlanStep {
                step_id: Uuid::new_v4(),
                description: format!("Test {:?}", st),
                step_type: st,
                dependencies: Vec::new(),
                required_capabilities: Vec::new(),
                estimated_duration_seconds: 10,
                status: crate::planner_agent::StepStatus::Pending,
                result: None,
                error_message: None,
            };
            let work = WorkItem {
                work_id: Uuid::new_v4(),
                plan_id: Uuid::new_v4(),
                step,
                status: WorkStatus::Pending,
                retry_count: 0,
                started_at: now,
                last_updated: now,
                output: None,
                error: None,
            };
            let work_id = work.work_id;
            agent.active_work.lock().await.insert(work_id, work);
            let result = agent.run_step(work_id).await;
            assert!(result.is_ok());
            let r = result.unwrap();
            assert!(r.success, "Step type {:?} should succeed", r);
        }
    }

    #[tokio::test]
    async fn test_worker_max_concurrent() {
        let config = WorkerAgentConfig {
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
            status: crate::planner_agent::StepStatus::Pending,
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
            status: crate::planner_agent::StepStatus::Pending,
            result: None,
            error_message: None,
        };

        assert!(agent.execute_step(Uuid::new_v4(), step1).await.is_ok());
        assert!(agent.execute_step(Uuid::new_v4(), step2).await.is_err());
    }
}
