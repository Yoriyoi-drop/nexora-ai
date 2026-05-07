//! Planner Agent
//! 
//! Agent untuk memecah task besar menjadi step-step yang lebih kecil.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use uuid::Uuid;
use serde_json::{Value, json};
use tracing::{debug, info, warn};

use crate::{
    Agent, AgentError, Result, AgentMessage, AgentResponse, AgentStatus,
    AgentContext, AgentStats, AgentConfig
};

/// Planner agent untuk task decomposition
pub struct PlannerAgent {
    /// Unique ID
    id: Uuid,
    /// Agent name
    name: String,
    /// Current status
    status: AgentStatus,
    /// Active plans
    active_plans: Arc<std::sync::Mutex<HashMap<Uuid, ExecutionPlan>>>,
    /// Planning strategies
    strategies: Vec<Box<dyn PlanningStrategy>>,
    /// Statistics
    stats: AgentStats,
    /// Configuration
    config: PlannerAgentConfig,
}

/// Configuration untuk planner agent
#[derive(Debug, Clone)]
pub struct PlannerAgentConfig {
    /// Maximum concurrent plans
    pub max_concurrent_plans: usize,
    /// Default planning depth
    pub default_planning_depth: u8,
    /// Enable adaptive planning
    pub enable_adaptive_planning: bool,
    /// Maximum plan complexity
    pub max_plan_complexity: u8,
}

/// Execution plan untuk task
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Plan ID
    pub plan_id: Uuid,
    /// Task description
    pub task_description: String,
    /// Plan steps
    pub steps: Vec<PlanStep>,
    /// Current step index
    pub current_step_index: usize,
    /// Plan status
    pub status: PlanStatus,
    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
    /// Plan metadata
    pub metadata: HashMap<String, Value>,
}

/// Individual step dalam plan
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// Step ID
    pub step_id: Uuid,
    /// Step description
    pub description: String,
    /// Step type
    pub step_type: StepType,
    /// Dependencies (step IDs yang harus selesai dulu)
    pub dependencies: Vec<Uuid>,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Estimated duration (seconds)
    pub estimated_duration_seconds: u64,
    /// Step status
    pub status: StepStatus,
    /// Step result
    pub result: Option<Value>,
    /// Error message (jika failed)
    pub error_message: Option<String>,
}

/// Step type
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum StepType {
    /// Data collection
    DataCollection,
    /// Analysis
    Analysis,
    /// Processing
    Processing,
    /// Generation
    Generation,
    /// Validation
    Validation,
    /// Communication
    Communication,
    /// Decision
    Decision,
    /// Custom step type
    Custom(String),
}

/// Plan status
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum PlanStatus {
    /// Plan sedang dibuat
    Planning,
    /// Plan siap dieksekusi
    Ready,
    /// Plan sedang dieksekusi
    Executing,
    /// Plan selesai
    Completed,
    /// Plan gagal
    Failed(String),
    /// Plan di-pause
    Paused,
    /// Plan di-cancel
    Cancelled,
}

/// Step status
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum StepStatus {
    /// Belum mulai
    Pending,
    /// Sedang dieksekusi
    Running,
    /// Selesai
    Completed,
    /// Gagal
    Failed(String),
    /// Di-skip
    Skipped,
}

/// Trait untuk planning strategy
#[async_trait]
pub trait PlanningStrategy: Send + Sync {
    /// Strategy name
    fn name(&self) -> &str;
    
    /// Can handle this task?
    async fn can_handle(&self, task: &str, context: &Value) -> bool;
    
    /// Create plan for task
    async fn create_plan(&self, task: &str, context: &Value, depth: u8) -> Result<ExecutionPlan>;
    
    /// Adapt existing plan
    async fn adapt_plan(&self, plan: &mut ExecutionPlan, feedback: &Value) -> Result<()>;
}

impl PlannerAgent {
    /// Create new planner agent
    pub fn new(config: PlannerAgentConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "PlannerAgent".to_string(),
            status: AgentStatus::Initializing,
            active_plans: Arc::new(std::sync::Mutex::new(HashMap::new())),
            strategies: Vec::new(),
            stats: AgentStats::default(),
            config,
        }
    }
    
    /// Add planning strategy
    pub fn add_strategy(&mut self, strategy: Box<dyn PlanningStrategy>) {
        self.strategies.push(strategy);
    }
    
    /// Create execution plan for task
    pub async fn create_plan(
        &self,
        task_description: String,
        context: &Value,
    ) -> Result<Uuid> {
        debug!("Creating plan for task: {}", task_description);
        
        // Check concurrent plan limit
        {
            let plans = self.active_plans.lock().unwrap();
            if plans.len() >= self.config.max_concurrent_plans {
                return Err(AgentError::ProcessingError(
                    format!("Maximum concurrent plans ({}) reached", self.config.max_concurrent_plans)
                ));
            }
        }
        
        // Find appropriate strategy
        let strategy = self.find_strategy(&task_description, context).await?;
        
        // Create plan
        let plan = strategy.create_plan(&task_description, context, self.config.default_planning_depth).await?;
        
        // Validate plan complexity
        if plan.steps.len() > self.config.max_plan_complexity as usize {
            return Err(AgentError::ProcessingError(
                format!("Plan complexity ({}) exceeds maximum ({})", 
                       plan.steps.len(), self.config.max_plan_complexity)
            ));
        }
        
        // Add to active plans
        {
            let mut plans = self.active_plans.lock().unwrap();
            plans.insert(plan.plan_id, plan.clone());
        }
        
        info!("Plan {} created for task: {}", plan.plan_id, task_description);
        Ok(plan.plan_id)
    }
    
    /// Execute next step in plan
    pub async fn execute_next_step(&self, plan_id: Uuid) -> Result<Option<PlanStep>> {
        debug!("Executing next step for plan: {}", plan_id);
        
        let mut plans = self.active_plans.lock().unwrap();
        if let Some(plan) = plans.get_mut(&plan_id) {
            // Find next executable step
            if let Some(step_index) = self.find_next_executable_step(plan) {
                let step = &mut plan.steps[step_index];
                step.status = StepStatus::Running;
                plan.current_step_index = step_index;
                plan.status = PlanStatus::Executing;
                plan.last_updated = chrono::Utc::now();
                
                info!("Executing step {} for plan {}", step.step_id, plan_id);
                Ok(Some(step.clone()))
            } else {
                // Check if plan is completed
                if self.is_plan_completed(plan) {
                    plan.status = PlanStatus::Completed;
                    info!("Plan {} completed", plan_id);
                }
                Ok(None)
            }
        } else {
            Err(AgentError::ProcessingError(format!("Plan {} not found", plan_id)))
        }
    }
    
    /// Complete step with result
    pub async fn complete_step(
        &self,
        plan_id: Uuid,
        step_id: Uuid,
        result: Value,
    ) -> Result<()> {
        debug!("Completing step {} for plan {}", step_id, plan_id);
        
        let mut plans = self.active_plans.lock().unwrap();
        if let Some(plan) = plans.get_mut(&plan_id) {
            // Find and update step
            if let Some(step) = plan.steps.iter_mut().find(|s| s.step_id == step_id) {
                step.status = StepStatus::Completed;
                step.result = Some(result);
                plan.last_updated = chrono::Utc::now();
                
                info!("Step {} completed for plan {}", step_id, plan_id);
                Ok(())
            } else {
                Err(AgentError::ProcessingError(format!("Step {} not found in plan {}", step_id, plan_id)))
            }
        } else {
            Err(AgentError::ProcessingError(format!("Plan {} not found", plan_id)))
        }
    }
    
    /// Fail step with error
    pub async fn fail_step(
        &self,
        plan_id: Uuid,
        step_id: Uuid,
        error: String,
    ) -> Result<()> {
        debug!("Failing step {} for plan {}: {}", step_id, plan_id, error);
        
        let mut plans = self.active_plans.lock().unwrap();
        if let Some(plan) = plans.get_mut(&plan_id) {
            // Find and update step
            if let Some(step) = plan.steps.iter_mut().find(|s| s.step_id == step_id) {
                step.status = StepStatus::Failed(error.clone());
                step.error_message = Some(error.clone());
                plan.last_updated = chrono::Utc::now();
                
                // Update plan status
                plan.status = PlanStatus::Failed(error.clone());
                
                warn!("Step {} failed for plan {}: {}", step_id, plan_id, error);
                Ok(())
            } else {
                Err(AgentError::ProcessingError(format!("Step {} not found in plan {}", step_id, plan_id)))
            }
        } else {
            Err(AgentError::ProcessingError(format!("Plan {} not found", plan_id)))
        }
    }
    
    /// Get plan information
    pub async fn get_plan(&self, plan_id: Uuid) -> Result<Option<ExecutionPlan>> {
        let plans = self.active_plans.lock().unwrap();
        Ok(plans.get(&plan_id).cloned())
    }
    
    /// List active plans
    pub async fn list_active_plans(&self) -> Vec<ExecutionPlan> {
        let plans = self.active_plans.lock().unwrap();
        plans.values().cloned().collect()
    }
    
    /// Adapt plan based on feedback
    pub async fn adapt_plan(
        &self,
        plan_id: Uuid,
        feedback: &Value,
    ) -> Result<()> {
        debug!("Adapting plan {} based on feedback", plan_id);
        
        if !self.config.enable_adaptive_planning {
            return Err(AgentError::ProcessingError("Adaptive planning is disabled".to_string()));
        }
        
        let strategy = self.find_strategy_for_adaptation(feedback).await?;
        
        let mut plans = self.active_plans.lock().unwrap();
        if let Some(plan) = plans.get_mut(&plan_id) {
            strategy.adapt_plan(plan, feedback).await?;
            plan.last_updated = chrono::Utc::now();
            
            info!("Plan {} adapted successfully", plan_id);
            Ok(())
        } else {
            Err(AgentError::ProcessingError(format!("Plan {} not found", plan_id)))
        }
    }
    
    /// Cancel plan
    pub async fn cancel_plan(&self, plan_id: Uuid) -> Result<()> {
        debug!("Cancelling plan: {}", plan_id);
        
        let mut plans = self.active_plans.lock().unwrap();
        if let Some(plan) = plans.get_mut(&plan_id) {
            plan.status = PlanStatus::Cancelled;
            plan.last_updated = chrono::Utc::now();
            
            info!("Plan {} cancelled", plan_id);
            Ok(())
        } else {
            Err(AgentError::ProcessingError(format!("Plan {} not found", plan_id)))
        }
    }
    
    /// Find appropriate strategy for task
    async fn find_strategy(&self, task: &str, context: &Value) -> Result<&dyn PlanningStrategy> {
        for strategy in &self.strategies {
            if strategy.can_handle(task, context).await {
                return Ok(strategy.as_ref());
            }
        }
        
        Err(AgentError::ProcessingError(
            format!("No strategy found for task: {}", task)
        ))
    }
    
    /// Find strategy for adaptation
    async fn find_strategy_for_adaptation(&self, _feedback: &Value) -> Result<&dyn PlanningStrategy> {
        // For now, use first available strategy
        if let Some(strategy) = self.strategies.first() {
            Ok(strategy.as_ref())
        } else {
            Err(AgentError::ProcessingError("No strategies available for adaptation".to_string()))
        }
    }
    
    /// Find next executable step
    fn find_next_executable_step(&self, plan: &ExecutionPlan) -> Option<usize> {
        for (index, step) in plan.steps.iter().enumerate() {
            if step.status == StepStatus::Pending {
                // Check dependencies
                let dependencies_met = step.dependencies.iter().all(|dep_id| {
                    plan.steps.iter().any(|s| s.step_id == *dep_id && s.status == StepStatus::Completed)
                });
                
                if dependencies_met {
                    return Some(index);
                }
            }
        }
        None
    }
    
    /// Check if plan is completed
    fn is_plan_completed(&self, plan: &ExecutionPlan) -> bool {
        plan.steps.iter().all(|step| {
            matches!(step.status, StepStatus::Completed | StepStatus::Skipped)
        })
    }
    
    /// Get planning statistics
    pub fn get_planning_stats(&self) -> PlanningStats {
        let plans = self.active_plans.lock().unwrap();
        let total_steps: usize = plans.values().map(|p| p.steps.len()).sum();
        let completed_steps: usize = plans.values()
            .flat_map(|p| p.steps.iter())
            .filter(|s| s.status == StepStatus::Completed)
            .count();
        
        PlanningStats {
            active_plans: plans.len(),
            total_steps,
            completed_steps,
            max_concurrent_plans: self.config.max_concurrent_plans,
            adaptive_planning_enabled: self.config.enable_adaptive_planning,
        }
    }
}

/// Planning statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct PlanningStats {
    pub active_plans: usize,
    pub total_steps: usize,
    pub completed_steps: usize,
    pub max_concurrent_plans: usize,
    pub adaptive_planning_enabled: bool,
}

#[async_trait]
impl Agent for PlannerAgent {
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn agent_type(&self) -> &str {
        "planner"
    }
    
    fn status(&self) -> AgentStatus {
        self.status.clone()
    }
    
    async fn initialize(&mut self, _config: AgentConfig) -> Result<()> {
        info!("Initializing PlannerAgent");
        
        // Add default strategies
        self.add_default_strategies();
        
        self.status = AgentStatus::Ready;
        Ok(())
    }
    
    async fn receive(&mut self, message: AgentMessage) -> Result<()> {
        debug!("PlannerAgent received message: {}", message.message_type);
        Ok(())
    }
    
    async fn process(&mut self, context: AgentContext) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();
        
        debug!("PlannerAgent processing request for session: {}", context.session_id);
        
        // Extract action from context
        let action = context.parameters.get("action").and_then(|v| v.as_str()).unwrap_or("status");
        
        let result = match action {
            "create_plan" => {
                let task = context.parameters.get("task")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("task required".to_string()))?;
                
                let task_context = context.parameters.get("context").cloned().unwrap_or(Value::Object(serde_json::Map::new()));
                
                let plan_id = self.create_plan(task.to_string(), &task_context).await?;
                
                json!({
                    "action": "create_plan",
                    "plan_id": plan_id,
                    "status": "created"
                })
            }
            
            "execute_step" => {
                let plan_id_str = context.parameters.get("plan_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("plan_id required".to_string()))?;
                
                let plan_id = Uuid::parse_str(plan_id_str)
                    .map_err(|_| AgentError::ProcessingError("Invalid plan_id".to_string()))?;
                
                let step = self.execute_next_step(plan_id).await?;
                
                match step {
                    Some(step) => json!({
                        "action": "execute_step",
                        "plan_id": plan_id,
                        "step": {
                            "step_id": step.step_id,
                            "description": step.description,
                            "step_type": step.step_type,
                            "dependencies": step.dependencies,
                            "required_capabilities": step.required_capabilities
                        }
                    }),
                    None => json!({
                        "action": "execute_step",
                        "plan_id": plan_id,
                        "message": "No executable steps available"
                    })
                }
            }
            
            "complete_step" => {
                let plan_id_str = context.parameters.get("plan_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("plan_id required".to_string()))?;
                
                let step_id_str = context.parameters.get("step_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("step_id required".to_string()))?;
                
                let result = context.parameters.get("result")
                    .cloned()
                    .unwrap_or(Value::Null);
                
                let plan_id = Uuid::parse_str(plan_id_str)
                    .map_err(|_| AgentError::ProcessingError("Invalid plan_id".to_string()))?;
                
                let step_id = Uuid::parse_str(step_id_str)
                    .map_err(|_| AgentError::ProcessingError("Invalid step_id".to_string()))?;
                
                self.complete_step(plan_id, step_id, result).await?;
                
                json!({
                    "action": "complete_step",
                    "plan_id": plan_id,
                    "step_id": step_id,
                    "status": "completed"
                })
            }
            
            "get_plan" => {
                let plan_id_str = context.parameters.get("plan_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("plan_id required".to_string()))?;
                
                let plan_id = Uuid::parse_str(plan_id_str)
                    .map_err(|_| AgentError::ProcessingError("Invalid plan_id".to_string()))?;
                
                let plan = self.get_plan(plan_id).await?;
                
                match plan {
                    Some(plan) => json!({
                        "action": "get_plan",
                        "plan": {
                            "plan_id": plan.plan_id,
                            "task_description": plan.task_description,
                            "status": plan.status,
                            "current_step_index": plan.current_step_index,
                            "steps": plan.steps.iter().map(|s| json!({
                                "step_id": s.step_id,
                                "description": s.description,
                                "step_type": s.step_type,
                                "status": s.status,
                                "dependencies": s.dependencies
                            })).collect::<Vec<_>>()
                        }
                    }),
                    None => json!({
                        "action": "get_plan",
                        "error": "Plan not found"
                    })
                }
            }
            
            "list_plans" => {
                let plans = self.list_active_plans().await;
                let plan_list: Vec<Value> = plans.iter()
                    .map(|p| json!({
                        "plan_id": p.plan_id,
                        "task_description": p.task_description,
                        "status": p.status,
                        "step_count": p.steps.len(),
                        "current_step": p.current_step_index
                    }))
                    .collect();
                
                json!({
                    "action": "list_plans",
                    "plans": plan_list,
                    "count": plan_list.len()
                })
            }
            
            "stats" => {
                let stats = self.get_planning_stats();
                json!({
                    "action": "stats",
                    "stats": stats
                })
            }
            
            _ => {
                return Err(AgentError::ProcessingError(format!("Unknown action: {}", action)));
            }
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Update stats
        self.stats.messages_processed += 1;
        self.stats.avg_processing_time_ms = 
            (self.stats.avg_processing_time_ms * (self.stats.messages_processed - 1) as f64 + 
             processing_time as f64) / self.stats.messages_processed as f64;
        self.stats.last_activity = chrono::Utc::now();
        
        let response = AgentResponse::success(
            context.session_id,
            result,
            processing_time,
        );
        
        Ok(response)
    }
    
    async fn respond(&mut self, _response: AgentResponse) -> Result<()> {
        debug!("PlannerAgent sending response");
        Ok(())
    }
    
    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down PlannerAgent");
        
        // Cancel all active plans
        let plan_ids: Vec<Uuid> = {
            let plans = self.active_plans.lock().unwrap();
            plans.keys().cloned().collect()
        };
        
        for plan_id in plan_ids {
            if let Err(e) = self.cancel_plan(plan_id).await {
                warn!("Failed to cancel plan {}: {}", plan_id, e);
            }
        }
        
        self.status = AgentStatus::Shutdown;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<bool> {
        // Check if we have strategies available
        Ok(!self.strategies.is_empty())
    }
    
    fn get_stats(&self) -> AgentStats {
        self.stats.clone()
    }
    
    fn get_config(&self) -> AgentConfig {
        self.config.clone().into()
    }
}

impl PlannerAgent {
    /// Add default planning strategies
    fn add_default_strategies(&mut self) {
        // Add simple sequential strategy
        self.add_strategy(Box::new(SimpleSequentialStrategy));
        
        // Add dependency-based strategy
        self.add_strategy(Box::new(DependencyBasedStrategy));
    }
}

/// Simple sequential planning strategy
struct SimpleSequentialStrategy;

#[async_trait]
impl PlanningStrategy for SimpleSequentialStrategy {
    fn name(&self) -> &str {
        "simple_sequential"
    }
    
    async fn can_handle(&self, _task: &str, _context: &Value) -> bool {
        true // Can handle any task
    }
    
    async fn create_plan(&self, task: &str, _context: &Value, _depth: u8) -> Result<ExecutionPlan> {
        let plan_id = Uuid::new_v4();
        let now = chrono::Utc::now();
        
        // Create simple sequential steps
        let analysis_step_id = Uuid::new_v4();
        let processing_step_id = Uuid::new_v4();
        let validation_step_id = Uuid::new_v4();
        
        let steps = vec![
            PlanStep {
                step_id: analysis_step_id,
                description: "Analyze task requirements".to_string(),
                step_type: StepType::Analysis,
                dependencies: Vec::new(),
                required_capabilities: vec!["analysis".to_string()],
                estimated_duration_seconds: 30,
                status: StepStatus::Pending,
                result: None,
                error_message: None,
            },
            PlanStep {
                step_id: processing_step_id,
                description: "Execute main task".to_string(),
                step_type: StepType::Processing,
                dependencies: vec![analysis_step_id],
                required_capabilities: vec!["processing".to_string()],
                estimated_duration_seconds: 60,
                status: StepStatus::Pending,
                result: None,
                error_message: None,
            },
            PlanStep {
                step_id: validation_step_id,
                description: "Validate results".to_string(),
                step_type: StepType::Validation,
                dependencies: vec![processing_step_id],
                required_capabilities: vec!["validation".to_string()],
                estimated_duration_seconds: 15,
                status: StepStatus::Pending,
                result: None,
                error_message: None,
            },
        ];
        
        Ok(ExecutionPlan {
            plan_id,
            task_description: task.to_string(),
            steps,
            current_step_index: 0,
            status: PlanStatus::Ready,
            created_at: now,
            last_updated: now,
            metadata: HashMap::new(),
        })
    }
    
    async fn adapt_plan(&self, _plan: &mut ExecutionPlan, _feedback: &Value) -> Result<()> {
        // Simple adaptation - not implemented for now
        Ok(())
    }
}

/// Dependency-based planning strategy
struct DependencyBasedStrategy;

#[async_trait]
impl PlanningStrategy for DependencyBasedStrategy {
    fn name(&self) -> &str {
        "dependency_based"
    }
    
    async fn can_handle(&self, _task: &str, _context: &Value) -> bool {
        true // Can handle any task
    }
    
    async fn create_plan(&self, task: &str, _context: &Value, _depth: u8) -> Result<ExecutionPlan> {
        let plan_id = Uuid::new_v4();
        let now = chrono::Utc::now();
        
        // Create dependency-based steps
        let data_step = PlanStep {
            step_id: Uuid::new_v4(),
            description: "Collect required data".to_string(),
            step_type: StepType::DataCollection,
            dependencies: Vec::new(),
            required_capabilities: vec!["data_collection".to_string()],
            estimated_duration_seconds: 45,
            status: StepStatus::Pending,
            result: None,
            error_message: None,
        };
        
        let analysis_step = PlanStep {
            step_id: Uuid::new_v4(),
            description: "Analyze collected data".to_string(),
            step_type: StepType::Analysis,
            dependencies: vec![data_step.step_id],
            required_capabilities: vec!["analysis".to_string()],
            estimated_duration_seconds: 30,
            status: StepStatus::Pending,
            result: None,
            error_message: None,
        };
        
        let processing_step = PlanStep {
            step_id: Uuid::new_v4(),
            description: "Process based on analysis".to_string(),
            step_type: StepType::Processing,
            dependencies: vec![analysis_step.step_id],
            required_capabilities: vec!["processing".to_string()],
            estimated_duration_seconds: 60,
            status: StepStatus::Pending,
            result: None,
            error_message: None,
        };
        
        let validation_step = PlanStep {
            step_id: Uuid::new_v4(),
            description: "Validate final results".to_string(),
            step_type: StepType::Validation,
            dependencies: vec![processing_step.step_id],
            required_capabilities: vec!["validation".to_string()],
            estimated_duration_seconds: 20,
            status: StepStatus::Pending,
            result: None,
            error_message: None,
        };
        
        Ok(ExecutionPlan {
            plan_id,
            task_description: task.to_string(),
            steps: vec![data_step, analysis_step, processing_step, validation_step],
            current_step_index: 0,
            status: PlanStatus::Ready,
            created_at: now,
            last_updated: now,
            metadata: HashMap::new(),
        })
    }
    
    async fn adapt_plan(&self, _plan: &mut ExecutionPlan, _feedback: &Value) -> Result<()> {
        // Dependency-based adaptation - not implemented for now
        Ok(())
    }
}

impl From<PlannerAgentConfig> for AgentConfig {
    fn from(config: PlannerAgentConfig) -> Self {
        AgentConfig {
            agent_id: "planner_agent".to_string(),
            agent_type: "planner".to_string(),
            max_concurrent_tasks: config.max_concurrent_plans,
            timeout_seconds: 45,
        }
    }
}

impl Default for PlannerAgentConfig {
    fn default() -> Self {
        Self {
            max_concurrent_plans: 50,
            default_planning_depth: 3,
            enable_adaptive_planning: true,
            max_plan_complexity: 10,
        }
    }
}
