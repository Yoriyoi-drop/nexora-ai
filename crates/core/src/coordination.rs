//! Multi-Model Coordination System
//! 
//! Mengkoordinasikan eksekusi multiple model spesialis dengan dependency management

use crate::types::{ModelId, IntentType, FusionResult};
use crate::error::{CoreError, CoreResult};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use uuid::Uuid;
use tracing::{debug, info, warn};

/// Model dependency graph untuk coordination
#[derive(Debug, Clone)]
pub struct ModelDependency {
    pub model: ModelId,
    pub depends_on: Vec<ModelId>,
    pub priority: i32, // Lower = higher priority
}

/// Multi-model coordinator
pub struct MultiModelCoordinator {
    /// Dependency graph
    dependencies: HashMap<ModelId, ModelDependency>,
    /// Active model instances
    model_instances: HashMap<ModelId, Arc<dyn ModelExecutor>>,
    /// Task semaphore untuk concurrency control
    task_semaphore: Arc<Semaphore>,
    /// Active tasks tracking
    active_tasks: Arc<RwLock<HashMap<String, TaskInfo>>>,
}

/// Task information untuk tracking
#[derive(Debug, Clone)]
struct TaskInfo {
    task_id: String,
    model: ModelId,
    status: TaskStatus,
    created_at: u64,
    dependencies: Vec<String>,
}

#[derive(Debug, Clone)]
enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
}

/// Trait untuk model executor
#[async_trait::async_trait]
pub trait ModelExecutor: Send + Sync {
    async fn execute(&self, input: &str, context: &str) -> CoreResult<String>;
    fn model_id(&self) -> ModelId;
    fn can_handle(&self, intent: IntentType) -> bool;
}

/// Simple model implementation untuk demo
pub struct SimpleModel {
    id: ModelId,
    name: String,
    supported_intents: HashSet<IntentType>,
}

impl SimpleModel {
    pub fn new(id: ModelId, name: String, supported_intents: Vec<IntentType>) -> Self {
        Self {
            id,
            name,
            supported_intents: supported_intents.into_iter().collect(),
        }
    }
}

#[async_trait::async_trait]
impl ModelExecutor for SimpleModel {
    async fn execute(&self, input: &str, _context: &str) -> CoreResult<String> {
        // Simulate processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        Ok(format!(
            "[{}] Processed: {}",
            self.name,
            input
        ))
    }
    
    fn model_id(&self) -> ModelId {
        self.id
    }
    
    fn can_handle(&self, intent: IntentType) -> bool {
        self.supported_intents.contains(&intent)
    }
}

impl MultiModelCoordinator {
    pub fn new(max_concurrent_tasks: usize) -> Self {
        Self {
            dependencies: Self::build_dependency_graph(),
            model_instances: HashMap::new(),
            task_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    fn build_dependency_graph() -> HashMap<ModelId, ModelDependency> {
        let mut deps = HashMap::new();
        
        // Core Controller - no dependencies
        deps.insert(ModelId::Controller, ModelDependency {
            model: ModelId::Controller,
            depends_on: vec![],
            priority: 0,
        });
        
        // Memory - depends on Controller
        deps.insert(ModelId::Memory, ModelDependency {
            model: ModelId::Memory,
            depends_on: vec![ModelId::Controller],
            priority: 1,
        });
        
        // Coding - depends on Memory
        deps.insert(ModelId::Coding, ModelDependency {
            model: ModelId::Coding,
            depends_on: vec![ModelId::Memory],
            priority: 2,
        });
        
        // Logic - depends on Controller
        deps.insert(ModelId::Logic, ModelDependency {
            model: ModelId::Logic,
            depends_on: vec![ModelId::Controller],
            priority: 1,
        });
        
        // Planner - depends on Logic and Memory
        deps.insert(ModelId::Planner, ModelDependency {
            model: ModelId::Planner,
            depends_on: vec![ModelId::Logic, ModelId::Memory],
            priority: 3,
        });
        
        // Ranking - depends on Logic
        deps.insert(ModelId::Ranking, ModelDependency {
            model: ModelId::Ranking,
            depends_on: vec![ModelId::Logic],
            priority: 2,
        });
        
        // Retrieval - depends on Memory
        deps.insert(ModelId::Retrieval, ModelDependency {
            model: ModelId::Retrieval,
            depends_on: vec![ModelId::Memory],
            priority: 2,
        });
        
        // Validator - depends on Logic
        deps.insert(ModelId::Validator, ModelDependency {
            model: ModelId::Validator,
            depends_on: vec![ModelId::Logic],
            priority: 2,
        });
        
        // Personality - depends on Logic
        deps.insert(ModelId::Personality, ModelDependency {
            model: ModelId::Personality,
            depends_on: vec![ModelId::Logic],
            priority: 2,
        });
        
        // Optimizer - depends on Coding and Validator
        deps.insert(ModelId::Optimizer, ModelDependency {
            model: ModelId::Optimizer,
            depends_on: vec![ModelId::Coding, ModelId::Validator],
            priority: 4,
        });
        
        deps
    }
    
    /// Register model instance
    pub fn register_model(&mut self, model: Arc<dyn ModelExecutor>) {
        let model_id = model.model_id();
        self.model_instances.insert(model_id, model);
        info!("Registered model: {:?}", model_id);
    }
    
    /// Coordinate multi-model execution
    pub async fn coordinate_execution(&self, 
        primary_intent: IntentType, 
        secondary_intents: &[IntentType], 
        input: &str, 
        context: &str
    ) -> CoreResult<FusionResult> {
        info!("Coordinating multi-model execution for intent: {:?}", primary_intent);
        
        // Determine which models to use
        let selected_models = self.select_models(primary_intent, secondary_intents)?;
        
        // Create execution plan
        let execution_plan = self.create_execution_plan(&selected_models)?;
        
        // Execute tasks according to plan
        let results = self.execute_plan(&execution_plan, input, context).await?;
        
        // Fuse results
        let mut model_responses = Vec::with_capacity(results.len());
        let mut response_sources = Vec::with_capacity(results.len());
        let mut success_count = 0;

        for result in &results {
            model_responses.push(result.output.clone());
            response_sources.push(result.model);
            if result.success {
                success_count += 1;
            }
        }

        let fused_response = self.generate_fused_response(&results);
        let has_conflicts = self.detect_conflicts(&results);
        let fusion_confidence = if results.is_empty() { 0.0 } else { success_count as f32 / results.len() as f32 };

        let fusion_result = FusionResult {
            model_responses,
            response_sources,
            response_count: results.len(),
            fused_response,
            fusion_confidence,
            has_conflicts,
            conflict_descriptions: if has_conflicts {
                self.describe_conflicts(&results)
            } else {
                Vec::new()
            },
        };
        
        info!("Multi-model coordination completed with {} models", selected_models.len());
        Ok(fusion_result)
    }
    
    fn select_models(&self, primary_intent: IntentType, secondary_intents: &[IntentType]) -> CoreResult<Vec<ModelId>> {
        let mut selected_models = Vec::with_capacity(1 + secondary_intents.len());
        
        let primary_model = self.find_model_for_intent(primary_intent)?;
        selected_models.push(primary_model);
        
        for &intent in secondary_intents {
            if let Ok(model) = self.find_model_for_intent(intent) {
                if !selected_models.contains(&model) {
                    selected_models.push(model);
                }
            }
        }
        
        // Sort by dependency and priority
        selected_models.sort_by(|a, b| {
            let dep_a = &self.dependencies[a];
            let dep_b = &self.dependencies[b];
            
            // First sort by dependency
            if dep_b.depends_on.contains(a) {
                return std::cmp::Ordering::Greater;
            }
            if dep_a.depends_on.contains(b) {
                return std::cmp::Ordering::Less;
            }
            
            // Then by priority
            dep_a.priority.cmp(&dep_b.priority)
        });
        
        Ok(selected_models)
    }
    
    fn find_model_for_intent(&self, intent: IntentType) -> CoreResult<ModelId> {
        // Simple mapping based on intent
        let model_id = match intent {
            IntentType::Coding => ModelId::Coding,
            IntentType::Memory => ModelId::Memory,
            IntentType::Debugging => ModelId::Coding,
            IntentType::Planning => ModelId::Planner,
            IntentType::Reasoning => ModelId::Logic,
            IntentType::Ranking => ModelId::Ranking,
            IntentType::Retrieval => ModelId::Retrieval,
            IntentType::Validation => ModelId::Validator,
            IntentType::Personality => ModelId::Personality,
            IntentType::Optimization => ModelId::Optimizer,
            IntentType::Unknown => ModelId::Controller,
        };
        
        if self.model_instances.contains_key(&model_id) {
            Ok(model_id)
        } else {
            Err(CoreError::ModelNotAvailable { model_id: model_id as u8 })
        }
    }
    
    fn create_execution_plan(&self, models: &[ModelId]) -> CoreResult<Vec<ExecutionStep>> {
        let mut plan = Vec::with_capacity(models.len());
        let mut completed = HashSet::with_capacity(models.len());
        
        while completed.len() < models.len() {
            let mut added_this_round = false;
            
            for &model in models {
                if completed.contains(&model) {
                    continue;
                }
                
                let deps = &self.dependencies[&model].depends_on;
                
                // Check if all dependencies are completed
                if deps.iter().all(|dep| completed.contains(dep)) {
                    let step = ExecutionStep {
                        model,
                        dependencies: deps.clone(),
                        step_id: Uuid::new_v4().to_string(),
                    };
                    plan.push(step);
                    completed.insert(model);
                    added_this_round = true;
                }
            }
            
            if !added_this_round {
                return Err(CoreError::TaskExecution("Circular dependency detected in execution plan".to_string()));
            }
        }
        
        Ok(plan)
    }
    
    async fn execute_plan(&self, plan: &[ExecutionStep], input: &str, context: &str) -> CoreResult<Vec<ModelResult>> {
        let total = plan.len();
        let mut completed: HashSet<usize> = HashSet::new();
        let mut results: HashMap<usize, ModelResult> = HashMap::new();
        let input = input.to_string();
        let context = context.to_string();

        while completed.len() < total {
            let ready: Vec<usize> = plan.iter().enumerate()
                .filter(|(i, _)| !completed.contains(&i))
                .filter(|(_, step)| step.dependencies.iter().all(|dep| {
                    completed.iter().any(|&j| plan[j].model == *dep)
                }))
                .map(|(i, _)| i)
                .collect();

            if ready.is_empty() {
                return Err(CoreError::TaskExecution(
                    "Circular dependency detected in execution plan".to_string()
                ));
            }

            // Execute ready steps in parallel
            let mut handles = Vec::with_capacity(ready.len());
            for &idx in &ready {
                let step = &plan[idx];
                let input = input.clone();
                let context = context.clone();
                let semaphore = self.task_semaphore.clone();
                let model = self.model_instances.get(&step.model).cloned();
                let model_id = step.model;

                handles.push(tokio::spawn(async move {
                    let _permit = semaphore.acquire_owned().await.map_err(|e| {
                        CoreError::TaskExecution(format!("Semaphore error: {}", e))
                    })?;
                    match model {
                        Some(executor) => {
                            let output = executor.execute(&input, &context).await?;
                            Ok::<_, CoreError>(output)
                        }
                        None => Err(CoreError::ModelNotAvailable { model_id: model_id as u8 }),
                    }
                }));
            }

            for (i, handle) in handles.into_iter().enumerate() {
                let idx = ready[i];
                let start_time = Self::current_timestamp();
                match handle.await {
                    Ok(Ok(output)) => {
                        let end_time = Self::current_timestamp();
                        results.insert(idx, ModelResult {
                            model: plan[idx].model,
                            output,
                            execution_time_ms: end_time - start_time,
                            success: true,
                            error: None,
                        });
                    }
                    Ok(Err(e)) => {
                        warn!("Model execution failed: {:?}, error: {}", plan[idx].model, e);
                        results.insert(idx, ModelResult {
                            model: plan[idx].model,
                            output: String::new(),
                            execution_time_ms: 0,
                            success: false,
                            error: Some(e.to_string()),
                        });
                    }
                    Err(e) => {
                        return Err(CoreError::TaskExecution(format!("Task join error: {}", e)));
                    }
                }
                completed.insert(idx);
            }
        }

        let ordered: Vec<ModelResult> = (0..total).map(|i| results.remove(&i).unwrap()).collect();
        Ok(ordered)
    }
    
    fn generate_fused_response(&self, results: &[ModelResult]) -> String {
        if results.is_empty() {
            return "No results to fuse".to_string();
        }
        
        if results.len() == 1 {
            return results[0].output.clone();
        }
        
        let mut fused = String::new();
        fused.push_str("🤖 Multi-Model Coordination Results:\n\n");
        
        for (i, result) in results.iter().enumerate() {
            fused.push_str(&format!(
                "{}. {} ({:.1}ms):\n   {}\n\n",
                i + 1,
                result.model.name(),
                result.execution_time_ms,
                if result.success {
                    &result.output
                } else {
                    "Unknown error"
                }
            ));
        }
        
        let success_rate = results.iter().filter(|r| r.success).count() as f32 / results.len() as f32;
        fused.push_str(&format!("📊 Success Rate: {:.1}%\n", success_rate * 100.0));
        
        fused
    }
    
    fn detect_conflicts(&self, results: &[ModelResult]) -> bool {
        // Simple conflict detection - check for contradictory outputs
        if results.len() < 2 {
            return false;
        }
        
        let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
        
        for i in 0..successful_results.len() {
            for j in (i + 1)..successful_results.len() {
                let output1 = &successful_results[i].output.to_lowercase();
                let output2 = &successful_results[j].output.to_lowercase();
                
                // Check for obvious contradictions
                if (output1.contains("yes") && output2.contains("no")) ||
                   (output1.contains("true") && output2.contains("false")) ||
                   (output1.contains("success") && output2.contains("error")) {
                    return true;
                }
            }
        }
        
        false
    }
    
    fn describe_conflicts(&self, results: &[ModelResult]) -> Vec<String> {
        let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
        let mut conflicts = Vec::with_capacity(successful_results.len().saturating_pow(2));
        
        for i in 0..successful_results.len() {
            for j in (i + 1)..successful_results.len() {
                let output1 = &successful_results[i].output.to_lowercase();
                let output2 = &successful_results[j].output.to_lowercase();
                
                if output1.contains("yes") && output2.contains("no") {
                    conflicts.push(format!(
                        "Contradiction between {:?} and {:?}: yes vs no",
                        successful_results[i].model,
                        successful_results[j].model
                    ));
                }
                
                if output1.contains("true") && output2.contains("false") {
                    conflicts.push(format!(
                        "Contradiction between {:?} and {:?}: true vs false",
                        successful_results[i].model,
                        successful_results[j].model
                    ));
                }
            }
        }
        
        conflicts
    }
    
    /// Get active tasks count
    pub async fn active_task_count(&self) -> usize {
        self.active_tasks.read().await.len()
    }
    
    /// Cancel all active tasks
    pub async fn cancel_all_tasks(&self) {
        let mut active_tasks = self.active_tasks.write().await;
        active_tasks.clear();
        info!("All active tasks cancelled");
    }
    
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Execution step dalam plan
#[derive(Debug, Clone)]
struct ExecutionStep {
    model: ModelId,
    dependencies: Vec<ModelId>,
    step_id: String,
}

/// Result dari model execution
#[derive(Debug, Clone)]
struct ModelResult {
    model: ModelId,
    output: String,
    execution_time_ms: u64,
    success: bool,
    error: Option<String>,
}

impl Default for MultiModelCoordinator {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::IntentType;
    
    #[tokio::test]
    async fn test_multi_model_coordination() {
        let mut coordinator = MultiModelCoordinator::new(5);
        
        // Register required models for the test
        let coding_model = Arc::new(SimpleModel::new(
            ModelId::Coding,
            "Coding Specialist".to_string(),
            vec![IntentType::Coding, IntentType::Debugging]
        ));
        
        coordinator.register_model(coding_model);
        
        // Test basic model selection
        let selected_models = coordinator.select_models(IntentType::Coding, &[]);
        assert!(selected_models.is_ok());
        let models = selected_models.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], ModelId::Coding);
        
        // Test that model is registered
        assert!(coordinator.model_instances.contains_key(&ModelId::Coding));
    }
    
    #[test]
    fn test_dependency_graph() {
        let coordinator = MultiModelCoordinator::new(5);
        
        // Test that Optimizer depends on Coding and Validator
        let optimizer_deps = &coordinator.dependencies[&ModelId::Optimizer];
        assert!(optimizer_deps.depends_on.contains(&ModelId::Coding));
        assert!(optimizer_deps.depends_on.contains(&ModelId::Validator));
        
        // Test that Controller has no dependencies
        let controller_deps = &coordinator.dependencies[&ModelId::Controller];
        assert!(controller_deps.depends_on.is_empty());
    }
}
