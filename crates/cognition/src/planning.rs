//! Planning Module - Task planning and execution strategies

use async_trait::async_trait;
use uuid::Uuid;
use nexora_foundation::FoundationResult;

/// Plan for executing a complex task
#[derive(Debug, Clone)]
pub struct Plan {
    pub id: Uuid,
    pub steps: Vec<PlanStep>,
    pub dependencies: Vec<Uuid>,
    pub estimated_duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct PlanStep {
    pub id: Uuid,
    pub action: String,
    pub parameters: serde_json::Value,
    pub dependencies: Vec<Uuid>,
    pub estimated_duration_ms: u64,
}

/// Planning strategy trait
#[async_trait]
pub trait PlanningStrategy: Send + Sync {
    /// Create a plan for the given goal
    async fn create_plan(&self, goal: &str, context: serde_json::Value) -> FoundationResult<Plan>;
    
    /// Optimize an existing plan
    async fn optimize_plan(&self, plan: &mut Plan) -> FoundationResult<()>;
    
    /// Validate a plan
    async fn validate_plan(&self, plan: &Plan) -> FoundationResult<bool>;
    
    /// Get strategy name
    fn strategy_name(&self) -> &str;
}

/// Default hierarchical planning strategy
pub struct HierarchicalPlanner;

#[async_trait]
impl PlanningStrategy for HierarchicalPlanner {
    async fn create_plan(&self, goal: &str, context: serde_json::Value) -> FoundationResult<Plan> {
        let sentences: Vec<&str> = goal.split(|c: char| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        
        let mut steps = Vec::new();
        let mut step_ids = Vec::new();
        let mut total_duration = 0u64;
        
        for (i, sentence) in sentences.iter().enumerate() {
            let step_id = Uuid::new_v4();
            let deps = if i > 0 {
                vec![step_ids[i - 1]]
            } else {
                vec![]
            };
            
            let action = sentence.to_string();
            let est_duration = (action.len() as u64).max(100) * 10;
            total_duration += est_duration;
            
            step_ids.push(step_id);
            steps.push(PlanStep {
                id: step_id,
                action,
                parameters: serde_json::Value::Object(serde_json::Map::new()),
                dependencies: deps,
                estimated_duration_ms: est_duration,
            });
        }
        
        if steps.is_empty() {
            let step_id = Uuid::new_v4();
            total_duration = 1000;
            steps.push(PlanStep {
                id: step_id,
                action: goal.to_string(),
                parameters: serde_json::Value::Object(serde_json::Map::new()),
                dependencies: vec![],
                estimated_duration_ms: 1000,
            });
            step_ids.push(step_id);
        }
        
        Ok(Plan {
            id: Uuid::new_v4(),
            steps,
            dependencies: step_ids,
            estimated_duration_ms: total_duration,
        })
    }
    
    async fn optimize_plan(&self, plan: &mut Plan) -> FoundationResult<()> {
        plan.steps.sort_by(|a, b| a.estimated_duration_ms.cmp(&b.estimated_duration_ms));
        plan.dependencies = plan.steps.iter().map(|s| s.id).collect();
        plan.estimated_duration_ms = plan.steps.iter().map(|s| s.estimated_duration_ms).sum();
        Ok(())
    }
    
    async fn validate_plan(&self, plan: &Plan) -> FoundationResult<bool> {
        if plan.steps.is_empty() {
            return Ok(false);
        }
        if plan.dependencies.is_empty() {
            return Ok(false);
        }
        Ok(true)
    }
    
    fn strategy_name(&self) -> &str {
        "hierarchical"
    }
}
