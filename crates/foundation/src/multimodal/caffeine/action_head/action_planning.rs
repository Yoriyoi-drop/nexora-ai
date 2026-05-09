//! Action planning module for CAFFEINE
//! 
//! Implements action sequence planning and reasoning

use crate::caffeine::types::*;
use crate::caffeine::error::Result;
use std::collections::HashMap;

/// Action planning module
pub struct ActionPlanningModule {
    config: crate::caffeine::config::ActionConfig,
    planner: ActionPlanner,
    reasoner: ActionReasoner,
}

impl ActionPlanningModule {
    /// Create new action planning module
    pub fn new(config: crate::caffeine::config::ActionConfig) -> Result<Self> {
        let planner = ActionPlanner::new(config.clone())?;
        let reasoner = ActionReasoner::new(config.clone())?;
        
        Ok(Self {
            config,
            planner,
            reasoner,
        })
    }
    
    /// Generate action plan from tokens
    pub fn generate(&mut self, tokens: &[UnifiedToken], inputs: &MultiModalInputs) -> Result<ActionPlan> {
        // Extract task context from inputs
        let task_context = self.extract_task_context(inputs)?;
        
        // Generate candidate actions
        let candidates = self.generate_candidate_actions(tokens, &task_context)?;
        
        // Reason about actions and select best sequence
        let planned_actions = self.reasoner.reason_and_select(candidates, &task_context)?;
        
        // Create action plan
        let description = self.generate_plan_description(&planned_actions, &task_context)?;
        let estimated_duration = self.estimate_duration(&planned_actions);
        let success_probability = self.calculate_success_probability(&planned_actions, &task_context);
        
        Ok(ActionPlan {
            actions: planned_actions,
            description,
            estimated_duration_ms: estimated_duration,
            success_probability,
        })
    }
    
    /// Extract task context from inputs
    fn extract_task_context(&self, inputs: &MultiModalInputs) -> Result<TaskContext> {
        let mut context = TaskContext {
            task_type: TaskType::Generation,
            instruction: None,
            environment: None,
            constraints: Vec::new(),
            goals: Vec::new(),
        };
        
        // Extract task type and instruction
        if let Some(ref text_input) = inputs.text {
            context.instruction = Some(text_input.text.clone());
            context.task_type = self.infer_task_type(&text_input.text)?;
        }
        
        // Extract environment information
        if let Some(ref image_input) = inputs.image {
            context.environment = Some(EnvironmentInfo {
                screen_size: (image_input.width, image_input.height),
                ui_elements: Vec::new(),
                available_actions: Vec::new(),
            });
        }
        
        // Extract constraints and goals
        self.extract_constraints_and_goals(&mut context, inputs)?;
        
        Ok(context)
    }
    
    /// Infer task type from text
    fn infer_task_type(&self, text: &str) -> Result<TaskType> {
        let text_lower = text.to_lowercase();
        
        if text_lower.contains("click") || text_lower.contains("tap") || text_lower.contains("press") {
            Ok(TaskType::Planning)
        } else if text_lower.contains("extract") || text_lower.contains("get") || text_lower.contains("find") {
            Ok(TaskType::Retrieval)
        } else if text_lower.contains("classify") || text_lower.contains("categorize") {
            Ok(TaskType::Classification)
        } else if text_lower.contains("summarize") || text_lower.contains("summary") {
            Ok(TaskType::Summarization)
        } else if text_lower.contains("translate") {
            Ok(TaskType::Translation)
        } else if text_lower.contains("reason") || text_lower.contains("think") || text_lower.contains("analyze") {
            Ok(TaskType::Reasoning)
        } else {
            Ok(TaskType::Generation)
        }
    }
    
    /// Extract constraints and goals from inputs
    fn extract_constraints_and_goals(&self, context: &mut TaskContext, inputs: &MultiModalInputs) -> Result<()> {
        // Extract from instruction text
        if let Some(ref instruction) = context.instruction {
            // Look for constraints
            if instruction.to_lowercase().contains("quick") || instruction.to_lowercase().contains("fast") {
                context.constraints.push("time_limited".to_string());
            }
            
            if instruction.to_lowercase().contains("careful") || instruction.to_lowercase().contains("precise") {
                context.constraints.push("high_accuracy".to_string());
            }
            
            // Look for goals
            if instruction.to_lowercase().contains("extract") {
                context.goals.push("extract_information".to_string());
            }
            
            if instruction.to_lowercase().contains("navigate") {
                context.goals.push("reach_destination".to_string());
            }
        }
        
        Ok(())
    }
    
    /// Generate candidate actions
    fn generate_candidate_actions(&self, tokens: &[UnifiedToken], context: &TaskContext) -> Result<Vec<Action>> {
        let mut candidates = Vec::new();
        
        // Generate actions based on task type
        match context.task_type {
            TaskType::Planning => {
                candidates.push(self.create_action(
                    ActionType::Click,
                    self.create_click_parameters(tokens, context)?,
                )?);
                
                candidates.push(self.create_action(
                    ActionType::Type,
                    self.create_type_parameters(tokens, context)?,
                )?);
            }
            TaskType::Retrieval => {
                candidates.push(self.create_action(
                    ActionType::Extract,
                    self.create_extract_parameters(tokens, context)?,
                )?);
            }
            TaskType::Classification => {
                candidates.push(self.create_action(
                    ActionType::Analyze,
                    self.create_analyze_parameters(tokens, context)?,
                )?);
            }
            _ => {
                // Default actions
                candidates.push(self.create_action(
                    ActionType::Navigate,
                    self.create_navigate_parameters(tokens, context)?,
                )?);
            }
        }
        
        Ok(candidates)
    }
    
    /// Create action with parameters
    fn create_action(&self, action_type: ActionType, parameters: HashMap<String, serde_json::Value>) -> Result<Action> {
        Ok(Action {
            action_type,
            parameters,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| crate::caffeine::error::CaffeineError::output_generation(&format!("Failed to get timestamp: {}", e)))?
                .as_secs_f32(),
            confidence: 0.8,
        })
    }
    
    /// Create click parameters
    fn create_click_parameters(&self, tokens: &[UnifiedToken], context: &TaskContext) -> Result<HashMap<String, serde_json::Value>> {
        let mut params = HashMap::new();
        
        // Extract click coordinates from tokens
        if let Some(spatial_token) = tokens.iter().find(|t| t.modality == ModalityType::Image) {
            if let Some((x, y, w, h)) = spatial_token.spatial_coords {
                params.insert("x".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(x as f64).ok_or_else(|| crate::caffeine::error::CaffeineError::output_generation("Failed to convert x to number"))?));
                params.insert("y".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(y as f64).ok_or_else(|| crate::caffeine::error::CaffeineError::output_generation("Failed to convert y to number"))?));
            }
        }
        
        // Add target if available
        if let Some(ref instruction) = context.instruction {
            if let Some(target) = self.extract_target_from_instruction(instruction) {
                params.insert("target".to_string(), serde_json::Value::String(target));
            }
        }
        
        Ok(params)
    }
    
    /// Create type parameters
    fn create_type_parameters(&self, tokens: &[UnifiedToken], context: &TaskContext) -> Result<HashMap<String, serde_json::Value>> {
        let mut params = HashMap::new();
        
        // Extract text to type from tokens
        let text_tokens: Vec<_> = tokens.iter()
            .filter(|t| t.modality == ModalityType::Text)
            .collect();
        
        if !text_tokens.is_empty() {
            let text_to_type = self.tokens_to_text(&text_tokens)?;
            params.insert("text".to_string(), serde_json::Value::String(text_to_type));
        }
        
        Ok(params)
    }
    
    /// Create extract parameters
    fn create_extract_parameters(&self, tokens: &[UnifiedToken], context: &TaskContext) -> Result<HashMap<String, serde_json::Value>> {
        let mut params = HashMap::new();
        
        // Extract what to extract
        if let Some(ref instruction) = context.instruction {
            if let Some(extract_target) = self.extract_target_from_instruction(instruction) {
                params.insert("target".to_string(), serde_json::Value::String(extract_target));
            }
        }
        
        // Add extraction method
        params.insert("method".to_string(), serde_json::Value::String("semantic".to_string()));
        
        Ok(params)
    }
    
    /// Create analyze parameters
    fn create_analyze_parameters(&self, tokens: &[UnifiedToken], context: &TaskContext) -> Result<HashMap<String, serde_json::Value>> {
        let mut params = HashMap::new();
        
        // Add analysis type
        params.insert("analysis_type".to_string(), serde_json::Value::String("classification".to_string()));
        
        // Add context
        if let Some(ref instruction) = context.instruction {
            params.insert("context".to_string(), serde_json::Value::String(instruction.clone()));
        }
        
        Ok(params)
    }
    
    /// Create navigate parameters
    fn create_navigate_parameters(&self, tokens: &[UnifiedToken], context: &TaskContext) -> Result<HashMap<String, serde_json::Value>> {
        let mut params = HashMap::new();
        
        // Extract destination
        if let Some(ref instruction) = context.instruction {
            if let Some(destination) = self.extract_destination_from_instruction(instruction) {
                params.insert("destination".to_string(), serde_json::Value::String(destination));
            }
        }
        
        // Add navigation method
        params.insert("method".to_string(), serde_json::Value::String("direct".to_string()));
        
        Ok(params)
    }
    
    /// Extract target from instruction
    fn extract_target_from_instruction(&self, instruction: &str) -> Option<String> {
        // Simple extraction - look for keywords
        let keywords = vec!["button", "link", "text", "image", "input", "menu"];
        
        for keyword in keywords {
            if instruction.to_lowercase().contains(keyword) {
                return Some(keyword.to_string());
            }
        }
        
        None
    }
    
    /// Extract destination from instruction
    fn extract_destination_from_instruction(&self, instruction: &str) -> Option<String> {
        // Simple extraction - look for navigation keywords
        let nav_keywords = vec!["home", "page", "section", "menu", "settings"];
        
        for keyword in nav_keywords {
            if instruction.to_lowercase().contains(keyword) {
                return Some(keyword.to_string());
            }
        }
        
        None
    }
    
    /// Convert tokens to text
    fn tokens_to_text(&self, tokens: &[&UnifiedToken]) -> Result<String> {
        let mut text = String::new();
        
        for token in tokens {
            // Simple token to text conversion
            let word = format!("word{}", token.token_id % 1000);
            text.push_str(&word);
            text.push(' ');
        }
        
        Ok(text.trim().to_string())
    }
    
    /// Generate plan description
    fn generate_plan_description(&self, actions: &[Action], context: &TaskContext) -> Result<String> {
        let mut description = format!("Action plan for {:?} task", context.task_type);
        
        if let Some(ref instruction) = context.instruction {
            description.push_str(&format!(" based on instruction: {}", instruction));
        }
        
        description.push_str(&format!("\nPlanned actions ({}):", actions.len()));
        
        for (i, action) in actions.iter().enumerate() {
            description.push_str(&format!("\n{}. {:?}", i + 1, action.action_type));
        }
        
        Ok(description)
    }
    
    /// Estimate duration of action plan
    fn estimate_duration(&self, actions: &[Action]) -> f32 {
        let base_duration = 100.0; // Base duration in ms
        
        let action_durations = match actions.len() {
            0 => 0.0,
            1 => base_duration,
            2 => base_duration * 1.5,
            3 => base_duration * 2.0,
            _ => base_duration * 2.5,
        };
        
        action_durations
    }
    
    /// Calculate success probability
    fn calculate_success_probability(&self, actions: &[Action], context: &TaskContext) -> f32 {
        let base_probability = 0.8;
        
        // Adjust based on complexity
        let complexity_factor = 1.0 - (actions.len() as f32 * 0.1).min(0.5);
        
        // Adjust based on constraints
        let constraint_factor = if context.constraints.is_empty() {
            1.0
        } else {
            1.0 - (context.constraints.len() as f32 * 0.05).min(0.3)
        };
        
        (base_probability * complexity_factor * constraint_factor).max(0.1)
    }
}

/// Task context
#[derive(Debug, Clone)]
pub struct TaskContext {
    pub task_type: TaskType,
    pub instruction: Option<String>,
    pub environment: Option<EnvironmentInfo>,
    pub constraints: Vec<String>,
    pub goals: Vec<String>,
}

/// Environment information
#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    pub screen_size: (usize, usize),
    pub ui_elements: Vec<UIElement>,
    pub available_actions: Vec<ActionType>,
}

/// Action planner
pub struct ActionPlanner {
    config: crate::caffeine::config::ActionConfig,
}

impl ActionPlanner {
    /// Create new action planner
    pub fn new(config: crate::caffeine::config::ActionConfig) -> Result<Self> {
        Ok(Self { config })
    }
    
    /// Plan action sequence
    pub fn plan(&self, context: &TaskContext) -> Result<Vec<Action>> {
        // Simple planning logic
        let mut actions = Vec::new();
        
        match context.task_type {
            TaskType::Planning => {
                // Add planning actions
                actions.push(Action {
                    action_type: ActionType::Wait,
                    parameters: HashMap::new(),
                    timestamp: 0.0,
                    confidence: 0.9,
                });
            }
            _ => {
                // Default action
                actions.push(Action {
                    action_type: ActionType::Navigate,
                    parameters: HashMap::new(),
                    timestamp: 0.0,
                    confidence: 0.7,
                });
            }
        }
        
        Ok(actions)
    }
}

/// Action reasoner
pub struct ActionReasoner {
    config: crate::caffeine::config::ActionConfig,
}

impl ActionReasoner {
    /// Create new action reasoner
    pub fn new(config: crate::caffeine::config::ActionConfig) -> Result<Self> {
        Ok(Self { config })
    }
    
    /// Reason about and select best actions
    pub fn reason_and_select(&self, candidates: Vec<Action>, context: &TaskContext) -> Result<Vec<Action>> {
        // Simple selection - filter by confidence and sort
        let mut filtered: Vec<_> = candidates.into_iter()
            .filter(|action| action.confidence > 0.5)
            .collect();
        
        // Sort by confidence
        filtered.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit number of actions
        filtered.truncate(self.config.max_action_steps);
        
        Ok(filtered)
    }
}
