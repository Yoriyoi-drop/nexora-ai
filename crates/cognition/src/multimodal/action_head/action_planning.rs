use crate::multimodal::error::Result;
use crate::multimodal::types::*;
use std::collections::HashMap;

const VOCAB_SIZE: usize = 50257;

fn embedding_lookup(token_id: usize) -> f32 {
    let idx = token_id % VOCAB_SIZE;
    (idx as f32 / VOCAB_SIZE as f32) * 2.0 - 1.0
}

pub struct ActionPlanningModule {
    config: crate::multimodal::config::ActionConfig,
    planner: ActionPlanner,
    reasoner: ActionReasoner,
}

impl ActionPlanningModule {
    pub fn new(config: crate::multimodal::config::ActionConfig) -> Result<Self> {
        Ok(Self {
            planner: ActionPlanner::new(config.clone())?,
            reasoner: ActionReasoner::new(config.clone())?,
            config,
        })
    }

    pub fn generate(
        &mut self,
        tokens: &[UnifiedToken],
        inputs: &MultiModalInputs,
    ) -> Result<ActionPlan> {
        let task_context = self.extract_task_context(inputs)?;
        let candidates = self.generate_candidate_actions(tokens, &task_context)?;
        let planned_actions = self.reasoner.reason_and_select(candidates, &task_context)?;
        let description = self.generate_plan_description(&planned_actions, &task_context)?;
        let estimated_duration = self.estimate_duration(&planned_actions);
        let success_probability =
            self.calculate_success_probability(&planned_actions, &task_context);

        Ok(ActionPlan {
            actions: planned_actions,
            description,
            estimated_duration_ms: estimated_duration,
            success_probability,
        })
    }

    fn extract_task_context(&self, inputs: &MultiModalInputs) -> Result<TaskContext> {
        let mut context = TaskContext {
            task_type: TaskType::Generation,
            instruction: None,
            environment: None,
            constraints: Vec::new(),
            goals: Vec::new(),
        };

        if let Some(ref text_input) = inputs.text {
            context.instruction = Some(text_input.text.clone());
            context.task_type = self.infer_task_type(&text_input.text)?;
        }
        if let Some(ref image_input) = inputs.image {
            context.environment = Some(EnvironmentInfo {
                screen_size: (image_input.width, image_input.height),
                ui_elements: Vec::new(),
                available_actions: Vec::new(),
            });
        }
        self.extract_constraints_and_goals(&mut context, inputs)?;
        Ok(context)
    }

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

    fn extract_constraints_and_goals(
        &self,
        context: &mut TaskContext,
        _inputs: &MultiModalInputs,
    ) -> Result<()> {
        if let Some(ref instruction) = context.instruction {
            let lower = instruction.to_lowercase();
            if lower.contains("quick") || lower.contains("fast") {
                context.constraints.push("time_limited".to_string());
            }
            if lower.contains("careful") || lower.contains("precise") {
                context.constraints.push("high_accuracy".to_string());
            }
            if lower.contains("extract") {
                context.goals.push("extract_information".to_string());
            }
            if lower.contains("navigate") {
                context.goals.push("reach_destination".to_string());
            }
        }
        Ok(())
    }

    fn generate_candidate_actions(
        &self,
        tokens: &[UnifiedToken],
        context: &TaskContext,
    ) -> Result<Vec<Action>> {
        let mut candidates = Vec::new();
        match context.task_type {
            TaskType::Planning => {
                candidates.push(self.create_action(ActionType::Click, self.create_click_parameters(tokens, context)?)?);
                candidates.push(self.create_action(ActionType::Type, self.create_type_parameters(tokens, context)?)?);
            }
            TaskType::Retrieval => {
                candidates.push(self.create_action(ActionType::Extract, self.create_extract_parameters(tokens, context)?)?);
            }
            TaskType::Classification => {
                candidates.push(self.create_action(ActionType::Analyze, self.create_analyze_parameters(tokens, context)?)?);
            }
            _ => {
                candidates.push(self.create_action(ActionType::Navigate, self.create_navigate_parameters(tokens, context)?)?);
            }
        }
        Ok(candidates)
    }

    fn create_action(
        &self,
        action_type: ActionType,
        parameters: HashMap<String, serde_json::Value>,
    ) -> Result<Action> {
        Ok(Action {
            action_type,
            parameters,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    crate::multimodal::error::CaffeineError::output_generation(&format!(
                        "Failed to get timestamp: {}", e
                    ))
                })?
                .as_secs_f32(),
            confidence: 0.8,
        })
    }

    fn create_click_parameters(
        &self,
        tokens: &[UnifiedToken],
        context: &TaskContext,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let mut params = HashMap::with_capacity(3);
        if let Some(spatial_token) = tokens.iter().find(|t| t.modality == ModalityType::Image) {
            if let Some((x, y, _w, _h)) = spatial_token.spatial_coords {
                params.insert("x".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(x as f64).unwrap_or(serde_json::Number::from(0))));
                params.insert("y".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(y as f64).unwrap_or(serde_json::Number::from(0))));
            }
        }
        if let Some(ref instruction) = context.instruction {
            if let Some(target) = self.extract_target_from_instruction(instruction) {
                params.insert("target".to_string(), serde_json::Value::String(target));
            }
        }
        Ok(params)
    }

    fn create_type_parameters(
        &self,
        tokens: &[UnifiedToken],
        _context: &TaskContext,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let mut params = HashMap::with_capacity(1);
        let text_tokens: Vec<_> = tokens.iter().filter(|t| t.modality == ModalityType::Text).collect();
        if !text_tokens.is_empty() {
            let text_to_type = self.tokens_to_text(&text_tokens)?;
            params.insert("text".to_string(), serde_json::Value::String(text_to_type));
        }
        Ok(params)
    }

    fn create_extract_parameters(
        &self,
        _tokens: &[UnifiedToken],
        context: &TaskContext,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let mut params = HashMap::with_capacity(2);
        if let Some(ref instruction) = context.instruction {
            if let Some(target) = self.extract_target_from_instruction(instruction) {
                params.insert("target".to_string(), serde_json::Value::String(target));
            }
        }
        params.insert("method".to_string(), serde_json::Value::String("semantic".to_string()));
        Ok(params)
    }

    fn create_analyze_parameters(
        &self,
        _tokens: &[UnifiedToken],
        context: &TaskContext,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let mut params = HashMap::with_capacity(2);
        params.insert("analysis_type".to_string(), serde_json::Value::String("classification".to_string()));
        if let Some(ref instruction) = context.instruction {
            params.insert("context".to_string(), serde_json::Value::String(instruction.clone()));
        }
        Ok(params)
    }

    fn create_navigate_parameters(
        &self,
        _tokens: &[UnifiedToken],
        context: &TaskContext,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let mut params = HashMap::with_capacity(2);
        if let Some(ref instruction) = context.instruction {
            if let Some(destination) = self.extract_destination_from_instruction(instruction) {
                params.insert("destination".to_string(), serde_json::Value::String(destination));
            }
        }
        params.insert("method".to_string(), serde_json::Value::String("direct".to_string()));
        Ok(params)
    }

    fn extract_target_from_instruction(&self, instruction: &str) -> Option<String> {
        let keywords = vec!["button", "link", "text", "image", "input", "menu"];
        for keyword in keywords {
            if instruction.to_lowercase().contains(keyword) {
                return Some(keyword.to_string());
            }
        }
        None
    }

    fn extract_destination_from_instruction(&self, instruction: &str) -> Option<String> {
        let nav_keywords = vec!["home", "page", "section", "menu", "settings"];
        for keyword in nav_keywords {
            if instruction.to_lowercase().contains(keyword) {
                return Some(keyword.to_string());
            }
        }
        None
    }

    fn tokens_to_text(&self, tokens: &[&UnifiedToken]) -> Result<String> {
        let common_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
            "shall", "should", "may", "might", "must", "i", "you", "he", "she", "it",
            "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
            "its", "our", "their", "this", "that", "these", "those", "some", "any", "no",
        ];

        let mut text = String::new();
        for token in tokens {
            let embed_val = embedding_lookup(token.token_id);
            let word_idx = (embed_val * 50.0 + 50.0).abs() as usize % common_words.len();
            text.push_str(common_words[word_idx]);
            text.push(' ');
        }
        Ok(text.trim().to_string())
    }

    fn generate_plan_description(
        &self,
        actions: &[Action],
        context: &TaskContext,
    ) -> Result<String> {
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

    fn estimate_duration(&self, actions: &[Action]) -> f32 {
        let mut duration = 0.0f32;
        for action in actions {
            let base = match action.action_type {
                ActionType::Click => 50.0,
                ActionType::Type => 200.0,
                ActionType::Extract => 150.0,
                ActionType::Analyze => 300.0,
                ActionType::Navigate => 100.0,
                ActionType::Wait => 500.0,
                _ => 100.0,
            };
            let param_overhead = action.parameters.len() as f32 * 10.0;
            duration += base + param_overhead;
        }
        duration
    }

    fn calculate_success_probability(&self, actions: &[Action], context: &TaskContext) -> f32 {
        let base_probability = 0.8;
        let complexity_factor = 1.0 - (actions.len() as f32 * 0.1).min(0.5);
        let constraint_factor = if context.constraints.is_empty() {
            1.0
        } else {
            1.0 - (context.constraints.len() as f32 * 0.05).min(0.3)
        };
        (base_probability * complexity_factor * constraint_factor).max(0.1)
    }
}

#[derive(Debug, Clone)]
pub struct TaskContext {
    pub task_type: TaskType,
    pub instruction: Option<String>,
    pub environment: Option<EnvironmentInfo>,
    pub constraints: Vec<String>,
    pub goals: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    pub screen_size: (usize, usize),
    pub ui_elements: Vec<UIElement>,
    pub available_actions: Vec<ActionType>,
}

pub struct ActionPlanner {
    _config: crate::multimodal::config::ActionConfig,
}

impl ActionPlanner {
    pub fn new(config: crate::multimodal::config::ActionConfig) -> Result<Self> {
        Ok(Self { _config: config })
    }

    pub fn plan(&self, context: &TaskContext) -> Result<Vec<Action>> {
        let mut actions = Vec::new();

        match context.task_type {
            TaskType::Planning => {
                actions.push(Action {
                    action_type: ActionType::Wait,
                    parameters: {
                        let mut p = HashMap::new();
                        p.insert("duration_ms".to_string(), serde_json::Value::Number(serde_json::Number::from(100)));
                        p
                    },
                    timestamp: 0.0,
                    confidence: 0.9,
                });
                actions.push(Action {
                    action_type: ActionType::Click,
                    parameters: {
                        let mut p = HashMap::new();
                        if let Some(ref instruction) = context.instruction {
                            if let Some(target) = self.extract_target(instruction) {
                                p.insert("target".to_string(), serde_json::Value::String(target));
                            }
                        }
                        p
                    },
                    timestamp: 0.1,
                    confidence: 0.85,
                });
                actions.push(Action {
                    action_type: ActionType::Type,
                    parameters: {
                        let mut p = HashMap::new();
                        if let Some(ref instruction) = context.instruction {
                            p.insert("text".to_string(), serde_json::Value::String(instruction.clone()));
                        }
                        p
                    },
                    timestamp: 0.2,
                    confidence: 0.8,
                });
            }
            TaskType::Retrieval => {
                actions.push(Action {
                    action_type: ActionType::Extract,
                    parameters: {
                        let mut p = HashMap::new();
                        p.insert("method".to_string(), serde_json::Value::String("all".to_string()));
                        p
                    },
                    timestamp: 0.0,
                    confidence: 0.85,
                });
            }
            TaskType::Classification | TaskType::Reasoning => {
                actions.push(Action {
                    action_type: ActionType::Analyze,
                    parameters: {
                        let mut p = HashMap::new();
                        p.insert("depth".to_string(), serde_json::Value::String("full".to_string()));
                        p
                    },
                    timestamp: 0.0,
                    confidence: 0.85,
                });
            }
            _ => {
                actions.push(Action {
                    action_type: ActionType::Navigate,
                    parameters: {
                        let mut p = HashMap::new();
                        if let Some(ref instruction) = context.instruction {
                            if let Some(dest) = self.extract_destination(instruction) {
                                p.insert("destination".to_string(), serde_json::Value::String(dest));
                            }
                        }
                        p
                    },
                    timestamp: 0.0,
                    confidence: 0.7,
                });
            }
        }

        Ok(actions)
    }

    fn extract_target(&self, instruction: &str) -> Option<String> {
        let keywords = vec!["button", "link", "field", "input", "menu", "icon"];
        for keyword in keywords {
            if instruction.to_lowercase().contains(keyword) {
                return Some(keyword.to_string());
            }
        }
        None
    }

    fn extract_destination(&self, instruction: &str) -> Option<String> {
        let keywords = vec!["home", "page", "section", "dashboard", "settings"];
        for keyword in keywords {
            if instruction.to_lowercase().contains(keyword) {
                return Some(keyword.to_string());
            }
        }
        None
    }
}

pub struct ActionReasoner {
    config: crate::multimodal::config::ActionConfig,
}

impl ActionReasoner {
    pub fn new(config: crate::multimodal::config::ActionConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn reason_and_select(
        &self,
        candidates: Vec<Action>,
        _context: &TaskContext,
    ) -> Result<Vec<Action>> {
        let mut filtered: Vec<_> = candidates
            .into_iter()
            .filter(|action| action.confidence > 0.5)
            .collect();
        filtered.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        filtered.truncate(self.config.max_action_steps);
        Ok(filtered)
    }
}
