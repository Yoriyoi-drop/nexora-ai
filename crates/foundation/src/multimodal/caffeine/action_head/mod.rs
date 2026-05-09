//! Agentic Action Head for CAFFEINE
//! 
//! Implements Stage 5: Agentic Action Head (dari Magma)
//! - Semantic output generation
//! - Spatial grounding (bounding boxes, segmentation)
//! - Action sequence planning

pub mod semantic_output;
pub mod spatial_grounding;
pub mod action_planning;
pub mod execution;

pub use semantic_output::*;
pub use spatial_grounding::*;
pub use action_planning::*;
pub use execution::*;

use crate::caffeine::types::*;
use crate::caffeine::config::BBoxFormat as ConfigBBoxFormat;
use crate::caffeine::error::Result;
use ndarray::ArrayD;

/// Agentic Action Head
pub struct AgenticActionHead {
    semantic_output: SemanticOutputGenerator,
    spatial_grounding: SpatialGroundingModule,
    action_planning: ActionPlanningModule,
    execution_engine: ExecutionEngine,
    config: crate::caffeine::config::ActionConfig,
}

impl AgenticActionHead {
    /// Create new action head
    pub fn new(config: crate::caffeine::config::ActionConfig) -> Result<Self> {
        let semantic_output = SemanticOutputGenerator::new(config.clone())?;
        let spatial_grounding = SpatialGroundingModule::new(config.clone())?;
        let action_planning = ActionPlanningModule::new(config.clone())?;
        let execution_engine = ExecutionEngine::new(config.clone())?;
        
        Ok(Self {
            semantic_output,
            spatial_grounding,
            action_planning,
            execution_engine,
            config,
        })
    }
    
    /// Process tokens and generate multimodal outputs with actions
    pub fn process(
        &mut self,
        tokens: Vec<UnifiedToken>,
        inputs: &MultiModalInputs,
    ) -> Result<MultiModalOutputs> {
        let mut outputs = MultiModalOutputs {
            text: None,
            image: None,
            audio: None,
            video: None,
            actions: Vec::new(),
            metrics: PerformanceMetrics::default(),
        };
        
        // Generate semantic outputs if enabled
        if self.config.enable_semantic_output {
            let semantic_outputs = self.semantic_output.generate(&tokens, inputs)?;
            outputs.text = semantic_outputs.text;
            outputs.image = semantic_outputs.image;
            outputs.audio = semantic_outputs.audio;
            outputs.video = semantic_outputs.video;
        }
        
        // Generate spatial grounding if enabled
        if self.config.enable_spatial_grounding {
            let grounding_results = self.spatial_grounding.generate(&tokens, inputs)?;
            
            // Convert grounding to actions
            for grounding in grounding_results {
                let action = self.grounding_to_action(grounding)?;
                outputs.actions.push(action);
            }
        }
        
        // Generate action planning if enabled
        if self.config.enable_action_planning {
            let action_plan = self.action_planning.generate(&tokens, inputs)?;
            
            // Convert plan to executable actions
            for planned_action in action_plan.actions {
                let action_output = self.planned_action_to_output(planned_action)?;
                outputs.actions.push(action_output);
            }
        }
        
        // Execute actions if available
        if !outputs.actions.is_empty() {
            let actions: Vec<Action> = outputs.actions.iter().map(|action_output| {
                Action {
                    action_type: action_output.action.action_type.clone(),
                    parameters: action_output.action.parameters.clone(),
                    confidence: action_output.action.confidence,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as f32,
                }
            }).collect();
            let execution_results = self.execution_engine.execute_batch(&actions)?;
            
            // Update action results
            for (action_output, result) in outputs.actions.iter_mut().zip(execution_results) {
                action_output.result = result;
            }
        }
        
        Ok(outputs)
    }
    
    /// Convert spatial grounding to action
    fn grounding_to_action(&self, grounding: SpatialGrounding) -> Result<ActionOutput> {
        let action = Action {
            action_type: ActionType::Extract,
            parameters: {
                let mut params = std::collections::HashMap::new();
                
                // Add bounding boxes as parameters
                let bbox_list: Vec<serde_json::Value> = grounding.bounding_boxes
                    .into_iter()
                    .map(|bbox| {
                        serde_json::json!({
                            "coords": bbox.coords,
                            "format": format!("{:?}", bbox.format),
                            "label": bbox.label,
                            "confidence": bbox.confidence
                        })
                    })
                    .collect();
                
                params.insert("bounding_boxes".to_string(), serde_json::Value::Array(bbox_list));
                params
            },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| crate::caffeine::error::CaffeineError::output_generation(&format!("Failed to get timestamp: {}", e)))?
                .as_secs_f32(),
            confidence: grounding.confidence_scores.iter().sum::<f32>() / grounding.confidence_scores.len() as f32,
        };
        
        Ok(ActionOutput {
            action,
            result: ExecutionResult::Success,
            execution_time_ms: 0.0,
        })
    }
    
    /// Convert planned action to action output
    fn planned_action_to_output(&self, planned_action: Action) -> Result<ActionOutput> {
        Ok(ActionOutput {
            action: planned_action,
            result: ExecutionResult::Success,
            execution_time_ms: 0.0,
        })
    }
    
    /// Get action head statistics
    pub fn get_stats(&self) -> ActionHeadStats {
        ActionHeadStats {
            semantic_output_enabled: self.config.enable_semantic_output,
            spatial_grounding_enabled: self.config.enable_spatial_grounding,
            action_planning_enabled: self.config.enable_action_planning,
            max_action_steps: self.config.max_action_steps,
            bbox_format: match self.config.bbox_format {
                        ConfigBBoxFormat::XYWH => BBoxFormat::XYWH,
                        ConfigBBoxFormat::XYXY => BBoxFormat::XYXY,
                        ConfigBBoxFormat::CXCYWH => BBoxFormat::CXCYWH,
                    },
        }
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: crate::caffeine::config::ActionConfig) -> Result<()> {
        self.config = config;
        
        // Reinitialize components with new config
        self.semantic_output = SemanticOutputGenerator::new(self.config.clone())?;
        self.spatial_grounding = SpatialGroundingModule::new(self.config.clone())?;
        self.action_planning = ActionPlanningModule::new(self.config.clone())?;
        self.execution_engine = ExecutionEngine::new(self.config.clone())?;
        
        Ok(())
    }
}

/// Action head statistics
#[derive(Debug, Clone)]
pub struct ActionHeadStats {
    pub semantic_output_enabled: bool,
    pub spatial_grounding_enabled: bool,
    pub action_planning_enabled: bool,
    pub max_action_steps: usize,
    pub bbox_format: BBoxFormat,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_time_ms: 0.0,
            encoding_time_ms: 0.0,
            query_time_ms: 0.0,
            tokenization_time_ms: 0.0,
            action_time_ms: 0.0,
            memory_usage_mb: 0.0,
            gpu_utilization_percent: 0.0,
        }
    }
}
