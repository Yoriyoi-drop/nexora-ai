//! Controller Core - Core logic dan routing untuk Core Controller

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::{debug, info, warn};

use crate::error::{CoreError, CoreResult};
use crate::types::{ContextInfo, IntentResult, IntentType, InputData, InputType, ModelId};
use parking_lot::RwLock as ParkingRwLock;

use super::controller_types::LruContextCache;
use crate::execution::controller_models::ModelProcessor;
use super::controller_types::{ControllerMetrics, ControllerState, RoutingDecision, SpecialistModel};

/// Core routing dan logic implementations
pub struct ControllerCore;

impl ControllerCore {
    pub fn map_intent_to_model(intent: IntentType) -> ModelId {
        match intent {
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
        }
    }

    pub fn calculate_routing_confidence(intent: IntentType, context: &ContextInfo) -> f32 {
        let base_confidence = match intent {
            IntentType::Coding => 0.9,
            IntentType::Memory => 0.8,
            IntentType::Debugging => 0.85,
            IntentType::Planning => 0.8,
            IntentType::Reasoning => 0.75,
            IntentType::Ranking => 0.7,
            IntentType::Retrieval => 0.8,
            IntentType::Validation => 0.75,
            IntentType::Personality => 0.7,
            IntentType::Optimization => 0.8,
            IntentType::Unknown => 0.3,
        };
        
        let adjusted_confidence = base_confidence * (0.5 + context.context_relevance * 0.5);
        adjusted_confidence.clamp(0.0, 1.0)
    }

    pub fn get_secondary_models(intents: &[crate::types::IntentScore]) -> Vec<ModelId> {
        let mut secondary_models = Vec::new();
        
        let mut sorted_intents = intents.to_vec();
        sorted_intents.sort_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        for intent_score in sorted_intents.iter().skip(1).take(3) {
            if intent_score.confidence > 0.7 {
                let model = Self::map_intent_to_model(intent_score.intent_type);
                if !secondary_models.contains(&model) && model != ModelId::Controller {
                    secondary_models.push(model);
                }
            }
        }
        
        secondary_models
    }

    pub fn generate_context_key(input_data: &InputData) -> String {
        let mut hasher = DefaultHasher::new();
        input_data.raw_input.hash(&mut hasher);
        input_data.input_type.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    pub fn find_alternative_model(
        specialist_models: &ParkingRwLock<HashMap<String, Box<dyn SpecialistModel>>>,
        target_model: ModelId,
        intent: &IntentResult,
    ) -> Option<ModelId> {
        let models = specialist_models.read();
        
        for (_name, model) in models.iter() {
            if model.model_type() != target_model {
                let capabilities = model.capabilities();
                let intent_name = format!("{:?}", intent.primary_intent);
                
                if capabilities.iter().any(|cap| cap.to_lowercase().contains(&intent_name.to_lowercase())) {
                    return Some(model.model_type());
                }
            }
        }
        
        None
    }

    pub async fn route_to_alternative_model(
        model: ModelId,
        intent: &IntentResult,
        context: &ContextInfo,
    ) -> CoreResult<RoutingDecision> {
        let routing_confidence = Self::calculate_routing_confidence(intent.primary_intent, context) * 0.8;
        
        Ok(RoutingDecision {
            target_model: model,
            routed_query: context.current_context.clone(),
            routing_confidence,
            routing_reasoning: format!(
                "Routed to alternative {} for intent {} (primary unavailable)",
                model.name(),
                intent.primary_intent.name()
            ),
            requires_multi_model: false,
            secondary_models: Vec::new(),
        })
    }

    pub async fn execute_task(
        routing: &RoutingDecision,
        original_input: &str,
        specialist_models: &ParkingRwLock<HashMap<String, Box<dyn SpecialistModel>>>,
    ) -> CoreResult<String> {
        debug!("Executing task on model: {:?}", routing.target_model);
        
        // Try specialist models first
        let models = specialist_models.read();
        let mut specialist_result = None;
        
        for (name, model) in models.iter() {
            if model.model_type() == routing.target_model {
                let context = ContextInfo::new(
                    routing.routed_query.clone(),
                    routing.target_model,
                );
                
                match model.process(original_input, &context).await {
                    Ok(result) => {
                        specialist_result = Some(result);
                        debug!("Specialist model {} processed successfully", name);
                        break;
                    }
                    Err(e) => {
                        warn!("Specialist model {} failed: {}", name, e);
                    }
                }
            }
        }
        
        // Fallback to built-in processing
        let model_result = if let Some(result) = specialist_result {
            result
        } else {
            ModelProcessor::process_with_model(routing.target_model, original_input).await?
        };
        
        // Handle multi-model coordination if needed
        let final_result = if routing.requires_multi_model {
            Self::coordinate_multi_models(routing, original_input, &model_result).await?
        } else {
            model_result
        };
        
        // Format the final response
        let result = format!(
            "=== Model Processing Result ===\n\
            Target Model: {}\n\
            Routing Confidence: {:.2}%\n\
            Routing Reasoning: {}\n\
            Multi-Model Coordination: {}\n\
            Secondary Models: {}\n\
            Processing Result:\n{}\n\
            === End Result ===",
            routing.target_model.name(),
            routing.routing_confidence * 100.0,
            routing.routing_reasoning,
            if routing.requires_multi_model { "Enabled" } else { "Disabled" },
            routing.secondary_models.iter()
                .map(|m| m.name().to_string())
                .collect::<Vec<_>>()
                .join(", "),
            final_result
        );
        
        Ok(result)
    }

    async fn coordinate_multi_models(
        routing: &RoutingDecision,
        input: &str,
        primary_result: &str,
    ) -> CoreResult<String> {
        let mut combined_results = vec![primary_result.to_string()];
        
        // Process with secondary models
        for &model_id in &routing.secondary_models {
            let secondary_result = ModelProcessor::process_with_model(model_id, input).await?;
            combined_results.push(secondary_result);
        }
        
        // Combine and synthesize results
        let combined = combined_results.join(" | ");
        Ok(format!("Synthesized: {}", combined))
    }

    pub fn current_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}
