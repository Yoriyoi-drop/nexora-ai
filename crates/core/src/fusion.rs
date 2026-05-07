//! Response fusion untuk Nexora Core

use crate::types::{ModelId, FusionResult, TaskExecution};
use crate::error::CoreResult;
use tracing::{debug, info};

/// Response fusion untuk menggabungkan response dari multiple models
pub struct ResponseFusion {
    fusion_threshold: f32,
}

impl ResponseFusion {
    pub fn new() -> Self {
        Self {
            fusion_threshold: 0.7,
        }
    }
    
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.fusion_threshold = threshold.clamp(0.0, 1.0);
        self
    }
    
    /// Fuse responses dari multiple models
    pub async fn fuse_responses(&self, tasks: &[TaskExecution]) -> CoreResult<FusionResult> {
        debug!("Fusing responses from {} tasks", tasks.len());
        
        let mut model_responses = Vec::new();
        let mut response_sources = Vec::new();
        
        for task in tasks {
            if let Some(output) = &task.task_output {
                model_responses.push(output.clone());
                response_sources.push(task.assigned_model);
            }
        }
        
        if model_responses.is_empty() {
            return Err(crate::error::CoreError::Fusion("No responses to fuse".to_string()));
        }
        
        let fused_response = self.generate_fused_response(&model_responses, &response_sources);
        let fusion_confidence = self.calculate_fusion_confidence(&model_responses);
        let has_conflicts = self.detect_conflicts(&model_responses);
        
        let fusion_result = FusionResult {
            model_responses: model_responses.clone(),
            response_sources,
            response_count: model_responses.len(),
            fused_response,
            fusion_confidence,
            has_conflicts,
            conflict_descriptions: if has_conflicts {
                self.describe_conflicts(&model_responses)
            } else {
                Vec::new()
            },
        };
        
        info!("Response fusion completed: confidence={:.2}, conflicts={}", 
              fusion_result.fusion_confidence, fusion_result.has_conflicts);
        
        Ok(fusion_result)
    }
    
    fn generate_fused_response(&self, responses: &[String], sources: &[ModelId]) -> String {
        if responses.len() == 1 {
            return responses[0].clone();
        }
        
        let mut fused = String::new();
        fused.push_str("Fused Response from Multiple Models:\n\n");
        
        for (i, (response, source)) in responses.iter().zip(sources.iter()).enumerate() {
            fused.push_str(&format!("{}. {}:\n", i + 1, source.name()));
            fused.push_str(&format!("   {}\n\n", response));
        }
        
        fused.push_str("Summary: Combined insights from multiple specialist models.");
        fused
    }
    
    fn calculate_fusion_confidence(&self, responses: &[String]) -> f32 {
        // Simplified confidence calculation
        let base_confidence = 0.8;
        let response_factor = (responses.len() as f32).min(3.0) / 3.0; // Max benefit from 3 responses
        
        (base_confidence + response_factor * 0.2).clamp(0.0, 1.0)
    }
    
    fn detect_conflicts(&self, responses: &[String]) -> bool {
        // Simplified conflict detection - check for contradictory statements
        if responses.len() < 2 {
            return false;
        }
        
        // Look for obvious contradictions (simplified)
        let mut has_positive = false;
        let mut has_negative = false;
        
        for response in responses {
            let response_lower = response.to_lowercase();
            if response_lower.contains("yes") || response_lower.contains("true") || response_lower.contains("success") {
                has_positive = true;
            }
            if response_lower.contains("no") || response_lower.contains("false") || response_lower.contains("error") {
                has_negative = true;
            }
        }
        
        has_positive && has_negative
    }
    
    fn describe_conflicts(&self, responses: &[String]) -> Vec<String> {
        let mut conflicts = Vec::new();
        
        if responses.len() >= 2 {
            let response_lower1 = responses[0].to_lowercase();
            let response_lower2 = responses[1].to_lowercase();
            
            if (response_lower1.contains("yes") && response_lower2.contains("no")) ||
               (response_lower1.contains("true") && response_lower2.contains("false")) {
                conflicts.push("Contradictory yes/no responses detected".to_string());
            }
            
            if (response_lower1.contains("error") && !response_lower2.contains("error")) ||
               (!response_lower1.contains("error") && response_lower2.contains("error")) {
                conflicts.push("Divergent error status between responses".to_string());
            }
        }
        
        conflicts
    }
}

impl Default for ResponseFusion {
    fn default() -> Self {
        Self::new()
    }
}
