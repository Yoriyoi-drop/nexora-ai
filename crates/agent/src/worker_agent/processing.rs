//! Worker agent step processing — inference-driven step execution.
//!
//! Single responsibility: handle the 8 step types (generation, analysis, etc.)
//! by calling the inference engine and producing structured output.

use serde_json::json;
use std::sync::Arc;

use crate::inference_agent::InferenceEngine;
use crate::planner_agent::StepType;
use crate::AgentError;
use crate::Result;

use super::types::build_prompt;
use super::WorkItem;

impl super::WorkerAgent {
    /// Call inference engine or return fallback message when engine unavailable.
    pub(crate) async fn infer_or_fallback(
        engine: Option<&Arc<dyn InferenceEngine>>,
        operation: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<String> {
        match engine {
            Some(eng) => eng
                .generate_tokens(uuid::Uuid::new_v4(), prompt, max_tokens)
                .await
                .map_err(|e| AgentError::ProcessingError {
                    operation: operation.into(),
                    reason: e.to_string(),
                }),
            None => Ok(format!(
                "[no engine] {} step deferred: engine unavailable",
                operation
            )),
        }
    }

    /// Execute a single step by routing to the appropriate handler based on step type.
    #[tracing::instrument(skip(self, work), fields(step_type = ?work.step.step_type, description = %work.step.description))]
    pub(crate) async fn process_step_inner(&self, work: &WorkItem) -> Result<serde_json::Value> {
        let engine = self.inference_engine.as_ref();
        let desc = &work.step.description;

        match work.step.step_type {
            StepType::Generation => {
                let prompt = build_prompt("Generate content for: ", desc);
                let result =
                    Self::infer_or_fallback(engine, "generate", &prompt, 512).await?;
                Ok(json!({
                    "type": "generation",
                    "description": desc,
                    "result": result,
                }))
            }
            StepType::Analysis => {
                let prompt = build_prompt("Analyze the following: ", desc);
                let result =
                    Self::infer_or_fallback(engine, "analysis", &prompt, 256).await?;
                Ok(json!({
                    "type": "analysis",
                    "description": desc,
                    "result": result,
                }))
            }
            StepType::Processing => {
                let prompt = build_prompt("Process the following: ", desc);
                let result =
                    Self::infer_or_fallback(engine, "processing", &prompt, 256).await?;
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
                let result =
                    Self::infer_or_fallback(engine, "validation", &prompt, 128).await?;
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
                let result =
                    Self::infer_or_fallback(engine, "data_collection", &prompt, 256).await?;
                Ok(json!({
                    "type": "data_collection",
                    "description": desc,
                    "result": result,
                }))
            }
            StepType::Communication => {
                let prompt = build_prompt("Compose a communication about: ", desc);
                let result =
                    Self::infer_or_fallback(engine, "communication", &prompt, 256).await?;
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
                let result =
                    Self::infer_or_fallback(engine, "decision", &prompt, 64).await?;
                let decision = if result.to_lowercase().contains("approve") {
                    "approved"
                } else {
                    "rejected"
                };
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
                let result =
                    Self::infer_or_fallback(engine, "custom", &prompt, 256).await?;
                Ok(json!({
                    "type": "custom",
                    "label": label,
                    "description": desc,
                    "result": result,
                }))
            }
        }
    }
}
