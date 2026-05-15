use async_trait::async_trait;
use serde_json::Value;
use std::sync::{Arc, OnceLock};

use crate::shared::{
    base_model::{
        FinishReason, GenerationMetadata, NxrInput, NxrModel, NxrModelError, NxrOutput,
        NxrStreamChunk, OutputData, PerformanceMetrics, ResourceUsage, StreamChunkData,
        ValidationResult, ModelStatistics, TokenOutput,
    },
    capability_spec::CapabilityVector,
    model_identity::{ModelMeta, ModelTier, NxrModelId},
    tokenizer_integration::NxrTokenizerRef,
};

macro_rules! define_foundation_model {
    ($name:ident, $id:ident, $tier:ident) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            pub tokenizer: Option<NxrTokenizerRef>,
        }

        impl Default for $name {
            fn default() -> Self {
                Self { tokenizer: None }
            }
        }

        impl $name {
            pub fn new() -> Self {
                Self::default()
            }

            pub fn with_tokenizer(t: NxrTokenizerRef) -> Self {
                Self { tokenizer: Some(t) }
            }

            pub async fn infer(&self, input: &NxrInput) -> Result<NxrOutput, NxrModelError> {
                <Self as NxrModel>::infer(self, input).await
            }
        }

        #[async_trait]
        impl NxrModel for $name {
            type Config = Value;
            type Metrics = Value;
            type State = Value;

            fn identity(&self) -> &ModelMeta {
                static META: OnceLock<ModelMeta> = OnceLock::new();
                META.get_or_init(|| {
                    ModelMeta::new(
                        NxrModelId::$id,
                        ModelTier::$tier,
                        "0.1.0".to_string(),
                        format!("Foundation model for {}", NxrModelId::$id.fullname()),
                    )
                })
            }

            fn capabilities(&self) -> &CapabilityVector {
                static CAPABILITIES: OnceLock<CapabilityVector> = OnceLock::new();
                CAPABILITIES.get_or_init(|| CapabilityVector::new(NxrModelId::$id))
            }

            fn config(&self) -> &Self::Config {
                static CONFIG: OnceLock<Value> = OnceLock::new();
                CONFIG.get_or_init(|| serde_json::json!({ "model": stringify!($id) }))
            }

            async fn state(&self) -> Result<Self::State, NxrModelError> {
                Ok(serde_json::json!({ "status": "ready", "model": stringify!($id) }))
            }

            async fn initialize(&mut self, _config: Self::Config) -> Result<(), NxrModelError> {
                Ok(())
            }

            async fn reset(&self) -> Result<(), NxrModelError> {
                Ok(())
            }

            async fn metrics(&self) -> Result<Self::Metrics, NxrModelError> {
                Ok(serde_json::json!({
                    "requests": 0,
                    "inference_time_ms": 0,
                    "tokens_generated": 0,
                }))
            }

            async fn infer(&self, input: &NxrInput) -> Result<NxrOutput, NxrModelError> {
                let (text, input_tokens) = match &input.data {
                    crate::shared::base_model::InputData::Text(text) => (text.clone(), None),
                    crate::shared::base_model::InputData::Tokens(tokens) => {
                        if let Some(tk) = &self.tokenizer {
                            (tk.read().decode(tokens), Some(tokens.clone()))
                        } else {
                            (format!("<{} token_ids>", tokens.len()), Some(tokens.clone()))
                        }
                    }
                    _ => ("structured input received".to_string(), None),
                };

                let model_name = stringify!($id);
                let (output_data, n_tokens, real_ids) = if let Some(tk) = &self.tokenizer {
                    let tok = tk.read();
                    let ids = input_tokens.unwrap_or_else(|| tok.encode(&text));
                    let count = ids.len();
                    let token_outputs: Vec<TokenOutput> = ids.iter().enumerate().map(|(pos, &id)| {
                        TokenOutput {
                            token_id: id,
                            text: tok.id_to_token(id).cloned().unwrap_or_default(),
                            log_prob: 0.0,
                            position: pos,
                        }
                    }).collect();
                    (OutputData::Tokens(token_outputs), count, Some(ids))
                } else {
                    let count = text.split_whitespace().count();
                    (OutputData::Text(format!("[{}] Processed input ({} tokens): {}", model_name, count, text)), count, None)
                };

                Ok(NxrOutput {
                    id: uuid::Uuid::new_v4(),
                    input_id: input.id,
                    timestamp: chrono::Utc::now(),
                    data: output_data,
                    metadata: GenerationMetadata {
                        finish_reason: FinishReason::EndOfSequence,
                        total_tokens: n_tokens,
                        generation_time_ms: n_tokens as u64,
                        model_version: self.identity().version.clone(),
                        seed: None,
                    },
                    performance: PerformanceMetrics {
                        tokens_per_second: n_tokens as f32,
                        memory_usage_gb: 0.1,
                        gpu_utilization: None,
                        cpu_utilization: 5.0,
                        network_usage_mbps: None,
                    },
                })
            }

            async fn infer_stream(
                &self,
                input: &NxrInput,
                callback: Arc<dyn Fn(NxrStreamChunk) + Send + Sync>,
            ) -> Result<(), NxrModelError> {
                let text = match &input.data {
                    crate::shared::base_model::InputData::Text(text) => text.clone(),
                    crate::shared::base_model::InputData::Tokens(_) => {
                        if let Some(tk) = &self.tokenizer {
                            if let crate::shared::base_model::InputData::Tokens(tokens) = &input.data {
                                tk.read().decode(tokens)
                            } else { String::new() }
                        } else { "structured input".to_string() }
                    }
                    _ => "structured input".to_string(),
                };

                let input_id = input.id;

                if let Some(tk) = &self.tokenizer {
                    let tok = tk.read();
                    let ids = tok.encode(&text);
                    for (pos, &id) in ids.iter().enumerate() {
                        let token_text = tok.id_to_token(id).cloned().unwrap_or_default();
                        callback(NxrStreamChunk {
                            id: uuid::Uuid::new_v4(),
                            input_id,
                            timestamp: chrono::Utc::now(),
                            data: StreamChunkData::TokenDelta(TokenOutput {
                                token_id: id,
                                text: token_text,
                                log_prob: 0.0,
                                position: pos,
                            }),
                            is_final: pos == ids.len() - 1,
                        });
                    }
                    if ids.is_empty() {
                        callback(NxrStreamChunk {
                            id: uuid::Uuid::new_v4(),
                            input_id,
                            timestamp: chrono::Utc::now(),
                            data: StreamChunkData::TextDelta(String::new()),
                            is_final: true,
                        });
                    }
                } else {
                    let words: Vec<&str> = text.split_whitespace().collect();
                    for (i, word) in words.iter().enumerate() {
                        callback(NxrStreamChunk {
                            id: uuid::Uuid::new_v4(),
                            input_id,
                            timestamp: chrono::Utc::now(),
                            data: StreamChunkData::TextDelta(format!("{} ", word)),
                            is_final: i == words.len() - 1,
                        });
                    }
                    if words.is_empty() {
                        callback(NxrStreamChunk {
                            id: uuid::Uuid::new_v4(),
                            input_id,
                            timestamp: chrono::Utc::now(),
                            data: StreamChunkData::TextDelta(String::new()),
                            is_final: true,
                        });
                    }
                }

                Ok(())
            }

            async fn update_config(&mut self, _config: Self::Config) -> Result<(), NxrModelError> {
                Ok(())
            }

            async fn validate(&self) -> Result<ValidationResult, NxrModelError> {
                Ok(ValidationResult {
                    is_valid: true,
                    errors: Vec::new(),
                    warnings: Vec::new(),
                    score: 1.0,
                })
            }

            async fn statistics(&self) -> Result<ModelStatistics, NxrModelError> {
                Ok(ModelStatistics::default())
            }

            async fn is_ready(&self) -> bool {
                true
            }

            async fn resource_usage(&self) -> Result<ResourceUsage, NxrModelError> {
                Ok(ResourceUsage {
                    memory_gb: 0.1,
                    cpu_percent: 5.0,
                    gpu_percent: None,
                    gpu_memory_gb: None,
                    disk_gb: 0.0,
                    network_mbps: 5.0,
                    active_connections: 0,
                    queue_size: 0,
                })
            }
        }
    };
}

define_foundation_model!(NxrOmnisModel, Omnis, Ultra);
define_foundation_model!(NxrVortexModel, Vortex, Apex);
define_foundation_model!(NxrAetherModel, Aether, Apex);
define_foundation_model!(NxrSpectraModel, Spectra, Pro);
define_foundation_model!(NxrNexumModel, Nexum, Apex);
define_foundation_model!(NxrAxiomModel, Axiom, Ultra);
define_foundation_model!(NxrCipherModel, Cipher, Pro);
define_foundation_model!(NxrSwiftModel, Swift, Edge);
define_foundation_model!(NxrKronosModel, Kronos, Core);
define_foundation_model!(NxrGenesisModel, Genesis, Ultra);
