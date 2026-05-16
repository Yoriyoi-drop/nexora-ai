//! Foundation model implementations (shared base implementation).
//! Each model now has its own directory-based implementation in the subdirectories.
//! This file is kept for reference but its types are deprecated in favor of the individual models.
//! 
//! To remove this file: delete it and the #[path] attributes in mod.rs

use async_trait::async_trait;
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::sync::Arc;
use std::time::Instant;

use crate::models::transformer::{CausalLM, TransformerConfig};
use crate::shared::{
    base_model::{
        FinishReason, GenerationMetadata, NxrInput, NxrModel, NxrModelError, NxrOutput,
        NxrStreamChunk, OutputData, PerformanceMetrics, ResourceUsage, StreamChunkData,
        ValidationResult, ModelStatistics, TokenOutput, InputData,
    },
    capability_spec::CapabilityVector,
    model_identity::{ModelMeta, ModelTier, NxrModelId},
    tokenizer_integration::NxrTokenizerRef,
};

fn transformer_config_for(model_id: NxrModelId) -> TransformerConfig {
    let base = TransformerConfig {
        vocab_size: 50257,
        max_seq_len: 2048,
        norm_eps: 1e-6,
        use_cache: true,
        rope_theta: 10000.0,
        ..Default::default()
    };
    match model_id {
        NxrModelId::Omnis | NxrModelId::Axiom | NxrModelId::Genesis => TransformerConfig {
            hidden_size: 768,
            num_heads: 12,
            num_kv_heads: 4,
            num_layers: 8,
            intermediate_size: 3072,
            ..base
        },
        NxrModelId::Vortex | NxrModelId::Aether | NxrModelId::Nexum => TransformerConfig {
            hidden_size: 512,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 6,
            intermediate_size: 2048,
            ..base
        },
        NxrModelId::Spectra | NxrModelId::Cipher => TransformerConfig {
            hidden_size: 384,
            num_heads: 6,
            num_kv_heads: 3,
            num_layers: 4,
            intermediate_size: 1536,
            ..base
        },
        NxrModelId::Kronos => TransformerConfig {
            hidden_size: 256,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 3,
            intermediate_size: 1024,
            ..base
        },
        NxrModelId::Swift => TransformerConfig {
            hidden_size: 128,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            intermediate_size: 512,
            ..base
        },
    }
}

fn byte_encode(text: &str) -> Vec<u32> {
    text.bytes().map(|b| b as u32).collect()
}

fn byte_decode(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().map(|&id| if id < 256 { id as u8 } else { b'?' }).collect();
    String::from_utf8_lossy(&bytes).to_string()
}

fn extract_params(input: &NxrInput) -> (usize, f32, usize) {
    let max_tokens = input.parameters.get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;
    let temperature = input.parameters.get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;
    let top_k = input.parameters.get("top_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;
    (max_tokens, temperature, top_k)
}

macro_rules! define_foundation_model {
    ($name:ident, $id:ident, $tier:ident) => {
        pub struct $name {
            pub tokenizer: Option<NxrTokenizerRef>,
            model: OnceLock<CausalLM>,
            model_config: TransformerConfig,
            inference_count: AtomicU64,
            total_generated: AtomicU64,
            total_time_ms: AtomicU64,
            start_time: chrono::DateTime<chrono::Utc>,
        }

        impl $name {
            pub fn new() -> Self {
                let config = transformer_config_for(NxrModelId::$id);
                Self {
                    tokenizer: None,
                    model: OnceLock::new(),
                    model_config: config,
                    inference_count: AtomicU64::new(0),
                    total_generated: AtomicU64::new(0),
                    total_time_ms: AtomicU64::new(0),
                    start_time: chrono::Utc::now(),
                }
            }

            pub fn with_tokenizer(t: NxrTokenizerRef) -> Self {
                let mut m = Self::new();
                m.tokenizer = Some(t);
                m
            }

            fn get_or_init_model(&self) -> &CausalLM {
                self.model.get_or_init(|| CausalLM::new(self.model_config.clone()))
            }

            fn encode_text(&self, text: &str) -> Vec<u32> {
                if let Some(ref tk) = self.tokenizer {
                    tk.read().encode(text)
                } else {
                    byte_encode(text)
                }
            }

            fn decode_ids(&self, ids: &[u32]) -> String {
                if let Some(ref tk) = self.tokenizer {
                    tk.read().decode(ids)
                } else {
                    byte_decode(ids)
                }
            }

            pub async fn infer(&self, input: &NxrInput) -> Result<NxrOutput, NxrModelError> {
                <Self as NxrModel>::infer(self, input).await
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("tokenizer", &self.tokenizer.is_some())
                    .field("model_initialized", &self.model.get().is_some())
                    .finish()
            }
        }

        impl Clone for $name {
            fn clone(&self) -> Self {
                Self {
                    tokenizer: self.tokenizer.clone(),
                    model: OnceLock::new(),
                    model_config: self.model_config.clone(),
                    inference_count: AtomicU64::new(self.inference_count.load(Ordering::Relaxed)),
                    total_generated: AtomicU64::new(self.total_generated.load(Ordering::Relaxed)),
                    total_time_ms: AtomicU64::new(self.total_time_ms.load(Ordering::Relaxed)),
                    start_time: self.start_time,
                }
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
                    .with_parameters(self.model_config.parameter_count() as u64)
                    .with_context_window(self.model_config.max_seq_len)
                })
            }

            fn capabilities(&self) -> &CapabilityVector {
                static CAPABILITIES: OnceLock<CapabilityVector> = OnceLock::new();
                CAPABILITIES.get_or_init(|| CapabilityVector::new(NxrModelId::$id))
            }

            fn config(&self) -> &Self::Config {
                static CONFIG: OnceLock<Value> = OnceLock::new();
                CONFIG.get_or_init(|| serde_json::json!({
                    "model": stringify!($id),
                    "parameters": self.model_config.parameter_count(),
                    "hidden_size": self.model_config.hidden_size,
                    "num_layers": self.model_config.num_layers,
                }))
            }

            async fn state(&self) -> Result<Self::State, NxrModelError> {
                Ok(serde_json::json!({
                    "status": if self.model.get().is_some() { "ready" } else { "uninitialized" },
                    "model": stringify!($id),
                    "inferences": self.inference_count.load(Ordering::Relaxed),
                }))
            }

            async fn initialize(&mut self, _config: Self::Config) -> Result<(), NxrModelError> {
                self.get_or_init_model();
                Ok(())
            }

            async fn reset(&self) -> Result<(), NxrModelError> {
                Ok(())
            }

            async fn metrics(&self) -> Result<Self::Metrics, NxrModelError> {
                let count = self.inference_count.load(Ordering::Relaxed);
                let tokens = self.total_generated.load(Ordering::Relaxed);
                let time_ms = self.total_time_ms.load(Ordering::Relaxed);
                Ok(serde_json::json!({
                    "inferences": count,
                    "tokens_generated": tokens,
                    "total_time_ms": time_ms,
                    "avg_tokens_per_second": if time_ms > 0 { tokens as f64 / time_ms as f64 * 1000.0 } else { 0.0 },
                }))
            }

            async fn infer(&self, input: &NxrInput) -> Result<NxrOutput, NxrModelError> {
                let model = self.get_or_init_model();

                let text = match &input.data {
                    InputData::Text(t) => t.clone(),
                    InputData::Tokens(tokens) => self.decode_ids(tokens),
                    _ => return Err(NxrModelError::Inference("Unsupported input type".to_string())),
                };

                let (max_tokens, temperature, top_k) = extract_params(input);
                let prompt_ids = self.encode_text(&text);

                let start = Instant::now();
                let (output_ids, _cache) = model.generate(&prompt_ids, max_tokens, temperature, top_k);
                let elapsed_ms = start.elapsed().as_millis() as u64;

                let output_text = self.decode_ids(&output_ids);
                let n_tokens = output_ids.len();

                self.inference_count.fetch_add(1, Ordering::Relaxed);
                self.total_generated.fetch_add(n_tokens as u64, Ordering::Relaxed);
                self.total_time_ms.fetch_add(elapsed_ms, Ordering::Relaxed);

                Ok(NxrOutput {
                    id: uuid::Uuid::new_v4(),
                    input_id: input.id,
                    timestamp: chrono::Utc::now(),
                    data: OutputData::Text(output_text),
                    metadata: GenerationMetadata {
                        finish_reason: if n_tokens < max_tokens { FinishReason::EndOfSequence } else { FinishReason::MaxTokens },
                        total_tokens: n_tokens,
                        generation_time_ms: elapsed_ms,
                        model_version: self.identity().version.clone(),
                        seed: None,
                    },
                    performance: PerformanceMetrics {
                        tokens_per_second: if elapsed_ms > 0 { n_tokens as f32 / elapsed_ms as f32 * 1000.0 } else { 0.0 },
                        memory_usage_gb: model.memory_bytes() as f32 / (1024.0 * 1024.0 * 1024.0),
                        gpu_utilization: None,
                        cpu_utilization: 100.0,
                        network_usage_mbps: None,
                    },
                })
            }

            async fn infer_stream(
                &self,
                input: &NxrInput,
                callback: Arc<dyn Fn(NxrStreamChunk) + Send + Sync>,
            ) -> Result<(), NxrModelError> {
                let model = self.get_or_init_model();

                let text = match &input.data {
                    InputData::Text(t) => t.clone(),
                    InputData::Tokens(tokens) => self.decode_ids(tokens),
                    _ => return Err(NxrModelError::Inference("Unsupported input type".to_string())),
                };

                let (max_tokens, temperature, top_k) = extract_params(input);
                let prompt_ids = self.encode_text(&text);
                let input_id = input.id;

                let mut cache = model.reset_cache();
                for &token_id in &prompt_ids {
                    model.forward(&[token_id], &mut cache);
                }

                let mut last_id = *prompt_ids.last().unwrap_or(&0);
                let mut count = 0usize;

                for pos in 0..max_tokens {
                    let logits = model.forward(&[last_id], &mut cache);
                    let next_id = crate::models::transformer::model::sample_token(&logits, temperature, top_k);
                    last_id = next_id;

                    let token_text = self.decode_ids(&[next_id]);
                    let is_final = next_id == 0 || pos == max_tokens - 1;

                    callback(NxrStreamChunk {
                        id: uuid::Uuid::new_v4(),
                        input_id,
                        timestamp: chrono::Utc::now(),
                        data: StreamChunkData::TextDelta(token_text),
                        is_final,
                    });

                    count += 1;
                    if next_id == 0 {
                        break;
                    }
                }

                self.inference_count.fetch_add(1, Ordering::Relaxed);
                self.total_generated.fetch_add(count as u64, Ordering::Relaxed);

                if count == 0 {
                    callback(NxrStreamChunk {
                        id: uuid::Uuid::new_v4(),
                        input_id,
                        timestamp: chrono::Utc::now(),
                        data: StreamChunkData::TextDelta(String::new()),
                        is_final: true,
                    });
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
                let count = self.inference_count.load(Ordering::Relaxed);
                Ok(ModelStatistics {
                    total_requests: count,
                    successful_requests: count,
                    failed_requests: 0,
                    avg_response_time_ms: if count > 0 {
                        self.total_time_ms.load(Ordering::Relaxed) as f64 / count as f64
                    } else { 0.0 },
                    total_tokens_generated: self.total_generated.load(Ordering::Relaxed),
                    uptime_seconds: (chrono::Utc::now() - self.start_time).num_seconds() as u64,
                    last_activity: Some(chrono::Utc::now()),
                })
            }

            async fn is_ready(&self) -> bool {
                true
            }

            async fn resource_usage(&self) -> Result<ResourceUsage, NxrModelError> {
                let params = self.model_config.parameter_count();
                let mem_gb = (params * 4) as f32 / (1024.0 * 1024.0 * 1024.0);
                Ok(ResourceUsage {
                    memory_gb: mem_gb,
                    cpu_percent: 5.0,
                    gpu_percent: None,
                    gpu_memory_gb: None,
                    disk_gb: 0.0,
                    network_mbps: 0.0,
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
