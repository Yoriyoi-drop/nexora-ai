//! CAFFEINE: Contrastive-Aware Fusion Framework with Efficient Instruction-following and Narrative Embodiment
//!
//! Foundation model multimodal holistik yang menggabungkan:
//! - Regional Contrastive Visual Encoder (dari CLIP)
//! - Hierarchical Tri-Query Former (dari BLIP-2)
//! - Unified Discrete Multimodal Token Space (dari MIO)
//! - Instruction-Aware Components (dari LLaVA)
//! - Agentic Action Head (dari Magma)
//!
//! CAFFEINE terintegrasi dengan ATQS untuk compression dan HAS-MoE-FFN untuk expert routing.

pub mod action_head;
pub mod cache;
pub mod config;
pub mod encoders;
pub mod error;
pub mod prelude;
pub mod qformer;
pub mod tokenizer;
pub mod types;
pub mod utils;

// Re-export main components
pub use config::*;
pub use error::*;
pub use types::*;

/// Main CAFFEINE implementation
pub struct Caffeine {
    config: CaffeineConfig,
    encoders: crate::caffeine::encoders::MultiModalEncoders,
    qformer: crate::caffeine::qformer::TriQueryFormer,
    tokenizer: crate::caffeine::tokenizer::UnifiedTokenizer,
    action_head: crate::caffeine::action_head::AgenticActionHead,

    // Integration with existing modules
    atqs_compression: Option<nexora_atqs::compression::adaptive_rank::CompressionEngine>,
    has_moe_router: Option<nexora_has_moe_ffn::routing::Router>,
    has_moe_experts: Option<Vec<nexora_has_moe_ffn::experts::Expert>>,

    // #4 Multimodal Cache
    cache: Option<std::sync::Arc<std::sync::Mutex<crate::caffeine::cache::MultiModalCache>>>,
    cache_config: crate::caffeine::cache::CacheConfig,
    streaming_video: Option<crate::caffeine::cache::StreamingVideoProcessor>,
}

#[derive(Debug, Clone, Default)]
pub struct MultimodalResult {
    pub processing_summary: String,
}

pub struct CaffeineProcessor {
    caffeine: Option<Caffeine>,
    cache: Option<std::sync::Arc<std::sync::Mutex<crate::caffeine::cache::MultiModalCache>>>,
}

impl Default for CaffeineProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl CaffeineProcessor {
    pub fn new() -> Self {
        let cache = if true {
            Some(crate::caffeine::cache::global_multimodal_store())
        } else {
            None
        };
        Self { caffeine: None, cache }
    }

    pub fn with_caffeine(config: CaffeineConfig) -> crate::caffeine::error::Result<Self> {
        let cache = if config.enable_cache {
            if config.use_global_cache {
                Some(crate::caffeine::cache::global_multimodal_store())
            } else {
                let cache_config = crate::caffeine::cache::CacheConfig {
                    max_ram_bytes: config.cache_max_ram_bytes,
                    max_ssd_bytes: config.cache_max_ssd_bytes,
                    ttl_secs: config.cache_ttl_secs,
                    compression: match config.cache_compression.as_str() {
                        "INT8" => crate::caffeine::cache::CompressionFormat::INT8,
                        "FP32" => crate::caffeine::cache::CompressionFormat::FP32,
                        _ => crate::caffeine::cache::CompressionFormat::FP16,
                    },
                    enable_dedup: config.cache_enable_dedup,
                    adaptive_max_pixels: config.cache_adaptive_max_pixels,
                    enable_hierarchical: config.cache_enable_hierarchical,
                    video_window_size: config.cache_video_window_size,
                    video_stride: config.cache_video_stride,
                    cleanup_interval_secs: 300,
                };
                Some(std::sync::Arc::new(std::sync::Mutex::new(
                    crate::caffeine::cache::MultiModalCache::new(cache_config),
                )))
            }
        } else {
            None
        };

        let mut caffeine = Caffeine::new(config)?;
        caffeine.cache = cache.clone();
        Ok(Self { caffeine: Some(caffeine), cache })
    }

    /// Get a reference to the cache for stats/external access.
    pub fn cache(&self) -> Option<&std::sync::Arc<std::sync::Mutex<crate::caffeine::cache::MultiModalCache>>> {
        self.cache.as_ref()
    }

    pub async fn process_multimodal(
        &mut self,
        inputs: &crate::MultiModalInputs,
    ) -> std::result::Result<MultimodalResult, CaffeineError> {
        if let Some(ref mut caffeine) = self.caffeine {
            // Pass cache reference to forward pass (#4 FIX 6)
            if self.cache.is_some() {
                caffeine.cache = self.cache.clone();
            }
            let outputs = caffeine.forward(inputs).await?;
            return Ok(MultimodalResult {
                processing_summary: format!(
                    "caffeine multimodal: text={}, image={}, audio={}, video={}, actions={}",
                    outputs.text.is_some(),
                    outputs.image.is_some(),
                    outputs.audio.is_some(),
                    outputs.video.is_some(),
                    outputs.actions.len(),
                ),
            });
        }

        let mut parts = Vec::new();
        if inputs.text.is_some() {
            parts.push("text");
        }
        if inputs.image.is_some() {
            parts.push("image");
        }
        if inputs.audio.is_some() {
            parts.push("audio");
        }
        if inputs.video.is_some() {
            parts.push("video");
        }

        Ok(MultimodalResult {
            processing_summary: format!(
                "input modalities: {} (Caffeine not initialized)",
                parts.join(", "),
            ),
        })
    }
}

impl Caffeine {
    /// Create new CAFFEINE instance
    pub fn new(config: CaffeineConfig) -> crate::caffeine::error::Result<Self> {
        let encoders =
            crate::caffeine::encoders::MultiModalEncoders::new(config.encoders_config.clone())?;
        let qformer = crate::caffeine::qformer::TriQueryFormer::new(config.qformer_config.clone())?;
        let tokenizer =
            crate::caffeine::tokenizer::UnifiedTokenizer::new(config.tokenizer_config.clone())?;
        let action_head =
            crate::caffeine::action_head::AgenticActionHead::new(config.action_config.clone())?;

        // Initialize ATQS compression if enabled
        let atqs_compression = if config.enable_atqs_compression {
            Some(
                nexora_atqs::compression::adaptive_rank::CompressionEngine::new(
                    config.atqs_config.clone().unwrap_or_default(),
                )?,
            )
        } else {
            None
        };

        // Initialize HAS-MoE-FFN router if enabled
        let has_moe_router = if config.enable_has_moe_routing {
            Some(nexora_has_moe_ffn::routing::Router::new(
                config
                    .has_moe_config
                    .clone()
                    .unwrap_or_default()
                    .hidden_size,
                config
                    .has_moe_config
                    .clone()
                    .unwrap_or_default()
                    .num_experts,
                config.has_moe_config.clone().unwrap_or_default().top_k,
            ))
        } else {
            None
        };

        // Initialize real HAS-MoE-FFN experts with Xavier init
        let has_moe_experts = if config.enable_has_moe_routing {
            let n_experts = config
                .has_moe_config
                .as_ref()
                .map(|c| c.num_experts)
                .unwrap_or(8);
            let hidden = config.model_dim;
            let intermediate = config.hidden_dim;
            let mut experts = Vec::with_capacity(n_experts);
            for _ in 0..n_experts {
                let mut expert =
                    nexora_has_moe_ffn::experts::Expert::new(hidden, intermediate, true, 0.1);
                expert.init_random();
                experts.push(expert);
            }
            Some(experts)
        } else {
            None
        };

        let cache_config = crate::caffeine::cache::CacheConfig {
            max_ram_bytes: config.cache_max_ram_bytes,
            max_ssd_bytes: config.cache_max_ssd_bytes,
            ttl_secs: config.cache_ttl_secs,
            compression: match config.cache_compression.as_str() {
                "INT8" => crate::caffeine::cache::CompressionFormat::INT8,
                "FP32" => crate::caffeine::cache::CompressionFormat::FP32,
                _ => crate::caffeine::cache::CompressionFormat::FP16,
            },
            enable_dedup: config.cache_enable_dedup,
            adaptive_max_pixels: config.cache_adaptive_max_pixels,
            enable_hierarchical: config.cache_enable_hierarchical,
            video_window_size: config.cache_video_window_size,
            video_stride: config.cache_video_stride,
            cleanup_interval_secs: 300,
        };

        let streaming_video = if config.enable_cache {
            Some(crate::caffeine::cache::StreamingVideoProcessor::new(cache_config.clone()))
        } else {
            None
        };

        Ok(Self {
            config,
            encoders,
            qformer,
            tokenizer,
            action_head,
            atqs_compression,
            has_moe_router,
            has_moe_experts,
            cache: None,
            cache_config,
            streaming_video,
        })
    }

    /// Forward pass through CAFFEINE pipeline
    pub async fn forward(
        &mut self,
        inputs: &crate::caffeine::types::MultiModalInputs,
    ) -> crate::caffeine::error::Result<crate::caffeine::types::MultiModalOutputs> {
        // Stage 1: Multi-modal encoding (with cache — #4 FIX 1,2,8)
        let encoded_features = self.encoders.encode_with_cache(inputs, &self.cache, &self.cache_config)?;

        // Stage 2: Tri-query transformation
        let query_features = self.qformer.transform(&encoded_features)?;

        // Stage 3: Tokenization
        let tokens = self.tokenizer.tokenize(&query_features)?;

        // Stage 4: Apply ATQS compression if enabled
        let compressed_tokens = if self.atqs_compression.is_some() {
            let tensor = self.tokens_to_tensor_for_atqs(&tokens)?;
            let compressor = self
                .atqs_compression
                .as_mut()
                .ok_or_else(|| CaffeineError::Config("ATQS compression unavailable".into()))?;
            let compressed = compressor.compress_tensor_data(&tensor)?;
            self.tensor_to_tokens(&compressed, &tokens)?
        } else {
            tokens
        };

        // Stage 5: Apply HAS-MoE-FFN routing if enabled
        let routed_tokens = if self.has_moe_router.is_some() {
            // Convert to format expected by HAS-MoE-FFN
            let tensor_input = self.tokens_to_tensor(&compressed_tokens)?;
            if let Some(ref mut router) = self.has_moe_router {
                // Use real MoE softmax confidence scores instead of fake 1/(i+1)
                let routing_with_weights = router.route_with_weights(&tensor_input).map_err(|e| {
                    crate::caffeine::error::CaffeineError::HasMoeRouting(format!("{}", e))
                })?;
                let routing_decisions: Vec<nexora_has_moe_ffn::types::RoutingDecision> =
                    routing_with_weights
                        .into_iter()
                        .flatten()
                        .map(|(expert_id, confidence)| {
                            nexora_has_moe_ffn::types::RoutingDecision {
                                expert_id,
                                confidence,
                                domain: None,
                            }
                        })
                        .collect();
                self.apply_routing(compressed_tokens, routing_decisions)?
            } else {
                compressed_tokens
            }
        } else {
            compressed_tokens
        };

        // Stage 6: Action head processing
        let outputs = self.action_head.process(routed_tokens, inputs).await?;

        Ok(outputs)
    }

    /// Convert tokens to tensor for ATQS compression
    fn tokens_to_tensor_for_atqs(
        &self,
        tokens: &[crate::caffeine::types::UnifiedToken],
    ) -> crate::caffeine::error::Result<ndarray::ArrayD<f32>> {
        let mut data = Vec::with_capacity(tokens.len() * 10);
        for token in tokens {
            data.push(token.token_id as f32 / 8192.0);
            let modality_val = match token.modality {
                crate::caffeine::types::ModalityType::Text => 0.0,
                crate::caffeine::types::ModalityType::Image => 0.25,
                crate::caffeine::types::ModalityType::Audio => 0.5,
                crate::caffeine::types::ModalityType::Video => 0.75,
                crate::caffeine::types::ModalityType::Action => 1.0,
            };
            data.push(modality_val);
            data.push(token.position as f32 / 2048.0);
            data.push(token.timestamp.unwrap_or(0.0));
            if let Some((x, y, w, h)) = token.spatial_coords {
                data.push(x);
                data.push(y);
                data.push(w);
                data.push(h);
            } else {
                data.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
            }
            let embedding_sum: f32 = token.embedding.iter().sum();
            data.push(embedding_sum / token.embedding.len() as f32);
        }
        let shape = vec![tokens.len(), 10];
        Ok(ndarray::ArrayD::from_shape_vec(shape, data)?)
    }

    /// Convert tensor back to tokens after ATQS compression
    fn tensor_to_tokens(
        &self,
        tensor: &ndarray::ArrayD<f32>,
        original: &[crate::caffeine::types::UnifiedToken],
    ) -> crate::caffeine::error::Result<Vec<crate::caffeine::types::UnifiedToken>> {
        let mut result = Vec::new();
        if tensor.ndim() >= 2 {
            let rows = tensor.shape()[0].min(original.len());
            for i in 0..rows {
                let orig = &original[i];
                result.push(crate::caffeine::types::UnifiedToken {
                    token_id: (tensor[[i, 0]].max(0.0).round() as usize),
                    modality: orig.modality,
                    embedding: orig.embedding.clone(),
                    position: (tensor[[i, 2]].max(0.0).round() as usize),
                    timestamp: Some(tensor[[i, 3]]),
                    spatial_coords: if tensor[[i, 4]] > 0.0 || tensor[[i, 5]] > 0.0 {
                        Some((
                            tensor[[i, 4]],
                            tensor[[i, 5]],
                            tensor[[i, 6]],
                            tensor[[i, 7]],
                        ))
                    } else {
                        None
                    },
                });
            }
        }
        Ok(result)
    }

    /// Convert tokens to tensor format for HAS-MoE-FFN
    fn tokens_to_tensor(
        &self,
        tokens: &[crate::caffeine::types::UnifiedToken],
    ) -> crate::caffeine::error::Result<ndarray::Array2<f32>> {
        // Convert tokens to tensor representation
        let mut data = Vec::with_capacity(tokens.len() * 768); // Assuming 768-dim embeddings
        for token in tokens {
            data.extend(token.embedding.iter());
        }

        let shape = (tokens.len(), 768);
        Ok(ndarray::Array2::from_shape_vec(shape, data)?)
    }

    /// Apply routing decisions to tokens
    fn apply_routing(
        &self,
        tokens: Vec<crate::caffeine::types::UnifiedToken>,
        routing_decisions: Vec<nexora_has_moe_ffn::types::RoutingDecision>,
    ) -> crate::caffeine::error::Result<Vec<crate::caffeine::types::UnifiedToken>> {
        // Implement actual routing logic with expert selection and modality-aware processing

        if tokens.is_empty() {
            return Ok(tokens);
        }

        if routing_decisions.is_empty() {
            // If no routing decisions, apply default processing
            return self.apply_default_routing(tokens);
        }

        // Group tokens by modality for specialized routing
        let mut text_tokens = Vec::new();
        let mut image_tokens = Vec::new();
        let mut audio_tokens = Vec::new();
        let mut video_tokens = Vec::new();
        let mut action_tokens = Vec::new();

        for (i, token) in tokens.iter().enumerate() {
            match token.modality {
                crate::caffeine::types::ModalityType::Text => text_tokens.push((i, token)),
                crate::caffeine::types::ModalityType::Image => image_tokens.push((i, token)),
                crate::caffeine::types::ModalityType::Audio => audio_tokens.push((i, token)),
                crate::caffeine::types::ModalityType::Video => video_tokens.push((i, token)),
                crate::caffeine::types::ModalityType::Action => action_tokens.push((i, token)),
            }
        }

        // Apply modality-specific routing
        let mut routed_tokens = Vec::new();
        routed_tokens.resize(tokens.len(), None);

        // Route each modality group with appropriate expert selection
        self.route_modality_group(&mut routed_tokens, text_tokens, &routing_decisions, "text")?;
        self.route_modality_group(
            &mut routed_tokens,
            image_tokens,
            &routing_decisions,
            "image",
        )?;
        self.route_modality_group(
            &mut routed_tokens,
            audio_tokens,
            &routing_decisions,
            "audio",
        )?;
        self.route_modality_group(
            &mut routed_tokens,
            video_tokens,
            &routing_decisions,
            "video",
        )?;
        self.route_modality_group(
            &mut routed_tokens,
            action_tokens,
            &routing_decisions,
            "action",
        )?;

        // Convert Option<UnifiedToken> to Vec<UnifiedToken>
        let result: Vec<crate::caffeine::types::UnifiedToken> = routed_tokens
            .into_iter()
            .filter_map(|token| token)
            .collect();

        Ok(result)
    }

    /// Apply routing to a specific modality group
    fn route_modality_group(
        &self,
        routed_tokens: &mut [Option<crate::caffeine::types::UnifiedToken>],
        modality_tokens: Vec<(usize, &crate::caffeine::types::UnifiedToken)>,
        routing_decisions: &[nexora_has_moe_ffn::types::RoutingDecision],
        modality_name: &str,
    ) -> crate::caffeine::error::Result<()> {
        if modality_tokens.is_empty() {
            return Ok(());
        }

        // Select expert + confidence for each token using real MoE routing
        let assignments =
            self.select_experts_for_modality(&modality_tokens, &routing_decisions, modality_name)?;

        // Apply real has-moe-ffn expert forward, weighted by routing confidence
        for ((token_idx, original_token), (expert_id, confidence)) in
            modality_tokens.iter().zip(assignments.iter())
        {
            let transformed_token =
                self.apply_expert_transformation(original_token, *expert_id, *confidence)?;
            routed_tokens[*token_idx] = Some(transformed_token);
        }

        Ok(())
    }

    /// Select appropriate experts for a modality group.
    /// Returns Vec<(expert_id, confidence)> using real MoE routing confidence.
    fn select_experts_for_modality(
        &self,
        tokens: &[(usize, &crate::caffeine::types::UnifiedToken)],
        routing_decisions: &[nexora_has_moe_ffn::types::RoutingDecision],
        _modality_name: &str,
    ) -> crate::caffeine::error::Result<Vec<(usize, f32)>> {
        let mut expert_assignments = Vec::with_capacity(tokens.len());

        let mut sorted_decisions = routing_decisions.to_vec();
        sorted_decisions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (_token_idx, _token) in tokens {
            let best = sorted_decisions.first().cloned().unwrap_or(nexora_has_moe_ffn::types::RoutingDecision {
                expert_id: 0,
                confidence: 1.0,
                domain: None,
            });
            expert_assignments.push((best.expert_id, best.confidence));
        }

        Ok(expert_assignments)
    }

    /// Apply real has-moe-ffn Expert forward pass to a token's embedding,
    /// weighted by the routing confidence. Blends expert output with original
    /// embedding: `result = (1 - blend) * original + blend * expert_output`
    /// where `blend = confidence * 0.5`.
    fn apply_expert_transformation(
        &self,
        token: &crate::caffeine::types::UnifiedToken,
        expert_id: usize,
        confidence: f32,
    ) -> crate::caffeine::error::Result<crate::caffeine::types::UnifiedToken> {
        let has_moe_experts = self.has_moe_experts.as_ref().ok_or_else(|| {
            crate::caffeine::error::CaffeineError::HasMoeRouting(
                "MoE experts not initialized — enable has_moe_routing in config".into(),
            )
        })?;

        let expert_idx = expert_id % has_moe_experts.len();
        let expert = &has_moe_experts[expert_idx];

        // Real expert forward: [hidden] → GELU(fc1(x) + b1) → fc2 + b2 → [hidden]
        let expert_output = expert.forward(&token.embedding);

        // Blend original with expert output: stronger confidence = more expert influence
        let blend = (confidence * 0.5).clamp(0.0, 0.8);
        let blended: Vec<f32> = token
            .embedding
            .iter()
            .zip(expert_output.iter())
            .map(|(e, o)| e * (1.0 - blend) + o * blend)
            .collect();

        Ok(crate::caffeine::types::UnifiedToken {
            embedding: blended,
            ..token.clone()
        })
    }

    /// Default routing when no specific decisions are available.
    /// Uses the first expert with equal blending weight.
    fn apply_default_routing(
        &self,
        tokens: Vec<crate::caffeine::types::UnifiedToken>,
    ) -> crate::caffeine::error::Result<Vec<crate::caffeine::types::UnifiedToken>> {
        if let Some(experts) = &self.has_moe_experts {
            let processed_tokens: Vec<crate::caffeine::types::UnifiedToken> = tokens
                .into_iter()
                .map(|token| {
                    let expert_idx = 0.min(experts.len() - 1);
                    let output = experts[expert_idx].forward(&token.embedding);
                    let blended: Vec<f32> = token
                        .embedding
                        .iter()
                        .zip(output.iter())
                        .map(|(e, o)| e * 0.7 + o * 0.3)
                        .collect();
                    crate::caffeine::types::UnifiedToken {
                        embedding: blended,
                        ..token
                    }
                })
                .collect();
            Ok(processed_tokens)
        } else {
            Ok(tokens)
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CaffeineConfig {
        &self.config
    }

    /// Get performance statistics (includes cache stats — #4 Multimodal Cache)
    pub fn get_performance_stats(&self) -> crate::caffeine::types::PerformanceStats {
        let cache_mb = self
            .cache
            .as_ref()
            .and_then(|c| c.lock().ok())
            .map(|g| (g.stats().ram_usage_bytes / (1024 * 1024)) as usize)
            .unwrap_or(0);

        let cache_hit_rate = self
            .cache
            .as_ref()
            .and_then(|c| c.lock().ok())
            .map(|g| g.stats().hit_rate)
            .unwrap_or(0.0);

        crate::caffeine::types::PerformanceStats {
            total_tokens_processed: 0,
            compression_ratio: self
                .atqs_compression
                .as_ref()
                .map(|c| c.get_compression_ratio())
                .unwrap_or(1.0),
            routing_efficiency: self
                .has_moe_router
                .as_ref()
                .map(|r| r.get_routing_stats().load_balance_score)
                .unwrap_or(1.0),
            average_latency_ms: 0.0,
            memory_usage_mb: cache_mb,
            cache_hit_rate: Some(cache_hit_rate),
            cache_enabled: self.cache.is_some(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caffeine_creation() {
        let config = CaffeineConfig::default();
        let caffeine = Caffeine::new(config);
        assert!(caffeine.is_ok());
    }
}
