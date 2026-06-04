use nexora_quantization::QFormat;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Hash, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelTier {
    Ultra,
    Apex,
    Pro,
    Core,
    Edge,
}

impl ModelTier {
    pub fn label(&self) -> &'static str {
        match self {
            ModelTier::Ultra => "Ultra",
            ModelTier::Apex => "Apex",
            ModelTier::Pro => "Pro",
            ModelTier::Core => "Core",
            ModelTier::Edge => "Edge",
        }
    }

    pub fn context_window(&self) -> usize {
        match self {
            ModelTier::Ultra => 5_000_000,
            ModelTier::Apex => 2_500_000,
            ModelTier::Pro => 1_000_000,
            ModelTier::Core => 500_000,
            ModelTier::Edge => 500_000,
        }
    }
}

/// Tensor parallelism sharding configuration.
/// When `num_shards > 1`, weight matrices are split across `num_shards` ranks.
/// Each rank holds a contiguous slice of the output dimension for attention/ffn weights.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardConfig {
    /// Total number of shards (1 = no sharding).
    pub num_shards: usize,
    /// This rank's index (0..num_shards).
    pub shard_rank: usize,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self { num_shards: 1, shard_rank: 0 }
    }
}

impl ShardConfig {
    pub fn is_sharded(&self) -> bool {
        self.num_shards > 1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub use_cache: bool,
    pub norm_eps: f32,
    // MoE
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub expert_intermediate_size: usize,
    /// Shared expert (DSv4-style): N SwiGLU MLPs running on every token.
    /// 0 = disabled. Ultra/Apex/Pro use 4 shared experts.
    pub shared_expert: usize,
    // Enable domain-aware expert pools (shared + tier-specific).
    pub use_domain_experts: bool,
    // Precision — unified quantization format (F16, BF16, Q8, Q6, Q5, Q4).
    // Controls weight storage format for safetensors I/O and GPU weight upload.
    // Internal GPU paths still use `use_half_precision` / `quantize_weights` flags
    // derived from this setting.
    pub quantization: QFormat,
    /// Legacy flag — derived from `quantization` in new code.
    /// True when quantization is F16 or BF16.
    pub use_half_precision: bool,
    /// Tensor parallelism sharding config.
    /// Default: no sharding (num_shards=1).
    pub shard: ShardConfig,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 100000,
            hidden_size: 768,
            num_heads: 12,
            num_kv_heads: 1,
            num_layers: 12,
            max_seq_len: 2048,
            intermediate_size: 3072,
            rope_theta: 10000.0,
            use_cache: true,
            norm_eps: 1e-6,
            num_experts: 0,
            top_k_experts: 0,
            expert_intermediate_size: 0,
            shared_expert: 0,
            use_domain_experts: false,
            quantization: QFormat::F16,
            use_half_precision: true,
            shard: ShardConfig::default(),
        }
    }
}

impl TransformerConfig {
    pub fn head_dim(&self) -> usize {
        if self.num_heads == 0 {
            0
        } else {
            self.hidden_size / self.num_heads
        }
    }

    pub fn num_groups(&self) -> usize {
        if self.num_kv_heads == 0 {
            0
        } else {
            self.num_heads / self.num_kv_heads
        }
    }

    /// Number of attention heads for this shard.
    pub fn num_heads_local(&self) -> usize {
        if self.shard.num_shards <= 1 {
            self.num_heads
        } else {
            self.num_heads / self.shard.num_shards
        }
    }

    /// Number of KV heads for this shard.
    pub fn num_kv_heads_local(&self) -> usize {
        if self.shard.num_shards <= 1 {
            self.num_kv_heads
        } else {
            self.num_kv_heads / self.shard.num_shards
        }
    }

    /// Intermediate size for this shard's FFN layers.
    pub fn intermediate_size_local(&self) -> usize {
        if self.shard.num_shards <= 1 {
            self.intermediate_size
        } else {
            self.intermediate_size / self.shard.num_shards
        }
    }

    /// Expert intermediate size for this shard's MoE layers.
    pub fn expert_intermediate_size_local(&self) -> usize {
        if self.shard.num_shards <= 1 {
            self.expert_intermediate_size
        } else {
            self.expert_intermediate_size / self.shard.num_shards
        }
    }

    pub fn is_sharded(&self) -> bool {
        self.shard.num_shards > 1
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0 && self.top_k_experts > 0
    }

    pub fn parameter_count(&self) -> usize {
        let head_dim = self.head_dim();
        let embedding = self.vocab_size * self.hidden_size;
        let per_layer_attn = {
            let q = self.hidden_size * self.hidden_size;
            let k = self.hidden_size * self.num_kv_heads * head_dim;
            let v = self.hidden_size * self.num_kv_heads * head_dim;
            let o = self.num_heads * head_dim * self.hidden_size;
            let norms = 2 * self.hidden_size;
            q + k + v + o + norms
        };
        // MoE: routed expert FFNs (shared_expert belum diimplementasi — gak dihitung)
        let per_layer_ffn = if self.is_moe() {
            let expert_ffn = 3 * self.hidden_size * self.expert_intermediate_size;
            self.num_experts * expert_ffn
        } else {
            3 * self.hidden_size * self.intermediate_size
        };
        let final_norm = self.hidden_size;
        let lm_head = self.vocab_size * self.hidden_size;
        // Weight tying: lm_head included once in embedding count
        embedding + self.num_layers * (per_layer_attn + per_layer_ffn) + final_norm + lm_head
    }

    pub fn active_parameters(&self) -> usize {
        if !self.is_moe() {
            return self.parameter_count();
        }
        let head_dim = self.head_dim();
        let embedding = self.vocab_size * self.hidden_size;
        let per_layer_attn = {
            let q = self.hidden_size * self.hidden_size;
            let k = self.hidden_size * self.num_kv_heads * head_dim;
            let v = self.hidden_size * self.num_kv_heads * head_dim;
            let o = self.num_heads * head_dim * self.hidden_size;
            let norms = 2 * self.hidden_size;
            q + k + v + o + norms
        };
        // Active: top-k routed experts + optional shared experts
        let active_expert = 3 * self.hidden_size * self.expert_intermediate_size;
        let shared_active = self.shared_expert * active_expert;
        let per_layer_ffn_active = shared_active + self.top_k_experts * active_expert;
        let final_norm = self.hidden_size;
        let lm_head = self.vocab_size * self.hidden_size;
        embedding + self.num_layers * (per_layer_attn + per_layer_ffn_active) + final_norm + lm_head
    }

    // ── Model presets ─────────────────────────────────────────────────────────

    /// Bytes per parameter based on quantization format.
    /// F16/BF16 = 2B, Q8 = 1B, Q6 = 0.75, Q5 = 0.625, Q4 = 0.5.
    pub fn bytes_per_param(&self) -> f64 {
        self.quantization.bits_per_element() as f64 / 8.0
    }

    pub fn preset(tier: ModelTier) -> Self {
        match tier {
            ModelTier::Ultra => Self {
                vocab_size: 100000,
                hidden_size: 6144,
                num_heads: 48,
                num_kv_heads: 1,
                num_layers: 48,
                max_seq_len: 5_000_000,
                intermediate_size: 16384,
                rope_theta: 500000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 256,
                top_k_experts: 8,
                expert_intermediate_size: 4096,
                shared_expert: 4,
                use_domain_experts: true,
                quantization: QFormat::Q4 { group_size: 128 },
                use_half_precision: true,
                shard: ShardConfig::default(),
            },
            ModelTier::Apex => Self {
                vocab_size: 100000,
                hidden_size: 4096,
                num_heads: 32,
                num_kv_heads: 1,
                num_layers: 40,
                max_seq_len: 2_500_000,
                intermediate_size: 11008,
                rope_theta: 500000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 256,
                top_k_experts: 8,
                expert_intermediate_size: 2048,
                shared_expert: 4,
                use_domain_experts: true,
                quantization: QFormat::Q4 { group_size: 128 },
                use_half_precision: true,
                shard: ShardConfig::default(),
            },
            ModelTier::Pro => Self {
                vocab_size: 100000,
                hidden_size: 3200,
                num_heads: 32,
                num_kv_heads: 1,
                num_layers: 32,
                max_seq_len: 1_000_000,
                intermediate_size: 8640,
                rope_theta: 10000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 8,
                top_k_experts: 2,
                expert_intermediate_size: 2048,
                shared_expert: 4,
                use_domain_experts: true,
                quantization: QFormat::Q4 { group_size: 128 },
                use_half_precision: true,
                shard: ShardConfig::default(),
            },
            ModelTier::Core => Self {
                vocab_size: 100000,
                hidden_size: 2048,
                num_heads: 16,
                num_kv_heads: 1,
                num_layers: 20,
                max_seq_len: 500_000,
                intermediate_size: 4096,
                rope_theta: 10000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 0,
                top_k_experts: 0,
                expert_intermediate_size: 0,
                shared_expert: 0,
                use_domain_experts: false,
                quantization: QFormat::Q4 { group_size: 128 },
                use_half_precision: true,
                shard: ShardConfig::default(),
            },
            ModelTier::Edge => Self {
                vocab_size: 100000,
                hidden_size: 2048,
                num_heads: 16,
                num_kv_heads: 1,
                num_layers: 24,
                max_seq_len: 500_000,
                intermediate_size: 5632,
                rope_theta: 10000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 0,
                top_k_experts: 0,
                expert_intermediate_size: 0,
                shared_expert: 0,
                use_domain_experts: false,
                quantization: QFormat::Q4 { group_size: 128 },
                use_half_precision: true,
                shard: ShardConfig::default(),
            },
        }
    }
}
