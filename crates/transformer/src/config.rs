use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
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
    // Precision
    pub use_half_precision: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 100000,
            hidden_size: 768,
            num_heads: 12,
            num_kv_heads: 4,
            num_layers: 12,
            max_seq_len: 2048,
            intermediate_size: 3072,
            rope_theta: 10000.0,
            use_cache: true,
            norm_eps: 1e-6,
            num_experts: 0,
            top_k_experts: 0,
            expert_intermediate_size: 0,
            use_half_precision: true,
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

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0 && self.top_k_experts > 0
    }

    pub fn parameter_count(&self) -> usize {
        let head_dim = self.head_dim();
        let embedding = self.vocab_size * self.hidden_size;
        let per_layer = {
            let q = self.hidden_size * self.hidden_size;
            let k = self.hidden_size * self.num_kv_heads * head_dim;
            let v = self.hidden_size * self.num_kv_heads * head_dim;
            let o = self.num_heads * head_dim * self.hidden_size;
            let attn = q + k + v + o;
            let shared_mlp = 3 * self.hidden_size * self.intermediate_size;
            let norms = 2 * self.hidden_size;
            attn + shared_mlp + norms
        };
        // MoE: expert FFNs replace the shared MLP
        let expert_params = if self.is_moe() {
            let expert_ffn = 3 * self.hidden_size * self.expert_intermediate_size;
            self.num_experts * expert_ffn
        } else {
            0
        };
        let final_norm = self.hidden_size;
        let lm_head = self.vocab_size * self.hidden_size;
        embedding + self.num_layers * (per_layer + expert_params) + final_norm + lm_head
    }

    pub fn active_parameters(&self) -> usize {
        if !self.is_moe() {
            return self.parameter_count();
        }
        let head_dim = self.head_dim();
        let embedding = self.vocab_size * self.hidden_size;
        let per_layer = {
            let q = self.hidden_size * self.hidden_size;
            let k = self.hidden_size * self.num_kv_heads * head_dim;
            let v = self.hidden_size * self.num_kv_heads * head_dim;
            let o = self.num_heads * head_dim * self.hidden_size;
            let attn = q + k + v + o;
            let shared_mlp = 3 * self.hidden_size * self.intermediate_size;
            let norms = 2 * self.hidden_size;
            attn + shared_mlp + norms
        };
        // Only top-k experts active per forward
        let active_expert = 3 * self.hidden_size * self.expert_intermediate_size;
        let final_norm = self.hidden_size;
        let lm_head = self.vocab_size * self.hidden_size;
        embedding + self.num_layers * (per_layer + self.top_k_experts * active_expert) + final_norm + lm_head
    }

    // ── Model presets ─────────────────────────────────────────────────────────

    pub fn preset(tier: ModelTier) -> Self {
        match tier {
            ModelTier::Ultra => Self {
                vocab_size: 100000,
                hidden_size: 6144,
                num_heads: 48,
                num_kv_heads: 8,
                num_layers: 48,
                max_seq_len: 5_000_000,
                intermediate_size: 16384,
                rope_theta: 500000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 8,
                top_k_experts: 2,
                expert_intermediate_size: 8192,
                use_half_precision: true,
            },
            ModelTier::Apex => Self {
                vocab_size: 100000,
                hidden_size: 4096,
                num_heads: 32,
                num_kv_heads: 8,
                num_layers: 40,
                max_seq_len: 2_500_000,
                intermediate_size: 11008,
                rope_theta: 500000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 6,
                top_k_experts: 2,
                expert_intermediate_size: 5504,
                use_half_precision: true,
            },
            ModelTier::Pro => Self {
                vocab_size: 100000,
                hidden_size: 3200,
                num_heads: 32,
                num_kv_heads: 8,
                num_layers: 32,
                max_seq_len: 1_000_000,
                intermediate_size: 8640,
                rope_theta: 10000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 4,
                top_k_experts: 2,
                expert_intermediate_size: 4320,
                use_half_precision: true,
            },
            ModelTier::Core => Self {
                vocab_size: 100000,
                hidden_size: 3200,
                num_heads: 32,
                num_kv_heads: 8,
                num_layers: 32,
                max_seq_len: 500_000,
                intermediate_size: 8640,
                rope_theta: 10000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 0,
                top_k_experts: 0,
                expert_intermediate_size: 0,
                use_half_precision: true,
            },
            ModelTier::Edge => Self {
                vocab_size: 100000,
                hidden_size: 2048,
                num_heads: 16,
                num_kv_heads: 4,
                num_layers: 24,
                max_seq_len: 500_000,
                intermediate_size: 5632,
                rope_theta: 10000.0,
                use_cache: true,
                norm_eps: 1e-6,
                num_experts: 0,
                top_k_experts: 0,
                expert_intermediate_size: 0,
                use_half_precision: true,
            },
        }
    }
}
