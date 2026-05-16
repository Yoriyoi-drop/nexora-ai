use ndarray::{Array1, Array2};
use rand::Rng;

use super::config::TransformerConfig;
use super::block::TransformerBlock;
use super::gqa::KVCacheEntry;
use super::rope::RoPE;

fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Array1::from_shape_fn(logits.len(), |i| if sum > 0.0 { exps[i] / sum } else { 1.0 / logits.len() as f32 })
}

pub(crate) fn sample_token(logits: &Array1<f32>, temperature: f32, top_k: usize) -> u32 {
    if temperature <= 0.0 {
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] { best = i; }
        }
        return best as u32;
    }
    let scaled: Array1<f32> = logits.mapv(|v| v / temperature);
    let probs = softmax(&scaled);

    if top_k > 0 && top_k < probs.len() {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(probs.len());
        let sum_k: f32 = indices[..k].iter().map(|&i| probs[i]).sum();
        if sum_k <= 0.0 { return indices[0] as u32; }
        let r: f32 = rand::random::<f32>() * sum_k;
        let mut cum = 0.0;
        for &i in &indices[..k] {
            cum += probs[i];
            if cum >= r { return i as u32; }
        }
        return indices[k - 1] as u32;
    }

    let r: f32 = rand::random();
    let mut cum = 0.0;
    for i in 0..probs.len() {
        cum += probs[i];
        if cum >= r { return i as u32; }
    }
    (probs.len() - 1) as u32
}

#[derive(Debug, Clone)]
pub struct CausalLM {
    pub config: TransformerConfig,
    pub token_embedding: Array2<f32>,
    pub blocks: Vec<TransformerBlock>,
    pub norm: super::rms_norm::RMSNorm,
    pub lm_head: Array2<f32>,
    pub rope: RoPE,
    pub precomputed_cos: Array1<f32>,
    pub precomputed_sin: Array1<f32>,
}

impl CausalLM {
    pub fn new(config: TransformerConfig) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (config.hidden_size as f32).sqrt().recip();

        let token_embedding = Array2::from_shape_fn(
            (config.vocab_size, config.hidden_size),
            |_| rng.gen::<f32>() * 2.0 * scale - scale,
        );

        let blocks = (0..config.num_layers)
            .map(|_| TransformerBlock::new(
                config.hidden_size,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim(),
                config.intermediate_size,
                config.norm_eps,
            ))
            .collect();

        let norm = super::rms_norm::RMSNorm::new(config.hidden_size, config.norm_eps);

        let lm_head = Array2::from_shape_fn(
            (config.vocab_size, config.hidden_size),
            |_| rng.gen::<f32>() * 2.0 * scale - scale,
        );

        let rope = RoPE::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let (cos_full, sin_full) = rope.precompute_freqs_cis();
        let half = config.head_dim() / 2;
        let precomputed_cos = cos_full.into_shape(config.max_seq_len * half).unwrap_or_else(|_| {
            Array1::zeros(config.max_seq_len * half)
        });
        let precomputed_sin = sin_full.into_shape(config.max_seq_len * half).unwrap_or_else(|_| {
            Array1::zeros(config.max_seq_len * half)
        });

        Self {
            config,
            token_embedding,
            blocks,
            norm,
            lm_head,
            rope,
            precomputed_cos,
            precomputed_sin,
        }
    }

    pub fn forward(&self, input_ids: &[u32], kv_cache: &mut Vec<KVCacheEntry>) -> Array1<f32> {
        let batch_size = 1;
        let seq_len = input_ids.len();

        let mut h = Array2::zeros((batch_size, self.config.hidden_size));

        if let Some(&token_id) = input_ids.last() {
            let tid = (token_id as usize).min(self.config.vocab_size - 1);
            for j in 0..self.config.hidden_size {
                h[[0, j]] = self.token_embedding[[tid, j]];
            }
        }

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let pos = kv_cache.first().map(|e| e.k.shape()[0]).unwrap_or(0);
            let half = self.config.head_dim() / 2;
            let cos_slice = if pos * half < self.precomputed_cos.len() {
                self.precomputed_cos.slice(ndarray::s![pos * half..(pos + 1) * half]).to_owned()
            } else {
                Array1::zeros(half)
            };
            let sin_slice = if pos * half < self.precomputed_sin.len() {
                self.precomputed_sin.slice(ndarray::s![pos * half..(pos + 1) * half]).to_owned()
            } else {
                Array1::zeros(half)
            };

            h = block.forward(&h, kv_cache, layer_idx, &cos_slice, &sin_slice);
        }

        h = self.norm.forward(&h);

        let h_row = h.row(0);
        let mut logits = Array1::zeros(self.config.vocab_size);
        for i in 0..self.config.vocab_size {
            let mut dot = 0.0;
            for j in 0..self.config.hidden_size {
                dot += h_row[j] * self.lm_head[[i, j]];
            }
            logits[i] = dot;
        }

        logits
    }

    pub fn reset_cache(&self) -> Vec<KVCacheEntry> {
        Vec::with_capacity(self.config.num_layers)
    }

    pub fn parameter_count(&self) -> usize {
        let mut count = self.token_embedding.len();
        count += self.lm_head.len();
        for block in &self.blocks {
            count += block.attention.wq.len();
            count += block.attention.wk.len();
            count += block.attention.wv.len();
            count += block.attention.wo.len();
            count += block.ffn.w1.len();
            count += block.ffn.w2.len();
            count += block.ffn.w3.len();
            count += block.attention_norm.weight.len();
            count += block.ffn_norm.weight.len();
        }
        count += self.norm.weight.len();
        count
    }

    pub fn from_checkpoint(config: TransformerConfig, path: &str) -> crate::FoundationResult<Self> {
        let mut model = Self::new(config);
        let loaded = crate::safetensors::load_safetensors(path)?;
        let get_arr = |name: &str| -> crate::FoundationResult<ndarray::ArrayD<f32>> {
            loaded.get(name).cloned().ok_or_else(|| {
                crate::FoundationError::Implementation(format!("Missing tensor: {}", name))
            })
        };
        model.token_embedding = get_arr("token_embedding")?
            .into_dimensionality::<ndarray::Ix2>().unwrap().to_owned();
        model.lm_head = get_arr("lm_head")?
            .into_dimensionality::<ndarray::Ix2>().unwrap().to_owned();
        model.norm.weight = get_arr("norm.weight")?
            .into_dimensionality::<ndarray::Ix1>().unwrap().to_owned();
        for (i, block) in model.blocks.iter_mut().enumerate() {
            let prefix = format!("blocks.{}.", i);
            block.attention_norm.weight = get_arr(&format!("{}attention_norm.weight", prefix))?
                .into_dimensionality::<ndarray::Ix1>().unwrap().to_owned();
            block.ffn_norm.weight = get_arr(&format!("{}ffn_norm.weight", prefix))?
                .into_dimensionality::<ndarray::Ix1>().unwrap().to_owned();
            block.attention.wq = get_arr(&format!("{}attention.wq", prefix))?
                .into_dimensionality::<ndarray::Ix2>().unwrap().to_owned();
            block.attention.wk = get_arr(&format!("{}attention.wk", prefix))?
                .into_dimensionality::<ndarray::Ix2>().unwrap().to_owned();
            block.attention.wv = get_arr(&format!("{}attention.wv", prefix))?
                .into_dimensionality::<ndarray::Ix2>().unwrap().to_owned();
            block.attention.wo = get_arr(&format!("{}attention.wo", prefix))?
                .into_dimensionality::<ndarray::Ix2>().unwrap().to_owned();
            block.ffn.w1 = get_arr(&format!("{}ffn.w1", prefix))?
                .into_dimensionality::<ndarray::Ix2>().unwrap().to_owned();
            block.ffn.w2 = get_arr(&format!("{}ffn.w2", prefix))?
                .into_dimensionality::<ndarray::Ix2>().unwrap().to_owned();
            block.ffn.w3 = get_arr(&format!("{}ffn.w3", prefix))?
                .into_dimensionality::<ndarray::Ix2>().unwrap().to_owned();
        }
        Ok(model)
    }

    pub fn memory_bytes(&self) -> usize {
        self.parameter_count() * 4
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        let mut cache = self.reset_cache();

        for &token_id in prompt_ids {
            self.forward(&[token_id], &mut cache);
        }

        let mut output = Vec::new();
        let mut last_id = *prompt_ids.last().unwrap_or(&0);

        for _ in 0..max_tokens {
            let logits = self.forward(&[last_id], &mut cache);
            let next_id = sample_token(&logits, temperature, top_k);
            output.push(next_id);
            if next_id == 0 {
                break;
            }
            last_id = next_id;
        }

        (output, cache)
    }
}
