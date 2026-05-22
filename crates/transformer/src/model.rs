use ndarray::{Array1, Array2};
use rand::Rng;

use super::block::TransformerBlock;
use super::config::TransformerConfig;
use super::gqa::{KVCacheEntry, PagedCacheReader};
use super::rope::RoPE;

fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Array1::from_shape_fn(logits.len(), |i| {
        if sum > 0.0 {
            exps[i] / sum
        } else {
            1.0 / logits.len() as f32
        }
    })
}

#[cfg(feature = "gpu")]
fn sample_token_gpu(logits: &Array1<f32>, temperature: f32, top_k: usize, seed: u64) -> u32 {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    if let Ok(ctx) = GpuContext::global() {
        let shape = vec![1, logits.len()];
        let cpu = ndarray::ArrayD::from_shape_vec(shape.clone(), logits.to_vec()).unwrap();
        let gpu = GpuTensor::from_cpu(&cpu).unwrap();
        match ctx.gpu_sample(&gpu, temperature, top_k as u32, seed) {
            Ok(result) => {
                let raw = result.to_cpu_raw_bytes();
                u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]])
            }
            Err(_) => sample_token(logits, temperature, top_k),
        }
    } else {
        sample_token(logits, temperature, top_k)
    }
}

pub fn sample_token(logits: &Array1<f32>, temperature: f32, top_k: usize) -> u32 {
    if temperature <= 0.0 {
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        return best as u32;
    }
    let scaled: Array1<f32> = logits.mapv(|v| v / temperature);
    let probs = softmax(&scaled);

    if top_k > 0 && top_k < probs.len() {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let k = top_k.min(probs.len());
        let sum_k: f32 = indices[..k].iter().map(|&i| probs[i]).sum();
        if sum_k <= 0.0 {
            return indices[0] as u32;
        }
        let r: f32 = rand::random::<f32>() * sum_k;
        let mut cum = 0.0;
        for &i in &indices[..k] {
            cum += probs[i];
            if cum >= r {
                return i as u32;
            }
        }
        return indices[k - 1] as u32;
    }

    let r: f32 = rand::random();
    let mut cum = 0.0;
    for i in 0..probs.len() {
        cum += probs[i];
        if cum >= r {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

#[derive(Debug)]
pub struct CausalLM {
    pub config: TransformerConfig,
    pub token_embedding: Array2<f32>,
    pub blocks: Vec<TransformerBlock>,
    pub norm: super::rms_norm::RMSNorm,
    pub lm_head: Array2<f32>,
    pub rope: RoPE,
    pub precomputed_cos: Array1<f32>,
    pub precomputed_sin: Array1<f32>,
    #[cfg(feature = "gpu")]
    pub(crate) lm_head_gpu: std::sync::Mutex<Option<nexora_autograd::gpu::GpuTensor>>,
}

#[cfg(not(feature = "gpu"))]
impl Clone for CausalLM {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            token_embedding: self.token_embedding.clone(),
            blocks: self.blocks.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
            rope: self.rope.clone(),
            precomputed_cos: self.precomputed_cos.clone(),
            precomputed_sin: self.precomputed_sin.clone(),
        }
    }
}

#[cfg(feature = "gpu")]
impl Clone for CausalLM {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            token_embedding: self.token_embedding.clone(),
            blocks: self.blocks.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
            rope: self.rope.clone(),
            precomputed_cos: self.precomputed_cos.clone(),
            precomputed_sin: self.precomputed_sin.clone(),
            lm_head_gpu: std::sync::Mutex::new(None),
        }
    }
}

impl CausalLM {
    pub fn new(config: TransformerConfig) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (config.hidden_size as f32).sqrt().recip();

        let token_embedding =
            Array2::from_shape_fn((config.vocab_size, config.hidden_size), |_| {
                rng.gen::<f32>() * 2.0 * scale - scale
            });

        let blocks = (0..config.num_layers)
            .map(|_| {
                TransformerBlock::new(
                    config.hidden_size,
                    config.num_heads,
                    config.num_kv_heads,
                    config.head_dim(),
                    config.intermediate_size,
                    config.norm_eps,
                )
            })
            .collect();

        let norm = super::rms_norm::RMSNorm::new(config.hidden_size, config.norm_eps);

        let lm_head = Array2::from_shape_fn((config.vocab_size, config.hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });

        let rope = RoPE::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let (cos_full, sin_full) = rope.precompute_freqs_cis();
        let half = config.head_dim() / 2;
        let precomputed_cos = cos_full
            .into_shape(config.max_seq_len * half)
            .unwrap_or_else(|_| Array1::zeros(config.max_seq_len * half));
        let precomputed_sin = sin_full
            .into_shape(config.max_seq_len * half)
            .unwrap_or_else(|_| Array1::zeros(config.max_seq_len * half));

        Self {
            config,
            token_embedding,
            blocks,
            norm,
            lm_head,
            rope,
            precomputed_cos,
            precomputed_sin,
            #[cfg(feature = "gpu")]
            lm_head_gpu: std::sync::Mutex::new(None),
        }
    }

    pub fn forward(&self, input_ids: &[u32], kv_cache: &mut Vec<KVCacheEntry>) -> Array1<f32> {
        #[cfg(feature = "gpu")]
        if let Ok(logits) = self.forward_gpu(input_ids, kv_cache) {
            return logits;
        }

        let batch_size = 1;

        let mut h = Array2::zeros((batch_size, self.config.hidden_size));

        if let Some(&token_id) = input_ids.last() {
            let tid = token_id as usize;
            assert!(
                tid < self.config.vocab_size,
                "Token ID {} out of range [0, {})",
                tid,
                self.config.vocab_size
            );
            h.row_mut(0).assign(&self.token_embedding.row(tid));
        }

        let pos = kv_cache.first().map(|e| e.k.shape()[0]).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        let cos_slice = if pos * half < self.precomputed_cos.len() {
            self.precomputed_cos
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };
        let sin_slice = if pos * half < self.precomputed_sin.len() {
            self.precomputed_sin
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, kv_cache, layer_idx, &cos_slice, &sin_slice);
        }

        h = self.norm.forward(&h);

        let logits = h.row(0).dot(&self.lm_head.t());

        logits
    }

    pub fn reset_cache(&self) -> Vec<KVCacheEntry> {
        Vec::with_capacity(self.config.num_layers)
    }

    /// Forward pass using a paged KV cache.
    /// `token_pos` is the current position being generated.
    pub fn forward_paged(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn PagedCacheReader,
        seq_id: u64,
    ) -> Array1<f32> {
        let batch_size = 1;
        let mut h = Array2::zeros((batch_size, self.config.hidden_size));

        if let Some(&token_id) = input_ids.last() {
            let tid = token_id as usize;
            assert!(
                tid < self.config.vocab_size,
                "Token ID {} out of range [0, {})",
                tid,
                self.config.vocab_size
            );
            h.row_mut(0).assign(&self.token_embedding.row(tid));
        }

        let token_pos = kv_cache.num_tokens(seq_id).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        let cos_slice = if token_pos * half < self.precomputed_cos.len() {
            self.precomputed_cos
                .slice(ndarray::s![token_pos * half..(token_pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };
        let sin_slice = if token_pos * half < self.precomputed_sin.len() {
            self.precomputed_sin
                .slice(ndarray::s![token_pos * half..(token_pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_paged(
                &h, kv_cache, seq_id, layer_idx, token_pos, &cos_slice, &sin_slice,
            );
        }

        h = self.norm.forward(&h);

        h.row(0).dot(&self.lm_head.t())
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

    pub fn from_checkpoint(
        config: TransformerConfig,
        path: &str,
    ) -> crate::TransformerResult<Self> {
        let mut model = Self::new(config);
        let loaded = crate::safetensors::load_safetensors(path)?;
        let get_arr = |name: &str| -> crate::TransformerResult<ndarray::ArrayD<f32>> {
            loaded.get(name).cloned().ok_or_else(|| {
                crate::TransformerError::Implementation(format!("Missing tensor: {}", name))
            })
        };
        fn to_fixed<D: ndarray::Dimension>(
            arr: ndarray::ArrayD<f32>,
            name: &str,
        ) -> crate::TransformerResult<ndarray::Array<f32, D>> {
            arr.into_dimensionality::<D>().map_err(|e| {
                crate::TransformerError::Implementation(format!(
                    "Shape mismatch for {}: {}",
                    name, e
                ))
            })
        }
        model.token_embedding =
            to_fixed::<ndarray::Ix2>(get_arr("token_embedding")?, "token_embedding")?.to_owned();
        model.lm_head = to_fixed::<ndarray::Ix2>(get_arr("lm_head")?, "lm_head")?.to_owned();
        model.norm.weight =
            to_fixed::<ndarray::Ix1>(get_arr("norm.weight")?, "norm.weight")?.to_owned();
        for (i, block) in model.blocks.iter_mut().enumerate() {
            let prefix = format!("blocks.{}.", i);
            let name = prefix.clone() + "attention_norm.weight";
            block.attention_norm.weight =
                to_fixed::<ndarray::Ix1>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "ffn_norm.weight";
            block.ffn_norm.weight = to_fixed::<ndarray::Ix1>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "attention.wq";
            block.attention.wq = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "attention.wk";
            block.attention.wk = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "attention.wv";
            block.attention.wv = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "attention.wo";
            block.attention.wo = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "ffn.w1";
            block.ffn.w1 = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "ffn.w2";
            block.ffn.w2 = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "ffn.w3";
            block.ffn.w3 = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
        }
        Ok(model)
    }

    #[cfg(feature = "gpu")]
    fn generate_gpu_impl(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        use nexora_autograd::gpu::GpuContext;

        let mut cache = self.reset_cache();

        // Prefill prompt tokens (use GPU forward)
        for &token_id in prompt_ids {
            if self.forward_gpu(&[token_id], &mut cache).is_err() {
                // Fallback to CPU if GPU fails
                self.forward(&[token_id], &mut cache);
            }
        }

        let mut output = Vec::new();
        let mut last_id = *prompt_ids.last().unwrap_or(&0);

        for _ in 0..max_tokens {
            match self.forward_gpu(&[last_id], &mut cache) {
                Ok(logits) => {
                    let next_id = sample_token_gpu(&logits, temperature, top_k, 12345);
                    output.push(next_id);
                    if next_id == 0 {
                        break;
                    }
                    last_id = next_id;
                }
                Err(_) => {
                    // Fallback to CPU
                    let logits = self.forward(&[last_id], &mut cache);
                    let next_id = sample_token(&logits, temperature, top_k);
                    output.push(next_id);
                    if next_id == 0 {
                        break;
                    }
                    last_id = next_id;
                }
            }
        }

        (output, cache)
    }

    /// GPU forward pass using GPU-resident KV cache.
    /// No CPU round-trip for K/V operations — eliminates O(n²) data transfer.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_cache(
        &self,
        input_ids: &[u32],
        cache: &mut [super::gqa::GpuKVCacheEntry],
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = GpuContext::global()?;
        let batch_size = 1;

        // 1. Token embedding: copy the embedding row for last token
        let h_data = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                Some(self.token_embedding.row(tid).to_vec())
            }
            None => None,
        };
        let h_cpu = if let Some(ref data) = h_data {
            ndarray::ArrayD::from_shape_vec(vec![batch_size, self.config.hidden_size], data.clone())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?
        } else {
            ndarray::ArrayD::zeros(vec![batch_size, self.config.hidden_size])
        };
        let mut h = GpuTensor::from_cpu(&h_cpu)?;

        // 2. Get RoPE cos/sin for current position
        let pos = cache.first().map(|e| e.seq_len).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        let cos_slice: Array1<f32> = if pos * half < self.precomputed_cos.len() {
            self.precomputed_cos
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };
        let sin_slice: Array1<f32> = if pos * half < self.precomputed_sin.len() {
            self.precomputed_sin
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };

        // 3. Forward through all transformer blocks on GPU with GPU-resident cache
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_with_cache(&h, cache, layer_idx, &cos_slice, &sin_slice)?;
        }

        // 4. Final RMSNorm on GPU
        h = self.norm.forward_gpu(&h)?;

        // 5. LM head: matmul(h, lm_head^T) on GPU with cached weight
        let mut guard = self.lm_head_gpu.lock().unwrap();
        let lm_head_cached = guard.get_or_insert_with(|| {
            let shape = vec![self.lm_head.shape()[0], self.lm_head.shape()[1]];
            let data = self.lm_head.as_slice().unwrap().to_vec();
            let cpu_arr = ndarray::ArrayD::from_shape_vec(shape, data).unwrap();
            let gpu = GpuTensor::from_cpu(&cpu_arr).unwrap();
            ctx.transpose(&gpu).unwrap()
        });
        let logits_gpu = ctx.matmul(&h, lm_head_cached)?;

        // 6. Download logits to CPU
        let logits_cpu = logits_gpu.to_cpu();
        let logits_flat: Vec<f32> = logits_cpu.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// Generate using GPU-resident KV cache for maximum performance.
    /// Creates GPU cache entries for all layers with pre-allocated buffers.
    #[cfg(feature = "gpu")]
    pub fn generate_gpu_with_cache(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> (Vec<u32>, Vec<super::gqa::GpuKVCacheEntry>) {
        use nexora_autograd::gpu::GpuContext;

        let ctx = match GpuContext::global() {
            Ok(c) => c,
            Err(_) => {
                // Fallback: CPU-only if GPU unavailable
                let (tokens, _) = self.generate(prompt_ids, max_tokens, temperature, top_k);
                return (tokens, Vec::new());
            }
        };

        // Create GPU-resident cache for all layers
        let mut gpu_cache: Vec<super::gqa::GpuKVCacheEntry> = (0..self.config.num_layers)
            .map(|_| {
                super::gqa::GpuKVCacheEntry::new(
                    self.config.num_kv_heads,
                    self.config.head_dim(),
                    self.config.max_seq_len,
                )
            })
            .collect();

        // Ensure fill_zero dispatches complete before any append
        ctx.flush();

        // Prefill prompt tokens
        for &token_id in prompt_ids {
            if self.forward_gpu_with_cache(&[token_id], &mut gpu_cache).is_err() {
                // Fallback: cannot use GPU cache — use Vec<KVCacheEntry> CPU path
                let mut cpu_cache = self.reset_cache();
                for t in &[prompt_ids, &[token_id]].concat() {
                    self.forward(&[*t], &mut cpu_cache);
                }
                // ... but we can't easily switch mid-generation, so just return
                let (tokens, _) = self.generate(prompt_ids, max_tokens, temperature, top_k);
                return (tokens, gpu_cache);
            }
        }

        let mut output = Vec::new();
        let mut last_id = *prompt_ids.last().unwrap_or(&0);

        for _ in 0..max_tokens {
            match self.forward_gpu_with_cache(&[last_id], &mut gpu_cache) {
                Ok(logits) => {
                    let next_id = sample_token_gpu(&logits, temperature, top_k, 12345);
                    output.push(next_id);
                    if next_id == 0 {
                        break;
                    }
                    last_id = next_id;
                }
                Err(_) => {
                    // Fallback to CPU sample (logits computation already failed)
                    let logits = self.forward(&[last_id], &mut self.reset_cache());
                    let next_id = sample_token(&logits, temperature, top_k);
                    output.push(next_id);
                    if next_id == 0 {
                        break;
                    }
                    last_id = next_id;
                }
            }
        }

        (output, gpu_cache)
    }

    pub fn memory_bytes(&self) -> usize {
        self.parameter_count() * 4
    }

    /// GPU forward pass: returns logits on CPU.
    /// Handles token embedding, all transformer blocks, final norm + lm_head on GPU.
    /// Falls back to CPU if GPU is unavailable.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Vec<KVCacheEntry>,
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        use ndarray::ArrayD;

        let ctx = GpuContext::global()?;
        let batch_size = 1;

        // 1. Token embedding: copy the embedding row for last token
        let h_data = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                Some(self.token_embedding.row(tid).to_vec())
            }
            None => None,
        };
        let h_cpu = if let Some(ref data) = h_data {
            ArrayD::from_shape_vec(vec![batch_size, self.config.hidden_size], data.clone())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?
        } else {
            ArrayD::zeros(vec![batch_size, self.config.hidden_size])
        };
        let mut h = GpuTensor::from_cpu(&h_cpu)?;

        // 2. Get RoPE cos/sin for current position
        let pos = kv_cache.first().map(|e| e.k.shape()[0]).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        let cos_slice: Array1<f32> = if pos * half < self.precomputed_cos.len() {
            self.precomputed_cos
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };
        let sin_slice: Array1<f32> = if pos * half < self.precomputed_sin.len() {
            self.precomputed_sin
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };

        // 3. Forward through all transformer blocks on GPU
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu(&h, kv_cache, layer_idx, &cos_slice, &sin_slice)?;
        }

        // 4. Final RMSNorm on GPU
        h = self.norm.forward_gpu(&h)?;

        // 5. LM head: matmul(h, lm_head^T) on GPU with cached weight
        let mut guard = self.lm_head_gpu.lock().unwrap();
        let lm_head_cached = guard.get_or_insert_with(|| {
            let shape = vec![self.lm_head.shape()[0], self.lm_head.shape()[1]];
            let data = self.lm_head.as_slice().unwrap().to_vec();
            let cpu_arr = ArrayD::from_shape_vec(shape, data).unwrap();
            let gpu = GpuTensor::from_cpu(&cpu_arr).unwrap();
            ctx.transpose(&gpu).unwrap()
        });
        let logits_gpu = ctx.matmul(&h, lm_head_cached)?;

        // 6. Download logits to CPU
        let logits_cpu = logits_gpu.to_cpu();
        let logits_flat: Vec<f32> = logits_cpu.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// Batched GPU forward pass: processes multiple sequences in a single batch.
    /// All GPU dispatches are coalesced into one queue.submit() via batch mode.
    /// Returns one logit vector per sequence.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_batched(
        &self,
        batch_tokens: &[u32],
        kv_caches: &mut [Vec<KVCacheEntry>],
    ) -> Result<Vec<Array1<f32>>, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = GpuContext::global()?;
        let batch_size = batch_tokens.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        ctx.begin_batch_mode();

        let mut all_logits = Vec::with_capacity(batch_size);
        for (seq_idx, cache) in kv_caches.iter_mut().enumerate() {
            let token_id = batch_tokens[seq_idx];

            // Embedding: upload single row [1, hidden] — non-blocking queue.write_buffer
            let tid = token_id as usize;
            let h_data = self.token_embedding.row(tid).to_vec();
            let h_cpu = ArrayD::from_shape_vec(vec![1, self.config.hidden_size], h_data)
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;
            let mut h = GpuTensor::from_cpu(&h_cpu)?;

            // RoPE cos/sin for this sequence's current position
            let pos = cache.first().map(|e| e.k.shape()[0]).unwrap_or(0);
            let half = self.config.head_dim() / 2;
            let cos_slice = if pos * half < self.precomputed_cos.len() {
                self.precomputed_cos
                    .slice(ndarray::s![pos * half..(pos + 1) * half])
                    .to_owned()
            } else {
                Array1::zeros(half)
            };
            let sin_slice = if pos * half < self.precomputed_sin.len() {
                self.precomputed_sin
                    .slice(ndarray::s![pos * half..(pos + 1) * half])
                    .to_owned()
            } else {
                Array1::zeros(half)
            };

            // Forward through all blocks with per-sequence KV cache
            for (layer_idx, block) in self.blocks.iter().enumerate() {
                h = block.forward_gpu(&h, cache, layer_idx, &cos_slice, &sin_slice)?;
            }

            h = self.norm.forward_gpu(&h)?;

            let mut guard = self.lm_head_gpu.lock().unwrap();
            let lm_head_cached = guard.get_or_insert_with(|| {
                let shape = vec![self.lm_head.shape()[0], self.lm_head.shape()[1]];
                let data = self.lm_head.as_slice().unwrap().to_vec();
                let cpu_arr = ArrayD::from_shape_vec(shape, data).unwrap();
                let gpu = GpuTensor::from_cpu(&cpu_arr).unwrap();
                ctx.transpose(&gpu).unwrap()
            });
            let logits_gpu = ctx.matmul(&h, lm_head_cached)?;

            let logits_cpu = logits_gpu.to_cpu();
            let logits_flat: Vec<f32> = logits_cpu.iter().copied().collect();
            all_logits.push(Array1::from_vec(logits_flat));
        }

        ctx.end_batch_mode();
        Ok(all_logits)
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, false)
    }

    pub fn generate_with_gpu(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        use_gpu: bool,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        let mut cache = self.reset_cache();

        if use_gpu {
            #[cfg(feature = "gpu")]
            {
                return self.generate_gpu_impl(prompt_ids, max_tokens, temperature, top_k);
            }
        }

        for &token_id in prompt_ids {
            self.forward(&[token_id], &mut cache);
        }

        let mut output = Vec::new();
        let mut last_id = *prompt_ids.last().unwrap_or(&0);

        for _ in 0..max_tokens {
            let logits = self.forward(&[last_id], &mut cache);
            let next_id = if use_gpu {
                #[cfg(feature = "gpu")]
                {
                    sample_token_gpu(&logits, temperature, top_k, 12345)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    sample_token(&logits, temperature, top_k)
                }
            } else {
                sample_token(&logits, temperature, top_k)
            };
            output.push(next_id);
            if next_id == 0 {
                break;
            }
            last_id = next_id;
        }

        (output, cache)
    }
}
