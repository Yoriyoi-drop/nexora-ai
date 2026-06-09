use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;
use ndarray::{Array1, Array2};
use crate::block::TransformerBlock;
use crate::gqa::{CpuKVCache, KVCacheEntry, KVCacheProvider, PagedCacheReader};
use crate::rope::RoPE;
use crate::model::builder::{BlockGpuWeights, CausalLM, GpuWeights};
use tracing::warn;
use crate::model::config::{sample_token, sample_token_gpu_keep_gpu};
#[cfg(feature = "gpu")]
use crate::model::config::sample_token_gpu;
use crate::{TransformerConfig, TransformerError, TransformerResult};

pub static GPU_FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);

impl CausalLM {

    /// Register a weight-change observer.
    /// Fires on every `sync_to_inference()` and `load_checkpoint()`.
    pub fn register_weight_observer(&self, observer: Box<dyn crate::observer::WeightObserver>) {
        self.weight_notifier.add(observer);
    }

    /// Notify all registered observers (called by weight-mutating code).
    pub fn notify_weight_changed(&self) {
        self.weight_notifier.notify();
    }

    /// Enable half-precision (f16) weight storage on GPU.
    /// Weights disimpan dalam packed f16 (2× VRAM hemat),
    /// lalu di-upconvert ke f32 tiap forward pass.
    pub fn with_half_precision(mut self) -> Self {
        self.use_half_precision = true;
        for block in &mut self.blocks {
            block.set_use_half_precision();
        }
        self
    }

    fn get_token_embedding(&self) -> TransformerResult<&Array2<f32>> {
        self.token_embedding.as_ref().ok_or_else(|| {
            TransformerError::Implementation("token_embedding not available — use readback_weights() or forward_gpu()".into())
        })
    }

    fn get_lm_head(&self) -> TransformerResult<&Array2<f32>> {
        if self.weight_tied {
            return self.get_token_embedding();
        }
        self.lm_head.as_ref().ok_or_else(|| {
            TransformerError::Implementation("lm_head not available".into())
        })
    }

    pub fn drop_cpu_weights(&mut self) {
        if !self.weight_tied {
            self.lm_head = None;
        }
        self.token_embedding = None;
        for block in &mut self.blocks {
            block.attention.drop_cpu_weights();
            block.ffn.drop_cpu_weights();
            block.attention_norm.drop_cpu_weight();
            block.ffn_norm.drop_cpu_weight();
        }
        self.norm.drop_cpu_weight();
    }

    /// Reset GPU weight caches so the next `forward()` re-uploads from CPU.
    /// Needed after training updates CPU weights via `sync_to_inference()`.
    pub fn reset_gpu_weights(&mut self) {
        #[cfg(feature = "gpu")]
        {
            self.gpu_weights = OnceLock::new();
            self.norm.gpu_weights = OnceLock::new();
            for block in &mut self.blocks {
                block.attention.gpu_weights = OnceLock::new();
                block.ffn.gpu_weights = OnceLock::new();
                block.attention_norm.gpu_weights = OnceLock::new();
                block.ffn_norm.gpu_weights = OnceLock::new();
            }
        }
    }

    pub fn forward(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn KVCacheProvider,
    ) -> TransformerResult<Array1<f32>> {
        #[cfg(feature = "gpu")]
        let gpu_available = nexora_autograd::gpu::GpuContext::global().is_ok();

        #[cfg(feature = "gpu")]
        if gpu_available {
            match self.forward_gpu_with_cache_provider(input_ids, kv_cache) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    if self.keep_on_gpu {
                        tracing::warn!(error = %e, "GPU forward pass failed, falling back to CPU");
                    } else {
                        tracing::error!(error = %e, "GPU forward pass failed, falling back to CPU");
                    }
                    GPU_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        let entries = kv_cache.as_cpu_entries().ok_or_else(|| {
            TransformerError::Implementation("CPU forward requires CpuKVCache".to_string())
        })?;
        self.forward_cpu_impl(input_ids, entries)
    }

    fn forward_cpu_impl(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Vec<KVCacheEntry>,
    ) -> TransformerResult<Array1<f32>> {
        if input_ids.is_empty() {
            return Err(TransformerError::Implementation(
                "forward_cpu_impl called with empty input_ids".into()
            ));
        }

        // Check that blocks are loaded (lazy loading must happen before forward)
        if self.lazy_loader.is_some() && self.blocks.first().map_or(true, |b| b.attention.wq.is_none()) {
            return Err(TransformerError::Implementation(
                "Blocks not loaded — call load_lazy_blocks() or ensure_blocks_loaded() before forward()".into()
            ));
        }

        let token_id = input_ids.last().copied().ok_or_else(|| {
            TransformerError::Implementation("forward_cpu_impl called with empty input_ids".into())
        })? as usize;
        if token_id >= self.config.vocab_size {
            return Err(TransformerError::Implementation(format!(
                "Token ID {} out of range [0, {})",
                token_id, self.config.vocab_size
            )));
        }

        let te = self.get_token_embedding()?;
        let mut h = Array2::zeros((1, self.config.hidden_size));
        h.row_mut(0).assign(&te.row(token_id));

        let pos = kv_cache.first().map(|e| e.seq_len()).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        if pos * half + half > self.precomputed_cos.len() && !self.precomputed_cos.is_empty() {
            return Err(TransformerError::Implementation(format!(
                "RoPE cos position {} out of range (len={}, half={})",
                pos,
                self.precomputed_cos.len(),
                half
            )));
        }
        let (cos_slice, sin_slice) = self.get_cos_sin_slices(pos);

        let (embed_vocab, _embed_hidden) = te.dim();
        if embed_vocab != self.config.vocab_size {
            return Err(TransformerError::Implementation(format!(
                "token_embedding vocab {} != config {}",
                embed_vocab, self.config.vocab_size
            )));
        }

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_with_collective(&h, kv_cache, layer_idx, cos_slice, sin_slice, collective)?;

            for (target_layer, injector) in &self.injectors {
                if *target_layer == layer_idx {
                    let mut guard = injector.lock().map_err(|e| {
                        TransformerError::Implementation(format!("Injector lock: {}", e))
                    })?;
                    guard.after_layer(layer_idx, &mut h, pos)?;
                }
            }
        }

        h = self.norm.forward(&h)?;

        // Apply LoRA PEFT to final hidden state before lm_head projection
        let logits = if let Some(ref adapters) = self.lora_adapters {
            let lora_out: Array2<f32> = adapters.iter().fold(
                Array2::zeros((1, self.config.vocab_size)),
                |acc, layer| {
                    acc + layer.q.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.k.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.v.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.o.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.w1.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.w2.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.w3.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                },
            );
            let lh = self.get_lm_head()?;
            h.row(0).dot(&lh.t()) + lora_out.row(0)
        } else {
            let lh = self.get_lm_head()?;
            h.row(0).dot(&lh.t())
        };
        Ok(logits)
    }

    pub fn reset_cache(&self) -> CpuKVCache {
        // Reset all injectors for a new sequence (fork-on-write for ring buffers)
        for (_, injector) in &self.injectors {
            if let Ok(mut guard) = injector.lock() {
                guard.reset();
            }
        }
        CpuKVCache::new(self.config.num_layers)
    }

    pub fn set_keep_on_gpu(&mut self, keep: bool) {
        self.keep_on_gpu = keep;
    }

    pub fn clear_gpu_cache(&self) {
        #[cfg(feature = "gpu")]
        if let Ok(mut cache) = self.gpu_cache.write() {
            if let Some(gpu_entries) = cache.as_mut() {
                gpu_entries.clear();
            }
        }
    }

    /// Forward pass using a paged KV cache.
    /// `token_pos` is the current position being generated.
    pub fn forward_paged(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn PagedCacheReader,
        seq_id: u64,
    ) -> TransformerResult<Array1<f32>> {
        let mut h = Array2::zeros((1, self.config.hidden_size));

        if let Some(&token_id) = input_ids.last() {
            let tid = token_id as usize;
            if tid >= self.config.vocab_size {
                return Err(TransformerError::Implementation(format!(
                    "Token ID {} out of range [0, {})",
                    tid, self.config.vocab_size
                )));
            }
            let te = self.get_token_embedding()?;
            h.row_mut(0).assign(&te.row(tid));
        }

        let token_pos = kv_cache.num_tokens(seq_id).unwrap_or(0);
        let (cos_slice, sin_slice) = self.get_cos_sin_slices(token_pos);

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_paged_with_collective(
                &h, kv_cache, seq_id, layer_idx, token_pos, cos_slice, sin_slice, collective,
            )?;
        }

        h = self.norm.forward(&h)?;

        let logits = if let Some(ref adapters) = self.lora_adapters {
            let lora_out: Array2<f32> = adapters.iter().fold(
                Array2::zeros((1, self.config.vocab_size)),
                |acc, layer| {
                    acc + layer.q.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.k.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.v.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.o.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.w1.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.w2.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                        + layer.w3.as_ref().map_or(Array2::zeros((1, self.config.vocab_size)), |l| l.apply(&h))
                },
            );
            let lh = self.get_lm_head()?;
            h.row(0).dot(&lh.t()) + lora_out.row(0)
        } else {
            let lh = self.get_lm_head()?;
            h.row(0).dot(&lh.t())
        };
        Ok(logits)
    }

    /// GPU forward with pre-uploaded cos/sin GPU tensors.
    /// cos/sin diupload SEKALI per step — tidak per-layer.
    /// cos_gpu, sin_gpu: [1, half] GPU tensors for the current position.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_precomputed_rope(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Vec<KVCacheEntry>,
        cos_gpu: &nexora_autograd::gpu::GpuTensor,
        sin_gpu: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = 1;
        let hidden_size = self.config.hidden_size;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                if tid >= self.config.vocab_size {
                    return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                        "Token ID {} out of range [0, {})",
                        tid, self.config.vocab_size
                    )));
                }
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gw.token_embedding.buffer(),
                        offset,
                        h.buffer(),
                        0,
                        row_bytes,
                    );
                    Ok(())
                })?;
                h
            }
            None => GpuTensor::zeros(&[batch_size, hidden_size])?,
        };

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_with_rope_gpu_collective(&h, kv_cache, layer_idx, cos_gpu, sin_gpu, collective)?;
        }

        h = self.norm.forward_gpu(&h)?;

        let logits = ctx.matmul(&h, &gw.lm_head_t)?;

        let logits_flat: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// GPU forward pass using GPU-resident KV cache.
    /// No CPU round-trip for K/V operations — eliminates O(n²) data transfer.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_cache(
        &self,
        input_ids: &[u32],
        cache: &mut [crate::gqa::GpuKVCacheEntry],
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = 1;
        let hidden_size = self.config.hidden_size;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                if tid >= self.config.vocab_size {
                    return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                        "Token ID {} out of range [0, {})",
                        tid, self.config.vocab_size
                    )));
                }
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gw.token_embedding.buffer(),
                        offset,
                        h.buffer(),
                        0,
                        row_bytes,
                    );
                    Ok(())
                })?;
                h
            }
            None => GpuTensor::zeros(&[batch_size, hidden_size])?,
        };

        let pos = cache.first().map(|e| e.seq_len).unwrap_or(0);
        let (cos_slice, sin_slice) = self.get_cos_sin_arrays(pos);

        // Upload cos/sin to GPU ONCE — reused across all layers
        let cos_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, cos_slice.len()], cos_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, sin_slice.len()], sin_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_with_cache_precomputed_rope_collective(
                &h, cache, layer_idx, &cos_gpu, &sin_gpu, collective,
            )?;

            // Run injectors (Echo-Net APSS, etc.) after each layer
            for (target_layer, injector) in &self.injectors {
                if *target_layer == layer_idx {
                    let mut guard = injector.lock().map_err(|e| {
                        nexora_autograd::gpu::GpuError::Unsupported(format!("injector lock: {}", e))
                    })?;
                    guard.after_layer_gpu(layer_idx, &mut h, pos, ctx)?;
                }
            }
        }

        h = self.norm.forward_gpu(&h)?;

        let logits = ctx.matmul(&h, &gw.lm_head_t)?;

        let logits_flat: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// Same as `forward_gpu_with_cache` but skips `to_cpu()` on the logits.
    /// Use with `generate_gpu_keep_gpu` to eliminate PCIe round-trip per token.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_cache_keep_gpu(
        &self,
        input_ids: &[u32],
        cache: &mut [crate::gqa::GpuKVCacheEntry],
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = 1;
        let hidden_size = self.config.hidden_size;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                if tid >= self.config.vocab_size {
                    return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                        "Token ID {} out of range [0, {})",
                        tid, self.config.vocab_size
                    )));
                }
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gw.token_embedding.buffer(),
                        offset,
                        h.buffer(),
                        0,
                        row_bytes,
                    );
                    Ok(())
                })?;
                h
            }
            None => GpuTensor::zeros(&[batch_size, hidden_size])?,
        };

        let pos = cache.first().map(|e| e.seq_len).unwrap_or(0);
        let (cos_slice, sin_slice) = self.get_cos_sin_arrays(pos);

        let cos_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, cos_slice.len()], cos_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, sin_slice.len()], sin_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_with_cache_precomputed_rope_collective(
                &h, cache, layer_idx, &cos_gpu, &sin_gpu, collective,
            )?;

            for (target_layer, injector) in &self.injectors {
                if *target_layer == layer_idx {
                    let mut guard = injector.lock().map_err(|e| {
                        nexora_autograd::gpu::GpuError::Unsupported(format!("injector lock: {}", e))
                    })?;
                    guard.after_layer_gpu(layer_idx, &mut h, pos, ctx)?;
                }
            }
        }

        h = self.norm.forward_gpu(&h)?;

        let logits = ctx.matmul(&h, &gw.lm_head_t)?;

        Ok(logits)
    }

    /// GPU forward via [`KVCacheProvider`].
    /// Routes to `GpuKVCache` for true zero-copy GPU→GPU cache,
    /// or auto-converts `CpuKVCache` to GPU cache to avoid O(n²) round-trip.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_cache_provider(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn KVCacheProvider,
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        self.forward_gpu_with_cache_provider_inner(input_ids, kv_cache, false)
    }

    fn forward_gpu_with_cache_provider_inner(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn KVCacheProvider,
        needs_sync_back: bool,
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = GpuContext::global()?;

        // Pure GPU path — no CPU round-trip for K/V
        if let Some(gpu_entries) = kv_cache.as_gpu_entries() {
            return self.forward_gpu_with_cache(input_ids, gpu_entries);
        }

        let num_layers = self.config.num_layers;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let max_seq = self.config.max_seq_len;

        // Get or create GPU-side cache mirroring the CpuKVCache
        let cpu_entries = kv_cache.as_cpu_entries().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "forward_gpu_with_cache_provider needs CpuKVCache or GpuKVCache".into(),
            )
        })?;

        let mut cache_guard = self.gpu_cache.write().map_err(|e| {
            nexora_autograd::gpu::GpuError::Unsupported(format!("GPU cache lock poisoned: {}", e))
        })?;
        if cache_guard.is_none() {
            let mut gpu_entries: Vec<crate::gqa::GpuKVCacheEntry> = (0..num_layers)
                .map(|_| {
                    crate::gqa::GpuKVCacheEntry::new(
                        &ctx,
                        n_kv_heads,
                        head_dim,
                        max_seq,
                        self.use_half_precision,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Upload existing CPU K/V to GPU (one-time O(seq_len) cost)
            for layer_idx in 0..num_layers {
                if layer_idx < cpu_entries.len() && cpu_entries[layer_idx].seq_len() > 0 {
                    let ce = &cpu_entries[layer_idx];
                    let seq = ce.seq_len();
                    let k_cpu =
                        ArrayD::from_shape_vec(vec![seq, n_kv_heads, head_dim], ce.k.clone())
                            .map_err(|e| {
                                nexora_autograd::gpu::GpuError::Unsupported(e.to_string())
                            })?;
                    let v_cpu =
                        ArrayD::from_shape_vec(vec![seq, n_kv_heads, head_dim], ce.v.clone())
                            .map_err(|e| {
                                nexora_autograd::gpu::GpuError::Unsupported(e.to_string())
                            })?;

                    let k_gpu = GpuTensor::from_cpu(&k_cpu)?;
                    let v_gpu = GpuTensor::from_cpu(&v_cpu)?;
                    let bytes = (seq * n_kv_heads * head_dim * 4) as u64;

                    ctx.batch_dispatch(|enc| {
                        enc.copy_buffer_to_buffer(
                            k_gpu.buffer(),
                            0,
                            gpu_entries[layer_idx].k.buffer(),
                            0,
                            bytes,
                        );
                        enc.copy_buffer_to_buffer(
                            v_gpu.buffer(),
                            0,
                            gpu_entries[layer_idx].v.buffer(),
                            0,
                            bytes,
                        );
                        Ok(())
                    })?;

                    gpu_entries[layer_idx].seq_len = seq;
                }
            }

            *cache_guard = Some(gpu_entries);
        }

        let gpu_entries = cache_guard.as_mut().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU cache not initialized after write".into(),
            )
        })?;

        // GPU forward with GPU-resident cache — no K/V round-trip
        let result = self.forward_gpu_with_cache(input_ids, gpu_entries)?;

        // Sync back to CPU only when explicitly needed (CpuKVCache conversion case).
        // When cache is GpuKVCache, the caller gets entries via as_gpu_entries() and
        // the early return above prevents reaching here entirely.
        if needs_sync_back {
            let kv_elems = n_kv_heads * head_dim;

            for layer_idx in 0..num_layers {
                let gpu_entry = &gpu_entries[layer_idx];
                let last_idx = gpu_entry.seq_len.wrapping_sub(1);
                let elem_size: u64 = if gpu_entry.f16_storage { 2 } else { 4 };
                let byte_off = (last_idx * kv_elems) as u64 * elem_size;
                let token_bytes = (kv_elems as u64) * elem_size;

                let staging_k = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("kv_sync_back_k"),
                    size: token_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let staging_v = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("kv_sync_back_v"),
                    size: token_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gpu_entry.k.buffer(),
                        byte_off,
                        &staging_k,
                        0,
                        token_bytes,
                    );
                    enc.copy_buffer_to_buffer(
                        gpu_entry.v.buffer(),
                        byte_off,
                        &staging_v,
                        0,
                        token_bytes,
                    );
                    Ok(())
                })?;

                ctx.sync();

                let read_back =
                    |staging: &wgpu::Buffer,
                     is_f16: bool|
                     -> Result<Vec<f32>, nexora_autograd::gpu::GpuError> {
                        let slice = staging.slice(..);
                        let (tx, rx) = std::sync::mpsc::channel();
                        slice.map_async(wgpu::MapMode::Read, move |r| {
                            let _ = tx.send(r);
                        });
                        // Called from spawn_blocking — sync device.poll(Wait) is acceptable here
                        ctx.device.poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(Duration::from_secs(30)),
                        });
                        rx.recv_timeout(Duration::from_secs(30))
                            .map_err(|_| {
                                nexora_autograd::gpu::GpuError::Timeout(
                                    "KV cache readback timed out after 30s".into(),
                                )
                            })?
                            .map_err(|e| {
                                nexora_autograd::gpu::GpuError::Device(format!("map_async: {e:?}"))
                            })?;
                        let out: Vec<f32> = {
                            let mapped = slice.get_mapped_range();
                            let result = if is_f16 {
                                mapped
                                    .chunks_exact(2)
                                    .map(|c| {
                                        let bits = u16::from_ne_bytes([c[0], c[1]]);
                                        half::f16::from_bits(bits).to_f32()
                                    })
                                    .collect()
                            } else {
                                mapped
                                    .chunks_exact(4)
                                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect()
                            };
                            result
                        };
                        staging.unmap();
                        Ok(out)
                    };

                let new_k = read_back(&staging_k, gpu_entry.f16_storage)?;
                let new_v = read_back(&staging_v, gpu_entry.f16_storage)?;

                if layer_idx < cpu_entries.len() {
                    let cpu_entry = &mut cpu_entries[layer_idx];
                    cpu_entry.k.extend_from_slice(&new_k);
                    cpu_entry.v.extend_from_slice(&new_v);
                }
            }
        }

        Ok(result)
    }

    /// Generate using GPU with pre-uploaded cos/sin.
    /// cos/sin diupload SEKALI ke GPU di awal, lalu slice_tensor GPU-side copy per step.
    /// Eliminates N_layers × 2 CPU→GPU uploads per step → 0 uploads.
    #[cfg(feature = "gpu")]
    fn generate_gpu_impl(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        use nexora_autograd::gpu::GpuContext;

        let ctx = match GpuContext::global() {
            Ok(c) => c,
            Err(_) => {
                return self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, false)
            }
        };

        // Create GPU-resident cache for all layers (zero CPU round-trip)
        let mut gpu_cache: Vec<crate::gqa::GpuKVCacheEntry> = match (0..self.config.num_layers)
            .map(|_| {
                crate::gqa::GpuKVCacheEntry::new(
                    &ctx,
                    self.config.num_kv_heads,
                    self.config.head_dim(),
                    self.config.max_seq_len,
                    self.use_half_precision,
                )
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(cache) => cache,
            Err(_) => {
                return self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, false)
            }
        };

        ctx.flush();

        // Prefill prompt tokens
        for &token_id in prompt_ids {
            if self
                .forward_gpu_with_cache(&[token_id], &mut gpu_cache)
                .is_err()
            {
                let (tokens, _) =
                    self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, false);
                return (tokens, Vec::new());
            }
        }

        let mut output = Vec::new();
        let mut last_id = *prompt_ids.last().unwrap_or(&0);

        for _ in 0..max_tokens {
            match self.forward_gpu_with_cache(&[last_id], &mut gpu_cache) {
                Ok(logits) => {
                    let next_id = sample_token_gpu(&logits, temperature, top_k, 0.0, 12345);
                    output.push(next_id);
                    if next_id == 0 {
                        break;
                    }
                    last_id = next_id;
                }
                Err(_) => {
                    let logits = self
                        .forward(&[last_id], &mut self.reset_cache())
                        .unwrap_or_else(|_| Array1::zeros(self.config.vocab_size));
                    let next_id = sample_token(&logits, temperature, top_k);
                    output.push(next_id);
                    if next_id == 0 {
                        break;
                    }
                    last_id = next_id;
                }
            }
        }

        // One-time GPU readback: convert GpuKVCacheEntry -> KVCacheEntry
        let cpu_cache: Vec<KVCacheEntry> = gpu_cache
            .iter()
            .map(|entry| {
                let k_cpu: ndarray::ArrayD<f32> = entry.k_view().to_cpu().unwrap_or_else(|e| {
                    tracing::warn!("KV cache GPU readback k failed: {e}");
                    ndarray::ArrayD::zeros(vec![0])
                });
                let v_cpu: ndarray::ArrayD<f32> = entry.v_view().to_cpu().unwrap_or_else(|e| {
                    tracing::warn!("KV cache GPU readback v failed: {e}");
                    ndarray::ArrayD::zeros(vec![0])
                });
                let kv_dim = entry.kv_heads * entry.head_dim;
                KVCacheEntry {
                    k: k_cpu.into_iter().collect(),
                    v: v_cpu.into_iter().collect(),
                    kv_dim,
                    num_kv_heads: entry.kv_heads,
                    head_dim: entry.head_dim,
                    k_compressed: Vec::new(),
                    v_compressed: Vec::new(),
                    k_scales: Vec::new(),
                    v_scales: Vec::new(),
                    compressed_seq_len: 0,
                }
            })
            .collect();

        (output, cpu_cache)
    }

    /// GPU-RESIDENT generation loop: logits NEVER leave GPU during the loop.
    /// Forward → `forward_gpu_with_cache_keep_gpu` (returns GpuTensor, no to_cpu).
    /// Sample  → `sample_token_gpu_keep_gpu` (takes GpuTensor, returns GpuTensor).
    /// Readback: only 4 bytes per token (token ID), plus one bulk readback at the end.
    /// Eliminates ~`vocab_size * 4` bytes per token PCIe traffic (e.g. 128KB/token for 32K vocab).
    #[cfg(feature = "gpu")]
    fn generate_gpu_keep_gpu_impl(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        use nexora_autograd::gpu::GpuContext;
        let ctx = match GpuContext::global() {
            Ok(c) => c,
            Err(_) => return self.generate(prompt_ids, max_tokens, temperature, top_k),
        };
        // Prefill: feed all prompt tokens through the GPU-forward path.
        // We still use the CPU forward for prefill (batched matmul isn't ready
        // for GPU-only KV cache). After prefill, switch to the GPU-resident loop.
        let mut output: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut last_id = *prompt_ids.last().unwrap_or(&0);
        let mut gpu_cache: Vec<crate::gqa::GpuKVCacheEntry> = match (0..self.config.num_layers)
            .map(|_| {
                crate::gqa::GpuKVCacheEntry::new(
                    &ctx,
                    self.config.num_kv_heads,
                    self.config.head_dim(),
                    self.config.max_seq_len,
                    self.use_half_precision,
                )
            })
            .collect()
        {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    "Failed to create GPU KV cache: {e}, falling back to CPU generation"
                );
                return self.generate(prompt_ids, max_tokens, temperature, top_k);
            }
        };

        // Prefill prompt tokens into GPU cache (one-by-one for correctness)
        for &id in prompt_ids {
            let _ = match self.forward_gpu_with_cache_keep_gpu(&[id], &mut gpu_cache) {
                Ok(_) => (),
                Err(e) => {
                    tracing::warn!("GPU prefill failed at token {id}: {e}, falling back");
                    return self.generate(prompt_ids, max_tokens, temperature, top_k);
                }
            };
        }

        let mut rng_seed = seed;

        // GPU-resident generation loop
        for _ in 0..max_tokens {
            let logits_gpu = match self.forward_gpu_with_cache_keep_gpu(&[last_id], &mut gpu_cache)
            {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!("GPU forward failed at step: {e}, falling back");
                    break;
                }
            };

            let token_gpu =
                match sample_token_gpu_keep_gpu(&logits_gpu, temperature, top_k, top_p, rng_seed) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::warn!("GPU sampling failed: {e}, falling back");
                        break;
                    }
                };

            // Read back only 4 bytes per token
            let raw = match token_gpu.to_cpu_raw_bytes() {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("GPU token readback failed: {e}, falling back");
                    break;
                }
            };
            let token = u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);

            output.push(token);
            last_id = token;
            rng_seed = rng_seed.wrapping_add(1);

            if token == 2 {
                break;
            }
        }

        // One-time GPU readback: convert GpuKVCacheEntry -> KVCacheEntry
        let cpu_cache: Vec<KVCacheEntry> = gpu_cache
            .iter()
            .map(|entry| {
                let k_cpu = entry.k_view().to_cpu().unwrap_or_else(|e| {
                    tracing::warn!("KV cache GPU readback k failed: {e}");
                    ndarray::ArrayD::zeros(vec![0])
                });
                let v_cpu = entry.v_view().to_cpu().unwrap_or_else(|e| {
                    tracing::warn!("KV cache GPU readback v failed: {e}");
                    ndarray::ArrayD::zeros(vec![0])
                });
                let kv_dim = entry.kv_heads * entry.head_dim;
                KVCacheEntry {
                    k: k_cpu.into_iter().collect(),
                    v: v_cpu.into_iter().collect(),
                    kv_dim,
                    num_kv_heads: entry.kv_heads,
                    head_dim: entry.head_dim,
                    k_compressed: Vec::new(),
                    v_compressed: Vec::new(),
                    k_scales: Vec::new(),
                    v_scales: Vec::new(),
                    compressed_seq_len: 0,
                }
            })
            .collect();

        (output, cpu_cache)
    }

    pub fn memory_bytes(&self) -> f64 {
        self.parameter_count() as f64 * self.config.bytes_per_param()
    }

    /// Bytes per parameter based on this model's config quantization format.
    pub fn bytes_per_param(&self) -> f64 {
        self.config.bytes_per_param()
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

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = 1;
        let hidden_size = self.config.hidden_size;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                if tid >= self.config.vocab_size {
                    return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                        "Token ID {} out of range [0, {})",
                        tid, self.config.vocab_size
                    )));
                }
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gw.token_embedding.buffer(),
                        offset,
                        h.buffer(),
                        0,
                        row_bytes,
                    );
                    Ok(())
                })?;
                h
            }
            None => GpuTensor::zeros(&[batch_size, hidden_size])?,
        };

        // 2. Get RoPE cos/sin for current position
        let pos = kv_cache.first().map(|e| e.seq_len()).unwrap_or(0);
        let (cos_slice, sin_slice) = self.get_cos_sin_arrays(pos);

        let collective = self.collective.as_ref();

        // 3. Forward through all transformer blocks on GPU
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_collective(&h, kv_cache, layer_idx, &cos_slice, &sin_slice, collective)?;

            // Run injectors (Echo-Net APSS) after each layer
            for (target_layer, injector) in &self.injectors {
                if *target_layer == layer_idx {
                    let mut guard = injector.lock().map_err(|e| {
                        nexora_autograd::gpu::GpuError::Unsupported(format!("injector lock: {}", e))
                    })?;
                    guard.after_layer_gpu(layer_idx, &mut h, pos, ctx)?;
                }
            }
        }

        // 4. Final RMSNorm on GPU
        h = self.norm.forward_gpu(&h)?;

        // 5. LM head: matmul(h, lm_head^T) on GPU with pre-transposed weight.
        let logits = ctx.matmul(&h, &gw.lm_head_t)?;

        // 6. Download logits to CPU (only at final return point)
        let logits_flat: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// GPU forward pass that enforces GPU residency — never falls back to CPU.
    /// If GPU fails, the error is propagated instead of silently falling back.
    /// Preferred method when `keep_on_gpu` is set.
    #[cfg(feature = "gpu")]
    pub fn forward_keep_gpu(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Vec<KVCacheEntry>,
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = 1;
        let hidden_size = self.config.hidden_size;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                if tid >= self.config.vocab_size {
                    return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                        "Token ID {} out of range [0, {})",
                        tid, self.config.vocab_size
                    )));
                }
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gw.token_embedding.buffer(),
                        offset,
                        h.buffer(),
                        0,
                        row_bytes,
                    );
                    Ok(())
                })?;
                h
            }
            None => GpuTensor::zeros(&[batch_size, hidden_size])?,
        };

        let pos = kv_cache.first().map(|e| e.seq_len()).unwrap_or(0);
        let (cos_slice, sin_slice) = self.get_cos_sin_arrays(pos);

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_collective(&h, kv_cache, layer_idx, &cos_slice, &sin_slice, collective)?;

            // Run injectors (Echo-Net APSS) after each layer
            for (target_layer, injector) in &self.injectors {
                if *target_layer == layer_idx {
                    let mut guard = injector.lock().map_err(|e| {
                        nexora_autograd::gpu::GpuError::Unsupported(format!("injector lock: {}", e))
                    })?;
                    guard.after_layer_gpu(layer_idx, &mut h, pos, ctx)?;
                }
            }
        }

        h = self.norm.forward_gpu(&h)?;

        let logits = ctx.matmul(&h, &gw.lm_head_t)?;

        let logits_flat: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// Truly batched GPU forward pass: processes ALL sequences in parallel
    /// through batched matmuls. QKV projections, output projections, FFN,
    /// final norm, and LM head are all batched [batch_size, hidden_size]
    /// matmuls. Per-sequence RoPE + attention is unavoidable due to
    /// per-sequence KV caches with independent lengths.
    ///
    /// Single LM head readback at the end — no per-sequence CPU sync.
    #[cfg(feature = "gpu")]
    fn prepare_f16_temps(
        &self,
        _gw: &GpuWeights,
        _ctx: &nexora_autograd::gpu::GpuContext,
        _num_layers: usize,
    ) -> Result<Option<(Vec<[nexora_autograd::gpu::GpuTensor; 7]>, Option<nexora_autograd::gpu::GpuTensor>)>, nexora_autograd::gpu::GpuError> {
        // F16 matmul uses native packed F16 weights directly via auto-dispatch in
        // GpuContext::matmul(). No upconversion needed — saves 2x VRAM bandwidth
        // and eliminates per-forward f16_packed→f32 overhead.
        Ok(None)
    }

    #[cfg(feature = "gpu")]
    fn forward_gpu_single_token_core(
        &self,
        batch_tokens: &[u32],
        ctx: &nexora_autograd::gpu::GpuContext,
        gpu_caches: &mut [Vec<crate::gqa::GpuKVCacheEntry>],
        _f16_temps: &Option<(Vec<[nexora_autograd::gpu::GpuTensor; 7]>, Option<nexora_autograd::gpu::GpuTensor>)>,
        gw: &GpuWeights,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::GpuTensor;

        let batch_size = batch_tokens.len();
        let hidden_size = self.config.hidden_size;
        let _num_layers = self.config.num_layers;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let vocab_size = self.config.vocab_size;
        let n_heads = self.config.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let half = head_dim / 2;

        // ── 1. Batched embedding: copy ALL token rows into one [batch_size, hidden_size] tensor ──
        let row_bytes = (hidden_size * 4) as u64;
        let mut h = GpuTensor::zeros(&[batch_size, hidden_size])?;
        for &token_id in batch_tokens.iter() {
            let tid = token_id as usize;
            if tid >= vocab_size {
                return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                    "Token ID {} out of range [0, {})",
                    tid, vocab_size
                )));
            }
        }
        ctx.batch_dispatch(|enc| {
            for (seq_idx, &token_id) in batch_tokens.iter().enumerate() {
                let offset = (token_id as usize * hidden_size * 4) as u64;
                let dst_offset = (seq_idx * hidden_size * 4) as u64;
                enc.copy_buffer_to_buffer(
                    gw.token_embedding.buffer(),
                    offset,
                    h.buffer(),
                    dst_offset,
                    row_bytes,
                );
            }
            Ok(())
        })?;

        // ── 2. Forward through all blocks (batched QKV/FFN, per-sequence attention) ──
        let q_dim = n_heads * head_dim;
        let kv_dim_total = n_kv_heads * head_dim;

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let block_gw = &gw.block_weights[layer_idx];

            // Batched RMSNorm
            let normed = block.attention_norm.forward_gpu(&h)?;

            // Batched QKV projection: 3 matmuls with [batch_size, hidden_size]
            let q_proj = if let Some(ref wq_i8) = block_gw.wq_i8 {
                ctx.matmul_int8_weight(&normed, wq_i8, block_gw.wq_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wq_scales required".into()))?, block_gw.wq_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wq_zero_points required".into()))?)?
            } else if let Some(ref wq_f16) = block_gw.wq_f16 {
                ctx.matmul(&normed, wq_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wq_t)?
            };
            let k_proj = if let Some(ref wk_i8) = block_gw.wk_i8 {
                ctx.matmul_int8_weight(&normed, wk_i8, block_gw.wk_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wk_scales required".into()))?, block_gw.wk_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wk_zero_points required".into()))?)?
            } else if let Some(ref wk_f16) = block_gw.wk_f16 {
                ctx.matmul(&normed, wk_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wk_t)?
            };
            let v_proj = if let Some(ref wv_i8) = block_gw.wv_i8 {
                ctx.matmul_int8_weight(&normed, wv_i8, block_gw.wv_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wv_scales required".into()))?, block_gw.wv_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wv_zero_points required".into()))?)?
            } else if let Some(ref wv_f16) = block_gw.wv_f16 {
                ctx.matmul(&normed, wv_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wv_t)?
            };

            // Batch-upload cos/sin for ALL sequences at once: [batch_size, half]
            let mut cos_flat = Vec::with_capacity(batch_size * half);
            let mut sin_flat = Vec::with_capacity(batch_size * half);
            for seq_idx in 0..batch_size {
                let pos = gpu_caches[seq_idx].first().map(|e| e.seq_len).unwrap_or(0);
                let (cos_slice, sin_slice) = self.get_cos_sin_arrays(pos);
                cos_flat.extend_from_slice(cos_slice.as_slice().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported("cos_slice not contiguous".into())
                })?);
                sin_flat.extend_from_slice(sin_slice.as_slice().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported("sin_slice not contiguous".into())
                })?);
            }
            let cos_gpu_batch = GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(vec![batch_size, half], cos_flat)
                    .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
            )?;
            let sin_gpu_batch = GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(vec![batch_size, half], sin_flat)
                    .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
            )?;

            let mut q_rows: Vec<GpuTensor> = Vec::with_capacity(batch_size);
            let mut k_rows: Vec<GpuTensor> = Vec::with_capacity(batch_size);
            let mut v_rows: Vec<GpuTensor> = Vec::with_capacity(batch_size);
            let mut cos_rows: Vec<GpuTensor> = Vec::with_capacity(batch_size);
            let mut sin_rows: Vec<GpuTensor> = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                q_rows.push(GpuTensor::zeros(&[1, q_dim])?);
                k_rows.push(GpuTensor::zeros(&[1, kv_dim_total])?);
                v_rows.push(GpuTensor::zeros(&[1, kv_dim_total])?);
                cos_rows.push(GpuTensor::zeros(&[1, half])?);
                sin_rows.push(GpuTensor::zeros(&[1, half])?);
            }

            // Single batched copy dispatch — all sequences' rows copied in one encoder pass
            ctx.batch_dispatch(|enc| {
                let qb = q_proj.buffer();
                let kb = k_proj.buffer();
                let vb = v_proj.buffer();
                let cb = cos_gpu_batch.buffer();
                let sb = sin_gpu_batch.buffer();
                for seq_idx in 0..batch_size {
                    enc.copy_buffer_to_buffer(qb, (seq_idx * q_dim * 4) as u64, q_rows[seq_idx].buffer(), 0, (q_dim * 4) as u64);
                    enc.copy_buffer_to_buffer(kb, (seq_idx * kv_dim_total * 4) as u64, k_rows[seq_idx].buffer(), 0, (kv_dim_total * 4) as u64);
                    enc.copy_buffer_to_buffer(vb, (seq_idx * kv_dim_total * 4) as u64, v_rows[seq_idx].buffer(), 0, (kv_dim_total * 4) as u64);
                    enc.copy_buffer_to_buffer(cb, (seq_idx * half * 4) as u64, cos_rows[seq_idx].buffer(), 0, (half * 4) as u64);
                    enc.copy_buffer_to_buffer(sb, (seq_idx * half * 4) as u64, sin_rows[seq_idx].buffer(), 0, (half * 4) as u64);
                }
                Ok(())
            })?;

            let mut attn_rows = Vec::with_capacity(batch_size);

            for seq_idx in 0..batch_size {
                // RoPE (per-sequence because positions differ)
                let q_rotated =
                    ctx.rotary_embedding(&q_rows[seq_idx], &cos_rows[seq_idx], &sin_rows[seq_idx], head_dim as u32)?;
                let k_rotated =
                    ctx.rotary_embedding(&k_rows[seq_idx], &cos_rows[seq_idx], &sin_rows[seq_idx], head_dim as u32)?;

                // Append K/V to per-sequence GPU cache
                let k_3d = k_rotated.reshape(vec![1, n_kv_heads, head_dim])?;
                let v_3d = v_rows[seq_idx].reshape(vec![1, n_kv_heads, head_dim])?;
                gpu_caches[seq_idx][layer_idx].append(&ctx, &k_3d, &v_3d)?;

                // Fused attention with cached K/V
                let total_seq = gpu_caches[seq_idx][layer_idx].seq_len;
                let (k_rep, v_rep) =
                    gpu_caches[seq_idx][layer_idx].get_repeated_kv(&ctx, n_heads as u32)?;
                let q_4d = q_rotated.reshape(vec![1, n_heads, 1, head_dim])?;
                let k_4d = k_rep.reshape(vec![1, n_heads, total_seq, head_dim])?;
                let v_4d = v_rep.reshape(vec![1, n_heads, total_seq, head_dim])?;

                let attn_out = ctx.fused_attention(&q_4d, &k_4d, &v_4d, scale, false)?;
                attn_rows.push(attn_out.reshape(vec![1, n_heads * head_dim])?);
            }

            // Stack per-sequence attention outputs → [batch_size, hidden_size]
            let attn_concat = GpuTensor::zeros(&[batch_size, hidden_size])?;
            ctx.batch_dispatch(|enc| {
                for (seq_idx, row) in attn_rows.iter().enumerate() {
                    enc.copy_buffer_to_buffer(
                        row.buffer(),
                        0,
                        attn_concat.buffer(),
                        (seq_idx * hidden_size * 4) as u64,
                        (hidden_size * 4) as u64,
                    );
                }
                Ok(())
            })?;

            // Batched output projection + residual
            let attn_proj = if let Some(ref wo_i8) = block_gw.wo_i8 {
                ctx.matmul_int8_weight(&attn_concat, wo_i8, block_gw.wo_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wo_scales required".into()))?, block_gw.wo_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wo_zero_points required".into()))?)?
            } else if let Some(ref wo_f16) = block_gw.wo_f16 {
                ctx.matmul(&attn_concat, wo_f16)?
            } else {
                ctx.matmul(&attn_concat, &block_gw.wo_t)?
            };
            h = ctx.add(&h, &attn_proj)?;
            h = crate::block::collective_gpu_all_reduce(&h, collective, ctx)?;

            // Batched FFN sub-block
            let normed_ffn = block.ffn_norm.forward_gpu(&h)?;
            let ffn_gate = if let Some(ref w1_i8) = block_gw.w1_i8 {
                ctx.matmul_int8_weight(&normed_ffn, w1_i8, block_gw.w1_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w1_scales required".into()))?, block_gw.w1_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w1_zero_points required".into()))?)?
            } else if let Some(ref w1_f16) = block_gw.w1_f16 {
                ctx.matmul(&normed_ffn, w1_f16)?
            } else {
                ctx.matmul(&normed_ffn, &block_gw.w1_t)?
            };
            let ffn_hidden = if let Some(ref w3_i8) = block_gw.w3_i8 {
                ctx.matmul_int8_weight(&normed_ffn, w3_i8, block_gw.w3_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w3_scales required".into()))?, block_gw.w3_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w3_zero_points required".into()))?)?
            } else if let Some(ref w3_f16) = block_gw.w3_f16 {
                ctx.matmul(&normed_ffn, w3_f16)?
            } else {
                ctx.matmul(&normed_ffn, &block_gw.w3_t)?
            };
            let ffn_silu_mul = ctx.mul(&ctx.silu(&ffn_gate)?, &ffn_hidden)?;
            let ffn_out = if let Some(ref w2_i8) = block_gw.w2_i8 {
                ctx.matmul_int8_weight(&ffn_silu_mul, w2_i8, block_gw.w2_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w2_scales required".into()))?, block_gw.w2_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w2_zero_points required".into()))?)?
            } else if let Some(ref w2_f16) = block_gw.w2_f16 {
                ctx.matmul(&ffn_silu_mul, w2_f16)?
            } else {
                ctx.matmul(&ffn_silu_mul, &block_gw.w2_t)?
            };
            h = ctx.add(&h, &ffn_out)?;
            h = crate::block::collective_gpu_all_reduce(&h, collective, ctx)?;
        }

        // ── 3. Final norm + SINGLE LM head matmul (batched) ──
        h = self.norm.forward_gpu(&h)?;
        let logits = if let Some(ref lm_head_i8) = gw.lm_head_i8 {
            ctx.matmul_int8_weight(&h, lm_head_i8, gw.lm_head_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: lm_head_scales required".into()))?, gw.lm_head_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: lm_head_zero_points required".into()))?)?
        } else if let Some(ref lm_head_f16) = gw.lm_head_f16 {
            ctx.matmul(&h, lm_head_f16)?
        } else {
            ctx.matmul(&h, &gw.lm_head_t)?
        };

        Ok(logits)
    }

    #[cfg(feature = "gpu")]
    pub fn forward_gpu_batched(
        &self,
        batch_tokens: &[u32],
        kv_caches: &mut [Vec<KVCacheEntry>],
    ) -> Result<Vec<Array1<f32>>, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = batch_tokens.len();
        let _hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let max_seq = self.config.max_seq_len;
        let vocab_size = self.config.vocab_size;
        let _n_heads = self.config.num_heads;
        let _scale = 1.0 / (head_dim as f32).sqrt();
        let _half = head_dim / 2;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let f16_temps = self.prepare_f16_temps(gw, &ctx, num_layers)?;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        ctx.begin_batch_mode();

        // Create per-sequence GPU KV caches (one Vec<GpuKVCacheEntry> per sequence)
        use crate::gqa::GpuKVCacheEntry;
        let mut gpu_caches: Vec<Vec<GpuKVCacheEntry>> = (0..batch_size)
            .map(|_| {
                (0..num_layers)
                    .map(|_| {
                        GpuKVCacheEntry::new(
                            &ctx,
                            n_kv_heads,
                            head_dim,
                            max_seq,
                            self.use_half_precision,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Sync existing CPU data to GPU caches (one-time O(seq) per sequence)
        for seq_idx in 0..batch_size {
            let cpu_cache = &kv_caches[seq_idx];
            for layer_idx in 0..num_layers {
                if layer_idx < cpu_cache.len() && cpu_cache[layer_idx].seq_len() > 0 {
                    let ce = &cpu_cache[layer_idx];
                    let seq = ce.seq_len();
                    let k_cpu =
                        ArrayD::from_shape_vec(vec![seq, n_kv_heads, head_dim], ce.k.clone())
                            .map_err(|e| {
                                nexora_autograd::gpu::GpuError::Unsupported(e.to_string())
                            })?;
                    let v_cpu =
                        ArrayD::from_shape_vec(vec![seq, n_kv_heads, head_dim], ce.v.clone())
                            .map_err(|e| {
                                nexora_autograd::gpu::GpuError::Unsupported(e.to_string())
                            })?;
                    let k_gpu = GpuTensor::from_cpu(&k_cpu)?;
                    let v_gpu = GpuTensor::from_cpu(&v_cpu)?;
                    let bytes = (seq * n_kv_heads * head_dim * 4) as u64;
                    ctx.batch_dispatch(|enc| {
                        enc.copy_buffer_to_buffer(
                            k_gpu.buffer(),
                            0,
                            gpu_caches[seq_idx][layer_idx].k.buffer(),
                            0,
                            bytes,
                        );
                        enc.copy_buffer_to_buffer(
                            v_gpu.buffer(),
                            0,
                            gpu_caches[seq_idx][layer_idx].v.buffer(),
                            0,
                            bytes,
                        );
                        Ok(())
                    })?;
                    gpu_caches[seq_idx][layer_idx].seq_len = seq;
                }
            }
        }

        let logits = self.forward_gpu_single_token_core(batch_tokens, &ctx, &mut gpu_caches, &f16_temps, gw)?;

        // ── 4. SINGLE readback — split into per-sequence logits ──
        let logits_data: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
        let all_logits: Vec<Array1<f32>> = logits_data
            .chunks(vocab_size)
            .map(|chunk| Array1::from_vec(chunk.to_vec()))
            .collect();

        // ── 5. Sync back new token's K/V from GPU cache to each CPU cache ──
        let kv_elems = n_kv_heads * head_dim;
        for seq_idx in 0..batch_size {
            let cpu_cache = &mut kv_caches[seq_idx];
            for layer_idx in 0..num_layers {
                let gpu_entry = &gpu_caches[seq_idx][layer_idx];
                if gpu_entry.seq_len == 0 {
                    continue;
                }
                let last_idx = gpu_entry.seq_len - 1;
                let elem_size: u64 = if gpu_entry.f16_storage { 2 } else { 4 };
                let byte_off = (last_idx * kv_elems) as u64 * elem_size;
                let token_bytes = (kv_elems as u64) * elem_size;

                let staging_k = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("batched_sync_k"),
                    size: token_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let staging_v = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("batched_sync_v"),
                    size: token_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gpu_entry.k.buffer(),
                        byte_off,
                        &staging_k,
                        0,
                        token_bytes,
                    );
                    enc.copy_buffer_to_buffer(
                        gpu_entry.v.buffer(),
                        byte_off,
                        &staging_v,
                        0,
                        token_bytes,
                    );
                    Ok(())
                })?;

                ctx.sync();

                let read_back =
                    |staging: &wgpu::Buffer,
                     is_f16: bool|
                     -> Result<Vec<f32>, nexora_autograd::gpu::GpuError> {
                        let slice = staging.slice(..);
                        let (tx, rx) = std::sync::mpsc::channel();
                        slice.map_async(wgpu::MapMode::Read, move |r| {
                            let _ = tx.send(r);
                        });
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(10);
                        let map_result = loop {
                            ctx.device.poll(wgpu::PollType::Wait {
                                submission_index: None,
                                timeout: Some(std::time::Duration::from_millis(100)),
                            });
                            match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                                Ok(result) => break result,
                                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                                    if std::time::Instant::now() > deadline {
                                        break Err(wgpu::BufferAsyncError);
                                    }
                                    continue;
                                }
                                Err(_) => break Err(wgpu::BufferAsyncError),
                            }
                        };
                        let _map_result = map_result.map_err(|e| {
                            nexora_autograd::gpu::GpuError::Device(format!("map_async: {e:?}"))
                        })?;
                        let out: Vec<f32> = {
                            let mapped = slice.get_mapped_range();
                            let result = if is_f16 {
                                mapped
                                    .chunks_exact(2)
                                    .map(|c| {
                                        let bits = u16::from_ne_bytes([c[0], c[1]]);
                                        half::f16::from_bits(bits).to_f32()
                                    })
                                    .collect()
                            } else {
                                mapped
                                    .chunks_exact(4)
                                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect()
                            };
                            result
                        };
                        staging.unmap();
                        Ok(out)
                    };

                    let new_k = read_back(&staging_k, gpu_entry.f16_storage)?;
                    let new_v = read_back(&staging_v, gpu_entry.f16_storage)?;

                    if layer_idx < cpu_cache.len() {
                    let entry = &mut cpu_cache[layer_idx];
                    entry.k.extend_from_slice(&new_k);
                    entry.v.extend_from_slice(&new_v);
                } else {
                    cpu_cache.push(KVCacheEntry {
                        k: new_k,
                        v: new_v,
                        kv_dim: kv_elems,
                        num_kv_heads: n_kv_heads,
                        head_dim,
                        k_compressed: Vec::new(),
                        v_compressed: Vec::new(),
                        k_scales: Vec::new(),
                        v_scales: Vec::new(),
                        compressed_seq_len: 0,
                    });
                }
            }
        }

        ctx.end_batch_mode();
        Ok(all_logits)
    }

    /// Truly batched GPU **prefill** forward pass.
    ///
    /// Like [`forward_gpu_batched`]], but operates on **existing GPU KV caches**
    /// instead of creating fresh ones from CPU entries. Key differences:
    ///
    /// 1. Skips the "create GpuKVCacheEntry + upload CPU K/V" phase
    /// 2. Skips the "sync GPU→CPU" phase (entries stay on GPU)
    /// 3. Returns `Vec<Option<Array1<f32>>>` — only reads back logits for
    ///    sequences where `needs_logits[i] == true`
    ///
    /// At each position, ALL active sequences' tokens are processed together
    /// with batched QKV projections, FFN, and LM head matmuls. The per-sequence
    /// RoPE + attention + KV cache append is unavoidable but uses GPU entries
    /// in-place with zero CPU round-trip.
    ///
    /// Designed to be called from `forward_prefill_batched` position-by-position:
    /// each call processes one token position across all sequences still in prefill.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_batched_prefill(
        &self,
        batch_tokens: &[u32],
        gpu_caches: &mut [Vec<crate::gqa::GpuKVCacheEntry>],
        needs_logits: &[bool],
    ) -> Result<Vec<Option<Array1<f32>>>, nexora_autograd::gpu::GpuError> {
        
        use nexora_autograd::gpu::GpuContext;

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = batch_tokens.len();
        let _hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let _n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let vocab_size = self.config.vocab_size;
        let _n_heads = self.config.num_heads;
        let _scale = 1.0 / (head_dim as f32).sqrt();
        let _half = head_dim / 2;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let f16_temps = self.prepare_f16_temps(gw, &ctx, num_layers)?;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        ctx.begin_batch_mode();

        let logits = self.forward_gpu_single_token_core(batch_tokens, &ctx, gpu_caches, &f16_temps, gw)?;

        // ── 4. Conditional readback: only for sequences at their last token ──
        let needs_any = needs_logits.iter().any(|&n| n);
        let all_logits: Vec<Option<Array1<f32>>> = if needs_any {
            let logits_data: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
            logits_data
                .chunks(vocab_size)
                .zip(needs_logits.iter())
                .map(|(chunk, &need)| {
                    if need {
                        Some(Array1::from_vec(chunk.to_vec()))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            vec![None; batch_size]
        };

        // ── SKIP Phase 5: No GPU→CPU sync — entries stay on GPU ──

        ctx.end_batch_mode();
        Ok(all_logits)
    }

    /// Batched GPU forward + GPU sampling — processes one token per sequence on GPU
    /// and samples next tokens entirely on device. Zero CPU readback of full logits
    /// (128KB/seq). Only token IDs (4 bytes/seq) are read back.
    ///
    /// Takes existing GPU KV cache entries (no CPU→GPU upload, no GPU→CPU sync).
    /// Per-sequence sampling parameters for temperature/top_k/top_p.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_batched_sample(
        &self,
        batch_tokens: &[u32],
        gpu_caches: &mut [Vec<crate::gqa::GpuKVCacheEntry>],
        temperatures: &[f32],
        top_ks: &[u32],
        top_ps: &[f32],
        seeds: &[u64],
    ) -> Result<Vec<u32>, nexora_autograd::gpu::GpuError> {
        
        use nexora_autograd::gpu::GpuContext;

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = batch_tokens.len();
        let _hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let _n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let _vocab_size = self.config.vocab_size;
        let _n_heads = self.config.num_heads;
        let _scale = 1.0 / (head_dim as f32).sqrt();
        let _half = head_dim / 2;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let f16_temps = self.prepare_f16_temps(gw, &ctx, num_layers)?;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        ctx.begin_batch_mode();

        let logits = self.forward_gpu_single_token_core(batch_tokens, &ctx, gpu_caches, &f16_temps, gw)?;

        // ── 4. GPU sampling (zero logit readback) — batched when params are uniform ──
        let token_ids = if batch_size > 1
            && temperatures.iter().all(|&t| (t - temperatures[0]).abs() < 1e-6)
            && top_ks.iter().all(|&k| k == top_ks[0])
            && top_ps.iter().all(|&p| (p - top_ps[0]).abs() < 1e-6)
        {
            // ── Fast batched path: single gpu_sample call + single readback ──
            // WGSL per-row seed mixing (seed ^ row * 0x9E3779B9) already
            // gives each row unique random values — one seed is sufficient.
            let tok = ctx.gpu_sample(
                &logits,
                temperatures[0],
                top_ks[0],
                top_ps[0],
                seeds[0],
            )?;
            let raw = tok.to_cpu_raw_bytes()?;
            raw.chunks_exact(4)
                .map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        } else {
            // ── Per-seq path: different params per sequence ──
            let mut ids = Vec::with_capacity(batch_size);
            for seq_idx in 0..batch_size {
                let logits_row = ctx.slice_tensor(&logits, seq_idx as u32, 1)?;
                let tok = ctx.gpu_sample(
                    &logits_row,
                    temperatures[seq_idx],
                    top_ks[seq_idx],
                    top_ps[seq_idx],
                    seeds[seq_idx],
                )?;
                let raw = tok.to_cpu_raw_bytes()?;
                ids.push(u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]));
            }
            ids
        };

        // ── SKIP Phase 5: No GPU→CPU sync — entries stay on GPU ──

        ctx.end_batch_mode();
        Ok(token_ids)
    }

    /// True GPU padded-batch prefill — processes ALL remaining tokens across ALL
    /// sequences in ONE forward pass (not position-by-position).
    ///
    /// Key differences from `forward_gpu_batched_prefill`:
    /// - Input is `&[Vec<u32>]` (per-seq remaining tokens) not `&[u32]` (one token each)
    /// - QKV/FFN/LM-head matmuls on `[total_tokens, hidden]` (not `[N, hidden]`)
    /// - Batched RoPE with per-token position IDs
    /// - Bulk KV append per sequence (one GPU copy, not N copies)
    /// - Multi-token attention per sequence with `causal=true`
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_batched_prefill_full(
        &self,
        batch_inputs: &[&[u32]],
        gpu_caches: &mut [Vec<crate::gqa::GpuKVCacheEntry>],
    ) -> Result<Vec<Array1<f32>>, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let n = batch_inputs.len();
        if n == 0 || n != gpu_caches.len() {
            return Ok(Vec::new());
        }
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let vocab_size = self.config.vocab_size;
        let n_heads = self.config.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let half = head_dim / 2;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let _f16_temps = self.prepare_f16_temps(gw, &ctx, num_layers)?;

        // ── 0b. Build flat token array + per-sequence offsets + position IDs ──
        let mut flat_tokens: Vec<u32> = Vec::new();
        let mut seq_offsets: Vec<(usize, usize)> = Vec::with_capacity(n);
        for seq_tokens in batch_inputs {
            let start = flat_tokens.len();
            flat_tokens.extend_from_slice(seq_tokens);
            seq_offsets.push((start, seq_tokens.len()));
        }
        let total_tokens = flat_tokens.len();
        if total_tokens == 0 {
            return Ok(vec![Array1::zeros(1); n]);
        }

        // Validate token IDs
        for &token_id in &flat_tokens {
            let tid = token_id as usize;
            if tid >= vocab_size {
                return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                    "Token ID {} out of range [0, {})",
                    tid, vocab_size
                )));
            }
        }

        ctx.begin_batch_mode();

        // ── 1. Batched embedding ──
        let row_bytes = (hidden_size * 4) as u64;
        let mut h = GpuTensor::zeros(&[total_tokens, hidden_size])?;
        ctx.batch_dispatch(|enc| {
            for (flat_idx, &token_id) in flat_tokens.iter().enumerate() {
                let offset = (token_id as usize * hidden_size * 4) as u64;
                let dst_offset = (flat_idx * hidden_size * 4) as u64;
                enc.copy_buffer_to_buffer(
                    gw.token_embedding.buffer(),
                    offset,
                    h.buffer(),
                    dst_offset,
                    row_bytes,
                );
            }
            Ok(())
        })?;

        // ── 1b. Build position-aware cos/sin for ALL tokens ──
        let mut cos_flat = Vec::with_capacity(total_tokens * half);
        let mut sin_flat = Vec::with_capacity(total_tokens * half);
        for (i, seq_tokens) in batch_inputs.iter().enumerate() {
            let start_pos = gpu_caches[i].first().map(|e| e.seq_len).unwrap_or(0);
            for j in 0..seq_tokens.len() {
                let pos = start_pos + j;
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
                cos_flat.extend_from_slice(cos_slice.as_slice().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported("cos_slice not contiguous".into())
                })?);
                sin_flat.extend_from_slice(sin_slice.as_slice().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported("sin_slice not contiguous".into())
                })?);
            }
        }
        let cos_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![total_tokens, half], cos_flat)
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![total_tokens, half], sin_flat)
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;

        // ── 2. Forward through all blocks ──
        let _q_dim = n_heads * head_dim;
        let _kv_dim_total = n_kv_heads * head_dim;

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let block_gw = &gw.block_weights[layer_idx];

            // ── 2a. Batched RMSNorm + QKV ──
            let normed = block.attention_norm.forward_gpu(&h)?;

            let q_proj = if let Some(ref wq_i8) = block_gw.wq_i8 {
                ctx.matmul_int8_weight(&normed, wq_i8, block_gw.wq_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wq_scales required".into()))?, block_gw.wq_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wq_zero_points required".into()))?)?
            } else if let Some(ref wq_f16) = block_gw.wq_f16 {
                ctx.matmul(&normed, wq_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wq_t)?
            };
            let k_proj = if let Some(ref wk_i8) = block_gw.wk_i8 {
                ctx.matmul_int8_weight(&normed, wk_i8, block_gw.wk_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wk_scales required".into()))?, block_gw.wk_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wk_zero_points required".into()))?)?
            } else if let Some(ref wk_f16) = block_gw.wk_f16 {
                ctx.matmul(&normed, wk_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wk_t)?
            };
            let v_proj = if let Some(ref wv_i8) = block_gw.wv_i8 {
                ctx.matmul_int8_weight(&normed, wv_i8, block_gw.wv_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wv_scales required".into()))?, block_gw.wv_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wv_zero_points required".into()))?)?
            } else if let Some(ref wv_f16) = block_gw.wv_f16 {
                ctx.matmul(&normed, wv_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wv_t)?
            };

            // ── 2b. Batched RoPE (per-token position) ──
            let q_rotated = ctx.rotary_embedding(&q_proj, &cos_gpu, &sin_gpu, head_dim as u32)?;
            let k_rotated = ctx.rotary_embedding(&k_proj, &cos_gpu, &sin_gpu, head_dim as u32)?;

            // ── 2c. Per-sequence: slice → bulk KV append → attention (causal=true) ──
            let mut attn_rows = Vec::with_capacity(n);
            for seq_idx in 0..n {
                let (start, remaining) = seq_offsets[seq_idx];

                let q_seq = ctx.slice_tensor(&q_rotated, start as u32, remaining as u32)?;
                let k_seq = ctx.slice_tensor(&k_rotated, start as u32, remaining as u32)?;
                let v_seq = ctx.slice_tensor(&v_proj, start as u32, remaining as u32)?;

                let k_3d = k_seq.reshape(vec![remaining, n_kv_heads, head_dim])?;
                let v_3d = v_seq.reshape(vec![remaining, n_kv_heads, head_dim])?;

                gpu_caches[seq_idx][layer_idx].bulk_append(&ctx, &k_3d, &v_3d, remaining)?;

                let total_seq = gpu_caches[seq_idx][layer_idx].seq_len;
                let (k_rep, v_rep) =
                    gpu_caches[seq_idx][layer_idx].get_repeated_kv(&ctx, n_heads as u32)?;

                let q_4d = q_seq.reshape(vec![1, n_heads, remaining, head_dim])?;
                let k_4d = k_rep.reshape(vec![1, n_heads, total_seq, head_dim])?;
                let v_4d = v_rep.reshape(vec![1, n_heads, total_seq, head_dim])?;

                let attn_out = ctx.fused_attention(&q_4d, &k_4d, &v_4d, scale, true)?;
                attn_rows.push(attn_out.reshape(vec![remaining, n_heads * head_dim])?);
            }

            // ── 2d. Gather attention outputs ──
            let attn_concat = GpuTensor::zeros(&[total_tokens, hidden_size])?;
            ctx.batch_dispatch(|enc| {
                for (seq_idx, row) in attn_rows.iter().enumerate() {
                    let (start, remaining) = seq_offsets[seq_idx];
                    enc.copy_buffer_to_buffer(
                        row.buffer(),
                        0,
                        attn_concat.buffer(),
                        (start * hidden_size * 4) as u64,
                        (remaining * hidden_size * 4) as u64,
                    );
                }
                Ok(())
            })?;

            // ── 2e. Batched Wo + residual ──
            let attn_proj = if let Some(ref wo_i8) = block_gw.wo_i8 {
                ctx.matmul_int8_weight(&attn_concat, wo_i8, block_gw.wo_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wo_scales required".into()))?, block_gw.wo_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wo_zero_points required".into()))?)?
            } else if let Some(ref wo_f16) = block_gw.wo_f16 {
                ctx.matmul(&attn_concat, wo_f16)?
            } else {
                ctx.matmul(&attn_concat, &block_gw.wo_t)?
            };
            h = ctx.add(&h, &attn_proj)?;
            h = crate::block::collective_gpu_all_reduce(&h, collective, ctx)?;

            // ── 2f. Batched FFN ──
            let normed_ffn = block.ffn_norm.forward_gpu(&h)?;
            let ffn_gate = if let Some(ref w1_i8) = block_gw.w1_i8 {
                ctx.matmul_int8_weight(&normed_ffn, w1_i8, block_gw.w1_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w1_scales required".into()))?, block_gw.w1_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w1_zero_points required".into()))?)?
            } else if let Some(ref w1_f16) = block_gw.w1_f16 {
                ctx.matmul(&normed_ffn, w1_f16)?
            } else {
                ctx.matmul(&normed_ffn, &block_gw.w1_t)?
            };
            let ffn_hidden = if let Some(ref w3_i8) = block_gw.w3_i8 {
                ctx.matmul_int8_weight(&normed_ffn, w3_i8, block_gw.w3_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w3_scales required".into()))?, block_gw.w3_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w3_zero_points required".into()))?)?
            } else if let Some(ref w3_f16) = block_gw.w3_f16 {
                ctx.matmul(&normed_ffn, w3_f16)?
            } else {
                ctx.matmul(&normed_ffn, &block_gw.w3_t)?
            };
            let ffn_silu_mul = ctx.mul(&ctx.silu(&ffn_gate)?, &ffn_hidden)?;
            let ffn_out = if let Some(ref w2_i8) = block_gw.w2_i8 {
                ctx.matmul_int8_weight(&ffn_silu_mul, w2_i8, block_gw.w2_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w2_scales required".into()))?, block_gw.w2_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w2_zero_points required".into()))?)?
            } else if let Some(ref w2_f16) = block_gw.w2_f16 {
                ctx.matmul(&ffn_silu_mul, w2_f16)?
            } else {
                ctx.matmul(&ffn_silu_mul, &block_gw.w2_t)?
            };
            h = ctx.add(&h, &ffn_out)?;
            h = crate::block::collective_gpu_all_reduce(&h, collective, ctx)?;
        }

        // ── 3. Final norm + batched LM head ──
        h = self.norm.forward_gpu(&h)?;
        let logits = if let Some(ref lm_head_i8) = gw.lm_head_i8 {
            ctx.matmul_int8_weight(&h, lm_head_i8, gw.lm_head_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: lm_head_scales required".into()))?, gw.lm_head_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: lm_head_zero_points required".into()))?)?
        } else if let Some(ref lm_head_f16) = gw.lm_head_f16 {
            ctx.matmul(&h, lm_head_f16)?
        } else {
            ctx.matmul(&h, &gw.lm_head_t)?
        };

        // ── 4. Readback: extract last-token logits per sequence ──
        let logits_data: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
        let mut results = Vec::with_capacity(n);
        for (start, remaining) in &seq_offsets {
            let last_idx = start + remaining - 1;
            let offset = last_idx * vocab_size;
            let chunk = &logits_data[offset..offset + vocab_size];
            results.push(Array1::from_vec(chunk.to_vec()));
        }

        ctx.end_batch_mode();
        Ok(results)
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, true)
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
            #[cfg(not(feature = "gpu"))]
            {
                warn!("use_gpu=true but GPU feature is not enabled — falling back to CPU");
            }
        }

        for &token_id in prompt_ids {
            if self.forward(&[token_id], &mut cache).is_err() {
                warn!("Forward failed during prefill for token {}", token_id);
            }
        }

        let mut output = Vec::new();
        let mut last_id = *prompt_ids.last().unwrap_or(&0);

        for _ in 0..max_tokens {
            let logits = self.forward(&[last_id], &mut cache).unwrap_or_else(|e| {
                warn!("Forward failed during generation: {e}");
                Array1::zeros(self.config.vocab_size)
            });
            let next_id = if use_gpu {
                #[cfg(feature = "gpu")]
                {
                    sample_token_gpu(&logits, temperature, top_k, 0.0, 12345)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    sample_token(&logits, temperature, top_k)
                }
            } else {
                sample_token(&logits, temperature, top_k)
            };
            output.push(next_id);
            // EOS token is 2 in MiniTokenizer; also stop on padding (0)
            if next_id == 0 || next_id == 2 {
                break;
            }
            last_id = next_id;
        }

        (output, cache.entries)
    }

    /// GPU-resident generation: logits stay GPU-native in the loop.
    /// Only 4 bytes per token read back (token ID), not full vocab_size vector.
    /// Falls back to `generate_with_gpu` if GPU unavailable or on error.
    #[cfg(feature = "gpu")]
    pub fn generate_with_gpu_keep_gpu(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        use nexora_autograd::gpu::GpuContext;

        match GpuContext::global() {
            Ok(_) => self.generate_gpu_keep_gpu_impl(
                prompt_ids,
                max_tokens,
                temperature,
                top_k,
                top_p,
                seed,
            ),
            Err(_) => {
                tracing::warn!("GPU context unavailable for GPU-resident generation, falling back to CPU generation");
                self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, false)
            }
        }
    }

    fn get_cos_sin_slices(&self, pos: usize) -> (&[f32], &[f32]) {
        let half = self.config.head_dim() / 2;
        let cos = if pos * half < self.precomputed_cos.len() {
            &self.precomputed_cos.as_slice().unwrap_or(&[])[pos * half..(pos + 1) * half]
        } else {
            &[]
        };
        let sin = if pos * half < self.precomputed_sin.len() {
            &self.precomputed_sin.as_slice().unwrap_or(&[])[pos * half..(pos + 1) * half]
        } else {
            &[]
        };
        (cos, sin)
    }

    fn get_cos_sin_arrays(&self, pos: usize) -> (ndarray::Array1<f32>, ndarray::Array1<f32>) {
        let half = self.config.head_dim() / 2;
        let cos = if pos * half < self.precomputed_cos.len() {
            self.precomputed_cos
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            ndarray::Array1::zeros(half)
        };
        let sin = if pos * half < self.precomputed_sin.len() {
            self.precomputed_sin
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            ndarray::Array1::zeros(half)
        };
        (cos, sin)
    }
}

