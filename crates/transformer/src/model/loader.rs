use ndarray::{Array1, Array2};
use crate::gqa::KVCacheEntry;
use crate::model::builder::{BlockGpuWeights, CausalLM, GpuWeights};
use crate::{TransformerConfig, TransformerError, TransformerResult};

impl CausalLM {
    /// Load all block weights from the lazy weight loader.
    /// Must be called after setting `lazy_loader` if weights are not already loaded.
    pub fn load_lazy_blocks(&mut self) -> TransformerResult<()> {
        let loader = self.lazy_loader.as_ref().ok_or_else(|| {
            TransformerError::Implementation("lazy_loader not set".into())
        })?;
        for i in 0..self.blocks.len() {
            loader.apply_block_weights(i, &mut self.blocks[i])?;
        }
        Ok(())
    }

    /// Ensure all lazy block weights are loaded before forward.
    /// If `unload_blocks_after_forward` is true, blocks are loaded
    /// on demand during `forward_cpu_impl`.
    pub fn ensure_blocks_loaded(&mut self) -> TransformerResult<()> {
        if self.blocks.iter().all(|b| b.attention.wq.is_some()) {
            return Ok(());
        }
        let loader = self.lazy_loader.as_ref().ok_or_else(|| {
            TransformerError::Implementation("Blocks not loaded and no lazy_loader set".into())
        })?;
        for i in 0..self.blocks.len() {
            let block = &mut self.blocks[i];
            if block.attention.wq.is_none() {
                loader.apply_block_weights(i, block)?;
            }
        }
        Ok(())
    }

    /// Unload all block weights from memory, freeing CPU RAM.
    /// After this call, `forward()` will fail — call `load_lazy_blocks()`
    /// or `ensure_blocks_loaded()` before the next forward.
    /// Also notifies the `LazyWeightLoader` to evict from its LRU cache.
    pub fn unload_blocks(&mut self) {
        for block in &mut self.blocks {
            block.attention.wq = None;
            block.attention.wk = None;
            block.attention.wv = None;
            block.attention.wo = None;
            block.ffn.w1 = None;
            block.ffn.w2 = None;
            block.ffn.w3 = None;
            block.attention_norm.weight = None;
            block.ffn_norm.weight = None;
        }
        if let Some(loader) = &self.lazy_loader {
            loader.evict_all_blocks();
        }
    }

    pub fn from_checkpoint(
        mut config: TransformerConfig,
        path: &str,
    ) -> crate::TransformerResult<Self> {
        let (loaded, meta) = crate::safetensors::load_safetensors_with_meta(path)?;
        // Apply quantization from file metadata if present
        if let Some(meta) = meta {
            if let Some(qstr) = meta.get("quantization") {
                match qstr.as_str() {
                    "F16" => config.quantization = nexora_quantization::QFormat::F16,
                    "BF16" => config.quantization = nexora_quantization::QFormat::BF16,
                    other => {
                        tracing::warn!(
                            "from_checkpoint: unknown quantization '{}' in file metadata, using config default",
                            other
                        );
                    }
                }
            }
        }
        let mut model = Self::new(config);
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
        model.token_embedding = Some(
            to_fixed::<ndarray::Ix2>(get_arr("token_embedding")?, "token_embedding")?.to_owned());
        model.lm_head = Some(
            to_fixed::<ndarray::Ix2>(get_arr("lm_head")?, "lm_head")?.to_owned());
        model.norm.weight = Some(
            to_fixed::<ndarray::Ix1>(get_arr("norm.weight")?, "norm.weight")?.to_owned());
        for (i, block) in model.blocks.iter_mut().enumerate() {
            let prefix = format!("blocks.{}.", i);
            let name = format!("{}attention_norm.weight", prefix);
            block.attention_norm.weight = Some(
                to_fixed::<ndarray::Ix1>(get_arr(&name)?, &name)?.to_owned());
            let name = format!("{}ffn_norm.weight", prefix);
            block.ffn_norm.weight = Some(
                to_fixed::<ndarray::Ix1>(get_arr(&name)?, &name)?.to_owned());
            let name = format!("{}attention.wq", prefix);
            block.attention.wq = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = format!("{}attention.wk", prefix);
            block.attention.wk = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = format!("{}attention.wv", prefix);
            block.attention.wv = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = format!("{}attention.wo", prefix);
            block.attention.wo = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = format!("{}ffn.w1", prefix);
            block.ffn.w1 = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = format!("{}ffn.w2", prefix);
            block.ffn.w2 = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = format!("{}ffn.w3", prefix);
            block.ffn.w3 = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            // Load expert weights if present in checkpoint
            if let Some(experts) = &mut block.experts {
                for (e_idx, expert) in experts.iter_mut().enumerate() {
                    let e_prefix = format!("{}experts.{}.", prefix, e_idx);
                    let w1_name = format!("{}w1", e_prefix);
                    if loaded.contains_key(&w1_name) {
                        expert.w1 = Some(to_fixed::<ndarray::Ix2>(get_arr(&w1_name)?, &w1_name)?.to_owned());
                        let w2_name = format!("{}w2", e_prefix);
                        expert.w2 = Some(to_fixed::<ndarray::Ix2>(get_arr(&w2_name)?, &w2_name)?.to_owned());
                        let w3_name = format!("{}w3", e_prefix);
                        expert.w3 = Some(to_fixed::<ndarray::Ix2>(get_arr(&w3_name)?, &w3_name)?.to_owned());
                    }
                }
            }
        }
        model.injectors = Vec::new();
        model.keep_on_gpu = {
            #[cfg(feature = "gpu")]
            {
                nexora_autograd::gpu::GpuContext::is_available()
            }
            #[cfg(not(feature = "gpu"))]
            {
                false
            }
        };
        Ok(model)
    }

    /// Load checkpoint directly to GPU — no CPU weight storage.
    /// CPU weights are never populated; all weight storage is GPU-only.
    /// Requires `gpu` feature. Returns CausalLM with `keep_on_gpu = true`.
    #[cfg(feature = "gpu")]
    pub fn from_checkpoint_gpu(
        mut config: TransformerConfig,
        path: &str,
    ) -> crate::TransformerResult<Self> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = GpuContext::global()
            .map_err(|e| crate::TransformerError::Implementation(format!("GPU unavailable: {e}")))?;
        let (loaded, meta) = crate::safetensors::load_safetensors_with_meta(path)?;
        // Apply quantization from file metadata if present
        if let Some(meta) = meta {
            if let Some(qstr) = meta.get("quantization") {
                match qstr.as_str() {
                    "F16" => config.quantization = nexora_quantization::QFormat::F16,
                    "BF16" => config.quantization = nexora_quantization::QFormat::BF16,
                    other => {
                        tracing::warn!(
                            "from_checkpoint_gpu: unknown quantization '{}' in file metadata, using config default",
                            other
                        );
                    }
                }
            }
        }
        let model = Self::new(config);

        let get_arr = |name: &str| -> crate::TransformerResult<ArrayD<f32>> {
            loaded.get(name).cloned().ok_or_else(|| {
                crate::TransformerError::Implementation(format!("Missing tensor: {}", name))
            })
        };
        let to_gpu = |arr: ArrayD<f32>| -> crate::TransformerResult<GpuTensor> {
            GpuTensor::from_cpu(&arr).map_err(|e| {
                crate::TransformerError::Implementation(format!("GPU upload failed: {e}"))
            })
        };

        let te_gpu = to_gpu(get_arr("token_embedding")?)?;
        let lh_gpu = to_gpu(get_arr("lm_head")?)?;
        let lm_head_t = ctx.transpose(&lh_gpu)
            .map_err(|e| crate::TransformerError::Implementation(format!("lm_head transpose: {e}")))?;
        let nw_gpu = to_gpu(get_arr("norm.weight")?)?;
        let _ = model.norm.gpu_weights.set(crate::rms_norm::RmsNormGpuWeights {
            weight: nw_gpu.clone(),
        });

        let mut block_weights = Vec::with_capacity(model.blocks.len());
        for (i, block) in model.blocks.iter().enumerate() {
            let p = format!("blocks.{}.", i);

            let an_gpu = to_gpu(get_arr(&format!("{}attention_norm.weight", p))?)?;
            let fn_gpu = to_gpu(get_arr(&format!("{}ffn_norm.weight", p))?)?;
            let _ = block.attention_norm.gpu_weights.set(
                crate::rms_norm::RmsNormGpuWeights { weight: an_gpu.clone() },
            );
            let _ = block.ffn_norm.gpu_weights.set(
                crate::rms_norm::RmsNormGpuWeights { weight: fn_gpu.clone() },
            );

            let wq = to_gpu(get_arr(&format!("{}attention.wq", p))?)?;
            let wk = to_gpu(get_arr(&format!("{}attention.wk", p))?)?;
            let wv = to_gpu(get_arr(&format!("{}attention.wv", p))?)?;
            let wo = to_gpu(get_arr(&format!("{}attention.wo", p))?)?;
            let wq_shape = wq.shape().to_vec();
            let wk_shape = wk.shape().to_vec();
            let wv_shape = wv.shape().to_vec();
            let wo_shape = wo.shape().to_vec();

            let use_f16 = model.use_half_precision && !model.quantize_weights;
            let (wq_t, wk_t, wv_t, wo_t, wq_f16, wk_f16, wv_f16, wo_f16) = if use_f16 {
                let wq_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&wq).map_err(|e| crate::TransformerError::Implementation(format!("{}wq t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}wq f16: {e}", p)))?;
                let wk_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&wk).map_err(|e| crate::TransformerError::Implementation(format!("{}wk t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}wk f16: {e}", p)))?;
                let wv_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&wv).map_err(|e| crate::TransformerError::Implementation(format!("{}wv t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}wv f16: {e}", p)))?;
                let wo_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&wo).map_err(|e| crate::TransformerError::Implementation(format!("{}wo t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}wo f16: {e}", p)))?;
                let d = GpuTensor::zeros(&[1]).map_err(|e| crate::TransformerError::Implementation(format!("dummy: {e}")))?;
                (d.clone(), d.clone(), d.clone(), d.clone(), Some(wq_f16), Some(wk_f16), Some(wv_f16), Some(wo_f16))
            } else {
                let wq_t = ctx.transpose(&wq).map_err(|e| crate::TransformerError::Implementation(format!("{}wq t: {e}", p)))?;
                let wk_t = ctx.transpose(&wk).map_err(|e| crate::TransformerError::Implementation(format!("{}wk t: {e}", p)))?;
                let wv_t = ctx.transpose(&wv).map_err(|e| crate::TransformerError::Implementation(format!("{}wv t: {e}", p)))?;
                let wo_t = ctx.transpose(&wo).map_err(|e| crate::TransformerError::Implementation(format!("{}wo t: {e}", p)))?;
                (wq_t, wk_t, wv_t, wo_t, None, None, None, None)
            };
            let att_wq_f16 = wq_f16.clone();
            let att_wk_f16 = wk_f16.clone();
            let att_wv_f16 = wv_f16.clone();
            let att_wo_f16 = wo_f16.clone();
            let _ = block.attention.gpu_weights.set(crate::gqa::GqaGpuWeights {
                wq_t: Some(wq_t.clone()), wk_t: Some(wk_t.clone()), wv_t: Some(wv_t.clone()), wo_t: Some(wo_t.clone()),
                wq_f16: att_wq_f16, wk_f16: att_wk_f16, wv_f16: att_wv_f16, wo_f16: att_wo_f16,
                wq_shape, wk_shape, wv_shape, wo_shape,
                is_f16: use_f16,
            });

            let w1 = to_gpu(get_arr(&format!("{}ffn.w1", p))?)?;
            let w2 = to_gpu(get_arr(&format!("{}ffn.w2", p))?)?;
            let w3 = to_gpu(get_arr(&format!("{}ffn.w3", p))?)?;
            let (w1_t, w2_t, w3_t, w1_f16, w2_f16, w3_f16) = if use_f16 {
                let w1_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&w1).map_err(|e| crate::TransformerError::Implementation(format!("{}w1 t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}w1 f16: {e}", p)))?;
                let w2_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&w2).map_err(|e| crate::TransformerError::Implementation(format!("{}w2 t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}w2 f16: {e}", p)))?;
                let w3_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&w3).map_err(|e| crate::TransformerError::Implementation(format!("{}w3 t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}w3 f16: {e}", p)))?;
                let d = GpuTensor::zeros(&[1]).map_err(|e| crate::TransformerError::Implementation(format!("dummy: {e}")))?;
                (d.clone(), d.clone(), d.clone(), Some(w1_f16), Some(w2_f16), Some(w3_f16))
            } else {
                let w1_t = ctx.transpose(&w1).map_err(|e| crate::TransformerError::Implementation(format!("{}w1 t: {e}", p)))?;
                let w2_t = ctx.transpose(&w2).map_err(|e| crate::TransformerError::Implementation(format!("{}w2 t: {e}", p)))?;
                let w3_t = ctx.transpose(&w3).map_err(|e| crate::TransformerError::Implementation(format!("{}w3 t: {e}", p)))?;
                (w1_t, w2_t, w3_t, None, None, None)
            };
            let ffn_w1_f16 = w1_f16.clone();
            let ffn_w2_f16 = w2_f16.clone();
            let ffn_w3_f16 = w3_f16.clone();
            let _ = block.ffn.gpu_weights.set(crate::swiglu::SwigluGpuWeights {
                w1_t: w1_t.clone(), w2_t: w2_t.clone(), w3_t: w3_t.clone(),
                w1_f16: ffn_w1_f16, w2_f16: ffn_w2_f16, w3_f16: ffn_w3_f16,
            });

            block_weights.push(BlockGpuWeights {
                attention_norm_weight: an_gpu,
                ffn_norm_weight: fn_gpu,
                wq_t, wk_t, wv_t, wo_t,
                w1_t, w2_t, w3_t,
                wq_i8: None, wk_i8: None, wv_i8: None, wo_i8: None,
                w1_i8: None, w2_i8: None, w3_i8: None,
                wq_scales: None, wk_scales: None, wv_scales: None, wo_scales: None,
                w1_scales: None, w2_scales: None, w3_scales: None,
                wq_zero_points: None, wk_zero_points: None, wv_zero_points: None, wo_zero_points: None,
                w1_zero_points: None, w2_zero_points: None, w3_zero_points: None,
                wq_f16, wk_f16, wv_f16, wo_f16,
                w1_f16, w2_f16, w3_f16,
            });
        }

        let _ = model.gpu_weights.set(GpuWeights {
            token_embedding: te_gpu,
            lm_head_t,
            lm_head_i8: None,
            lm_head_scales: None,
            lm_head_zero_points: None,
            lm_head_f16: None,
            norm_weight: nw_gpu,
            block_weights,
        });

        Ok(model)
    }

    /// Per-channel asymmetric int8 quantization.
    /// Returns (packed u32 data, per-channel scales, per-channel zero_points)
    /// where:
    ///   - packed: each u32 holds 4 int8 values in little-endian byte order
    ///   - scales[f32; N]: per-output-channel scale factor
    ///   - zero_points[f32; N]: per-output-channel dequant bias (precomputed as -zp_i8 * scale)
    /// Dequant formula in shader: val = q_i8 * scales[col] + zero_points[col]
    #[cfg(feature = "gpu")]
    fn quantize_weight(weight: &Array2<f32>) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let shape = weight.shape();
        let n = shape[0];
        let k = shape[1];

        let mut packed = Vec::with_capacity((n * k + 3) / 4);
        let mut scales = Vec::with_capacity(n);
        let mut zero_points = Vec::with_capacity(n);

        for i in 0..n {
            let row = weight.row(i);
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for &v in row.iter() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }

            let (scale_i, dequant_bias_i) = if max_val == min_val {
                (1.0, 0.0)
            } else {
                let s = (max_val - min_val) / 255.0;
                let inv_s = 1.0 / s;
                // Compute int8 zero_point: q = round(v / s + 128), then convert to signed
                // so that q = zero_point_i8 + round(v / s) and zero_point_i8 maps to v=0 in [-128,127] range
                // Standard: zero_point = round(-min / s), then q = round(v/s + zero_point), clamp to [-128,127]
                let zp_f32 = (-min_val * inv_s).round().clamp(-128.0, 127.0);
                let zp_i8 = zp_f32 as i32 as i8;
                // Precompute dequant bias: -zp_i8 * s
                (s, -(zp_i8 as f32) * s)
            };

            scales.push(scale_i);
            zero_points.push(dequant_bias_i);

            // Pack this row
            let inv_scale_i = 1.0 / scale_i;
            for (j, &v) in row.iter().enumerate() {
                // Asymmetric: q_i8 = round(v / s + zero_point_i8), clamped to [-128, 127]
                let q_f32 = (v * inv_scale_i).round().clamp(-128.0, 127.0);
                let q = q_f32 as i32 as i8 as u8;
                let idx = i * k + j;
                let word_idx = idx / 4;
                let byte_off = idx % 4;
                if byte_off == 0 {
                    if word_idx >= packed.len() {
                        packed.push(0);
                    }
                }
                packed[word_idx] |= (q as u32) << (byte_off * 8);
            }
        }

        (packed, scales, zero_points)
    }

    /// Pre-upload ALL model weights to GPU persistent buffer.
    /// Token embedding, lm_head (pre-transposed), norm weight, and per-block
    /// attention/FFN weights (all pre-transposed) are uploaded once and reused.
    /// Called once at the start of any GPU forward pass.
    ///
    /// If `self.quantize_weights` is true, also uploads int8 quantized variants
    /// of weight matrices. The int8 kernels dequantize on-the-fly in the WGSL
    /// shader, reducing GPU memory bandwidth ~4×.
    #[cfg(feature = "gpu")]
    pub fn preupload_weights_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuError, GpuTensor};

        if self.gpu_weights.get().is_some() {
            return Ok(());
        }

        let ctx = GpuContext::global()?;

        let mk_gpu = |arr: &Array2<f32>| -> Result<GpuTensor, GpuError> {
            let shape = vec![arr.shape()[0], arr.shape()[1]];
            let data = arr
                .as_slice()
                .ok_or_else(|| GpuError::Unsupported("non-contiguous weight".into()))?
                .to_vec();
            Ok(GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(shape, data)
                    .map_err(|e| GpuError::Unsupported(e.to_string()))?,
            )?)
        };

        let mk_gpu_1d = |arr: &Array1<f32>| -> Result<GpuTensor, GpuError> {
            let shape = vec![arr.len()];
            let data = arr.to_vec();
            Ok(GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(shape, data)
                    .map_err(|e| GpuError::Unsupported(e.to_string()))?,
            )?)
        };

        let token_embedding = mk_gpu(self.token_embedding.as_ref().ok_or_else(|| {
            GpuError::Unsupported("token_embedding not available".into())
        })?)?;

        let lm_head_gpu = mk_gpu(self.lm_head.as_ref().ok_or_else(|| {
            GpuError::Unsupported("lm_head not available".into())
        })?)?;
        // F16 mode: f32 transposed weight ga dipake — cukup f16 packed doang.
        // `zeros(&[1])` adalah DUMMY older (4 bytes) buat ngisi struct field,
        // BUKAN matriks ukuran penuh. VRAM hemat: (vocab_size × hidden_size × 4) bytes.
        let lm_head_t = ctx.transpose(&lm_head_gpu)?;

        let (lm_head_i8, lm_head_scales, lm_head_zero_points) = if self.quantize_weights {
            let lh = self.lm_head.as_ref().ok_or_else(|| {
                GpuError::Unsupported("lm_head not available".into())
            })?;
            let shape = vec![lh.shape()[0], lh.shape()[1]];
            let (packed, sc, zp) = Self::quantize_weight(lh);
            let n_out = sc.len();
            let sc_t = GpuTensor::from_cpu(&ArrayD::from_shape_vec(vec![n_out], sc)
                .map_err(|e| GpuError::Unsupported(e.to_string()))?)?;
            let zp_t = GpuTensor::from_cpu(&ArrayD::from_shape_vec(vec![n_out], zp)
                .map_err(|e| GpuError::Unsupported(e.to_string()))?)?;
            (Some(GpuTensor::from_cpu_i8_packed(shape, &packed)?), Some(sc_t), Some(zp_t))
        } else {
            (None, None, None)
        };

        let norm_weight = {
            let w = self.norm.weight.as_ref().ok_or_else(|| {
                GpuError::Unsupported("norm.weight not available for GPU upload".into())
            })?;
            mk_gpu_1d(w)?
        };

        let mut block_weights = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let attention_norm_weight = {
                let w = block.attention_norm.weight.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("attention_norm.weight not available".into())
                })?;
                mk_gpu_1d(w)?
            };
            let ffn_norm_weight = {
                let w = block.ffn_norm.weight.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("ffn_norm.weight not available".into())
                })?;
                mk_gpu_1d(w)?
            };

            let wq = mk_gpu(block.attention.wq.as_ref().ok_or_else(|| {
                GpuError::Unsupported("attention.wq not available".into())
            })?)?;
            let wk = mk_gpu(block.attention.wk.as_ref().ok_or_else(|| {
                GpuError::Unsupported("attention.wk not available".into())
            })?)?;
            let wv = mk_gpu(block.attention.wv.as_ref().ok_or_else(|| {
                GpuError::Unsupported("attention.wv not available".into())
            })?)?;
            let wo = mk_gpu(block.attention.wo.as_ref().ok_or_else(|| {
                GpuError::Unsupported("attention.wo not available".into())
            })?)?;
            // MoE blocks: upload first expert's weights sebagai pengganti dense FFN weights
            let ffn_err = |name: &str| GpuError::Unsupported(format!("{name} not available (MoE expert used instead)"));
            let (ffn_w1, ffn_w2, ffn_w3): (&Array2<f32>, &Array2<f32>, &Array2<f32>) =
                if let Some(experts) = &block.experts {
                    let e = &experts[0];
                    (e.w1.as_ref().ok_or_else(|| ffn_err("expert[0].w1"))?,
                     e.w2.as_ref().ok_or_else(|| ffn_err("expert[0].w2"))?,
                     e.w3.as_ref().ok_or_else(|| ffn_err("expert[0].w3"))?)
                } else {
                    (block.ffn.w1.as_ref().ok_or_else(|| ffn_err("ffn.w1"))?,
                     block.ffn.w2.as_ref().ok_or_else(|| ffn_err("ffn.w2"))?,
                     block.ffn.w3.as_ref().ok_or_else(|| ffn_err("ffn.w3"))?)
                };

            let w1 = mk_gpu(ffn_w1)?;
            let w2 = mk_gpu(ffn_w2)?;
            let w3 = mk_gpu(ffn_w3)?;

            let mk_scale_zp = |arr: &Array2<f32>| -> Result<_, GpuError> {
                let (p, sc, zp) = Self::quantize_weight(arr);
                let shape = vec![arr.shape()[0], arr.shape()[1]];
                let n_out = sc.len();
                let sc_t = GpuTensor::from_cpu(&ArrayD::from_shape_vec(vec![n_out], sc)
                    .map_err(|e| GpuError::Unsupported(e.to_string()))?)?;
                let zp_t = GpuTensor::from_cpu(&ArrayD::from_shape_vec(vec![n_out], zp)
                    .map_err(|e| GpuError::Unsupported(e.to_string()))?)?;
                let i8_t = GpuTensor::from_cpu_i8_packed(shape, &p)?;
                Ok((i8_t, sc_t, zp_t))
            };

            let (wq_i8, wq_scales, wq_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(block.attention.wq.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("attention.wq not available".into())
                })?)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (wk_i8, wk_scales, wk_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(block.attention.wk.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("attention.wk not available".into())
                })?)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (wv_i8, wv_scales, wv_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(block.attention.wv.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("attention.wv not available".into())
                })?)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (wo_i8, wo_scales, wo_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(block.attention.wo.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("attention.wo not available".into())
                })?)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (w1_i8, w1_scales, w1_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(ffn_w1)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (w2_i8, w2_scales, w2_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(ffn_w2)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (w3_i8, w3_scales, w3_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(ffn_w3)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };

            let use_f16 = self.use_half_precision && !self.quantize_weights;
            // Fix: transpose dulu, BARU pack f16 — matmul WGSL expect transposed weights.
            // Ini benerin bug: dulu f16 di-pack tanpa transpose, hasil matmul salah.
            let wq_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wq)?)?)
            } else {
                None
            };
            let wk_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wk)?)?)
            } else {
                None
            };
            let wv_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wv)?)?)
            } else {
                None
            };
            let wo_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wo)?)?)
            } else {
                None
            };
            let w1_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w1)?)?)
            } else {
                None
            };
            let w2_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w2)?)?)
            } else {
                None
            };
            let w3_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w3)?)?)
            } else {
                None
            };

            // F16 mode: f32 transposed weights (wq_t..w3_t) ga dipake —
            // semua matmul pake f16→f32 upconvert temp per-forward.
            // `zeros(&[1])` DUMMY older (4 bytes per weight, total ~28 bytes),
            // BUKAN 7 matriks ukuran penuh. VRAM hemat ~50% per block.
            let (wq_t, wk_t, wv_t, wo_t, w1_t, w2_t, w3_t) = if use_f16 {
                let d = GpuTensor::zeros(&[1])?;
                (d.clone(), d.clone(), d.clone(), d.clone(), d.clone(), d.clone(), d)
            } else {
                (
                    ctx.transpose(&wq)?,
                    ctx.transpose(&wk)?,
                    ctx.transpose(&wv)?,
                    ctx.transpose(&wo)?,
                    ctx.transpose(&w1)?,
                    ctx.transpose(&w2)?,
                    ctx.transpose(&w3)?,
                )
            };

            block_weights.push(BlockGpuWeights {
                attention_norm_weight,
                ffn_norm_weight,
                wq_t,
                wk_t,
                wv_t,
                wo_t,
                w1_t,
                w2_t,
                w3_t,
                wq_i8,
                wk_i8,
                wv_i8,
                wo_i8,
                w1_i8,
                w2_i8,
                w3_i8,
                wq_scales,
                wk_scales,
                wv_scales,
                wo_scales,
                w1_scales,
                w2_scales,
                w3_scales,
                wq_zero_points,
                wk_zero_points,
                wv_zero_points,
                wo_zero_points,
                w1_zero_points,
                w2_zero_points,
                w3_zero_points,
                wq_f16,
                wk_f16,
                wv_f16,
                wo_f16,
                w1_f16,
                w2_f16,
                w3_f16,
            });
        }

        let lm_head_f16 = if self.use_half_precision && !self.quantize_weights {
            Some(ctx.f32_to_f16_packed(&ctx.transpose(&lm_head_gpu)?)?)
        } else {
            None
        };

        self.gpu_weights
            .set(GpuWeights {
                token_embedding,
                lm_head_t,
                lm_head_i8,
                lm_head_scales,
                lm_head_zero_points,
                lm_head_f16,
                norm_weight,
                block_weights,
            })
            .map_err(|_| {
                nexora_autograd::gpu::GpuError::Unsupported("weights already set".into())
            })?;

        Ok(())
    }

    /// Readback ALL weights from CPU (or GPU if CPU weights dropped).
    /// Returns (name, ArrayD) pairs for safetensors save or training init.
    pub fn readback_weights(&self) -> crate::TransformerResult<Vec<(String, ndarray::ArrayD<f32>)>> {
        use ndarray::ArrayD;
        let mut tensors: Vec<(String, ArrayD<f32>)> =
            Vec::with_capacity(3 + 9 * self.blocks.len() + 3 * self.config.num_experts * self.blocks.len());

        // token_embedding
        if let Some(te) = &self.token_embedding {
            tensors.push(("token_embedding".to_owned(), te.clone().into_dyn()));
        } else {
            #[cfg(feature = "gpu")]
            {
                let gw = self.gpu_weights.get().ok_or_else(||
                    crate::TransformerError::Implementation("token_embedding unavailable — no CPU or GPU copy".into()))?;
                let cpu = gw.token_embedding.to_cpu().map_err(|e|
                    crate::TransformerError::Implementation(format!("token_embedding GPU readback: {e}")))?;
                tensors.push(("token_embedding".to_owned(), cpu));
            }
            #[cfg(not(feature = "gpu"))]
            return Err(crate::TransformerError::Implementation(
                "token_embedding unavailable — no CPU copy and GPU feature disabled".into()));
        }

        // lm_head
        if let Some(lh) = &self.lm_head {
            tensors.push(("lm_head".to_owned(), lh.clone().into_dyn()));
        } else {
            #[cfg(feature = "gpu")]
            {
                let gw = self.gpu_weights.get().ok_or_else(||
                    crate::TransformerError::Implementation("lm_head unavailable — no CPU or GPU copy".into()))?;
                // lm_head_t is [hidden, vocab] transposed; re-transpose to get [vocab, hidden]
                let cpu = gw.lm_head_t.to_cpu().map_err(|e|
                    crate::TransformerError::Implementation(format!("lm_head GPU readback: {e}")))?;
                let cpu_2d = cpu.into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| crate::TransformerError::Implementation(format!("lm_head shape: {e}")))?;
                let re_t = cpu_2d.reversed_axes();
                tensors.push(("lm_head".to_owned(), re_t.into_dyn()));
            }
            #[cfg(not(feature = "gpu"))]
            return Err(crate::TransformerError::Implementation(
                "lm_head unavailable — no CPU copy and GPU feature disabled".into()));
        }

        // norm.weight (1D)
        if let Some(nw) = &self.norm.weight {
            tensors.push(("norm.weight".to_owned(), nw.clone().into_dyn()));
        } else {
            let cpu = self.norm.readback_weight().map_err(|e|
                crate::TransformerError::Implementation(format!("norm.weight readback failed: {e}")))?;
            tensors.push(("norm.weight".to_owned(), cpu.into_dyn()));
        }

        for (i, block) in self.blocks.iter().enumerate() {
            let p = format!("blocks.{}.", i);

            // attention_norm.weight
            if let Some(w) = &block.attention_norm.weight {
                tensors.push((format!("{}attention_norm.weight", p), w.clone().into_dyn()));
            } else {
                let cpu = block.attention_norm.readback_weight().map_err(|e|
                    crate::TransformerError::Implementation(format!("{}attention_norm.weight readback: {e}", p)))?;
                tensors.push((format!("{}attention_norm.weight", p), cpu.into_dyn()));
            }

            // ffn_norm.weight
            if let Some(w) = &block.ffn_norm.weight {
                tensors.push((format!("{}ffn_norm.weight", p), w.clone().into_dyn()));
            } else {
                let cpu = block.ffn_norm.readback_weight().map_err(|e|
                    crate::TransformerError::Implementation(format!("{}ffn_norm.weight readback: {e}", p)))?;
                tensors.push((format!("{}ffn_norm.weight", p), cpu.into_dyn()));
            }

            // GQA weights — readback from GPU if no CPU copy, else use CPU weights
            if let Some(w) = &block.attention.wq {
                tensors.push((format!("{}attention.wq", p), w.clone().into_dyn()));
                if let Some(wk) = &block.attention.wk {
                    tensors.push((format!("{}attention.wk", p), wk.clone().into_dyn()));
                }
                if let Some(wv) = &block.attention.wv {
                    tensors.push((format!("{}attention.wv", p), wv.clone().into_dyn()));
                }
                if let Some(wo) = &block.attention.wo {
                    tensors.push((format!("{}attention.wo", p), wo.clone().into_dyn()));
                }
            } else {
                let g = block.attention.readback_weights().map_err(|e|
                    crate::TransformerError::Implementation(format!("{}attention readback: {:?}", p, e)))?;
                tensors.push((format!("{}attention.wq", p), g.0.into_dyn()));
                tensors.push((format!("{}attention.wk", p), g.1.into_dyn()));
                tensors.push((format!("{}attention.wv", p), g.2.into_dyn()));
                tensors.push((format!("{}attention.wo", p), g.3.into_dyn()));
            }

            // SwiGLU weights — same pattern
            if let Some(w) = &block.ffn.w1 {
                tensors.push((format!("{}ffn.w1", p), w.clone().into_dyn()));
                if let Some(w2) = &block.ffn.w2 {
                    tensors.push((format!("{}ffn.w2", p), w2.clone().into_dyn()));
                }
                if let Some(w3) = &block.ffn.w3 {
                    tensors.push((format!("{}ffn.w3", p), w3.clone().into_dyn()));
                }
            } else {
                let g = block.ffn.readback_weights().map_err(|e|
                    crate::TransformerError::Implementation(format!("{}ffn readback: {:?}", p, e)))?;
                tensors.push((format!("{}ffn.w1", p), g.0.into_dyn()));
                tensors.push((format!("{}ffn.w2", p), g.1.into_dyn()));
                tensors.push((format!("{}ffn.w3", p), g.2.into_dyn()));
            }

            // Expert weights
            if let Some(experts) = &block.experts {
                for (e_idx, expert) in experts.iter().enumerate() {
                    let e_prefix = format!("{}experts.{}.", p, e_idx);
                    if let Some(w) = &expert.w1 {
                        tensors.push((format!("{}w1", e_prefix), w.clone().into_dyn()));
                    }
                    if let Some(w2) = &expert.w2 {
                        tensors.push((format!("{}w2", e_prefix), w2.clone().into_dyn()));
                    }
                    if let Some(w3) = &expert.w3 {
                        tensors.push((format!("{}w3", e_prefix), w3.clone().into_dyn()));
                    }
                }
            }
        }

        Ok(tensors)
    }

    /// Save checkpoint by reading back from GPU (or using resident CPU weights).
    pub fn save_checkpoint(&self, path: &str) -> crate::TransformerResult<()> {
        let tensors = self.readback_weights()?;
        let refs: Vec<(&str, ndarray::ArrayD<f32>)> = tensors
            .iter()
            .map(|(name, arr)| (name.as_str(), arr.clone()))
            .collect();
        let mut meta = std::collections::HashMap::new();
        meta.insert(
            "quantization".to_string(),
            self.config.quantization.dtype_name().to_string(),
        );
        crate::safetensors::save_safetensors_with_meta(
            path,
            &refs,
            crate::safetensors::SaveDtype::F32,
            Some(meta),
        )
    }

}
