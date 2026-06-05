

use super::gpu_tensor::{GpuDtype, GpuTensor};
use super::gpu_types::*;
use super::wgsl::*;
#[cfg(feature = "cuda")]
use super::cuda::CudaTensor;
/// Maximum workgroups per dimension for wgpu/WebGPU (65535).
/// Any dispatch exceeding this must be chunked across multiple dispatches
/// using buffer-slice bindings with per-chunk offsets.
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Default workgroup size used by most Nexora WGSL shaders.
const DEFAULT_WORKGROUP_SIZE: u32 = 256;

impl GpuContext {
    // ── Private helpers ──

    fn compile_shader(
        &mut self,
        name: &'static str,
        bindings: &[wgpu::BindGroupLayoutEntry],
        wgsl: &'static str,
        entry: &'static str,
    ) -> Result<(), GpuError> {
        self.compile_pipeline(name, bindings, std::borrow::Cow::Borrowed(wgsl), entry)
    }

    /// Create a uniform buffer and write u32 config data (common pattern).
    fn uniform_cfg_u32(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
        let bytes = bytemuck::cast_slice(data);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytes);
        buf
    }

    // ====================================================================
    // SECTION: Gradient & MoE
    // ====================================================================

    pub fn gradient_allreduce(
        &self,
        grad_buffers: &GpuTensor,
        out: &GpuTensor,
        num_replicas: u32,
    ) -> Result<(), GpuError> {
        let numel = out.numel() as u32;
        let pipeline = self
            .pipelines
            .get("gradient_allreduce")
            .ok_or_else(|| GpuError::Pipeline("gradient_allreduce not compiled".into()))?;

        let cfg_buf = self.uniform_cfg_u32("gradient_allreduce_cfg", &[numel, num_replicas, 0, 0]);

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, grad_buffers.buffer()), (1, out.buffer())],
            &[(2, &cfg_buf)],
            numel, 256, 4,
        );
        Ok(())
    }

    /// MoE scatter-add: scatters weighted expert outputs into an accumulated output buffer.
    /// Processes one expert group — the kernel dispatches sequentially (no atomicAdd needed).
    /// - `expert_out`: [n_tokens, hidden_size] expert output
    /// - `indices`: [n_tokens] u32 — which batch row each token maps to
    /// - `weights`: [n_tokens] f32 — routing weight for each token
    /// - `output`: [batch_size, hidden_size] accumulated output (in-place read-write)
    pub fn moe_scatter_add(
        &self,
        expert_out: &GpuTensor,
        indices: &GpuTensor,
        weights: &GpuTensor,
        output: &GpuTensor,
    ) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("moe_scatter_add")
            .ok_or_else(|| GpuError::Pipeline("moe_scatter_add not compiled".into()))?;
        let hidden_size = output.shape()[1] as u32;
        let n_tokens = expert_out.shape()[0] as u32;
        let total = n_tokens * hidden_size;

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("moe_scatter_add_cfg"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 2] = [hidden_size, n_tokens];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        self.dispatch_1d_chunked(
            pipeline,
            &[
                (0, expert_out.buffer()),
                (1, indices.buffer()),
                (2, weights.buffer()),
                (3, output.buffer()),
            ],
            &[(4, &cfg_buf)],
            total, 256, 4,
        );
        Ok(())
    }

    // ====================================================================
    // SECTION: Shader Compilation (sampling/norm pipelines)
    // ====================================================================

    pub(crate) fn compile_causal_softmax(&mut self) -> Result<(), GpuError> {
        self.compile_shader("causal_softmax", &[storage_binding(0, true), storage_binding(1, false), uniform_binding(2)], CAUSAL_SOFTMAX_WGSL, "causal_softmax_main")
    }

    pub(crate) fn compile_l2_norm(&mut self) -> Result<(), GpuError> {
        self.compile_shader("l2_norm", &[storage_binding(0, true), storage_binding(1, false), uniform_binding(2)], L2_NORM_WGSL, "l2_norm_main")
    }

    pub(crate) fn compile_gradient_clip(&mut self) -> Result<(), GpuError> {
        self.compile_shader("gradient_clip", &[storage_binding(0, false), storage_binding(1, true), storage_binding(2, false), uniform_binding(3)], GRADIENT_CLIP_WGSL, "main")
    }

    pub(crate) fn compile_temperature_scale(&mut self) -> Result<(), GpuError> {
        self.compile_shader("temperature_scale", &[storage_binding(0, false), uniform_binding(1)], TEMPERATURE_SCALE_WGSL, "temperature_scale_main")
    }

    pub(crate) fn compile_top_k_mask(&mut self) -> Result<(), GpuError> {
        self.compile_shader("top_k_mask", &[storage_binding(0, false), uniform_binding(1)], TOP_K_MASK_WGSL, "top_k_mask_main")
    }

    pub(crate) fn compile_top_p_mask(&mut self) -> Result<(), GpuError> {
        self.compile_shader("top_p_mask", &[storage_binding(0, false), uniform_binding(1)], TOP_P_MASK_WGSL, "top_p_mask_main")
    }

    pub(crate) fn compile_multinomial_sample(&mut self) -> Result<(), GpuError> {
        self.compile_shader("multinomial_sample", &[storage_binding(0, true), storage_binding(1, false), uniform_binding(2)], MULTINOMIAL_SAMPLE_WGSL, "multinomial_main")
    }

    pub(crate) fn compile_dropout_mask(&mut self) -> Result<(), GpuError> {
        self.compile_shader("dropout_mask", &[storage_binding(0, false), uniform_binding(1)], DROPOUT_MASK_WGSL, "dropout_mask_main")
    }

    // ====================================================================
    // SECTION: Buffer Operations (fill, scale, norm, sampling)
    // ====================================================================

    /// Generate dropout mask directly on GPU — avoids CPU random + upload.
    /// Fills `mask` buffer with 0.0 or `scale` based on random samples.
    pub fn generate_dropout_mask(
        &self,
        mask: &GpuTensor,
        rate: f32,
        scale: f32,
        seed: u32,
    ) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("dropout_mask")
            .ok_or_else(|| GpuError::Pipeline("dropout_mask not compiled".into()))?;
        let numel = u32::try_from(mask.numel()).unwrap_or(u32::MAX);
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dropout_mask_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [
            numel,
            f32::to_bits(rate),
            f32::to_bits(scale),
            seed,
        ];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));
        self.dispatch_1d_chunked(pipeline, &[(0, mask.buffer())], &[(1, &cfg_buf)], numel, 256, 4);
        Ok(())
    }

    /// Zero out a GPU tensor buffer in-place.
    pub fn fill_zero(&self, t: &GpuTensor) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("fill_zero")
            .ok_or_else(|| GpuError::Pipeline("fill_zero not compiled".into()))?;
        let numel = u32::try_from(t.numel()).unwrap_or(u32::MAX);
        self.dispatch_1d_chunked(pipeline, &[(0, t.buffer())], &[], numel, 256, 4);
        Ok(())
    }

    /// Zero out a GPU buffer using u32 writes — works for any dtype.
    /// Dispatches ceil(byte_size / 4 / 256) workgroups, each writes 0u per u32.
    pub fn fill_zero_u32(&self, t: &GpuTensor) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("fill_zero_u32")
            .ok_or_else(|| GpuError::Pipeline("fill_zero_u32 not compiled".into()))?;
        let buf_size = t.buffer().size();
        let u32_count = u32::try_from(buf_size / 4).unwrap_or(u32::MAX);
        self.dispatch_1d_chunked(pipeline, &[(0, t.buffer())], &[], u32_count, 256, 4);
        Ok(())
    }

    /// Fill a GPU tensor buffer with a constant value in-place.
    pub fn fill_constant(&self, t: &GpuTensor, value: f32) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("fill_constant")
            .ok_or_else(|| GpuError::Pipeline("fill_constant not compiled".into()))?;
        let numel = u32::try_from(t.numel()).unwrap_or(u32::MAX);
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fill_const_cfg"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 2] = [numel, f32::to_bits(value)];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));
        self.dispatch_1d_chunked(pipeline, &[(0, t.buffer())], &[(1, &cfg_buf)], numel, 256, 4);
        Ok(())
    }

    /// Scale a GPU tensor buffer in-place (multiply each element by `scale`).
    pub fn scale_inplace(&self, t: &GpuTensor, scale: f32) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("scale_inplace")
            .ok_or_else(|| GpuError::Pipeline("scale_inplace not compiled".into()))?;
        let numel = u32::try_from(t.numel()).unwrap_or(u32::MAX);
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scale_cfg"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 2] = [numel, f32::to_bits(scale)];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));
        self.dispatch_1d_chunked(pipeline, &[(0, t.buffer())], &[(1, &cfg_buf)], numel, 256, 4);
        Ok(())
    }

    /// Compute L2 norm (sum of squares) for a GPU tensor. Returns scalar tensor.
    pub fn l2_norm(&self, t: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let pipeline = self
            .pipelines
            .get("l2_norm")
            .ok_or_else(|| GpuError::Pipeline("l2_norm not compiled".into()))?;
        let numel = u32::try_from(t.numel()).unwrap_or(u32::MAX);
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("l2_norm_out"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("l2_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [numel, 0, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l2_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: t.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch(pipeline, &bg, (1, 1, 1));
        Ok(GpuTensor {
            shape: vec![1],
            buffer: out,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// Causal softmax: softmax over last dim with causal mask.
    pub fn causal_softmax(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let shape = input.shape();
        let batch = u32::try_from(shape[0]).unwrap_or(u32::MAX);
        let dim = u32::try_from(shape[1]).unwrap_or(u32::MAX);
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("causal_softmax_out"),
            size: (input.numel() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cs_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [batch, dim, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));
        let pipeline = self
            .pipelines
            .get("causal_softmax")
            .ok_or_else(|| GpuError::Pipeline("causal_softmax not compiled".into()))?;
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cs_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch(pipeline, &bg, (batch, 1, 1));
        Ok(GpuTensor {
            shape: shape.to_vec(),
            buffer: out,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ====================================================================
    // SECTION: GPU Sampling (temperature, top-k, top-p, multinomial)
    // ====================================================================

    /// GPU sampling: temperature → softmax → top-k → top-p → multinomial
    pub fn gpu_sample(
        &self,
        logits: &GpuTensor,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
    ) -> Result<GpuTensor, GpuError> {
        let shape = logits.shape();
        let batch = shape[0];
        let vocab = u32::try_from(shape[1]).unwrap_or(u32::MAX);
        let numel = logits.numel();

        // Working copy of logits (pool-allocated)
        let work_buf = {
            let pooled = self.alloc_buffer(
                (numel * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            self.with_encoder(|enc| {
                enc.copy_buffer_to_buffer(
                    logits.buffer(),
                    0,
                    &pooled.buffer,
                    0,
                    (numel * 4) as u64,
                );
            });
            pooled.buffer
        };

        // Step 1: temperature scale (in-place on work_buf)
        if (temperature - 1.0).abs() > 1e-6 && temperature > 0.0 {
            let pipeline = self
                .pipelines
                .get("temperature_scale")
                .ok_or_else(|| GpuError::Pipeline("temperature_scale not compiled".into()))?;
            let temp_buf = self.alloc_buffer(
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let temp_cfg: [u32; 4] = [
                f32::to_bits(temperature),
                u32::try_from(numel).unwrap_or(u32::MAX),
                0,
                0,
            ];
            self.queue
                .write_buffer(&temp_buf.buffer, 0, bytemuck::cast_slice(&temp_cfg));
            let workgroups = u32::try_from(numel).unwrap_or(u32::MAX);
            self.dispatch_1d_chunked(
                pipeline,
                &[(0, &work_buf)],
                &[(1, &temp_buf.buffer)],
                workgroups, 256, 4,
            );
        }

        // Step 2: softmax (via ctx.softmax — uses reusable encoder internally)
        let logits_gpu = GpuTensor {
            shape: shape.clone(),
            buffer: work_buf,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let probs = self.softmax(&logits_gpu)?;

        // Copy probs → fresh work buffer for top-k mask (avoids mutating softmax output)
        let masked_buf = {
            let pooled = self.alloc_buffer(
                (numel * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            self.with_encoder(|enc| {
                enc.copy_buffer_to_buffer(probs.buffer(), 0, &pooled.buffer, 0, (numel * 4) as u64);
            });
            pooled.buffer
        };

        // Step 3: top-k mask (in-place on masked_buf)
        let masked = if top_k > 0 && top_k < vocab {
            let pipeline = self
                .pipelines
                .get("top_k_mask")
                .ok_or_else(|| GpuError::Pipeline("top_k_mask not compiled".into()))?;
            let cfg_buf = self.alloc_buffer(
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let cfg: [u32; 4] = [vocab, top_k, 0, 0];
            self.queue
                .write_buffer(&cfg_buf.buffer, 0, bytemuck::cast_slice(&cfg));
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("topk_bg"),
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: masked_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cfg_buf.buffer.as_entire_binding(),
                    },
                ],
            });
            self.dispatch(pipeline, &bg, (batch as u32, 1, 1));
            GpuTensor {
                shape: shape.clone(),
                buffer: masked_buf,
                dtype: GpuDtype::F32,
                device_id: 0,
            }
        } else {
            GpuTensor {
                shape: shape.clone(),
                buffer: masked_buf,
                dtype: GpuDtype::F32,
                device_id: 0,
            }
        };

        // Step 3b: top-p mask (in-place on masked buffer, after top-k)
        if top_p > 0.0 && top_p < 1.0 {
            let pipeline = self
                .pipelines
                .get("top_p_mask")
                .ok_or_else(|| GpuError::Pipeline("top_p_mask not compiled".into()))?;
            let cfg_buf = self.alloc_buffer(
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let cfg: [u32; 4] = [vocab, f32::to_bits(top_p), 0, 0];
            self.queue
                .write_buffer(&cfg_buf.buffer, 0, bytemuck::cast_slice(&cfg));
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("topp_bg"),
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: masked.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cfg_buf.buffer.as_entire_binding(),
                    },
                ],
            });
            self.dispatch(pipeline, &bg, (batch as u32, 1, 1));
        }

        // Step 4: multinomial sample
        let pipeline = self
            .pipelines
            .get("multinomial_sample")
            .ok_or_else(|| GpuError::Pipeline("multinomial_sample not compiled".into()))?;
        let out = self.alloc_buffer(
            (batch * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let cfg_buf = self.alloc_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let cfg: [u32; 4] = [vocab, seed as u32, (seed >> 32) as u32, 0];
        self.queue
            .write_buffer(&cfg_buf.buffer, 0, bytemuck::cast_slice(&cfg));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sample_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: masked.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cfg_buf.buffer.as_entire_binding(),
                },
            ],
        });
        self.dispatch(pipeline, &bg, (batch as u32, 1, 1));

        Ok(GpuTensor {
            shape: vec![batch],
            buffer: out.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// GPU sampling from F16 input: converts F16→F32 on GPU, then runs standard
    /// f32 sampling pipeline. Avoids CPU-side F16→F32 conversion and re-upload.
    pub fn gpu_sample_f16(
        &self,
        logits_f16: &GpuTensor,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
    ) -> Result<GpuTensor, GpuError> {
        let logits_f32 = self.f16_packed_to_f32(logits_f16)?;
        self.gpu_sample(&logits_f32, temperature, top_k, top_p, seed)
    }

    /// Batched GPU sampling: stack B logit vectors into [B, vocab] and sample
    /// once on GPU via F16→F32 conversion. Returns one token ID per sequence.
    /// All computation stays on GPU — only token IDs are read back.
    pub fn gpu_sample_batched_f16(
        &self,
        batch_logits: &[&GpuTensor],
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
    ) -> Result<GpuTensor, GpuError> {
        if batch_logits.is_empty() {
            return Ok(GpuTensor {
                shape: vec![0],
                buffer: self.alloc_buffer(0, wgpu::BufferUsages::COPY_SRC).buffer,
                dtype: GpuDtype::F32,
                device_id: 0,
            });
        }
        // Convert each F16 tensor to F32 on GPU
        let f32_tensors: Vec<GpuTensor> = batch_logits
            .iter()
            .map(|t| self.f16_packed_to_f32(t))
            .collect::<Result<Vec<_>, _>>()?;

        let batch = f32_tensors.len();
        let vocab = f32_tensors[0].numel();
        let total_elems = batch * vocab;

        // Allocate contiguous [B, vocab] F32 buffer
        let flat_buf = self.alloc_buffer(
            (total_elems * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        self.with_encoder(|enc| {
            let mut offset: u64 = 0;
            for t in &f32_tensors {
                enc.copy_buffer_to_buffer(
                    t.buffer(),
                    0,
                    &flat_buf.buffer,
                    offset,
                    (vocab * 4) as u64,
                );
                offset += (vocab * 4) as u64;
            }
        });

        let stacked = GpuTensor {
            shape: vec![batch, vocab],
            buffer: flat_buf.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        self.gpu_sample(&stacked, temperature, top_k, top_p, seed)
    }

    // ── Singleton access ──────────────────────────────────────────────────────


    pub fn matmul_dynamic(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
    ) -> Result<(GpuTensor, std::time::Duration), GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::ShapeMismatch("Matmul requires 2D tensors".into()));
        }

        let max_dim = a_shape[0].max(a_shape[1]).max(b_shape[0]).max(b_shape[1]);

        // Use simple matmul for small matrices (fusion overhead dominates)
        if max_dim < 512 {
            self.matmul(a, b)
                .map(|tensor| (tensor, std::time::Duration::from_secs(0)))
        } else {
            // Use matmul for large matrices (fusion beneficial)
            self.matmul(a, b)
                .map(|tensor| (tensor, std::time::Duration::from_secs(0)))
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.1: TILED MATMUL
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_matmul_tiled(&mut self, tile: u32) -> Result<(), GpuError> {
        if self.pipelines.contains_key("matmul_tiled") {
            return Ok(());
        }
        let wgsl =
            std::borrow::Cow::Owned(MATMUL_TILED_WGSL.replace("{{TILE_SIZE}}", &tile.to_string()));
        self.compile_pipeline(
            "matmul_tiled",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            wgsl,
            "matmul_tiled_main",
        )
    }
    pub(crate) fn compile_rotary_embedding(&mut self) -> Result<(), GpuError> {
        self.compile_shader("rotary_embedding", &[storage_binding(0, false), storage_binding(1, true), storage_binding(2, true), uniform_binding(3)], ROTARY_EMBEDDING_WGSL, "rotary_main")
    }

    pub(crate) fn compile_adam_step(&mut self) -> Result<(), GpuError> {
        self.compile_shader("adam_step", &[storage_binding(0, false), storage_binding(1, true), storage_binding(2, false), storage_binding(3, false), uniform_binding(4)], ADAM_WGSL, "main")
    }

    pub(crate) fn compile_repeat_heads(&mut self) -> Result<(), GpuError> {
        self.compile_shader("repeat_heads", &[storage_binding(0, true), storage_binding(1, false), uniform_binding(2)], REPEAT_HEADS_WGSL, "repeat_heads_main")
    }

    pub fn matmul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }
        if a_shape[1] != b_shape[0] {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }

        // Auto-dispatch to F16 matmul kernel when weight tensor is packed F16.
        // Activation (a) stays f32 — no upconversion needed.
        if a.dtype == GpuDtype::F32 && b.dtype == GpuDtype::F16 {
            return self.matmul_f16(a, b);
        }
        // Both f32: standard path
        if a.dtype != GpuDtype::F32 || b.dtype != GpuDtype::F32 {
            return Err(GpuError::Unsupported(format!(
                "matmul: unsupported dtypes a={:?} b={:?}",
                a.dtype, b.dtype
            )));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        let m_u32 = u32::try_from(m)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let k_u32 = u32::try_from(k)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let n_u32 = u32::try_from(n)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let tile = self.caps.adaptive_tile_size(m, n, k);
        let tile_u32 = u32::try_from(tile)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let c_size = (m_u32 as u64) * (n_u32 as u64) * 4;
        let limit = self.caps.max_storage_buffer_binding_size;
        if c_size > limit {
            return Err(GpuError::Buffer(format!(
                "matmul output {m}×{n} = {c_size} B exceeds device limit {limit} B"
            )));
        }
        let c_buffer = self.alloc_or_create_buffer(
            c_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let dims_buf = self.alloc_or_create_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let dims_data: [u32; 4] = [m_u32, k_u32, n_u32, tile_u32];
        self.queue
            .write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims_data));

        let pipeline = self
            .pipelines
            .get("matmul_tiled")
            .ok_or_else(|| GpuError::Pipeline("matmul_tiled not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_tiled_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let wgx = m_u32.div_ceil(tile_u32);
        let wgy = n_u32.div_ceil(tile_u32);
        self.dispatch(pipeline, &bind_group, (wgx, wgy, 1));

        Ok(GpuTensor {
            shape: vec![a_shape[0], b_shape[1]],
            buffer: c_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.1b: F16 PACKED WEIGHT MATMUL
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_matmul_f16_tiled(&mut self, tile: u32) -> Result<(), GpuError> {
        if self.pipelines.contains_key("matmul_f16_tiled") {
            return Ok(());
        }
        let wgsl = std::borrow::Cow::Owned(
            MATMUL_F16_TILED_WGSL.replace("{{TILE_SIZE}}", &tile.to_string()),
        );
        self.compile_pipeline(
            "matmul_f16_tiled",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            wgsl,
            "matmul_f16_main",
        )
    }

    /// F16 packed weight matmul: activation (f32) × weight (packed F16).
    /// Weight is stored as packed F16 (2 f16 per u32), halving VRAM reads.
    /// Accumulation stays in f32 for precision.
    /// Output is f32 with shape [a.shape[0], b_packed.shape[1]].
    pub fn matmul_f16(&self, a: &GpuTensor, b_packed: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let a_shape = a.shape();
        let b_shape = b_packed.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }
        // b is packed F16: [K, N] in f32 terms, but stored as u32 with 2 f16 per u32
        // a_flat = M*K f32 elements, b_flat = K*N/2 u32 elements
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        if b_packed.dtype != GpuDtype::F16 {
            return Err(GpuError::Unsupported(
                "matmul_f16: b_packed must have dtype F16".into(),
            ));
        }
        let m_u32 = u32::try_from(m)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let k_u32 = u32::try_from(k)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let n_u32 = u32::try_from(n)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let tile = self.caps.adaptive_tile_size(m, n, k);
        let tile_u32 = u32::try_from(tile)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let c_size = (m_u32 as u64) * (n_u32 as u64) * 4;
        let limit = self.caps.max_storage_buffer_binding_size;
        if c_size > limit {
            return Err(GpuError::Buffer(format!(
                "matmul_f16 output {m}×{n} = {c_size} B exceeds device limit {limit} B"
            )));
        }
        let c_buffer = self.alloc_or_create_buffer(
            c_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let dims_buf = self.alloc_or_create_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let dims_data: [u32; 4] = [m_u32, k_u32, n_u32, tile_u32];
        self.queue
            .write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims_data));

        let pipeline = self
            .pipelines
            .get("matmul_f16_tiled")
            .ok_or_else(|| GpuError::Pipeline("matmul_f16_tiled not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_f16_tiled_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_packed.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let wgx = m_u32.div_ceil(tile_u32);
        let wgy = n_u32.div_ceil(tile_u32);
        self.dispatch(pipeline, &bind_group, (wgx, wgy, 1));

        Ok(GpuTensor {
            shape: vec![m, n],
            buffer: c_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.2: INT8 QUANTIZED MATMUL
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_matmul_int8_tiled(&mut self, tile: u32) -> Result<(), GpuError> {
        if self.pipelines.contains_key("matmul_int8_tiled") {
            return Ok(());
        }
        let wgsl = std::borrow::Cow::Owned(
            MATMUL_INT8_TILED_WGSL.replace("{{TILE_SIZE}}", &tile.to_string()),
        );
        self.compile_pipeline(
            "matmul_int8_tiled",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            wgsl,
            "matmul_int8_main",
        )
    }

    /// Quantized matmul: A (int8 packed) × B (f32) → C (f32).
    ///
    /// `a` must have dtype `GpuDtype::I8`. Internally the int8 values are
    /// unpacked, dequantized with `scale`, and accumulated in f32.
    /// `b` must be f32.
    pub fn matmul_int8(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        scale: f32,
    ) -> Result<GpuTensor, GpuError> {
        if a.dtype() != GpuDtype::I8 {
            return Err(GpuError::Dtype(format!(
                "matmul_int8 expects I8 for A, got {:?}",
                a.dtype()
            )));
        }
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }
        if a_shape[1] != b_shape[0] {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        let m_u32 = u32::try_from(m)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let k_u32 = u32::try_from(k)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let n_u32 = u32::try_from(n)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let tile = self.caps.adaptive_tile_size(m, n, k);
        let tile_u32 = u32::try_from(tile)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let c_size = (m_u32 as u64) * (n_u32 as u64) * 4;
        let limit = self.caps.max_storage_buffer_binding_size;
        if c_size > limit {
            return Err(GpuError::Buffer(format!(
                "matmul_int8 output {m}×{n} = {c_size} B exceeds device limit {limit} B"
            )));
        }
        let c_buffer = self.alloc_or_create_buffer(
            c_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let dims_buf = self.alloc_or_create_buffer(
            20,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let dims_data: [u32; 4] = [m_u32, k_u32, n_u32, tile_u32];
        let mut uniform_bytes = Vec::with_capacity(20);
        uniform_bytes.extend_from_slice(bytemuck::cast_slice(&dims_data));
        uniform_bytes.extend_from_slice(bytemuck::bytes_of(&scale));
        self.queue.write_buffer(&dims_buf, 0, &uniform_bytes);

        let pipeline = self
            .pipelines
            .get("matmul_int8_tiled")
            .ok_or_else(|| GpuError::Pipeline("matmul_int8_tiled not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_int8_tiled_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let wgx = m_u32.div_ceil(tile_u32);
        let wgy = n_u32.div_ceil(tile_u32);
        self.dispatch(pipeline, &bind_group, (wgx, wgy, 1));

        Ok(GpuTensor {
            shape: vec![a_shape[0], b_shape[1]],
            buffer: c_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.2b: INT8 WEIGHT MATMUL
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_matmul_int4_weight(&mut self, tile: u32) -> Result<(), GpuError> {
        if self.pipelines.contains_key("matmul_int4_weight") {
            return Ok(());
        }
        let wgsl = std::borrow::Cow::Owned(
            MATMUL_INT4_WEIGHT_WGSL.replace("{{TILE_SIZE}}", &tile.to_string()),
        );
        self.compile_pipeline(
            "matmul_int4_weight",
            &[
                storage_binding(0, true),   // binding 0: A (f32 activations)
                storage_binding(1, true),   // binding 1: B (packed u32 Q4 weights)
                storage_binding(2, false),  // binding 2: C (f32 output, read_write)
                uniform_binding(3),         // binding 3: Uniforms { M, K, N, GroupSize, Tile }
                storage_binding(4, true),   // binding 4: scales (f32 per-group per-column)
            ],
            wgsl,
            "matmul_int4_weight_main",
        )
    }

    pub(crate) fn compile_matmul_int8_weight(&mut self, tile: u32) -> Result<(), GpuError> {
        if self.pipelines.contains_key("matmul_int8_weight") {
            return Ok(());
        }
        let wgsl = std::borrow::Cow::Owned(
            MATMUL_INT8_WEIGHT_WGSL.replace("{{TILE_SIZE}}", &tile.to_string()),
        );
        self.compile_pipeline(
            "matmul_int8_weight",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
                storage_binding(4, true),
                storage_binding(5, true),
            ],
            wgsl,
            "matmul_int8_weight_main",
        )
    }

    /// Quantized matmul: A (f32 activations) × B (packed int8 weights) → C (f32).
    ///
    /// `b` must have dtype `GpuDtype::I8` and shape [N, K] (ORIGINAL orientation —
    /// NOT the conventional transposed layout). The shader reads B[j][k] and
    /// computes C[i][j] = sum_k A[i][k] × dequantize(B[j][k]).
    ///
    /// This is designed for the model inference pattern `activation @ weight^T`
    /// where the original weight matrix W[output_dim, hidden] is stored as int8
    /// and the transpose is handled on-the-fly in the shader.
    pub fn matmul_int8_weight(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        scales: &GpuTensor,
        zero_points: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        if b.dtype() != GpuDtype::I8 {
            return Err(GpuError::Dtype(format!(
                "matmul_int8_weight expects I8 for B (weight), got {:?}",
                b.dtype()
            )));
        }
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }
        // A[M, K] × B[N, K] → Not standard. Check K matches: a_shape[1] == b_shape[1]
        if a_shape[1] != b_shape[1] {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[0];
        let m_u32 = u32::try_from(m)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let k_u32 = u32::try_from(k)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let n_u32 = u32::try_from(n)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let tile = self.caps.adaptive_tile_size(m, n, k);
        let tile_u32 = u32::try_from(tile)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let c_size = (m_u32 as u64) * (n_u32 as u64) * 4;
        let limit = self.caps.max_storage_buffer_binding_size;
        if c_size > limit {
            return Err(GpuError::Buffer(format!(
                "matmul_int8_weight output {m}×{n} = {c_size} B exceeds device limit {limit} B"
            )));
        }
        let c_buffer = self.alloc_or_create_buffer(
            c_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let dims_buf = self.alloc_or_create_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let dims_data: [u32; 4] = [m_u32, k_u32, n_u32, tile_u32];
        self.queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims_data));

        let pipeline = self
            .pipelines
            .get("matmul_int8_weight")
            .ok_or_else(|| GpuError::Pipeline("matmul_int8_weight not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_int8_weight_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scales.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: zero_points.buffer().as_entire_binding(),
                },
            ],
        });

        let wgx = m_u32.div_ceil(tile_u32);
        let wgy = n_u32.div_ceil(tile_u32);
        self.dispatch(pipeline, &bind_group, (wgx, wgy, 1));

        Ok(GpuTensor {
            shape: vec![a_shape[0], b_shape[0]],
            buffer: c_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// INT4 quantized matmul: A (f32 activations [M, K]) × packed Q4 B ([K/2, N]) → C (f32 [M, N]).
    ///
    /// Q4 packing: 2 values per byte (low nibble = even K, high nibble = odd K).
    /// `b_packed` must have shape `[K/2, N]` and dtype `U8`.
    /// `scales` must have shape `[groups, N]` where groups = ceil(K / group_size).
    /// Dequant: `val = (q4 - 8) * scale` (symmetric, center at 0).
    pub fn matmul_int4_weight(
        &self,
        a: &GpuTensor,
        b_packed: &GpuTensor,
        scales: &GpuTensor,
        group_size: usize,
    ) -> Result<GpuTensor, GpuError> {
        if b_packed.dtype() != GpuDtype::I8 {
            return Err(GpuError::Dtype(format!(
                "matmul_int4_weight expects I8 for B (packed Q4), got {:?}",
                b_packed.dtype()
            )));
        }
        let a_shape = a.shape();
        let b_shape = b_packed.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }
        // A[M, K] × B[K/2, N] — K from A must be twice the first dim of B
        if a_shape[1] != b_shape[0] * 2 {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        let _groups = k.div_ceil(group_size);

        let m_u32 = u32::try_from(m)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let k_u32 = u32::try_from(k)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let n_u32 = u32::try_from(n)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let gs_u32 = u32::try_from(group_size)
            .map_err(|_| GpuError::Unsupported("group_size overflow".into()))?;

        let tile = self.caps.adaptive_tile_size(m, n, k);
        let tile_u32 = u32::try_from(tile)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let c_size = (m_u32 as u64) * (n_u32 as u64) * 4;
        let limit = self.caps.max_storage_buffer_binding_size;
        if c_size > limit {
            return Err(GpuError::Buffer(format!(
                "matmul_int4_weight output {m}×{n} = {c_size} B exceeds device limit {limit} B"
            )));
        }
        let c_buffer = self.alloc_or_create_buffer(
            c_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let dims_buf = self.alloc_or_create_buffer(
            20, // M + K + N + GroupSize + Tile = 5 × u32
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let dims_data: [u32; 5] = [m_u32, k_u32, n_u32, gs_u32, tile_u32];
        self.queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims_data));

        let pipeline = self
            .pipelines
            .get("matmul_int4_weight")
            .ok_or_else(|| GpuError::Pipeline("matmul_int4_weight not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_int4_weight_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_packed.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scales.buffer().as_entire_binding(),
                },
            ],
        });

        let wgx = m_u32.div_ceil(tile_u32);
        let wgy = n_u32.div_ceil(tile_u32);
        self.dispatch(pipeline, &bind_group, (wgx, wgy, 1));

        Ok(GpuTensor {
            shape: vec![a_shape[0], b_shape[1]],
            buffer: c_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PACKED F16 CONVERSION HELPERS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Convert f32 tensor to packed F16 (2 f16 per u32). Returns tensor with GpuDtype::F16.
    pub fn f32_to_f16_packed(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {

        use crate::gpu_mixed::dispatch_f32_to_f16_packed;
        let pipeline = self
            .pipelines
            .get("f32_to_f16_packed")
            .ok_or_else(|| GpuError::Pipeline("f32_to_f16_packed not compiled".into()))?;
        dispatch_f32_to_f16_packed(self, &pipeline.pipeline, input)
    }

    /// Convert packed F16 tensor back to f32. Input must have dtype GpuDtype::F16.
    pub fn f16_packed_to_f32(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        use crate::gpu_mixed::dispatch_f16_packed_to_f32;
        if input.dtype() != GpuDtype::F16 {
            return Err(GpuError::Dtype(format!(
                "f16_packed_to_f32 expects F16 input, got {:?}",
                input.dtype()
            )));
        }
        let pipeline = self
            .pipelines
            .get("f16_packed_to_f32")
            .ok_or_else(|| GpuError::Pipeline("f16_packed_to_f32 not compiled".into()))?;
        dispatch_f16_packed_to_f32(self, &pipeline.pipeline, input)
    }

    pub(crate) fn compile_elementwise(&mut self) -> Result<(), GpuError> {
        self.compile_shader("elementwise", &[storage_binding(0, true), storage_binding(1, true), storage_binding(2, false), uniform_binding(3)], ELEMENTWISE_WGSL, "elementwise_main")
    }

    pub(crate) fn compile_elementwise_inplace(&mut self) -> Result<(), GpuError> {
        self.compile_shader("elementwise_inplace", &[storage_binding(0, false), uniform_binding(1)], ELEMENTWISE_INPLACE_WGSL, "elementwise_inplace_main")
    }

    /// Dispatch element-wise/unary op: output = op(a)

    pub fn elementwise_unary(&self, a: &GpuTensor, op: ElemOp) -> Result<GpuTensor, GpuError> {
        let shape = a.shape();
        let numel = a.numel();

        let out_buffer = self.alloc_or_create_buffer(
            (numel * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let cfg_buf = self.alloc_or_create_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let cfg_data: [u32; 4] = [numel as u32, op as u32, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("elementwise")
            .ok_or_else(|| GpuError::Pipeline("elementwise not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, a.buffer()), (1, a.buffer()), (2, &out_buffer)],
            &[(3, &cfg_buf)],
            numel as u32, 256, 4,
        );

        Ok(GpuTensor {
            shape: shape.to_vec(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    pub fn elementwise_unary_inplace(
        &self,
        tensor: &mut GpuTensor,
        op: ElemOp,
    ) -> Result<(), GpuError> {
        let numel = tensor.numel();
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("elementwise_inplace_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [numel as u32, op as u32, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("elementwise_inplace")
            .ok_or_else(|| GpuError::Pipeline("elementwise_inplace not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, tensor.buffer())],
            &[(1, &cfg_buf)],
            numel as u32, 256, 4,
        );
        Ok(())
    }

    pub fn gelu_inplace(&self, tensor: &mut GpuTensor) -> Result<(), GpuError> {
        self.elementwise_unary_inplace(tensor, ElemOp::Gelu)
    }

    /// Broadcast a GPU tensor to a target shape.
    /// Efficient scalar path (1 element → N) avoids full CPU round-trip.
    /// General path reads tensor to CPU, broadcasts via ndarray, re-uploads.
    pub fn broadcast_tensor(
        &self,
        tensor: &GpuTensor,
        target: &[usize],
    ) -> Result<GpuTensor, GpuError> {
        let shape = tensor.shape();
        let target_numel: usize = target.iter().product();

        if shape == target {
            return Ok(tensor.clone());
        }

        if tensor.numel() == 1 {
            let scalar = tensor.to_cpu_first_element()?;
            let data = vec![scalar; target_numel];
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("broadcast_scalar"),
                size: (target_numel * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&data));

            crate::gpu::gpu_observability::PCIE_WRITE_BYTES.fetch_add(
                (target_numel * 4) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );

            return Ok(GpuTensor {
                shape: target.to_vec(),
                buffer,
                dtype: tensor.dtype,
                device_id: tensor.device_id,
            });
        }

        let cpu_data = tensor.to_cpu()?;
        let bc_data = match cpu_data.broadcast(target) {
            Some(view) => view.to_owned(),
            None => {
                return Err(GpuError::ShapeMismatch(format!(
                    "Cannot broadcast {:?} to {:?}",
                    shape, target
                )));
            }
        };
        GpuTensor::from_cpu(&bc_data)
    }

    /// Dispatch element-wise/binary op: output = op(a, b).
    /// Automatically broadcasts shapes (including scalar→tensor) before dispatch.
    pub fn elementwise_binary(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        op: ElemOp,
    ) -> Result<GpuTensor, GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape != b_shape {
            let target = nexora_autograd_core::broadcast::broadcast_shapes(&a_shape, &b_shape);
            let a_bc = if a_shape != target {
                self.broadcast_tensor(a, &target)?
            } else {
                a.clone()
            };
            let b_bc = if b_shape != target {
                self.broadcast_tensor(b, &target)?
            } else {
                b.clone()
            };
            return self.elementwise_binary(&a_bc, &b_bc, op);
        }

        let numel = a.numel();

        let out_buffer = self.alloc_or_create_buffer(
            (numel * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let cfg_buf = self.alloc_or_create_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let cfg_data: [u32; 4] = [numel as u32, op as u32, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("elementwise")
            .ok_or_else(|| GpuError::Pipeline("elementwise not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, a.buffer()), (1, b.buffer()), (2, &out_buffer)],
            &[(3, &cfg_buf)],
            numel as u32, 256, 4,
        );

        Ok(GpuTensor {
            shape: a_shape.to_vec(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_binary(a, b, ElemOp::Add)
    }

    pub fn add_inplace(&self, a: &mut GpuTensor, b: &GpuTensor) -> Result<(), GpuError> {
        let result = self.add(&a.view_as(a.shape()), b)?;
        a.buffer = result.buffer;
        Ok(())
    }

    pub fn sub(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> { self.elementwise_binary(a, b, ElemOp::Sub) }
    pub fn mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> { self.elementwise_binary(a, b, ElemOp::Mul) }
    pub fn div(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> { self.elementwise_binary(a, b, ElemOp::Div) }
    pub fn sqrt(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> { self.elementwise_unary(a, ElemOp::Sqrt) }
    pub fn exp(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> { self.elementwise_unary(a, ElemOp::Exp) }
    pub fn relu(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> { self.elementwise_unary(a, ElemOp::Relu) }
    pub fn sigmoid(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> { self.elementwise_unary(a, ElemOp::Sigmoid) }
    pub fn silu(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> { self.elementwise_unary(a, ElemOp::Silu) }
    pub fn gelu(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> { self.elementwise_unary(a, ElemOp::Gelu) }
    pub fn tanh(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> { self.elementwise_unary(a, ElemOp::Tanh) }

    // ── LeakyReLU (needs custom cfg, not plain elementwise_unary) ──
    pub fn leaky_relu_inplace(
        &self,
        tensor: &mut GpuTensor,
        negative_slope: f32,
    ) -> Result<(), GpuError> {
        let numel = tensor.numel();
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("leaky_relu_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [
            numel as u32,
            ElemOp::LeakyRelu as u32,
            f32::to_bits(negative_slope),
            0,
        ];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("elementwise_inplace")
            .ok_or_else(|| GpuError::Pipeline("elementwise_inplace not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, tensor.buffer())],
            &[(1, &cfg_buf)],
            numel as u32, 256, 4,
        );
        Ok(())
    }

    /// LeakyReLU on GPU: returns new tensor
    pub fn leaky_relu(
        &self,
        input: &GpuTensor,
        negative_slope: f32,
    ) -> Result<GpuTensor, GpuError> {
        let mut result = input.clone();
        self.leaky_relu_inplace(&mut result, negative_slope)?;
        Ok(result)
    }
    /// Rotary embedding (RoPE) in-place on GPU.
    /// Applies rotary position embeddings to [total_rows, dim] tensor.
    /// cos/sin: [half] arrays for current position (precomputed via precompute_freqs_cis).
    pub fn rotary_embedding(
        &self,
        x: &GpuTensor,
        cos: &GpuTensor,
        sin: &GpuTensor,
        head_dim: u32,
    ) -> Result<GpuTensor, GpuError> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "RoPE GPU: x must be 2D [total_rows, dim]".into(),
            ));
        }
        let total_rows = shape[0] as u32;
        let dim = shape[1] as u32;
        if dim != head_dim {
            return Err(GpuError::ShapeMismatch(
                format!("RoPE GPU: dim {} != head_dim {}", dim, head_dim),
            ));
        }
        let half = head_dim / 2;

        // Create output buffer
        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rotary_output"),
            size: (shape[0] * shape[1] * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Copy input to output first (in-place kernel writes to output)
        self.with_encoder(|enc| {
            enc.copy_buffer_to_buffer(
                x.buffer(),
                0,
                &out_buffer,
                0,
                (shape[0] * shape[1] * 4) as u64,
            );
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rotary_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [total_rows, dim, half, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("rotary_embedding")
            .ok_or_else(|| GpuError::Pipeline("rotary_embedding not compiled".into()))?;

        let num_pairs = total_rows * half;
        self.dispatch_1d_chunked(
            pipeline,
            &[(0, &out_buffer), (1, cos.buffer()), (2, sin.buffer())],
            &[(3, &cfg_buf)],
            num_pairs, 256, 4,
        );

        Ok(GpuTensor {
            shape: shape.clone(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// Repeat KV heads to match Q heads for GQA.
    /// src: [seq, kv_heads, dim] → dst: [q_heads, seq, dim]
    /// Output is HEAD-MAJOR (compatible with fused_attention's [B, H, S, D] layout).
    /// The WGSL shader writes in head-major layout: element (qh, s, d) at qh*seq*dim + s*dim + d.
    pub fn repeat_heads(
        &self,
        src: &GpuTensor,
        kv_heads: u32,
        q_heads: u32,
        dim: u32,
    ) -> Result<GpuTensor, GpuError> {
        let shape = src.shape();
        if shape.len() != 3 {
            return Err(GpuError::ShapeMismatch(
                "repeat_heads: src must be 3D [seq, kv_heads, dim]".into(),
            ));
        }
        let seq = shape[0] as u32;

        let numel_dst = (seq * q_heads * dim) as usize;
        let dst_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repeat_heads_output"),
            size: (numel_dst * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repeat_heads_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [seq, kv_heads, q_heads, dim];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("repeat_heads")
            .ok_or_else(|| GpuError::Pipeline("repeat_heads not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, src.buffer()), (1, &dst_buffer)],
            &[(2, &cfg_buf)],
            numel_dst as u32, 256, 4,
        );

        Ok(GpuTensor {
            shape: vec![q_heads as usize, seq as usize, dim as usize],
            buffer: dst_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.2: REDUCE KERNELS
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_reduce(&mut self, op: ReduceOp) -> Result<(), GpuError> {
        let key = match op {
            ReduceOp::Sum => "reduce_sum",
            ReduceOp::Max => "reduce_max",
            ReduceOp::Min => "reduce_min",
        };
        if self.pipelines.contains_key(key) {
            return Ok(());
        }
        let wgsl = std::borrow::Cow::Owned(
            REDUCE_WGSL_TEMPLATE.replace("{{OP}}", &(op as u32).to_string()),
        );
        self.compile_pipeline(
            key,
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            wgsl,
            "reduce_main",
        )
    }

    /// Reduce 1D tensor to scalar (sum / max / min)

    pub fn reduce(&self, input: &GpuTensor, op: ReduceOp) -> Result<GpuTensor, GpuError> {
        let numel = input.numel();
        if numel == 0 {
            return Ok(GpuTensor {
                shape: vec![1],
                buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("reduce_output"),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                dtype: GpuDtype::F32,
                device_id: 0,
            });
        }

        let key = match op {
            ReduceOp::Sum => "reduce_sum",
            ReduceOp::Max => "reduce_max",
            ReduceOp::Min => "reduce_min",
        };

        let pipeline = self
            .pipelines
            .get(key)
            .ok_or_else(|| GpuError::Pipeline(format!("{} not compiled", key)))?;

        // Multi-pass: each workgroup reduces 256 elements → partial results
        // Then reduce partials until scalar remains
        let mut current_buffer = input.buffer().clone();
        let mut current_numel = numel;

        loop {
            let num_groups = current_numel.div_ceil(256);
            let num_groups_u32 = num_groups as u32;

            let out_size = (num_groups * 4) as u64;
            let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduce_partial"),
                size: out_size.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let needs_chunking = num_groups_u32 > MAX_WORKGROUPS_PER_DIM;

            if needs_chunking {
                let elems_per_full_chunk = MAX_WORKGROUPS_PER_DIM as u64 * 256;
                let mut input_offset: u64 = 0;
                let mut output_offset: u64 = 0;
                let mut remaining = current_numel as u64;

                while remaining > 0 {
                    let chunk_elems = remaining.min(elems_per_full_chunk);
                    let chunk_groups = chunk_elems.div_ceil(256) as u32;
                    let chunk_bytes = chunk_elems * 4;
                    let chunk_out_bytes = chunk_groups as u64 * 4;

                    let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("reduce_cfg_chunk"),
                        size: 8,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let cfg_data: [u32; 2] = [chunk_elems as u32, chunk_groups];
                    self.queue
                        .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

                    let bind_group = self.device.create_bind_group(
                        &wgpu::BindGroupDescriptor {
                            label: Some(&format!("{}_bind_group_chunk", key)),
                            layout: &pipeline.bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::Buffer(
                                        wgpu::BufferBinding {
                                            buffer: &current_buffer,
                                            offset: input_offset,
                                            size: wgpu::BufferSize::new(chunk_bytes),
                                        },
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Buffer(
                                        wgpu::BufferBinding {
                                            buffer: &out_buffer,
                                            offset: output_offset,
                                            size: wgpu::BufferSize::new(chunk_out_bytes),
                                        },
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: cfg_buf.as_entire_binding(),
                                },
                            ],
                        },
                    );

                    self.dispatch_impl(pipeline, &bind_group, (chunk_groups, 1, 1));

                    input_offset += chunk_bytes;
                    output_offset += chunk_out_bytes;
                    remaining -= chunk_elems;
                }
            } else {
                let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("reduce_cfg"),
                    size: 8,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let cfg_data: [u32; 2] = [current_numel as u32, num_groups as u32];
                self.queue
                    .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

                let bind_group = self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some(&format!("{}_bind_group", key)),
                        layout: &pipeline.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: current_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: cfg_buf.as_entire_binding(),
                            },
                        ],
                    },
                );

                self.dispatch(pipeline, &bind_group, (num_groups as u32, 1, 1));
            }

            if num_groups == 1 {
                return Ok(GpuTensor {
                    shape: vec![1],
                    buffer: out_buffer,
                    dtype: GpuDtype::F32,
                    device_id: 0,
                });
            }

            current_buffer = out_buffer;
            current_numel = num_groups;
        }
    }

    /// Convenience: reduce sum to scalar
    pub fn sum(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.reduce(input, ReduceOp::Sum)
    }

    /// Convenience: reduce max to scalar
    pub fn max(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.reduce(input, ReduceOp::Max)
    }

    /// Convenience: reduce min to scalar
    pub fn min(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.reduce(input, ReduceOp::Min)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.1: SOFTMAX
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_softmax(&mut self) -> Result<(), GpuError> {
        self.compile_shader("softmax", &[storage_binding(0, true), storage_binding(1, false), uniform_binding(2)], SOFTMAX_WGSL, "softmax_main")
    }

    /// Softmax along last axis. Input shape: [rows, dim].
    pub fn softmax(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "Softmax GPU: input must be 2D".into(),
            ));
        }
        let rows = shape[0] as u32;
        let dim = shape[1] as u32;
        let numel = input.numel();

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [rows, dim, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("softmax")
            .ok_or_else(|| GpuError::Pipeline("softmax not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (rows, 1, 1));

        Ok(GpuTensor {
            shape: shape.to_vec(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.2: RMSNORM
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_rms_norm(&mut self) -> Result<(), GpuError> {
        self.compile_shader("rms_norm", &[storage_binding(0, true), storage_binding(1, true), storage_binding(2, false), uniform_binding(3)], RMSNORM_WGSL, "rms_norm_main")
    }

    /// RMSNorm: out = x / rms(x) * weight. Shapes: x[batch, dim], weight[dim].
    pub fn rms_norm(
        &self,
        x: &GpuTensor,
        weight: &GpuTensor,
        eps: f32,
    ) -> Result<GpuTensor, GpuError> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch("RMSNorm GPU: x must be 2D".into()));
        }
        let batch = shape[0] as u32;
        let dim = shape[1] as u32;
        let numel = x.numel();

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [batch, dim, f32::to_bits(eps), 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("rms_norm")
            .ok_or_else(|| GpuError::Pipeline("rms_norm not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms_norm_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        Ok(GpuTensor {
            shape: shape.to_vec(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    pub(crate) fn compile_rms_norm_backward(&mut self) -> Result<(), GpuError> {
        self.compile_shader("rms_norm_backward", &[storage_binding(0, true), storage_binding(1, true), storage_binding(2, true), storage_binding(3, false), storage_binding(4, false), uniform_binding(5)], RMSNORM_BACKWARD_WGSL, "rms_norm_bwd_main")
    }

    pub fn rms_norm_backward(
        &self,
        input: &GpuTensor,
        weight: &GpuTensor,
        grad: &GpuTensor,
        eps: f32,
    ) -> Result<(GpuTensor, GpuTensor), GpuError> {
        let shape = input.shape();
        let batch = shape[0] as u32;
        let dim = shape[1] as u32;
        let numel = input.numel();

        let dx_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_dx"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dw_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_dw_atomic"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Zero-init dw
        let dw_tmp = GpuTensor {
            shape: vec![dim as usize],
            buffer: dw_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        self.fill_zero_u32(&dw_tmp)?;

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_bwd_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [batch, dim, f32::to_bits(eps), 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let pipeline = self
            .pipelines
            .get("rms_norm_backward")
            .ok_or_else(|| GpuError::Pipeline("rms_norm_backward not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms_norm_bwd_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dx_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dw_tmp.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        let dx = GpuTensor {
            shape: shape.to_vec(),
            buffer: dx_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let dw = GpuTensor {
            shape: vec![dim as usize],
            buffer: dw_tmp.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        Ok((dx, dw))
    }

    // ── LayerNorm Backward ──────────────────────────────────────────────

    pub(crate) fn compile_layer_norm_backward(&mut self) -> Result<(), GpuError> {
        self.compile_shader("layer_norm_backward", &[storage_binding(0, true), storage_binding(1, true), storage_binding(2, true), storage_binding(3, true), storage_binding(4, false), storage_binding(5, false), storage_binding(6, false), uniform_binding(7)], LAYERNORM_BACKWARD_WGSL, "layer_norm_bwd_main")
    }

    pub fn layer_norm_backward(
        &self,
        input: &GpuTensor,
        weight: &GpuTensor,
        bias: &GpuTensor,
        grad: &GpuTensor,
        eps: f32,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor), GpuError> {
        let shape = input.shape();
        let batch = shape[0] as u32;
        let dim = shape[1] as u32;
        let numel = input.numel();

        let dx_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_dx"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dw_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_dw_atomic"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let db_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_db_atomic"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dw_tmp = GpuTensor {
            shape: vec![dim as usize],
            buffer: dw_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let db_tmp = GpuTensor {
            shape: vec![dim as usize],
            buffer: db_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        self.fill_zero_u32(&dw_tmp)?;
        self.fill_zero_u32(&db_tmp)?;

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_bwd_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [batch, dim, f32::to_bits(eps), 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let pipeline = self
            .pipelines
            .get("layer_norm_backward")
            .ok_or_else(|| GpuError::Pipeline("layer_norm_backward not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_bwd_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dx_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dw_tmp.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: db_tmp.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        let dx = GpuTensor {
            shape: shape.to_vec(),
            buffer: dx_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let dw = GpuTensor {
            shape: vec![dim as usize],
            buffer: dw_tmp.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let db = GpuTensor {
            shape: vec![dim as usize],
            buffer: db_tmp.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        Ok((dx, dw, db))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.3: CROSS-ENTROPY LOSS
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_cross_entropy(&mut self) -> Result<(), GpuError> {
        self.compile_shader("cross_entropy", &[storage_binding(0, true), storage_binding(1, true), storage_binding(2, false), uniform_binding(3)], CROSS_ENTROPY_WGSL, "cross_entropy_main")
    }

    /// Cross-entropy loss. logits: [batch, classes], targets: [batch] (u32 as f32).
    /// Returns per-sample loss: [batch].

    pub fn cross_entropy(
        &self,
        logits: &GpuTensor,
        targets: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "CrossEntropy GPU: logits must be 2D".into(),
            ));
        }
        let batch = shape[0] as u32;
        let classes = shape[1] as u32;

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cross_entropy_output"),
            size: (batch as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cross_entropy_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [batch, classes, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("cross_entropy")
            .ok_or_else(|| GpuError::Pipeline("cross_entropy not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cross_entropy_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: targets.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        Ok(GpuTensor {
            shape: vec![batch as usize],
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    pub(crate) fn compile_cross_entropy_backward(&mut self) -> Result<(), GpuError> {
        self.compile_shader("cross_entropy_backward", &[storage_binding(0, true), storage_binding(1, true), storage_binding(2, true), storage_binding(3, false), uniform_binding(4)], CROSS_ENTROPY_BACKWARD_WGSL, "cross_entropy_bwd_main")
    }

    /// Cross-entropy backward: d_logits = grad * (softmax - one_hot(targets)).
    /// softmax/grad/d_logits: [batch, classes]. targets: [batch] (u32).
    pub fn cross_entropy_backward(
        &self,
        softmax: &GpuTensor,
        grad: &GpuTensor,
        targets: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        let shape = softmax.shape();
        let total = softmax.numel() as u32;

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cross_entropy_bwd_out"),
            size: (total as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cross_entropy_bwd_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [shape[0] as u32, shape[1] as u32, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let pipeline = self
            .pipelines
            .get("cross_entropy_backward")
            .ok_or_else(|| GpuError::Pipeline("cross_entropy_backward not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, softmax.buffer()), (1, grad.buffer()), (2, targets.buffer()), (3, &out_buffer)],
            &[(4, &cfg_buf)],
            total, 256, 4,
        );

        Ok(GpuTensor {
            shape: shape.clone(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.4: EMBEDDING
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_embedding(&mut self) -> Result<(), GpuError> {
        self.compile_shader("embedding", &[storage_binding(0, true), storage_binding(1, true), storage_binding(2, false), uniform_binding(3)], EMBEDDING_WGSL, "embedding_main")
    }

    pub(crate) fn compile_embedding_backward(&mut self) -> Result<(), GpuError> {
        self.compile_shader("embedding_backward", &[storage_binding(0, true), storage_binding(1, true), storage_binding(2, false), uniform_binding(3)], EMBEDDING_BACKWARD_WGSL, "embedding_backward_main")
    }

    /// Embedding backward: scatter-add grad into d_weight.
    /// ids: [N] u32, grad: [N, D] f32, output: d_weight [V, D] f32 (must be zero-initialized).
    pub fn embedding_backward(
        &self,
        ids: &GpuTensor,
        grad: &GpuTensor,
        vocab_size: usize,
    ) -> Result<GpuTensor, GpuError> {
        let grad_shape = grad.shape();
        let dim = grad_shape[grad_shape.len() - 1];
        let num_ids = ids.numel();
        let d_weight = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding_d_weight"),
            size: (vocab_size * dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Zero-init d_weight
        let d_weight_tmp = GpuTensor {
            shape: vec![vocab_size, dim],
            buffer: d_weight,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        self.fill_zero_u32(&d_weight_tmp)?;
        let d_weight_buf = d_weight_tmp.buffer;

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding_bwd_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [vocab_size as u32, dim as u32, num_ids as u32, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let pipeline = self
            .pipelines
            .get("embedding_backward")
            .ok_or_else(|| GpuError::Pipeline("embedding_backward not compiled".into()))?;

        let total_threads = (num_ids * dim) as u32;
        self.dispatch_1d_chunked(
            pipeline,
            &[(0, ids.buffer()), (1, grad.buffer()), (2, &d_weight_buf)],
            &[(3, &cfg_buf)],
            total_threads, 256, 4,
        );

        Ok(GpuTensor {
            shape: vec![vocab_size, dim],
            buffer: d_weight_buf,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// Embedding lookup: out[i * dim + d] = weight[tokens[i] * dim + d].
    /// ids: [seq_len], weight: [vocab, dim].

    pub fn embedding(&self, ids: &GpuTensor, weight: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let w_shape = weight.shape();
        if w_shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "Embedding GPU: weight must be 2D [vocab, dim]".into(),
            ));
        }
        let seq_len = ids.numel() as u32;
        let dim = w_shape[1] as u32;

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding_output"),
            size: (seq_len as u64) * (dim as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let vocab_size = w_shape[0] as u32;
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [vocab_size, dim, seq_len, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("embedding")
            .ok_or_else(|| GpuError::Pipeline("embedding not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, ids.buffer()), (1, weight.buffer()), (2, &out_buffer)],
            &[(3, &cfg_buf)],
            seq_len * dim, 256, 4,
        );

        Ok(GpuTensor {
            shape: vec![seq_len as usize, dim as usize],
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.5: LAYERNORM
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_layer_norm(&mut self) -> Result<(), GpuError> {
        self.compile_shader("layer_norm", &[storage_binding(0, true), storage_binding(1, true), storage_binding(2, true), storage_binding(3, false), uniform_binding(4)], LAYERNORM_WGSL, "layer_norm_main")
    }

    /// LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
    /// Shapes: x[batch, dim], weight[dim], bias[dim]
    pub fn layer_norm(
        &self,
        x: &GpuTensor,
        weight: &GpuTensor,
        bias: &GpuTensor,
        eps: f32,
    ) -> Result<GpuTensor, GpuError> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "LayerNorm GPU: x must be 2D".into(),
            ));
        }
        let batch = shape[0] as u32;
        let dim = shape[1] as u32;
        let numel = x.numel();

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [batch, dim, f32::to_bits(eps), 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("layer_norm")
            .ok_or_else(|| GpuError::Pipeline("layer_norm not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        Ok(GpuTensor {
            shape: vec![batch as usize, dim as usize],
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.6: TRANSPOSE + MATMUL BACKWARD
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_transpose(&mut self) -> Result<(), GpuError> {
        self.compile_shader("transpose", &[storage_binding(0, true), storage_binding(1, false), uniform_binding(2)], TRANSPOSE_WGSL, "transpose_main")
    }

    /// Transpose a 2D tensor [rows, cols] → [cols, rows]

    pub fn transpose(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "Transpose GPU: input must be 2D".into(),
            ));
        }
        let rows = shape[0] as u32;
        let cols = shape[1] as u32;
        let numel = input.numel();

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transpose_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transpose_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [rows, cols, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("transpose")
            .ok_or_else(|| GpuError::Pipeline("transpose not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, input.buffer()), (1, &out_buffer)],
            &[(2, &cfg_buf)],
            numel as u32, 256, 4,
        );

        Ok(GpuTensor {
            shape: vec![cols as usize, rows as usize],
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// Matmul backward: grad_a = grad_out @ b^T, grad_b = a^T @ grad_out
    pub fn matmul_backward(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        grad_out: &GpuTensor,
    ) -> Result<(GpuTensor, GpuTensor), GpuError> {
        let b_t = self.transpose(b)?;
        let grad_a = self.matmul(grad_out, &b_t)?;

        let a_t = self.transpose(a)?;
        let grad_b = self.matmul(&a_t, grad_out)?;

        Ok((grad_a, grad_b))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 3: FUSED ATTENTION (Flash Attention-style)
    // ═══════════════════════════════════════════════════════════════════════════

    pub(crate) fn compile_fused_attention(&mut self) -> Result<(), GpuError> {
        self.compile_shader("fused_attention", &[storage_binding(0, true), storage_binding(1, true), storage_binding(2, true), storage_binding(3, false), uniform_binding(4), uniform_binding(5)], FUSED_ATTENTION_WGSL, "fused_attention_main")
    }

    /// Fused attention: O = softmax(Q @ K^T / scale) @ V
    /// Shapes: Q/K/V = [batch, heads, seq, dim], output = [batch, heads, seq, dim]
    /// Uses online softmax (Flash Attention-style) with causal masking.
    pub fn fused_attention(
        &self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        scale: f32,
        causal: bool,
    ) -> Result<GpuTensor, GpuError> {
        let q_shape = q.shape();
        if q_shape.len() != 4 {
            return Err(GpuError::ShapeMismatch(
                "Fused Attention: Q must be 4D [B,H,S,D]".into(),
            ));
        }
        let batch = q_shape[0] as u32;
        let heads = q_shape[1] as u32;
        let seq_len = q_shape[2] as u32;
        let dim = q_shape[3] as u32;
        let numel = q.numel();

        // CUDA FlashAttention path (preferred when CUDA backend is active)
        // Avoids Vec<u8>/Vec<f32> intermediate allocations by working directly
        // with the mapped wgpu staging buffer and CUDA dtoh_sync_copy.
        #[cfg(feature = "cuda")]
        if let Some(cuda) = self.cuda {
            // Batch QKV into a single contiguous readback (1 staging buffer instead of 3)
            let q_size = (q.numel() * 4) as u64;
            let k_size = (k.numel() * 4) as u64;
            let v_size = (v.numel() * 4) as u64;
            let total_bytes = q_size + k_size + v_size;
            let staging = self
                .device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fused_attn_qkv_staging"),
                    size: total_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fused_attn_qkv_encoder"),
                });
            self.flush();
            encoder.copy_buffer_to_buffer(q.buffer(), 0, &staging, 0, q_size);
            encoder.copy_buffer_to_buffer(k.buffer(), 0, &staging, q_size, k_size);
            encoder.copy_buffer_to_buffer(v.buffer(), 0, &staging, q_size + k_size, v_size);
            self.queue.submit(Some(encoder.finish()));

            // Read QKV from staging, upload to CUDA — NO Vec<u8> intermediate
            let q_len = q.numel();
            let k_len = k.numel();
            let v_len = v.numel();
            let (q_cuda, k_cuda, v_cuda) = {
                let _limiter_token = self.readback_limiter
                    .acquire(std::time::Duration::from_secs(30))
                    .then_some(())
                    .ok_or_else(|| GpuError::Timeout("GPU readback limiter".into()))?;
                let slice = staging.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                let deadline = std::time::Instant::now()
                    + std::time::Duration::from_secs(30);
                let result = loop {
                    self.device.poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(std::time::Duration::from_millis(100)),
                    });
                    match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(Ok(())) => {
                            let mapped = slice.get_mapped_range();
                            let qkv_f32: &[f32] = bytemuck::cast_slice(&mapped);

                            let q_cuda = CudaTensor::from_cpu(&cuda.device, q.shape().clone(), &qkv_f32[..q_len]);
                            let k_cuda = CudaTensor::from_cpu(&cuda.device, k.shape().clone(), &qkv_f32[q_len..q_len + k_len]);
                            let v_cuda = CudaTensor::from_cpu(&cuda.device, v.shape().clone(), &qkv_f32[q_len + k_len..q_len + k_len + v_len]);

                            drop(mapped);
                            staging.unmap();

                            match (q_cuda, k_cuda, v_cuda) {
                                (Ok(q), Ok(k), Ok(v)) => break (q, k, v),
                                (e, _, _) => break Err(e.unwrap_err()),
                                (_, e, _) => break Err(e.unwrap_err()),
                                (_, _, e) => break Err(e.unwrap_err()),
                            }
                        }
                        Ok(Err(_)) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            break Err(GpuError::Timeout("fused_attn QKV readback".into()));
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            if std::time::Instant::now() > deadline {
                                break Err(GpuError::Timeout("fused_attn QKV readback".into()));
                            }
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    }
                };
                crate::gpu::gpu_observability::PCIE_READ_BYTES
                    .fetch_add(total_bytes, std::sync::atomic::Ordering::Relaxed);
                result.map_err(|e| e)?
            };

            let result_cuda = cuda
                .fused_attention(&q_cuda, &k_cuda, &v_cuda, scale, causal)
                .map_err(|e| GpuError::Compute(format!("CUDA fused_attention: {e}")))?;

            // Write result to wgpu output buffer via staging — NO Vec<f32> intermediate
            let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fused_attention_output_cuda"),
                size: (numel * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            {
                let result_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fused_attn_result_staging"),
                    size: (numel * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let rslice = result_staging.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                rslice.map_async(wgpu::MapMode::Write, move |r| {
                    let _ = tx.send(r);
                });
                let deadline = std::time::Instant::now()
                    + std::time::Duration::from_secs(30);
                loop {
                    self.device.poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(std::time::Duration::from_millis(100)),
                    });
                    match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(Ok(())) => {
                            let mut mapped = rslice.get_mapped_range_mut();
                            let result_f32: &mut [f32] = bytemuck::cast_slice_mut(&mut mapped);
                            cuda.device
                                .dtoh_sync_copy(&result_cuda.buffer, result_f32)
                                .map_err(|e| GpuError::Transfer(format!("CUDA result dtoh: {e}")))?;
                            drop(mapped);
                            result_staging.unmap();

                            let mut encoder = self
                                .device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("fused_attn_result_encoder"),
                                });
                            encoder.copy_buffer_to_buffer(
                                &result_staging, 0, &out_buffer, 0, (numel * 4) as u64,
                            );
                            self.queue.submit(Some(encoder.finish()));
                            break;
                        }
                        Ok(Err(_)) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            return Err(GpuError::Timeout("fused_attn result staging map".into()));
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            if std::time::Instant::now() > deadline {
                                return Err(GpuError::Timeout("fused_attn result staging map".into()));
                            }
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    }
                }
            }

            return Ok(GpuTensor {
                shape: q_shape.clone(),
                buffer: out_buffer,
                dtype: GpuDtype::F32,
                device_id: 0,
            });
        }

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg1_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_cfg1"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg1: [u32; 4] = [batch, heads, seq_len, dim];
        self.queue
            .write_buffer(&cfg1_buf, 0, bytemuck::cast_slice(&cfg1));

        let causal_flag: u32 = if causal { 1 } else { 0 };
        let cfg2_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_cfg2"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg2: [u32; 4] = [f32::to_bits(scale), causal_flag, 0, 0];
        self.queue
            .write_buffer(&cfg2_buf, 0, bytemuck::cast_slice(&cfg2));

        let pipeline = self
            .pipelines
            .get("fused_attention")
            .ok_or_else(|| GpuError::Pipeline("fused_attention not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_attention_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cfg1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cfg2_buf.as_entire_binding(),
                },
            ],
        });

        let num_wg = batch * heads * seq_len;
        let mut cfg2_mut: [u32; 4] = [f32::to_bits(scale), causal_flag, 0, 0];
        if num_wg > MAX_WORKGROUPS_PER_DIM {
            let mut remaining = num_wg;
            let mut base: u32 = 0;
            while remaining > 0 {
                let chunk = remaining.min(MAX_WORKGROUPS_PER_DIM);
                cfg2_mut[2] = base;
                self.queue
                    .write_buffer(&cfg2_buf, 0, bytemuck::cast_slice(&cfg2_mut));
                self.dispatch_impl(pipeline, &bind_group, (chunk, 1, 1));
                base += chunk;
                remaining -= chunk;
            }
        } else {
            self.dispatch(pipeline, &bind_group, (num_wg, 1, 1));
        }

        Ok(GpuTensor {
            shape: q_shape.clone(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ── Fused Attention Backward ─────────────────────────────────────
    // Pure GPU kernel for causal_attention backward.
    // Input: Q/K/V/dO as [B,H,S,D], Output: dQ/dK/dV as [B,H,S,D].
    // Allocates dQ/dK/dV internally (caller must ensure zero-init if needed).

    pub(crate) fn compile_fused_attention_backward(&mut self) -> Result<(), GpuError> {
        self.compile_shader("fused_attention_backward", &[storage_binding(0, true), storage_binding(1, true), storage_binding(2, true), storage_binding(3, true), storage_binding(4, false), storage_binding(5, false), storage_binding(6, false), uniform_binding(7), uniform_binding(8)], FUSED_ATTENTION_BACKWARD_WGSL, "fused_attn_backward_main")
    }

    /// Fused attention backward: compute dQ, dK, dV from Q, K, V, dO.
    /// All shapes: [batch, heads, seq, dim]. Uses causal masking.
    /// dK/dV are atomically accumulated (must be zero-initialized).
    pub fn fused_attention_backward(
        &self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        grad: &GpuTensor,
        scale: f32,
        causal: bool,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor), GpuError> {
        let shape = q.shape();
        if shape.len() != 4 {
            return Err(GpuError::ShapeMismatch(
                "Fused Attention backward: Q must be 4D [B,H,S,D]".into(),
            ));
        }
        let batch = shape[0] as u32;
        let heads = shape[1] as u32;
        let seq_len = shape[2] as u32;
        let dim = shape[3] as u32;
        let numel = q.numel();
        let byte_size = (numel * 4) as u64;

        // Allocate output buffers
        let dq_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_dq"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dk_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_dk_atomic"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dv_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_dv_atomic"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Zero-init dK and dV (atomic buffers start non-zero from pool)
        let dk_tmp = GpuTensor {
            shape: shape.clone(),
            buffer: dk_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let dv_tmp = GpuTensor {
            shape: shape.clone(),
            buffer: dv_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        self.fill_zero_u32(&dk_tmp)?;
        self.fill_zero_u32(&dv_tmp)?;

        let cfg1_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attn_bwd_cfg1"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // batch_heads = batch * heads (heads treated as flat batch dim in shader)
        let cfg1: [u32; 4] = [batch * heads, seq_len, dim, f32::to_bits(scale)];
        self.queue
            .write_buffer(&cfg1_buf, 0, bytemuck::cast_slice(&cfg1));

        let causal_flag: u32 = if causal { 1 } else { 0 };
        let cfg2_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attn_bwd_cfg2"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut cfg2_mut: [u32; 4] = [causal_flag, 0, 0, 0];
        self.queue
            .write_buffer(&cfg2_buf, 0, bytemuck::cast_slice(&cfg2_mut));

        let pipeline = self
            .pipelines
            .get("fused_attention_backward")
            .ok_or_else(|| GpuError::Pipeline("fused_attention_backward not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_attention_backward_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dq_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dk_tmp.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dv_tmp.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: cfg1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: cfg2_buf.as_entire_binding(),
                },
            ],
        });

        let num_wg = batch * heads * seq_len;
        if num_wg > MAX_WORKGROUPS_PER_DIM {
            let mut remaining = num_wg;
            let mut base: u32 = 0;
            while remaining > 0 {
                let chunk = remaining.min(MAX_WORKGROUPS_PER_DIM);
                cfg2_mut[1] = base;
                self.queue
                    .write_buffer(&cfg2_buf, 0, bytemuck::cast_slice(&cfg2_mut));
                self.dispatch_impl(pipeline, &bind_group, (chunk, 1, 1));
                base += chunk;
                remaining -= chunk;
            }
        } else {
            self.dispatch(pipeline, &bind_group, (num_wg, 1, 1));
        }

        let dq = GpuTensor {
            shape: shape.clone(),
            buffer: dq_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let dk = GpuTensor {
            shape: shape.clone(),
            buffer: dk_tmp.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let dv = GpuTensor {
            shape: shape.clone(),
            buffer: dv_tmp.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };

        Ok((dq, dk, dv))
    }

}
