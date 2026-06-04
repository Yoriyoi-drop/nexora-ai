//! Expert modules for HAS-MoE-FFN

use serde::{Deserialize, Serialize};
use tracing::warn;

/// GELU activation function (standalone, for sharing across modules)
pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    let x_cubed = x * x * x;
    let tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed);
    x * 0.5 * (1.0 + tanh_arg.tanh())
}

/// Expert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub use_dropout: bool,
    pub dropout_rate: f32,
}

/// Individual expert in the MoE system
pub struct Expert {
    config: ExpertConfig,
    fc1_weights: Option<Vec<Vec<f32>>>,
    fc1_bias: Option<Vec<f32>>,
    fc2_weights: Option<Vec<Vec<f32>>>,
    fc2_bias: Option<Vec<f32>>,

    /// Q4 quantized weights: (packed_bytes, scales) for fc1 [intermediate × hidden]
    fc1_q4: Option<(Vec<u8>, Vec<f32>)>,
    /// Q4 quantized weights: (packed_bytes, scales) for fc2 [hidden × intermediate]
    fc2_q4: Option<(Vec<u8>, Vec<f32>)>,
    /// Group size used for Q4 quantization (default 128)
    q4_group_size: Option<usize>,

    /// Expert specialization domain (set by ExpertPoolConfig).
    pub domain: Option<String>,
    /// Model tier this expert belongs to ("shared", "ultra", "apex", etc.).
    pub tier: Option<String>,
    #[cfg(feature = "gpu")]
    fc1_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "gpu")]
    fc1_bias_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "gpu")]
    fc2_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "gpu")]
    fc2_bias_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    /// Q4 packed GPU tensors: (fc1_q4_gpu, fc1_scales_gpu, fc2_q4_gpu, fc2_scales_gpu)
    #[cfg(feature = "gpu")]
    fc1_q4_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "gpu")]
    fc1_q4_scales_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "gpu")]
    fc2_q4_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "gpu")]
    fc2_q4_scales_gpu: std::sync::OnceLock<Option<nexora_autograd::gpu::GpuTensor>>,
    #[cfg(feature = "cuda")]
    fc1_cuda: std::sync::OnceLock<Option<nexora_autograd::gpu::cuda::CudaTensor>>,
    #[cfg(feature = "cuda")]
    fc1_bias_cuda: std::sync::OnceLock<Option<nexora_autograd::gpu::cuda::CudaTensor>>,
    #[cfg(feature = "cuda")]
    fc2_cuda: std::sync::OnceLock<Option<nexora_autograd::gpu::cuda::CudaTensor>>,
    #[cfg(feature = "cuda")]
    fc2_bias_cuda: std::sync::OnceLock<Option<nexora_autograd::gpu::cuda::CudaTensor>>,
}

impl Expert {
    /// Create new expert
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        use_dropout: bool,
        dropout_rate: f32,
    ) -> Self {
        let config = ExpertConfig {
            hidden_size,
            intermediate_size,
            use_dropout,
            dropout_rate,
        };

        Self {
            config,
            fc1_weights: None,
            fc1_bias: None,
            fc2_weights: None,
            fc2_bias: None,
            fc1_q4: None,
            fc2_q4: None,
            q4_group_size: None,
            domain: None,
            tier: None,
            #[cfg(feature = "gpu")]
            fc1_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            fc1_bias_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            fc2_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            fc2_bias_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            fc1_q4_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            fc1_q4_scales_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            fc2_q4_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "gpu")]
            fc2_q4_scales_gpu: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            fc1_cuda: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            fc1_bias_cuda: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            fc2_cuda: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            fc2_bias_cuda: std::sync::OnceLock::new(),
        }
    }

    fn get_fc1_weights(&self) -> Result<&Vec<Vec<f32>>, String> {
        self.fc1_weights.as_ref().ok_or_else(|| {
            warn!("fc1_weights not initialized — call init_random()");
            "fc1_weights not initialized — call init_random()".to_string()
        })
    }

    fn get_fc1_bias(&self) -> Result<&Vec<f32>, String> {
        self.fc1_bias.as_ref().ok_or_else(|| {
            warn!("fc1_bias not initialized — call init_random()");
            "fc1_bias not initialized — call init_random()".to_string()
        })
    }

    fn get_fc2_weights(&self) -> Result<&Vec<Vec<f32>>, String> {
        self.fc2_weights.as_ref().ok_or_else(|| {
            warn!("fc2_weights not initialized — call init_random()");
            "fc2_weights not initialized — call init_random()".to_string()
        })
    }

    fn get_fc2_bias(&self) -> Result<&Vec<f32>, String> {
        self.fc2_bias.as_ref().ok_or_else(|| {
            warn!("fc2_bias not initialized — call init_random()");
            "fc2_bias not initialized — call init_random()".to_string()
        })
    }

    pub fn init_random(&mut self) {
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;
        self.fc1_weights = Some(Self::init_weights(h, i));
        self.fc1_bias = Some(vec![0.0; i]);
        self.fc2_weights = Some(Self::init_weights(i, h));
        self.fc2_bias = Some(vec![0.0; h]);
    }

    /// Quantize fc1/fc2 weights to Q4 packed format using symmetric per-group quantization.
    /// Stores packed bytes + scales, replacing the FP32 weights in memory.
    /// Call this after init_random() or loading FP32 weights to save ~4× VRAM on GPU.
    pub fn quantize_to_q4(&mut self, group_size: usize) {
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;
        if let Some(ref w1) = self.fc1_weights {
            // fc1_w is [intermediate, hidden]. quantize_fp32_to_q4 expects [K=hidden, N=intermediate].
            // Build transposed flat: fc1_w^T flattened as [hidden, intermediate].
            let mut flat_t = Vec::with_capacity(h * i);
            for k in 0..h {
                for j in 0..i {
                    flat_t.push(w1[j][k]);
                }
            }
            let (packed, scales) = nexora_quantization::gemm::quantize_fp32_to_q4(&flat_t, i, h, group_size);
            self.fc1_q4 = Some((packed, scales));
        }
        if let Some(ref w2) = self.fc2_weights {
            // fc2_w is [hidden, intermediate]. quantize_fp32_to_q4 expects [K=intermediate, N=hidden].
            // Build transposed flat: fc2_w^T flattened as [intermediate, hidden].
            let mut flat_t = Vec::with_capacity(i * h);
            for k in 0..i {
                for j in 0..h {
                    flat_t.push(w2[j][k]);
                }
            }
            let (packed, scales) = nexora_quantization::gemm::quantize_fp32_to_q4(&flat_t, h, i, group_size);
            self.fc2_q4 = Some((packed, scales));
        }
        self.fc1_weights = None;
        self.fc2_weights = None;
        self.q4_group_size = Some(group_size);
    }

    /// Drop FP32 weight matrices but keep biases + Q4 data.
    /// Biases are NOT quantized and must be retained for Q4 forward.
    pub fn drop_cpu_fp32_keep_q4(&mut self) {
        self.fc1_weights = None;
        self.fc2_weights = None;
        // Biases are retained — they are not quantized
    }

    pub fn drop_cpu_weights(&mut self) {
        self.fc1_weights = None;
        self.fc1_bias = None;
        self.fc2_weights = None;
        self.fc2_bias = None;
        self.fc1_q4 = None;
        self.fc2_q4 = None;
        self.q4_group_size = None;
    }

    pub fn has_weights(&self) -> bool {
        self.fc1_weights.is_some()
    }

    /// Initialize weights with Xavier/Glorot initialization
    fn init_weights(input_size: usize, output_size: usize) -> Vec<Vec<f32>> {
        let scale = (2.0 / (input_size + output_size) as f32).sqrt();
        let mut weights = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            let row: Vec<f32> = (0..input_size)
                .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
                .collect();
            weights.push(row);
        }
        weights
    }

    /// Forward pass through expert
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        {
            let gpu_result = self.forward_gpu(input);
            if let Some(result) = gpu_result {
                return result;
            }
            tracing::warn!("MoE expert GPU forward failed, falling back to CPU");
        }

        // First linear layer + activation
        let hidden = self.fc1_forward(input);
        let activated = self.apply_gelu(&hidden);

        // Apply dropout if enabled
        let dropped = if self.config.use_dropout {
            self.apply_dropout(&activated)
        } else {
            activated
        };

        // Second linear layer
        let output = self.fc2_forward(&dropped);

        output
    }

    /// GPU-accelerated forward pass
    #[cfg(feature = "gpu")]
    fn forward_gpu(&self, input: &[f32]) -> Option<Vec<f32>> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        use tracing::warn;
        let ctx = match GpuContext::global() {
            Ok(ctx) => ctx,
            Err(e) => {
                warn!("GPU forward failed (GPU unavailable): {e}");
                return None;
            }
        };
        let dim = input.len();

        // Use cached weights — avoids re-flattening + re-upload per forward
        let (w1, b1, w2, b2) = self.ensure_weights_gpu()?;

        // Upload input as [1, hidden_size]
        let input_gpu = GpuTensor::from_slice(vec![1, dim], input).ok()?;

        let mut hidden_gpu = ctx.matmul(&input_gpu, w1).ok()?;
        hidden_gpu = ctx.add(&hidden_gpu, b1).ok()?;

        ctx.gelu_inplace(&mut hidden_gpu).ok()?;

        let act_gpu = if self.config.use_dropout {
            let rate = self.config.dropout_rate;
            let scale = 1.0 / (1.0 - rate);
            let inter_size = hidden_gpu.shape()[1];
            let mask_gpu = GpuTensor::zeros(&[1, inter_size]).ok()?;
            ctx.generate_dropout_mask(&mask_gpu, rate, scale, 42).ok()?;
            ctx.mul(&hidden_gpu, &mask_gpu).ok()?
        } else {
            hidden_gpu
        };

        let out_gpu = ctx.matmul(&act_gpu, w2).ok()?;
        let out_gpu = ctx.add(&out_gpu, b2).ok()?;

        let out_cpu = out_gpu.to_cpu().ok()?;
        Some(out_cpu.iter().copied().collect())
    }

    /// Lazily upload and transpose expert weights to GPU — cached via OnceLock.
    /// Returns (fc1_w_t, fc1_bias, fc2_w_t, fc2_bias) as GpuTensors or None.
    #[cfg(feature = "gpu")]
    fn ensure_weights_gpu(&self) -> Option<(&nexora_autograd::gpu::GpuTensor, &nexora_autograd::gpu::GpuTensor, &nexora_autograd::gpu::GpuTensor, &nexora_autograd::gpu::GpuTensor)> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global().ok()?;
        let fc1_w = self.fc1_weights.as_ref()?;
        let fc1_b = self.fc1_bias.as_ref()?;
        let fc2_w = self.fc2_weights.as_ref()?;
        let fc2_b = self.fc2_bias.as_ref()?;
        let w1 = self.fc1_gpu.get_or_init(|| {
            let shape = vec![fc1_w.len(), self.config.hidden_size];
            let flat: Vec<f32> = fc1_w.iter().flatten().copied().collect();
            let gpu = GpuTensor::from_slice(shape, &flat).ok()?;
            ctx.transpose(&gpu).ok()
        }).as_ref()?;
        let b1 = self.fc1_bias_gpu.get_or_init(|| {
            GpuTensor::from_slice(vec![1, fc1_b.len()], fc1_b).ok()
        }).as_ref()?;
        let w2 = self.fc2_gpu.get_or_init(|| {
            let shape = vec![fc2_w.len(), self.config.intermediate_size];
            let flat: Vec<f32> = fc2_w.iter().flatten().copied().collect();
            let gpu = GpuTensor::from_slice(shape, &flat).ok()?;
            ctx.transpose(&gpu).ok()
        }).as_ref()?;
        let b2 = self.fc2_bias_gpu.get_or_init(|| {
            GpuTensor::from_slice(vec![1, fc2_b.len()], fc2_b).ok()
        }).as_ref()?;
        Some((w1, b1, w2, b2))
    }

    /// Batched GPU forward: processes N tokens in one GPU call.
    /// Returns None if GPU unavailable (CPU fallback handled by caller).
    #[cfg(feature = "gpu")]
    pub fn forward_batched_gpu(&self, inputs: &ndarray::Array2<f32>) -> Option<ndarray::Array2<f32>> {
        self.forward_batched_gpu_inner(inputs)
            .and_then(|t| t.to_cpu().ok())
            .and_then(|d| d.into_dimensionality::<ndarray::Ix2>().ok())
    }

    /// Batched GPU forward that keeps result on GPU (no readback).
    /// Returns the raw GPU tensor for deferred readback / fused scatter-add.
    #[cfg(feature = "gpu")]
    pub fn forward_batched_gpu_keep_gpu(&self, inputs: &ndarray::Array2<f32>) -> Option<nexora_autograd::gpu::GpuTensor> {
        self.forward_batched_gpu_inner(inputs)
    }

    /// Internal batched wgpu forward — all GPU ops, no readback.
    /// Prefers Q4 path if quantized weights are available, falls back to FP32.
    #[cfg(feature = "gpu")]
    fn forward_batched_gpu_inner(&self, inputs: &ndarray::Array2<f32>) -> Option<nexora_autograd::gpu::GpuTensor> {
        if self.fc1_q4.is_some() {
            return self.forward_batched_gpu_q4(inputs);
        }
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let (w1, b1, w2, b2) = self.ensure_weights_gpu()?;
        let ctx = GpuContext::global().ok()?;
        let (batch, dim) = inputs.dim();
        let input_data = inputs.as_slice()?;
        let input_gpu = GpuTensor::from_slice(vec![batch, dim], input_data).ok()?;
        let mut hidden = ctx.matmul(&input_gpu, w1).ok()?;
        hidden = ctx.add(&hidden, b1).ok()?;
        ctx.gelu_inplace(&mut hidden).ok()?;
        let output = ctx.matmul(&hidden, w2).ok()?;
        ctx.add(&output, b2).ok()
    }

    /// Batched wgpu forward via INT4 quantized weights.
    /// Uses `matmul_int4_weight` kernel for compute directly from Q4 packed data.
    #[cfg(feature = "gpu")]
    fn forward_batched_gpu_q4(&self, inputs: &ndarray::Array2<f32>) -> Option<nexora_autograd::gpu::GpuTensor> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global().ok()?;
        let (ref packed1, ref scales1) = self.fc1_q4.as_ref()?;
        let (ref packed2, ref scales2) = self.fc2_q4.as_ref()?;
        let gs = self.q4_group_size?;
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;
        let (batch, dim) = inputs.dim();

        // Lazily upload Q4 packed + scales to GPU via OnceLock
        let b_q4 = self.fc1_q4_gpu.get_or_init(|| {
            let shape = vec![h, i]; // original [K, N] = [hidden, intermediate]
            GpuTensor::from_cpu_q4_packed(shape, packed1).ok()
        }).as_ref()?;
        let b_scales = self.fc1_q4_scales_gpu.get_or_init(|| {
            let groups = h.div_ceil(gs);
            let shape = vec![groups, i];
            GpuTensor::from_slice(shape, scales1).ok()
        }).as_ref()?;
        let w2_q4 = self.fc2_q4_gpu.get_or_init(|| {
            let shape = vec![i, h]; // original [K, N] = [intermediate, hidden]
            GpuTensor::from_cpu_q4_packed(shape, packed2).ok()
        }).as_ref()?;
        let w2_scales = self.fc2_q4_scales_gpu.get_or_init(|| {
            let groups = i.div_ceil(gs);
            let shape = vec![groups, h];
            GpuTensor::from_slice(shape, scales2).ok()
        }).as_ref()?;

        let input_data = inputs.as_slice()?;
        let input_gpu = GpuTensor::from_slice(vec![batch, dim], input_data).ok()?;

        // fc1: [batch, hidden] @ Q4([hidden, intermediate]) → [batch, intermediate]
        // matmul_int4_weight: A[M,K] × B_packed[K/2,N] → C[M,N]
        // A = input [batch, hidden] → K=hidden
        // B_packed = q4 with shape [hidden, intermediate] → [K/2, N]
        // Note: from_cpu_q4_packed stores byte layout as [K, N] but shader reads [K/2, N]*N bytes
        let mut hidden = ctx.matmul_int4_weight(&input_gpu, b_q4, b_scales, gs).ok()?;
        // GELU activation — in-place
        ctx.gelu_inplace(&mut hidden).ok()?;

        // fc2: [batch, intermediate] @ Q4([intermediate, hidden]) → [batch, hidden]
        let output = ctx.matmul_int4_weight(&hidden, w2_q4, w2_scales, gs).ok()?;
        Some(output)
    }

    /// Lazily upload expert weights to CUDA — cached via OnceLock.
    /// Returns (fc1, fc1_bias, fc2, fc2_bias) as CudaTensors or None.
    #[cfg(feature = "cuda")]
    pub(crate) fn ensure_weights_cuda(&self, cuda: &nexora_autograd::gpu::cuda::CudaRuntime) -> Option<(&nexora_autograd::gpu::cuda::CudaTensor, &nexora_autograd::gpu::cuda::CudaTensor, &nexora_autograd::gpu::cuda::CudaTensor, &nexora_autograd::gpu::cuda::CudaTensor)> {
        use nexora_autograd::gpu::cuda::CudaTensor;
        let fc1_w = self.fc1_weights.as_ref()?;
        let fc1_b = self.fc1_bias.as_ref()?;
        let fc2_w = self.fc2_weights.as_ref()?;
        let fc2_b = self.fc2_bias.as_ref()?;
        let w1 = self.fc1_cuda.get_or_init(|| {
            let flat: Vec<f32> = fc1_w.iter().flatten().copied().collect();
            // Store transposed: [hidden_size, intermediate_size]
            CudaTensor::from_cpu(&cuda.device, vec![fc1_w.len(), self.config.hidden_size], &flat).ok()
        }).as_ref()?;
        let b1 = self.fc1_bias_cuda.get_or_init(|| {
            CudaTensor::from_cpu(&cuda.device, vec![1, fc1_b.len()], fc1_b).ok()
        }).as_ref()?;
        let w2 = self.fc2_cuda.get_or_init(|| {
            let flat: Vec<f32> = fc2_w.iter().flatten().copied().collect();
            // Store transposed: [intermediate_size, hidden_size]
            CudaTensor::from_cpu(&cuda.device, vec![fc2_w.len(), self.config.intermediate_size], &flat).ok()
        }).as_ref()?;
        let b2 = self.fc2_bias_cuda.get_or_init(|| {
            CudaTensor::from_cpu(&cuda.device, vec![1, fc2_b.len()], fc2_b).ok()
        }).as_ref()?;
        Some((w1, b1, w2, b2))
    }

    /// Batched CUDA forward: processes N tokens via cuBLAS (no CPU round-trip for weights).
    /// Returns None if CUDA unavailable.
    #[cfg(feature = "cuda")]
    pub fn forward_batched_cuda(&self, inputs: &ndarray::Array2<f32>) -> Option<ndarray::Array2<f32>> {
        use nexora_autograd::gpu::{GpuContext, GpuBackend};
        let ctx = GpuContext::global().ok()?;
        if ctx.backend() != GpuBackend::Cuda {
            return None;
        }
        let cuda = ctx.cuda_runtime()?;
        let (w1, b1, w2, b2) = self.ensure_weights_cuda(cuda)?;

        let n = inputs.shape()[0];
        let dim = inputs.shape()[1];

        // Upload input to GPU (zero-copy contiguous path, fallback to iter)
        let input_data: Vec<f32> = inputs.as_slice().map_or_else(
            || inputs.iter().copied().collect(),
            |s| s.to_vec(),
        );
        let input_gpu = nexora_autograd::gpu::cuda::CudaTensor::from_cpu(
            &cuda.device, vec![n, dim], &input_data,
        ).ok()?;

        // fc1: [n, dim] @ [inter, dim] → [n, inter]
        // w1 stored as [inter, dim]; cuBLAS transposes both operands internally
        let mut hidden = cuda.matmul(&input_gpu, w1).ok()?;
        hidden = cuda.add(&hidden, b1).ok()?;
        cuda.gelu_inplace(&mut hidden).ok()?;

        // fc2: [n, inter] @ [dim, inter] → [n, dim]
        // w2 stored as [dim, inter]; cuBLAS handles transpose
        let output = cuda.matmul(&hidden, w2).ok()?;
        let output = cuda.add(&output, b2).ok()?;

        // Readback
        let out_cpu = output.to_cpu_vec(&cuda.device).ok()?;
        let out_shape = vec![n, self.config.hidden_size];
        let out_flat: Vec<f32> = out_cpu;
        ndarray::Array2::from_shape_vec(out_shape, out_flat).ok()
    }

    /// CPU batched forward via Q4 quantized weights.
    /// Uses `matmul_int4` from `nexora_quantization::gemm` for compute directly from Q4.
    fn forward_batched_q4_cpu(&self, inputs: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>, String> {
        let (ref packed1, ref scales1) = self.fc1_q4.as_ref().ok_or_else(|| "fc1_q4 not available".to_string())?;
        let (ref packed2, ref scales2) = self.fc2_q4.as_ref().ok_or_else(|| "fc2_q4 not available".to_string())?;
        let gs = self.q4_group_size.ok_or_else(|| "q4_group_size not set".to_string())?;
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;
        let n = inputs.shape()[0];

        let input_flat = inputs.as_slice().ok_or_else(|| "input not contiguous".to_string())?;

        // fc1: [N × H] → [N × I]
        // A = inputs [N, H]; Q4 B has K=H, N=I
        let mut hidden = vec![0.0; n * i];
        nexora_quantization::gemm::matmul_int4(
            input_flat, packed1, scales1,
            &mut hidden,
            n, i, h, gs,
        );
        // Add fc1 bias (not quantized)
        if let Some(ref b1) = self.fc1_bias {
            for row in 0..n {
                for col in 0..i {
                    hidden[row * i + col] += b1[col];
                }
            }
        }

        // GELU activation
        let activated: Vec<f32> = hidden.iter().map(|&x| gelu(x)).collect();

        // fc2: [N × I] → [N × H]
        let mut output = vec![0.0; n * h];
        nexora_quantization::gemm::matmul_int4(
            &activated, packed2, scales2,
            &mut output,
            n, h, i, gs,
        );
        // Add fc2 bias (not quantized)
        if let Some(ref b2) = self.fc2_bias {
            for row in 0..n {
                for col in 0..h {
                    output[row * h + col] += b2[col];
                }
            }
        }

        ndarray::Array2::from_shape_vec((n, h), output)
            .map_err(|e| format!("Q4 fc2 output shape: {e}"))
    }

    /// Batched forward: processes N tokens, tries CUDA → wgpu → Q4 CPU → FP32 CPU fallback.
    /// Returns `Err` if weights are not initialized.
    pub fn forward_batched(&self, inputs: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>, String> {
        #[cfg(feature = "cuda")]
        if let Some(result) = self.forward_batched_cuda(inputs) {
            return Ok(result);
        }
        #[cfg(feature = "gpu")]
        if let Some(result) = self.forward_batched_gpu(inputs) {
            return Ok(result);
        }
        // Q4 CPU fallback (no GPU available, but Q4 weights exist)
        if self.fc1_q4.is_some() {
            return self.forward_batched_q4_cpu(inputs);
        }
        let (_n, h) = inputs.dim();
        let i_size = self.config.intermediate_size;

        let fc1_w = self.fc1_weights.as_ref().ok_or_else(|| "fc1_weights not initialized".to_string())?;
        let fc1_b = self.fc1_bias.as_ref().ok_or_else(|| "fc1_bias not initialized".to_string())?;
        let fc2_w = self.fc2_weights.as_ref().ok_or_else(|| "fc2_weights not initialized".to_string())?;
        let fc2_b = self.fc2_bias.as_ref().ok_or_else(|| "fc2_bias not initialized".to_string())?;

        // Build weight matrices for batched matmul
        let w1_flat: Vec<f32> = fc1_w.iter().flat_map(|r| r.iter()).copied().collect();
        let w1 = ndarray::Array2::from_shape_vec((i_size, h), w1_flat).map_err(|e| format!("fc1 shape: {e}"))?;
        let w2_flat: Vec<f32> = fc2_w.iter().flat_map(|r| r.iter()).copied().collect();
        let w2 = ndarray::Array2::from_shape_vec((h, i_size), w2_flat).map_err(|e| format!("fc2 shape: {e}"))?;
        let b1 = ndarray::Array1::from_vec(fc1_b.clone());
        let b2 = ndarray::Array1::from_vec(fc2_b.clone());

        // fc1: [N × H] @ [H × I] + [I] → [N × I]
        let hidden = inputs.dot(&w1.t()) + &b1;

        // GELU activation
        let activated = hidden.mapv(gelu);

        // Dropout in training
        let dropped = if self.config.use_dropout {
            let rate = self.config.dropout_rate;
            let scale = 1.0 / (1.0 - rate);
            activated.mapv(|x| {
                if rand::random::<f32>() < rate { 0.0 } else { x * scale }
            })
        } else {
            activated
        };

        // fc2: [N × I] @ [I × H] + [H] → [N × H]
        Ok(dropped.dot(&w2.t()) + b2)
    }

    /// First linear layer forward pass
    fn fc1_forward(&self, input: &[f32]) -> Vec<f32> {
        let (w, b) = match (self.get_fc1_weights(), self.get_fc1_bias()) {
            (Ok(w), Ok(b)) => (w, b),
            _ => {
                warn!("fc1_forward failed: weights or bias not initialized");
                return Vec::new();
            }
        };
        w.iter()
            .zip(b.iter())
            .map(|(weights, bias)| {
                let mut sum = *bias;
                for (w, x) in weights.iter().zip(input.iter()) {
                    sum += w * x;
                }
                sum
            })
            .collect()
    }

    /// Second linear layer forward pass
    fn fc2_forward(&self, input: &[f32]) -> Vec<f32> {
        let (w, b) = match (self.get_fc2_weights(), self.get_fc2_bias()) {
            (Ok(w), Ok(b)) => (w, b),
            _ => {
                warn!("fc2_forward failed: weights or bias not initialized");
                return Vec::new();
            }
        };
        w.iter()
            .zip(b.iter())
            .map(|(weights, bias)| {
                let mut sum = *bias;
                for (w, x) in weights.iter().zip(input.iter()) {
                    sum += w * x;
                }
                sum
            })
            .collect()
    }

    /// GELU activation function
    fn apply_gelu(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| gelu(x)).collect()
    }

    /// Apply dropout during training
    fn apply_dropout(&self, input: &[f32]) -> Vec<f32> {
        let rate = self.config.dropout_rate;
        let scale = 1.0 / (1.0 - rate);
        input
            .iter()
            .map(|&x| {
                if rand::random::<f32>() < rate {
                    0.0
                } else {
                    x * scale
                }
            })
            .collect()
    }

    /// Get expert configuration
    pub fn config(&self) -> &ExpertConfig {
        &self.config
    }

    /// Get expert size information
    pub fn size_info(&self) -> (usize, usize) {
        (self.config.hidden_size, self.config.intermediate_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_expert() -> Expert {
        let mut e = Expert::new(4, 8, false, 0.0);
        e.init_random();
        e
    }

    #[test]
    fn test_gelu_shape() {
        let x = gelu(0.0);
        assert!((x - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_gelu_positive() {
        let x = gelu(1.0);
        assert!(x > 0.8);
    }

    #[test]
    fn test_gelu_negative() {
        let x = gelu(-1.0);
        assert!(x < 0.0);
        assert!(x > -0.2);
    }

    #[test]
    fn test_new_expert_config() {
        let mut e = Expert::new(16, 32, true, 0.1);
        e.init_random();
        let (h, i) = e.size_info();
        assert_eq!(h, 16);
        assert_eq!(i, 32);
        let fc1 = e.get_fc1_weights().unwrap();
        assert_eq!(fc1.len(), 32);
        assert_eq!(fc1[0].len(), 16);
        let fc2 = e.get_fc2_weights().unwrap();
        assert_eq!(fc2.len(), 16);
        assert_eq!(fc2[0].len(), 32);
    }

    #[test]
    fn test_forward_output_dim() {
        let e = small_expert();
        let input = vec![0.5; 4];
        let output = e.forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_forward_deterministic_without_dropout() {
        let e = small_expert();
        let input = vec![1.0, 0.5, 0.0, -0.5];
        let out1 = e.forward(&input);
        let out2 = e.forward(&input);
        assert_eq!(out1.len(), out2.len());
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fc1_forward_output_dim() {
        let e = small_expert();
        let input = vec![0.5; 4];
        let hidden = e.fc1_forward(&input);
        assert_eq!(hidden.len(), 8);
    }

    #[test]
    fn test_fc2_forward_output_dim() {
        let e = small_expert();
        let input = vec![0.5; 8];
        let output = e.fc2_forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_apply_gelu() {
        let e = small_expert();
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let activated = e.apply_gelu(&input);
        assert_eq!(activated.len(), 4);
        assert!(activated[1].abs() < 1e-5);
        assert!(activated[0] < activated[2]);
    }

    #[test]
    fn test_apply_dropout_with_zero_rate() {
        let e = Expert::new(4, 8, true, 0.0);
        let input = vec![1.0; 4];
        let dropped = e.apply_dropout(&input);
        assert_eq!(dropped.len(), 4);
        for v in &dropped {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_apply_dropout_with_full_rate() {
        let e = Expert::new(4, 8, true, 1.0);
        let input = vec![1.0; 4];
        let dropped = e.apply_dropout(&input);
        for v in &dropped {
            assert!((v - 0.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_forward_zero_input() {
        let e = small_expert();
        let input = vec![0.0; 4];
        let output = e.forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_config() {
        let e = small_expert();
        let config = e.config();
        assert_eq!(config.hidden_size, 4);
        assert_eq!(config.intermediate_size, 8);
    }

    #[test]
    fn test_init_weights_dimensions() {
        let w = Expert::init_weights(10, 5);
        assert_eq!(w.len(), 5);
        assert_eq!(w[0].len(), 10);
    }

    #[test]
    fn test_quantize_to_q4_stores_data() {
        let mut e = Expert::new(16, 32, false, 0.0);
        e.init_random();
        e.quantize_to_q4(8); // small group_size for test
        let (packed1, scales1) = e.fc1_q4.as_ref().unwrap();
        let (packed2, scales2) = e.fc2_q4.as_ref().unwrap();
        // fc1: [intermediate=32, hidden=16] → K=16, N=32 → packed len = (16/2)*32 = 256
        assert_eq!(packed1.len(), 256);
        // groups = 16/8 = 2 → scales len = 2 * 32 = 64
        assert_eq!(scales1.len(), 64);
        // fc2: [hidden=16, intermediate=32] → K=32, N=16 → packed len = (32/2)*16 = 256
        assert_eq!(packed2.len(), 256);
        // groups = 32/8 = 4 → scales len = 4 * 16 = 64
        assert_eq!(scales2.len(), 64);
        assert_eq!(e.q4_group_size, Some(8));
    }

    #[test]
    fn test_quantize_q4_forward_cpu_matches() {
        let mut e = Expert::new(8, 16, false, 0.0);
        e.init_random();
        let input = ndarray::Array2::from_shape_vec((2, 8), (0..16).map(|v| v as f32).collect()).unwrap();
        // Reference: CPU forward via FP32
        let fp32_out = e.forward_batched(&input).unwrap();

        // Quantize (auto-drops FP32) → re-forward via Q4 CPU fallback
        e.quantize_to_q4(16);
        // With FP32 dropped, forward_batched should use Q4 path
        // (currently falls back to CPU matmul_int4 via quantized data)
        let q4_out = e.forward_batched(&input).unwrap();

        // Q4 introduces quantization error (~1/16 of value range per group).
        // For small random weights range [-2,2], error up to ±1 per element is acceptable.
        let max_err = fp32_out.iter()
            .zip(q4_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 2.0, "Q4 max error too large: {max_err}");
        // Verify outputs are not just zeros
        let q4_norm: f32 = q4_out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(q4_norm > 0.01, "Q4 output is all zeros");
    }

    #[test]
    fn test_drop_cpu_fp32_keep_q4() {
        let mut e = Expert::new(4, 8, false, 0.0);
        e.init_random();
        // quantize_to_q4 auto-drops FP32 weights now
        e.quantize_to_q4(4);
        assert!(e.fc1_weights.is_none());
        assert!(e.fc1_q4.is_some());
        // Manual drop is idempotent
        e.drop_cpu_fp32_keep_q4();
        assert!(e.fc1_weights.is_none());
        assert!(e.fc1_bias.is_some());  // biases retained for Q4 forward
        assert!(e.fc2_weights.is_none());
        assert!(e.fc2_bias.is_some());
        assert!(e.fc1_q4.is_some());
        assert!(e.fc2_q4.is_some());
    }
}
