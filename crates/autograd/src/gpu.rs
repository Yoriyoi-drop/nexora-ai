use std::collections::HashMap;
use once_cell::sync::OnceCell;
use ndarray::ArrayD;
use thiserror::Error;

use crate::gpu_caps::GpuCapabilities;

// ─── Errors ────────────────────────────────────────────────────────────────────

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("No GPU adapter found")]
    NoAdapter,
    #[error("GPU device error: {0}")]
    Device(String),
    #[error("Buffer creation failed: {0}")]
    Buffer(String),
    #[error("Compute shader error: {0}")]
    Shader(String),
    #[error("GPU not initialized. Call GpuContext::init() first")]
    NotInitialized,
    #[error("MatMul shape mismatch: a={0:?} b={1:?}")]
    MatMulShape(Vec<usize>, Vec<usize>),
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
    #[error("Pipeline compilation failed: {0}")]
    Pipeline(String),
    #[error("Element-wise shape mismatch: a={0:?} b={1:?}")]
    ElemShape(Vec<usize>, Vec<usize>),
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
}

// ─── Singleton Context ─────────────────────────────────────────────────────────

static GPU_CTX: OnceCell<GpuContext> = OnceCell::new();

// ─── GpuContext ────────────────────────────────────────────────────────────────

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub caps: GpuCapabilities,
    pipelines: HashMap<String, CompiledPipeline>,
}

struct CompiledPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// GPU adapter info for display
pub struct GpuAdapterInfo {
    pub name: String,
    pub backend: wgpu::Backend,
}

// ─── Element-wise op codes ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ElemOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Neg = 4,
    Exp = 5,
    Ln = 6,
    Powf = 7,
    Sqrt = 8,
    Relu = 9,
    Gelu = 10,
    Sigmoid = 11,
    Tanh = 12,
    Silu = 13,
}

// ─── Reduce op codes ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReduceOp {
    Sum = 0,
    Max = 1,
    Min = 2,
}

// ─── Binding helpers ────────────────────────────────────────────────────────────

fn storage_binding(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ─── Implementation ────────────────────────────────────────────────────────────

impl GpuContext {
    // ── Pipeline compilation helper ────────────────────────────────────────────

    fn compile_pipeline(
        &mut self,
        name: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
        wgsl: std::borrow::Cow<'static, str>,
        entry_point: &str,
    ) -> Result<(), GpuError> {
        let layout = self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{}_layout", name)),
                entries,
            },
        );

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{}_shader", name)),
                source: wgpu::ShaderSource::Wgsl(wgsl),
            });

        let pipeline_layout = self.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}_pipeline_layout", name)),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            },
        );

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{}_pipeline", name)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        self.pipelines.insert(
            name.into(),
            CompiledPipeline { pipeline, bind_group_layout: layout },
        );
        Ok(())
    }

    // ── Initialization ────────────────────────────────────────────────────────

    fn new_blocking() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }

    async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Nexora GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        let caps = GpuCapabilities::detect(&device, &adapter);
        let mut ctx = Self {
            device,
            queue,
            caps,
            pipelines: HashMap::new(),
        };

        ctx.compile_all_pipelines()?;

        Ok(ctx)
    }

    fn compile_all_pipelines(&mut self) -> Result<(), GpuError> {
        let tile = self.caps.optimal_tile_size();
        self.compile_matmul_tiled(tile)?;
        self.compile_elementwise()?;
        self.compile_reduce(ReduceOp::Sum)?;
        self.compile_reduce(ReduceOp::Max)?;
        self.compile_reduce(ReduceOp::Min)?;
        self.compile_softmax()?;
        self.compile_rms_norm()?;
        self.compile_cross_entropy()?;
        self.compile_embedding()?;
        self.compile_layer_norm()?;
        self.compile_transpose()?;
        self.compile_fused_attention()?;
        Ok(())
    }

    // ── Singleton access ──────────────────────────────────────────────────────

    pub fn init() -> Result<&'static Self, GpuError> {
        GPU_CTX.get_or_try_init(Self::new_blocking)
    }

    pub fn global() -> Result<&'static Self, GpuError> {
        GPU_CTX.get().ok_or(GpuError::NotInitialized)
    }

    pub fn is_available() -> bool {
        GPU_CTX.get().is_some()
    }

    pub fn adapter_info(&self) -> GpuAdapterInfo {
        GpuAdapterInfo {
            name: self.caps.device_name.clone(),
            backend: self.caps.backend,
        }
    }

    fn dispatch(
        &self,
        pipeline: &CompiledPipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&pipeline.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.1: TILED MATMUL
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_matmul_tiled(&mut self, tile: u32) -> Result<(), GpuError> {
        let wgsl = std::borrow::Cow::Owned(
            MATMUL_TILED_WGSL.replace("{{TILE_SIZE}}", &tile.to_string())
        );
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

    pub fn matmul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }
        if a_shape[1] != b_shape[0] {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }

        let m = a_shape[0] as u32;
        let k = a_shape[1] as u32;
        let n = b_shape[1] as u32;
        let tile = self.caps.optimal_tile_size();

        let c_size = (m as u64) * (n as u64) * 4;
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_output"),
            size: c_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dims_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_dims"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dims_data: [u32; 4] = [m, k, n, tile];
        self.queue
            .write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims_data));

        let pipeline = self.pipelines.get("matmul_tiled").ok_or_else(|| {
            GpuError::Pipeline("matmul_tiled not compiled".into())
        })?;

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

        let wgx = (m + tile - 1) / tile;
        let wgy = (n + tile - 1) / tile;
        self.dispatch(pipeline, &bind_group, (wgx, wgy, 1));

        Ok(GpuTensor {
            shape: vec![a_shape[0], b_shape[1]],
            buffer: c_buffer,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.3: ELEMENT-WISE + ACTIVATION
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_elementwise(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "elementwise",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            std::borrow::Cow::Borrowed(ELEMENTWISE_WGSL),
            "elementwise_main",
        )
    }

    /// Dispatch element-wise/unary op: output = op(a)
    pub fn elementwise_unary(
        &self,
        a: &GpuTensor,
        op: ElemOp,
    ) -> Result<GpuTensor, GpuError> {
        let shape = a.shape();
        let numel = a.numel();

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("elementwise_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("elementwise_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [numel as u32, op as u32, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self.pipelines.get("elementwise").ok_or_else(|| {
            GpuError::Pipeline("elementwise not compiled".into())
        })?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("elementwise_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.buffer().as_entire_binding(),
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

        let wg = (numel as u32 + 255) / 256;
        self.dispatch(pipeline, &bind_group, (wg, 1, 1));

        Ok(GpuTensor {
            shape: shape.to_vec(),
            buffer: out_buffer,
        })
    }

    /// Dispatch element-wise/binary op: output = op(a, b)
    pub fn elementwise_binary(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        op: ElemOp,
    ) -> Result<GpuTensor, GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape != b_shape {
            return Err(GpuError::ElemShape(a_shape, b_shape));
        }
        let numel = a.numel();

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("elementwise_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("elementwise_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [numel as u32, op as u32, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self.pipelines.get("elementwise").ok_or_else(|| {
            GpuError::Pipeline("elementwise not compiled".into())
        })?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("elementwise_binary_bind_group"),
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
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        let wg = (numel as u32 + 255) / 256;
        self.dispatch(pipeline, &bind_group, (wg, 1, 1));

        Ok(GpuTensor {
            shape: a_shape.to_vec(),
            buffer: out_buffer,
        })
    }

    /// Convenience: element-wise add on GPU
    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_binary(a, b, ElemOp::Add)
    }

    /// Convenience: element-wise sub on GPU
    pub fn sub(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_binary(a, b, ElemOp::Sub)
    }

    /// Convenience: element-wise mul on GPU
    pub fn mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_binary(a, b, ElemOp::Mul)
    }

    /// Convenience: element-wise div on GPU
    pub fn div(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_binary(a, b, ElemOp::Div)
    }

    /// Convenience: exp on GPU
    pub fn exp(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Exp)
    }

    /// Convenience: relu on GPU
    pub fn relu(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Relu)
    }

    /// Convenience: sigmoid on GPU
    pub fn sigmoid(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Sigmoid)
    }

    /// Convenience: silu on GPU
    pub fn silu(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Silu)
    }

    /// Convenience: gelu on GPU
    pub fn gelu(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Gelu)
    }

    /// Convenience: tanh on GPU
    pub fn tanh(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Tanh)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.2: REDUCE KERNELS
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_reduce(&mut self, op: ReduceOp) -> Result<(), GpuError> {
        let wgsl = std::borrow::Cow::Owned(
            REDUCE_WGSL_TEMPLATE.replace("{{OP}}", &(op as u32).to_string())
        );
        let key = match op {
            ReduceOp::Sum => "reduce_sum",
            ReduceOp::Max => "reduce_max",
            ReduceOp::Min => "reduce_min",
        };
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
            });
        }

        let key = match op {
            ReduceOp::Sum => "reduce_sum",
            ReduceOp::Max => "reduce_max",
            ReduceOp::Min => "reduce_min",
        };

        let pipeline = self.pipelines.get(key).ok_or_else(|| {
            GpuError::Pipeline(format!("{} not compiled", key))
        })?;

        // Multi-pass: each workgroup reduces 256 elements → partial results
        // Then reduce partials until scalar remains
        let mut current_buffer = input.buffer().clone();
        let mut current_numel = numel;

        loop {
            let num_groups = (current_numel + 255) / 256;

            let out_size = (num_groups * 4) as u64;
            let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduce_partial"),
                size: out_size.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduce_cfg"),
                size: 8,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let cfg_data: [u32; 2] = [current_numel as u32, num_groups as u32];
            self.queue
                .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            });

            self.dispatch(pipeline, &bind_group, (num_groups as u32, 1, 1));

            if num_groups == 1 {
                return Ok(GpuTensor {
                    shape: vec![1],
                    buffer: out_buffer,
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

    fn compile_softmax(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "softmax",
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            std::borrow::Cow::Borrowed(SOFTMAX_WGSL),
            "softmax_main",
        )
    }

    /// Softmax along last axis. Input shape: [rows, dim].
    pub fn softmax(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch("Softmax GPU: input must be 2D".into()));
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
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self.pipelines.get("softmax").ok_or_else(|| {
            GpuError::Pipeline("softmax not compiled".into())
        })?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cfg_buf.as_entire_binding() },
            ],
        });

        self.dispatch(pipeline, &bind_group, (rows, 1, 1));

        Ok(GpuTensor { shape: shape.to_vec(), buffer: out_buffer })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.2: RMSNORM
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_rms_norm(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "rms_norm",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            std::borrow::Cow::Borrowed(RMSNORM_WGSL),
            "rms_norm_main",
        )
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
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self.pipelines.get("rms_norm").ok_or_else(|| {
            GpuError::Pipeline("rms_norm not compiled".into())
        })?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms_norm_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cfg_buf.as_entire_binding() },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        Ok(GpuTensor { shape: shape.to_vec(), buffer: out_buffer })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.3: CROSS-ENTROPY LOSS
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_cross_entropy(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "cross_entropy",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            std::borrow::Cow::Borrowed(CROSS_ENTROPY_WGSL),
            "cross_entropy_main",
        )
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
            return Err(GpuError::ShapeMismatch("CrossEntropy GPU: logits must be 2D".into()));
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
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self.pipelines.get("cross_entropy").ok_or_else(|| {
            GpuError::Pipeline("cross_entropy not compiled".into())
        })?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cross_entropy_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: logits.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: targets.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cfg_buf.as_entire_binding() },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        Ok(GpuTensor { shape: vec![batch as usize], buffer: out_buffer })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.4: EMBEDDING
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_embedding(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "embedding",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            std::borrow::Cow::Borrowed(EMBEDDING_WGSL),
            "embedding_main",
        )
    }

    /// Embedding lookup: out[i * dim + d] = weight[tokens[i] * dim + d].
    /// ids: [seq_len], weight: [vocab, dim].
    pub fn embedding(
        &self,
        ids: &GpuTensor,
        weight: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        let w_shape = weight.shape();
        if w_shape.len() != 2 {
            return Err(GpuError::ShapeMismatch("Embedding GPU: weight must be 2D [vocab, dim]".into()));
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
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self.pipelines.get("embedding").ok_or_else(|| {
            GpuError::Pipeline("embedding not compiled".into())
        })?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embedding_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ids.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cfg_buf.as_entire_binding() },
            ],
        });

        self.dispatch(pipeline, &bind_group, ((seq_len * dim + 255) / 256, 1, 1));

        Ok(GpuTensor { shape: vec![seq_len as usize, dim as usize], buffer: out_buffer })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.5: LAYERNORM
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_layer_norm(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "layer_norm",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, true),
                storage_binding(3, false),
                uniform_binding(4),
            ],
            std::borrow::Cow::Borrowed(LAYERNORM_WGSL),
            "layer_norm_main",
        )
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
            return Err(GpuError::ShapeMismatch("LayerNorm GPU: x must be 2D".into()));
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
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self.pipelines.get("layer_norm").ok_or_else(|| {
            GpuError::Pipeline("layer_norm not compiled".into())
        })?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bias.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cfg_buf.as_entire_binding() },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        Ok(GpuTensor { shape: vec![batch as usize, dim as usize], buffer: out_buffer })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.6: TRANSPOSE + MATMUL BACKWARD
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_transpose(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "transpose",
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            std::borrow::Cow::Borrowed(TRANSPOSE_WGSL),
            "transpose_main",
        )
    }

    /// Transpose a 2D tensor [rows, cols] → [cols, rows]
    pub fn transpose(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch("Transpose GPU: input must be 2D".into()));
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
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self.pipelines.get("transpose").ok_or_else(|| {
            GpuError::Pipeline("transpose not compiled".into())
        })?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transpose_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cfg_buf.as_entire_binding() },
            ],
        });

        self.dispatch(pipeline, &bind_group, ((numel as u32 + 255) / 256, 1, 1));

        Ok(GpuTensor { shape: vec![cols as usize, rows as usize], buffer: out_buffer })
    }

    /// Matmul backward: grad_a = grad_out @ b^T, grad_b = a^T @ grad_out
    pub fn matmul_backward(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        grad_out: &GpuTensor,
    ) -> Result<(GpuTensor, GpuTensor), GpuError> {
        let b_t = self.transpose(b)?;
        let grad_a = self.matmul(&grad_out, &b_t)?;

        let a_t = self.transpose(a)?;
        let grad_b = self.matmul(&a_t, &grad_out)?;

        Ok((grad_a, grad_b))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 3: FUSED ATTENTION (Flash Attention-style)
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_fused_attention(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "fused_attention",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, true),
                storage_binding(3, false),
                uniform_binding(4),
                uniform_binding(5),
            ],
            std::borrow::Cow::Borrowed(FUSED_ATTENTION_WGSL),
            "fused_attention_main",
        )
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
            return Err(GpuError::ShapeMismatch("Fused Attention: Q must be 4D [B,H,S,D]".into()));
        }
        let batch = q_shape[0] as u32;
        let heads = q_shape[1] as u32;
        let seq_len = q_shape[2] as u32;
        let dim = q_shape[3] as u32;
        let numel = q.numel();

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
        self.queue.write_buffer(&cfg1_buf, 0, bytemuck::cast_slice(&cfg1));

        let causal_flag: u32 = if causal { 1 } else { 0 };
        let cfg2_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_cfg2"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg2: [u32; 4] = [f32::to_bits(scale), causal_flag, 0, 0];
        self.queue.write_buffer(&cfg2_buf, 0, bytemuck::cast_slice(&cfg2));

        let pipeline = self.pipelines.get("fused_attention").ok_or_else(|| {
            GpuError::Pipeline("fused_attention not compiled".into())
        })?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_attention_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cfg1_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cfg2_buf.as_entire_binding() },
            ],
        });

        let num_wg = batch * heads * seq_len;
        self.dispatch(pipeline, &bind_group, (num_wg, 1, 1));

        Ok(GpuTensor { shape: q_shape.clone(), buffer: out_buffer })
    }
}

const MATMUL_TILED_WGSL: &str = r#"
const TILE_SIZE: u32 = {{TILE_SIZE}};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    Tile: u32,
};

@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_tiled_main(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>,
                     @builtin(workgroup_id) wg_id: vec3<u32>) {
    let row = wg_id.x * TILE_SIZE + lid.x;
    let col = wg_id.y * TILE_SIZE + lid.y;

    var sum = 0.0;
    let num_tiles = (dims.K + TILE_SIZE - 1) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load tile of A
        let a_row = row;
        let a_col = t * TILE_SIZE + lid.y;
        if (a_row < dims.M && a_col < dims.K) {
            tile_a[lid.x][lid.y] = a[a_row * dims.K + a_col];
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        // Load tile of B
        let b_row = t * TILE_SIZE + lid.x;
        let b_col = col;
        if (b_row < dims.K && b_col < dims.N) {
            tile_b[lid.x][lid.y] = b[b_row * dims.N + b_col];
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        // Accumulate
        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        c[row * dims.N + col] = sum;
    }
}
"#;

// ── Phase 1.3: Element-wise + Activation ──────────────────────────────────────

const ELEMENTWISE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Cfg {
    numel: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(3) var<uniform> cfg: Cfg;

fn gelu_f32(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

fn sigmoid_f32(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn elementwise_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.numel) {
        return;
    }

    let a_val = a[i];
    // For binary ops, b[i] is used; for unary, b[i] is same as a[i] or ignored
    let b_val = b[i];

    switch cfg.op {
        case 0: { // Add
            out[i] = a_val + b_val;
        }
        case 1: { // Sub
            out[i] = a_val - b_val;
        }
        case 2: { // Mul
            out[i] = a_val * b_val;
        }
        case 3: { // Div
            out[i] = a_val / b_val;
        }
        case 4: { // Neg
            out[i] = -a_val;
        }
        case 5: { // Exp
            out[i] = exp(a_val);
        }
        case 6: { // Ln
            out[i] = log(a_val);
        }
        case 7: { // Powf
            out[i] = pow(a_val, b_val);
        }
        case 8: { // Sqrt
            out[i] = sqrt(a_val);
        }
        case 9: { // Relu
            out[i] = max(a_val, 0.0);
        }
        case 10: { // Gelu
            out[i] = gelu_f32(a_val);
        }
        case 11: { // Sigmoid
            out[i] = sigmoid_f32(a_val);
        }
        case 12: { // Tanh
            out[i] = tanh(a_val);
        }
        case 13: { // Silu
            out[i] = a_val * sigmoid_f32(a_val);
        }
        default: {
            out[i] = a_val;
        }
    }
}
"#;

// ── Phase 1.2: Reduce ─────────────────────────────────────────────────────────

const REDUCE_WGSL_TEMPLATE: &str = r#"
const BLOCK_SIZE: u32 = 256;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Cfg {
    numel: u32,
    num_groups: u32,
};

@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> shared: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn reduce_main(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wg_id: vec3<u32>) {
    let group_idx = wg_id.x;
    let items_per_group = (cfg.numel + cfg.num_groups - 1) / cfg.num_groups;

    // Load
    var val: f32;
    let base = group_idx * items_per_group;
    let idx = base + lid.x;
    if (idx < cfg.numel) {
        val = input[idx];
    } else {
        val = 0.0;
    }
    shared[lid.x] = val;
    workgroupBarrier();

    // Tree-reduce
    var stride = BLOCK_SIZE / 2;
    while (stride > 0u) {
        if (lid.x < stride) {
            {{REDUCE_BODY}}
        }
        workgroupBarrier();
        stride = stride / 2;
    }

    // Write result
    if (lid.x == 0u) {
        output[group_idx] = shared[0];
    }
}
"#;

// ── Phase 2.1: Softmax (stable, per-row) ──────────────────────────────────────

const SOFTMAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn softmax_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let batch = cfg.x;
    let dim = cfg.y;
    let row = wg.x;
    if (row >= batch) { return; }

    let base = row * dim;

    // shared memory for tree-reduce
    var shared: array<f32, BLOCK_SIZE>;

    // === Pass 1: find row max ===
    var mx: f32 = -3.402823e+38;
    var i = lid.x;
    while (i < dim) {
        let v = input[base + i];
        if (v > mx) { mx = v; }
        i += BLOCK_SIZE;
    }
    shared[lid.x] = mx;
    workgroupBarrier();

    // tree-reduce max
    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            if (shared[lid.x] < shared[lid.x + stride]) {
                shared[lid.x] = shared[lid.x + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = shared[0u];
    workgroupBarrier();

    // === Pass 2: exp(x - max) and sum ===
    var sum: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        let e = exp(input[base + i] - row_max);
        output[base + i] = e;
        sum += e;
        i += BLOCK_SIZE;
    }
    shared[lid.x] = sum;
    workgroupBarrier();

    // tree-reduce sum
    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            shared[lid.x] = shared[lid.x] + shared[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = shared[0u];
    workgroupBarrier();

    // === Pass 3: normalize ===
    i = lid.x;
    while (i < dim) {
        output[base + i] = output[base + i] / row_sum;
        i += BLOCK_SIZE;
    }
}
"#;

// ── Phase 2.2: RMSNorm ───────────────────────────────────────────────────────

const RMSNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn rms_norm_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let dim = cfg.y;
    let eps = bitcast<f32>(cfg.z);
    let row = wg.x;

    let base = row * dim;

    // === Pass 1: sum(x²) ===
    var shared: array<f32, BLOCK_SIZE>;
    var ss: f32 = 0.0;
    var i = lid.x;
    while (i < dim) {
        let v = input[base + i];
        ss += v * v;
        i += BLOCK_SIZE;
    }
    shared[lid.x] = ss;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            shared[lid.x] = shared[lid.x] + shared[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let rms = sqrt(shared[0u] / f32(dim) + eps);
    workgroupBarrier();

    // === Pass 2: normalize ===
    i = lid.x;
    while (i < dim) {
        output[base + i] = (input[base + i] / rms) * weight[i];
        i += BLOCK_SIZE;
    }
}
"#;

// ── Phase 2.3: Cross-Entropy ─────────────────────────────────────────────────

const CROSS_ENTROPY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn cross_entropy_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let batch = cfg.x;
    let num_classes = cfg.y;
    let row = wg.x;
    if (row >= batch) { return; }

    let base = row * num_classes;
    let target = targets[row];

    // === Pass 1: find row max (stable) ===
    var shared: array<f32, BLOCK_SIZE>;
    var mx: f32 = -3.402823e+38;
    var i = lid.x;
    while (i < num_classes) {
        let v = logits[base + i];
        if (v > mx) { mx = v; }
        i += BLOCK_SIZE;
    }
    shared[lid.x] = mx;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            if (shared[lid.x] < shared[lid.x + stride]) {
                shared[lid.x] = shared[lid.x + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = shared[0u];
    workgroupBarrier();

    // === Pass 2: sum(exp(x - max)) ===
    var sum: f32 = 0.0;
    i = lid.x;
    while (i < num_classes) {
        sum += exp(logits[base + i] - row_max);
        i += BLOCK_SIZE;
    }
    shared[lid.x] = sum;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            shared[lid.x] = shared[lid.x] + shared[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let log_sum_exp = log(shared[0u]) + row_max;
    workgroupBarrier();

    // === Pass 3: loss = log_sum_exp - logits[target] ===
    if (lid.x == 0u) {
        let target_logit = logits[base + target];
        losses[row] = log_sum_exp - target_logit;
    }
}
"#;

// ── Phase 2.4: Embedding (gather) ────────────────────────────────────────────

const EMBEDDING_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn embedding_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let vocab_size = cfg.x;
    let dim = cfg.y;
    let seq_len = cfg.z;

    let idx = gid.x;
    if (idx >= seq_len * dim) { return; }

    let d = idx % dim;
    let s = idx / dim;
    let token_id = ids[s];
    output[idx] = weight[token_id * dim + d];
}
"#;

// ── Phase 2.5: LayerNorm ────────────────────────────────────────────────────

const LAYERNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn layer_norm_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let dim = cfg.y;
    let eps = bitcast<f32>(cfg.z);
    let row = wg.x;

    let base = row * dim;
    var shared: array<f32, BLOCK_SIZE>;

    // === Pass 1: mean ===
    var sum: f32 = 0.0;
    var i = lid.x;
    while (i < dim) {
        sum += input[base + i];
        i += BLOCK_SIZE;
    }
    shared[lid.x] = sum;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            shared[lid.x] = shared[lid.x] + shared[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean = shared[0u] / f32(dim);
    workgroupBarrier();

    // === Pass 2: variance ===
    var var_sum: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        let diff = input[base + i] - mean;
        var_sum += diff * diff;
        i += BLOCK_SIZE;
    }
    shared[lid.x] = var_sum;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            shared[lid.x] = shared[lid.x] + shared[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let variance = shared[0u] / f32(dim);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // === Pass 3: normalize ===
    i = lid.x;
    while (i < dim) {
        let normalized = (input[base + i] - mean) * inv_std;
        output[base + i] = normalized * weight[i] + bias[i];
        i += BLOCK_SIZE;
    }
}
"#;

// ── Phase 2.6: Transpose ────────────────────────────────────────────────────

const TRANSPOSE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn transpose_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = cfg.x;
    let cols = cfg.y;
    let idx = gid.x;
    if (idx >= rows * cols) { return; }
    let r = idx / cols;
    let c = idx % cols;
    output[c * rows + r] = input[r * cols + c];
}
"#;

// ── Phase 3: Fused Attention (Flash Attention-style) ─────────────────────────

const FUSED_ATTENTION_WGSL: &str = r#"
const BLOCK_SIZE = 256u;
const TILE_SIZE = 32u;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> cfg1: vec4<u32>;
@group(0) @binding(5) var<uniform> cfg2: vec4<u32>;

var<workgroup> q_shared: array<f32, BLOCK_SIZE>;
var<workgroup> kv_shared: array<f32, BLOCK_SIZE>;
var<workgroup> score_shared: array<f32, TILE_SIZE>;
var<workgroup> exp_shared: array<f32, TILE_SIZE>;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn fused_attention_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch = cfg1.x;
    let heads = cfg1.y;
    let seq_len = cfg1.z;
    let dim = cfg1.w;
    let scale = bitcast<f32>(cfg2.x);
    let causal = cfg2.y;

    let tid = lid.x;

    let wg_flat = wg_id.x;
    let q_pos = wg_flat % seq_len;
    let head = (wg_flat / seq_len) % heads;
    let batch_idx = wg_flat / (seq_len * heads);

    if (batch_idx >= batch) { return; }
    if (tid >= dim) { return; }

    let head_stride = seq_len * dim;
    let batch_stride = heads * head_stride;
    let q_off = batch_idx * batch_stride + head * head_stride + q_pos * dim;
    let kv_base = batch_idx * batch_stride + head * head_stride;

    // Load Q row
    q_shared[tid] = q[q_off + tid];
    workgroupBarrier();

    // Online softmax state
    var m: f32 = -3.402823e+38;
    var d: f32 = 0.0;
    var o: f32 = 0.0;

    // Loop over KV tiles
    var tile_start: u32 = 0u;
    while (tile_start < seq_len) {
        let tile_end = min(tile_start + TILE_SIZE, seq_len);
        let tile_size = tile_end - tile_start;

        // === Step 1: compute scores for this tile ===
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;

            kv_shared[tid] = k[kv_base + k_pos * dim + tid];
            workgroupBarrier();

            var partial = q_shared[tid] * kv_shared[tid];
            scratch[tid] = partial;
            workgroupBarrier();

            var stride = BLOCK_SIZE / 2u;
            while (stride > 0u) {
                if (tid < stride) {
                    scratch[tid] = scratch[tid] + scratch[tid + stride];
                }
                workgroupBarrier();
                stride = stride / 2u;
            }

            if (tid == 0u) {
                var s = scratch[0u] / scale;
                if (causal == 1u && k_pos > q_pos) {
                    s = -3.402823e+38;
                }
                score_shared[ki] = s;
            }
            workgroupBarrier();
        }

        // === Step 2: max of tile scores ===
        if (tid < tile_size) {
            scratch[tid] = score_shared[tid];
        } else {
            scratch[tid] = -3.402823e+38;
        }
        workgroupBarrier();

        var stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) {
                if (scratch[tid] < scratch[tid + stride]) {
                    scratch[tid] = scratch[tid + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let m_tile = scratch[0u];
        workgroupBarrier();

        // === Step 3: exp(score - m_tile) and sum ===
        if (tid < tile_size) {
            let e = exp(score_shared[tid] - m_tile);
            exp_shared[tid] = e;
            scratch[tid] = e;
        } else {
            scratch[tid] = 0.0;
        }
        workgroupBarrier();

        stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) {
                scratch[tid] = scratch[tid] + scratch[tid + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let sum_exp_tile = scratch[0u];
        workgroupBarrier();

        // === Step 4: Online softmax update ===
        let m_new = max(m, m_tile);
        let old_scale = exp(m - m_new);
        let tile_scale = exp(m_tile - m_new);

        d = old_scale * d + tile_scale * sum_exp_tile;
        o = o * old_scale;

        // === Step 5: Accumulate V ===
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;
            kv_shared[tid] = v[kv_base + k_pos * dim + tid];
            workgroupBarrier();

            o += tile_scale * exp_shared[ki] * kv_shared[tid];
            workgroupBarrier();
        }

        m = m_new;
        tile_start += TILE_SIZE;
    }

    // === Final: output = o / d ===
    output[q_off + tid] = o / d;
}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
//  GpuTensor
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct GpuTensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) buffer: wgpu::Buffer,
}

unsafe impl Send for GpuTensor {}
unsafe impl Sync for GpuTensor {}

impl GpuTensor {
    fn ctx() -> Result<&'static GpuContext, GpuError> {
        GpuContext::global()
    }

    pub fn from_cpu(data: &ArrayD<f32>) -> Result<Self, GpuError> {
        let ctx = Self::ctx()?;
        let shape = data.shape().to_vec();
        let flat: &[u8] = bytemuck::cast_slice(data.as_slice().ok_or_else(|| {
            GpuError::Buffer("non-contiguous array".into())
        })?);

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::from_cpu"),
            size: flat.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        ctx.queue.write_buffer(&buffer, 0, flat);
        Ok(Self { shape, buffer })
    }

    pub fn zeros(shape: &[usize]) -> Result<Self, GpuError> {
        let ctx = Self::ctx()?;
        let numel: usize = shape.iter().product();
        let byte_size = (numel * 4) as u64;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::zeros"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::zeros_staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });
        {
            let mut view = staging.slice(..).get_mapped_range_mut();
            view.copy_from_slice(&vec![0u8; byte_size as usize]);
        }
        staging.unmap();

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&staging, 0, &buffer, 0, byte_size);
        ctx.queue.submit(Some(encoder.finish()));

        Ok(Self {
            shape: shape.to_vec(),
            buffer,
        })
    }

    pub fn to_cpu(&self) -> ArrayD<f32> {
        let ctx = Self::ctx().expect("GPU context not initialized");
        let byte_size = (self.numel() * 4) as u64;

        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::to_cpu_staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_size);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        ctx.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .expect("channel closed")
            .expect("buffer mapping failed");

        let mapped = slice.get_mapped_range();
        let data: &[f32] = bytemuck::cast_slice(&*mapped);
        let result = ArrayD::from_shape_vec(self.shape.clone(), data.to_vec())
            .expect("shape mismatch in GPU→CPU transfer");
        staging.unmap();
        result
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn device_id(&self) -> usize {
        0
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}
