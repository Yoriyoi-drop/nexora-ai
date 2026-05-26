use crate::gpu::{GpuContext, GpuTensor};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub fn compile_nan_check_pipeline(
    ctx: &mut GpuContext,
) -> Result<wgpu::ComputePipeline, crate::gpu::GpuError> {
    ctx.compile_pipeline_cached(
        "nan_check",
        &[
            crate::gpu::storage_binding(0, true),
            crate::gpu::storage_binding(1, false),
        ],
        std::borrow::Cow::Borrowed(NAN_CHECK_WGSL),
        "main",
    )
}

pub fn compile_nan_check_reduce_pipeline(
    ctx: &mut GpuContext,
) -> Result<wgpu::ComputePipeline, crate::gpu::GpuError> {
    ctx.compile_pipeline_cached(
        "nan_check_reduce",
        &[
            crate::gpu::storage_binding(0, true),
            crate::gpu::storage_binding(1, false),
            crate::gpu::uniform_binding(2),
        ],
        std::borrow::Cow::Borrowed(NAN_CHECK_REDUCE_WGSL),
        "main",
    )
}

pub fn compile_nan_check_final_pipeline(
    ctx: &mut GpuContext,
) -> Result<wgpu::ComputePipeline, crate::gpu::GpuError> {
    ctx.compile_pipeline_cached(
        "nan_check_final",
        &[crate::gpu::storage_binding(0, false)],
        std::borrow::Cow::Borrowed(NAN_CHECK_FINAL_WGSL),
        "main",
    )
}

pub fn has_nan_or_inf_gpu(
    ctx: &GpuContext,
    tensor: &GpuTensor,
    pipelines: &(
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
    ),
) -> bool {
    let (flag_pipeline, reduce_pipeline, final_pipeline) = pipelines;

    let n = tensor.numel() as u32;
    if n == 0 {
        return false;
    }

    let flag_workgroups = ((n + 63) / 64).max(1);
    let flag_size = (flag_workgroups * 4) as u64;
    let flag_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nan_flag_buf"),
        size: flag_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nan_check_bg"),
        layout: &flag_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tensor.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: flag_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nan_check_encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nan_check"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(flag_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(flag_workgroups, 1, 1);
    }

    if flag_workgroups > 1 {
        let mut remaining = flag_workgroups;
        let mut src_buf = flag_buf;
        let mut reduce_round = 0;

        while remaining > 1 {
            let dst_workgroups = ((remaining + 63) / 64).max(1);
            let dst_size = (dst_workgroups * 4).max(4) as u64;
            let dst_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("nan_reduce_buf_{}", reduce_round)),
                size: dst_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let n_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nan_n_buf"),
                size: 4,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue
                .write_buffer(&n_buf, 0, bytemuck::bytes_of(&remaining));

            let reduce_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nan_reduce_bg"),
                layout: &reduce_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: n_buf.as_entire_binding(),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("nan_reduce_{}", reduce_round)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(reduce_pipeline);
                cpass.set_bind_group(0, &reduce_bg, &[]);
                cpass.dispatch_workgroups(dst_workgroups, 1, 1);
            }

            src_buf = dst_buf;
            remaining = dst_workgroups;
            reduce_round += 1;
        }

        let final_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nan_final_bg"),
            layout: &final_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buf.as_entire_binding(),
            }],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nan_final"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(final_pipeline);
            cpass.set_bind_group(0, &final_bg, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        {
            let result_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nan_final_result"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder2 = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("nan_final_copy"),
                });
            encoder2.copy_buffer_to_buffer(&src_buf, 0, &result_buf, 0, 4);

            let _ = std::mem::replace(&mut encoder, encoder2);

            ctx.queue.submit(Some(encoder.finish()));

            let slice = result_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            ctx.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            let data = slice.get_mapped_range();
            let flag = data[0] != 0;
            drop(data);
            result_buf.unmap();
            return flag;
        }
    }

    {
        let final_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nan_final_bg"),
            layout: &final_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: flag_buf.as_entire_binding(),
            }],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nan_final"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(final_pipeline);
            cpass.set_bind_group(0, &final_bg, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        let result_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nan_result"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&flag_buf, 0, &result_buf, 0, 4);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = result_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice.get_mapped_range();
        let flag = data[0] != 0;
        drop(data);
        result_buf.unmap();
        return flag;
    }
}

pub struct GpuNanDetector {
    pub enabled: bool,
    pub abort_on_nan: bool,
    pipelines: Option<(
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
    )>,
    nan_detected: Arc<AtomicBool>,
}

impl GpuNanDetector {
    pub fn new(enabled: bool, abort_on_nan: bool) -> Self {
        Self {
            enabled,
            abort_on_nan,
            pipelines: None,
            nan_detected: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn ensure_pipelines(&mut self, ctx: &mut GpuContext) -> Result<(), crate::gpu::GpuError> {
        if self.pipelines.is_some() {
            return Ok(());
        }
        let pipelines = (
            compile_nan_check_pipeline(ctx)?,
            compile_nan_check_reduce_pipeline(ctx)?,
            compile_nan_check_final_pipeline(ctx)?,
        );
        self.pipelines = Some(pipelines);
        Ok(())
    }

    pub fn check(
        &mut self,
        ctx: &mut GpuContext,
        tensor: &GpuTensor,
        label: &str,
    ) -> Result<bool, crate::gpu::GpuError> {
        if !self.enabled {
            return Ok(false);
        }
        self.ensure_pipelines(ctx)?;
        let Some(pipelines) = self.pipelines.as_ref() else {
            return Ok(false);
        };
        let has_nan = has_nan_or_inf_gpu(ctx, tensor, pipelines);
        if has_nan {
            self.nan_detected.store(true, Ordering::SeqCst);
            tracing::warn!(
                "[GPU NaN DETECTED] in '{}' (size={}, shape={:?})",
                label,
                tensor.numel(),
                tensor.shape()
            );
            if self.abort_on_nan {
                return Err(crate::gpu::GpuError::Unsupported(format!(
                    "GPU NaN detected in '{}' — aborting",
                    label
                )));
            }
        }
        Ok(has_nan)
    }

    pub fn was_nan_detected(&self) -> bool {
        self.nan_detected.load(Ordering::SeqCst)
    }

    pub fn reset(&self) {
        self.nan_detected.store(false, Ordering::SeqCst);
    }
}

const NAN_CHECK_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&input);
    if i >= n { return; }
    let v = input[i];
    var flag: u32 = 0u;
    if isNan(v) || isInf(v) {
        flag = 1u;
    }
    flags[id.x] = flag;
}
";

const NAN_CHECK_REDUCE_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> n_in: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = n_in;
    let workgroup_n = (n + 63u) / 64u;
    if i >= workgroup_n { return; }

    var acc: u32 = 0u;
    let base = i * 64u;
    for (var j: u32 = 0u; j < 64u && base + j < n; j = j + 1u) {
        acc = acc | input[base + j];
    }
    output[i] = acc;
}
";

const NAN_CHECK_FINAL_WGSL: &str = r"
@group(0) @binding(0) var<storage, read_write> flag: array<u32>;

@compute @workgroup_size(1)
fn main() {
    // flag[0] already set from reduce; this is a no-op marker
    // but ensures the pipeline has a valid entry point
}
";
