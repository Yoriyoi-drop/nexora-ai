use crate::gpu::{GpuContext, GpuError, GpuTensor};
use ndarray::ArrayD;

// ─── GpuPageTable ──────────────────────────────────────────────────────────────

/// Manages paged KV cache blocks on GPU.
/// Each page holds `page_size` KV entries for a single layer + head.
pub struct GpuPageTable {
    /// Flat buffer of all KV data: [num_pages, page_size, 2, head_dim]
    pub data: GpuTensor,
    /// Page table: [num_pages] → offset in `data` (page index)
    pub page_table: GpuTensor,
    /// Free list: [max_pages] → page indices available for allocation
    pub free_list: GpuTensor,
    pub page_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_pages: usize,
    free_count: u32,
    pub cpu_free: Vec<u32>,
}

impl GpuPageTable {
    pub fn new(
        ctx: &GpuContext,
        max_pages: usize,
        page_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self, GpuError> {
        let data = GpuTensor::zeros(&[
            max_pages, page_size, 2, // K and V
            head_dim,
        ])?;

        let page_table = GpuTensor::zeros(&[max_pages])?;

        // Free list initialized with sequential page indices
        let mut free_indices: Vec<f32> = (0..max_pages).map(|i| i as f32).collect();
        let free_list = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![max_pages], free_indices)
                .map_err(|_| GpuError::Buffer("shape error".into()))?,
        )?;

        let cpu_free: Vec<u32> = (0..max_pages).map(|i| i as u32).rev().collect();

        Ok(Self {
            data,
            page_table,
            free_list,
            page_size,
            num_heads,
            head_dim,
            max_pages,
            free_count: max_pages as u32,
            cpu_free,
        })
    }

    /// Number of free pages remaining.
    pub fn available_pages(&self) -> usize {
        self.cpu_free.len()
    }

    /// Allocate a page. Returns the page index, or None if OOM.
    pub fn alloc(&mut self) -> Option<u32> {
        self.cpu_free.pop()
    }

    /// Free a page index, returning it to the pool.
    pub fn free(&mut self, page_idx: u32) {
        if (page_idx as usize) < self.max_pages && !self.cpu_free.contains(&page_idx) {
            self.cpu_free.push(page_idx);
        }
    }

    /// Sync CPU free list to GPU free_list tensor (e.g. after alloc/free batch).
    pub fn sync_free_list_to_gpu(&self, ctx: &GpuContext) -> Result<(), GpuError> {
        let mut data = vec![0.0f32; self.max_pages];
        for (i, &idx) in self.cpu_free.iter().enumerate() {
            data[i] = idx as f32;
        }
        let arr = ndarray::ArrayD::from_shape_vec(vec![self.max_pages], data)
            .map_err(|_| GpuError::Buffer("shape error".into()))?;
        let temp = GpuTensor::from_cpu(&arr)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sync_free_list"),
            });
        encoder.copy_buffer_to_buffer(
            temp.buffer(),
            0,
            self.free_list.buffer(),
            0,
            (self.max_pages * 4) as u64,
        );
        ctx.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

// ─── Compile pipelines ─────────────────────────────────────────────────────────

pub fn compile_gather_pages_pipeline(ctx: &mut GpuContext) -> Result<wgpu::ComputePipeline, crate::gpu::GpuError> {
    ctx.compile_pipeline_cached(
        "gather_pages",
        &[
            crate::gpu::storage_binding(0, true),
            crate::gpu::storage_binding(1, true),
            crate::gpu::storage_binding(2, false),
            crate::gpu::uniform_binding(3),
        ],
        std::borrow::Cow::Borrowed(GATHER_PAGES_WGSL),
        "main",
    )
}

pub fn compile_scatter_pages_pipeline(ctx: &mut GpuContext) -> Result<wgpu::ComputePipeline, crate::gpu::GpuError> {
    ctx.compile_pipeline_cached(
        "scatter_pages",
        &[
            crate::gpu::storage_binding(0, true),
            crate::gpu::storage_binding(1, false),
            crate::gpu::uniform_binding(2),
        ],
        std::borrow::Cow::Borrowed(SCATTER_PAGES_WGSL),
        "main",
    )
}

pub struct GpuKvCachePipelines {
    pub gather: wgpu::ComputePipeline,
    pub scatter: wgpu::ComputePipeline,
}

pub fn compile_kv_cache_pipelines(ctx: &mut GpuContext) -> Result<GpuKvCachePipelines, crate::gpu::GpuError> {
    Ok(GpuKvCachePipelines {
        gather: compile_gather_pages_pipeline(ctx)?,
        scatter: compile_scatter_pages_pipeline(ctx)?,
    })
}

// ─── Dispatch ──────────────────────────────────────────────────────────────────

/// Gather paged KV data into contiguous output.
/// `page_ids`: [num_pages] GPU tensor of page indices.
/// `page_table`: page table data.
/// `out`: [page_size, num_pages, 2, head_dim] contiguous output.
pub fn dispatch_gather_pages(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    page_table: &GpuPageTable,
    page_ids: &GpuTensor,
    out: &GpuTensor,
) -> Result<(), GpuError> {
    let num_pages = page_ids.numel() as u32;

    let cfg_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gather_pages_cfg"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cfg: [u32; 4] = [
        num_pages,
        page_table.page_size as u32,
        page_table.num_heads as u32,
        page_table.head_dim as u32,
    ];
    ctx.queue
        .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gather_pages_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: page_table.data.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: page_ids.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cfg_buf.as_entire_binding(),
            },
        ],
    });

    let total_elements = num_pages * page_table.page_size as u32 * 2 * page_table.head_dim as u32;
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gather_pages_encoder"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather_pages"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((total_elements + 255) / 256, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    Ok(())
}

/// Scatter contiguous KV data into paged storage.
pub fn dispatch_scatter_pages(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    page_table: &GpuPageTable,
    page_ids: &GpuTensor,
    input: &GpuTensor,
) -> Result<(), GpuError> {
    let num_pages = page_ids.numel() as u32;

    let cfg_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scatter_pages_cfg"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cfg: [u32; 4] = [
        num_pages,
        page_table.page_size as u32,
        page_table.num_heads as u32,
        page_table.head_dim as u32,
    ];
    ctx.queue
        .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scatter_pages_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: page_table.data.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: page_ids.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cfg_buf.as_entire_binding(),
            },
        ],
    });

    let total_elements = num_pages * page_table.page_size as u32 * 2 * page_table.head_dim as u32;
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_pages_encoder"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter_pages"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((total_elements + 255) / 256, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    Ok(())
}

// ─── WGSL Shaders ──────────────────────────────────────────────────────────────

const GATHER_PAGES_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> pool: array<f32>;
@group(0) @binding(1) var<storage, read> page_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let num_pages = cfg.x;
    let page_size = cfg.y;
    let num_heads = cfg.z;
    let head_dim = cfg.w;

    let entries_per_page = page_size * 2u * head_dim;
    let total = num_pages * entries_per_page;

    let i = id.x;
    if i >= total { return; }

    let page_idx = i / entries_per_page;
    let within_page = i % entries_per_page;
    let block_id = page_ids[page_idx];
    let src_offset = block_id * entries_per_page + within_page;
    output[i] = pool[src_offset];
}
";

const SCATTER_PAGES_WGSL: &str = r"
@group(0) @binding(0) var<storage, read_write> pool: array<f32>;
@group(0) @binding(1) var<storage, read> page_ids: array<u32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let num_pages = cfg.x;
    let page_size = cfg.y;
    let num_heads = cfg.z;
    let head_dim = cfg.w;

    let entries_per_page = page_size * 2u * head_dim;
    let total = num_pages * entries_per_page;

    let i = id.x;
    if i >= total { return; }

    let page_idx = i / entries_per_page;
    let within_page = i % entries_per_page;
    let block_id = page_ids[page_idx];
    let dst_offset = block_id * entries_per_page + within_page;
    pool[dst_offset] = input[i];
}
";
