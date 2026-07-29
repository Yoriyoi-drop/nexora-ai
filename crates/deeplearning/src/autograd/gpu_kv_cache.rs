use crate::autograd::gpu::{GpuContext, GpuError, GpuTensor};
use std::sync::atomic::{AtomicBool, Ordering};

// ─── GpuPageTable ──────────────────────────────────────────────────────────────

/// Manages paged KV cache blocks on GPU.
/// Each page holds `page_size` KV entries for a single layer + head.
pub struct GpuPageTable {
    /// Flat buffer of all KV data: [num_pages, page_size, 2, head_dim]
    pub data: GpuTensor,
    /// Page table: [num_pages] → offset in `data` (page index)
    pub page_table: GpuTensor,
    /// Free list GPU mirror — synced from `cpu_free` lazily.
    /// Only read by future shader code; gather/scatter use separate `page_ids` param.
    pub free_list: GpuTensor,
    pub page_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_pages: usize,
    free_count: u32,
    /// CPU-side free list for O(1) allocation (pop from end).
    pub cpu_free: Vec<u32>,
    /// Bitmap tracking which pages are free: bit=1 means free, bit=0 means allocated.
    /// u64 words — 64 pages per word.
    free_bitmap: Vec<u64>,
    /// True if cpu_free changed since last GPU sync.
    dirty: AtomicBool,
}

impl GpuPageTable {
    pub fn new(
        _ctx: &GpuContext,
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
        let free_indices: Vec<f32> = (0..max_pages).map(|i| i as f32).collect();
        let free_list = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![max_pages], free_indices)
                .map_err(|_| GpuError::Buffer("shape error".into()))?,
        )?;

        let cpu_free: Vec<u32> = (0..max_pages).map(|i| i as u32).rev().collect();

        let num_words = (max_pages + 63) / 64;
        let mut free_bitmap = vec![0u64; num_words];
        for i in 0..num_words {
            let remaining = max_pages - i * 64;
            let bits_in_word = 64.min(remaining);
            free_bitmap[i] = if bits_in_word == 64 {
                !0u64
            } else {
                (1u64 << bits_in_word) - 1
            };
        }

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
            free_bitmap,
            dirty: AtomicBool::new(false),
        })
    }

    /// Number of free pages remaining.
    pub fn available_pages(&self) -> usize {
        self.cpu_free.len()
    }

    /// Allocate a page. Returns the page index, or None if OOM.
    pub fn alloc(&mut self) -> Option<u32> {
        let page = self.cpu_free.pop();
        if let Some(idx) = page {
            let word = (idx as usize) / 64;
            let bit = 1u64 << ((idx as usize) % 64);
            self.free_bitmap[word] &= !bit;
            self.free_count -= 1;
            self.dirty.store(true, Ordering::Release);
        }
        page
    }

    /// Free a page index, returning it to the pool. O(1) via bitmap.
    pub fn free(&mut self, page_idx: u32) {
        if (page_idx as usize) >= self.max_pages {
            return;
        }
        let word = (page_idx as usize) / 64;
        let bit = 1u64 << ((page_idx as usize) % 64);
        if self.free_bitmap[word] & bit == 0 {
            self.free_bitmap[word] |= bit;
            self.cpu_free.push(page_idx);
            self.free_count += 1;
            self.dirty.store(true, Ordering::Release);
        }
    }

    /// Sync CPU free list to GPU if dirty. No-op if already synced.
    /// Clears the dirty flag after successful sync.
    pub fn sync_free_list_if_dirty(&self, ctx: &GpuContext) -> Result<(), GpuError> {
        if !self.dirty.load(Ordering::Acquire) {
            return Ok(());
        }
        self.sync_free_list_to_gpu(ctx)?;
        self.dirty.store(false, Ordering::Release);
        Ok(())
    }

    /// Force-sync CPU free list to GPU free_list tensor.
    pub fn sync_free_list_to_gpu(&self, ctx: &GpuContext) -> Result<(), GpuError> {
        let mut data = vec![0.0f32; self.max_pages];
        for (i, &idx) in self.cpu_free.iter().enumerate() {
            data[i] = idx as f32;
        }
        let arr = ndarray::ArrayD::from_shape_vec(vec![self.max_pages], data)
            .map_err(|_| GpuError::Buffer("shape error".into()))?;
        let temp = GpuTensor::from_cpu(&arr)?;
        ctx.with_encoder(|enc| {
            enc.copy_buffer_to_buffer(
                temp.buffer(),
                0,
                self.free_list.buffer(),
                0,
                (self.max_pages * 4) as u64,
            );
        });
        Ok(())
    }
}

// ─── Compile pipelines ─────────────────────────────────────────────────────────

pub fn compile_gather_pages_pipeline(
    ctx: &mut GpuContext,
) -> Result<wgpu::ComputePipeline, crate::autograd::gpu::GpuError> {
    ctx.compile_pipeline_cached(
        "gather_pages",
        &[
            crate::autograd::gpu::storage_binding(0, true),
            crate::autograd::gpu::storage_binding(1, true),
            crate::autograd::gpu::storage_binding(2, false),
            crate::autograd::gpu::uniform_binding(3),
        ],
        std::borrow::Cow::Borrowed(GATHER_PAGES_WGSL),
        "main",
    )
}

pub fn compile_scatter_pages_pipeline(
    ctx: &mut GpuContext,
) -> Result<wgpu::ComputePipeline, crate::autograd::gpu::GpuError> {
    ctx.compile_pipeline_cached(
        "scatter_pages",
        &[
            crate::autograd::gpu::storage_binding(0, true),
            crate::autograd::gpu::storage_binding(1, false),
            crate::autograd::gpu::uniform_binding(2),
        ],
        std::borrow::Cow::Borrowed(SCATTER_PAGES_WGSL),
        "main",
    )
}

pub struct GpuKvCachePipelines {
    pub gather: wgpu::ComputePipeline,
    pub scatter: wgpu::ComputePipeline,
}

pub fn compile_kv_cache_pipelines(
    ctx: &mut GpuContext,
) -> Result<GpuKvCachePipelines, crate::autograd::gpu::GpuError> {
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

    let cfg_buf = ctx.alloc_buffer(
        16,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let cfg: [u32; 4] = [
        num_pages,
        page_table.page_size as u32,
        page_table.num_heads as u32,
        page_table.head_dim as u32,
    ];
    ctx.queue
        .write_buffer(&cfg_buf.buffer, 0, bytemuck::cast_slice(&cfg));

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
                resource: cfg_buf.buffer.as_entire_binding(),
            },
        ],
    });

    let total_elements = num_pages * page_table.page_size as u32 * 2 * page_table.head_dim as u32;
    ctx.with_encoder(|enc| {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather_pages"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(total_elements.div_ceil(256), 1, 1);
    });
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

    let cfg_buf = ctx.alloc_buffer(
        16,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let cfg: [u32; 4] = [
        num_pages,
        page_table.page_size as u32,
        page_table.num_heads as u32,
        page_table.head_dim as u32,
    ];
    ctx.queue
        .write_buffer(&cfg_buf.buffer, 0, bytemuck::cast_slice(&cfg));

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
                resource: cfg_buf.buffer.as_entire_binding(),
            },
        ],
    });

    let total_elements = num_pages * page_table.page_size as u32 * 2 * page_table.head_dim as u32;
    ctx.with_encoder(|enc| {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter_pages"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(total_elements.div_ceil(256), 1, 1);
    });
    Ok(())
}

// ─── WGSL Shaders ──────────────────────────────────────────────────────────────

const GATHER_PAGES_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> pool: array<f32>;
@group(0) @binding(1) var<storage, read> page_ids: array<f32>;
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
    let block_id = u32(page_ids[page_idx]);
    let src_offset = block_id * entries_per_page + within_page;
    output[i] = pool[src_offset];
}
";

const SCATTER_PAGES_WGSL: &str = r"
@group(0) @binding(0) var<storage, read_write> pool: array<f32>;
@group(0) @binding(1) var<storage, read> page_ids: array<f32>;
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
    let block_id = u32(page_ids[page_idx]);
    let dst_offset = block_id * entries_per_page + within_page;
    pool[dst_offset] = input[i];
}
";
