use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuVendor::Nvidia => write!(f, "NVIDIA"),
            GpuVendor::Amd => write!(f, "AMD"),
            GpuVendor::Intel => write!(f, "Intel"),
            GpuVendor::Apple => write!(f, "Apple"),
            GpuVendor::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub vendor: GpuVendor,
    pub backend: wgpu::Backend,
    pub device_name: String,
    pub device_type: wgpu::DeviceType,
    pub supports_subgroups: bool,
    pub supports_subgroup_barrier: bool,
    pub supports_f16: bool,
    pub supports_f64: bool,
    pub max_workgroup_size: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_shared_memory: u32,
    pub max_storage_buffer_binding_size: u64,
    pub max_bind_groups: u32,
    pub dedicated_video_memory: u64,
    pub shared_system_memory: u64,
    pub supports_timestamp_query: bool,
    pub supports_timestamp_query_inside_passes: bool,
    pub supports_timestamp_query_inside_encoders: bool,
}

impl GpuCapabilities {
    pub fn detect(device: &wgpu::Device, adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let limits = device.limits();
        let features = device.features();

        let vendor = match info.vendor {
            0x10DE => GpuVendor::Nvidia,
            0x1002 | 0x1022 => GpuVendor::Amd,
            0x8086 => GpuVendor::Intel,
            0x106B => GpuVendor::Apple,
            _ => GpuVendor::Unknown,
        };

        let supports_subgroups = features.contains(wgpu::Features::SUBGROUP);
        let supports_subgroup_barrier = features.contains(wgpu::Features::SUBGROUP_BARRIER);
        let supports_f16 = features.contains(wgpu::Features::SHADER_F16);
        let supports_f64 = features.contains(wgpu::Features::SHADER_F64);

        let dedicated_video_memory = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => match limits.max_storage_buffer_binding_size {
                size if size > 1 << 30 => size,
                _ => 8 << 30,
            },
            _ => 0,
        };

        let shared_system_memory = match info.device_type {
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu => {
                limits.max_storage_buffer_binding_size
            }
            _ => 0,
        };

        let supports_timestamp_query = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let supports_timestamp_query_inside_passes = features.contains(
            wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        );
        let supports_timestamp_query_inside_encoders = features.contains(
            wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        );

        Self {
            vendor,
            backend: info.backend,
            device_name: info.name,
            device_type: info.device_type,
            supports_subgroups,
            supports_subgroup_barrier,
            supports_f16,
            supports_f64,
            max_workgroup_size: limits.max_compute_workgroup_size_x
                * limits.max_compute_workgroup_size_y
                * limits.max_compute_workgroup_size_z,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_shared_memory: limits.max_compute_workgroup_storage_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_bind_groups: limits.max_bind_groups,
            dedicated_video_memory,
            shared_system_memory,
            supports_timestamp_query,
            supports_timestamp_query_inside_passes,
            supports_timestamp_query_inside_encoders,
        }
    }

    pub fn optimal_tile_size(&self) -> u32 {
        let preferred = if self.vendor == GpuVendor::Nvidia {
            16
        } else {
            8
        };

        let max_threads = self.max_compute_invocations_per_workgroup;
        let tile = preferred.min(32);

        if tile * tile > max_threads {
            (max_threads as f64).sqrt().floor() as u32
        } else {
            tile
        }
    }

    pub fn optimal_fused_tile_size(&self) -> u32 {
        let candidates = [32, 16, 8];
        for &tile in candidates.iter() {
            let wg_threads = tile * tile;
            let shared_bytes = 2 * tile * tile * 4;
            if wg_threads <= self.max_compute_invocations_per_workgroup
                && shared_bytes <= self.max_compute_shared_memory
            {
                return tile;
            }
        }
        8
    }

    pub fn optimal_workgroup_size(&self, tile: u32) -> u32 {
        let preferred = tile * tile;
        preferred.min(self.max_compute_invocations_per_workgroup)
    }

    pub fn display(&self) -> String {
        format!(
            "GPU: {} | Vendor: {} | Type: {:?} | Backend: {:?}\n\
             Subgroups: {} | Max WG: {} | Max SMEM: {}B\n\
             FP16: {} | FP64: {}\n\
             Timestamp query: {} | Inside passes: {} | Inside encoders: {}\n\
             VRAM: {} | Shared Mem: {}",
            self.device_name,
            self.vendor,
            self.device_type,
            self.backend,
            if self.supports_subgroups { "yes" } else { "no" },
            self.max_workgroup_size,
            self.max_compute_shared_memory,
            if self.supports_f16 { "yes" } else { "no" },
            if self.supports_f64 { "yes" } else { "no" },
            if self.supports_timestamp_query { "yes" } else { "no" },
            if self.supports_timestamp_query_inside_passes {
                "yes"
            } else {
                "no"
            },
            if self.supports_timestamp_query_inside_encoders {
                "yes"
            } else {
                "no"
            },
            format_size(self.dedicated_video_memory),
            format_size(self.shared_system_memory),
        )
    }
}

fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    while size > 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    format!("{:.1}{}", size, UNITS[unit_idx])
}
