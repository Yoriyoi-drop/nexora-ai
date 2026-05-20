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
        let supports_subgroup_barrier =
            features.contains(wgpu::Features::SUBGROUP_BARRIER);

        let dedicated_video_memory = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => {
                match limits.max_storage_buffer_binding_size {
                    size if size > 1 << 30 => size,
                    _ => 8 << 30,
                }
            }
            _ => 0,
        };

        let shared_system_memory = match info.device_type {
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu => {
                limits.max_storage_buffer_binding_size
            }
            _ => 0,
        };

        Self {
            vendor,
            backend: info.backend,
            device_name: info.name,
            device_type: info.device_type,
            supports_subgroups,
            supports_subgroup_barrier,
            max_workgroup_size: limits.max_compute_workgroup_size_x
                * limits.max_compute_workgroup_size_y
                * limits.max_compute_workgroup_size_z,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup:
                limits.max_compute_invocations_per_workgroup,
            max_compute_shared_memory: limits.max_compute_workgroup_storage_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_bind_groups: limits.max_bind_groups,
            dedicated_video_memory,
            shared_system_memory,
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

    pub fn optimal_workgroup_size(&self, tile: u32) -> u32 {
        let preferred = tile * tile;
        preferred.min(self.max_compute_invocations_per_workgroup)
    }

    pub fn display(&self) -> String {
        format!(
            "GPU: {} | Vendor: {} | Type: {:?} | Backend: {:?}\n\
             Subgroups: {} | Max WG: {} | Max SMEM: {}B\n\
             VRAM: {} | Shared Mem: {}",
            self.device_name,
            self.vendor,
            self.device_type,
            self.backend,
            if self.supports_subgroups { "yes" } else { "no" },
            self.max_workgroup_size,
            self.max_compute_shared_memory,
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

