//! Tensor Scheduling Layer
//!
//! Mengatur memory checkpointing, asynchronous execution, tensor paging,
//! gradient accumulation, dan VRAM reuse. Tujuannya memungkinkan graf besar
//! berjalan bahkan pada perangkat dengan resource terbatas.

pub mod async_exec;
pub mod memory;
pub mod paging;

pub use async_exec::*;
pub use memory::*;
pub use paging::*;

/// Konfigurasi tensor scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub enable_checkpointing: bool,
    pub checkpoint_frequency: usize,
    pub enable_async: bool,
    pub enable_tensor_paging: bool,
    pub max_vram_mb: f64,
    pub gradient_accumulation_steps: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            enable_checkpointing: true,
            checkpoint_frequency: 4,
            enable_async: true,
            enable_tensor_paging: true,
            max_vram_mb: 4096.0,
            gradient_accumulation_steps: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_config_default() {
        let c = SchedulerConfig::default();
        assert!(c.enable_checkpointing);
        assert_eq!(c.checkpoint_frequency, 4);
        assert_eq!(c.max_vram_mb, 4096.0);
    }
}
