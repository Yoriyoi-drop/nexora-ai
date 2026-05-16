//! Tensor Scheduling Layer
//!
//! Mengatur memory checkpointing, asynchronous execution, tensor paging,
//! gradient accumulation, dan VRAM reuse. Tujuannya memungkinkan graf besar
//! berjalan bahkan pada perangkat dengan resource terbatas.

pub mod memory;
pub mod async_exec;
pub mod paging;

pub use memory::*;
pub use async_exec::*;
pub use paging::*;

use crate::DLResult;
use crate::gnac::canvas::NeuralGraph;

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
