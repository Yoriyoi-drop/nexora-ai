//! Deployment via Distillation Node
//!
//! Mode: Guided Distillation & Custom Distillation.
//! Sistem memverifikasi kompatibilitas, menjalankan teacher-student compression,
//! dan menghasilkan optimized inference graph untuk target deployment:
//! Edge TPU, mobile, browser runtime, embedded system, cloud inference cluster.

pub mod teacher_student;
pub mod export;

pub use teacher_student::*;
pub use export::*;

use crate::DLResult;
use crate::gnac::canvas::NeuralGraph;

/// Konfigurasi knowledge distillation
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    pub temperature: f64,
    pub alpha: f64,
    pub student_depth: usize,
    pub student_width: usize,
    pub target_hardware: String,
}
