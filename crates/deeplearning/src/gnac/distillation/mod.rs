//! Deployment via Distillation Node
//!
//! Mode: Guided Distillation & Custom Distillation.
//! Sistem memverifikasi kompatibilitas, menjalankan teacher-student compression,
//! dan menghasilkan optimized inference graph untuk target deployment:
//! Edge TPU, mobile, browser runtime, embedded system, cloud inference cluster.

pub mod export;
pub mod teacher_student;

pub use export::*;
pub use teacher_student::*;

/// Konfigurasi knowledge distillation
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    pub temperature: f64,
    pub alpha: f64,
    pub student_depth: usize,
    pub student_width: usize,
    pub target_hardware: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distillation_config_debug() {
        let cfg = DistillationConfig {
            temperature: 2.0,
            alpha: 0.5,
            student_depth: 4,
            student_width: 2,
            target_hardware: "mobile".to_string(),
        };
        assert!(!format!("{:?}", cfg).is_empty());
    }
}
