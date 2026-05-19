//! Guided Intervention System
//!
//! Ketika SmartTensor mendeteksi anomali (exploding gradient, dead activation,
//! unstable attention, mode collapse), Diagnostic Assistant aktif otomatis.
//! Menjelaskan masalah dalam bahasa natural dan menawarkan auto-fix, guided tuning,
//! atau kontrol manual penuh.

pub mod detector;
pub mod assistant;

pub use detector::*;
pub use assistant::*;
