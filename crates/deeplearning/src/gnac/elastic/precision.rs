use crate::gnac::DType;

/// Precision scaling — turunkan precision untuk efisiensi
pub struct PrecisionScaler;

impl PrecisionScaler {
    /// Pilih precision optimal berdasarkan hardware target
    pub fn select_precision(hardware: &str) -> DType {
        match hardware {
            "edge_tpu" | "mobile" => DType::F16,
            "browser" => DType::F16,
            "cpu" => DType::F32,
            "gpu" => DType::BF16,
            "tpu" => DType::BF16,
            _ => DType::F32,
        }
    }

    /// Estimasi speedup dari precision scaling
    pub fn estimated_speedup(from: &DType, to: &DType) -> f64 {
        match (from, to) {
            (DType::F32, DType::F16) => 2.0,
            (DType::F32, DType::BF16) => 1.8,
            (DType::F16, DType::BF16) => 1.0,
            (DType::F32, DType::I32) => 1.2,
            _ => 1.0,
        }
    }
}
