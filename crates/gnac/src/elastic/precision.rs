use crate::DType;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_precision_edge_tpu() {
        assert_eq!(PrecisionScaler::select_precision("edge_tpu"), DType::F16);
    }

    #[test]
    fn test_select_precision_cpu() {
        assert_eq!(PrecisionScaler::select_precision("cpu"), DType::F32);
    }

    #[test]
    fn test_select_precision_gpu() {
        assert_eq!(PrecisionScaler::select_precision("gpu"), DType::BF16);
    }

    #[test]
    fn test_select_precision_unknown() {
        assert_eq!(PrecisionScaler::select_precision("unknown"), DType::F32);
    }

    #[test]
    fn test_estimated_speedup_f32_to_f16() {
        assert!((PrecisionScaler::estimated_speedup(&DType::F32, &DType::F16) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_estimated_speedup_no_change() {
        assert!((PrecisionScaler::estimated_speedup(&DType::F16, &DType::F16) - 1.0).abs() < 1e-5);
    }
}
