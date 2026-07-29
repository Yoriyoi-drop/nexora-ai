use crate::gnac::canvas::{ActivationStats, GradientStatus};
use crate::gnac::{DType, TensorDesc};

/// Metadata lengkap untuk SmartTensor
#[derive(Debug, Clone)]
pub struct SmartTensorMetadata {
    pub tensor: TensorDesc,
    pub gradient: GradientStatus,
    pub entropy_score: f32,
    pub activation: ActivationStats,
    pub bandwidth_estimate: f64, // MB/s
    pub memory_cost: usize,      // bytes
    pub is_frozen: bool,
    pub is_grad_required: bool,
}

impl SmartTensorMetadata {
    pub fn new(tensor: TensorDesc) -> Self {
        let memory_cost = tensor.numel
            * match tensor.dtype {
                DType::F32 | DType::I32 => 4,
                DType::F64 | DType::I64 => 8,
                DType::F16 | DType::BF16 => 2,
                DType::U8 | DType::Bool => 1,
            };

        SmartTensorMetadata {
            memory_cost,
            tensor,
            gradient: GradientStatus::Stable,
            entropy_score: 0.0,
            activation: ActivationStats::new(),
            bandwidth_estimate: 0.0,
            is_frozen: false,
            is_grad_required: true,
        }
    }

    pub fn update_entropy(&mut self, distribution: &[f32]) {
        let total: f32 = distribution.iter().sum();
        if total <= 0.0 {
            self.entropy_score = 0.0;
            return;
        }
        self.entropy_score = -distribution
            .iter()
            .map(|&p| {
                let normalized = p / total;
                if normalized > 0.0 {
                    normalized * normalized.log2()
                } else {
                    0.0
                }
            })
            .sum::<f32>();
    }

    pub fn estimate_bandwidth(&mut self, flops: u64) {
        // Estimasi bandwidth berdasarkan ukuran tensor dan FLOPs
        let bytes_per_element = self.memory_cost / self.tensor.numel.max(1);
        let elements_per_second = flops as f64 / 2.0;
        self.bandwidth_estimate = (self.tensor.numel as f64 * bytes_per_element as f64)
            * elements_per_second
            / 1_000_000.0;
    }

    pub fn freeze(&mut self) {
        self.is_frozen = true;
        self.is_grad_required = false;
    }

    pub fn unfreeze(&mut self) {
        self.is_frozen = false;
        self.is_grad_required = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnac::TensorDesc;

    fn meta() -> SmartTensorMetadata {
        SmartTensorMetadata::new(TensorDesc::new(vec![2, 3], crate::gnac::DType::F32))
    }

    #[test]
    fn test_smart_tensor_metadata_new() {
        let m = meta();
        assert_eq!(m.tensor.shape, vec![2, 3]);
        assert_eq!(m.memory_cost, 2 * 3 * 4);
        assert_eq!(m.gradient, GradientStatus::Stable);
        assert!(!m.is_frozen);
        assert!(m.is_grad_required);
    }

    #[test]
    fn test_smart_tensor_metadata_new_f64() {
        let desc = TensorDesc::new(vec![10], crate::gnac::DType::F64);
        let m = SmartTensorMetadata::new(desc);
        assert_eq!(m.memory_cost, 10 * 8);
    }

    #[test]
    fn test_update_entropy() {
        let mut m = meta();
        m.update_entropy(&[0.5, 0.5]);
        assert!(m.entropy_score > 0.0);
    }

    #[test]
    fn test_update_entropy_zero_total() {
        let mut m = meta();
        m.update_entropy(&[0.0, 0.0]);
        assert_eq!(m.entropy_score, 0.0);
    }

    #[test]
    fn test_estimate_bandwidth() {
        let mut m = meta();
        m.estimate_bandwidth(1_000_000);
        assert!(m.bandwidth_estimate > 0.0);
    }

    #[test]
    fn test_freeze() {
        let mut m = meta();
        m.freeze();
        assert!(m.is_frozen);
        assert!(!m.is_grad_required);
    }

    #[test]
    fn test_unfreeze() {
        let mut m = meta();
        m.freeze();
        m.unfreeze();
        assert!(!m.is_frozen);
        assert!(m.is_grad_required);
    }
}
