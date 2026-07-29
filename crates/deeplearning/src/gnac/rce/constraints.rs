use crate::gnac::DLResult;

/// Batasan resource untuk constraint-aware architecture design
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub max_vram_mb: f64,
    pub target_latency_ms: f64,
    pub max_cloud_cost_per_hour: f64,
    pub target_hardware: HardwareTarget,
    pub energy_ceiling_joules: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HardwareTarget {
    EdgeTPU,
    Mobile,
    CloudGPU,
    TPU,
}

impl ResourceConstraints {
    pub fn edge_tpu() -> Self {
        ResourceConstraints {
            max_vram_mb: 1024.0,
            target_latency_ms: 10.0,
            max_cloud_cost_per_hour: 0.0,
            target_hardware: HardwareTarget::EdgeTPU,
            energy_ceiling_joules: 5.0,
        }
    }

    pub fn mobile() -> Self {
        ResourceConstraints {
            max_vram_mb: 2048.0,
            target_latency_ms: 50.0,
            max_cloud_cost_per_hour: 0.0,
            target_hardware: HardwareTarget::Mobile,
            energy_ceiling_joules: 10.0,
        }
    }

    pub fn cloud_gpu() -> Self {
        ResourceConstraints {
            max_vram_mb: 80_000.0,
            target_latency_ms: 100.0,
            max_cloud_cost_per_hour: 3.0,
            target_hardware: HardwareTarget::CloudGPU,
            energy_ceiling_joules: 500.0,
        }
    }

    pub fn tpu() -> Self {
        ResourceConstraints {
            max_vram_mb: 160_000.0,
            target_latency_ms: 50.0,
            max_cloud_cost_per_hour: 5.0,
            target_hardware: HardwareTarget::TPU,
            energy_ceiling_joules: 200.0,
        }
    }

    pub fn validate(&self, vram_mb: f64, latency_ms: f64) -> DLResult<()> {
        if vram_mb > self.max_vram_mb {
            return Err(crate::gnac::DeepLearningError::Configuration {
                reason: format!(
                    "VRAM {:.1}MB exceeds limit {:.1}MB for {:?}",
                    vram_mb, self.max_vram_mb, self.target_hardware
                ),
            });
        }
        if latency_ms > self.target_latency_ms {
            return Err(crate::gnac::DeepLearningError::Configuration {
                reason: format!(
                    "Latency {:.1}ms exceeds target {:.1}ms for {:?}",
                    latency_ms, self.target_latency_ms, self.target_hardware
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_tpu_constraints() {
        let c = ResourceConstraints::edge_tpu();
        assert_eq!(c.target_hardware, HardwareTarget::EdgeTPU);
        assert_eq!(c.max_vram_mb, 1024.0);
    }

    #[test]
    fn test_mobile_constraints() {
        let c = ResourceConstraints::mobile();
        assert_eq!(c.target_hardware, HardwareTarget::Mobile);
    }

    #[test]
    fn test_cloud_gpu_constraints() {
        let c = ResourceConstraints::cloud_gpu();
        assert_eq!(c.target_hardware, HardwareTarget::CloudGPU);
    }

    #[test]
    fn test_validate_ok() {
        let c = ResourceConstraints::cloud_gpu();
        assert!(c.validate(1000.0, 50.0).is_ok());
    }

    #[test]
    fn test_validate_vram_exceeded() {
        let c = ResourceConstraints::edge_tpu();
        assert!(c.validate(2000.0, 5.0).is_err());
    }

    #[test]
    fn test_validate_latency_exceeded() {
        let c = ResourceConstraints::edge_tpu();
        assert!(c.validate(500.0, 100.0).is_err());
    }
}
