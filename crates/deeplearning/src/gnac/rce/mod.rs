//! Resource Cognition Engine (RCE)
//!
//! Sistem kesadaran resource real-time. Setiap perubahan graf langsung
//! menghitung estimasi VRAM, FLOPs, inference latency, tensor bandwidth,
//! cloud cost, dan konsumsi energi. Mendukung constraint-aware architecture design.

pub mod constraints;
pub mod estimator;

pub use constraints::*;
pub use estimator::*;

/// Laporan resource untuk satu graf atau subgraf
#[derive(Debug, Clone)]
pub struct ResourceReport {
    pub total_vram_mb: f64,
    pub total_flops: u64,
    pub inference_latency_ms: f64,
    pub tensor_bandwidth_gbs: f64,
    pub estimated_cloud_cost_per_hour: f64,
    pub estimated_energy_joules: f64,
    pub parameter_count: usize,
    pub activation_memory_mb: f64,
}

impl ResourceReport {
    pub fn new() -> Self {
        ResourceReport {
            total_vram_mb: 0.0,
            total_flops: 0,
            inference_latency_ms: 0.0,
            tensor_bandwidth_gbs: 0.0,
            estimated_cloud_cost_per_hour: 0.0,
            estimated_energy_joules: 0.0,
            parameter_count: 0,
            activation_memory_mb: 0.0,
        }
    }

    /// Estimasi biaya: 1 FLOP ≈ 1e-12 kWh (FP32 pada GPU modern)
    pub fn estimate_cloud_cost(flops: u64, vram_mb: f64) -> f64 {
        let energy_kwh = flops as f64 * 1e-12;
        let vram_cost = vram_mb / 1024.0 * 0.10; // $0.10/GB-hour
        energy_kwh * 0.12 + vram_cost // $0.12/kWh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_report_new() {
        let r = ResourceReport::new();
        assert_eq!(r.total_flops, 0);
        assert_eq!(r.parameter_count, 0);
    }

    #[test]
    fn test_estimate_cloud_cost_zero() {
        let cost = ResourceReport::estimate_cloud_cost(0, 0.0);
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_estimate_cloud_cost_nonzero() {
        let cost = ResourceReport::estimate_cloud_cost(1_000_000_000_000, 1000.0);
        assert!(cost > 0.0);
    }
}
