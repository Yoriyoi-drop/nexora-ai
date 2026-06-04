// Integration test: GPU Mixed Precision (FP16/BF16) correctness
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_mixed_precision -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::gpu_mixed::{GpuDType, MixedPrecisionConfig, LossScaler};

    #[test]
    fn test_loss_scaler_basic() {
        let scaler = LossScaler::new(2.0, 1000, 2.0, 0.5);
        
        assert_eq!(scaler.scale, 2.0);
        assert_eq!(scaler.growth_interval, 1000);
        assert_eq!(scaler.growth_factor, 2.0);
        assert_eq!(scaler.backoff_factor, 0.5);
        
        println!("LossScaler basic OK");
    }

    #[test]
    fn test_loss_scaler_growth() {
        let mut scaler = LossScaler::new(1.0, 2, 2.0, 0.5);
        
        // Growth after interval steps
        scaler.update_growth();
        assert_eq!(scaler.scale, 2.0);
        
        scaler.update_growth();
        assert_eq!(scaler.scale, 4.0);
        
        println!("LossScaler growth OK");
    }

    #[test]
    fn test_loss_scaler_backoff() {
        let mut scaler = LossScaler::new(4.0, 1000, 2.0, 0.5);
        
        // Backoff on overflow
        scaler.update_backoff();
        assert_eq!(scaler.scale, 2.0);
        
        scaler.update_backoff();
        assert_eq!(scaler.scale, 1.0);
        
        println!("LossScaler backoff OK");
    }

    #[test]
    fn test_mixed_precision_config() {
        let config = MixedPrecisionConfig {
            compute_dtype: GpuDType::F16,
            master_weights: true,
            loss_scaling: LossScaler::new(2.0, 1000, 2.0, 0.5),
        };
        
        assert!(matches!(config.compute_dtype, GpuDType::F16));
        assert!(config.master_weights);
        assert_eq!(config.loss_scaling.scale, 2.0);
        
        println!("MixedPrecisionConfig OK");
    }

    #[test]
    fn test_gpu_dtype_enum() {
        let f32 = GpuDType::F32;
        let f16 = GpuDType::F16;
        let bf16 = GpuDType::BF16;
        
        // Test that all variants exist
        match f32 {
            GpuDType::F32 => {},
            _ => panic!("F32 variant failed"),
        }
        match f16 {
            GpuDType::F16 => {},
            _ => panic!("F16 variant failed"),
        }
        match bf16 {
            GpuDType::BF16 => {},
            _ => panic!("BF16 variant failed"),
        }
        
        println!("GpuDType enum OK");
    }

    #[test]
    fn test_mixed_precision_default_config() {
        let config = MixedPrecisionConfig::default();
        
        // Default should be FP32 compute, no master weights, no loss scaling
        assert!(matches!(config.compute_dtype, GpuDType::F32));
        assert!(!config.master_weights);
        assert_eq!(config.loss_scaling.scale, 1.0);
        
        println!("MixedPrecisionConfig default OK");
    }
}
