// Integration test: GPU Mixed Precision (FP16/BF16) correctness
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_mixed_precision -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::gpu_mixed::{GpuDType, GpuLossScaler};

    #[test]
    fn test_loss_scaler_basic() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let scaler = GpuLossScaler::new(&ctx, 2.0).unwrap();
        
        assert!((scaler.scale() - 2.0).abs() < 1e-6);
        
        println!("LossScaler basic OK");
    }

    #[test]
    fn test_gpu_dtype_enum() {
        let f32 = GpuDType::F32;
        let f16 = GpuDType::F16;
        let bf16 = GpuDType::BF16;
        
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
    fn test_gpu_dtype_methods() {
        assert_eq!(GpuDType::F32.bytes_per_element(), 4);
        assert_eq!(GpuDType::F16.bytes_per_element(), 2);
        assert_eq!(GpuDType::BF16.bytes_per_element(), 2);
        
        assert!(GpuDType::F16.is_half());
        assert!(GpuDType::BF16.is_half());
        assert!(!GpuDType::F32.is_half());
        
        println!("GpuDType methods OK");
    }
}
