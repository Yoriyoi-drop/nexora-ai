// Integration test: GPU Attention correctness
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_attention -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::tensor::Tensor;

    #[test]
    fn test_gpu_attention_basic() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Q, K, V: [batch=2, heads=4, seq=8, head_dim=16]
        let q_data = ArrayD::from_shape_vec(
            vec![2, 4, 8, 16],
            (0..1024).map(|i| (i as f32) * 0.01).collect(),
        ).unwrap();
        let k_data = ArrayD::from_shape_vec(
            vec![2, 4, 8, 16],
            (0..1024).map(|i| (i as f32) * 0.02).collect(),
        ).unwrap();
        let v_data = ArrayD::from_shape_vec(
            vec![2, 4, 8, 16],
            (0..1024).map(|i| (i as f32) * 0.03).collect(),
        ).unwrap();

        let gq = GpuTensor::from_cpu(&q_data).unwrap();
        let gk = GpuTensor::from_cpu(&k_data).unwrap();
        let gv = GpuTensor::from_cpu(&v_data).unwrap();

        // Compute attention
        let output = ctx.attention(&gq, &gk, &gv, None, None).unwrap();
        let cpu_output = output.to_cpu();

        // Check output shape
        assert_eq!(cpu_output.shape(), &[2, 4, 8, 16]);
        
        println!("GPU attention basic [2,4,8,16] OK");
    }

    #[test]
    fn test_gpu_attention_with_mask() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Q, K, V: [batch=1, heads=2, seq=4, head_dim=8]
        let q_data = ArrayD::from_shape_vec(
            vec![1, 2, 4, 8],
            (0..64).map(|i| (i as f32) * 0.1).collect(),
        ).unwrap();
        let k_data = ArrayD::from_shape_vec(
            vec![1, 2, 4, 8],
            (0..64).map(|i| (i as f32) * 0.05).collect(),
        ).unwrap();
        let v_data = ArrayD::from_shape_vec(
            vec![1, 2, 4, 8],
            (0..64).map(|i| (i as f32) * 0.02).collect(),
        ).unwrap();

        // Causal mask
        let mask_data = ArrayD::from_shape_vec(
            vec![1, 1, 4, 4],
            vec![0.0, -1e9, -1e9, -1e9,
                 0.0, 0.0, -1e9, -1e9,
                 0.0, 0.0, 0.0, -1e9,
                 0.0, 0.0, 0.0, 0.0],
        ).unwrap();

        let gq = GpuTensor::from_cpu(&q_data).unwrap();
        let gk = GpuTensor::from_cpu(&k_data).unwrap();
        let gv = GpuTensor::from_cpu(&v_data).unwrap();
        let gmask = GpuTensor::from_cpu(&mask_data).unwrap();

        let output = ctx.attention(&gq, &gk, &gv, Some(&gmask), None).unwrap();
        let cpu_output = output.to_cpu();

        assert_eq!(cpu_output.shape(), &[1, 2, 4, 8]);
        
        println!("GPU attention with causal mask OK");
    }

    #[test]
    fn test_gpu_attention_softmax_stability() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Test with large values to check numerical stability
        let q_data = ArrayD::from_shape_vec(
            vec![1, 1, 4, 8],
            vec![100.0, 200.0, 300.0, 400.0,
                 100.0, 200.0, 300.0, 400.0,
                 100.0, 200.0, 300.0, 400.0,
                 100.0, 200.0, 300.0, 400.0],
        ).unwrap();
        let k_data = ArrayD::from_shape_vec(
            vec![1, 1, 4, 8],
            vec![50.0, 100.0, 150.0, 200.0,
                 50.0, 100.0, 150.0, 200.0,
                 50.0, 100.0, 150.0, 200.0,
                 50.0, 100.0, 150.0, 200.0],
        ).unwrap();
        let v_data = ArrayD::from_shape_vec(
            vec![1, 1, 4, 8],
            (0..32).map(|i| (i as f32) * 0.1).collect(),
        ).unwrap();

        let gq = GpuTensor::from_cpu(&q_data).unwrap();
        let gk = GpuTensor::from_cpu(&k_data).unwrap();
        let gv = GpuTensor::from_cpu(&v_data).unwrap();

        let output = ctx.attention(&gq, &gk, &gv, None, None).unwrap();
        let cpu_output = output.to_cpu();

        // Check for NaN/Inf
        for i in 0..32 {
            let val = cpu_output.as_slice().unwrap()[i];
            assert!(
                val.is_finite(),
                "Attention produced non-finite value at index {}: {}",
                i, val
            );
        }
        
        println!("GPU attention numerical stability OK");
    }
}
