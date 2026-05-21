// Integration test: GPU Softmax correctness vs CPU reference
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_softmax -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::ops::activations::softmax;
    use nexora_autograd::tensor::Tensor;

    #[test]
    fn test_gpu_softmax_correctness_basic() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Simple 2D case
        let logits = ArrayD::from_shape_vec(vec![1, 4], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        
        // GPU result
        let g = GpuTensor::from_cpu(&logits).unwrap();
        let gpu_result = ctx.softmax(&g).unwrap();
        let gpu_cpu = gpu_result.to_cpu();

        // CPU reference
        let t = Tensor::new(logits);
        let cpu_result = softmax(&t, 1).unwrap();
        let cpu_data = cpu_result.data();

        for i in 0..4 {
            let diff = (gpu_cpu[[0, i]] - cpu_data[[0, i]]).abs();
            assert!(
                diff < 1e-4,
                "softmax[0,{}] mismatch: GPU={} CPU={} diff={}",
                i, gpu_cpu[[0, i]], cpu_data[[0, i]], diff
            );
        }
        println!("GPU softmax basic [1,4] OK");
    }

    #[test]
    fn test_gpu_softmax_correctness_batch() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Batch of 3 sequences, vocab size 8
        let logits = ArrayD::from_shape_vec(
            vec![3, 8],
            (0..24).map(|i| (i as f32) * 0.5).collect(),
        ).unwrap();
        
        let g = GpuTensor::from_cpu(&logits).unwrap();
        let gpu_result = ctx.softmax(&g).unwrap();
        let gpu_cpu = gpu_result.to_cpu();

        let t = Tensor::new(logits.clone());
        let cpu_result = softmax(&t, 1).unwrap();
        let cpu_data = cpu_result.data();

        for i in 0..3 {
            for j in 0..8 {
                let diff = (gpu_cpu[[i, j]] - cpu_data[[i, j]]).abs();
                assert!(
                    diff < 1e-4,
                    "softmax[{},{}] mismatch",
                    i, j
                );
            }
        }
        println!("GPU softmax batch [3,8] OK");
    }

    #[test]
    fn test_gpu_softmax_sum_to_one() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Verify softmax sums to 1.0 for each row
        let logits = ArrayD::from_shape_vec(
            vec![5, 16],
            (0..80).map(|i| (i as f32) * 0.1 - 4.0).collect(),
        ).unwrap();
        
        let g = GpuTensor::from_cpu(&logits).unwrap();
        let gpu_result = ctx.softmax(&g).unwrap();
        let gpu_cpu = gpu_result.to_cpu();

        for i in 0..5 {
            let mut sum = 0.0f32;
            for j in 0..16 {
                sum += gpu_cpu[[i, j]];
            }
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "softmax row {} sums to {}, expected 1.0",
                i, sum
            );
        }
        println!("GPU softmax sum-to-one property OK");
    }

    #[test]
    fn test_gpu_softmax_numerical_stability() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Test with large values (numerical stability)
        let logits = ArrayD::from_shape_vec(
            vec![1, 4],
            vec![1000.0, 1001.0, 1002.0, 1003.0],
        ).unwrap();
        
        let g = GpuTensor::from_cpu(&logits).unwrap();
        let gpu_result = ctx.softmax(&g).unwrap();
        let gpu_cpu = gpu_result.to_cpu();

        // Should not produce NaN or Inf
        for i in 0..4 {
            assert!(
                gpu_cpu[[0, i]].is_finite(),
                "softmax produced non-finite value at index {}: {}",
                i, gpu_cpu[[0, i]]
            );
        }
        println!("GPU softmax numerical stability OK");
    }
}
