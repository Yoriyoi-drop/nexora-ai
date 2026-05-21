// Integration test: GPU Matmul correctness vs CPU reference
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_matmul -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::ops::matmul::matmul;
    use nexora_autograd::tensor::Tensor;

    #[test]
    fn test_gpu_matmul_correctness_small() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Small matmul [4, 8] x [8, 4]
        let a_data = ArrayD::from_shape_vec(
            vec![4, 8],
            (0..32).map(|i| (i as f32) * 0.1).collect(),
        ).unwrap();
        let b_data = ArrayD::from_shape_vec(
            vec![8, 4],
            (0..32).map(|i| (i as f32) * 0.05).collect(),
        ).unwrap();

        // GPU result
        let ga = GpuTensor::from_cpu(&a_data).unwrap();
        let gb = GpuTensor::from_cpu(&b_data).unwrap();
        let gpu_result = ctx.matmul(&ga, &gb).unwrap();
        let gpu_cpu = gpu_result.to_cpu();

        // CPU reference
        let a_tensor = Tensor::new(a_data);
        let b_tensor = Tensor::new(b_data);
        let cpu_result = matmul(&a_tensor, &b_tensor).unwrap();
        let cpu_data = cpu_result.data();

        // Compare
        for i in 0..4 {
            for j in 0..4 {
                let diff = (gpu_cpu[[i, j]] - cpu_data[[i, j]]).abs();
                assert!(
                    diff < 1e-4,
                    "matmul[{},{}] mismatch: GPU={} CPU={} diff={}",
                    i, j, gpu_cpu[[i, j]], cpu_data[[i, j]], diff
                );
            }
        }
        println!("GPU matmul correctness [4,8]x[8,4] OK");
    }

    #[test]
    fn test_gpu_matmul_correctness_medium() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Medium matmul [64, 128] x [128, 64]
        let a_data = ArrayD::from_shape_vec(
            vec![64, 128],
            (0..8192).map(|i| (i as f32) * 0.01).collect(),
        ).unwrap();
        let b_data = ArrayD::from_shape_vec(
            vec![128, 64],
            (0..8192).map(|i| (i as f32) * 0.02).collect(),
        ).unwrap();

        let ga = GpuTensor::from_cpu(&a_data).unwrap();
        let gb = GpuTensor::from_cpu(&b_data).unwrap();
        let gpu_result = ctx.matmul(&ga, &gb).unwrap();
        let gpu_cpu = gpu_result.to_cpu();

        let a_tensor = Tensor::new(a_data);
        let b_tensor = Tensor::new(b_data);
        let cpu_result = matmul(&a_tensor, &b_tensor).unwrap();
        let cpu_data = cpu_result.data();

        // Sample check (not all elements to save time)
        for i in 0..64 {
            for j in 0..64 {
                let diff = (gpu_cpu[[i, j]] - cpu_data[[i, j]]).abs();
                assert!(
                    diff < 1e-3,
                    "matmul[{},{}] mismatch: GPU={} CPU={} diff={}",
                    i, j, gpu_cpu[[i, j]], cpu_data[[i, j]], diff
                );
            }
        }
        println!("GPU matmul correctness [64,128]x[128,64] OK");
    }

    #[test]
    fn test_gpu_matmul_square() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Square matmul [32, 32] x [32, 32]
        let a_data = ArrayD::from_shape_vec(
            vec![32, 32],
            (0..1024).map(|i| (i as f32) * 0.1).collect(),
        ).unwrap();
        let b_data = ArrayD::from_shape_vec(
            vec![32, 32],
            (0..1024).map(|i| (i as f32) * 0.05).collect(),
        ).unwrap();

        let ga = GpuTensor::from_cpu(&a_data).unwrap();
        let gb = GpuTensor::from_cpu(&b_data).unwrap();
        let gpu_result = ctx.matmul(&ga, &gb).unwrap();
        let gpu_cpu = gpu_result.to_cpu();

        let a_tensor = Tensor::new(a_data);
        let b_tensor = Tensor::new(b_data);
        let cpu_result = matmul(&a_tensor, &b_tensor).unwrap();
        let cpu_data = cpu_result.data();

        for i in 0..32 {
            for j in 0..32 {
                let diff = (gpu_cpu[[i, j]] - cpu_data[[i, j]]).abs();
                assert!(
                    diff < 1e-4,
                    "matmul[{},{}] mismatch",
                    i, j
                );
            }
        }
        println!("GPU matmul square [32,32] OK");
    }

    #[test]
    fn test_gpu_matmul_broadcast() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Test with different shapes
        let a_data = ArrayD::from_shape_vec(
            vec![16, 32],
            (0..512).map(|i| (i as f32) * 0.1).collect(),
        ).unwrap();
        let b_data = ArrayD::from_shape_vec(
            vec![32, 24],
            (0..768).map(|i| (i as f32) * 0.05).collect(),
        ).unwrap();

        let ga = GpuTensor::from_cpu(&a_data).unwrap();
        let gb = GpuTensor::from_cpu(&b_data).unwrap();
        let gpu_result = ctx.matmul(&ga, &gb).unwrap();
        let gpu_cpu = gpu_result.to_cpu();

        assert_eq!(gpu_cpu.shape(), &[16, 24]);
        println!("GPU matmul broadcast [16,32]x[32,24] OK");
    }
}
