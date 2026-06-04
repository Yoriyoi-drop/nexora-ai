// Integration test: GPU LayerNorm/RMSNorm correctness vs CPU reference
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_layernorm -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::ops::nn::{layer_norm_2d, rms_norm_2d};
    use nexora_autograd::tensor::Tensor;

    #[test]
    fn test_gpu_rmsnorm_correctness() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Input [4, 16]
        let x = ArrayD::from_shape_vec(
            vec![4, 16],
            (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect(),
        ).unwrap();
        let weight = ArrayD::from_shape_vec(
            vec![16],
            (0..16).map(|i| (i as f32) * 0.5 + 0.1).collect(),
        ).unwrap();
        let epsilon = 1e-6;

        // GPU result
        let gx = GpuTensor::from_cpu(&x).unwrap();
        let gw = GpuTensor::from_cpu(&weight).unwrap();
        let gpu_result = ctx.rms_norm(&gx, &gw, epsilon).unwrap();
        let gpu_cpu = gpu_result.to_cpu().unwrap();

        // CPU reference
        let tx = Tensor::new(x);
        let tw = Tensor::new(weight);
        let cpu_result = rms_norm_2d(&tx, &tw, epsilon);
        let cpu_data = cpu_result.data();

        for i in 0..4 {
            for j in 0..16 {
                let diff = (gpu_cpu[[i, j]] - cpu_data[[i, j]]).abs();
                assert!(
                    diff < 1e-4,
                    "rms_norm[{},{}] mismatch: GPU={} CPU={} diff={}",
                    i, j, gpu_cpu[[i, j]], cpu_data[[i, j]], diff
                );
            }
        }
        println!("GPU RMSNorm [4,16] OK");
    }

    #[test]
    fn test_gpu_layernorm_correctness() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Input [8, 32]
        let x = ArrayD::from_shape_vec(
            vec![8, 32],
            (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect(),
        ).unwrap();
        let weight = ArrayD::from_shape_vec(
            vec![32],
            (0..32).map(|i| (i as f32) * 0.3 + 0.2).collect(),
        ).unwrap();
        let bias = ArrayD::from_shape_vec(
            vec![32],
            (0..32).map(|i| (i as f32) * 0.1).collect(),
        ).unwrap();
        let epsilon = 1e-5;

        // GPU result
        let gx = GpuTensor::from_cpu(&x).unwrap();
        let gw = GpuTensor::from_cpu(&weight).unwrap();
        let gb = GpuTensor::from_cpu(&bias).unwrap();
        let gpu_result = ctx.layer_norm(&gx, &gw, &gb, epsilon).unwrap();
        let gpu_cpu = gpu_result.to_cpu().unwrap();

        // CPU reference
        let tx = Tensor::new(x);
        let tw = Tensor::new(weight);
        let tb = Tensor::new(bias);
        let cpu_result = layer_norm_2d(&tx, Some(&tw), Some(&tb), epsilon);
        let cpu_data = cpu_result.data();

        for i in 0..8 {
            for j in 0..32 {
                let diff = (gpu_cpu[[i, j]] - cpu_data[[i, j]]).abs();
                assert!(
                    diff < 1e-4,
                    "layer_norm[{},{}] mismatch",
                    i, j
                );
            }
        }
        println!("GPU LayerNorm [8,32] OK");
    }

    #[test]
    fn test_gpu_rmsnorm_zero_mean() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Test RMSNorm property: output should have zero mean after weighting
        let x = ArrayD::from_shape_vec(
            vec![2, 8],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        ).unwrap();
        let weight = ArrayD::from_shape_vec(vec![8], vec![1.0; 8]).unwrap();
        
        let gx = GpuTensor::from_cpu(&x).unwrap();
        let gw = GpuTensor::from_cpu(&weight).unwrap();
        let gpu_result = ctx.rms_norm(&gx, &gw, 1e-6).unwrap();
        let gpu_cpu = gpu_result.to_cpu().unwrap();

        // Check that each row is normalized
        for i in 0..2 {
            let mut sum_sq = 0.0f32;
            for j in 0..8 {
                sum_sq += gpu_cpu[[i, j]] * gpu_cpu[[i, j]];
            }
            let rms = (sum_sq / 8.0).sqrt();
            assert!(
                (rms - 1.0).abs() < 1e-4,
                "RMSNorm row {} has RMS {}, expected ~1.0",
                i, rms
            );
        }
        println!("GPU RMSNorm normalization property OK");
    }
}
