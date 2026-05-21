// Integration test: GPU Adam optimizer correctness
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_adam -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::gpu_adam::GpuAdam;
    use nexora_autograd::tensor::Tensor;
    use nexora_autograd::Device;

    #[test]
    fn test_gpu_adam_step() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Simple parameter [4]
        let param_data = ArrayD::from_shape_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grad_data = ArrayD::from_shape_vec(vec![4], vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        // Create GPU tensors
        let gparam = GpuTensor::from_cpu(&param_data).unwrap();
        let ggrad = GpuTensor::from_cpu(&grad_data).unwrap();

        // Create Adam optimizer
        let mut adam = GpuAdam::new(
            vec![gparam.clone()],
            0.001,  // lr
            0.9,    // beta1
            0.999,  // beta2
            1e-8,   // eps
            0.0,    // weight_decay
        );

        // Perform step
        adam.step(&ctx, &[gparam], &[ggrad]).unwrap();

        // Parameters should have changed
        let updated = gparam.to_cpu();
        for i in 0..4 {
            assert!(
                (updated[[i]] - param_data[[i]]).abs() > 1e-6,
                "param[{}] should have changed",
                i
            );
        }
        
        println!("GPU Adam step OK");
    }

    #[test]
    fn test_gpu_adam_multiple_steps() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Parameter [8]
        let param_data = ArrayD::from_shape_vec(
            vec![8],
            (0..8).map(|i| (i as f32) * 0.5 + 1.0).collect(),
        ).unwrap();
        let grad_data = ArrayD::from_shape_vec(
            vec![8],
            (0..8).map(|i| (i as f32) * 0.1 + 0.05).collect(),
        ).unwrap();

        let gparam = GpuTensor::from_cpu(&param_data).unwrap();
        let ggrad = GpuTensor::from_cpu(&grad_data).unwrap();

        let mut adam = GpuAdam::new(
            vec![gparam.clone()],
            0.01,   // lr
            0.9,    // beta1
            0.999,  // beta2
            1e-8,   // eps
            0.01,   // weight_decay
        );

        // Perform 5 steps
        for _ in 0..5 {
            adam.step(&ctx, &[gparam.clone()], &[ggrad.clone()]).unwrap();
        }

        let updated = gparam.to_cpu();
        println!("GPU Adam 5 steps OK - params changed from {:?} to {:?}", 
                 param_data.as_slice().unwrap(), 
                 updated.as_slice().unwrap());
    }

    #[test]
    fn test_gpu_adam_weight_decay() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Test weight decay effect
        let param_data = ArrayD::from_shape_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grad_data = ArrayD::from_shape_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap();

        let gparam = GpuTensor::from_cpu(&param_data).unwrap();
        let ggrad = GpuTensor::from_cpu(&grad_data).unwrap();

        // Adam with weight decay
        let mut adam = GpuAdam::new(
            vec![gparam.clone()],
            0.001,
            0.9,
            0.999,
            1e-8,
            0.1,  // high weight decay
        );

        adam.step(&ctx, &[gparam], &[ggrad]).unwrap();

        let updated = gparam.to_cpu();
        // With weight decay and zero gradient, params should decay toward zero
        for i in 0..4 {
            assert!(
                updated[[i]] < param_data[[i]],
                "param[{}] should decay with weight decay",
                i
            );
        }
        
        println!("GPU Adam weight decay OK");
    }
}
