// Integration test: Full training step on GPU (forward + backward + optimizer)
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_training_step -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::gpu_adam::GpuAdam;
    use nexora_autograd::tensor::Tensor;
    use nexora_autograd::Device;
    use nexora_autograd::ops;

    #[test]
    fn test_gpu_training_step_simple() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Simple linear layer: W[4,8], bias[8]
        let w_data = ArrayD::from_shape_vec(
            vec![4, 8],
            (0..32).map(|i| (i as f32) * 0.1).collect(),
        ).unwrap();
        let bias_data = ArrayD::from_shape_vec(vec![8], vec![0.1; 8]).unwrap();
        let x_data = ArrayD::from_shape_vec(
            vec![2, 4],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ).unwrap();
        let target_data = ArrayD::from_shape_vec(vec![2], vec![3.0, 7.0]).unwrap();

        // Create tensors
        let mut w = Tensor::new(w_data);
        w.set_requires_grad(true);
        let mut bias = Tensor::new(bias_data);
        bias.set_requires_grad(true);
        let x = Tensor::new(x_data);
        let target = Tensor::new(target_data);

        // Move to GPU
        let gw = GpuTensor::from_cpu(&w.data()).unwrap();
        let gbias = GpuTensor::from_cpu(&bias.data()).unwrap();
        let gx = GpuTensor::from_cpu(&x.data()).unwrap();
        w.set_storage(nexora_autograd::Storage::Gpu(gw));
        w.set_device(Device::Gpu(0));
        bias.set_storage(nexora_autograd::Storage::Gpu(gbias));
        bias.set_device(Device::Gpu(0));
        x.set_storage(nexora_autograd::Storage::Gpu(gx));
        x.set_device(Device::Gpu(0));

        // Forward
        let logits = ops::matmul(&x, &w);
        let logits_bias = ops::add(&logits, &bias);
        let loss = ops::cross_entropy_loss(&logits_bias, &target);

        // Backward
        loss.backward();

        // Check gradients
        assert!(w.grad().is_some(), "W should have gradient");
        assert!(bias.grad().is_some(), "bias should have gradient");
        
        println!("GPU training step (forward+backward) OK");
    }

    #[test]
    fn test_gpu_training_step_with_optimizer() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Linear layer
        let w_data = ArrayD::from_shape_vec(
            vec![4, 8],
            (0..32).map(|i| (i as f32) * 0.1).collect(),
        ).unwrap();
        let bias_data = ArrayD::from_shape_vec(vec![8], vec![0.0; 8]).unwrap();
        let x_data = ArrayD::from_shape_vec(
            vec![2, 4],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ).unwrap();
        let target_data = ArrayD::from_shape_vec(vec![2], vec![0.0, 1.0]).unwrap();

        let mut w = Tensor::new(w_data);
        w.set_requires_grad(true);
        let mut bias = Tensor::new(bias_data);
        bias.set_requires_grad(true);
        let x = Tensor::new(x_data);
        let target = Tensor::new(target_data);

        // Move to GPU
        let gw = GpuTensor::from_cpu(&w.data()).unwrap();
        let gbias = GpuTensor::from_cpu(&bias.data()).unwrap();
        let gx = GpuTensor::from_cpu(&x.data()).unwrap();

        // Clone before moving into storage so optimizer keeps references
        let gw_opt = gw.clone();
        let gbias_opt = gbias.clone();
        w.set_storage(nexora_autograd::Storage::Gpu(gw));
        w.set_device(Device::Gpu(0));
        bias.set_storage(nexora_autograd::Storage::Gpu(gbias));
        bias.set_device(Device::Gpu(0));
        x.set_storage(nexora_autograd::Storage::Gpu(gx));
        x.set_device(Device::Gpu(0));

        // Create optimizer
        let mut adam = GpuAdam::new(&ctx, &[&gw_opt, &gbias_opt], 0.01).unwrap();

        // Training loop: 3 steps
        let initial_loss = {
            let logits = ops::matmul(&x, &w);
            let logits_bias = ops::add(&logits, &bias);
            let loss = ops::cross_entropy_loss(&logits_bias, &target);
            loss.data()[[0]]
        };

        for step in 0..3 {
            // Forward
            let logits = ops::matmul(&x, &w);
            let logits_bias = ops::add(&logits, &bias);
            let loss = ops::cross_entropy_loss(&logits_bias, &target);

            // Backward
            loss.backward();

            // Get gradients as GPU tensors
            let w_grad_arr = w.grad().unwrap();
            let bias_grad_arr = bias.grad().unwrap();
            let gw_grad = GpuTensor::from_cpu(&w_grad_arr).unwrap();
            let gbias_grad = GpuTensor::from_cpu(&bias_grad_arr).unwrap();

            // Optimizer step
            adam.step(&ctx, &[&gw_opt, &gbias_opt], &[&gw_grad, &gbias_grad]).unwrap();

            // Update parameters
            let w_updated = gw_opt.to_cpu().unwrap();
            let bias_updated = gbias_opt.to_cpu().unwrap();
            w.set_data(ArrayD::from_shape_vec(vec![4, 8], w_updated.into_raw_vec()).unwrap());
            bias.set_data(ArrayD::from_shape_vec(vec![8], bias_updated.into_raw_vec()).unwrap());

            // Zero gradients
            w.zero_grad();
            bias.zero_grad();

            println!("Step {} loss: {}", step, loss.data()[[0]]);
        }

        // Loss should decrease
        let final_loss = {
            let logits = ops::matmul(&x, &w);
            let logits_bias = ops::add(&logits, &bias);
            let loss = ops::cross_entropy_loss(&logits_bias, &target);
            loss.data()[[0]]
        };

        assert!(
            final_loss < initial_loss,
            "Loss should decrease: initial={} final={}",
            initial_loss, final_loss
        );
        
        println!("GPU training step with optimizer OK - loss decreased from {} to {}", 
                 initial_loss, final_loss);
    }
}
