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
    use nexora_autograd::ops::losses::cross_entropy_loss;

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
        let target_data = ArrayD::from_shape_vec(vec![2], vec![3, 7]).unwrap();

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
        let logits = x.matmul(&w).unwrap();
        let logits_bias = logits.add(&bias).unwrap();
        let loss = cross_entropy_loss(&logits_bias, &target).unwrap();

        // Backward
        loss.backward(None).unwrap();

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
        let target_data = ArrayD::from_shape_vec(vec![2], vec![0, 1]).unwrap();

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

        // Create optimizer
        let mut adam = GpuAdam::new(
            vec![gw.clone(), gbias.clone()],
            0.01,
            0.9,
            0.999,
            1e-8,
            0.0,
        );

        // Training loop: 3 steps
        let initial_loss = {
            let logits = x.matmul(&w).unwrap();
            let logits_bias = logits.add(&bias).unwrap();
            let loss = cross_entropy_loss(&logits_bias, &target).unwrap();
            loss.data()[[0]]
        };

        for step in 0..3 {
            // Forward
            let logits = x.matmul(&w).unwrap();
            let logits_bias = logits.add(&bias).unwrap();
            let loss = cross_entropy_loss(&logits_bias, &target).unwrap();

            // Backward
            loss.backward(None).unwrap();

            // Get gradients as GPU tensors
            let gw_grad = GpuTensor::from_cpu(&w.grad().unwrap().data()).unwrap();
            let gbias_grad = GpuTensor::from_cpu(&bias.grad().unwrap().data()).unwrap();

            // Optimizer step
            adam.step(&ctx, &[gw.clone(), gbias.clone()], &[gw_grad, gbias_grad]).unwrap();

            // Update parameters
            let w_updated = gw.to_cpu();
            let bias_updated = gbias.to_cpu();
            w.set_data(ArrayD::from(w_updated.into_raw_vec()).into_shape(vec![4, 8]).unwrap());
            bias.set_data(ArrayD::from(bias_updated.into_raw_vec()).into_shape(vec![8]).unwrap());

            // Zero gradients
            w.zero_grad();
            bias.zero_grad();

            println!("Step {} loss: {}", step, loss.data()[[0]]);
        }

        // Loss should decrease
        let final_loss = {
            let logits = x.matmul(&w).unwrap();
            let logits_bias = logits.add(&bias).unwrap();
            let loss = cross_entropy_loss(&logits_bias, &target).unwrap();
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
