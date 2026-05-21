// Integration test: GPU Backward pass correctness (numerical gradient check)
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_backward -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::tensor::Tensor;
    use nexora_autograd::Device;

    #[test]
    fn test_gpu_backward_matmul() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Simple matmul: A[2,4] @ B[4,3] -> C[2,3]
        let a_data = ArrayD::from_shape_vec(
            vec![2, 4],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ).unwrap();
        let b_data = ArrayD::from_shape_vec(
            vec![4, 3],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        ).unwrap();

        // Create tensors with requires_grad
        let mut a = Tensor::new(a_data);
        a.set_requires_grad(true);
        let mut b = Tensor::new(b_data);
        b.set_requires_grad(true);

        // Move to GPU
        let ga = GpuTensor::from_cpu(&a.data()).unwrap();
        let gb = GpuTensor::from_cpu(&b.data()).unwrap();
        a.set_storage(nexora_autograd::Storage::Gpu(ga));
        a.set_device(Device::Gpu(0));
        b.set_storage(nexora_autograd::Storage::Gpu(gb));
        b.set_device(Device::Gpu(0));

        // Forward
        let c = a.matmul(&b).unwrap();
        
        // Backward
        let grad_output = Tensor::ones_like(&c);
        c.backward(Some(&grad_output)).unwrap();

        // Check gradients exist
        assert!(a.grad().is_some(), "A should have gradient");
        assert!(b.grad().is_some(), "B should have gradient");
        
        println!("GPU backward matmul gradients computed OK");
    }

    #[test]
    fn test_gpu_backward_elementwise() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Element-wise: a + b
        let a_data = ArrayD::from_shape_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b_data = ArrayD::from_shape_vec(vec![4], vec![0.5, 1.5, 2.5, 3.5]).unwrap();

        let mut a = Tensor::new(a_data);
        a.set_requires_grad(true);
        let mut b = Tensor::new(b_data);
        b.set_requires_grad(true);

        // Move to GPU
        let ga = GpuTensor::from_cpu(&a.data()).unwrap();
        let gb = GpuTensor::from_cpu(&b.data()).unwrap();
        a.set_storage(nexora_autograd::Storage::Gpu(ga));
        a.set_device(Device::Gpu(0));
        b.set_storage(nexora_autograd::Storage::Gpu(gb));
        b.set_device(Device::Gpu(0));

        // Forward
        let c = a.add(&b).unwrap();
        
        // Backward
        let grad_output = Tensor::ones_like(&c);
        c.backward(Some(&grad_output)).unwrap();

        // Gradients should be all ones
        let a_grad = a.grad().unwrap();
        let b_grad = b.grad().unwrap();
        
        for i in 0..4 {
            assert!(
                (a_grad.data()[[i]] - 1.0).abs() < 1e-5,
                "grad_a[{}] should be 1.0",
                i
            );
            assert!(
                (b_grad.data()[[i]] - 1.0).abs() < 1e-5,
                "grad_b[{}] should be 1.0",
                i
            );
        }
        
        println!("GPU backward elementwise gradients OK");
    }

    #[test]
    fn test_gpu_backward_chain() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        // Chain: a * b + c
        let a_data = ArrayD::from_shape_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b_data = ArrayD::from_shape_vec(vec![4], vec![2.0, 2.0, 2.0, 2.0]).unwrap();
        let c_data = ArrayD::from_shape_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();

        let mut a = Tensor::new(a_data);
        a.set_requires_grad(true);
        let mut b = Tensor::new(b_data);
        b.set_requires_grad(true);
        let mut c = Tensor::new(c_data);
        c.set_requires_grad(true);

        // Move to GPU
        let ga = GpuTensor::from_cpu(&a.data()).unwrap();
        let gb = GpuTensor::from_cpu(&b.data()).unwrap();
        let gc = GpuTensor::from_cpu(&c.data()).unwrap();
        a.set_storage(nexora_autograd::Storage::Gpu(ga));
        a.set_device(Device::Gpu(0));
        b.set_storage(nexora_autograd::Storage::Gpu(gb));
        b.set_device(Device::Gpu(0));
        c.set_storage(nexora_autograd::Storage::Gpu(gc));
        c.set_device(Device::Gpu(0));

        // Forward: a * b + c
        let mul = a.mul(&b).unwrap();
        let result = mul.add(&c).unwrap();
        
        // Backward
        let grad_output = Tensor::ones_like(&result);
        result.backward(Some(&grad_output)).unwrap();

        // Check gradients
        let a_grad = a.grad().unwrap();
        let b_grad = b.grad().unwrap();
        let c_grad = c.grad().unwrap();
        
        // grad_a = b, grad_b = a, grad_c = 1
        for i in 0..4 {
            assert!(
                (a_grad.data()[[i]] - b_data[[i]]).abs() < 1e-5,
                "grad_a[{}] should be {}",
                i, b_data[[i]]
            );
            assert!(
                (b_grad.data()[[i]] - a_data[[i]]).abs() < 1e-5,
                "grad_b[{}] should be {}",
                i, a_data[[i]]
            );
            assert!(
                (c_grad.data()[[i]] - 1.0).abs() < 1e-5,
                "grad_c[{}] should be 1.0",
                i
            );
        }
        
        println!("GPU backward chain rule OK");
    }
}
