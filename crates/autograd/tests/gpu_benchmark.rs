// GPU benchmark test — measures matmul throughput on GPU vs CPU
// Run: cargo test --features gpu -p nexora-autograd --test gpu_benchmark -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::tensor::Tensor;
    use nexora_autograd::ops::matmul::matmul;

    #[test]
    fn test_gpu_matmul_small() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        println!("GPU initialized!");
        // Small matmul [64, 64] x [64, 64]
        let a_data = ArrayD::from_shape_vec(vec![64, 64], (0..4096).map(|i| i as f32).collect()).unwrap();
        let b_data = ArrayD::from_shape_vec(vec![64, 64], (0..4096).map(|i| (i * 2) as f32).collect()).unwrap();
        let a_gpu = GpuTensor::from_cpu(&a_data).unwrap();
        let b_gpu = GpuTensor::from_cpu(&b_data).unwrap();
        let result = ctx.matmul(&a_gpu, &b_gpu).unwrap();
        let cpu = result.to_cpu();
        println!("GPU matmul [64,64] OK — first element: {}", cpu[[0, 0]]);
    }

    #[test]
    fn test_gpu_ops_roundtrip() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        // Test elementwise: a + b
        let a = ArrayD::from_shape_vec(vec![128], (0..128).map(|i| i as f32).collect()).unwrap();
        let b = ArrayD::from_shape_vec(vec![128], (0..128).map(|i| (i * 3) as f32).collect()).unwrap();
        let ga = GpuTensor::from_cpu(&a).unwrap();
        let gb = GpuTensor::from_cpu(&b).unwrap();
        let sum = ctx.add(&ga, &gb).unwrap();
        let sum_cpu = sum.to_cpu();
        assert!((sum_cpu[[0]] - 0.0).abs() < 1e-5, "sum[0] should be 0");
        assert!((sum_cpu[[1]] - 4.0).abs() < 1e-5, "sum[1] should be 4");
        println!("GPU elementwise add OK");

        // Test reduce: sum
        let sum_all = ctx.reduce(&ga, nexora_autograd::gpu::ReduceOp::Sum).unwrap();
        let sum_all_cpu = sum_all.to_cpu();
        let expected: f32 = (0..128).sum::<i32>() as f32;
        assert!((sum_all_cpu[[0]] - expected).abs() < 1.0, "sum_all mismatch");
        println!("GPU reduce sum OK");

        // Test fill_zero
        ctx.fill_zero(&ga).unwrap();
        let zeroed = ga.to_cpu();
        assert!(zeroed[[0]].abs() < 1e-6, "fill_zero failed");
        assert!(zeroed[[127]].abs() < 1e-6, "fill_zero failed");
        println!("GPU fill_zero OK");
    }

    #[test]
    fn test_gpu_matmul_benchmark() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let sizes = [128, 256, 512, 1024];
        for &n in &sizes {
            let a_data = ArrayD::from_shape_vec(vec![n, n], (0..n*n).map(|i| (i % 100) as f32).collect()).unwrap();
            let b_data = ArrayD::from_shape_vec(vec![n, n], (0..n*n).map(|i| ((i * 3) % 100) as f32).collect()).unwrap();
            let ga = GpuTensor::from_cpu(&a_data).unwrap();
            let gb = GpuTensor::from_cpu(&b_data).unwrap();

            let start = std::time::Instant::now();
            let iterations = match n {
                128 => 100,
                256 => 50,
                512 => 20,
                1024 => 5,
                _ => 1,
            };
            for _ in 0..iterations {
                let _ = ctx.matmul(&ga, &gb).unwrap();
            }
            let elapsed = start.elapsed();
            let avg = elapsed / iterations as u32;
            let gflops = 2.0 * (n as f64).powi(3) / avg.as_secs_f64() / 1e9;
            println!("GPU matmul [{n:>4}x{n:>4}] x {iterations:>3} = {avg:?} avg, {gflops:.2} GFLOPS");
        }
    }

    #[test]
    fn test_gpu_roundtrip_simple() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        // Simple buffer roundtrip: write 4 f32 values, read them back via elementwise op
        let data = ArrayD::from_shape_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ones = ArrayD::from_shape_vec(vec![4], vec![10.0, 10.0, 10.0, 10.0]).unwrap();
        let a = GpuTensor::from_cpu(&data).unwrap();
        let b = GpuTensor::from_cpu(&ones).unwrap();
        let sum = ctx.add(&a, &b).unwrap();
        let cpu = sum.to_cpu();
        println!("add result: {:?}", cpu);
        assert!((cpu[[0]] - 11.0).abs() < 1e-5);
        assert!((cpu[[3]] - 14.0).abs() < 1e-5);
        println!("GPU roundtrip simple OK");

        // Now test softmax
        let logits = ArrayD::from_shape_vec(vec![1, 4], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let g = GpuTensor::from_cpu(&logits).unwrap();
        let result = ctx.softmax(&g).unwrap();
        let cpu = result.to_cpu();
        let e: Vec<f32> = vec![0.0f32, 1.0, 2.0, 3.0].iter().map(|x| x.exp()).collect();
        let s: f32 = e.iter().sum();
        println!("softmax result: {:?}", cpu);
        let expected: Vec<f32> = e.iter().map(|x| x/s).collect();
        println!("softmax expected: {:?}", expected);
        for i in 0..4 {
            assert!((cpu[[0, i]] - expected[i]).abs() < 1e-4,
                "softmax[{}] mismatch: got {}, expected {}", i, cpu[[0, i]], expected[i]);
        }
        println!("GPU softmax basic OK");
    }

    #[test]
    fn test_gpu_sampler_pipeline() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let vocab: u32 = 16;
        let batch: usize = 1;

        // Peak at token 7
        let mut data = vec![0.0f32; batch * vocab as usize];
        data[7] = 10.0;
        let logits = ArrayD::from_shape_vec(vec![batch, vocab as usize], data).unwrap();
        let gpu = GpuTensor::from_cpu(&logits).unwrap();

        // Greedy sample: top_k=1, temp=1.0
        let result = ctx.gpu_sample(&gpu, 1.0, 1, 12345).unwrap();
        // Read back as raw bytes (output is u32[], not f32[])
        let bytes = result.to_cpu_raw_bytes();
        let token = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        println!("gpu_sample top_k=1: token={token}");
        assert_eq!(token, 7, "should pick peak 7, got {token}");

        // No top-k: should still pick near 7 with high temp
        let result2 = ctx.gpu_sample(&gpu, 1.0, 0, 6789).unwrap();
        let bytes2 = result2.to_cpu_raw_bytes();
        let token2 = u32::from_ne_bytes([bytes2[0], bytes2[1], bytes2[2], bytes2[3]]);
        println!("gpu_sample top_k=0: token={token2}");

        println!("GPU sampler pipeline OK");
    }

    #[test]
    fn test_gpu_training_roundtrip() {
        use nexora_autograd::{Tensor, cross_entropy_loss, Device};
        let _ctx = GpuContext::init().expect("GPU context init failed");

        // Create CPU reference tensor, move to GPU
        let data = ArrayD::from_shape_vec(vec![8], (0..8).map(|i| i as f32).collect()).unwrap();
        let t = Tensor::new(data);
        t.set_requires_grad(true);

        // Move storage to GPU
        let gpu_storage = GpuTensor::from_cpu(&t.data()).unwrap();
        t.set_storage(nexora_autograd::Storage::Gpu(gpu_storage));
        t.set_device(Device::Gpu(0));

        // Verify tensor is on GPU
        match t.storage() {
            nexora_autograd::Storage::Gpu(_) => println!("Tensor is on GPU ✓"),
            _ => panic!("Tensor should be on GPU"),
        }
        println!("GPU training roundtrip OK");
    }
}
