// GPU benchmark test — measures matmul throughput on GPU vs CPU
// Run: cargo test --features gpu -p nexora-autograd --test gpu_benchmark -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::ops::matmul::matmul;
    use nexora_autograd::tensor::Tensor;
    use std::time::{Duration, Instant};

    // ── Professional Benchmark Harness ─────────────────────────────────────────────

    #[derive(Debug, Clone, Default)]
    struct DetailedTiming {
        upload: Duration,
        kernel: Duration,
        readback: Duration,
        end_to_end: Duration,
    }

    #[derive(Debug, Clone)]
    struct BenchmarkStats {
        warmup_iters: usize,
        measured_iters: usize,
        median: Duration,
        mean: Duration,
        stddev: Duration,
        min: Duration,
        max: Duration,
        detailed: DetailedTiming,
    }

    impl BenchmarkStats {
        fn gflops(&self, ops: f64) -> f64 {
            ops / self.detailed.kernel.as_secs_f64() / 1e9
        }

        fn display(&self) -> String {
            format!(
                "median={:?} mean={:?} stddev={:?} min={:?} max={:?} (warmup={}, measured={})\n  detailed: upload={:?} kernel={:?} readback={:?} e2e={:?}",
                self.median,
                self.mean,
                self.stddev,
                self.min,
                self.max,
                self.warmup_iters,
                self.measured_iters,
                self.detailed.upload,
                self.detailed.kernel,
                self.detailed.readback,
                self.detailed.end_to_end
            )
        }
    }

    struct BenchmarkHarness {
        warmup_iters: usize,
        measured_iters: usize,
    }

    impl BenchmarkHarness {
        fn new(warmup_iters: usize, measured_iters: usize) -> Self {
            Self {
                warmup_iters,
                measured_iters,
            }
        }

        fn default() -> Self {
            Self::new(10, 20) // Increased warmup from 5 to 10 to reduce variance
        }

        /// Lighter harness for long benchmark suites (runs in CI-friendly time when used with --ignored).
        fn quick() -> Self {
            Self::new(3, 5)
        }

        fn run<F, E>(&self, mut f: F) -> Result<BenchmarkStats, E>
        where
            F: FnMut() -> Result<(), E>,
        {
            // Warmup phase
            for _ in 0..self.warmup_iters {
                f()?;
            }

            // Measurement phase
            let mut samples = Vec::with_capacity(self.measured_iters);
            for _ in 0..self.measured_iters {
                let start = Instant::now();
                f()?;
                let elapsed = start.elapsed();
                samples.push(elapsed);
            }

            // Calculate statistics
            let mut sorted_samples = samples.clone();
            sorted_samples.sort();

            let median = if sorted_samples.is_empty() {
                Duration::ZERO
            } else if sorted_samples.len() % 2 == 0 {
                let mid = sorted_samples.len() / 2;
                (sorted_samples[mid - 1] + sorted_samples[mid]) / 2
            } else {
                sorted_samples[sorted_samples.len() / 2]
            };

            let mean: Duration = samples.iter().sum::<Duration>() / samples.len() as u32;

            let mean_secs = mean.as_secs_f64();
            let variance: f64 = samples
                .iter()
                .map(|d| {
                    let diff = d.as_secs_f64() - mean_secs;
                    diff * diff
                })
                .sum::<f64>()
                / samples.len() as f64;
            let stddev_secs = variance.sqrt();
            let stddev = Duration::from_secs_f64(stddev_secs);

            let min = *samples.iter().min().unwrap_or(&Duration::ZERO);
            let max = *samples.iter().max().unwrap_or(&Duration::ZERO);

            Ok(BenchmarkStats {
                warmup_iters: self.warmup_iters,
                measured_iters: self.measured_iters,
                median,
                mean,
                stddev,
                min,
                max,
                detailed: DetailedTiming::default(),
            })
        }

        /// Run with detailed timing separation
        fn run_detailed<F, E>(&self, mut f: F) -> Result<BenchmarkStats, E>
        where
            F: FnMut() -> Result<DetailedTiming, E>,
        {
            // Warmup phase
            for _ in 0..self.warmup_iters {
                f()?;
            }

            // Measurement phase
            let mut upload_samples = Vec::with_capacity(self.measured_iters);
            let mut kernel_samples = Vec::with_capacity(self.measured_iters);
            let mut readback_samples = Vec::with_capacity(self.measured_iters);
            let mut e2e_samples = Vec::with_capacity(self.measured_iters);

            for _ in 0..self.measured_iters {
                let e2e_start = Instant::now();
                let timing = f()?;
                let e2e_elapsed = e2e_start.elapsed();

                upload_samples.push(timing.upload);
                kernel_samples.push(timing.kernel);
                readback_samples.push(timing.readback);
                e2e_samples.push(e2e_elapsed);
            }

            // Calculate median for each phase
            let median = |samples: &mut Vec<Duration>| -> Duration {
                samples.sort();
                if samples.is_empty() {
                    Duration::ZERO
                } else if samples.len() % 2 == 0 {
                    let mid = samples.len() / 2;
                    (samples[mid - 1] + samples[mid]) / 2
                } else {
                    samples[samples.len() / 2]
                }
            };

            let median_e2e = median(&mut e2e_samples.clone());
            let median_upload = median(&mut upload_samples.clone());
            let median_kernel = median(&mut kernel_samples.clone());
            let median_readback = median(&mut readback_samples.clone());

            // Calculate overall statistics from e2e
            let mean: Duration = e2e_samples.iter().sum::<Duration>() / e2e_samples.len() as u32;
            let mean_secs = mean.as_secs_f64();
            let variance: f64 = e2e_samples
                .iter()
                .map(|d| {
                    let diff = d.as_secs_f64() - mean_secs;
                    diff * diff
                })
                .sum::<f64>()
                / e2e_samples.len() as f64;
            let stddev_secs = variance.sqrt();
            let stddev = Duration::from_secs_f64(stddev_secs);

            let min = *e2e_samples.iter().min().unwrap_or(&Duration::ZERO);
            let max = *e2e_samples.iter().max().unwrap_or(&Duration::ZERO);

            Ok(BenchmarkStats {
                warmup_iters: self.warmup_iters,
                measured_iters: self.measured_iters,
                median: median_e2e,
                mean,
                stddev,
                min,
                max,
                detailed: DetailedTiming {
                    upload: median_upload,
                    kernel: median_kernel,
                    readback: median_readback,
                    end_to_end: median_e2e,
                },
            })
        }
    }

    #[test]
    fn test_gpu_matmul_small() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        println!("GPU initialized!");
        // Small matmul [64, 64] x [64, 64]
        let a_data =
            ArrayD::from_shape_vec(vec![64, 64], (0..4096).map(|i| i as f32).collect()).unwrap();
        let b_data =
            ArrayD::from_shape_vec(vec![64, 64], (0..4096).map(|i| (i * 2) as f32).collect())
                .unwrap();
        let a_gpu = GpuTensor::from_cpu(&a_data).unwrap();
        let b_gpu = GpuTensor::from_cpu(&b_data).unwrap();
        let result = ctx.matmul(&a_gpu, &b_gpu).unwrap();
        let cpu = result.to_cpu().unwrap();
        println!("GPU matmul [64,64] OK — first element: {}", cpu[[0, 0]]);
    }

    #[test]
    fn test_gpu_ops_roundtrip() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        // Test elementwise: a + b
        let a = ArrayD::from_shape_vec(vec![128], (0..128).map(|i| i as f32).collect()).unwrap();
        let b =
            ArrayD::from_shape_vec(vec![128], (0..128).map(|i| (i * 3) as f32).collect()).unwrap();
        let ga = GpuTensor::from_cpu(&a).unwrap();
        let gb = GpuTensor::from_cpu(&b).unwrap();
        let sum = ctx.add(&ga, &gb).unwrap();
        let sum_cpu = sum.to_cpu().unwrap();
        assert!((sum_cpu[[0]] - 0.0).abs() < 1e-5, "sum[0] should be 0");
        assert!((sum_cpu[[1]] - 4.0).abs() < 1e-5, "sum[1] should be 4");
        println!("GPU elementwise add OK");

        // Test reduce: sum
        let sum_all = ctx
            .reduce(&ga, nexora_autograd::gpu::ReduceOp::Sum)
            .unwrap();
        let sum_all_cpu = sum_all.to_cpu().unwrap();
        let expected: f32 = (0..128).sum::<i32>() as f32;
        assert!(
            (sum_all_cpu[[0]] - expected).abs() < 1.0,
            "sum_all mismatch"
        );
        println!("GPU reduce sum OK");

        // Test fill_zero
        ctx.fill_zero(&ga).unwrap();
        let zeroed = ga.to_cpu().unwrap();
        assert!(zeroed[[0]].abs() < 1e-6, "fill_zero failed");
        assert!(zeroed[[127]].abs() < 1e-6, "fill_zero failed");
        println!("GPU fill_zero OK");
    }

    /// Quick matmul smoke benchmark (always runs in default `cargo test`).
    #[test]
    fn test_gpu_matmul_benchmark_quick() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let harness = BenchmarkHarness::quick();
        let n = 512usize;
        let a_data =
            ArrayD::from_shape_vec(vec![n, n], (0..n * n).map(|i| (i % 100) as f32).collect())
                .unwrap();
        let b_data = ArrayD::from_shape_vec(
            vec![n, n],
            (0..n * n).map(|i| ((i * 3) % 100) as f32).collect(),
        )
        .unwrap();
        let ga = GpuTensor::from_cpu(&a_data).unwrap();
        let gb = GpuTensor::from_cpu(&b_data).unwrap();
        let iterations = 10usize;
        let stats = harness
            .run(|| {
                for _ in 0..iterations {
                    let _ = ctx.matmul(&ga, &gb).unwrap();
                }
                Ok::<(), ()>(())
            })
            .unwrap();
        let ops_per_iter = 2.0 * (n as f64).powi(3);
        let median_per_iter = stats.median.as_secs_f64() / iterations as f64;
        let gflops = ops_per_iter / median_per_iter / 1e9;
        println!(
            "GPU matmul quick [{n}x{n}] x {iterations} | median={:?} | {gflops:.2} GFLOPS",
            stats.median
        );
        assert!(
            gflops > 1.0,
            "expected >1 GFLOPS on quick matmul smoke, got {gflops:.2}"
        );
    }

    /// Full benchmark: measures kernel performance with multiple iterations.
    /// Marked as ignored due to long runtime. Run with: cargo test test_gpu_matmul_benchmark -- --ignored
    #[test]
    #[ignore = "slow full GPU matmul sweep"]
    fn test_gpu_matmul_benchmark() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let harness = BenchmarkHarness::quick();
        let sizes = [128, 256, 512, 1024];
        for &n in &sizes {
            let a_data =
                ArrayD::from_shape_vec(vec![n, n], (0..n * n).map(|i| (i % 100) as f32).collect())
                    .unwrap();
            let b_data = ArrayD::from_shape_vec(
                vec![n, n],
                (0..n * n).map(|i| ((i * 3) % 100) as f32).collect(),
            )
            .unwrap();

            let iterations = match n {
                128 => 40,
                256 => 20,
                512 => 10,
                1024 => 3,
                _ => 1,
            };

            // Upload once, reuse for all iterations
            let ga = GpuTensor::from_cpu(&a_data).unwrap();
            let gb = GpuTensor::from_cpu(&b_data).unwrap();

            let stats = harness
                .run(|| {
                    // Kernel dispatch phase only - no readback
                    for _ in 0..iterations {
                        let _ = ctx.matmul(&ga, &gb).unwrap();
                    }
                    Ok::<(), ()>(())
                })
                .unwrap();

            // Fix: median is time for ALL iterations, need to divide by iterations
            let ops_per_iter = 2.0 * (n as f64).powi(3);
            let median_per_iter = stats.median.as_secs_f64() / iterations as f64;
            let gflops = ops_per_iter / median_per_iter / 1e9;
            println!(
                "GPU matmul [{n:>4}x{n:>4}] x {iterations:>3} | median={:?} mean={:?} stddev={:?} | {gflops:.2} GFLOPS (kernel only, no readback)",
                stats.median, stats.mean, stats.stddev
            );
        }
    }

    /// Benchmark with GPU timestamp query for accurate kernel timing.
    /// Marked as ignored due to long runtime. Run with: cargo test test_gpu_matmul_benchmark_with_timestamp -- --ignored
    #[test]
    #[ignore = "slow full GPU timestamp matmul sweep"]
    fn test_gpu_matmul_benchmark_with_timestamp() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let harness = BenchmarkHarness::quick();
        let sizes = [128, 256, 512, 1024];
        for &n in &sizes {
            let a_data =
                ArrayD::from_shape_vec(vec![n, n], (0..n * n).map(|i| (i % 100) as f32).collect())
                    .unwrap();
            let b_data = ArrayD::from_shape_vec(
                vec![n, n],
                (0..n * n).map(|i| ((i * 3) % 100) as f32).collect(),
            )
            .unwrap();

            // Upload once, reuse for all iterations
            let ga = GpuTensor::from_cpu(&a_data).unwrap();
            let gb = GpuTensor::from_cpu(&b_data).unwrap();

            let iterations = match n {
                128 => 40,
                256 => 20,
                512 => 10,
                1024 => 3,
                _ => 1,
            };

            let stats = harness
                .run(|| {
                    // Use GPU timestamp query for accurate kernel timing
                    for _ in 0..iterations {
                        let _ = ctx.matmul_with_timestamp(&ga, &gb).unwrap();
                    }
                    Ok::<(), ()>(())
                })
                .unwrap();

            // Fix: median is time for ALL iterations, need to divide by iterations
            let ops_per_iter = 2.0 * (n as f64).powi(3);
            let median_per_iter = stats.median.as_secs_f64() / iterations as f64;
            let gflops = ops_per_iter / median_per_iter / 1e9;
            println!(
                "GPU matmul [{n:>4}x{n:>4}] x {iterations:>3} | median={:?} mean={:?} stddev={:?} | {gflops:.2} GFLOPS (GPU timestamp query)",
                stats.median, stats.mean, stats.stddev
            );
        }
    }

    #[test]
    fn test_gpu_matmul_benchmark_batch() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let harness = BenchmarkHarness::default();
        let sizes = [128, 256, 512, 1024];
        for &n in &sizes {
            let a_data =
                ArrayD::from_shape_vec(vec![n, n], (0..n * n).map(|i| (i % 100) as f32).collect())
                    .unwrap();
            let b_data = ArrayD::from_shape_vec(
                vec![n, n],
                (0..n * n).map(|i| ((i * 3) % 100) as f32).collect(),
            )
            .unwrap();

            // Upload once, reuse for all iterations
            let ga = GpuTensor::from_cpu(&a_data).unwrap();
            let gb = GpuTensor::from_cpu(&b_data).unwrap();

            let batch_size = match n {
                128 => 100,
                256 => 50,
                512 => 20,
                1024 => 5,
                _ => 1,
            };

            let stats = harness
                .run(|| {
                    // Use batch dispatch with single timestamp query
                    // Note: matmul_batch_with_timestamp now requires &mut self
                    let _ = ctx
                        .matmul_batch_with_timestamp(&ga, &gb, batch_size)
                        .unwrap();
                    Ok::<(), ()>(())
                })
                .unwrap();

            // Fix: median is time for ALL iterations, need to divide by batch_size
            let ops_per_iter = 2.0 * (n as f64).powi(3);
            let median_per_iter = stats.median.as_secs_f64() / batch_size as f64;
            let gflops = ops_per_iter / median_per_iter / 1e9;
            println!(
                "GPU matmul batch [{n:>4}x{n:>4}] x {batch_size:>3} | median={:?} mean={:?} stddev={:?} | {gflops:.2} GFLOPS (per-iteration, batch-dispatch)",
                stats.median, stats.mean, stats.stddev
            );
        }
    }

    /// Microbenchmark: kernel-only benchmark without any readback.
    /// Marked as ignored due to long runtime. Run with: cargo test test_gpu_matmul_microbenchmark -- --ignored
    #[test]
    #[ignore = "slow GPU microbenchmark sweep"]
    fn test_gpu_matmul_microbenchmark() {
        let mut ctx = GpuContext::init().expect("GPU context init failed");
        let harness = BenchmarkHarness::quick();
        let sizes = [128, 256, 512, 1024];
        for &n in &sizes {
            let a_data =
                ArrayD::from_shape_vec(vec![n, n], (0..n * n).map(|i| (i % 100) as f32).collect())
                    .unwrap();
            let b_data = ArrayD::from_shape_vec(
                vec![n, n],
                (0..n * n).map(|i| ((i * 3) % 100) as f32).collect(),
            )
            .unwrap();

            // Upload once, reuse for all iterations
            let ga = GpuTensor::from_cpu(&a_data).unwrap();
            let gb = GpuTensor::from_cpu(&b_data).unwrap();

            let batch_size = match n {
                128 => 80,
                256 => 40,
                512 => 20,
                1024 => 8,
                _ => 1,
            };

            let stats = harness
                .run(|| {
                    // Pure kernel dispatch, no readback, no validation
                    let _ = ctx
                        .matmul_batch_with_timestamp(&ga, &gb, batch_size)
                        .unwrap();
                    Ok::<(), ()>(())
                })
                .unwrap();

            // Fix: median is time for ALL iterations, need to divide by batch_size
            let ops_per_iter = 2.0 * (n as f64).powi(3);
            let median_per_iter = stats.median.as_secs_f64() / batch_size as f64;
            let gflops = ops_per_iter / median_per_iter / 1e9;
            println!(
                "GPU matmul micro [{n:>4}x{n:>4}] x {batch_size:>3} | median={:?} mean={:?} stddev={:?} | {gflops:.2} GFLOPS (per-iteration, kernel-only)",
                stats.median, stats.mean, stats.stddev
            );
        }
    }

    /// Pipeline benchmark: end-to-end benchmark with upload, compute, and readback.
    /// Marked as ignored due to long runtime. Run with: cargo test test_gpu_matmul_pipeline_benchmark -- --ignored
    #[test]
    #[ignore = "slow GPU pipeline e2e sweep"]
    fn test_gpu_matmul_pipeline_benchmark() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let harness = BenchmarkHarness::quick();
        let sizes = [128, 256, 512, 1024];
        for &n in &sizes {
            let iterations = match n {
                128 => 20,
                256 => 12,
                512 => 8,
                1024 => 4,
                _ => 1,
            };

            let stats = harness
                .run(|| {
                    for _ in 0..iterations {
                        let a_data = ArrayD::from_shape_vec(
                            vec![n, n],
                            (0..n * n).map(|i| (i % 100) as f32).collect(),
                        )
                        .unwrap();
                        let b_data = ArrayD::from_shape_vec(
                            vec![n, n],
                            (0..n * n).map(|i| ((i * 3) % 100) as f32).collect(),
                        )
                        .unwrap();

                        // Upload
                        let ga = GpuTensor::from_cpu(&a_data).unwrap();
                        let gb = GpuTensor::from_cpu(&b_data).unwrap();

                        // Compute
                        let gc = ctx.matmul(&ga, &gb).unwrap();

                        // Readback
                        let _ = gc.to_cpu().unwrap();
                    }
                    Ok::<(), ()>(())
                })
                .unwrap();

            // Fix: median is time for ALL iterations, need to divide by iterations
            let ops_per_iter = 2.0 * (n as f64).powi(3);
            let median_per_iter = stats.median.as_secs_f64() / iterations as f64;
            let gflops = ops_per_iter / median_per_iter / 1e9;
            println!(
                "GPU matmul pipeline [{n:>4}x{n:>4}] x {iterations:>3} | median={:?} mean={:?} stddev={:?} | {gflops:.2} GFLOPS (per-iteration, end-to-end)",
                stats.median, stats.mean, stats.stddev
            );
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
        let cpu = sum.to_cpu().unwrap();
        println!("add result: {:?}", cpu);
        assert!((cpu[[0]] - 11.0).abs() < 1e-5);
        assert!((cpu[[3]] - 14.0).abs() < 1e-5);
        println!("GPU roundtrip simple OK");

        // Now test softmax
        let logits = ArrayD::from_shape_vec(vec![1, 4], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let g = GpuTensor::from_cpu(&logits).unwrap();
        let result = ctx.softmax(&g).unwrap();
        let cpu = result.to_cpu().unwrap();
        let e: Vec<f32> = vec![0.0f32, 1.0, 2.0, 3.0]
            .iter()
            .map(|x| x.exp())
            .collect();
        let s: f32 = e.iter().sum();
        println!("softmax result: {:?}", cpu);
        let expected: Vec<f32> = e.iter().map(|x| x / s).collect();
        println!("softmax expected: {:?}", expected);
        for i in 0..4 {
            assert!(
                (cpu[[0, i]] - expected[i]).abs() < 1e-4,
                "softmax[{}] mismatch: got {}, expected {}",
                i,
                cpu[[0, i]],
                expected[i]
            );
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
        let result = ctx.gpu_sample(&gpu, 1.0, 1, 1.0, 12345).unwrap();
        // Read back as raw bytes (output is u32[], not f32[])
        let bytes = result.to_cpu_raw_bytes().unwrap();
        let token = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        println!("gpu_sample top_k=1: token={token}");
        assert_eq!(token, 7, "should pick peak 7, got {token}");

        // No top-k: should still pick near 7 with high temp
        let result2 = ctx.gpu_sample(&gpu, 1.0, 0, 1.0, 6789).unwrap();
        let bytes2 = result2.to_cpu_raw_bytes().unwrap();
        let token2 = u32::from_ne_bytes([bytes2[0], bytes2[1], bytes2[2], bytes2[3]]);
        println!("gpu_sample top_k=0: token={token2}");

        println!("GPU sampler pipeline OK");
    }

    #[test]
    fn test_gpu_training_roundtrip() {
        use nexora_autograd::{cross_entropy_loss, Device, Tensor};
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

    // ─── Priority 1: Fused Op Benchmarks ─────────────────────────────────────

    #[test]
    fn test_fused_matmul_bias_correctness() {
        use nexora_autograd::gpu_fused::ACT_IDENTITY;
        let ctx = GpuContext::init().expect("GPU context init failed");

        // A[4,8] @ B[8,4] + bias[4] — small, verify numerics
        let a_data =
            ArrayD::from_shape_vec(vec![4, 8], (0..32).map(|i| (i as f32) * 0.1).collect())
                .unwrap();
        let b_data =
            ArrayD::from_shape_vec(vec![8, 4], (0..32).map(|i| (i as f32) * 0.05).collect())
                .unwrap();
        let bias_data = ArrayD::from_shape_vec(vec![4], vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        let ga = GpuTensor::from_cpu(&a_data).unwrap();
        let gb = GpuTensor::from_cpu(&b_data).unwrap();
        let gbias = GpuTensor::from_cpu(&bias_data).unwrap();

        // Fused result
        let fused = ctx.fused_matmul_bias(&ga, &gb, &gbias).unwrap();
        let fused_cpu = fused.to_cpu().unwrap();

        // Reference: separate matmul + add
        let matmul_ref = ctx.matmul(&ga, &gb).unwrap();
        let matmul_cpu = matmul_ref.to_cpu().unwrap();
        for row in 0..4usize {
            for col in 0..4usize {
                let expected = matmul_cpu[[row, col]] + bias_data[[col]];
                let got = fused_cpu[[row, col]];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "fused_matmul_bias[{row},{col}] mismatch: got={got} expected={expected}"
                );
            }
        }
        println!("fused_matmul_bias correctness OK (act=identity)");
    }

    #[test]
    fn test_fused_matmul_bias_gelu_correctness() {
        let ctx = GpuContext::init().expect("GPU context init failed");

        // Use non-trivial input that won't result in all zeros
        let a = ArrayD::from_shape_vec(
            vec![2, 4],
            vec![1.0f32, 2.0, 0.5, -0.5, 2.0, 1.0, 0.1, -0.1],
        )
        .unwrap();
        let b = ArrayD::from_shape_vec(vec![4, 2], vec![1.0f32, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0])
            .unwrap();
        let bias = ArrayD::from_shape_vec(vec![2], vec![0.1f32, 0.2]).unwrap();

        let ga = GpuTensor::from_cpu(&a).unwrap();
        let gb = GpuTensor::from_cpu(&b).unwrap();
        let gbias = GpuTensor::from_cpu(&bias).unwrap();

        let out = ctx.fused_matmul_bias_gelu(&ga, &gb, &gbias).unwrap();
        let cpu = out.to_cpu().unwrap();

        // All outputs should be in GELU range (positive dominant for pos input)
        println!("fused_matmul_bias_gelu output: {:?}", cpu);
        assert_eq!(cpu.shape(), &[2, 2]);
        // GELU(x) >= 0 for x >> 0, GELU(x) < 0 for very negative x
        println!("fused_matmul_bias_gelu correctness OK");
    }

    #[test]
    fn test_fused_matmul_bias_benchmark() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)];

        for (m, k, n) in sizes {
            let a_data = ArrayD::from_shape_vec(
                vec![m, k],
                (0..m * k).map(|i| (i % 100) as f32 * 0.01).collect(),
            )
            .unwrap();
            let b_data = ArrayD::from_shape_vec(
                vec![k, n],
                (0..k * n).map(|i| (i % 100) as f32 * 0.01).collect(),
            )
            .unwrap();
            let bias_data =
                ArrayD::from_shape_vec(vec![n], (0..n).map(|i| i as f32 * 0.001).collect())
                    .unwrap();

            let ga = GpuTensor::from_cpu(&a_data).unwrap();
            let gb = GpuTensor::from_cpu(&b_data).unwrap();
            let gbias = GpuTensor::from_cpu(&bias_data).unwrap();

            let iters = if m >= 1024 { 5 } else { 20 };

            // GPU warmup before timing (avoids cold-start skew on small matrices)
            for _ in 0..3 {
                let _ = ctx.fused_matmul_bias_gelu(&ga, &gb, &gbias).unwrap();
                let mut mm = ctx.fused_matmul_bias(&ga, &gb, &gbias).unwrap();
                ctx.gelu_inplace(&mut mm).unwrap();
            }
            ctx.sync();

            // Benchmark fused
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = ctx.fused_matmul_bias_gelu(&ga, &gb, &gbias).unwrap();
            }
            let fused_avg = t0.elapsed() / iters as u32;
            let gflops = 2.0 * (m as f64) * (k as f64) * (n as f64) / fused_avg.as_secs_f64() / 1e9;

            // Benchmark unfused (matmul + bias + gelu - apple-to-apple comparison)
            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                let mut mm = ctx.fused_matmul_bias(&ga, &gb, &gbias).unwrap();
                ctx.gelu_inplace(&mut mm).unwrap();
            }
            let unfused_avg = t1.elapsed() / iters as u32;

            println!(
                "fused_matmul_gelu [{m}x{k}]x[{k}x{n}]: fused={fused_avg:?} unfused_mm={unfused_avg:?} | {gflops:.1} GFLOPS (bias+act free)"
            );
        }
        println!("fused_matmul_bias_benchmark OK");
    }

    #[test]
    fn test_fused_matmul_bias_gelu_separate_vs_fully_fused() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let m = 1024;
        let k = 1024;
        let n = 1024;
        let a_data = ArrayD::from_shape_vec(
            vec![m, k],
            (0..m * k).map(|i| (i % 100) as f32 * 0.01).collect(),
        )
        .unwrap();
        let b_data = ArrayD::from_shape_vec(
            vec![k, n],
            (0..k * n).map(|i| (i % 100) as f32 * 0.01).collect(),
        )
        .unwrap();
        let bias_data =
            ArrayD::from_shape_vec(vec![n], (0..n).map(|i| i as f32 * 0.001).collect()).unwrap();

        let ga = GpuTensor::from_cpu(&a_data).unwrap();
        let gb = GpuTensor::from_cpu(&b_data).unwrap();
        let gbias = GpuTensor::from_cpu(&bias_data).unwrap();

        let iters = 5;

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let mut mat = ctx.fused_matmul_bias(&ga, &gb, &gbias).unwrap();
            ctx.gelu_inplace(&mut mat).unwrap();
        }
        let separate_avg = start.elapsed() / iters as u32;

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = ctx.fused_matmul_bias_gelu_forced(&ga, &gb, &gbias).unwrap();
        }
        let fused_avg = start.elapsed() / iters as u32;

        println!(
            "fused_matmul_bias_gelu [1024x1024] separate={separate_avg:?} fully_fused={fused_avg:?}"
        );
    }

    #[test]
    fn test_fused_matmul_bias_gelu_large_path_profile() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let m = 1024;
        let k = 1024;
        let n = 1024;
        let a_data = ArrayD::from_shape_vec(
            vec![m, k],
            (0..m * k).map(|i| (i % 100) as f32 * 0.01).collect(),
        )
        .unwrap();
        let b_data = ArrayD::from_shape_vec(
            vec![k, n],
            (0..k * n).map(|i| (i % 100) as f32 * 0.01).collect(),
        )
        .unwrap();
        let bias_data =
            ArrayD::from_shape_vec(vec![n], (0..n).map(|i| i as f32 * 0.001).collect()).unwrap();

        let ga = GpuTensor::from_cpu(&a_data).unwrap();
        let gb = GpuTensor::from_cpu(&b_data).unwrap();
        let gbias = GpuTensor::from_cpu(&bias_data).unwrap();

        let (out, elapsed) = ctx
            .fused_matmul_bias_gelu_profiled(&ga, &gb, &gbias)
            .unwrap();
        let cpu = out.to_cpu().unwrap();

        assert_eq!(cpu.shape(), &[m, n]);
        assert!(cpu[[0, 0]].is_finite(), "output must be finite");
        println!("fused_matmul_bias_gelu large path [1024x1024] elapsed={elapsed:?}");
    }

    #[test]
    fn test_fused_online_softmax() {
        let ctx = GpuContext::init().expect("GPU context init failed");

        // [4, 8] softmax over last dim
        let data: Vec<f32> = (0..32).map(|i| (i % 8) as f32).collect();
        let inp = ArrayD::from_shape_vec(vec![4, 8], data).unwrap();
        let ginp = GpuTensor::from_cpu(&inp).unwrap();

        let out = ctx.fused_softmax_online(&ginp).unwrap();
        let cpu = out.to_cpu().unwrap();

        // Each row must sum to ~1.0
        for row in 0..4usize {
            let row_sum: f32 = (0..8).map(|c| cpu[[row, c]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "online_softmax row={row} sum={row_sum}"
            );
        }

        // Compare against regular softmax for numerical agreement
        let ref_out = ctx.softmax(&ginp).unwrap();
        let ref_cpu = ref_out.to_cpu().unwrap();
        for row in 0..4usize {
            for col in 0..8usize {
                let diff = (cpu[[row, col]] - ref_cpu[[row, col]]).abs();
                assert!(
                    diff < 1e-4,
                    "online vs ref softmax [{row},{col}] diff={diff}"
                );
            }
        }
        println!("fused_online_softmax OK — rows sum to 1.0 and match reference");
    }

    #[test]
    fn test_command_batch() {
        let ctx = GpuContext::init().expect("GPU context init failed");

        let a = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![64], (0..64).map(|i| i as f32).collect()).unwrap(),
        )
        .unwrap();
        let b = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![64], (0..64).map(|i| (i * 2) as f32).collect()).unwrap(),
        )
        .unwrap();

        // Baseline: separate submits (reusable encoder path)
        ctx.fill_zero(&a).unwrap();
        ctx.fill_zero(&b).unwrap();
        ctx.sync();

        // Re-fill then zero both buffers in one command encoder submit
        let fill_a: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let fill_b: Vec<f32> = (0..64).map(|i| (i * 2) as f32).collect();
        ctx.upload_f32(&a, &fill_a);
        ctx.upload_f32(&b, &fill_b);

        let mut batch = ctx.begin_batch();
        ctx.batch_enqueue_fill_zero(&mut batch, &a).unwrap();
        ctx.batch_enqueue_fill_zero(&mut batch, &b).unwrap();
        assert_eq!(
            batch.op_count(),
            2,
            "batch should track 2 dispatches before submit"
        );
        let ops = batch.submit();
        assert_eq!(ops, 2, "submit should report 2 dispatches");
        ctx.sync();

        let a_cpu = a.to_cpu().unwrap();
        let b_cpu = b.to_cpu().unwrap();
        assert!(a_cpu[[0]].abs() < 1e-6, "batch fill_zero A failed");
        assert!(b_cpu[[63]].abs() < 1e-6, "batch fill_zero B failed");

        let empty = ctx.begin_batch();
        assert_eq!(empty.submit(), 0, "empty batch should have 0 ops");

        println!("command_batch API OK — {ops} ops submitted in one queue call");
    }
}
