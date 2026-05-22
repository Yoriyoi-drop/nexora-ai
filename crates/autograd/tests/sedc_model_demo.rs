// SEDC Integration Demo — compress NXR model weights and measure results.
// Run: cargo test --features gpu -p nexora-autograd --test sedc_model_demo -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rand::Rng;

    /// NXR Omnis model weight shapes (hidden_size=768)
    fn omnis_shapes() -> Vec<(&'static str, usize, usize)> {
        vec![
            ("attention.wq", 768, 768),
            ("attention.wo", 768, 768),
            ("attention.wk", 256, 768),
            ("attention.wv", 256, 768),
            ("ffn.w1", 3072, 768),
            ("ffn.w2", 768, 3072),
            ("ffn.w3", 3072, 768),
            ("token_embedding", 256, 768),
            ("lm_head", 256, 768),
        ]
    }

    fn low_rank_matrix(m: usize, n: usize, effective_rank: usize) -> Array2<f32> {
        let mut rng = rand::thread_rng();
        let a = Array2::from_shape_fn((m, effective_rank), |(_, _)| rng.gen::<f32>() * 2.0 - 1.0);
        let b = Array2::from_shape_fn((effective_rank, n), |(_, _)| rng.gen::<f32>() * 2.0 - 1.0);
        a.dot(&b)
    }

    fn full_rank_matrix(m: usize, n: usize) -> Array2<f32> {
        let mut rng = rand::thread_rng();
        Array2::from_shape_fn((m, n), |(_, _)| rng.gen::<f32>() * 2.0 - 1.0)
    }

    #[test]
    fn demo_sedc_on_omnis_weights() {
        use nexora_autograd::gpu_sedc::{SedcConfig, SedcCompressor};

        let shapes = omnis_shapes();
        let config = SedcConfig {
            alpha: 0.85,
            rbs_bits: 2,
            oversamples: 10,
            power_iters: 2,
            tolerance: 0.05,
            min_fuse_ratio: 1.1,
        };

        println!("\n═══════════════════════════════════════════════════════════════");
        println!("  SEDC Compression Demo — NXR Model Weights");
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Config: alpha={}, oversamples={}, power_iters={}, tolerance={}",
            config.alpha, config.oversamples, config.power_iters, config.tolerance);
        println!("───────────────────────────────────────────────────────────────");
        println!("  {:<22} {:<14} {:<14} {:<10} {:<10} {:<8}",
            "Layer", "Shape", "Original P.", "Rank", "Compressed P.", "Ratio");
        println!("───────────────────────────────────────────────────────────────");

        let compressor = SedcCompressor::new(config);
        let mut total_orig = 0usize;
        let mut total_comp = 0usize;
        let mut total_error = 0.0_f32;
        let mut count = 0u32;

        for (name, m, n) in &shapes {
            let w = if *m > 1000 {
                let effective_rank = (*m).min(*n) / 4;
                low_rank_matrix(*m, *n, effective_rank)
            } else {
                full_rank_matrix(*m, *n)
            };

            let result = compressor.compress_weight(&w, 0).unwrap();

            let original_params = m * n;
            let compressed_params = result.rank * (m + n);
            let ratio = if compressed_params > 0 {
                original_params as f32 / compressed_params as f32
            } else {
                1.0
            };

            total_orig += original_params;
            total_comp += compressed_params;
            total_error += result.relative_error;
            count += 1;

            println!("  {:<22} {:>4}×{:<8} {:<14} {:<10} {:<10} {:<8.2}×",
                name, m, n,
                original_params,
                result.rank,
                compressed_params,
                ratio,
            );
        }

        let avg_error = total_error / count as f32;
        let total_ratio = total_orig as f32 / total_comp.max(1) as f32;

        println!("───────────────────────────────────────────────────────────────");
        println!("  {:<22} {:<14} {:<14} {:<10} {:<10} {:<8.2}×",
            "TOTAL", "", total_orig, "", total_comp, total_ratio);
        println!("  Avg relative error: {:.4}", avg_error);
        println!("═══════════════════════════════════════════════════════════════\n");

        assert!(total_ratio >= 1.0, "SEDC should compress, got ratio {:.2}", total_ratio);
    }
}
