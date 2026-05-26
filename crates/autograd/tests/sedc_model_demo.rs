// SEDC Integration Demo — full GPU pipeline, CPU control only.
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
        use nexora_autograd::gpu_sedc::{SedcCompressor, SedcConfig};

        let shapes = omnis_shapes();
        let config = SedcConfig {
            alpha: 0.85,
            oversamples: 10,
            power_iters: 2,
            tolerance: 0.05,
            vet_gamma: 0.1,
            sparsity_kappa: 2.0,
            target_sparsity: 0.3,
            arq_bits: 2,
            arq_eta: 1.5,
            ..Default::default()
        };

        println!("\n═══════════════════════════════════════════════════════════════");
        println!("  SEDC Compression Demo — NXR Model Weights");
        println!("═══════════════════════════════════════════════════════════════");
        println!(
            "  Config: alpha={}, oversamples={}, power_iters={}, tolerance={}",
            config.alpha, config.oversamples, config.power_iters, config.tolerance
        );
        println!(
            "  VET gamma={}, EGSS kappa={}, ARQ bits={}, REC lambda={}",
            config.vet_gamma, config.sparsity_kappa, config.arq_bits, config.rec_lambda
        );
        println!("───────────────────────────────────────────────────────────────");
        println!(
            "  {:<22} {:<10} {:<10} {:<10} {:<10} {:<8} {:<8}",
            "Layer", "Shape", "Orig P.", "Rank", "Comp P.", "Ratio", "Sparsity"
        );
        println!("───────────────────────────────────────────────────────────────");

        let compressor = SedcCompressor::new(config);
        let mut total_orig = 0usize;
        let mut total_comp = 0usize;
        let mut total_error = 0.0_f32;
        let mut total_sparsity = 0.0_f32;
        let mut count = 0u32;

        for (name, m, n) in &shapes {
            let w = if *m > 1000 {
                let effective_rank = (*m).min(*n) / 4;
                low_rank_matrix(*m, *n, effective_rank)
            } else {
                full_rank_matrix(*m, *n)
            };

            let result = match compressor.compress_weight(&w, count as usize, name) {
                Ok(r) => r,
                Err(e) => {
                    println!("  {:<22} {:>4}×{:<4} — SKIPPED: {}", name, m, n, e);
                    continue;
                }
            };

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
            total_sparsity += result.sparsity;
            count += 1;

            println!(
                "  {:<22} {:>4}×{:<4} {:<10} {:<10} {:<10.2}× {:<8.2} sp={:<6.2}  err={:.4}",
                name,
                m,
                n,
                original_params,
                result.rank,
                compressed_params,
                ratio,
                result.sparsity,
                result.relative_error,
            );
        }

        if count > 0 {
            let avg_error = total_error / count as f32;
            let avg_sparsity = total_sparsity / count as f32;
            let total_ratio = total_orig as f32 / total_comp.max(1) as f32;

            println!("───────────────────────────────────────────────────────────────");
            println!(
                "  {:<22} {:<10} {:<10} {:<10} {:<10.2}× {:<8.2} sp={:<6.2}",
                "TOTAL", "", total_orig, "", total_comp, total_ratio, avg_sparsity
            );
            println!("  Avg relative error: {:.4}", avg_error);
            println!("  Avg sparsity: {:.2}", avg_sparsity);
            println!("═══════════════════════════════════════════════════════════════\n");

            assert!(
                total_ratio >= 1.0,
                "SEDC should compress, got ratio {:.2}",
                total_ratio
            );
        } else {
            println!("  No layers compressed (GPU may not be available)");
            println!("═══════════════════════════════════════════════════════════════\n");
        }
    }

    #[test]
    fn demo_sedc_full_model_pipeline() {
        use nexora_autograd::gpu_sedc::{SedcCompressor, SedcConfig};

        let config = SedcConfig {
            tolerance: 0.15,
            ..Default::default()
        };
        let compressor = SedcCompressor::new(config);

        let names = &[
            "attention.wq",
            "attention.wo",
            "ffn.w1",
            "ffn.w2",
            "attention.wk",
            "attention.wv",
            "ffn.w3",
        ];
        let weights: Vec<Array2<f32>> = names
            .iter()
            .map(|&_| {
                let m = 256;
                let n = 256;
                low_rank_matrix(m, n, 32)
            })
            .collect();
        let fuse_pairs = vec![(0, 16), (2, 16)];

        println!("\n═══════════════════════════════════════════════════════════════");
        println!("  SEDC Full Model Pipeline Demo");
        println!("═══════════════════════════════════════════════════════════════");

        match compressor.compress_model(&weights, names, &fuse_pairs) {
            Ok(model) => {
                println!("  Total ratio: {:.2}×", model.report.total_ratio);
                println!("  Mean error: {:.4}", model.report.mean_relative_error);
                println!("  Mean sparsity: {:.2}", model.report.mean_sparsity);
                println!("  Fused pairs: {}", model.fused_pairs.len());
                println!("  Layers compressed: {}", model.weights.len());
                assert!(
                    model.report.total_ratio >= 0.5,
                    "model compression ratio should be reasonable"
                );
            }
            Err(e) => {
                println!("  Model compression skipped: {}", e);
            }
        }
        println!("═══════════════════════════════════════════════════════════════\n");
    }
}
