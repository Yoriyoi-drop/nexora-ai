    use super::*;
    use ndarray::Array1;
    use crate::TransformerConfig;
    use nexora_quantization::QFormat;

    fn small_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            max_seq_len: 64,
            intermediate_size: 64,
            rope_theta: 10000.0,
            use_cache: true,
            norm_eps: 1e-6,
            num_experts: 0,
            top_k_experts: 0,
            expert_intermediate_size: 0,
            quantization: QFormat::F16,
            use_half_precision: true,
            shard: Default::default(),
            shared_expert: 0,
            use_domain_experts: false,
        }
    }

    fn small_model() -> CausalLM {
        CausalLM::new(small_config())
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = Array1::from(vec![2.0, 1.0, 0.1, -1.0, 3.0]);
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_softmax_uniform_when_all_equal() {
        let logits = Array1::from(vec![5.0; 10]);
        let probs = softmax(&logits);
        for &p in probs.iter() {
            assert!((p - 0.1).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_nan_input() {
        let logits = Array1::from(vec![f32::NEG_INFINITY, f32::NEG_INFINITY]);
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_token_greedy() {
        let logits = Array1::from(vec![1.0, 5.0, 2.0, 0.5, 3.0]);
        let token = sample_token(&logits, 0.0, 0);
        assert_eq!(token, 1); // highest logit at index 1
    }

    #[test]
    fn test_sample_token_greedy_last() {
        let logits = Array1::from(vec![0.1, 0.2, 0.3, 10.0]);
        let token = sample_token(&logits, 0.0, 0);
        assert_eq!(token, 3);
    }

    #[test]
    fn test_sample_token_temperature_zero_is_greedy() {
        let logits = Array1::from(vec![0.0, 0.0, 100.0, 0.0]);
        let token = sample_token(&logits, 0.0, 0);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_token_top_k() {
        let logits = Array1::from(vec![0.0, 10.0, 0.0, 9.0, 0.0]);
        // With top_k=2, should sample from {1, 3}
        for _ in 0..20 {
            let token = sample_token(&logits, 1.0, 2);
            assert!(
                token == 1 || token == 3,
                "top_k=2 should only pick from top 2, got {}",
                token
            );
        }
    }

    #[test]
    fn test_causal_lm_new_shapes() {
        let cfg = small_config();
        let model = small_model();
        assert_eq!(model.token_embedding.as_ref().unwrap().dim(), (100, 32));
        assert_eq!(model.lm_head.as_ref().unwrap().dim(), (100, 32));
        assert_eq!(model.norm.weight.as_ref().unwrap().len(), 32);
        assert_eq!(model.blocks.len(), 2);
        assert_eq!(
            model.precomputed_cos.len(),
            cfg.max_seq_len * cfg.head_dim() / 2
        );
        assert_eq!(
            model.precomputed_sin.len(),
            cfg.max_seq_len * cfg.head_dim() / 2
        );
    }

    #[test]
    fn test_causal_lm_forward_shape() {
        let mut model = small_model();
        model.set_keep_on_gpu(false);
        let mut cache = model.reset_cache();
        let logits = model.forward(&[0, 1], &mut cache).unwrap();
        assert_eq!(logits.len(), 100);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_causal_lm_forward_single_token() {
        let mut model = small_model();
        model.set_keep_on_gpu(false);
        let mut cache = model.reset_cache();
        let logits = model.forward(&[5], &mut cache).unwrap();
        assert_eq!(logits.len(), 100);
    }

    #[test]
    fn test_causal_lm_forward_out_of_range_token() {
        let model = small_model();
        let mut cache = model.reset_cache();
        let result = model.forward(&[999], &mut cache);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_returns_tokens() {
        let model = small_model();
        let (tokens, cache) = model.generate_with_gpu(&[0, 1], 10, 1.0, 0, false);
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= 10);
        if !cache.is_empty() {
            assert!(!cache[0].k.is_empty());
        }
    }

    #[test]
    fn test_generate_single_token() {
        let model = small_model();
        let (tokens, _) = model.generate_with_gpu(&[5], 5, 1.0, 0, false);
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= 5);
    }

    #[test]
    fn test_parameter_count() {
        let model = small_model();
        let count = model.parameter_count();
        let cfg = small_config();
        let expected = cfg.parameter_count();
        assert_eq!(count, expected);
    }

    #[test]
    fn test_reset_cache() {
        let model = small_model();
        let cache = model.reset_cache();
        assert_eq!(cache.num_layers(), 0);
    }

    #[test]
    fn test_memory_bytes() {
        let model = small_model();
        let bytes = model.memory_bytes();
        let expected = model.parameter_count() as f64 * model.bytes_per_param();
        assert!((bytes - expected).abs() < 1.0);
    }

    #[test]
    fn test_keep_on_gpu_default() {
        let model = small_model();
        #[cfg(feature = "gpu")]
        assert_eq!(model.keep_on_gpu, nexora_autograd::gpu::GpuContext::is_available());
        #[cfg(not(feature = "gpu"))]
        assert!(!model.keep_on_gpu);
    }

    #[test]
    fn test_set_keep_on_gpu() {
        let mut model = small_model();
        model.set_keep_on_gpu(false);
        assert!(!model.keep_on_gpu);
    }

    #[test]
    fn test_forward_paged_shape() {
        use crate::gqa::PagedCacheReader;

        struct DummyPagedCache;
        impl PagedCacheReader for DummyPagedCache {
            fn read(
                &self,
                _seq_id: u64,
                _layer: usize,
                _token_pos: usize,
            ) -> Option<(Vec<f32>, Vec<f32>)> {
                None
            }
            fn num_tokens(&self, _seq_id: u64) -> Option<usize> {
                Some(1)
            }
            fn append(
                &mut self,
                _seq_id: u64,
                _layer: usize,
                _token_pos: usize,
                _k: &[f32],
                _v: &[f32],
            ) {
            }
        }

        let model = small_model();
        let mut cache = DummyPagedCache;
        let result = model.forward_paged(&[1, 2], &mut cache, 0);
        assert!(result.is_ok());
        if let Ok(logits) = result {
            assert_eq!(logits.len(), 100);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_forward_gpu_int8_matches_cpu() {
        use crate::GpuKVCacheEntry;
        use nexora_autograd::gpu::GpuContext;

        let _ctx = GpuContext::init().expect("GPU context init failed");

        let model = small_model();
        let mut cache = model.reset_cache();
        let cpu_logits = model.forward(&[0, 1], &mut cache).unwrap();

        let mut model_gpu = small_model();
        model_gpu.quantize_weights = true;
        let mut gpu_cache: Vec<GpuKVCacheEntry> = (0..model_gpu.config.num_layers)
            .map(|_| {
                GpuKVCacheEntry::new(
                    &GpuContext::global().unwrap(),
                    model_gpu.config.num_kv_heads,
                    model_gpu.config.head_dim(),
                    model_gpu.config.max_seq_len,
                    false,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let gpu_logits = model_gpu
            .forward_gpu_with_cache(&[0, 1], &mut gpu_cache)
            .unwrap();

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(gpu_logits.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(
                diff < 1.0,
                "int8 logit mismatch at [{i}]: cpu={c} gpu={g} diff={diff}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_forward_gpu_f16_matches_cpu() {
        use crate::GpuKVCacheEntry;
        use nexora_autograd::gpu::GpuContext;

        let _ctx = GpuContext::init().expect("GPU context init failed");

        let model = small_model();
        let mut cache = model.reset_cache();
        let cpu_logits = model.forward(&[0, 1], &mut cache).unwrap();

        let mut model_gpu = small_model();
        model_gpu.use_half_precision = true;
        let mut gpu_cache: Vec<GpuKVCacheEntry> = (0..model_gpu.config.num_layers)
            .map(|_| {
                GpuKVCacheEntry::new(
                    &GpuContext::global().unwrap(),
                    model_gpu.config.num_kv_heads,
                    model_gpu.config.head_dim(),
                    model_gpu.config.max_seq_len,
                    true,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let gpu_logits = model_gpu
            .forward_gpu_with_cache(&[0, 1], &mut gpu_cache)
            .unwrap();

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(gpu_logits.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(
                diff < 0.1,
                "F16 logit mismatch at [{i}]: cpu={c} gpu={g} diff={diff}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_forward_gpu_f16_kv_cache_matches_cpu() {
        use crate::GpuKVCacheEntry;
        use nexora_autograd::gpu::GpuContext;

        let _ctx = GpuContext::init().expect("GPU context init failed");

        let model = small_model();
        let mut cache = model.reset_cache();
        let cpu_logits = model.forward(&[1, 2, 3], &mut cache).unwrap();

        let mut model_gpu = small_model();
        model_gpu.use_half_precision = true;
        let mut gpu_cache: Vec<GpuKVCacheEntry> = (0..model_gpu.config.num_layers)
            .map(|_| {
                GpuKVCacheEntry::new(
                    &GpuContext::global().unwrap(),
                    model_gpu.config.num_kv_heads,
                    model_gpu.config.head_dim(),
                    model_gpu.config.max_seq_len,
                    true,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // Multi-token prefill (3 tokens)
        let gpu_logits = model_gpu
            .forward_gpu_with_cache(&[1, 2, 3], &mut gpu_cache)
            .unwrap();

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(gpu_logits.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(
                diff < 0.1,
                "F16 KV cache logit mismatch at [{i}]: cpu={c} gpu={g} diff={diff}"
            );
        }
    }

    #[test]
    fn test_collect_weights_for_sedc() {
        let model = small_model();
        let (weights, names, fuse_pairs) = model.collect_weights_for_sedc();
        let per_block = 7;
        let expected = 2 + model.blocks.len() * per_block; // token_embedding + lm_head + blocks*7
        assert_eq!(weights.len(), expected);
        assert_eq!(names.len(), expected);
        assert_eq!(fuse_pairs.len(), model.blocks.len().saturating_sub(1));
    }
