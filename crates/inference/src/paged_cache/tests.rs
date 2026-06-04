use super::*;



    fn test_config() -> PagedCacheConfig {
        PagedCacheConfig {
            block_size: 4,
            max_blocks: 64,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            max_seq_len: 64,
            f16_storage: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_register_and_remove_sequence() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(1);
        assert!(cache.has_sequence(1));
        cache.remove_sequence(1);
        assert!(!cache.has_sequence(1));
    }

    #[test]
    fn test_append_and_read_single_block() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(42);

        let k_row = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let v_row = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7];

        cache.append(42, 0, 0, &k_row, &v_row);

        let (k_read, v_read) = cache.read(42, 0, 0).unwrap();
        assert_eq!(k_read.len(), 8);
        assert!((k_read[0] - 0.1).abs() < 1e-6);
        assert!((v_read[0] - 1.0).abs() < 1e-6);

        assert_eq!(cache.num_tokens(42), Some(1));
    }

    #[test]
    fn test_get_or_alloc_block_sequential() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(99);

        // pos=0: should allocate block 0
        let b0 = cache.get_or_alloc_block(99, 0, 0);
        assert_eq!(b0, Some(0));

        // pos=1: should reuse block 0
        let b1 = cache.get_or_alloc_block(99, 0, 1);
        assert_eq!(b1, Some(0));

        // pos=4: should allocate block 1
        let b4 = cache.get_or_alloc_block(99, 0, 4);
        assert_eq!(b4, Some(1));

        // pos=5: should reuse block 1
        let b5 = cache.get_or_alloc_block(99, 0, 5);
        assert_eq!(b5, Some(1));
    }

    #[test]
    fn test_append_multiple_blocks() {
        let config = test_config();
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(7);

        // Fill 5 tokens (= 2 blocks: first block 4 tokens, second block 1 token)
        for pos in 0..5 {
            // Each position writes k_row where k_row[0] = f32::from(pos) / 10.0
            let k_row: Vec<f32> = (0..8).map(|i| (pos * 8 + i) as f32 / 10.0).collect();
            let v_row: Vec<f32> = (0..8).map(|i| (pos * 8 + i) as f32).collect();
            cache.append(7, 0, pos, &k_row, &v_row);
        }

        assert_eq!(cache.num_tokens(7), Some(5));

        // Verify all tokens readable
        for pos in 0..5 {
            let (k, v) = cache.read(7, 0, pos).unwrap();
            let expected_k0 = (pos * 8) as f32 / 10.0;
            let expected_v0 = (pos * 8) as f32;
            assert!(
                (k[0] - expected_k0).abs() < 1e-6,
                "pos={}: expected k[0]={}, got k[0]={}",
                pos,
                expected_k0,
                k[0]
            );
            assert!(
                (v[0] - expected_v0).abs() < 1e-6,
                "pos={}: expected v[0]={}, got v[0]={}",
                pos,
                expected_v0,
                v[0]
            );
        }
    }

    #[test]
    fn test_to_flat_cache() {
        let config = test_config();
        let mut cache = PagedKVCache::new(config);
        cache.register_sequence(99);

        let k_row: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let v_row: Vec<f32> = (0..8).map(|i| i as f32 * 10.0).collect();

        for pos in 0..3 {
            cache.append(99, 0, pos, &k_row, &v_row);
            cache.append(99, 1, pos, &k_row, &v_row);
        }

        let flat = cache.to_flat_cache(99).unwrap();
        assert_eq!(flat.len(), 2); // 2 layers
        assert_eq!(flat[0].k.len(), 3 * 8); // 3 tokens * 8 cols
        assert_eq!(flat[0].v.len(), 3 * 8);
    }

    #[test]
    fn test_remove_sequence_frees_blocks() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(1);
        cache.register_sequence(2);

        let k_row = vec![0.0; 8];
        let v_row = vec![0.0; 8];

        for pos in 0..8 {
            cache.append(1, 0, pos, &k_row, &v_row);
            cache.append(2, 0, pos, &k_row, &v_row);
        }

        let stats_before = cache.stats();
        assert_eq!(stats_before.num_sequences, 2);

        cache.remove_sequence(1);

        let stats_after = cache.stats();
        assert_eq!(stats_after.num_sequences, 1);
        assert_eq!(stats_after.total_blocks, stats_before.total_blocks); // blocks still allocated
    }

    #[test]
    fn test_clear_all() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(1);
        cache.register_sequence(2);

        let k_row = vec![1.0; 8];
        let v_row = vec![2.0; 8];
        cache.append(1, 0, 0, &k_row, &v_row);
        cache.append(2, 0, 0, &k_row, &v_row);

        cache.clear();
        assert_eq!(cache.stats().num_sequences, 0);
        assert_eq!(cache.total_blocks(), 0);
    }

    #[test]
    fn test_memory_usage() {
        let mut config = test_config();
        config.f16_storage = false; // use f32 for predictable sizing in this test
        let cache = PagedKVCache::new(config.clone());
        let per_block = config.block_size * config.num_kv_heads * config.head_dim * 4 * 2;
        assert!(cache.memory_usage_bytes() == 0);

        // After allocating blocks, memory should increase
        let mut cache2 = PagedKVCache::new(config);
        cache2.register_sequence(1);
        let k_row = vec![0.0; 8];
        let v_row = vec![0.0; 8];
        for pos in 0..4 {
            cache2.append(1, 0, pos, &k_row, &v_row);
            cache2.append(1, 1, pos, &k_row, &v_row);
        }
        // One block per layer (2 layers) = 2 blocks total
        assert_eq!(cache2.total_blocks(), 2);
        assert!(cache2.memory_usage_bytes() >= 2 * per_block);
    }

    #[test]
    fn test_f16_memory_half() {
        let mut config = test_config();
        let per_block = config.block_size * config.num_kv_heads * config.head_dim * 4 * 2;
        config.f16_storage = true;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(1);
        let k_row = vec![0.0; 8];
        let v_row = vec![0.0; 8];
        for pos in 0..4 {
            cache.append(1, 0, pos, &k_row, &v_row);
            cache.append(1, 1, pos, &k_row, &v_row);
        }

        let f16_bytes = cache.memory_usage_bytes();

        // Compare with f32 version
        let mut f32_config = test_config();
        f32_config.f16_storage = false;
        let mut f32_cache = PagedKVCache::new(f32_config);
        f32_cache.register_sequence(1);
        for pos in 0..4 {
            f32_cache.append(1, 0, pos, &k_row, &v_row);
            f32_cache.append(1, 1, pos, &k_row, &v_row);
        }
        let f32_bytes = f32_cache.memory_usage_bytes();

        assert!(
            f16_bytes < f32_bytes,
            "f16 memory ({f16_bytes}) should be less than f32 ({f32_bytes})"
        );
        let data_savings = f32_bytes - f16_bytes;
        assert!(
            data_savings >= per_block / 2,
            "f16 data savings ({data_savings}) should be at least half a block ({})",
            per_block / 2
        );
    }

    #[test]
    fn test_append_token_from_array() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(1);

        let k_arr = Array2::from_shape_vec((1, 8), (0..8).map(|i| i as f32).collect()).unwrap();
        let v_arr =
            Array2::from_shape_vec((1, 8), (0..8).map(|i| i as f32 * 2.0).collect()).unwrap();

        cache.append_token(1, 0, 0, &k_arr, &v_arr);

        let (k, v) = cache.read(1, 0, 0).unwrap();
        assert!((k[3] - 3.0).abs() < 1e-6);
        assert!((v[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_unregistered_sequence() {
        let cache = PagedKVCache::new(test_config());
        assert!(!cache.has_sequence(999));
        assert!(cache.to_flat_cache(999).is_none());
    }

    #[test]
    fn test_to_flat_cache_empty() {
        let cache = PagedKVCache::new(test_config());
        // No sequences registered
        assert!(cache.to_flat_cache(1).is_none());
    }

    #[test]
    fn test_alloc_stats() {
        let mut cache = PagedKVCache::new(test_config());
        cache.register_sequence(1);
        assert_eq!(cache.num_allocated, 0);

        let k_row = vec![0.0; 8];
        let v_row = vec![0.0; 8];
        cache.append(1, 0, 0, &k_row, &v_row);

        // First append allocates one block per layer
        assert!(cache.num_allocated > 0);
    }

    #[test]
    fn test_forward_paged_integration() {
        use ndarray::Array1;
        use nexora_transformer::{CausalLM, TransformerConfig};

        let cfg = PagedCacheConfig {
            block_size: 4,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            max_blocks: 1024,
            max_seq_len: 512,
            f16_storage: false,
            ..Default::default()
        };

        let mc = TransformerConfig {
            hidden_size: 512,
            intermediate_size: 1024,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 2,
            max_seq_len: 512,
            vocab_size: 256,
            ..Default::default()
        };

        let model = CausalLM::new(mc);
        let mut paged_cache = PagedKVCache::new(cfg);
        paged_cache.register_sequence(1);

        let input = [1u32, 2u32, 3u32];
        for (i, &token) in input.iter().enumerate() {
            let logits = model
                .forward_paged(&[token], &mut paged_cache, 1)
                .expect("forward_paged should succeed in test");
            assert_eq!(logits.len(), 256);
            if i > 0 {
                let got = paged_cache.num_tokens(1).unwrap_or(0);
                assert_eq!(
                    got,
                    i + 1,
                    "after token {i}, cache should have {} entries",
                    i + 1
                );
            }
        }

        assert_eq!(paged_cache.num_tokens(1), Some(3));
        assert!(paged_cache.total_blocks() > 0);
    }

    // ─── PrefixDAG sharing tests ─────────────────────────────────────────────

    #[test]
    fn test_prefix_sharing_basic() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);

        let kv_dim = config.num_kv_heads * config.head_dim;
        let k_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 10.0).collect();

        // Fill seq 10 with 4 tokens (1 block)
        for pos in 0..4 {
            cache.append(10, 0, pos, &k_row, &v_row);
            cache.append(10, 1, pos, &k_row, &v_row);
        }

        let total_before = cache.total_blocks();
        let refcount_before = cache.blocks[0][0].ref_count;

        // Share prefix (4 tokens = 1 full block per layer) from 10 → 20
        let shared = cache.share_prefix_in_blocks(20, 10, 4);
        assert_eq!(shared, 2, "should share 1 block per layer (2 layers)");

        // No new blocks allocated — just ref_count bump
        assert_eq!(cache.total_blocks(), total_before);
        assert_eq!(
            cache.blocks[0][0].ref_count,
            refcount_before + 1,
            "ref_count should be incremented"
        );
        assert!(
            cache.has_sequence(20),
            "dst sequence should exist after share"
        );

        // Both sequences can read the same data from shared blocks
        for pos in 0..4 {
            let (k10, v10) = cache.read(10, 0, pos).unwrap();
            let (k20, v20) = cache.read(20, 0, pos).unwrap();
            for i in 0..kv_dim {
                assert!((k10[i] - k20[i]).abs() < 1e-6, "K mismatch at pos {pos}");
                assert!((v10[i] - v20[i]).abs() < 1e-6, "V mismatch at pos {pos}");
            }
        }
    }

    #[test]
    fn test_prefix_sharing_cow_on_write() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);

        let kv_dim = config.num_kv_heads * config.head_dim;

        // Fill seq 10 with distinct per-position data
        for pos in 0..4 {
            let k: Vec<f32> = (0..kv_dim).map(|i| (pos * kv_dim + i) as f32).collect();
            let v: Vec<f32> = (0..kv_dim).map(|i| (pos * kv_dim + i + 100) as f32).collect();
            cache.append(10, 0, pos, &k, &v);
            cache.append(10, 1, pos, &k, &v);
        }

        let shared = cache.share_prefix_in_blocks(20, 10, 4);
        assert_eq!(shared, 2, "should share 1 block per layer");

        let blocks_before_cow = cache.total_blocks();
        assert_eq!(cache.blocks[0][0].ref_count, 2);

        // Seq 20 writes to position 0 — triggers COW (block ref_count=2)
        let cow_k: Vec<f32> = (200..200 + kv_dim).map(|i| i as f32).collect();
        let cow_v: Vec<f32> = (300..300 + kv_dim).map(|i| i as f32).collect();
        cache.append(20, 0, 0, &cow_k.as_slice(), &cow_v.as_slice());

        // After COW: new block allocated, original ref_count decremented
        assert_eq!(
            cache.total_blocks(),
            blocks_before_cow + 1,
            "COW should allocate one new block"
        );
        assert_eq!(
            cache.blocks[0][0].ref_count,
            1,
            "original block ref_count decremented after COW"
        );

        // Seq 10 data should be unchanged
        let (k10, v10) = cache.read(10, 0, 0).unwrap();
        let expected_k: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
        let expected_v: Vec<f32> = (0..kv_dim).map(|i| (i + 100) as f32).collect();
        for i in 0..kv_dim {
            assert!(
                (k10[i] - expected_k[i]).abs() < 1e-6,
                "seq 10 K[{i}] changed after COW: expected {}, got {}",
                expected_k[i],
                k10[i]
            );
            assert!(
                (v10[i] - expected_v[i]).abs() < 1e-6,
                "seq 10 V[{i}] changed after COW"
            );
        }

        // Seq 20 data should be the new value
        let (k20, v20) = cache.read(20, 0, 0).unwrap();
        for i in 0..kv_dim {
            assert!(
                (k20[i] - cow_k[i]).abs() < 1e-6,
                "seq 20 K[{i}] should be cow value after COW"
            );
            assert!(
                (v20[i] - cow_v[i]).abs() < 1e-6,
                "seq 20 V[{i}] should be cow value after COW"
            );
        }
    }

    #[test]
    fn test_prefix_sharing_cow_data_integrity() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);

        let kv_dim = config.num_kv_heads * config.head_dim;

        // Fill seq 10 with 8 tokens (2 blocks)
        for pos in 0..8 {
            let k: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i) as f32).collect();
            let v: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i + 1000) as f32).collect();
            cache.append(10, 0, pos, &k, &v);
            cache.append(10, 1, pos, &k, &v);
        }

        // Share first 4 tokens (1 block) from 10 to 20
        cache.share_prefix_in_blocks(20, 10, 4);

        // Seq 20 writes to position 0 and 4 (position 4 falls in NEW block, no COW)
        let cow_k: Vec<f32> = (0..kv_dim).map(|i| 999.0 + i as f32).collect();
        let cow_v: Vec<f32> = (0..kv_dim).map(|i| 888.0 + i as f32).collect();
        cache.append(20, 0, 0, &cow_k, &cow_v);

        let new_k: Vec<f32> = (0..kv_dim).map(|i| 777.0 + i as f32).collect();
        let new_v: Vec<f32> = (0..kv_dim).map(|i| 666.0 + i as f32).collect();
        cache.append(20, 0, 4, &new_k, &new_v);

        // Verify seq 10 unchanged at position 0
        let (k10_p0, v10_p0) = cache.read(10, 0, 0).unwrap();
        for i in 0..kv_dim {
            assert!((k10_p0[i] - i as f32).abs() < 1e-6, "seq 10 K[0][{i}] corrupted");
            assert!((v10_p0[i] - (i + 1000) as f32).abs() < 1e-6, "seq 10 V[0][{i}] corrupted");
        }

        // Verify seq 10 unchanged at position 4
        let (k10_p4, v10_p4) = cache.read(10, 0, 4).unwrap();
        for i in 0..kv_dim {
            assert!((k10_p4[i] - (4 * 10 + i) as f32).abs() < 1e-6, "seq 10 K[4][{i}] corrupted");
        }

        // Verify seq 20 has cow value at position 0
        let (k20_p0, v20_p0) = cache.read(20, 0, 0).unwrap();
        for i in 0..kv_dim {
            assert!((k20_p0[i] - (999.0 + i as f32)).abs() < 1e-6, "seq 20 K[0][{i}] wrong after COW");
        }

        // Verify seq 20 has new value at position 4 (new block, not shared)
        let (k20_p4, v20_p4) = cache.read(20, 0, 4).unwrap();
        for i in 0..kv_dim {
            assert!((k20_p4[i] - (777.0 + i as f32)).abs() < 1e-6, "seq 20 K[4][{i}] wrong");
        }

        // Verify seq 20 still sees shared data at positions 1-3 (no COW on those)
        for pos in 1..4 {
            let (k20, v20) = cache.read(20, 0, pos).unwrap();
            let (k10, v10) = cache.read(10, 0, pos).unwrap();
            for i in 0..kv_dim {
                assert!((k20[i] - k10[i]).abs() < 1e-6, "seq 20 K[{pos}][{i}] diverged from shared");
                assert!((v20[i] - v10[i]).abs() < 1e-6, "seq 20 V[{pos}][{i}] diverged from shared");
            }
        }
    }

    #[test]
    fn test_prefix_sharing_multiple_sequences() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);
        cache.register_sequence(30);

        let kv_dim = config.num_kv_heads * config.head_dim;
        let k_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();

        for pos in 0..4 {
            cache.append(10, 0, pos, &k_row, &v_row);
            cache.append(10, 1, pos, &k_row, &v_row);
        }

        // Share from 10 to both 20 and 30
        cache.share_prefix_in_blocks(20, 10, 4);
        cache.share_prefix_in_blocks(30, 10, 4);

        // All three sequences share the same physical block
        assert_eq!(cache.blocks[0][0].ref_count, 3, "3-way sharing ref_count");

        // COW on seq 20 — decrements one ref_count
        let cow_k: Vec<f32> = (0..kv_dim).map(|i| 555.0 + i as f32).collect();
        let cow_v: Vec<f32> = (0..kv_dim).map(|i| 555.0 + i as f32).collect();
        cache.append(20, 0, 0, &cow_k, &cow_v);

        // Original block now has ref_count=2 (shared by 10 and 30)
        assert_eq!(cache.blocks[0][0].ref_count, 2, "after COW on 20, ref_count drops to 2");

        // COW on seq 30 — decrements another ref_count
        cache.append(30, 0, 0, &cow_k, &cow_v);
        assert_eq!(cache.blocks[0][0].ref_count, 1, "after COW on 30, ref_count drops to 1");
    }

    #[test]
    fn test_prefix_sharing_noop_invalid() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);

        let kv_dim = config.num_kv_heads * config.head_dim;
        let k_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();

        for pos in 0..4 {
            cache.append(10, 0, pos, &k_row, &v_row);
            cache.append(10, 1, pos, &k_row, &v_row);
        }

        // Invalid dst seq
        assert_eq!(cache.share_prefix_in_blocks(999, 10, 4), 0, "invalid dst");
        // Invalid src seq
        assert_eq!(cache.share_prefix_in_blocks(10, 999, 4), 0, "invalid src");
        // prefix_tokens = 0
        assert_eq!(cache.share_prefix_in_blocks(10, 10, 0), 0, "zero prefix");
    }

    #[test]
    fn test_prefix_sharing_free_after_share() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);

        let kv_dim = config.num_kv_heads * config.head_dim;
        let k_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();

        for pos in 0..4 {
            cache.append(10, 0, pos, &k_row, &v_row);
            cache.append(10, 1, pos, &k_row, &v_row);
        }

        cache.share_prefix_in_blocks(20, 10, 4);
        assert_eq!(cache.blocks[0][0].ref_count, 2);

        // Remove seq 20 — ref_count should drop
        cache.remove_sequence(20);
        assert_eq!(cache.blocks[0][0].ref_count, 1, "ref_count decremented after removing shared seq");
    }

    #[test]
    fn test_prefix_sharing_append_after_share() {
        let mut config = test_config();
        config.block_size = 4;
        let mut cache = PagedKVCache::new(config.clone());
        cache.register_sequence(10);
        cache.register_sequence(20);

        let kv_dim = config.num_kv_heads * config.head_dim;

        // Seq 10: 4 tokens
        for pos in 0..4 {
            let k: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i) as f32).collect();
            let v: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i + 100) as f32).collect();
            cache.append(10, 0, pos, &k, &v);
            cache.append(10, 1, pos, &k, &v);
        }

        // Share prefix + seq 20 appends 2 more tokens
        cache.share_prefix_in_blocks(20, 10, 4);

        // Append position 4 and 5 to seq 20 (new block)
        for pos in 4..6 {
            let k: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i) as f32).collect();
            let v: Vec<f32> = (0..kv_dim).map(|i| (pos * 10 + i + 100) as f32).collect();
            cache.append(20, 0, pos, &k, &v);
            cache.append(20, 1, pos, &k, &v);
        }

        assert_eq!(cache.num_tokens(20), Some(6), "seq 20 should have 6 tokens after append");

        // Seq 10 still has 4 tokens
        assert_eq!(cache.num_tokens(10), Some(4), "seq 10 should have 4 tokens");

        // Verify shared prefix data still matches
        for pos in 0..4 {
            let (k10, v10) = cache.read(10, 0, pos).unwrap();
            let (k20, v20) = cache.read(20, 0, pos).unwrap();
            for i in 0..kv_dim {
                assert!((k10[i] - k20[i]).abs() < 1e-6, "shared prefix K mismatch at pos {pos}");
                assert!((v10[i] - v20[i]).abs() < 1e-6, "shared prefix V mismatch at pos {pos}");
            }
        }

        // Verify new tokens in seq 20
        for pos in 4..6 {
            let (k20, _) = cache.read(20, 0, pos).unwrap();
            assert!((k20[0] - (pos * 10) as f32).abs() < 1e-6, "appended K wrong at pos {pos}");
        }
    }

    // ─── PrefixTrie tests ────────────────────────────────────────────────────

    #[test]
    fn test_prefix_trie_insert_and_find_shared() {
        use crate::continuous_batching::PrefixTrie;

        let mut trie = PrefixTrie::new();

        // Insert seq 1 with prompt [1,2,3,4,5]
        trie.insert(1, &[1, 2, 3, 4, 5]);

        // Insert seq 2 with prompt [1,2,3,6,7,8] — shared prefix [1,2,3]
        trie.insert(2, &[1, 2, 3, 6, 7, 8]);

        // Find shared prefix for seq 2
        let result = trie.find_shared_prefix(&[1, 2, 3, 6, 7, 8], 2);
        assert!(result.is_some(), "should find shared prefix");
        let (len, src) = result.unwrap();
        assert_eq!(len, 3, "should find 3 shared tokens");
        assert_eq!(src, 1, "should find seq 1 as source");

        // Find shared prefix for seq 1
        let result = trie.find_shared_prefix(&[1, 2, 3, 4, 5], 1);
        assert!(result.is_some(), "should find shared prefix for seq 1");
        let (len, src) = result.unwrap();
        assert_eq!(len, 3, "should find 3 shared tokens for seq 1 too");
        assert_eq!(src, 2, "should find seq 2 as source");

        // No match
        let result = trie.find_shared_prefix(&[9, 10, 11], 3);
        assert!(result.is_none(), "no match for unknown prefix");
    }

    #[test]
    fn test_prefix_trie_exact_match() {
        use crate::continuous_batching::PrefixTrie;

        let mut trie = PrefixTrie::new();
        trie.insert(1, &[1, 2, 3, 4]);
        trie.insert(2, &[1, 2, 3, 4]);

        let result = trie.find_shared_prefix(&[1, 2, 3, 4], 3);
        assert!(result.is_some(), "exact match should return Some");
        let (len, src) = result.unwrap();
        assert_eq!(len, 4, "exact match should return full length");
        assert!(src == 1 || src == 2, "exact match should find any matching seq");
    }

    #[test]
    fn test_prefix_trie_empty_prompt() {
        use crate::continuous_batching::PrefixTrie;

        let mut trie = PrefixTrie::new();
        trie.insert(1, &[1, 2, 3]);

        let result = trie.find_shared_prefix(&[], 2);
        assert!(result.is_none(), "empty prompt should match nothing");
    }

    #[test]
    fn test_paged_vs_flat_cache_parity() {
        use ndarray::Array1;
        use nexora_transformer::{CausalLM, KVCacheEntry, TransformerConfig};

        let cfg = PagedCacheConfig {
            block_size: 4,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            max_blocks: 1024,
            max_seq_len: 512,
            f16_storage: false,
            ..Default::default()
        };

        let mc = TransformerConfig {
            hidden_size: 512,
            intermediate_size: 1024,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 2,
            max_seq_len: 512,
            vocab_size: 256,
            ..Default::default()
        };

        let model = CausalLM::new(mc);
        let mut flat_cache = model.reset_cache();
        let mut paged_cache = PagedKVCache::new(cfg);
        paged_cache.register_sequence(1);

        let tokens = [5u32, 10u32, 15u32];
        for (i, &token) in tokens.iter().enumerate() {
            let logits_flat = model
                .forward(&[token], &mut flat_cache)
                .expect("forward should succeed in test");
            let logits_paged = model
                .forward_paged(&[token], &mut paged_cache, 1)
                .expect("forward_paged should succeed in test");

            for j in 0..logits_flat.len() {
                let diff = (logits_flat[j] - logits_paged[j]).abs();
                assert!(
                    diff < 1e-4,
                    "mismatch at token {i}, logit {j}: flat={} paged={}",
                    logits_flat[j],
                    logits_paged[j]
                );
            }
        }
    }
