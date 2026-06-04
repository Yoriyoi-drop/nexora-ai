use super::*;
use ndarray::array;

fn small_gqa() -> GQA {
        let mut gqa = GQA::new(8, 4, 2, 4);
        gqa.init_random(8, 4, 2, 4);
        gqa
    }

    #[test]
    fn test_gqa_new_shapes() {
        let gqa = small_gqa();
        assert_eq!(gqa.num_heads, 4);
        assert_eq!(gqa.num_kv_heads, 2);
        assert_eq!(gqa.head_dim, 4);
        assert_eq!(gqa.num_groups, 2);
        assert_eq!(gqa.wq.as_ref().unwrap().dim(), (16, 8));
        assert_eq!(gqa.wk.as_ref().unwrap().dim(), (8, 8));
        assert_eq!(gqa.wv.as_ref().unwrap().dim(), (8, 8));
        assert_eq!(gqa.wo.as_ref().unwrap().dim(), (8, 16));
    }

    #[test]
    fn test_gqa_forward_shape_no_cache() {
        let gqa = small_gqa();
        let x = array![[1.0; 8]];
        let cos = vec![0.0; 2];
        let sin = vec![1.0; 2];
        let out = gqa.forward(&x, None, 0, &cos, &sin).unwrap();
        assert_eq!(out.dim(), (1, 8));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gqa_forward_with_kv_shape_and_cache() {
        let gqa = small_gqa();
        let x = array![[1.0; 8]];
        let cos = vec![0.0; 2];
        let sin = vec![1.0; 2];
        let mut cache = vec![KVCacheEntry {
            k: vec![],
            v: vec![],
            kv_dim: 8,
            num_kv_heads: 1,
            head_dim: 8,
            k_compressed: Vec::new(),
            v_compressed: Vec::new(),
            k_scales: Vec::new(),
            v_scales: Vec::new(),
            compressed_seq_len: 0,
        }];
        let out = gqa.forward_with_kv(&x, &mut cache, 0, &cos, &sin).unwrap();
        assert_eq!(out.dim(), (1, 8));
        assert!(cache[0].seq_len() > 0);
        assert!(!cache[0].k.is_empty());
    }

    #[test]
    fn test_gqa_forward_batch2() {
        let gqa = small_gqa();
        let x = array![[1.0; 8], [2.0; 8]];
        let cos = vec![0.5; 2];
        let sin = vec![0.5; 2];
        let out = gqa.forward(&x, None, 0, &cos, &sin).unwrap();
        assert_eq!(out.dim(), (2, 8));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_kv_cache_entry_seq_len() {
        let entry = KVCacheEntry {
            k: vec![1.0; 16],
            v: vec![2.0; 16],
            kv_dim: 8,
            num_kv_heads: 1,
            head_dim: 8,
            k_compressed: Vec::new(),
            v_compressed: Vec::new(),
            k_scales: Vec::new(),
            v_scales: Vec::new(),
            compressed_seq_len: 0,
        };
        assert_eq!(entry.seq_len(), 2);
        assert_eq!(
            KVCacheEntry {
                k: vec![],
                v: vec![],
                kv_dim: 0,
                num_kv_heads: 0,
                head_dim: 0,
                k_compressed: Vec::new(),
                v_compressed: Vec::new(),
                k_scales: Vec::new(),
                v_scales: Vec::new(),
                compressed_seq_len: 0,
            }
            .seq_len(),
            0
        );
    }

    #[test]
    fn test_cpu_kv_cache_append_read() {
        let mut cache = CpuKVCache::new(0);
        assert_eq!(cache.num_layers(), 0);
        cache.append(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.num_layers(), 1);
        assert_eq!(cache.seq_len(0), 1);
        assert_eq!(cache.get_k(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.get_v(0), &[5.0, 6.0, 7.0, 8.0]);
        cache.append(0, &[9.0; 4], &[10.0; 4]);
        assert_eq!(cache.seq_len(0), 2);
        assert_eq!(cache.get_k(0).len(), 8);
        cache.clear();
        assert_eq!(cache.num_layers(), 0);
    }

    #[test]
    fn test_cpu_kv_cache_out_of_range() {
        let cache = CpuKVCache::new(0);
        assert_eq!(cache.get_k(99), &[] as &[f32]);
        assert_eq!(cache.get_v(99), &[] as &[f32]);
        assert_eq!(cache.seq_len(99), 0);
    }