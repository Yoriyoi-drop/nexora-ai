// Integration test: GPU KV Cache functionality
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_kv_cache -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::gpu_kv_cache::GpuKVCache;

    #[test]
    fn test_kv_cache_init() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        let cache = GpuKVCache::new(
            &ctx.device,
            32,  // num_layers
            4,   // num_heads
            16,  // head_dim
            128, // max_seq_len
            1,   // batch_size
        );
        
        assert_eq!(cache.num_layers(), 32);
        assert_eq!(cache.max_seq_len(), 128);
        
        println!("KV Cache init OK");
    }

    #[test]
    fn test_kv_cache_append() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        let mut cache = GpuKVCache::new(
            &ctx.device,
            2,   // num_layers
            2,   // num_heads
            8,   // head_dim
            16,  // max_seq_len
            1,   // batch_size
        );
        
        // Append key/value for position 0
        let k_data = ArrayD::from_shape_vec(
            vec![1, 2, 2, 8],
            (0..32).map(|i| (i as f32) * 0.1).collect(),
        ).unwrap();
        let v_data = ArrayD::from_shape_vec(
            vec![1, 2, 2, 8],
            (0..32).map(|i| (i as f32) * 0.05).collect(),
        ).unwrap();
        
        let gk = GpuTensor::from_cpu(&k_data).unwrap();
        let gv = GpuTensor::from_cpu(&v_data).unwrap();
        
        cache.append(0, &gk, &gv).unwrap();
        
        assert_eq!(cache.current_len(), 1);
        
        println!("KV Cache append OK");
    }

    #[test]
    fn test_kv_cache_multiple_appends() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        let mut cache = GpuKVCache::new(
            &ctx.device,
            1,   // num_layers
            2,   // num_heads
            8,   // head_dim
            32,  // max_seq_len
            1,   // batch_size
        );
        
        // Append 5 positions
        for pos in 0..5 {
            let k_data = ArrayD::from_shape_vec(
                vec![1, 2, 2, 8],
                (0..32).map(|i| (i as f32 + pos as f32) * 0.1).collect(),
            ).unwrap();
            let v_data = ArrayD::from_shape_vec(
                vec![1, 2, 2, 8],
                (0..32).map(|i| (i as f32 + pos as f32) * 0.05).collect(),
            ).unwrap();
            
            let gk = GpuTensor::from_cpu(&k_data).unwrap();
            let gv = GpuTensor::from_cpu(&v_data).unwrap();
            
            cache.append(pos, &gk, &gv).unwrap();
        }
        
        assert_eq!(cache.current_len(), 5);
        
        println!("KV Cache multiple appends OK");
    }

    #[test]
    fn test_kv_cache_get() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        let mut cache = GpuKVCache::new(
            &ctx.device,
            1,   // num_layers
            2,   // num_heads
            8,   // head_dim
            16,  // max_seq_len
            1,   // batch_size
        );
        
        // Append data
        let k_data = ArrayD::from_shape_vec(
            vec![1, 2, 2, 8],
            (0..32).map(|i| (i as f32) * 0.1).collect(),
        ).unwrap();
        let v_data = ArrayD::from_shape_vec(
            vec![1, 2, 2, 8],
            (0..32).map(|i| (i as f32) * 0.05).collect(),
        ).unwrap();
        
        let gk = GpuTensor::from_cpu(&k_data).unwrap();
        let gv = GpuTensor::from_cpu(&v_data).unwrap();
        
        cache.append(0, &gk, &gv).unwrap();
        
        // Get keys/values
        let (k_out, v_out) = cache.get(0, 0, 1).unwrap();
        
        assert_eq!(k_out.shape(), &[1, 2, 2, 8]);
        assert_eq!(v_out.shape(), &[1, 2, 2, 8]);
        
        println!("KV Cache get OK");
    }

    #[test]
    fn test_kv_cache_reset() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        let mut cache = GpuKVCache::new(
            &ctx.device,
            1,   // num_layers
            2,   // num_heads
            8,   // head_dim
            16,  // max_seq_len
            1,   // batch_size
        );
        
        // Append data
        let k_data = ArrayD::from_shape_vec(
            vec![1, 2, 2, 8],
            (0..32).map(|i| (i as f32) * 0.1).collect(),
        ).unwrap();
        let v_data = ArrayD::from_shape_vec(
            vec![1, 2, 2, 8],
            (0..32).map(|i| (i as f32) * 0.05).collect(),
        ).unwrap();
        
        let gk = GpuTensor::from_cpu(&k_data).unwrap();
        let gv = GpuTensor::from_cpu(&v_data).unwrap();
        
        cache.append(0, &gk, &gv).unwrap();
        assert_eq!(cache.current_len(), 1);
        
        // Reset
        cache.reset();
        assert_eq!(cache.current_len(), 0);
        
        println!("KV Cache reset OK");
    }
}
