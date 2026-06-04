// Integration test: GPU KV Cache functionality
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_kv_cache -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::gpu_kv_cache::GpuPageTable;

    #[test]
    fn test_gpu_page_table_alloc_free() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        let mut pt = GpuPageTable::new(
            &ctx,
            64,    // max_pages
            16,    // page_size
            4,     // num_heads
            64,    // head_dim
        ).unwrap();
        
        assert_eq!(pt.available_pages(), 64);
        
        // Allocate a page
        let page1 = pt.alloc().expect("alloc should succeed");
        assert_eq!(pt.available_pages(), 63);
        
        // Free it
        pt.free(page1);
        assert_eq!(pt.available_pages(), 64);
        
        println!("GPU PageTable alloc/free OK");
    }

    #[test]
    fn test_gpu_page_table_exhaustion() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        
        let mut pt = GpuPageTable::new(
            &ctx,
            4,     // max_pages
            16,    // page_size
            4,     // num_heads
            64,    // head_dim
        ).unwrap();
        
        // Allocate all pages
        let mut pages = Vec::new();
        for _ in 0..4 {
            pages.push(pt.alloc().expect("should allocate"));
        }
        
        // Should be no more pages
        assert!(pt.alloc().is_none());
        
        // Free all
        for p in pages {
            pt.free(p);
        }
        assert_eq!(pt.available_pages(), 4);
        
        println!("GPU PageTable exhaustion OK");
    }
}
