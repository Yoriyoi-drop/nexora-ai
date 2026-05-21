// Integration test: GPU Memory Pool functionality
// Run: cargo test --features gpu -p nexora-autograd --test gpu/test_memory_pool -- --nocapture

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    use nexora_autograd::gpu_memory::GpuMemoryPool;

    #[test]
    fn test_memory_pool_basic_alloc() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let pool = GpuMemoryPool::new(&ctx.device);
        
        // Allocate buffer
        let size = 1024u64;
        let buffer = pool.alloc(size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        
        assert!(buffer.size >= size);
        
        // Deallocate
        pool.dealloc(buffer);
        
        println!("Memory pool basic alloc/dealloc OK");
    }

    #[test]
    fn test_memory_pool_reuse() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let pool = GpuMemoryPool::new(&ctx.device);
        
        let size = 2048u64;
        
        // Allocate and deallocate multiple times
        for _ in 0..5 {
            let buffer = pool.alloc(size, wgpu::BufferUsages::STORAGE);
            pool.dealloc(buffer);
        }
        
        println!("Memory pool reuse OK");
    }

    #[test]
    fn test_memory_pool_different_sizes() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let pool = GpuMemoryPool::new(&ctx.device);
        
        let sizes = vec![512, 1024, 2048, 4096, 8192];
        
        for size in sizes {
            let buffer = pool.alloc(size, wgpu::BufferUsages::STORAGE);
            assert!(buffer.size >= size);
            pool.dealloc(buffer);
        }
        
        println!("Memory pool different sizes OK");
    }

    #[test]
    fn test_memory_pool_workspace() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let pool = GpuMemoryPool::new(&ctx.device);
        
        // Reset workspace
        pool.reset_workspace();
        
        // Allocate from workspace (small temporary buffers)
        let buf1 = pool.alloc_from_workspace(256, wgpu::BufferUsages::STORAGE);
        let buf2 = pool.alloc_from_workspace(512, wgpu::BufferUsages::STORAGE);
        
        pool.reset_workspace();
        
        println!("Memory pool workspace OK");
    }

    #[test]
    fn test_memory_pool_stats() {
        let ctx = GpuContext::init().expect("GPU context init failed");
        let pool = GpuMemoryPool::new(&ctx.device);
        
        let stats_before = pool.get_stats();
        
        // Allocate some buffers
        let buf1 = pool.alloc(1024, wgpu::BufferUsages::STORAGE);
        let buf2 = pool.alloc(2048, wgpu::BufferUsages::STORAGE);
        
        let stats_after = pool.get_stats();
        
        // Stats should change
        assert!(stats_after.total_allocated > stats_before.total_allocated);
        
        pool.dealloc(buf1);
        pool.dealloc(buf2);
        
        println!("Memory pool stats OK - allocated: {} bytes", stats_after.total_allocated);
    }
}
