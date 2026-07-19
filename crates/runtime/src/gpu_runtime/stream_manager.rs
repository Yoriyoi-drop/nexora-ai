use dashmap::DashMap;
use std::sync::atomic::AtomicU64;
use uuid::Uuid;

/// GPU Stream Manager — manage multiple CUDA streams untuk concurrent execution
pub struct GpuStreamManager {
    streams: DashMap<Uuid, GpuStream>,
    #[allow(dead_code)]
    next_id: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct GpuStream {
    pub id: Uuid,
    pub gpu_id: usize,
    pub priority: u32,
    pub kernels_queued: u64,
    pub kernels_completed: u64,
}

impl GpuStreamManager {
    pub fn new() -> Self {
        Self {
            streams: DashMap::new(),
            next_id: AtomicU64::new(0),
        }
    }

    pub fn create_stream(&self, gpu_id: usize, priority: u32) -> Uuid {
        let id = Uuid::new_v4();
        self.streams.insert(
            id,
            GpuStream {
                id,
                gpu_id,
                priority,
                kernels_queued: 0,
                kernels_completed: 0,
            },
        );
        id
    }

    pub fn assign_to_stream(&self, stream_id: &Uuid) -> Option<GpuStream> {
        self.streams
            .get_mut(stream_id)
            .map(|mut s| {
                s.kernels_queued += 1;
                s.clone()
            })
    }

    pub fn complete_kernel(&self, stream_id: &Uuid) {
        if let Some(mut s) = self.streams.get_mut(stream_id) {
            s.kernels_completed += 1;
        }
    }

    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    pub fn active_streams(&self, gpu_id: usize) -> Vec<Uuid> {
        self.streams
            .iter()
            .filter(|e| e.gpu_id == gpu_id)
            .map(|e| e.id)
            .collect()
    }
}

impl Default for GpuStreamManager {
    fn default() -> Self {
        Self::new()
    }
}
