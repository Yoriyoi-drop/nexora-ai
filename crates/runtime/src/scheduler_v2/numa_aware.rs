use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// NUMA-aware scheduler — track memory node dan assign task ke node terdekat
pub struct NumaAwareScheduler {
    node_count: usize,
    node_memory_total: Vec<u64>,
    node_memory_used: Vec<AtomicUsize>,
    node_cpu_usage: Vec<AtomicUsize>,
    task_to_node: DashMap<uuid::Uuid, u32>,
}

impl NumaAwareScheduler {
    pub fn new(node_count: usize, memory_per_node: Vec<u64>) -> Self {
        let total = if memory_per_node.len() == node_count {
            memory_per_node
        } else {
            vec![64 * 1024 * 1024 * 1024; node_count]
        };
        Self {
            node_count,
            node_memory_total: total,
            node_memory_used: (0..node_count).map(|_| AtomicUsize::new(0)).collect(),
            node_cpu_usage: (0..node_count).map(|_| AtomicUsize::new(0)).collect(),
            task_to_node: DashMap::new(),
        }
    }

    /// Assign task ke NUMA node dengan resource paling tersedia
    pub fn assign_node(&self, task_id: uuid::Uuid, required_memory: u64, preferred: Option<u32>) -> u32 {
        let node = preferred.unwrap_or_else(|| {
            let mut best = 0u32;
            let mut best_free = 0u64;
            for i in 0..self.node_count {
                let used = self.node_memory_used[i].load(Ordering::Relaxed) as u64;
                let free = self.node_memory_total[i].saturating_sub(used);
                if free > best_free {
                    best_free = free;
                    best = i as u32;
                }
            }
            best
        });

        let idx = node as usize;
        if idx < self.node_count {
            self.node_memory_used[idx]
                .fetch_add(required_memory as usize, Ordering::Relaxed);
            self.node_cpu_usage[idx].fetch_add(1, Ordering::Relaxed);
        }
        self.task_to_node.insert(task_id, node);
        node
    }

    pub fn release_node(&self, task_id: &uuid::Uuid, memory: u64) {
        if let Some((_, node)) = self.task_to_node.remove(task_id) {
            let idx = node as usize;
            if idx < self.node_count {
                self.node_memory_used[idx]
                    .fetch_sub(memory as usize, Ordering::Relaxed);
                self.node_cpu_usage[idx].fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    pub fn node_utilization(&self) -> Vec<f64> {
        (0..self.node_count)
            .map(|i| {
                self.node_memory_used[i].load(Ordering::Relaxed) as f64
                    / self.node_memory_total[i] as f64
            })
            .collect()
    }

    pub fn task_count_per_node(&self) -> Vec<usize> {
        (0..self.node_count)
            .map(|i| {
                self.task_to_node
                    .iter()
                    .filter(|e| *e.value() == i as u32)
                    .count()
            })
            .collect()
    }
}
