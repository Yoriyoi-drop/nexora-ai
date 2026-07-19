use crossbeam::channel;
use std::sync::Arc;
use tracing::debug;

use crate::dag::TaskId;

/// Work-stealing scheduler using crossbeam channels
pub struct WorkStealingScheduler {
    global_tx: channel::Sender<TaskId>,
    global_rx: Arc<channel::Receiver<TaskId>>,
    local_queues: Vec<channel::Sender<TaskId>>,
    local_receivers: Vec<channel::Receiver<TaskId>>,
}

impl WorkStealingScheduler {
    pub fn new(num_workers: usize) -> Self {
        let (global_tx, global_rx) = channel::unbounded::<TaskId>();

        let mut local_txs = Vec::with_capacity(num_workers);
        let mut local_rxs = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let (ltx, lrx) = channel::unbounded::<TaskId>();
            local_txs.push(ltx);
            local_rxs.push(lrx);
        }

        Self {
            global_tx,
            global_rx: Arc::new(global_rx),
            local_queues: local_txs,
            local_receivers: local_rxs,
        }
    }

    pub fn push_global(&self, task_id: TaskId) {
        let _ = self.global_tx.send(task_id);
    }

    pub fn push_local(&self, worker_idx: usize, task_id: TaskId) {
        if worker_idx < self.local_queues.len() {
            let _ = self.local_queues[worker_idx].send(task_id);
        }
    }

    /// Try to pop: local queue first, then global, then steal from others
    pub fn pop(&self, worker_idx: usize) -> Option<TaskId> {
        if worker_idx >= self.local_receivers.len() {
            return None;
        }

        // 1. Local queue
        if let Ok(task) = self.local_receivers[worker_idx].try_recv() {
            return Some(task);
        }

        // 2. Global queue
        if let Ok(task) = self.global_rx.try_recv() {
            return Some(task);
        }

        // 3. Try stealing from others (random order)
        let len = self.local_receivers.len();
        let start = (worker_idx + 1) % len;
        for i in 0..len {
            let idx = (start + i) % len;
            if idx == worker_idx {
                continue;
            }
            if let Ok(task) = self.local_receivers[idx].try_recv() {
                debug!("stealing task {} from worker {}", task, idx);
                return Some(task);
            }
        }

        None
    }

    pub fn worker_count(&self) -> usize {
        self.local_queues.len()
    }

    pub fn is_empty(&self) -> bool {
        self.global_rx.is_empty()
            && self.local_receivers.iter().all(|r| r.is_empty())
    }
}
