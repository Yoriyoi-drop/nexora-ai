use dashmap::DashMap;
use parking_lot::Mutex;
use std::time::{Duration, Instant};

use crate::scheduler_v2::dag::TaskId;

#[derive(Debug, Clone)]
struct DeadlineEntry {
    task_id: TaskId,
    deadline: Instant,
    priority: u32,
}

/// Deadline scheduler — tasks with earliest deadline executed first
pub struct DeadlineScheduler {
    queue: Mutex<Vec<DeadlineEntry>>,
    deadlines: DashMap<TaskId, Instant>,
}

impl DeadlineScheduler {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(Vec::new()),
            deadlines: DashMap::new(),
        }
    }

    pub fn add(&self, task_id: TaskId, deadline: Instant, priority: u32) {
        self.deadlines.insert(task_id, deadline);
        let mut queue = self.queue.lock();
        queue.push(DeadlineEntry {
            task_id,
            deadline,
            priority,
        });
        queue.sort_by(|a, b| {
            a.deadline
                .cmp(&b.deadline)
                .then_with(|| b.priority.cmp(&a.priority))
        });
    }

    pub fn pop(&self) -> Option<TaskId> {
        let mut queue = self.queue.lock();
        while let Some(entry) = queue.first() {
            if entry.deadline < Instant::now() {
                // Deadline missed — prioritize anyway
                return queue.pop().map(|e| e.task_id);
            }
            return Some(queue.remove(0).task_id);
        }
        None
    }

    pub fn deadline_for(&self, task_id: &TaskId) -> Option<Instant> {
        self.deadlines.get(task_id).map(|d| *d)
    }

    pub fn time_until_deadline(&self, task_id: &TaskId) -> Option<Duration> {
        self.deadlines
            .get(task_id)
            .map(|d| d.saturating_duration_since(Instant::now()))
    }

    pub fn missed_deadlines(&self) -> Vec<TaskId> {
        let now = Instant::now();
        self.queue
            .lock()
            .iter()
            .filter(|e| e.deadline < now)
            .map(|e| e.task_id)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.queue.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.lock().is_empty()
    }
}

impl Default for DeadlineScheduler {
    fn default() -> Self {
        Self::new()
    }
}
