use parking_lot::Mutex;
use priority_queue::PriorityQueue;
use std::cmp::Reverse;

use crate::scheduler_v2::dag::TaskId;

/// Priority queue untuk task scheduling
/// Higher priority = executed first
pub struct TaskPriorityQueue {
    queue: Mutex<PriorityQueue<TaskId, Reverse<u32>>>,
}

impl TaskPriorityQueue {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(PriorityQueue::new()),
        }
    }

    pub fn push(&self, task_id: TaskId, priority: u32) {
        self.queue.lock().push(task_id, Reverse(priority));
    }

    pub fn pop(&self) -> Option<TaskId> {
        self.queue.lock().pop().map(|(id, _)| id)
    }

    pub fn peek(&self) -> Option<TaskId> {
        self.queue.lock().peek().map(|(id, _)| *id)
    }

    pub fn change_priority(&self, task_id: &TaskId, new_priority: u32) {
        self.queue
            .lock()
            .change_priority(task_id, Reverse(new_priority));
    }

    pub fn remove(&self, task_id: &TaskId) {
        self.queue.lock().remove(task_id);
    }

    pub fn len(&self) -> usize {
        self.queue.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.lock().is_empty()
    }
}

impl Default for TaskPriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}
