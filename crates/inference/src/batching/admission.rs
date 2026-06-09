#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-in-first-out — oldest ready sequences are processed first.
    Fifo,
    /// Priority with aging — base priority + time-in-queue boost.
    /// Long-waiting sequences get priority boost to prevent starvation.
    PriorityAging,
    /// Shortest job first by remaining generation tokens.
    ShortestRemaining,
    /// Token-bucket fairness — each sequence gets a fair share of the batch.
    TokenBucket,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        Self::Fifo
    }
}
