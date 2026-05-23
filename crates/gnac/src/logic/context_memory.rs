use std::collections::VecDeque;
use uuid::Uuid;

/// Context Memory Node — memori jangka panjang untuk konteks
#[derive(Debug, Clone)]
pub struct ContextMemoryNode {
    pub id: Uuid,
    pub name: String,
    pub memory_size: usize,
    pub state_dim: usize,
    pub memory_buffer: VecDeque<Vec<f64>>,
    pub attention_weights: Vec<f64>,
}

impl ContextMemoryNode {
    pub fn new(name: &str, memory_size: usize, state_dim: usize) -> Self {
        ContextMemoryNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            memory_size,
            state_dim,
            memory_buffer: VecDeque::with_capacity(memory_size),
            attention_weights: Vec::new(),
        }
    }

    /// Push state baru ke memori
    pub fn push(&mut self, state: Vec<f64>) {
        if self.memory_buffer.len() >= self.memory_size {
            self.memory_buffer.pop_front();
        }
        self.memory_buffer.push_back(state);
    }

    /// Retrieval dengan attention
    pub fn retrieve(&self, query: &[f64]) -> Vec<f64> {
        if self.memory_buffer.is_empty() {
            return vec![0.0; self.state_dim];
        }

        let mut scores: Vec<f64> = self
            .memory_buffer
            .iter()
            .map(|mem| {
                query
                    .iter()
                    .zip(mem.iter())
                    .map(|(q, m)| (q - m).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for s in &mut scores {
                *s = (-(*s) / sum).exp();
            }
            let norm_sum: f64 = scores.iter().sum();
            for s in &mut scores {
                *s /= norm_sum;
            }
        }

        scores
            .iter()
            .zip(self.memory_buffer.iter())
            .map(|(&w, mem)| mem.iter().map(|&v| v * w).collect::<Vec<_>>())
            .reduce(|a, b| a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
            .unwrap_or_default()
    }

    pub fn clear(&mut self) {
        self.memory_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_memory_new() {
        let m = ContextMemoryNode::new("mem", 5, 3);
        assert_eq!(m.memory_size, 5);
        assert_eq!(m.state_dim, 3);
    }

    #[test]
    fn test_push_and_retrieve() {
        let mut m = ContextMemoryNode::new("mem", 2, 2);
        m.push(vec![1.0, 2.0]);
        m.push(vec![3.0, 4.0]);
        assert_eq!(m.memory_buffer.len(), 2);
        let result = m.retrieve(&[1.0, 2.0]);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_retrieve_empty() {
        let m = ContextMemoryNode::new("mem", 5, 3);
        let result = m.retrieve(&[1.0, 2.0, 3.0]);
        assert_eq!(result, vec![0.0; 3]);
    }

    #[test]
    fn test_push_boundary() {
        let mut m = ContextMemoryNode::new("mem", 2, 2);
        m.push(vec![1.0, 2.0]);
        m.push(vec![3.0, 4.0]);
        m.push(vec![5.0, 6.0]); // Should evict oldest
        assert_eq!(m.memory_buffer.len(), 2);
        assert_eq!(m.memory_buffer[0], vec![3.0, 4.0]);
    }

    #[test]
    fn test_clear() {
        let mut m = ContextMemoryNode::new("mem", 5, 2);
        m.push(vec![1.0, 2.0]);
        m.clear();
        assert!(m.memory_buffer.is_empty());
    }
}
