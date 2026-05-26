use super::shuffle::ShuffleBuffer;
use crate::types::DataSample;

pub struct BatchIterator {
    batch_size: usize,
    shuffle_buffer: ShuffleBuffer,
}

impl BatchIterator {
    pub fn new(batch_size: usize, shuffle_buffer_size: usize) -> Self {
        Self {
            batch_size,
            shuffle_buffer: ShuffleBuffer::new(shuffle_buffer_size),
        }
    }

    pub fn push(&mut self, samples: Vec<DataSample>) {
        self.shuffle_buffer.push(samples);
    }

    pub fn next_batch(&mut self) -> Vec<DataSample> {
        let available = self.shuffle_buffer.len();
        if available == 0 {
            return Vec::new();
        }

        let take = self.batch_size.min(available);
        self.shuffle_buffer.drain(take)
    }

    pub fn remaining(&self) -> usize {
        self.shuffle_buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceInfo, SourceCategory};
    use uuid::Uuid;

    fn sample(text: &str) -> DataSample {
        DataSample {
            id: Uuid::new_v4(),
            text: text.into(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: SourceInfo {
                name: "test".into(),
                url: None,
                trust_score: 0.5,
                category: SourceCategory::Other,
                fetch_timestamp: 0,
            },
            stats: SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        }
    }

    #[test]
    fn test_batch_iterator_new() {
        let iter = BatchIterator::new(8, 100);
        assert_eq!(iter.remaining(), 0);
    }

    #[test]
    fn test_push_and_next_batch() {
        let mut iter = BatchIterator::new(4, 100);
        let samples = vec![
            sample("a"), sample("b"), sample("c"), sample("d"),
            sample("e"), sample("f"),
        ];
        iter.push(samples);
        assert_eq!(iter.remaining(), 6);

        let batch = iter.next_batch();
        assert_eq!(batch.len(), 4);
        assert_eq!(iter.remaining(), 2);
    }

    #[test]
    fn test_next_batch_less_than_batch_size() {
        let mut iter = BatchIterator::new(10, 100);
        iter.push(vec![sample("a"), sample("b")]);
        let batch = iter.next_batch();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_next_batch_empty() {
        let mut iter = BatchIterator::new(8, 100);
        let batch = iter.next_batch();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_multiple_pushes() {
        let mut iter = BatchIterator::new(3, 100);
        iter.push(vec![sample("a"), sample("b")]);
        iter.push(vec![sample("c"), sample("d")]);
        assert_eq!(iter.remaining(), 4);

        let b1 = iter.next_batch();
        assert_eq!(b1.len(), 3);
        let b2 = iter.next_batch();
        assert_eq!(b2.len(), 1);
    }
}
