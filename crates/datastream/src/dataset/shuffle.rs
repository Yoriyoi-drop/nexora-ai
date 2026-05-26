use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

use crate::types::DataSample;

pub struct ShuffleBuffer {
    capacity: usize,
    buffer: Vec<DataSample>,
    rng: StdRng,
    seen: usize,
}

impl ShuffleBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
            rng: StdRng::from_entropy(),
            seen: 0,
        }
    }

    pub fn push(&mut self, samples: Vec<DataSample>) {
        for sample in samples {
            if self.buffer.len() < self.capacity {
                self.buffer.push(sample);
            } else {
                let idx = self.rng.gen_range(0..self.seen + 1);
                if idx < self.capacity {
                    self.buffer[idx] = sample;
                }
            }
            self.seen += 1;
        }
    }

    pub fn drain(&mut self, count: usize) -> Vec<DataSample> {
        let actual = count.min(self.buffer.len());
        if actual == 0 {
            return Vec::new();
        }

        let mut samples: Vec<DataSample> = self.buffer.drain(0..actual).collect();
        samples.shuffle(&mut self.rng);
        samples
    }

    pub fn shuffle(&mut self) {
        self.buffer.shuffle(&mut self.rng);
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

pub fn shuffle_shards(shards: &mut [crate::dataset::scanner::ShardPath]) {
    let mut rng = StdRng::from_entropy();
    shards.shuffle(&mut rng);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::compression::Compression;
    use crate::dataset::scanner::{ShardPath, ShardScanner};
    use std::path::PathBuf;

    fn sample(text: &str) -> DataSample {
        DataSample {
            id: uuid::Uuid::new_v4(),
            text: text.into(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: crate::types::SourceInfo {
                name: "test".into(),
                url: None,
                trust_score: 0.5,
                category: crate::types::SourceCategory::Other,
                fetch_timestamp: 0,
            },
            stats: crate::types::SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        }
    }

    #[test]
    fn test_shuffle_buffer_new() {
        let buf = ShuffleBuffer::new(100);
        assert_eq!(buf.capacity(), 100);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_shuffle_buffer_push_and_drain() {
        let mut buf = ShuffleBuffer::new(10);
        let samples = vec![sample("a"), sample("b"), sample("c")];
        buf.push(samples);
        assert_eq!(buf.len(), 3);
        assert!(!buf.is_empty());
        let drained = buf.drain(2);
        assert_eq!(drained.len(), 2);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_shuffle_buffer_drain_more_than_available() {
        let mut buf = ShuffleBuffer::new(10);
        buf.push(vec![sample("a")]);
        let drained = buf.drain(100);
        assert_eq!(drained.len(), 1);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_shuffle_buffer_drain_empty() {
        let mut buf = ShuffleBuffer::new(10);
        let drained = buf.drain(5);
        assert!(drained.is_empty());
    }

    #[test]
    fn test_shuffle_buffer_clear() {
        let mut buf = ShuffleBuffer::new(10);
        buf.push(vec![sample("a"), sample("b")]);
        assert!(!buf.is_empty());
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_shuffle_buffer_exceed_capacity() {
        let mut buf = ShuffleBuffer::new(2);
        buf.push(vec![sample("a"), sample("b"), sample("c"), sample("d")]);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_shuffle_shards() {
        let mut shards = vec![
            ShardPath {
                path: PathBuf::from("a.arrow"),
                compression: Compression::None,
                size_bytes: 10,
                split: "train".into(),
            },
            ShardPath {
                path: PathBuf::from("b.arrow"),
                compression: Compression::None,
                size_bytes: 20,
                split: "train".into(),
            },
            ShardPath {
                path: PathBuf::from("c.arrow"),
                compression: Compression::None,
                size_bytes: 30,
                split: "val".into(),
            },
        ];
        let original = shards.clone();
        shuffle_shards(&mut shards);
        // Order may (or may not) change, but length must stay
        assert_eq!(shards.len(), original.len());
    }
}
