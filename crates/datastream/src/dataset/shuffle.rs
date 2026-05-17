use rand::seq::SliceRandom;
use rand::rngs::StdRng;
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
