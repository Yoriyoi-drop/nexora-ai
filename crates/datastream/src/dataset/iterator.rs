use crate::types::DataSample;
use super::shuffle::ShuffleBuffer;

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
