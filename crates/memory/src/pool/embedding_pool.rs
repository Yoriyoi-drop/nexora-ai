use parking_lot::RwLock;

/// Pre-allocated embedding buffer
pub struct EmbeddingBuffer {
    pub data: Vec<f32>,
    pub dim: usize,
    pub in_use: bool,
}

impl EmbeddingBuffer {
    pub fn new(dim: usize) -> Self {
        Self {
            data: vec![0.0; dim],
            dim,
            in_use: false,
        }
    }

    pub fn fill(&mut self, values: &[f32]) {
        let len = values.len().min(self.dim);
        self.data[..len].copy_from_slice(&values[..len]);
    }
}

pub struct EmbeddingPool {
    pool: RwLock<Vec<EmbeddingBuffer>>,
    dim: usize,
    capacity: usize,
}

impl EmbeddingPool {
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            pool: RwLock::new(Vec::with_capacity(capacity)),
            dim,
            capacity,
        }
    }

    pub fn acquire(&self) -> EmbeddingBuffer {
        let mut pool = self.pool.write();
        if let Some(pos) = pool.iter().position(|b| !b.in_use) {
            let mut buf = pool.swap_remove(pos);
            buf.in_use = true;
            return buf;
        }
        let mut buf = EmbeddingBuffer::new(self.dim);
        buf.in_use = true;
        buf
    }

    pub fn release(&self, mut buf: EmbeddingBuffer) {
        buf.in_use = false;
        let mut pool = self.pool.write();
        if pool.len() < self.capacity {
            pool.push(buf);
        }
    }

    pub fn size(&self) -> usize {
        self.pool.read().len()
    }
}
