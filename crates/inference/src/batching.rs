use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;

/// Key for grouping compatible requests into a batch.
/// Requests with the same model and compatible sampling params can be batched.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BatchKey {
    pub model_id: String,
    pub temperature_bucket: u8,
    pub top_k: u32,
    pub top_p_bucket: u8,
}

impl BatchKey {
    pub fn from_request(req: &crate::InferenceRequest) -> Self {
        let t_bucket = (req.temperature.clamp(0.0, 5.0) * 10.0) as u8;
        let p_bucket = (req.top_p.clamp(0.0, 1.0) * 10.0) as u8;
        Self {
            model_id: req.model_id.clone(),
            temperature_bucket: t_bucket,
            top_k: req.top_k,
            top_p_bucket: p_bucket,
        }
    }
}

/// A single request within a batch, with a channel to send its response.
#[derive(Debug)]
pub struct BatchRequest {
    pub request_id: Uuid,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub response_tx: tokio::sync::mpsc::Sender<crate::InferenceResponse>,
}

/// A batch of compatible requests to be processed together.
#[derive(Debug)]
pub struct Batch {
    pub batch_id: Uuid,
    pub key: BatchKey,
    pub requests: Vec<BatchRequest>,
    pub created_at: Instant,
}

impl Batch {
    pub fn new(key: BatchKey) -> Self {
        Self {
            batch_id: Uuid::new_v4(),
            key,
            requests: Vec::with_capacity(32),
            created_at: Instant::now(),
        }
    }

    pub fn is_full(&self, max_size: usize) -> bool {
        self.requests.len() >= max_size
    }

    pub fn add_request(&mut self, req: BatchRequest) {
        self.requests.push(req);
    }

    pub fn size(&self) -> usize {
        self.requests.len()
    }
}

/// Collects pending requests and forms them into batches by compatibility.
pub struct BatchCollector {
    pending: HashMap<BatchKey, Batch>,
    max_batch_size: usize,
    max_collect_time_ms: u64,
}

impl BatchCollector {
    pub fn new(max_batch_size: usize, max_collect_time_ms: u64) -> Self {
        Self {
            pending: HashMap::new(),
            max_batch_size,
            max_collect_time_ms,
        }
    }

    /// Add a request to the pending pool. Returns the batch key it was grouped into.
    pub fn add_request(&mut self, request: crate::InferenceRequest, response_tx: tokio::sync::mpsc::Sender<crate::InferenceResponse>) -> BatchKey {
        let key = BatchKey::from_request(&request);
        let entry = self.pending.entry(key.clone()).or_insert_with(|| Batch::new(key.clone()));
        entry.add_request(BatchRequest {
            request_id: request.request_id,
            prompt: request.prompt,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            response_tx,
        });
        key
    }

    /// Drain all ready batches (full or timed out).
    /// Returns batches that are ready for processing.
    pub fn drain_ready(&mut self) -> Vec<Batch> {
        let pending_len = self.pending.len();
        let mut ready = Vec::with_capacity(pending_len);
        let now = Instant::now();

        let mut to_retain: Vec<BatchKey> = Vec::with_capacity(pending_len);
        let mut to_drain: Vec<BatchKey> = Vec::with_capacity(pending_len);
        for (key, batch) in &self.pending {
            if batch.is_full(self.max_batch_size) || now.duration_since(batch.created_at).as_millis() as u64 >= self.max_collect_time_ms {
                to_drain.push(key.clone());
            } else {
                to_retain.push(key.clone());
            }
        }
        for key in &to_drain {
            if let Some(batch) = self.pending.remove(key) {
                ready.push(batch);
            }
        }
        // Re-insert fresh batches for drained keys
        for key in &to_drain {
            if !self.pending.contains_key(key) {
                self.pending.insert(key.clone(), Batch::new(key.clone()));
            }
        }

        ready
    }

    /// Drain ALL pending batches (used during shutdown).
    pub fn drain_all(&mut self) -> Vec<Batch> {
        let batches: Vec<Batch> = self.pending.drain().map(|(_, b)| b).collect();
        batches
    }

    pub fn pending_count(&self) -> usize {
        self.pending.values().map(|b| b.size()).sum()
    }

    pub fn batch_count(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn make_request(model_id: &str, temperature: f32, top_k: u32, top_p: f32) -> crate::InferenceRequest {
        crate::InferenceRequest {
            request_id: Uuid::new_v4(),
            model_id: model_id.to_string(),
            temperature,
            top_k,
            top_p,
            ..crate::InferenceRequest::default()
        }
    }

    #[test]
    fn test_batch_key_equality() {
        let req1 = make_request("model-a", 0.7, 40, 0.9);
        let req2 = make_request("model-a", 0.7, 40, 0.9);
        let req3 = make_request("model-b", 0.7, 40, 0.9);

        let k1 = BatchKey::from_request(&req1);
        let k2 = BatchKey::from_request(&req2);
        let k3 = BatchKey::from_request(&req3);

        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn test_batch_collector_grouping() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut collector = BatchCollector::new(4, 5000);

        let req1 = make_request("model-a", 0.7, 40, 0.9);
        let req2 = make_request("model-a", 0.7, 40, 0.9);
        let req3 = make_request("model-b", 0.7, 40, 0.9);

        collector.add_request(req1.clone(), tx.clone());
        collector.add_request(req2.clone(), tx.clone());
        collector.add_request(req3.clone(), tx.clone());

        assert_eq!(collector.batch_count(), 2);
        assert_eq!(collector.pending_count(), 3);
    }

    #[test]
    fn test_batch_drain_full() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut collector = BatchCollector::new(2, 5000);

        let req1 = make_request("model-a", 0.7, 40, 0.9);
        let req2 = make_request("model-a", 0.7, 40, 0.9);
        let req3 = make_request("model-a", 0.7, 40, 0.9);

        collector.add_request(req1, tx.clone());
        assert_eq!(collector.drain_ready().len(), 0);

        collector.add_request(req2, tx.clone());
        let ready = collector.drain_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].size(), 2);

        collector.add_request(req3, tx.clone());
        assert_eq!(collector.pending_count(), 1);
    }

    #[test]
    fn test_batch_collector_empty_drain() {
        let mut collector = BatchCollector::new(4, 5000);
        assert!(collector.drain_ready().is_empty());
        assert!(collector.drain_all().is_empty());
    }

    #[test]
    fn test_batch_collector_drain_all() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut collector = BatchCollector::new(10, 5000);

        let req1 = make_request("model-a", 0.7, 40, 0.9);
        let req2 = make_request("model-b", 0.7, 40, 0.9);

        collector.add_request(req1, tx.clone());
        collector.add_request(req2, tx.clone());

        let all = collector.drain_all();
        assert_eq!(all.len(), 2);
        assert_eq!(collector.pending_count(), 0);
    }
}
