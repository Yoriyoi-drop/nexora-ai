use std::collections::VecDeque;
use uuid::Uuid;
use tokio::sync::mpsc;

use crate::{InferenceRequest, InferenceResponse};

/// A grouping key for batching compatible requests together.
#[derive(Debug, Clone)]
pub struct BatchKey {
    pub(crate) model_id: String,
    pub(crate) priority: u8,
}

impl BatchKey {
    pub fn from_request(request: &InferenceRequest) -> Self {
        Self {
            model_id: request.model_id.clone(),
            priority: request.priority,
        }
    }
}

/// A single request within a batch.
#[derive(Debug)]
pub struct BatchRequest {
    pub request_id: Uuid,
    pub prompt: String,
    pub response_tx: mpsc::Sender<InferenceResponse>,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: u32,
}

/// A batch of requests ready for processing.
#[derive(Debug)]
pub struct Batch {
    pub batch_id: Uuid,
    pub requests: Vec<BatchRequest>,
}

impl Batch {
    fn new(requests: Vec<BatchRequest>) -> Self {
        Self {
            batch_id: Uuid::new_v4(),
            requests,
        }
    }

    /// Returns the number of requests in this batch.
    pub fn size(&self) -> usize {
        self.requests.len()
    }
}

/// Collects incoming requests into batches based on configurable limits.
#[derive(Debug)]
pub struct BatchCollector {
    max_batch_size: usize,
    max_queue_depth: usize,
    pending_requests: VecDeque<BatchRequest>,
    ready_batches: VecDeque<Batch>,
}

impl BatchCollector {
    pub fn new(max_batch_size: usize, max_queue_depth: usize) -> Self {
        Self {
            max_batch_size,
            max_queue_depth,
            pending_requests: VecDeque::new(),
            ready_batches: VecDeque::new(),
        }
    }

    pub fn add_request(
        &mut self,
        request: InferenceRequest,
        response_tx: mpsc::Sender<InferenceResponse>,
    ) {
        if self.pending_requests.len() >= self.max_queue_depth {
            tracing::warn!("Batch collector queue full, dropping request");
            return;
        }
        self.pending_requests.push_back(BatchRequest {
            request_id: request.request_id,
            prompt: request.prompt,
            response_tx,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
        });
        if self.pending_requests.len() >= self.max_batch_size {
            self.flush_ready();
        }
    }

    fn flush_ready(&mut self) {
        if self.pending_requests.is_empty() {
            return;
        }
        let mut batch_requests = Vec::with_capacity(self.max_batch_size);
        while let Some(req) = self.pending_requests.pop_front() {
            batch_requests.push(req);
            if batch_requests.len() >= self.max_batch_size {
                break;
            }
        }
        if !batch_requests.is_empty() {
            self.ready_batches.push_back(Batch::new(batch_requests));
        }
    }

    pub fn drain_ready(&mut self) -> Vec<Batch> {
        self.flush_ready();
        self.ready_batches.drain(..).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.pending_requests.is_empty() && self.ready_batches.is_empty()
    }

    pub fn contains_batch(&self, batch_id: Uuid) -> bool {
        self.ready_batches.iter().any(|b| b.batch_id == batch_id)
    }

    pub fn batch_count(&self) -> usize {
        self.ready_batches.len()
    }

    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }
}
