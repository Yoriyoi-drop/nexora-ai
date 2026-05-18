use crate::{FinishReason, GeneratedToken, InferenceRequest};

/// Lifecycle state of a single sequence in the continuous batching pool.
#[derive(Debug, Clone, PartialEq)]
pub enum SeqState {
    /// Request received, not yet started processing prompt
    New,
    /// Still processing prompt tokens (building up KV cache)
    Prefilling,
    /// Generating new tokens (all prompt tokens have KV cache)
    Generating,
    /// Finished (max tokens, EOS, stop, error)
    Finished(FinishReason),
}

/// A single sequence being processed by the continuous batching engine.
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique sequence ID (used as PagedKVCache sequence ID)
    pub id: u64,
    /// Prompt token IDs
    pub prompt: Vec<u32>,
    /// Tokens generated so far
    pub generated: Vec<GeneratedToken>,
    /// Current lifecycle state
    pub state: SeqState,
    /// Maximum tokens to generate (including prompt)
    pub max_tokens: u32,
    /// Sampling temperature
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: u32,
    /// EOS token ID (0 by default)
    pub eos_token_id: u32,
    /// How many prompt tokens have been processed through the model
    pub prompt_pos: usize,
}

impl Sequence {
    /// Create a new sequence from an inference request.
    /// Consumes `request.input_tokens` as the prompt.
    pub fn from_request(id: u64, request: &InferenceRequest) -> Self {
        Self {
            id,
            prompt: request.input_tokens.clone(),
            generated: Vec::with_capacity(request.max_tokens as usize),
            state: SeqState::New,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            eos_token_id: 0,
            prompt_pos: 0,
        }
    }

    /// Total tokens processed so far (prompt + generated).
    pub fn total_tokens(&self) -> usize {
        self.prompt_pos + self.generated.len()
    }

    /// Whether this sequence has reached its max tokens limit.
    pub fn reached_max_tokens(&self) -> bool {
        self.total_tokens() >= self.max_tokens as usize
    }

    /// Whether the sequence still has prompt tokens to process.
    pub fn has_pending_prompt(&self) -> bool {
        self.prompt_pos < self.prompt.len()
    }

    /// The next input token for the model forward pass.
    /// Returns `None` if the sequence is finished or has no tokens.
    pub fn next_input_token(&self) -> Option<u32> {
        if self.has_pending_prompt() {
            Some(self.prompt[self.prompt_pos])
        } else {
            self.generated.last().map(|t| t.token_id)
        }
    }

    /// Advance prompt position by one token (called after prefill forward pass).
    pub fn advance_prompt(&mut self) {
        if self.prompt_pos < self.prompt.len() {
            self.prompt_pos += 1;
        }
    }

    /// Append a generated token to this sequence.
    pub fn push_token(&mut self, token: GeneratedToken) {
        self.generated.push(token);
    }

    /// Finish the sequence with the given reason.
    pub fn finish(&mut self, reason: FinishReason) {
        self.state = SeqState::Finished(reason);
    }

    /// Whether this sequence is finished.
    pub fn is_finished(&self) -> bool {
        matches!(self.state, SeqState::Finished(_))
    }

    /// Whether the sequence is ready for a forward pass step.
    pub fn is_ready(&self) -> bool {
        match self.state {
            SeqState::New => !self.prompt.is_empty(),
            SeqState::Prefilling => self.has_pending_prompt(),
            SeqState::Generating => !self.reached_max_tokens(),
            SeqState::Finished(_) => false,
        }
    }
}
