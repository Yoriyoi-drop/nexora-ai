use parking_lot::RwLock;
use std::sync::Arc;

pub type NxrTokenizerRef = Arc<RwLock<nexora_foundation::tokenizer::BpeTokenizer>>;
