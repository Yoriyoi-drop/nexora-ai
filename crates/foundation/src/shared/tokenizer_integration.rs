use std::sync::Arc;
use parking_lot::RwLock;

pub type NxrTokenizerRef = Arc<RwLock<nexora_tokenizer::BpeTokenizer>>;
