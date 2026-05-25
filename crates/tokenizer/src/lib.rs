//! Nexora Tokenizer - Text processing dan tokenization
//!
//! Module ini menyediakan fungsi tokenization untuk Nexora AI system

pub mod bpe_tokenizer;
pub mod pretokenizer;
pub mod special_tokens;
pub mod tokenizer_core;
pub mod tokenizer_io;
pub mod trie;
pub mod unicode_normalizer;
pub mod vocab_builder;

pub use bpe_tokenizer::{BpeConfig, BpeTokenizer, TokenizerStats};
pub use pretokenizer::{
    pretokenize, pretokenize_with_config, PieceType, PreTokenized, PreTokenizedPiece, PreTokenizer,
    PreTokenizerConfig,
};
pub use special_tokens::{
    get_special_token_id, get_special_token_str, is_special_token_str, SpecialTokenID,
    SpecialTokens,
};
pub use tokenizer_core::{
    create_tokenizer, decode_tokens, tokenize_text, MergeRule, TokenPair, TokenizerConfig,
    TokenizerCore,
};
pub use tokenizer_io::{
    encode_text, load_tokenizer, save_tokenizer, TokenizerComparison, TokenizerIO,
};
pub use trie::{create_trie, lookup_sequence, Trie, TrieNode, TrieStats};
pub use unicode_normalizer::{
    normalize_nfc, normalize_nfd, normalize_nfkc, normalize_nfkd, normalize_text,
    normalize_text_with_config, NormalizationConfig, NormalizationForm, UnicodeNormalizer,
};
pub use vocab_builder::{
    build_vocab_from_file, build_vocab_from_texts, create_byte_level_vocab, VocabBuilder,
    VocabBuilderConfig, VocabBuilderStats, VocabEntry,
};

/// Unified tokenizer trait for encoding text and decoding token IDs.
pub trait Tokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String>;
    fn vocab_size(&self) -> usize;
}
