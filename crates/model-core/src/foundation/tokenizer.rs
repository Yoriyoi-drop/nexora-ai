//! Fallback tokenizer utilities — byte-level encoding/decoding when no NxrTokenizer is set.
//!
//! Single responsibility: provide stateless byte↔token-id conversion.

/// Encode a text string into byte-based token IDs (fallback, no real tokenizer).
pub fn byte_encode(text: &str) -> Vec<u32> {
    text.bytes().map(|b| b as u32).collect()
}

/// Decode byte-based token IDs back into a string (fallback, no real tokenizer).
pub fn byte_decode(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids
        .iter()
        .map(|&id| if id < 256 { id as u8 } else { b'?' })
        .collect();
    String::from_utf8_lossy(&bytes).to_string()
}
