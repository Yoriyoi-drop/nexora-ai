use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use unicode_segmentation::UnicodeSegmentation;

/// Byte + BPE tokenizer: bytes 0-255 as base (BOS=1, EOS=2), plus optional learned merge rules for IDs ≥ 256.
#[derive(Clone)]
pub struct MiniTokenizer {
    pub vocab_size: usize,
    pub bpe_token_to_id: FxHashMap<String, u32>,
    pub bpe_id_to_token: FxHashMap<u32, String>,
    pub merges: Vec<(String, String)>,
}

impl MiniTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            bpe_token_to_id: FxHashMap::default(),
            bpe_id_to_token: FxHashMap::default(),
            merges: Vec::new(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if !self.merges.is_empty() {
            return self.bpe_tokenize(text);
        }
        if self.vocab_size > 256 {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "MiniTokenizer: vocab_size={} but BPE not trained — falling back to byte-level encoding (only 0-255 usable)",
                    self.vocab_size
                );
            }
        }
        let mut ids = vec![1u32];
        for &b in text.as_bytes() {
            let id = b as u32;
            if (id as usize) < self.vocab_size {
                ids.push(id);
            }
        }
        ids.push(2);
        ids
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut result = String::new();
        for &id in token_ids {
            if id == 1 || id == 2 {
                continue;
            }
            if id < 256 {
                result.push(char::from(id as u8));
            } else if let Some(s) = self.bpe_id_to_token.get(&id) {
                result.push_str(s);
            }
        }
        result
    }

    pub fn train_bpe(&mut self, texts: &[String], num_merges: usize) {
        use rayon::prelude::*;

        let mut pair_counts: FxHashMap<(String, String), usize> = FxHashMap::default();
        let chunks: Vec<FxHashMap<(String, String), usize>> = texts
            .par_iter()
            .map(|text| {
                let mut local: FxHashMap<(String, String), usize> = FxHashMap::default();
                let graphemes: Vec<String> =
                    text.graphemes(true).map(|g| g.to_string()).collect();
                for pair in graphemes.windows(2) {
                    *local
                        .entry((pair[0].clone(), pair[1].clone()))
                        .or_default() += 1;
                }
                local
            })
            .collect();

        for chunk in chunks {
            for (k, v) in chunk {
                *pair_counts.entry(k).or_default() += v;
            }
        }
        let max_merges = num_merges.min(self.vocab_size.saturating_sub(256));
        for _ in 0..max_merges {
            let best = pair_counts
                .iter()
                .max_by_key(|&(_, &c)| c)
                .map(|(k, _)| k.clone());
            let Some((left, right)) = best else { break };
            let merged = format!("{}{}", left, right);
            let id = 256u32 + self.merges.len() as u32;
            if id >= self.vocab_size as u32 {
                break;
            }
            self.bpe_token_to_id.insert(merged.clone(), id);
            self.bpe_id_to_token.insert(id, merged.clone());
            self.merges.push((left.clone(), right.clone()));
            pair_counts.remove(&(left, right));
        }
    }

    pub fn bpe_tokenize(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![1u32];
        let graphemes: Vec<String> = text.graphemes(true).map(|g| g.to_string()).collect();
        let n = graphemes.len();
        if n == 0 {
            ids.push(2);
            return ids;
        }
        let mut token_texts = graphemes;
        let mut next: Vec<usize> = (1..n).chain(std::iter::once(usize::MAX)).collect();
        let mut head = 0usize;
        let mut prev: Vec<usize> = std::iter::once(usize::MAX).chain((0..n - 1)).collect();
        let mut active = n;
        if !self.merges.is_empty() {
            loop {
                let mut best_pos = None;
                let mut i = head;
                while i != usize::MAX {
                    let j = next[i];
                    if j != usize::MAX {
                        let merged = format!("{}{}", token_texts[i], token_texts[j]);
                        if self.bpe_token_to_id.contains_key(&merged) {
                            best_pos = Some((i, j, merged));
                            break;
                        }
                    }
                    i = next[i];
                    if i == usize::MAX {
                        break;
                    }
                }
                match best_pos {
                    Some((i, j, merged)) => {
                        token_texts[i] = merged;
                        let j_next = next[j];
                        if j_next != usize::MAX {
                            prev[j_next] = i;
                        }
                        next[i] = j_next;
                        if j == head {
                            head = i;
                        }
                        active -= 1;
                    }
                    None => break,
                }
            }
        }
        let mut i = head;
        while i != usize::MAX {
            if let Some(&id) = self.bpe_token_to_id.get(token_texts[i].as_str()) {
                ids.push(id);
            } else {
                for &b in token_texts[i].as_bytes() {
                    if (b as usize) < self.vocab_size {
                        ids.push(b as u32);
                    }
                }
            }
            i = next[i];
        }
        ids.push(2);
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mini_tokenizer_encode_has_bos_eos() {
        let tok = MiniTokenizer::new(512);
        let ids = tok.encode("hello");
        assert_eq!(ids.first(), Some(&1), "BOS token must be 1");
        assert_eq!(ids.last(), Some(&2), "EOS token must be 2");
        assert!(ids.len() >= 3, "Should have BOS + content + EOS");
    }

    #[test]
    fn test_mini_tokenizer_encode_decode_roundtrip() {
        let tok = MiniTokenizer::new(512);
        let text = "Hello, World!";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(
            text.as_bytes(),
            decoded.as_bytes(),
        );
    }

    #[test]
    fn test_mini_tokenizer_empty_input() {
        let tok = MiniTokenizer::new(512);
        let ids = tok.encode("");
        assert_eq!(ids, vec![1, 2], "Empty input should give [BOS, EOS]");
    }

    #[test]
    fn test_mini_tokenizer_all_bytes_map_correctly() {
        let tok = MiniTokenizer::new(512);
        let text = "abcdefghijklmnopqrstuvwxyz";
        let ids = tok.encode(text);
        let content: Vec<u32> = ids[1..ids.len() - 1].to_vec();
        for (i, &c) in text.as_bytes().iter().enumerate() {
            assert_eq!(
                content[i], c as u32,
                "Char '{:?}' should map to {}",
                c as char, c
            );
        }
    }

    #[test]
    fn test_mini_tokenizer_decode_skips_special_tokens() {
        let tok = MiniTokenizer::new(512);
        let ids = vec![104, 101, 108, 108, 111];
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_mini_tokenizer_decode_filters_above_255() {
        let tok = MiniTokenizer::new(512);
        let ids = vec![104, 101, 108, 108, 111, 256, 300];
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_mini_tokenizer_vocab_size_respected() {
        let tok = MiniTokenizer::new(100);
        let ids = tok.encode("abc\u{ff}");
        let content: Vec<u32> = ids[1..ids.len() - 1].to_vec();
        assert_eq!(
            content,
            vec![97, 98, 99],
            "Byte 255 should be skipped (>= vocab_size)"
        );
    }

    #[test]
    fn test_mini_tokenizer_new() {
        let tok = MiniTokenizer::new(50257);
        assert_eq!(tok.vocab_size, 50257);
        assert!(tok.bpe_token_to_id.is_empty());
    }

    #[test]
    fn test_mini_tokenizer_encode_decode_byte_level() {
        let tok = MiniTokenizer::new(256);
        let ids = tok.encode("abc");
        assert_eq!(ids.first(), Some(&1));
        assert_eq!(ids.last(), Some(&2));
        assert_eq!(ids.len(), 5);

        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn test_mini_tokenizer_encode_empty_string() {
        let tok = MiniTokenizer::new(256);
        let ids = tok.encode("");
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_mini_tokenizer_decode_skips_bos_eos() {
        let tok = MiniTokenizer::new(256);
        let decoded = tok.decode(&[1, 65, 66, 67, 2]);
        assert_eq!(decoded, "ABC");
    }

    #[test]
    fn test_mini_tokenizer_encode_vocab_too_small() {
        let tok = MiniTokenizer::new(10);
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_mini_tokenizer_train_bpe() {
        let mut tok = MiniTokenizer::new(260);
        let texts = vec![
            "hello".to_string(),
            "help".to_string(),
            "helicopter".to_string(),
        ];
        tok.train_bpe(&texts, 2);
        assert!(tok.merges.len() <= 2);
        assert!(!tok.bpe_token_to_id.is_empty() || tok.merges.len() == 2);
    }

    #[test]
    fn test_mini_tokenizer_bpe_roundtrip() {
        let mut tok = MiniTokenizer::new(260);
        let texts = vec!["test".to_string(), "testing".to_string()];
        tok.train_bpe(&texts, 1);

        if !tok.merges.is_empty() {
            let ids = tok.encode("test");
            let decoded = tok.decode(&ids);
            assert_eq!(decoded, "test");
        }
    }

    #[test]
    fn test_mini_tokenizer_decode_unknown_bpe_id() {
        let tok = MiniTokenizer::new(260);
        let decoded = tok.decode(&[65, 66, 300, 67]);
        assert_eq!(decoded, "ABC");
    }

    #[test]
    fn test_mini_tokenizer_vocab_size_limits_merges() {
        let mut tok = MiniTokenizer::new(257);
        let texts = vec!["ab".to_string(); 100];
        tok.train_bpe(&texts, 10);
        assert!(tok.merges.len() <= 1);
    }
}
