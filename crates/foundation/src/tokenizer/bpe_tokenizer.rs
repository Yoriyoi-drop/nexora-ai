use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::tokenizer::Tokenizer;

pub type TokenId = u32;
type Pair = (TokenId, TokenId);

const SPACE_SENTINEL: &str = "\u{0120}";
const SENTINEL_BYTE: u8 = 0x20;

#[derive(Debug, Clone, Eq, PartialEq)]
struct PairFreq {
    pair: Pair,
    freq: u32,
}

impl Ord for PairFreq {
    fn cmp(&self, other: &Self) -> Ordering {
        self.freq
            .cmp(&other.freq)
            .then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for PairFreq {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeConfig {
    pub vocab_size: usize,
    pub special_tokens: HashMap<String, u32>,
    pub min_frequency: u32,
    pub unknown_token: String,
    pub pad_token: String,
    pub bos_token: String,
    pub eos_token: String,
    pub space_sentinel: String,
}

impl Default for BpeConfig {
    fn default() -> Self {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<unk>".to_string(), 0);
        special_tokens.insert("<pad>".to_string(), 1);
        special_tokens.insert("<bos>".to_string(), 2);
        special_tokens.insert("<eos>".to_string(), 3);
        Self {
            vocab_size: 30000,
            special_tokens,
            min_frequency: 2,
            unknown_token: "<unk>".to_string(),
            pad_token: "<pad>".to_string(),
            bos_token: "<bos>".to_string(),
            eos_token: "<eos>".to_string(),
            space_sentinel: SPACE_SENTINEL.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    config: BpeConfig,
    vocab: FxHashMap<String, TokenId>,
    reverse_vocab: FxHashMap<TokenId, String>,
    merge_ranks: FxHashMap<Pair, u32>,
    merges: Vec<Pair>,
    next_id: TokenId,
}

fn byte_char(b: u8) -> String {
    match b {
        0 => String::from("<NUL>"),
        9 => String::from("<TAB>"),
        10 => String::from("<LF>"),
        13 => String::from("<CR>"),
        32 => String::from("<SP>"),
        127 => String::from("<DEL>"),
        0..=31 | 128..=159 => {
            let cp = 256u32 + b as u32;
            char::from_u32(cp)
                .map(|c| c.to_string())
                .unwrap_or_else(|| format!("<BYTE_{}>", b))
        }
        _ => (b as char).to_string(),
    }
}

impl BpeTokenizer {
    pub fn new(config: BpeConfig) -> Self {
        let mut vocab = FxHashMap::default();
        let mut reverse_vocab = FxHashMap::default();
        let mut next_id = 0u32;

        for (token, &id) in &config.special_tokens {
            vocab.insert(token.clone(), id);
            reverse_vocab.insert(id, token.clone());
            next_id = next_id.max(id + 1);
        }

        let base_start = next_id;
        for byte in 0..=255u8 {
            let token = byte_char(byte);
            if !vocab.contains_key(&token) {
                let id = next_id;
                vocab.insert(token.clone(), id);
                reverse_vocab.insert(id, token);
                next_id += 1;
            }
        }

        info!(
            "BpeTokenizer: {} special, {} byte tokens, next_id={}",
            config.special_tokens.len(),
            next_id - base_start,
            next_id,
        );

        Self {
            config,
            vocab,
            reverse_vocab,
            merge_ranks: FxHashMap::default(),
            merges: Vec::new(),
            next_id,
        }
    }

    pub fn train(&mut self, corpus: &str) -> Result<(), Box<dyn std::error::Error>> {
        let train_start = std::time::Instant::now();
        let corpus_bytes = corpus.len();
        let corpus_lines = corpus.lines().count();

        info!(
            "Byte-level BPE training: vocab_size={}, {} bytes, {} lines",
            self.config.vocab_size, corpus_bytes, corpus_lines,
        );

        let processed: Vec<String> = corpus
            .par_lines()
            .map(|line| {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    return String::new();
                }
                let mut out = String::with_capacity(trimmed.len() + 4);
                let mut first = true;
                for word in trimmed.split_whitespace() {
                    if !first {
                        out.push(' ');
                    }
                    out.push_str(SPACE_SENTINEL);
                    out.push_str(word);
                    first = false;
                }
                out
            })
            .collect();
        let text = processed.join("\n");

        let mut words_flat: Vec<Vec<TokenId>> = Vec::new();
        let mut word_freqs: Vec<u32> = Vec::new();

        {
            let mut buf = Vec::new();
            let mut reading_word = false;
            for &b in text.as_bytes() {
                if b == SENTINEL_BYTE || !b.is_ascii_whitespace() {
                    if !reading_word && b == SENTINEL_BYTE {
                        reading_word = true;
                        buf.clear();
                        buf.push(b as TokenId);
                    } else if reading_word && b != SENTINEL_BYTE {
                        buf.push(b as TokenId);
                    }
                } else {
                    if reading_word && !buf.is_empty() {
                        words_flat.push(buf.clone());
                        word_freqs.push(1);
                    }
                    reading_word = false;
                }
            }
            if reading_word && !buf.is_empty() {
                words_flat.push(buf);
                word_freqs.push(1);
            }
        }

        let mut word_ids: Vec<Vec<TokenId>> = Vec::new();
        let mut word_freqs_deduped: Vec<u32> = Vec::new();
        {
            let mut seen: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
            for w in words_flat.iter() {
                let hash = fx_hash(w);
                let entry = seen.entry(hash).or_default();
                let mut found = false;
                for &existing in entry.iter() {
                    if word_ids[existing] == *w {
                        word_freqs_deduped[existing] += 1;
                        found = true;
                        break;
                    }
                }
                if !found {
                    entry.push(word_ids.len());
                    word_ids.push(w.clone());
                    word_freqs_deduped.push(1);
                }
            }
        }

        let unique_words = word_ids.len();
        let total_words: u32 = word_freqs_deduped.iter().sum();
        info!(
            "Corpus: {} unique words, {} total, initial vocab={}",
            unique_words,
            total_words,
            self.vocab.len(),
        );

        let mut pair_freqs: FxHashMap<Pair, u32> = FxHashMap::default();
        let mut pair_to_words: FxHashMap<Pair, Vec<usize>> = FxHashMap::default();
        let mut heap: BinaryHeap<PairFreq> = BinaryHeap::new();

        for (idx, tokens) in word_ids.iter().enumerate() {
            let freq = word_freqs_deduped[idx];
            for pair in tokens.windows(2) {
                let p = (pair[0], pair[1]);
                *pair_freqs.entry(p).or_insert(0) += freq;
                pair_to_words.entry(p).or_default().push(idx);
            }
        }

        for (&p, &f) in &pair_freqs {
            heap.push(PairFreq { pair: p, freq: f });
        }

        let merge_start = std::time::Instant::now();
        let log_interval = (self.config.vocab_size / 20).max(100);
        let max_merges = self.config.vocab_size.saturating_sub(self.vocab.len());
        let mut merge_count = 0usize;

        while self.vocab.len() < self.config.vocab_size {
            let (pair, freq) = loop {
                let entry = match heap.pop() {
                    Some(e) => e,
                    None => return Ok(()),
                };
                let current = pair_freqs.get(&entry.pair).copied().unwrap_or(0);
                if current == entry.freq && current >= self.config.min_frequency {
                    break (entry.pair, entry.freq);
                }
                if current == 0 {
                    pair_freqs.remove(&entry.pair);
                }
                if heap.is_empty() {
                    return Ok(());
                }
            };

            let (id1, id2) = pair;
            let new_id = self.next_id;
            self.next_id += 1;

            let s1 = self.reverse_vocab.get(&id1).cloned().unwrap_or_else(|| "?".to_string());
            let s2 = self.reverse_vocab.get(&id2).cloned().unwrap_or_else(|| "?".to_string());
            let merged_str = format!("{}{}", s1, s2);

            self.vocab.insert(merged_str.clone(), new_id);
            self.reverse_vocab.insert(new_id, merged_str);
            self.merge_ranks.insert(pair, merge_count as u32);
            self.merges.push(pair);
            merge_count += 1;

            let affected = pair_to_words.remove(&pair).unwrap_or_default();

            for &idx in &affected {
                let freq = word_freqs_deduped[idx];
                let tokens = &word_ids[idx];

                let old_pairs: Vec<Pair> = tokens.windows(2).map(|w| (w[0], w[1])).collect();

                for &p in &old_pairs {
                    if let Some(count) = pair_freqs.get_mut(&p) {
                        *count = count.saturating_sub(freq);
                    }
                    if let Some(indices) = pair_to_words.get_mut(&p) {
                        indices.retain(|&i| i != idx);
                    }
                }

                let mut new_tokens = Vec::with_capacity(tokens.len());
                let mut i = 0;
                while i < tokens.len() {
                    if i + 1 < tokens.len() && tokens[i] == id1 && tokens[i + 1] == id2 {
                        new_tokens.push(new_id);
                        i += 2;
                    } else {
                        new_tokens.push(tokens[i]);
                        i += 1;
                    }
                }
                word_ids[idx] = new_tokens;

                let new_pairs: Vec<Pair> = word_ids[idx].windows(2).map(|w| (w[0], w[1])).collect();
                for &p in &new_pairs {
                    *pair_freqs.entry(p).or_insert(0) += freq;
                    pair_to_words.entry(p).or_default().push(idx);
                }

                let mut pushed = FxHashMap::default();
                for &p in &old_pairs {
                    if p != pair && !pushed.contains_key(&p) {
                        if let Some(&f) = pair_freqs.get(&p) {
                            heap.push(PairFreq { pair: p, freq: f });
                            pushed.insert(p, ());
                        }
                    }
                }
                for &p in &new_pairs {
                    if !pushed.contains_key(&p) {
                        if let Some(&f) = pair_freqs.get(&p) {
                            heap.push(PairFreq {
                                pair: p,
                                freq: f,
                            });
                            pushed.insert(p, ());
                        }
                    }
                }
            }

            if merge_count % log_interval == 0 {
                let elapsed = merge_start.elapsed();
                let speed = merge_count as f64 / elapsed.as_secs_f64().max(0.001);
                info!(
                    "  Merge {}/{} ({:.0}%) @ {:.0}/s, '{}'+'{}' freq={}",
                    merge_count,
                    max_merges,
                    merge_count as f64 * 100.0 / max_merges.max(1) as f64,
                    speed,
                    s1,
                    s2,
                    freq,
                );
            }
        }

        let total = train_start.elapsed();
        let merge_time = merge_start.elapsed();
        let speed = self.merges.len() as f64 / merge_time.as_secs_f64().max(0.001);
        info!(
            "BPE done in {:?} ({:.3}s): {} vocab, {} merges, {:.0}/s",
            total,
            total.as_secs_f64(),
            self.vocab.len(),
            self.merges.len(),
            speed,
        );
        Ok(())
    }

    pub fn train_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = fs::File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        let corpus = std::str::from_utf8(&mmap)?;
        let result = self.train(corpus);
        drop(mmap);
        drop(file);
        result
    }

    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut tokens = Vec::with_capacity(text.len());
        if let Some(&bos_id) = self.config.special_tokens.get(&self.config.bos_token) {
            tokens.push(bos_id);
        }

        for word in text.split_whitespace() {
            let mut word_ids: Vec<TokenId> = Vec::with_capacity(word.len() + 1);
            word_ids.push(SENTINEL_BYTE as TokenId);
            for &b in word.as_bytes() {
                word_ids.push(b as TokenId);
            }
            tokens.extend(self.bpe_encode_word(&word_ids));
        }

        if let Some(&eos_id) = self.config.special_tokens.get(&self.config.eos_token) {
            tokens.push(eos_id);
        }
        tokens
    }

    fn bpe_encode_word(&self, word: &[TokenId]) -> Vec<TokenId> {
        let mut syms = word.to_vec();
        loop {
            let (best_i, _best_rank) = {
                let mut best = None;
                let mut best_rank = u32::MAX;
                for i in 0..syms.len().saturating_sub(1) {
                    let p = (syms[i], syms[i + 1]);
                    if let Some(&rank) = self.merge_ranks.get(&p) {
                        if rank < best_rank {
                            best_rank = rank;
                            best = Some(i);
                        }
                    }
                }
                match best {
                    Some(i) => (i, best_rank),
                    None => break,
                }
            };

            let left = &syms[best_i];
            let right = &syms[best_i + 1];

            let combined = match (self.reverse_vocab.get(left), self.reverse_vocab.get(right)) {
                (Some(l), Some(r)) => format!("{}{}", l, r),
                _ => break,
            };

            match self.vocab.get(&combined) {
                Some(&new_id) => {
                    syms[best_i] = new_id;
                    syms.remove(best_i + 1);
                }
                None => break,
            }
        }
        syms
    }

    pub fn decode(&self, token_ids: &[TokenId]) -> String {
        let mut bytes = Vec::with_capacity(token_ids.len() * 4);
        for &id in token_ids {
            if id < 256 {
                bytes.push(id as u8);
            } else if let Some(token) = self.reverse_vocab.get(&id) {
                if !self.config.special_tokens.contains_key(token.as_str()) {
                    bytes.extend_from_slice(token.as_bytes());
                }
            }
        }
        let result = String::from_utf8(bytes).unwrap_or_default();
        result.replace(SPACE_SENTINEL, " ")
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.vocab.get(token).copied()
    }

    pub fn id_to_token(&self, id: TokenId) -> Option<&String> {
        self.reverse_vocab.get(&id)
    }

    pub fn merge_rank(&self, left: TokenId, right: TokenId) -> Option<u32> {
        self.merge_ranks.get(&(left, right)).copied()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let config_json = serde_json::to_string_pretty(&self.config)?;
        fs::write(path.join("config.json"), config_json)?;

        let mut vocab_entries: Vec<_> = self.vocab.iter().collect();
        vocab_entries.sort_by_key(|(_, &id)| id);
        let mut vocab_file = fs::File::create(path.join("vocab.txt"))?;
        for (token, &id) in &vocab_entries {
            writeln!(vocab_file, "{} {}", token, id)?;
        }

        let mut merges_file = fs::File::create(path.join("merges.txt"))?;
        for &(id1, id2) in &self.merges {
            let s1 = self.reverse_vocab.get(&id1).map(|s| s.as_str()).unwrap_or("?");
            let s2 = self.reverse_vocab.get(&id2).map(|s| s.as_str()).unwrap_or("?");
            writeln!(merges_file, "{} {}", s1, s2)?;
        }

        info!(
            "Saved to {}: {} vocab, {} merges",
            path.display(),
            self.vocab.len(),
            self.merges.len(),
        );
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let config_json = fs::read_to_string(path.join("config.json"))?;
        let config: BpeConfig = serde_json::from_str(&config_json)?;
        let mut t = Self::new(config);

        let vocab_file = BufReader::new(fs::File::open(path.join("vocab.txt"))?);
        for line in vocab_file.lines() {
            let line = line?;
            if let Some((token, id_str)) = line.split_once(' ') {
                if let Ok(id) = id_str.parse::<TokenId>() {
                    t.vocab.insert(token.to_string(), id);
                    t.reverse_vocab.insert(id, token.to_string());
                    if id >= t.next_id {
                        t.next_id = id + 1;
                    }
                }
            }
        }

        let merges_path = path.join("merges.txt");
        if merges_path.exists() {
            let mfile = BufReader::new(fs::File::open(merges_path)?);
            for line in mfile.lines() {
                let line = line?;
                if let Some((t1, t2)) = line.split_once(' ') {
                    let id1 = t.vocab.get(t1).copied().unwrap_or(0);
                    let id2 = t.vocab.get(t2).copied().unwrap_or(0);
                    t.merges.push((id1, id2));
                    t.merge_ranks.insert((id1, id2), t.merge_ranks.len() as u32);
                }
            }
        }

        info!(
            "Loaded from {}: {} vocab, {} merges",
            path.display(),
            t.vocab.len(),
            t.merges.len(),
        );
        Ok(t)
    }

    pub fn get_stats(&self) -> TokenizerStats {
        let max_len = self.vocab.keys().map(|t| t.len()).max().unwrap_or(0);
        TokenizerStats {
            vocab_size: self.vocab.len(),
            merge_count: self.merges.len(),
            special_tokens_count: self.config.special_tokens.len(),
            max_token_length: max_len,
        }
    }

    pub fn add_word(&mut self, word: &str, _frequency: u32) -> Result<(), Box<dyn std::error::Error>> {
        if self.vocab.len() >= self.config.vocab_size {
            return Err("Vocabulary size limit reached".into());
        }
        if !self.vocab.contains_key(word) {
            let id = self.next_id;
            self.next_id += 1;
            self.vocab.insert(word.to_string(), id);
            self.reverse_vocab.insert(id, word.to_string());
        }
        Ok(())
    }

    pub fn unknown_token(&self) -> &str { &self.config.unknown_token }
    pub fn pad_token(&self) -> &str { &self.config.pad_token }
    pub fn bos_token(&self) -> &str { &self.config.bos_token }
    pub fn eos_token(&self) -> &str { &self.config.eos_token }
    pub fn space_sentinel(&self) -> &str { &self.config.space_sentinel }
    pub fn merges(&self) -> &[Pair] { &self.merges }
    pub fn merge_ranks(&self) -> &FxHashMap<Pair, u32> { &self.merge_ranks }
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenizerStats {
    pub vocab_size: usize,
    pub merge_count: usize,
    pub special_tokens_count: usize,
    pub max_token_length: usize,
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new(BpeConfig::default())
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(BpeTokenizer::encode(self, text))
    }
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        Ok(BpeTokenizer::decode(self, ids))
    }
    fn vocab_size(&self) -> usize {
        BpeTokenizer::vocab_size(self)
    }
}

fn fx_hash(tokens: &[TokenId]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    tokens.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BpeConfig {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<unk>".to_string(), 0);
        special_tokens.insert("<bos>".to_string(), 1);
        special_tokens.insert("<eos>".to_string(), 2);
        BpeConfig {
            vocab_size: 300,
            special_tokens,
            min_frequency: 1,
            unknown_token: "<unk>".to_string(),
            pad_token: "<pad>".to_string(),
            bos_token: "<bos>".to_string(),
            eos_token: "<eos>".to_string(),
            space_sentinel: SPACE_SENTINEL.to_string(),
        }
    }

    #[test]
    fn test_byte_vocab() {
        let t = BpeTokenizer::new(test_config());
        assert!(t.vocab_size() > 256);
        assert!(t.token_to_id("<unk>").is_some());
        assert!(t.token_to_id("<EOT>").is_none());
    }

    #[test]
    fn test_basic() {
        let mut t = BpeTokenizer::new(test_config());
        t.train("hello world hello there world wide web").unwrap();
        let ids = t.encode("hello world");
        assert!(!ids.is_empty());
        let dec = t.decode(&ids);
        assert!(!dec.is_empty());
        assert!(t.get_stats().vocab_size > 0);
    }

    #[test]
    fn test_unicode() {
        let mut t = BpeTokenizer::new(test_config());
        t.train("hello café naïve wéèb à propos").unwrap();
        let src = "café wéèb";
        let ids = t.encode(src);
        let dec = t.decode(&ids);
        assert_eq!(src, dec, "Unicode: '{}' != '{}'", src, dec);
    }

    #[test]
    fn test_emoji() {
        let mut t = BpeTokenizer::new(test_config());
        t.train("hello 🎉 🌍 🚀 world").unwrap();
        let src = "🎉 🌍";
        let ids = t.encode(src);
        let dec = t.decode(&ids);
        assert_eq!(src, dec, "Emoji: '{}' != '{}'", src, dec);
    }

    #[test]
    fn test_japanese() {
        let mut t = BpeTokenizer::new(test_config());
        t.train("こんにちは世界 今日はいい天気").unwrap();
        let src = "こんにちは世界";
        let ids = t.encode(src);
        let dec = t.decode(&ids);
        assert_eq!(src, dec, "Japanese: '{}' != '{}'", src, dec);
    }

    #[test]
    fn test_arabic() {
        let mut t = BpeTokenizer::new(test_config());
        t.train("مرحبا بالعالم اليوم جميل").unwrap();
        let src = "مرحبا بالعالم";
        let ids = t.encode(src);
        let dec = t.decode(&ids);
        assert_eq!(src, dec, "Arabic: '{}' != '{}'", src, dec);
    }

    #[test]
    fn test_space() {
        let mut t = BpeTokenizer::new(test_config());
        t.train("hello world foo bar baz qux").unwrap();
        let src = "hello world";
        let ids = t.encode(src);
        let dec = t.decode(&ids);
        assert_eq!(src, dec, "Space: '{}' != '{}'", src, dec);
    }

    #[test]
    fn test_multi_space() {
        let mut t = BpeTokenizer::new(test_config());
        t.train("hello   world   foo").unwrap();
        let src = "hello   world";
        let ids = t.encode(src);
        let dec = t.decode(&ids);
        assert_eq!(src, dec, "Multi-space: '{}' != '{}'", src, dec);
    }

    #[test]
    fn test_roundtrip() {
        let mut t = BpeTokenizer::new(test_config());
        t.train("the quick brown fox jumps over the lazy dog").unwrap();
        let cases = vec![
            "the quick brown fox",
            "hello world",
            "a b c d e f g",
            "test",
        ];
        for src in &cases {
            let ids = t.encode(src);
            let dec = t.decode(&ids);
            assert_eq!(src, &dec, "Roundtrip: '{}' != '{}'", src, dec);
        }
    }

    #[test]
    fn test_save_load() {
        let mut t = BpeTokenizer::new(test_config());
        t.train("save and load roundtrip test data").unwrap();
        let dir = std::env::temp_dir().join("test_bpe_saveload");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap_or(());
        t.save(&dir).unwrap();
        let loaded = BpeTokenizer::load(&dir).unwrap();
        assert_eq!(t.vocab_size(), loaded.vocab_size());
        assert_eq!(t.merges.len(), loaded.merges.len());
        let src = "test data roundtrip";
        assert_eq!(t.encode(src), loaded.encode(src));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_deterministic_save() {
        let mut t1 = BpeTokenizer::new(test_config());
        let mut t2 = BpeTokenizer::new(test_config());
        let corpus = "deterministic save output identical";
        t1.train(corpus).unwrap();
        t2.train(corpus).unwrap();
        let d1 = std::env::temp_dir().join("test_det_1");
        let d2 = std::env::temp_dir().join("test_det_2");
        let _ = std::fs::remove_dir_all(&d1);
        let _ = std::fs::remove_dir_all(&d2);
        std::fs::create_dir_all(&d1).unwrap_or(());
        std::fs::create_dir_all(&d2).unwrap_or(());
        t1.save(&d1).unwrap();
        t2.save(&d2).unwrap();
        let v1 = std::fs::read_to_string(d1.join("vocab.txt")).unwrap_or_default();
        let v2 = std::fs::read_to_string(d2.join("vocab.txt")).unwrap_or_default();
        assert_eq!(v1, v2, "Deterministic vocab save failed");
        let _ = std::fs::remove_dir_all(&d1);
        let _ = std::fs::remove_dir_all(&d2);
    }

    #[test]
    fn test_priority_queue() {
        let mut heap: BinaryHeap<PairFreq> = BinaryHeap::new();
        heap.push(PairFreq { pair: (1, 2), freq: 5 });
        heap.push(PairFreq { pair: (3, 4), freq: 10 });
        heap.push(PairFreq { pair: (5, 6), freq: 3 });
        assert_eq!(heap.pop().unwrap().freq, 10);
        assert_eq!(heap.pop().unwrap().freq, 5);
        assert_eq!(heap.pop().unwrap().freq, 3);
    }

    #[test]
    fn test_large_corpus() {
        let mut t = BpeTokenizer::new(test_config());
        let corpus = (0..1000)
            .map(|i| format!("word_{} data_{} test_{}", i, i % 50, i % 30))
            .collect::<Vec<_>>()
            .join(" ");
        t.train(&corpus).unwrap();
        assert!(t.vocab_size() > 260);
        let src = "word_42 test_7";
        let ids = t.encode(src);
        let dec = t.decode(&ids);
        assert_eq!(src, dec);
    }
}
