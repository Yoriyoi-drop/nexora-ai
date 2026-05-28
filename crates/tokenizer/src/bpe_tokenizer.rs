use crate::Tokenizer;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use tracing::{debug, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeConfig {
    pub vocab_size: usize,
    pub special_tokens: HashMap<String, u32>,
    pub min_frequency: u32,
    pub unknown_token: String,
    pub pad_token: String,
    pub bos_token: String,
    pub eos_token: String,
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    config: BpeConfig,
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    merges: Vec<(String, String)>,
    unicode_to_byte: HashMap<char, u8>,
    byte_to_unicode: HashMap<u8, char>,
}

impl BpeTokenizer {
    pub fn new(config: BpeConfig) -> Self {
        let mut tokenizer = Self {
            config,
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            merges: Vec::new(),
            unicode_to_byte: HashMap::new(),
            byte_to_unicode: HashMap::new(),
        };
        tokenizer.init_unicode_mapping();
        tokenizer
    }

    fn init_unicode_mapping(&mut self) {
        let mut unicode_to_byte = HashMap::with_capacity(256);
        let mut byte_to_unicode = HashMap::with_capacity(256);
        for byte in 0..=255u8 {
            let ch = match byte {
                0 | 32 | 127..=160 => char::from_u32(256 + byte as u32).unwrap_or('\u{FFFD}'),
                _ => char::from(byte),
            };
            unicode_to_byte.insert(ch, byte);
            byte_to_unicode.insert(byte, ch);
        }
        self.unicode_to_byte = unicode_to_byte;
        self.byte_to_unicode = byte_to_unicode;
    }

    pub fn train(&mut self, corpus: &str) -> Result<(), Box<dyn std::error::Error>> {
        let train_start = std::time::Instant::now();
        let corpus_chars = corpus.len();
        let corpus_lines = corpus.lines().count();
        let corpus_words = corpus.split_whitespace().count();

        info!(
            "Starting BPE training: vocab_size={}, corpus={} chars, {} lines, {} words",
            self.config.vocab_size, corpus_chars, corpus_lines, corpus_words,
        );

        let mut vocab_set: HashSet<String> = HashSet::with_capacity(self.config.vocab_size);
        let mut word_freqs: HashMap<Vec<String>, u32> =
            HashMap::with_capacity(self.config.vocab_size);

        for line in corpus.lines() {
            let processed = self.preprocess_line(line);
            for word in processed.split_whitespace() {
                let tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                *word_freqs.entry(tokens).or_insert(0) += 1;
                for ch in word.chars() {
                    vocab_set.insert(ch.to_string());
                }
            }
        }

        let unique_chars = vocab_set.len();
        let unique_words = word_freqs.len();

        info!(
            "Corpus stats: {} unique chars, {} unique word types, {:.2} avg chars/word",
            unique_chars,
            unique_words,
            if corpus_words > 0 { corpus_chars as f64 / corpus_words as f64 } else { 0.0 },
        );
        info!("Initial vocabulary size: {}", unique_chars);

        for (token, _) in &self.config.special_tokens {
            vocab_set.insert(token.clone());
        }

        let mut vocab: HashMap<String, u32> = vocab_set
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        let mut rev_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        let mut next_id = vocab.len() as u32;
        let mut merges = Vec::with_capacity(self.config.vocab_size);
        let merge_loop_start = std::time::Instant::now();
        let log_interval = (self.config.vocab_size / 20).max(100);

        while vocab.len() < self.config.vocab_size {
            let best = find_most_frequent_pair(&word_freqs, &rev_vocab);
            let ((s1, s2), freq) = match best {
                Some(p) => p,
                None => break,
            };

            if freq < self.config.min_frequency {
                break;
            }

            let new_token = s1.clone() + &s2;
            if vocab.contains_key(&new_token) {
                break;
            }

            vocab.insert(new_token.clone(), next_id);
            rev_vocab.insert(next_id, new_token.clone());
            next_id += 1;
            merges.push((s1.clone(), s2.clone()));

            update_word_freqs(&mut word_freqs, &s1, &s2, &new_token);

            let merge_count = merges.len();
            if merge_count % log_interval == 0 {
                let elapsed = merge_loop_start.elapsed();
                let merge_speed = merge_count as f64 / elapsed.as_secs_f64().max(0.001);
                info!(
                    "  Merge progress: {}/{} ({:.0}%), {:.0} merges/s, current pair: '{}'+'{}'",
                    merge_count,
                    self.config.vocab_size.saturating_sub(vocab_set.len()),
                    merge_count as f64 * 100.0 / self.config.vocab_size.saturating_sub(vocab_set.len()).max(1) as f64,
                    merge_speed,
                    s1, s2,
                );
            }

            debug!(
                "Added merge: {} + {} -> {} (freq: {})",
                s1, s2, new_token, freq
            );
        }

        self.vocab.clear();
        self.reverse_vocab.clear();
        for (token, &id) in &vocab {
            self.vocab.insert(token.clone(), id);
            self.reverse_vocab.insert(id, token.clone());
        }
        self.merges = merges;

        let total_time = train_start.elapsed();
        let merge_time = merge_loop_start.elapsed();
        let merge_speed = self.merges.len() as f64 / merge_time.as_secs_f64().max(0.001);
        let corpus_mb = corpus_chars as f64 / 1_048_576.0;

        info!(
            "BPE training completed in {:?} ({:.3}s)",
            total_time,
            total_time.as_secs_f64(),
        );
        info!(
            "  Corpus: {:.2} MB chars, {} unique chars → {} vocab ({} merges)",
            corpus_mb, unique_chars, self.vocab.len(), self.merges.len(),
        );
        info!(
            "  Merge loop: {:.2}s, {:.0} merges/s, {:.0} chars/s",
            merge_time.as_secs_f64(),
            merge_speed,
            corpus_chars as f64 / total_time.as_secs_f64().max(0.001),
        );
        Ok(())
    }

    fn preprocess_line(&self, line: &str) -> String {
        line.to_lowercase()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(text.len());
        if let Some(&bos_id) = self.config.special_tokens.get(&self.config.bos_token) {
            tokens.push(bos_id);
        }
        let processed = self.preprocess_line(text);
        for word in processed.split_whitespace() {
            tokens.extend(self.encode_word(word));
        }
        if let Some(&eos_id) = self.config.special_tokens.get(&self.config.eos_token) {
            tokens.push(eos_id);
        }
        tokens
    }

    fn encode_word(&self, word: &str) -> Vec<u32> {
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best_rank = u32::MAX;
            let mut best_i = None;
            for i in 0..tokens.len().saturating_sub(1) {
                let merged = tokens[i].clone() + &tokens[i + 1];
                if let Some(&id) = self.vocab.get(&merged) {
                    if id < best_rank {
                        best_rank = id;
                        best_i = Some(i);
                    }
                }
            }
            match best_i {
                Some(i) => {
                    let merged = tokens[i].clone() + &tokens[i + 1];
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                }
                None => break,
            }
        }
        let unk_id = self
            .config
            .special_tokens
            .get(&self.config.unknown_token)
            .copied()
            .unwrap_or(0);
        tokens
            .into_iter()
            .map(|t| self.vocab.get(&t).copied().unwrap_or(unk_id))
            .collect()
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut text = String::new();
        for &id in token_ids {
            if let Some(token) = self.reverse_vocab.get(&id) {
                if !self.config.special_tokens.contains_key(token) {
                    text.push_str(token);
                }
            }
        }
        text
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub fn id_to_token(&self, id: u32) -> Option<&String> {
        self.reverse_vocab.get(&id)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let config_json = serde_json::to_string_pretty(&self.config)?;
        fs::write(path.join("config.json"), config_json)?;

        let mut vocab_file = fs::File::create(path.join("vocab.txt"))?;
        for (token, &id) in &self.vocab {
            writeln!(vocab_file, "{} {}", token, id)?;
        }

        let mut merges_file = fs::File::create(path.join("merges.txt"))?;
        for (t1, t2) in &self.merges {
            writeln!(merges_file, "{} {}", t1, t2)?;
        }

        let unicode_data = (self.unicode_to_byte.clone(), self.byte_to_unicode.clone());
        let unicode_json = serde_json::to_string_pretty(&unicode_data)?;
        fs::write(path.join("unicode.json"), unicode_json)?;

        info!("Tokenizer saved to: {}", path.display());
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let config_json = fs::read_to_string(path.join("config.json"))?;
        let config: BpeConfig = serde_json::from_str(&config_json)?;
        let mut tokenizer = Self::new(config);

        let vocab_file = BufReader::new(fs::File::open(path.join("vocab.txt"))?);
        for line in vocab_file.lines() {
            let line = line?;
            if let Some((token, id_str)) = line.split_once(' ') {
                if let Ok(id) = id_str.parse::<u32>() {
                    tokenizer.vocab.insert(token.to_string(), id);
                    tokenizer.reverse_vocab.insert(id, token.to_string());
                }
            }
        }

        let merges_path = path.join("merges.txt");
        if merges_path.exists() {
            let merges_file = BufReader::new(fs::File::open(merges_path)?);
            for line in merges_file.lines() {
                let line = line?;
                if let Some((t1, t2)) = line.split_once(' ') {
                    tokenizer.merges.push((t1.to_string(), t2.to_string()));
                }
            }
        }

        let unicode_path = path.join("unicode.json");
        if unicode_path.exists() {
            let json = fs::read_to_string(unicode_path)?;
            let (u2b, b2u): (HashMap<char, u8>, HashMap<u8, char>) = serde_json::from_str(&json)?;
            tokenizer.unicode_to_byte = u2b;
            tokenizer.byte_to_unicode = b2u;
        }

        info!("Tokenizer loaded from: {}", path.display());
        Ok(tokenizer)
    }

    pub fn get_stats(&self) -> TokenizerStats {
        TokenizerStats {
            vocab_size: self.vocab.len(),
            merge_count: self.merges.len(),
            special_tokens_count: self.config.special_tokens.len(),
            max_token_length: self.vocab.keys().map(|t| t.len()).max().unwrap_or(0),
        }
    }

    pub fn add_word(
        &mut self,
        word: &str,
        _frequency: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.vocab.len() >= self.config.vocab_size {
            return Err("Vocabulary size limit reached".into());
        }
        if !self.vocab.contains_key(word) {
            let id = self.vocab.len() as u32;
            self.vocab.insert(word.to_string(), id);
            self.reverse_vocab.insert(id, word.to_string());
        }
        Ok(())
    }

    pub fn unknown_token(&self) -> &str {
        &self.config.unknown_token
    }
    pub fn pad_token(&self) -> &str {
        &self.config.pad_token
    }
    pub fn bos_token(&self) -> &str {
        &self.config.bos_token
    }
    pub fn eos_token(&self) -> &str {
        &self.config.eos_token
    }
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

fn find_most_frequent_pair(
    word_freqs: &HashMap<Vec<String>, u32>,
    rev_vocab: &HashMap<u32, String>,
) -> Option<((String, String), u32)> {
    let mut token_to_id: HashMap<&str, u32> = HashMap::with_capacity(rev_vocab.len());
    for (id, token) in rev_vocab {
        token_to_id.insert(token.as_str(), *id);
    }

    let mut pair_freqs: HashMap<(u32, u32), u32> = HashMap::new();
    for (tokens, freq) in word_freqs {
        for i in 0..tokens.len().saturating_sub(1) {
            if let (Some(&id1), Some(&id2)) = (
                token_to_id.get(tokens[i].as_str()),
                token_to_id.get(tokens[i + 1].as_str()),
            ) {
                *pair_freqs.entry((id1, id2)).or_insert(0) += freq;
            }
        }
    }

    let ((id1, id2), freq) = pair_freqs.into_iter().max_by_key(|&(_, f)| f)?;

    let s1 = rev_vocab.get(&id1)?.clone();
    let s2 = rev_vocab.get(&id2)?.clone();
    Some(((s1, s2), freq))
}

fn update_word_freqs(
    word_freqs: &mut HashMap<Vec<String>, u32>,
    s1: &str,
    s2: &str,
    new_token: &str,
) {
    let mut new_freqs: HashMap<Vec<String>, u32> = HashMap::with_capacity(word_freqs.len());
    for (tokens, freq) in word_freqs.drain() {
        let mut new_tokens = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == s1 && tokens[i + 1] == s2 {
                new_tokens.push(new_token.to_string());
                i += 2;
            } else {
                new_tokens.push(tokens[i].clone());
                i += 1;
            }
        }
        *new_freqs.entry(new_tokens).or_insert(0) += freq;
    }
    *word_freqs = new_freqs;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_tokenizer_basic() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig {
            vocab_size: 100,
            special_tokens: HashMap::new(),
            min_frequency: 1,
            unknown_token: "<unk>".to_string(),
            pad_token: "<pad>".to_string(),
            bos_token: "<bos>".to_string(),
            eos_token: "<eos>".to_string(),
        });
        let corpus = "hello world hello there world wide web";
        tokenizer.train(corpus).unwrap();
        let tokens = tokenizer.encode("hello world");
        assert!(!tokens.is_empty());
        let decoded = tokenizer.decode(&tokens);
        assert!(!decoded.is_empty());
        let stats = tokenizer.get_stats();
        assert!(stats.vocab_size > 0);
    }

    #[test]
    fn test_unicode_mapping() {
        let tokenizer = BpeTokenizer::new(BpeConfig::default());
        assert!(!tokenizer.unicode_to_byte.is_empty());
        assert!(!tokenizer.byte_to_unicode.is_empty());
        assert!(tokenizer.unicode_to_byte.contains_key(&'a'));
        assert!(tokenizer.unicode_to_byte.contains_key(&char::from_u32(288).unwrap()));
    }

    #[test]
    fn test_special_tokens() {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<unk>".to_string(), 0);
        special_tokens.insert("<pad>".to_string(), 1);
        special_tokens.insert("<bos>".to_string(), 2);
        special_tokens.insert("<eos>".to_string(), 3);
        let tokenizer = BpeTokenizer::new(BpeConfig {
            vocab_size: 1000,
            special_tokens,
            min_frequency: 1,
            unknown_token: "<unk>".to_string(),
            pad_token: "<pad>".to_string(),
            bos_token: "<bos>".to_string(),
            eos_token: "<eos>".to_string(),
        });
        let tokens = tokenizer.encode("test");
        assert!(tokens.len() >= 2);
    }

    #[test]
    fn test_save_load() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        let corpus = "test training data for tokenizer";
        tokenizer.train(corpus).unwrap();
        let temp_dir = std::env::temp_dir().join("test_tokenizer_save_load");
        std::fs::create_dir_all(&temp_dir).unwrap_or(());
        tokenizer.save(&temp_dir).unwrap();
        let loaded = BpeTokenizer::load(&temp_dir).unwrap();
        assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());
        fs::remove_dir_all(&temp_dir).unwrap_or(());
    }
}
