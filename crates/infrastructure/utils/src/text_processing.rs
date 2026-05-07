//! Text processing utilities untuk Nexora

use regex::Regex;
use anyhow::Result;
use std::collections::HashMap;

pub struct TextProcessor;

impl TextProcessor {
    /// Preprocess text untuk AI processing
    pub fn preprocess(text: &str) -> Result<String> {
        let processed = text
            .trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>();
        
        let normalized = Self::normalize_whitespace(&processed);
        Ok(normalized)
    }
    
    /// Normalize whitespace
    pub fn normalize_whitespace(text: &str) -> String {
        let re = Regex::new(r"\s+").unwrap();
        re.replace_all(text.trim(), " ").to_string()
    }
    
    /// Remove punctuation
    pub fn remove_punctuation(text: &str) -> String {
        let re = Regex::new(r"[^\w\s]").unwrap();
        re.replace_all(text, "").to_string()
    }
    
    /// Remove numbers
    pub fn remove_numbers(text: &str) -> String {
        let re = Regex::new(r"\d+").unwrap();
        re.replace_all(text, "").to_string()
    }
    
    /// Remove stopwords
    pub fn remove_stopwords(text: &str, language: &str) -> String {
        let stopwords = Self::get_stopwords(language);
        let words: Vec<&str> = text.split_whitespace().collect();
        let filtered: Vec<&str> = words.into_iter()
            .filter(|word| !stopwords.contains(&word.to_lowercase()))
            .collect();
        filtered.join(" ")
    }
    
    /// Get stopwords for different languages
    fn get_stopwords(language: &str) -> Vec<String> {
        match language.to_lowercase().as_str() {
            "en" => vec![
                "a".to_string(), "an".to_string(), "and".to_string(), "are".to_string(), "as".to_string(), "at".to_string(), "be".to_string(), "but".to_string(), "by".to_string(), "for".to_string(), "if".to_string(), "in".to_string(), "into".to_string(), "is".to_string(), "it".to_string(),
                "no".to_string(), "not".to_string(), "of".to_string(), "on".to_string(), "or".to_string(), "such".to_string(), "that".to_string(), "the".to_string(), "their".to_string(), "then".to_string(), "there".to_string(), "these".to_string(),
                "they".to_string(), "this".to_string(), "to".to_string(), "was".to_string(), "will".to_string(), "with".to_string(), "the".to_string(), "and".to_string(), "is".to_string(), "in".to_string(), "at".to_string(), "of".to_string(), "a".to_string(), "an".to_string()
            ],
            "id" => vec![
                "yang".to_string(), "di".to_string(), "ke".to_string(), "dari".to_string(), "pada".to_string(), "untuk".to_string(), "dengan".to_string(), "adalah".to_string(), "itu".to_string(), "ini".to_string(), "dan".to_string(),
                "atau".to_string(), "tapi".to_string(), "jika".to_string(), "karena".to_string(), "seperti".to_string(), "juga".to_string(), "sudah".to_string(), "belum".to_string(), "akan".to_string(), "bisa".to_string(),
                "dapat".to_string(), "oleh".to_string(), "terhadap".to_string(), "antara".to_string(), "dalam".to_string(), "tanpa".to_string(), "hanya".to_string(), "saja".to_string(), "lebih".to_string(), "paling".to_string()
            ],
            _ => vec![], // No stopwords for unsupported languages
        }
    }
    
    /// Tokenize text
    pub fn tokenize(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| word.to_lowercase())
            .collect()
    }
    
    /// Tokenize by character
    pub fn tokenize_chars(text: &str) -> Vec<String> {
        text.chars().map(|c| c.to_string()).collect()
    }
    
    /// Tokenize by sentence
    pub fn tokenize_sentences(text: &str) -> Vec<String> {
        let re = Regex::new(r"[.!?]+").unwrap();
        let sentences: Vec<&str> = re.split(text).collect();
        sentences.into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
    
    /// Extract n-grams
    pub fn extract_ngrams(text: &str, n: usize) -> Vec<String> {
        let tokens = Self::tokenize(text);
        if tokens.len() < n {
            return vec![];
        }
        
        (0..=tokens.len() - n)
            .map(|i| tokens[i..i + n].join(" "))
            .collect()
    }
    
    /// Extract character n-grams
    pub fn extract_char_ngrams(text: &str, n: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < n {
            return vec![];
        }
        
        (0..=chars.len() - n)
            .map(|i| chars[i..i + n].iter().collect())
            .collect()
    }
    
    /// Stem text (simple implementation)
    pub fn stem(text: &str, language: &str) -> String {
        let words = Self::tokenize(text);
        let stemmed: Vec<String> = words.into_iter()
            .map(|word| Self::stem_word(&word, language))
            .collect();
        stemmed.join(" ")
    }
    
    /// Simple stemming for individual word
    fn stem_word(word: &str, language: &str) -> String {
        match language.to_lowercase().as_str() {
            "en" => Self::stem_english(word),
            "id" => Self::stem_indonesian(word),
            _ => word.to_string(),
        }
    }
    
    /// Simple English stemming
    fn stem_english(word: &str) -> String {
        let mut stemmed = word.to_string();
        
        // Remove common suffixes
        let suffixes = vec!["ing", "ed", "er", "est", "ly", "s"];
        for suffix in &suffixes {
            if stemmed.ends_with(suffix) {
                stemmed = stemmed[..stemmed.len() - suffix.len()].to_string();
                break;
            }
        }
        
        stemmed
    }
    
    /// Simple Indonesian stemming
    fn stem_indonesian(word: &str) -> String {
        let mut stemmed = word.to_string();
        
        // Remove common Indonesian suffixes
        let suffixes = vec!["kan", "an", "i", "kah", "lah", "pun", "nya"];
        for suffix in &suffixes {
            if stemmed.ends_with(suffix) {
                stemmed = stemmed[..stemmed.len() - suffix.len()].to_string();
                break;
            }
        }
        
        // Remove common prefixes
        let prefixes = vec!["ber", "ter", "se", "pe", "me", "di", "ke"];
        for prefix in &prefixes {
            if stemmed.starts_with(prefix) {
                stemmed = stemmed[prefix.len()..].to_string();
                break;
            }
        }
        
        stemmed
    }
    
    /// Lemmatize text (improved implementation)
    pub fn lemmatize(text: &str, language: &str) -> String {
        // Enhanced lemmatization with common patterns
        let words = Self::tokenize(text);
        let mut lemmatized_words = Vec::new();
        
        for word in words {
            let lemma = match language.to_lowercase().as_str() {
                "en" | "english" => Self::lemmatize_english(&word),
                "id" | "indonesian" => Self::lemmatize_indonesian(&word),
                _ => word.clone(), // Return original for unsupported languages
            };
            lemmatized_words.push(lemma);
        }
        
        lemmatized_words.join(" ")
    }
    
    /// Simple English lemmatization with common patterns
    fn lemmatize_english(word: &str) -> String {
        let word_lower = word.to_lowercase();
        
        // Common English lemmatization patterns
        match word_lower.as_str() {
            // Plurals to singular
            w if w.ends_with("ies") && w.len() > 3 => {
                let base = &w[..w.len()-3];
                if base == "f" || base == "th" {
                    format!("{}ves", base)
                } else {
                    format!("{}y", base)
                }
            },
            w if w.ends_with("es") && w.len() > 2 => {
                let base = &w[..w.len()-2];
                // Handle words like "boxes" -> "box", "watches" -> "watch"
                if base.ends_with("s") || base.ends_with("sh") || base.ends_with("ch") || base.ends_with("x") || base.ends_with("z") {
                    base.to_string()
                } else {
                    word.to_string() // Keep original if pattern doesn't match
                }
            },
            w if w.ends_with('s') && w.len() > 1 && !w.ends_with("ss") => {
                w[..w.len()-1].to_string()
            },
            // Verb forms
            w if w.ends_with("ing") && w.len() > 4 => {
                let base = &w[..w.len()-3];
                if base.ends_with('e') {
                    base[..base.len()-1].to_string()
                } else {
                    base.to_string()
                }
            },
            w if w.ends_with("ed") && w.len() > 3 => {
                let base = &w[..w.len()-2];
                if base.ends_with('e') {
                    base[..base.len()-1].to_string()
                } else {
                    base.to_string()
                }
            },
            // Adjectives to adverbs
            w if w.ends_with("ly") && w.len() > 3 => {
                w[..w.len()-2].to_string()
            },
            // Comparative/Superlative
            w if w.ends_with("er") && w.len() > 3 => {
                w[..w.len()-2].to_string()
            },
            w if w.ends_with("est") && w.len() > 4 => {
                w[..w.len()-3].to_string()
            },
            _ => word.to_string()
        }
    }
    
    /// Simple Indonesian lemmatization with common patterns
    fn lemmatize_indonesian(word: &str) -> String {
        let word_lower = word.to_lowercase();
        
        // Common Indonesian affix removal
        match word_lower.as_str() {
            // Remove common prefixes
            w if w.starts_with("ber") && w.len() > 3 => w[3..].to_string(),
            w if w.starts_with("ter") && w.len() > 3 => w[3..].to_string(),
            w if w.starts_with("me") && w.len() > 2 => w[2..].to_string(),
            w if w.starts_with("mem") && w.len() > 3 => w[3..].to_string(),
            w if w.starts_with("men") && w.len() > 3 => w[3..].to_string(),
            w if w.starts_with("peng") && w.len() > 4 => w[4..].to_string(),
            w if w.starts_with("pen") && w.len() > 3 => w[3..].to_string(),
            w if w.starts_with("pe") && w.len() > 2 => w[2..].to_string(),
            // Remove common suffixes
            w if w.ends_with("kan") && w.len() > 3 => w[..w.len()-3].to_string(),
            w if w.ends_with("an") && w.len() > 3 => w[..w.len()-2].to_string(),
            w if w.ends_with("i") && w.len() > 2 => w[..w.len()-1].to_string(),
            _ => word.to_string()
        }
    }
    
    /// Extract keywords using TF-IDF (simplified)
    pub fn extract_keywords_tfidf(texts: &[&str], top_k: usize) -> Vec<(String, f64)> {
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let total_docs = texts.len();
        
        // Count word frequencies and document frequencies
        for text in texts {
            let words = Self::tokenize(text);
            let mut doc_words: std::collections::HashSet<String> = std::collections::HashSet::new();
            
            for word in &words {
                *word_freq.entry(word.clone()).or_insert(0) += 1;
                doc_words.insert(word.clone());
            }
            
            for word in doc_words {
                *doc_freq.entry(word).or_insert(0) += 1;
            }
        }
        
        // Calculate TF-IDF scores
        let mut tfidf_scores: Vec<(String, f64)> = word_freq.into_iter()
            .map(|(word, freq)| {
                let tf = freq as f64;
                let df = *doc_freq.get(&word).unwrap_or(&1) as f64;
                let idf = (total_docs as f64 / df).ln();
                let tfidf = tf * idf;
                (word, tfidf)
            })
            .collect();
        
        // Sort by TF-IDF score and take top k
        tfidf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        tfidf_scores.into_iter().take(top_k).collect()
    }
    
    /// Extract keywords using frequency
    pub fn extract_keywords_frequency(text: &str, min_length: usize, top_k: usize) -> Vec<(String, usize)> {
        let words = Self::tokenize(text);
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        
        for word in words {
            if word.len() >= min_length {
                *word_freq.entry(word).or_insert(0) += 1;
            }
        }
        
        let mut freq_vec: Vec<(String, usize)> = word_freq.into_iter().collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
        freq_vec.into_iter().take(top_k).collect()
    }
    
    /// Calculate text similarity using cosine similarity
    pub fn calculate_similarity(text1: &str, text2: &str) -> Result<f64> {
        let tokens1 = Self::tokenize(text1);
        let tokens2 = Self::tokenize(text2);
        
        let mut all_tokens: std::collections::HashSet<String> = std::collections::HashSet::new();
        all_tokens.extend(tokens1.iter().cloned());
        all_tokens.extend(tokens2.iter().cloned());
        
        let mut vec1 = Vec::new();
        let mut vec2 = Vec::new();
        
        for token in &all_tokens {
            let count1 = tokens1.iter().filter(|&t| t == token).count();
            let count2 = tokens2.iter().filter(|&t| t == token).count();
            vec1.push(count1 as f64);
            vec2.push(count2 as f64);
        }
        
        let dot_product: f64 = vec1.iter().zip(&vec2).map(|(a, b)| a * b).sum();
        let magnitude1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let magnitude2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (magnitude1 * magnitude2))
        }
    }
    
    /// Summarize text using extractive summarization
    pub fn summarize_text(text: &str, max_sentences: usize) -> String {
        let sentences = Self::tokenize_sentences(text);
        if sentences.len() <= max_sentences {
            return text.to_string();
        }
        
        // Calculate sentence scores based on word frequency
        let words = Self::tokenize(text);
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for word in words {
            *word_freq.entry(word).or_insert(0) += 1;
        }
        
        let mut sentence_scores: Vec<(usize, f64)> = sentences.iter()
            .enumerate()
            .map(|(i, sentence)| {
                let sentence_words = Self::tokenize(sentence);
                let score: f64 = sentence_words.iter()
                    .map(|word| *word_freq.get(word).unwrap_or(&0) as f64)
                    .sum();
                (i, score)
            })
            .collect();
        
        // Sort by score and take top sentences
        sentence_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_indices: Vec<usize> = sentence_scores.into_iter()
            .take(max_sentences)
            .map(|(i, _)| i)
            .collect();
        
        // Sort by original order and join
        let mut top_indices = top_indices;
        top_indices.sort();
        
        top_indices.into_iter()
            .map(|i| &sentences[i])
            .cloned()
            .collect::<Vec<String>>()
            .join(" ")
    }
    
    /// Detect language (simple heuristic)
    pub fn detect_language(text: &str) -> String {
        let words = Self::tokenize(text);
        if words.is_empty() {
            return "unknown".to_string();
        }
        
        // Simple language detection based on common words
        let en_words = ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with"];
        let id_words = ["yang", "di", "ke", "dari", "pada", "untuk", "dengan", "adalah", "itu", "ini"];
        
        let mut en_count = 0;
        let mut id_count = 0;
        
        for word in &words {
            if en_words.contains(&word.as_str()) {
                en_count += 1;
            }
            if id_words.contains(&word.as_str()) {
                id_count += 1;
            }
        }
        
        if en_count > id_count {
            "en".to_string()
        } else if id_count > en_count {
            "id".to_string()
        } else {
            "unknown".to_string()
        }
    }
    
    /// Clean text for processing
    pub fn clean_text(text: &str) -> String {
        let cleaned = text
            .chars()
            .filter(|c| c.is_ascii() && (c.is_alphanumeric() || c.is_whitespace()))
            .collect::<String>();
        Self::normalize_whitespace(&cleaned)
    }
    
    /// Extract entities (simple implementation)
    pub fn extract_entities(text: &str) -> Vec<String> {
        // This is a very simplified entity extraction
        // In production, you'd use NLP libraries
        
        let mut entities = Vec::new();
        
        // Extract capitalized words (potential proper nouns)
        let re = Regex::new(r"\b[A-Z][a-z]+\b").unwrap();
        for cap in re.find_iter(text) {
            entities.push(cap.as_str().to_string());
        }
        
        // Extract email addresses
        let email_re = Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap();
        for cap in email_re.find_iter(text) {
            entities.push(cap.as_str().to_string());
        }
        
        // Extract URLs
        let url_re = Regex::new(r"https?://[^\s]+").unwrap();
        for cap in url_re.find_iter(text) {
            entities.push(cap.as_str().to_string());
        }
        
        entities
    }
    
    /// Calculate readability score (simplified Flesch Reading Ease)
    pub fn calculate_readability(text: &str) -> f64 {
        let sentences = Self::tokenize_sentences(text);
        let words = Self::tokenize(text);
        
        if sentences.is_empty() || words.is_empty() {
            return 0.0;
        }
        
        let avg_sentence_length = words.len() as f64 / sentences.len() as f64;
        let avg_syllables = Self::count_syllables(text) as f64 / words.len() as f64;
        
        // Simplified Flesch Reading Ease formula
        206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
    }
    
    /// Count syllables (simplified)
    fn count_syllables(text: &str) -> usize {
        let words = Self::tokenize(text);
        let mut total_syllables = 0;
        
        for word in words {
            let syllable_count = Self::count_word_syllables(&word);
            total_syllables += syllable_count.max(1); // At least 1 syllable per word
        }
        
        total_syllables
    }
    
    /// Count syllables in a word (very simplified)
    fn count_word_syllables(word: &str) -> usize {
        let vowels = vec!['a', 'e', 'i', 'o', 'u', 'y'];
        let mut syllable_count = 0;
        let chars: Vec<char> = word.chars().collect();
        
        for i in 0..chars.len() {
            if vowels.contains(&chars[i]) {
                // Check if it's not part of a diphthong
                if i == 0 || !vowels.contains(&chars[i - 1]) {
                    syllable_count += 1;
                }
            }
        }
        
        // Adjust for silent 'e'
        if word.ends_with('e') && syllable_count > 1 {
            syllable_count -= 1;
        }
        
        syllable_count
    }
    
    /// Translate text (improved implementation with basic phrase mapping)
    pub fn translate(text: &str, from: &str, to: &str) -> Result<String> {
        if from == to {
            return Ok(text.to_string());
        }
        
        // Basic phrase dictionary for common translations
        let translations = Self::get_basic_translation_dict(from, to);
        
        // Simple word-by-word translation for demonstration
        let words = Self::tokenize(text);
        let mut translated_words = Vec::new();
        
        for word in words {
            let word_lower = word.to_lowercase();
            if let Some(translated) = translations.get(&word_lower) {
                translated_words.push(translated.clone());
            } else {
                // If no translation found, keep original with indicator
                translated_words.push(format!("[{}]", word));
            }
        }
        
        Ok(translated_words.join(" "))
    }
    
    /// Get basic translation dictionary for common phrases
    fn get_basic_translation_dict(from: &str, to: &str) -> std::collections::HashMap<String, String> {
        let mut dict = std::collections::HashMap::new();
        
        match (from.to_lowercase().as_str(), to.to_lowercase().as_str()) {
            ("en" | "english", "id" | "indonesian") => {
                dict.insert("hello".to_string(), "halo".to_string());
                dict.insert("world".to_string(), "dunia".to_string());
                dict.insert("thank".to_string(), "terima kasih".to_string());
                dict.insert("you".to_string(), "kamu".to_string());
                dict.insert("good".to_string(), "baik".to_string());
                dict.insert("morning".to_string(), "pagi".to_string());
                dict.insert("night".to_string(), "malam".to_string());
                dict.insert("yes".to_string(), "ya".to_string());
                dict.insert("no".to_string(), "tidak".to_string());
                dict.insert("please".to_string(), "tolong".to_string());
                dict.insert("sorry".to_string(), "maaf".to_string());
                dict.insert("help".to_string(), "bantuan".to_string());
                dict.insert("love".to_string(), "cinta".to_string());
                dict.insert("friend".to_string(), "teman".to_string());
                dict.insert("family".to_string(), "keluarga".to_string());
                dict.insert("home".to_string(), "rumah".to_string());
                dict.insert("water".to_string(), "air".to_string());
                dict.insert("food".to_string(), "makanan".to_string());
            },
            ("id" | "indonesian", "en" | "english") => {
                dict.insert("halo".to_string(), "hello".to_string());
                dict.insert("dunia".to_string(), "world".to_string());
                dict.insert("terima".to_string(), "thank".to_string());
                dict.insert("kasih".to_string(), "love".to_string());
                dict.insert("kamu".to_string(), "you".to_string());
                dict.insert("baik".to_string(), "good".to_string());
                dict.insert("pagi".to_string(), "morning".to_string());
                dict.insert("malam".to_string(), "night".to_string());
                dict.insert("ya".to_string(), "yes".to_string());
                dict.insert("tidak".to_string(), "no".to_string());
                dict.insert("tolong".to_string(), "please".to_string());
                dict.insert("maaf".to_string(), "sorry".to_string());
                dict.insert("bantuan".to_string(), "help".to_string());
                dict.insert("cinta".to_string(), "love".to_string());
                dict.insert("teman".to_string(), "friend".to_string());
                dict.insert("keluarga".to_string(), "family".to_string());
                dict.insert("rumah".to_string(), "home".to_string());
                dict.insert("air".to_string(), "water".to_string());
                dict.insert("makanan".to_string(), "food".to_string());
            },
            _ => {
                // For unsupported language pairs, return empty dict
                // In production, you'd integrate with translation APIs like Google Translate
            }
        }
        
        dict
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_processor() {
        let text = "Hello, World! This is a test text.";
        
        // Test preprocessing
        let processed = TextProcessor::preprocess(text).unwrap();
        assert_eq!(processed, "hello world this is a test text");
        
        // Test tokenization
        let tokens = TextProcessor::tokenize(text);
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test", "text"]);
        
        // Test n-grams
        let bigrams = TextProcessor::extract_ngrams(text, 2);
        assert!(bigrams.contains(&"hello world".to_string()));
        assert!(bigrams.contains(&"world this".to_string()));
        
        // Test stemming
        let stemmed = TextProcessor::stem("running dogs", "en");
        assert_eq!(stemmed, "runn dog");
        
        // Test keyword extraction
        let keywords = TextProcessor::extract_keywords_frequency(text, 3, 5);
        assert!(!keywords.is_empty());
        
        // Test similarity
        let similarity = TextProcessor::calculate_similarity("hello world", "hello there").unwrap();
        assert!(similarity > 0.0 && similarity < 1.0);
        
        // Test summarization
        let long_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four.";
        let summary = TextProcessor::summarize_text(long_text, 2);
        assert!(summary.split('.').count() <= 3); // Account for potential trailing period
        
        // Test language detection
        let language = TextProcessor::detect_language("the cat is on the table");
        assert_eq!(language, "en");
        
        let language = TextProcessor::detect_language("kucing itu di atas meja");
        assert_eq!(language, "id");
        
        // Test entity extraction
        let text_with_entities = "Contact us at info@example.com or visit https://example.com";
        let entities = TextProcessor::extract_entities(text_with_entities);
        assert!(entities.iter().any(|e| e.contains("example.com")));
        
        // Test readability
        let readability = TextProcessor::calculate_readability(text);
        assert!(readability > 0.0);
    }
}
