//! Data Deduplicator - Rust implementation
//! 
//! Advanced deduplication using MinHash LSH and similarity detection

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use crate::DataEntry;

/// Data deduplicator with multiple algorithms
#[derive(Debug, Clone)]
pub struct DataDeduplicator {
    algorithm: DeduplicationAlgorithm,
    pub similarity_threshold: f32,
    minhash_lsh: Option<MinHashLSH>,
    exact_hashes: HashMap<u64, String>,
}

/// Deduplication algorithms
#[derive(Debug, Clone)]
pub enum DeduplicationAlgorithm {
    ExactHash,
    MinHashLSH { num_hashes: usize, signature_size: usize },
    JaccardSimilarity { ngram_size: usize },
    CosineSimilarity,
    SemanticSimilarity,
}

/// MinHash LSH implementation for similarity detection
#[derive(Debug, Clone)]
pub struct MinHashLSH {
    num_hashes: usize,
    signature_size: usize,
    document_signatures: HashMap<String, Vec<u64>>,
    lsh_buckets: HashMap<Vec<u64>, Vec<String>>,
}

impl MinHashLSH {
    /// Create new MinHash LSH instance
    pub fn new(num_hashes: usize, signature_size: usize) -> Self {
        Self {
            num_hashes,
            signature_size,
            document_signatures: HashMap::new(),
            lsh_buckets: HashMap::new(),
        }
    }
    
    /// Create MinHash signature from document
    pub fn create_signature(&self, content: &str) -> Vec<u64> {
        let shingles = self.create_shingles(content, 3);
        let mut signature = Vec::with_capacity(self.num_hashes);
        
        for i in 0..self.num_hashes {
            let mut min_hash = u64::MAX;
            let seed = i as u64;
            
            for shingle in &shingles {
                let hash = self.hash_djb2(shingle, seed);
                if hash < min_hash {
                    min_hash = hash;
                }
            }
            
            signature.push(min_hash);
        }
        
        signature
    }
    
    /// Create n-gram shingles from content
    fn create_shingles(&self, content: &str, ngram_size: usize) -> Vec<String> {
        let chars: Vec<char> = content.chars().collect();
        let mut shingles = Vec::new();
        
        if chars.len() < ngram_size {
            shingles.push(content.to_string());
            return shingles;
        }
        
        for i in 0..=chars.len() - ngram_size {
            let shingle: String = chars[i..i + ngram_size].iter().collect();
            shingles.push(shingle);
        }
        
        shingles
    }
    
    /// DJB2 hash function
    fn hash_djb2(&self, text: &str, seed: u64) -> u64 {
        let mut hash = 5381u64.wrapping_add(seed);
        for c in text.chars() {
            hash = hash.wrapping_mul(33).wrapping_add(c as u64);
        }
        hash
    }
    
    /// Add document to LSH index
    pub fn add_document(&mut self, doc_id: String, signature: Vec<u64>) {
        // Create LSH bands
        let bands = self.create_lsh_bands(&signature);
        
        for band in bands {
            self.lsh_buckets
                .entry(band)
                .or_insert_with(Vec::new)
                .push(doc_id.clone());
        }
        
        self.document_signatures.insert(doc_id, signature);
    }
    
    /// Create LSH bands from signature
    fn create_lsh_bands(&self, signature: &[u64]) -> Vec<Vec<u64>> {
        let bands_per_row = self.num_hashes / self.signature_size;
        let mut bands = Vec::new();
        
        for i in 0..self.signature_size {
            let start = i * bands_per_row;
            let end = std::cmp::min(start + bands_per_row, signature.len());
            let band = signature[start..end].to_vec();
            bands.push(band);
        }
        
        bands
    }
    
    /// Find similar documents
    pub fn find_similar(&self, doc_id: &str, signature: &[u64]) -> Vec<String> {
        let mut similar_docs = HashSet::new();
        let bands = self.create_lsh_bands(signature);
        
        for band in bands {
            if let Some(candidates) = self.lsh_buckets.get(&band) {
                for candidate in candidates {
                    if candidate != doc_id {
                        similar_docs.insert(candidate.clone());
                    }
                }
            }
        }
        
        similar_docs.into_iter().collect()
    }
    
    /// Estimate Jaccard similarity from MinHash signatures
    pub fn estimate_similarity(&self, sig1: &[u64], sig2: &[u64]) -> f32 {
        if sig1.len() != sig2.len() {
            return 0.0;
        }
        
        let matches = sig1.iter().zip(sig2.iter())
            .filter(|(a, b)| a == b)
            .count();
        
        matches as f32 / sig1.len() as f32
    }
}

impl DataDeduplicator {
    /// Create new deduplicator
    pub fn new(algorithm: DeduplicationAlgorithm, similarity_threshold: f32) -> Self {
        let minhash_lsh = match &algorithm {
            DeduplicationAlgorithm::MinHashLSH { num_hashes, signature_size } => {
                Some(MinHashLSH::new(*num_hashes, *signature_size))
            }
            _ => None,
        };
        
        Self {
            algorithm,
            similarity_threshold,
            minhash_lsh,
            exact_hashes: HashMap::new(),
        }
    }
    
    /// Check if entry is duplicate
    pub fn is_duplicate(&mut self, entry: &DataEntry) -> Result<bool> {
        match &self.algorithm {
            DeduplicationAlgorithm::ExactHash => {
                let hash = self.compute_content_hash(&entry.content);
                if self.exact_hashes.contains_key(&hash) {
                    Ok(true)
                } else {
                    self.exact_hashes.insert(hash, entry.content.clone());
                    Ok(false)
                }
            }
            DeduplicationAlgorithm::MinHashLSH { .. } => {
                if let Some(lsh) = &mut self.minhash_lsh {
                    let signature = lsh.create_signature(&entry.content);
                    let similar = lsh.find_similar(&entry.id, &signature);
                    
                    for similar_id in similar {
                        if let Some(existing_sig) = lsh.document_signatures.get(&similar_id) {
                            let similarity = lsh.estimate_similarity(&signature, existing_sig);
                            if similarity >= self.similarity_threshold {
                                return Ok(true);
                            }
                        }
                    }
                    
                    lsh.add_document(entry.id.clone(), signature);
                    Ok(false)
                } else {
                    Err(anyhow::anyhow!("MinHash LSH not initialized"))
                }
            }
            DeduplicationAlgorithm::JaccardSimilarity { ngram_size } => {
                // Implement Jaccard similarity
                self.is_duplicate_jaccard(entry, *ngram_size)
            }
            DeduplicationAlgorithm::CosineSimilarity => {
                // Implement cosine similarity
                self.is_duplicate_cosine(entry)
            }
            DeduplicationAlgorithm::SemanticSimilarity => {
                // Implement semantic similarity using word embeddings and context
                self.is_duplicate_semantic(entry)
            }
        }
    }
    
    /// Check duplicate using exact hash
    fn compute_content_hash(&self, content: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Check duplicate using Jaccard similarity
    fn is_duplicate_jaccard(&self, entry: &DataEntry, ngram_size: usize) -> Result<bool> {
        // For simplicity, we'll use a basic implementation
        // In practice, you'd compare against existing entries
        let current_shingles = self.create_jaccard_shingles(&entry.content, ngram_size);
        
        // This is a simplified check - in practice, you'd maintain a shingle index
        for _existing_id in self.exact_hashes.keys() {
            // Compare with existing entries (simplified)
            if current_shingles.len() > 0 {
                let similarity = 0.5; // Placeholder calculation
                if similarity >= self.similarity_threshold {
                    return Ok(true);
                }
            }
        }
        
        Ok(false)
    }
    
    /// Create Jaccard shingles
    fn create_jaccard_shingles(&self, content: &str, ngram_size: usize) -> HashSet<String> {
        let chars: Vec<char> = content.chars().collect();
        let mut shingles = HashSet::new();
        
        if chars.len() < ngram_size {
            shingles.insert(content.to_string());
            return shingles;
        }
        
        for i in 0..=chars.len() - ngram_size {
            let shingle: String = chars[i..i + ngram_size].iter().collect();
            shingles.insert(shingle);
        }
        
        shingles
    }
    
    /// Check duplicate using cosine similarity
    fn is_duplicate_cosine(&self, entry: &DataEntry) -> Result<bool> {
        // Create TF-IDF vector for the entry
        let current_vector = self.create_tfidf_vector(&entry.content);
        
        // Compare with existing entries
        for (_existing_id, existing_content) in &self.exact_hashes {
            let existing_vector = self.create_tfidf_vector_from_content(existing_content);
            let similarity = self.cosine_similarity(&current_vector, &existing_vector);
            if similarity >= self.similarity_threshold {
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// Create TF-IDF vector from content
    fn create_tfidf_vector(&self, content: &str) -> Vec<f32> {
        let words: Vec<String> = content
            .to_lowercase()
            .split_whitespace()
            .map(|word| word.chars().filter(|c| c.is_alphanumeric()).collect())
            .filter(|word: &String| !word.is_empty())
            .collect();
        
        if words.is_empty() {
            return Vec::new();
        }
        
        // Count word frequencies
        let mut word_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for word in &words {
            *word_counts.entry(word.clone()).or_insert(0) += 1;
        }
        
        // Create vector based on word hash (simplified embedding)
        let mut vector = vec![0.0f32; 256]; // Fixed size vector
        
        for (word, count) in word_counts {
            let hash = self.compute_word_hash(&word);
            let index = (hash % 256) as usize;
            
            // Simple TF-IDF approximation
            let tf = count as f32 / words.len() as f32;
            let idf = 1.0; // Simplified IDF
            vector[index] += tf * idf;
        }
        
        // Normalize vector
        let norm: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
        
        vector
    }
    
    /// Create TF-IDF vector from stored content string
    fn create_tfidf_vector_from_content(&self, content: &str) -> Vec<f32> {
        self.create_tfidf_vector(content)
    }
    
    /// Compute word hash for vector indexing
    fn compute_word_hash(&self, word: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        word.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if vec1.is_empty() || vec2.is_empty() {
            return 0.0;
        }
        
        let min_len = vec1.len().min(vec2.len());
        let mut dot_product = 0.0f32;
        let mut norm1 = 0.0f32;
        let mut norm2 = 0.0f32;
        
        for i in 0..min_len {
            dot_product += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        
        norm1 = norm1.sqrt();
        norm2 = norm2.sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }
    
    /// Check duplicate using semantic similarity
    fn is_duplicate_semantic(&self, entry: &DataEntry) -> Result<bool> {
        // Create semantic embedding for the entry
        let current_embedding = self.create_semantic_embedding(&entry.content);
        
        // Compare with existing entries
        for (_existing_id, existing_content) in &self.exact_hashes {
            let existing_embedding = self.create_semantic_embedding(existing_content);
            let similarity = self.semantic_similarity(&current_embedding, &existing_embedding);
            if similarity >= self.similarity_threshold {
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// Create semantic embedding using word-level features and context
    fn create_semantic_embedding(&self, content: &str) -> SemanticEmbedding {
        let words: Vec<String> = content
            .to_lowercase()
            .split_whitespace()
            .map(|word| word.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
            .filter(|word| !word.is_empty())
            .collect();
        
        // Extract semantic features
        let word_embeddings = self.create_word_embeddings(&words);
        let syntactic_features = self.extract_syntactic_features(content);
        let semantic_features = self.extract_semantic_features(&words);
        
        // Combine into unified embedding
        let combined_vector = self.combine_features(&word_embeddings, &syntactic_features, &semantic_features);
        
        SemanticEmbedding {
            vector: combined_vector,
            word_count: words.len(),
            sentence_count: content.split('.').count(),
            complexity_score: self.calculate_complexity(content),
        }
    }
    
    /// Create simple word embeddings based on character patterns
    fn create_word_embeddings(&self, words: &[String]) -> Vec<f32> {
        let mut embeddings = vec![0.0f32; 128]; // Fixed size embedding
        
        for (i, word) in words.iter().enumerate().take(128) {
            let word_hash = self.compute_word_hash(word);
            let embedding_value = (word_hash % 1000) as f32 / 1000.0;
            embeddings[i % 128] += embedding_value;
        }
        
        // Normalize
        let norm: f32 = embeddings.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut embeddings {
                *value /= norm;
            }
        }
        
        embeddings
    }
    
    /// Extract syntactic features from text
    fn extract_syntactic_features(&self, content: &str) -> Vec<f32> {
        let sentences = content.split('.').filter(|s| !s.trim().is_empty()).count();
        let questions = content.matches('?').count();
        let exclamations = content.matches('!').count();
        let commas = content.matches(',').count();
        let total_chars = content.chars().count();
        
        vec![
            sentences as f32 / total_chars.max(1) as f32,
            questions as f32 / total_chars.max(1) as f32,
            exclamations as f32 / total_chars.max(1) as f32,
            commas as f32 / total_chars.max(1) as f32,
        ]
    }
    
    /// Extract semantic features from words
    fn extract_semantic_features(&self, words: &[String]) -> Vec<f32> {
        let mut tech_terms = 0;
        let mut action_words = 0;
        let mut descriptive_words = 0;
        
        // Simple keyword-based semantic analysis
        for word in words {
            if self.is_technical_term(word) {
                tech_terms += 1;
            } else if self.is_action_word(word) {
                action_words += 1;
            } else if self.is_descriptive_word(word) {
                descriptive_words += 1;
            }
        }
        
        let total = words.len().max(1);
        vec![
            tech_terms as f32 / total as f32,
            action_words as f32 / total as f32,
            descriptive_words as f32 / total as f32,
        ]
    }
    
    /// Check if word is a technical term
    fn is_technical_term(&self, word: &str) -> bool {
        let tech_terms = vec![
            "algorithm", "function", "variable", "class", "method", "api",
            "database", "server", "client", "network", "protocol", "interface",
            "implementation", "optimization", "performance", "security", "encryption"
        ];
        tech_terms.contains(&word)
    }
    
    /// Check if word is an action word
    fn is_action_word(&self, word: &str) -> bool {
        let action_words = vec![
            "create", "update", "delete", "get", "set", "run", "execute",
            "process", "handle", "manage", "control", "start", "stop", "build"
        ];
        action_words.contains(&word)
    }
    
    /// Check if word is descriptive
    fn is_descriptive_word(&self, word: &str) -> bool {
        let descriptive_words = vec![
            "beautiful", "efficient", "fast", "slow", "large", "small",
            "complex", "simple", "important", "critical", "essential", "optional"
        ];
        descriptive_words.contains(&word)
    }
    
    /// Combine different feature vectors into unified embedding
    fn combine_features(&self, word_embeddings: &[f32], syntactic: &[f32], semantic: &[f32]) -> Vec<f32> {
        let mut combined = Vec::with_capacity(128 + syntactic.len() + semantic.len());
        combined.extend_from_slice(word_embeddings);
        combined.extend_from_slice(syntactic);
        combined.extend_from_slice(semantic);
        
        // Pad or truncate to fixed size
        while combined.len() < 256 {
            combined.push(0.0);
        }
        combined.truncate(256);
        
        combined
    }
    
    /// Calculate text complexity
    fn calculate_complexity(&self, content: &str) -> f32 {
        let words = content.split_whitespace().count();
        let sentences = content.split('.').filter(|s| !s.trim().is_empty()).count();
        let avg_word_length: f32 = content
            .split_whitespace()
            .map(|word| word.chars().count() as f32)
            .sum::<f32>() / words.max(1) as f32;
        
        // Simple complexity score
        (avg_word_length / 10.0 + words as f32 / sentences.max(1) as f32 / 20.0).min(1.0)
    }
    
    /// Calculate semantic similarity between embeddings
    fn semantic_similarity(&self, emb1: &SemanticEmbedding, emb2: &SemanticEmbedding) -> f32 {
        // Vector similarity (70% weight)
        let vector_sim = self.cosine_similarity(&emb1.vector, &emb2.vector);
        
        // Structural similarity (20% weight)
        let length_ratio = (emb1.word_count as f32).min(emb2.word_count as f32) / 
                          (emb1.word_count as f32).max(emb2.word_count as f32);
        let sentence_sim = 1.0 - (emb1.sentence_count as f32 - emb2.sentence_count as f32).abs() / 
                          (emb1.sentence_count + emb2.sentence_count).max(1) as f32;
        let structural_sim = (length_ratio + sentence_sim) / 2.0;
        
        // Complexity similarity (10% weight)
        let complexity_sim = 1.0 - (emb1.complexity_score - emb2.complexity_score).abs();
        
        // Weighted combination
        vector_sim * 0.7 + structural_sim * 0.2 + complexity_sim * 0.1
    }
}

/// Semantic embedding structure
#[derive(Debug, Clone)]
struct SemanticEmbedding {
    vector: Vec<f32>,
    word_count: usize,
    sentence_count: usize,
    complexity_score: f32,
}

impl DataDeduplicator {
    /// Remove duplicate entries from a collection
    pub fn deduplicate_entries(&mut self, entries: &mut Vec<DataEntry>) -> Result<usize> {
        let initial_len = entries.len();
        let mut unique_entries = Vec::new();
        
        for entry in entries.iter() {
            if !self.is_duplicate(entry)? {
                unique_entries.push(entry.clone());
            }
        }
        
        *entries = unique_entries;
        Ok(initial_len - entries.len())
    }
    
    /// Get deduplication statistics
    pub fn get_statistics(&self) -> DeduplicationStatistics {
        DeduplicationStatistics {
            algorithm: format!("{:?}", self.algorithm),
            similarity_threshold: self.similarity_threshold,
            exact_hashes_count: self.exact_hashes.len(),
            minhash_documents: self.minhash_lsh
                .as_ref()
                .map(|lsh| lsh.document_signatures.len())
                .unwrap_or(0),
        }
    }
}

/// Deduplication statistics
#[derive(Debug, Clone)]
pub struct DeduplicationStatistics {
    pub algorithm: String,
    pub similarity_threshold: f32,
    pub exact_hashes_count: usize,
    pub minhash_documents: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_hash_deduplication() {
        let mut deduplicator = DataDeduplicator::new(
            DeduplicationAlgorithm::ExactHash,
            0.0
        );
        
        let entry1 = DataEntry::new("Test content".to_string());
        let entry2 = DataEntry::new("Test content".to_string());
        let entry3 = DataEntry::new("Different content".to_string());
        
        assert!(!deduplicator.is_duplicate(&entry1).unwrap());
        assert!(deduplicator.is_duplicate(&entry2).unwrap());
        assert!(!deduplicator.is_duplicate(&entry3).unwrap());
    }

    #[test]
    fn test_minhash_lsh() {
        let lsh = MinHashLSH::new(100, 10);
        let signature1 = lsh.create_signature("test document one");
        let signature2 = lsh.create_signature("test document two");
        let signature3 = lsh.create_signature("completely different");
        
        let sim12 = lsh.estimate_similarity(&signature1, &signature2);
        let sim13 = lsh.estimate_similarity(&signature1, &signature3);
        
        assert!(sim12 > sim13);
    }
}
