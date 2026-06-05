//! Context Module - Context management, evolution, pruning, and retrieval
//!
//! Real implementation using:
//! - In-memory context store with Arc<RwLock<…>>
//! - Bag-of-words TF-IDF for semantic relevance scoring
//! - Importance-based LRU eviction
//! - evolve_context: inserts new info as weighted entry; re-scores existing entries
//! - prune_context: removes entries below importance threshold
//! - retrieve_relevant: TF-IDF cosine similarity without external embedding backend

use async_trait::async_trait;
use nexora_foundation_types::{FoundationError, FoundationResult};
use std::collections::HashMap;
use uuid::Uuid;

/// Context window for tracking conversation/task state
#[derive(Debug, Clone)]
pub struct ContextWindow {
    pub id: Uuid,
    pub entries: Vec<ContextEntry>,
    pub metadata: ContextMetadata,
    pub max_size: usize,
}

#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub id: Uuid,
    pub content: String,
    pub entry_type: ContextType,
    pub importance: f32,
    pub timestamp: i64,
    pub embeddings: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub enum ContextType {
    UserInput,
    SystemOutput,
    InternalThought,
    ExternalInfo,
    MemoryRetrieval,
}

#[derive(Debug, Clone)]
pub struct ContextMetadata {
    pub created_at: i64,
    pub updated_at: i64,
    pub total_entries: usize,
    pub tags: Vec<String>,
}

// ─── TF-IDF helpers ───────────────────────────────────────────────────────────

/// Tokenise text into lowercase alpha tokens longer than 3 chars.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| {
            t.chars()
                .filter(|c| c.is_alphabetic())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|t| t.len() > 3)
        .collect()
}

/// Term-frequency map for a document.
fn tf(tokens: &[String]) -> HashMap<String, f32> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for t in tokens {
        *counts.entry(t.clone()).or_insert(0) += 1;
    }
    let total = tokens.len().max(1) as f32;
    counts
        .into_iter()
        .map(|(k, v)| (k, v as f32 / total))
        .collect()
}

/// Cosine similarity between two TF maps.
fn cosine_similarity(a: &HashMap<String, f32>, b: &HashMap<String, f32>) -> f32 {
    let dot: f32 = a
        .iter()
        .filter_map(|(k, va)| b.get(k).map(|vb| va * vb))
        .sum();
    let norm_a: f32 = a.values().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = b.values().map(|v| v * v).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).min(1.0)
}

/// Compute a simple importance score for an entry given a reference vocabulary.
/// importance = normalised TF-IDF: mean TF of tokens that appear in corpus_idf.
fn score_importance(content: &str, corpus_idf: &HashMap<String, f32>) -> f32 {
    let tokens = tokenize(content);
    if tokens.is_empty() || corpus_idf.is_empty() {
        return 0.5; // neutral default
    }
    let tf_map = tf(&tokens);
    let weighted: f32 = tf_map
        .iter()
        .filter_map(|(t, tf_val)| corpus_idf.get(t).map(|idf| tf_val * idf))
        .sum();
    (weighted / tf_map.len().max(1) as f32).min(1.0).max(0.0)
}

/// Build a corpus-level IDF map from all context entries.
fn build_corpus_idf(entries: &[ContextEntry]) -> HashMap<String, f32> {
    let n = entries.len() as f32;
    if n == 0.0 {
        return HashMap::new();
    }
    let mut df: HashMap<String, usize> = HashMap::new();
    for entry in entries {
        let unique_terms: std::collections::HashSet<String> =
            tokenize(&entry.content).into_iter().collect();
        for t in unique_terms {
            *df.entry(t).or_insert(0) += 1;
        }
    }
    df.into_iter()
        .map(|(t, df_val)| (t, (n / df_val as f32).ln() + 1.0))
        .collect()
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

// ─── ContextManager trait ─────────────────────────────────────────────────────

#[async_trait]
pub trait ContextManager: Send + Sync {
    async fn create_context(&self, max_size: usize) -> FoundationResult<Uuid>;
    async fn add_entry(&self, context_id: Uuid, entry: ContextEntry) -> FoundationResult<()>;
    async fn get_context(&self, context_id: Uuid) -> FoundationResult<Option<ContextWindow>>;
    async fn evolve_context(
        &self,
        context_id: Uuid,
        new_info: &str,
    ) -> FoundationResult<ContextWindow>;
    async fn prune_context(&self, context_id: Uuid, threshold: f32) -> FoundationResult<usize>;
    async fn retrieve_relevant(
        &self,
        context_id: Uuid,
        query: &str,
        limit: usize,
    ) -> FoundationResult<Vec<ContextEntry>>;
}

// ─── DefaultContextManager ────────────────────────────────────────────────────

/// In-memory context manager with real TF-IDF based retrieval.
pub struct DefaultContextManager {
    contexts: std::sync::Arc<tokio::sync::RwLock<HashMap<Uuid, ContextWindow>>>,
}

impl DefaultContextManager {
    pub fn new() -> Self {
        Self {
            contexts: std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }
}

impl Default for DefaultContextManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContextManager for DefaultContextManager {
    async fn create_context(&self, max_size: usize) -> FoundationResult<Uuid> {
        let id = Uuid::new_v4();
        let now = now_unix();
        let window = ContextWindow {
            id,
            entries: Vec::new(),
            metadata: ContextMetadata {
                created_at: now,
                updated_at: now,
                total_entries: 0,
                tags: Vec::new(),
            },
            max_size,
        };
        self.contexts.write().await.insert(id, window);
        Ok(id)
    }

    async fn add_entry(&self, context_id: Uuid, entry: ContextEntry) -> FoundationResult<()> {
        let mut contexts = self.contexts.write().await;
        let window = contexts.get_mut(&context_id).ok_or_else(|| {
            FoundationError::Implementation(format!("Context {} not found", context_id))
        })?;

        // Evict lowest-importance entry when at capacity
        if window.entries.len() >= window.max_size {
            if let Some(min_pos) = window
                .entries
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.importance
                        .partial_cmp(&b.1.importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                window.entries.remove(min_pos);
            }
        }

        window.entries.push(entry);
        window.metadata.total_entries = window.entries.len();
        window.metadata.updated_at = now_unix();
        Ok(())
    }

    async fn get_context(&self, context_id: Uuid) -> FoundationResult<Option<ContextWindow>> {
        Ok(self.contexts.read().await.get(&context_id).cloned())
    }

    /// Add `new_info` as a new `ExternalInfo` entry, then re-score all existing
    /// entries using the updated corpus IDF so importance weights reflect the
    /// full current window.
    async fn evolve_context(
        &self,
        context_id: Uuid,
        new_info: &str,
    ) -> FoundationResult<ContextWindow> {
        if new_info.trim().is_empty() {
            return Err(FoundationError::Implementation(
                "Cannot evolve context with empty information.".to_string(),
            ));
        }

        let mut contexts = self.contexts.write().await;
        let window = contexts.get_mut(&context_id).ok_or_else(|| {
            FoundationError::Implementation(format!("Context {} not found", context_id))
        })?;

        // Step 1: insert the new information entry with a neutral importance
        let new_entry = ContextEntry {
            id: Uuid::new_v4(),
            content: new_info.to_string(),
            entry_type: ContextType::ExternalInfo,
            importance: 0.5,
            timestamp: now_unix(),
            embeddings: None,
        };
        if window.entries.len() >= window.max_size {
            // Evict least important
            if let Some(min_pos) = window
                .entries
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.importance
                        .partial_cmp(&b.1.importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                window.entries.remove(min_pos);
            }
        }
        window.entries.push(new_entry);

        // Step 2: rebuild IDF from the full (updated) corpus
        let idf = build_corpus_idf(&window.entries);

        // Step 3: re-score every entry's importance using TF-IDF
        for entry in window.entries.iter_mut() {
            let recalculated = score_importance(&entry.content, &idf);
            // Blend old importance with new score to avoid sudden shifts
            entry.importance = 0.6 * recalculated + 0.4 * entry.importance;
        }

        window.metadata.total_entries = window.entries.len();
        window.metadata.updated_at = now_unix();

        Ok(window.clone())
    }

    /// Remove all entries whose `importance < threshold`. Returns number removed.
    async fn prune_context(&self, context_id: Uuid, threshold: f32) -> FoundationResult<usize> {
        let mut contexts = self.contexts.write().await;
        let window = contexts.get_mut(&context_id).ok_or_else(|| {
            FoundationError::Implementation(format!("Context {} not found", context_id))
        })?;

        let before = window.entries.len();
        window.entries.retain(|e| e.importance >= threshold);
        let removed = before - window.entries.len();

        window.metadata.total_entries = window.entries.len();
        window.metadata.updated_at = now_unix();

        Ok(removed)
    }

    /// Return up to `limit` entries sorted by TF-IDF cosine similarity to `query`.
    async fn retrieve_relevant(
        &self,
        context_id: Uuid,
        query: &str,
        limit: usize,
    ) -> FoundationResult<Vec<ContextEntry>> {
        if query.trim().is_empty() {
            return Err(FoundationError::Implementation(
                "Query cannot be empty for retrieval.".to_string(),
            ));
        }

        let contexts = self.contexts.read().await;
        let window = contexts.get(&context_id).ok_or_else(|| {
            FoundationError::Implementation(format!("Context {} not found", context_id))
        })?;

        if window.entries.is_empty() {
            return Ok(vec![]);
        }

        let query_tokens = tokenize(query);
        let query_tf = tf(&query_tokens);

        // Score each entry by cosine(query, entry)
        let mut scored: Vec<(f32, &ContextEntry)> = window
            .entries
            .iter()
            .map(|e| {
                let entry_tokens = tokenize(&e.content);
                let entry_tf = tf(&entry_tokens);
                let sim = cosine_similarity(&query_tf, &entry_tf);
                // Blend similarity with stored importance for final rank
                let rank = 0.7 * sim + 0.3 * e.importance;
                (rank, e)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let results = scored
            .into_iter()
            .take(limit)
            .map(|(_, e)| e.clone())
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn make_ctx(max: usize) -> (DefaultContextManager, Uuid) {
        let mgr = DefaultContextManager::new();
        let id = mgr.create_context(max).await.unwrap();
        (mgr, id)
    }

    fn entry(content: &str, importance: f32) -> ContextEntry {
        ContextEntry {
            id: Uuid::new_v4(),
            content: content.to_string(),
            entry_type: ContextType::UserInput,
            importance,
            timestamp: 0,
            embeddings: None,
        }
    }

    #[tokio::test]
    async fn test_create_and_get() {
        let (mgr, id) = make_ctx(10).await;
        let ctx = mgr.get_context(id).await.unwrap();
        assert!(ctx.is_some());
        assert_eq!(ctx.unwrap().max_size, 10);
    }

    #[tokio::test]
    async fn test_add_entry_evicts_on_full() {
        let (mgr, id) = make_ctx(2).await;
        mgr.add_entry(id, entry("first important message", 0.9))
            .await
            .unwrap();
        mgr.add_entry(id, entry("second less important", 0.2))
            .await
            .unwrap();
        // Third entry should evict the lowest-importance one (0.2)
        mgr.add_entry(id, entry("third new entry", 0.7))
            .await
            .unwrap();
        let ctx = mgr.get_context(id).await.unwrap().unwrap();
        assert_eq!(ctx.entries.len(), 2);
        assert!(ctx.entries.iter().any(|e| e.content.contains("first")));
    }

    #[tokio::test]
    async fn test_evolve_context() {
        let (mgr, id) = make_ctx(10).await;
        mgr.add_entry(id, entry("Rust is a systems programming language", 0.5))
            .await
            .unwrap();
        let evolved = mgr
            .evolve_context(id, "Rust has excellent memory safety guarantees")
            .await
            .unwrap();
        assert_eq!(evolved.entries.len(), 2);
        // All importances should be updated (non-zero)
        assert!(evolved.entries.iter().all(|e| e.importance > 0.0));
    }

    #[tokio::test]
    async fn test_prune_context() {
        let (mgr, id) = make_ctx(10).await;
        mgr.add_entry(id, entry("high importance entry", 0.9))
            .await
            .unwrap();
        mgr.add_entry(id, entry("low importance entry", 0.1))
            .await
            .unwrap();
        let removed = mgr.prune_context(id, 0.5).await.unwrap();
        assert_eq!(removed, 1);
        let ctx = mgr.get_context(id).await.unwrap().unwrap();
        assert_eq!(ctx.entries.len(), 1);
    }

    #[tokio::test]
    async fn test_retrieve_relevant() {
        let (mgr, id) = make_ctx(10).await;
        mgr.add_entry(id, entry("Rust memory safety borrow checker", 0.8))
            .await
            .unwrap();
        mgr.add_entry(id, entry("Python machine learning neural networks", 0.7))
            .await
            .unwrap();
        mgr.add_entry(id, entry("Rust async tokio runtime performance", 0.6))
            .await
            .unwrap();

        let results = mgr
            .retrieve_relevant(id, "Rust async performance", 2)
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        // The Rust-related entries should rank above Python
        assert!(results[0].content.contains("Rust"));
    }

    #[tokio::test]
    async fn test_retrieve_empty_query_errors() {
        let (mgr, id) = make_ctx(5).await;
        assert!(mgr.retrieve_relevant(id, "  ", 5).await.is_err());
    }
}
