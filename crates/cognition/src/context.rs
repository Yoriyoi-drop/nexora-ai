//! Context Module - Context management and evolution

use async_trait::async_trait;
use uuid::Uuid;
use std::collections::HashMap;
use nexora_foundation::{FoundationError, FoundationResult};

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

/// Context manager trait
#[async_trait]
pub trait ContextManager: Send + Sync {
    async fn create_context(&self, max_size: usize) -> FoundationResult<Uuid>;
    async fn add_entry(&self, context_id: Uuid, entry: ContextEntry) -> FoundationResult<()>;
    async fn get_context(&self, context_id: Uuid) -> FoundationResult<Option<ContextWindow>>;
    async fn evolve_context(&self, context_id: Uuid, new_info: &str) -> FoundationResult<ContextWindow>;
    async fn prune_context(&self, context_id: Uuid, threshold: f32) -> FoundationResult<usize>;
    async fn retrieve_relevant(&self, context_id: Uuid, query: &str, limit: usize) -> FoundationResult<Vec<ContextEntry>>;
}

/// Default context manager implementation
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
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

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

        let mut contexts = self.contexts.write().await;
        contexts.insert(id, window);
        Ok(id)
    }

    async fn add_entry(&self, context_id: Uuid, entry: ContextEntry) -> FoundationResult<()> {
        let mut contexts = self.contexts.write().await;
        let window = contexts.get_mut(&context_id)
            .ok_or_else(|| FoundationError::Implementation(
                format!("Context {} not found", context_id),
            ))?;

        if window.entries.len() >= window.max_size {
            window.entries.sort_by(|a, b| a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal));
            window.entries.pop();
        }

        window.entries.push(entry);
        window.metadata.total_entries = window.entries.len();
        window.metadata.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        Ok(())
    }

    async fn get_context(&self, context_id: Uuid) -> FoundationResult<Option<ContextWindow>> {
        let contexts = self.contexts.read().await;
        Ok(contexts.get(&context_id).cloned())
    }

    async fn evolve_context(&self, context_id: Uuid, new_info: &str) -> FoundationResult<ContextWindow> {
        let mut contexts = self.contexts.write().await;
        let window = contexts.get_mut(&context_id)
            .ok_or_else(|| FoundationError::Implementation(
                format!("Context {} not found", context_id),
            ))?;

        let entry = ContextEntry {
            id: Uuid::new_v4(),
            content: new_info.to_string(),
            entry_type: ContextType::ExternalInfo,
            importance: 0.5,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            embeddings: None,
        };

        if window.entries.len() >= window.max_size {
            window.entries.sort_by(|a, b| a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal));
            window.entries.pop();
        }

        window.entries.push(entry);
        window.metadata.total_entries = window.entries.len();
        window.metadata.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        Ok(window.clone())
    }

    async fn prune_context(&self, context_id: Uuid, threshold: f32) -> FoundationResult<usize> {
        let mut contexts = self.contexts.write().await;
        let window = contexts.get_mut(&context_id)
            .ok_or_else(|| FoundationError::Implementation(
                format!("Context {} not found", context_id),
            ))?;

        let before = window.entries.len();
        window.entries.retain(|e| e.importance >= threshold);
        let pruned = before - window.entries.len();
        window.metadata.total_entries = window.entries.len();
        window.metadata.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        Ok(pruned)
    }

    async fn retrieve_relevant(&self, context_id: Uuid, query: &str, limit: usize) -> FoundationResult<Vec<ContextEntry>> {
        let contexts = self.contexts.read().await;
        let window = contexts.get(&context_id)
            .ok_or_else(|| FoundationError::Implementation(
                format!("Context {} not found", context_id),
            ))?;

        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(f32, &ContextEntry)> = window.entries.iter()
            .map(|e| {
                let content_lower = e.content.to_lowercase();
                let overlap = query_words.iter()
                    .filter(|w| content_lower.contains(*w))
                    .count();
                let relevance = overlap as f32 / query_words.len().max(1) as f32;
                let score = relevance * 0.7 + e.importance * 0.3;
                (score, e)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(limit).map(|(_, e)| e.clone()).collect())
    }
}
