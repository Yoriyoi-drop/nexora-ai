//! Context Module - Context management and evolution

use async_trait::async_trait;
use uuid::Uuid;
use std::collections::HashMap;
use nexora_foundation_traits::FoundationResult;

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
    /// Create a new context window
    async fn create_context(&self, max_size: usize) -> FoundationResult<Uuid>;
    
    /// Add entry to context
    async fn add_entry(&self, context_id: Uuid, entry: ContextEntry) -> FoundationResult<()>;
    
    /// Retrieve context
    async fn get_context(&self, context_id: Uuid) -> FoundationResult<Option<ContextWindow>>;
    
    /// Evolve context based on new information
    async fn evolve_context(&self, context_id: Uuid, new_info: &str) -> FoundationResult<ContextWindow>;
    
    /// Prune low-importance entries
    async fn prune_context(&self, context_id: Uuid, threshold: f32) -> FoundationResult<usize>;
    
    /// Get relevant context for query
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
