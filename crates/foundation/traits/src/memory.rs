//! Memory Backend Trait
//!
//! Defines the interface for memory implementations (episodic, semantic, working)

use async_trait::async_trait;
use uuid::Uuid;
use crate::FoundationResult;

/// Memory entry
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub content: Vec<u8>,
    pub metadata: MemoryMetadata,
    pub timestamp: i64,
}

#[derive(Debug, Clone)]
pub struct MemoryMetadata {
    pub importance: f32,
    pub tags: Vec<String>,
    pub access_count: u32,
    pub last_access: i64,
}

/// Memory query
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub query: String,
    pub limit: usize,
    pub min_importance: Option<f32>,
    pub tags: Option<Vec<String>>,
}

/// Memory search result
#[derive(Debug, Clone)]
pub struct MemoryResult {
    pub entries: Vec<MemoryEntry>,
    pub total_found: usize,
}

/// Core memory backend trait
#[async_trait]
pub trait MemoryBackend: Send + Sync {
    /// Store memory entry
    async fn store(&self, entry: MemoryEntry) -> FoundationResult<Uuid>;
    
    /// Retrieve memory entry by ID
    async fn retrieve(&self, id: Uuid) -> FoundationResult<Option<MemoryEntry>>;
    
    /// Search memory
    async fn search(&self, query: MemoryQuery) -> FoundationResult<MemoryResult>;
    
    /// Update memory entry
    async fn update(&self, id: Uuid, entry: MemoryEntry) -> FoundationResult<bool>;
    
    /// Delete memory entry
    async fn delete(&self, id: Uuid) -> FoundationResult<bool>;
    
    /// Get memory statistics
    async fn stats(&self) -> FoundationResult<MemoryStats>;
    
    /// Clear old/low-importance memories
    async fn cleanup(&self, threshold: f32) -> FoundationResult<usize>;
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub total_size_bytes: u64,
    pub avg_importance: f32,
}
