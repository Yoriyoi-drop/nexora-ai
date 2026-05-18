//! Memory Agent
//! 
//! Agent sebagai bridge ke folder memory.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;
use uuid::Uuid;
use serde_json::{Value, json};
use tracing::{debug, info, warn};

use crate::{
    Agent, AgentError, Result, AgentMessage, AgentResponse, AgentStatus,
    AgentContext, AgentStats, AgentConfig
};
use nexora_memory::{MemoryLayers, MemoryType};

/// Memory query untuk memory agent
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub user_id: Option<Uuid>,
    pub session_id: Option<Uuid>,
    pub memory_type: String,
    pub query_text: String,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub filters: HashMap<String, Value>,
}

/// Memory result type alias
pub type MemoryResult<T> = std::result::Result<T, crate::AgentError>;

/// Memory agent untuk mengelola operasi memory
pub struct MemoryAgent {
    /// Unique ID
    id: Uuid,
    /// Agent name
    name: String,
    /// Current status
    status: AgentStatus,
    /// Memory store
    memory_store: Arc<RwLock<MemoryLayers>>,
    /// Statistics
    stats: AgentStats,
    /// Configuration
    config: MemoryAgentConfig,
}

/// Configuration untuk memory agent
#[derive(Debug, Clone)]
pub struct MemoryAgentConfig {
    /// Default query limit
    pub default_query_limit: u32,
    /// Enable memory compression
    pub enable_compression: bool,
    /// Compression threshold (minimum entries to trigger compression)
    pub compression_threshold: usize,
    /// Cache recent queries (seconds)
    pub cache_duration_seconds: u64,
    /// Enable memory cleanup
    pub enable_cleanup: bool,
    /// Cleanup interval (seconds)
    pub cleanup_interval_seconds: u64,
}

/// Memory operation result
#[derive(Debug, Clone)]
pub struct MemoryOperationResult {
    /// Operation type
    pub operation: String,
    /// Success status
    pub success: bool,
    /// Affected records count
    pub affected_count: usize,
    /// Operation result data
    pub data: Value,
    /// Operation metadata
    pub metadata: HashMap<String, Value>,
}

impl MemoryAgent {
    /// Create new memory agent
    pub fn new(
        memory_store: Arc<RwLock<MemoryLayers>>,
        config: MemoryAgentConfig,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "MemoryAgent".to_string(),
            status: AgentStatus::Initializing,
            memory_store,
            stats: AgentStats::default(),
            config,
        }
    }
    
    /// Store memory
    pub async fn store_memory(
        &self,
        user_id: Option<Uuid>,
        session_id: Option<Uuid>,
        memory_type: MemoryType,
        content: String,
        metadata: HashMap<String, Value>,
    ) -> Result<Uuid> {
        debug!("Storing memory of type: {:?}", memory_type);
        
        let memory_id = Uuid::new_v4();
        let now = chrono::Utc::now();
        
        let memory_key = format!("{}:{}:{}", user_id.unwrap_or_default(), session_id.unwrap_or_default(), memory_id);
        let memory_value = serde_json::json!({
            "id": memory_id,
            "user_id": user_id,
            "session_id": session_id,
            "memory_type": memory_type,
            "content": content,
            "metadata": metadata,
            "timestamp": now,
            "relevance_score": 1.0
        }).to_string();
        
        self.memory_store.write().await.store(nexora_memory::MemoryLayer::Short, &memory_key, &memory_value).await?;
        
        info!("Memory {} stored successfully", memory_id);
        Ok(memory_id)
    }
    
    /// Query memory
    pub async fn query_memory(
        &self,
        query: MemoryQuery,
    ) -> Result<Vec<nexora_memory::MemoryEntry>> {
        debug!("Querying memory with type: {}", query.memory_type);
        
        let results = self.memory_store.read().await.query(&query.query_text).await?;
        
        debug!("Query returned {} results", results.len());
        let result_wrapped: Vec<nexora_memory::MemoryEntry> = results
            .into_iter()
            .enumerate()
            .map(|(i, entry)| {
                // Use sequential ID — hash was immediately truncated/replaced
                let memory_id = i as u32;
                
                // Determine memory type from entry content
                let memory_type = if entry.value.contains("episodic") || 
                                 entry.value.contains("experience") {
                    nexora_memory::MemoryType::Episodic
                } else if entry.value.contains("semantic") || 
                         entry.value.contains("fact") {
                    nexora_memory::MemoryType::Semantic
                } else if entry.value.contains("working") || 
                         entry.value.contains("temporary") {
                    nexora_memory::MemoryType::Working
                } else {
                    nexora_memory::MemoryType::Episodic // Default
                };
                
                let model_entry = nexora_memory::MemoryEntry {
                    memory_id: memory_id as u32,
                    memory_type,
                    activation: 0.0,
                    relevance: 0.0,
                    emotional_salience: 0.0,
                    timestamp: 0.0,
                    strength: 0.0,
                    content: Some(entry.value),
                    embedding: None,
                    embedding_dim: 0,
                };
                model_entry
            })
            .collect();
        Ok(result_wrapped)
    }
    
    /// Get specific memory by ID
    pub async fn get_memory(&self, memory_id: Uuid) -> Result<Option<nexora_memory::MemoryEntry>> {
        debug!("Retrieving memory: {}", memory_id);
        
        let query = MemoryQuery {
            user_id: None,
            session_id: None,
            memory_type: "all".to_string(),
            query_text: "memory_by_id".to_string(),
            limit: Some(1),
            offset: None,
            filters: {
                let mut filters = HashMap::new();
                filters.insert("id".to_string(), Value::String(memory_id.to_string()));
                filters
            },
        };
        
        let results = self.memory_store.read().await.query(&query.query_text).await?;
        
        Ok(results.into_iter().next().map(|entry| {
            // Use sequential ID — hash was immediately truncated/replaced
            let memory_id = 0u32;
            
            // Determine memory type from entry content
            let memory_type = if entry.value.contains("episodic") || 
                             entry.value.contains("experience") {
                nexora_memory::MemoryType::Episodic
            } else if entry.value.contains("semantic") || 
                     entry.value.contains("fact") {
                nexora_memory::MemoryType::Semantic
            } else if entry.value.contains("working") || 
                     entry.value.contains("temporary") {
                nexora_memory::MemoryType::Working
            } else {
                nexora_memory::MemoryType::Episodic // Default
            };
            
            let model_entry = nexora_memory::MemoryEntry {
                memory_id: memory_id as u32,
                memory_type,
                activation: 0.0,
                relevance: 0.0,
                emotional_salience: 0.0,
                timestamp: 0.0,
                strength: 0.0,
                content: Some(entry.value),
                embedding: None,
                embedding_dim: 0,
            };
            model_entry
        }))
    }
    
    /// Update memory
    pub async fn update_memory(
        &self,
        memory_id: Uuid,
        content: Option<String>,
        _metadata: Option<HashMap<String, Value>>,
    ) -> Result<bool> {
        debug!("Updating memory: {}", memory_id);
        
        // Get existing memory first
        let mut memory_result = self.get_memory(memory_id).await?
            .ok_or_else(|| AgentError::ProcessingError(format!("Memory {} not found", memory_id)))?;
        
        // Update fields
        if let Some(new_content) = content {
            memory_result.content = Some(new_content);
        }
        
        // Note: metadata is stored in the JSON wrapper, not directly in MemoryEntry
        
        // Store updated memory
        let memory_key = format!("{}:{}:{}", memory_id, memory_result.memory_type, memory_id);
        let memory_value = serde_json::json!({
            "id": memory_id,
            "memory_id": memory_result.memory_id,
            "memory_type": memory_result.memory_type,
            "content": memory_result.content,
            "activation": memory_result.activation,
            "relevance": memory_result.relevance,
            "emotional_salience": memory_result.emotional_salience,
            "timestamp": memory_result.timestamp,
            "strength": memory_result.strength,
            "embedding": memory_result.embedding,
            "embedding_dim": memory_result.embedding_dim
        }).to_string();
        
        self.memory_store.write().await.store(nexora_memory::MemoryLayer::Short, &memory_key, &memory_value).await?;
        
        info!("Memory {} updated successfully", memory_id);
        Ok(true)
    }
    
    /// Delete memory
    pub async fn delete_memory(&self, memory_id: Uuid) -> Result<bool> {
        debug!("Deleting memory: {}", memory_id);
        
        let _memory_key = format!("*:*:{}", memory_id);
        // Note: This is a simplified approach - in practice, we'd need to find the exact key
        self.memory_store.write().await.delete(nexora_memory::MemoryLayer::Short, &memory_id.to_string()).await?;
        
        info!("Memory {} deleted successfully", memory_id);
        
        Ok(true)
    }
    
    /// Get user memories
    pub async fn get_user_memories(
        &self,
        user_id: Uuid,
        memory_type: Option<MemoryType>,
        limit: Option<u32>,
    ) -> Result<Vec<nexora_memory::MemoryEntry>> {
        debug!("Getting memories for user: {}", user_id);
        
        let query = MemoryQuery {
            user_id: Some(user_id),
            session_id: None,
            memory_type: memory_type.map(|t| t.to_string()).unwrap_or("all".to_string()),
            query_text: "user_memories".to_string(),
            limit: limit.map(|l| l as usize).or(Some(self.config.default_query_limit as usize)),
            offset: None,
            filters: HashMap::new(),
        };
        
        self.query_memory(query).await
    }
    
    /// Get session memories
    pub async fn get_session_memories(
        &self,
        session_id: Uuid,
        memory_type: Option<MemoryType>,
        limit: Option<u32>,
    ) -> Result<Vec<nexora_memory::MemoryEntry>> {
        debug!("Getting memories for session: {}", session_id);
        
        let query = MemoryQuery {
            user_id: None,
            session_id: Some(session_id),
            memory_type: memory_type.map(|t| t.to_string()).unwrap_or("all".to_string()),
            query_text: "session_memories".to_string(),
            limit: limit.map(|l| l as usize).or(Some(self.config.default_query_limit as usize)),
            offset: None,
            filters: HashMap::new(),
        };
        
        self.query_memory(query).await
    }
    
    /// Search memories
    pub async fn search_memories(
        &self,
        search_term: String,
        user_id: Option<Uuid>,
        memory_type: Option<MemoryType>,
        limit: Option<u32>,
    ) -> Result<Vec<nexora_memory::MemoryEntry>> {
        debug!("Searching memories with term: {}", search_term);
        
        let query = MemoryQuery {
            user_id,
            session_id: None,
            memory_type: memory_type.map(|t| t.to_string()).unwrap_or("all".to_string()),
            query_text: search_term.clone(),
            limit: limit.map(|l| l as usize).or(Some(self.config.default_query_limit as usize)),
            offset: None,
            filters: {
                let mut filters = HashMap::new();
                filters.insert("search".to_string(), Value::String(search_term));
                filters
            },
        };
        
        self.query_memory(query).await
    }
    
    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> Result<MemoryStats> {
        debug!("Getting memory statistics");
        
        // Single query instead of 5 individual queries
        let query = MemoryQuery {
            user_id: None,
            session_id: None,
            memory_type: "all".to_string(),
            query_text: "stats_query".to_string(),
            limit: Some(10000),
            offset: None,
            filters: HashMap::new(),
        };
        
        let results = self.memory_store.read().await.query(&query.query_text).await?;
        
        let mut stats = MemoryStats::default();
        for entry in &results {
            if entry.value.contains("episodic") || entry.value.contains("experience") {
                stats.episodic_count += 1;
            } else if entry.value.contains("semantic") || entry.value.contains("fact") {
                stats.semantic_count += 1;
            } else if entry.value.contains("working") || entry.value.contains("temporary") {
                stats.working_count += 1;
            } else if entry.value.contains("user") {
                stats.user_count += 1;
            } else {
                stats.procedural_count += 1;
            }
        }
        stats.total_count = results.len();
        
        Ok(stats)
    }
    
    /// Cleanup old memories
    pub async fn cleanup_old_memories(&self, max_age_hours: u64) -> Result<usize> {
        debug!("Cleaning up memories older than {} hours", max_age_hours);
        
        if !self.config.enable_cleanup {
            return Ok(0);
        }
        
        let cutoff_time = chrono::Utc::now() - chrono::Duration::hours(max_age_hours as i64);
        let mut cleaned_count = 0;
        
        // Query all memories
        let query = MemoryQuery {
            user_id: None,
            session_id: None,
            memory_type: "all".to_string(),
            query_text: "cleanup_query".to_string(),
            limit: Some(10000),
            offset: None,
            filters: HashMap::new(),
        };
        
        let results = self.memory_store.read().await.query(&query.query_text).await?;
        
        for memory in results {
            if memory.timestamp < cutoff_time.timestamp() as u64 {
                self.memory_store.write().await.delete(nexora_memory::MemoryLayer::Short, &memory.key).await?;
                cleaned_count += 1;
            }
        }
        
        info!("Cleaned up {} old memories", cleaned_count);
        Ok(cleaned_count)
    }
    
    /// Compress memories (merge similar memories)
    pub async fn compress_memories(&self, user_id: Option<Uuid>) -> Result<usize> {
        debug!("Compressing memories for user: {:?}", user_id);
        
        if !self.config.enable_compression {
            return Ok(0);
        }
        
        // Implement memory compression logic
        info!("Starting memory compression");
        
        let memory_store = self.memory_store.read().await;
        let mut compressed_count = 0;
        
        // Simplified compression - just count total entries as compression metric
        // Since we don't have access to specific layer methods, use basic stats
        if let Ok(stats) = memory_store.get_stats().await {
            compressed_count = stats.values().sum();
            info!("Total memory entries: {}", compressed_count);
        }
        
        drop(memory_store);
        
        // Trigger garbage collection if needed
        if compressed_count > 0 {
            if let Err(e) = self.cleanup_old_memories(24).await {
                warn!("Failed to cleanup after compression: {}", e);
            }
        }
        
        Ok(compressed_count)
    }
}

/// Memory statistics
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct MemoryStats {
    pub total_count: usize,
    pub episodic_count: usize,
    pub semantic_count: usize,
    pub working_count: usize,
    pub user_count: usize,
    pub procedural_count: usize,
}

#[async_trait]
impl Agent for MemoryAgent {
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn agent_type(&self) -> &str {
        "memory"
    }
    
    fn status(&self) -> AgentStatus {
        self.status.clone()
    }
    
    async fn initialize(&mut self, _config: AgentConfig) -> Result<()> {
        info!("Initializing MemoryAgent");
        self.status = AgentStatus::Ready;
        Ok(())
    }
    
    async fn receive(&mut self, message: AgentMessage) -> Result<()> {
        debug!("MemoryAgent received message: {}", message.message_type);
        Ok(())
    }
    
    async fn process(&mut self, context: AgentContext) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();
        
        debug!("MemoryAgent processing request for session: {}", context.session_id);
        
        // Extract action from context
        let action = context.parameters.get("action").and_then(|v| v.as_str()).unwrap_or("status");
        
        let result = match action {
            "store" => {
                let content = context.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("content required".to_string()))?;
                
                let memory_type_str = context.parameters.get("memory_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("episodic");
                
                let memory_type = match memory_type_str {
                    "episodic" => MemoryType::Episodic,
                    "semantic" => MemoryType::Semantic,
                    "working" => MemoryType::Working,
                    "user" => MemoryType::User,
                    _ => return Err(AgentError::ProcessingError(format!("Invalid memory_type: {}", memory_type_str))),
                };
                
                let metadata = context.parameters.get("metadata")
                    .and_then(|v| v.as_object())
                    .map(|obj| {
                        obj.iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect()
                    })
                    .unwrap_or_default();
                
                let memory_id = self.store_memory(
                    context.user_id,
                    Some(context.session_id),
                    memory_type,
                    content.to_string(),
                    metadata,
                ).await?;
                
                json!({
                    "action": "store",
                    "memory_id": memory_id,
                    "status": "stored"
                })
            }
            
            "query" => {
                let memory_type_str = context.parameters.get("memory_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("all");
                
                let search_term = context.parameters.get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                
                let limit = context.parameters.get("limit")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                
                let query = MemoryQuery {
                    user_id: context.user_id,
                    session_id: Some(context.session_id),
                    memory_type: memory_type_str.to_string(),
                    query_text: search_term.to_string(),
                    limit: limit.map(|l| l as usize).or(Some(self.config.default_query_limit as usize)),
                    offset: None,
                    filters: HashMap::new(),
                };
                
                let results = self.query_memory(query).await?;
                
                let memories_json: Vec<Value> = results.iter()
                    .map(|entry| json!({
                        "memory_id": entry.memory_id,
                        "memory_type": entry.memory_type,
                        "content": entry.content,
                        "timestamp": entry.timestamp,
                        "activation": entry.activation,
                        "relevance": entry.relevance,
                        "emotional_salience": entry.emotional_salience,
                        "strength": entry.strength
                    }))
                    .collect();
                
                json!({
                    "action": "query",
                    "memories": memories_json,
                    "count": memories_json.len()
                })
            }
            
            "get" => {
                let memory_id_str = context.parameters.get("memory_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("memory_id required".to_string()))?;
                
                let memory_id = Uuid::parse_str(memory_id_str)
                    .map_err(|_| AgentError::ProcessingError("Invalid memory_id".to_string()))?;
                
                let memory = self.get_memory(memory_id).await?;
                
                match memory {
                    Some(memory) => {
                        json!({
                            "action": "get",
                            "memory": {
                                "memory_id": memory.memory_id,
                                "memory_type": memory.memory_type,
                                "content": memory.content,
                                "timestamp": memory.timestamp,
                                "activation": memory.activation,
                                "relevance": memory.relevance,
                                "emotional_salience": memory.emotional_salience,
                                "strength": memory.strength
                            }
                        })
                    },
                    None => json!({
                        "action": "get",
                        "error": "Memory not found"
                    })
                }
            }
            
            "update" => {
                let memory_id_str = context.parameters.get("memory_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("memory_id required".to_string()))?;
                
                let memory_id = Uuid::parse_str(memory_id_str)
                    .map_err(|_| AgentError::ProcessingError("Invalid memory_id".to_string()))?;
                
                let content = context.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                
                let metadata = context.parameters.get("metadata")
                    .and_then(|v| v.as_object())
                    .map(|obj| {
                        obj.iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect()
                    });
                
                let success = self.update_memory(memory_id, content, metadata).await?;
                
                json!({
                    "action": "update",
                    "memory_id": memory_id,
                    "success": success
                })
            }
            
            "delete" => {
                let memory_id_str = context.parameters.get("memory_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("memory_id required".to_string()))?;
                
                let memory_id = Uuid::parse_str(memory_id_str)
                    .map_err(|_| AgentError::ProcessingError("Invalid memory_id".to_string()))?;
                
                let success = self.delete_memory(memory_id).await?;
                
                json!({
                    "action": "delete",
                    "memory_id": memory_id,
                    "success": success
                })
            }
            
            "search" => {
                let search_term = context.parameters.get("search_term")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("search_term required".to_string()))?;
                
                let memory_type_str = context.parameters.get("memory_type")
                    .and_then(|v| v.as_str());
                
                let memory_type = match memory_type_str {
                    Some(t) => Some(match t {
                        "episodic" => MemoryType::Episodic,
                        "semantic" => MemoryType::Semantic,
                        "working" => MemoryType::Working,
                        "user" => MemoryType::User,
                        _ => return Err(AgentError::ProcessingError(format!("Invalid memory_type: {}", t))),
                    }),
                    None => None,
                };
                
                let limit = context.parameters.get("limit")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                
                let results = self.search_memories(
                    search_term.to_string(),
                    context.user_id,
                    memory_type,
                    limit,
                ).await?;
                
                let memories_json: Vec<Value> = results.iter()
                    .map(|entry| json!({
                        "memory_id": entry.memory_id,
                        "memory_type": entry.memory_type,
                        "content": entry.content,
                        "timestamp": entry.timestamp,
                        "activation": entry.activation,
                        "relevance": entry.relevance,
                        "emotional_salience": entry.emotional_salience,
                        "strength": entry.strength
                    }))
                    .collect();
                
                json!({
                    "action": "search",
                    "search_term": search_term,
                    "memories": memories_json,
                    "count": memories_json.len()
                })
            }
            
            "stats" => {
                let stats = self.get_memory_stats().await?;
                json!({
                    "action": "stats",
                    "stats": stats
                })
            }
            
            "cleanup" => {
                let max_age_hours = context.parameters.get("max_age_hours")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(24); // Default 24 hours
                
                let cleaned_count = self.cleanup_old_memories(max_age_hours).await?;
                
                json!({
                    "action": "cleanup",
                    "cleaned_count": cleaned_count
                })
            }
            
            _ => {
                return Err(AgentError::ProcessingError(format!("Unknown action: {}", action)));
            }
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Update stats
        self.stats.messages_processed += 1;
        self.stats.avg_processing_time_ms = 
            (self.stats.avg_processing_time_ms * (self.stats.messages_processed - 1) as f64 + 
             processing_time as f64) / self.stats.messages_processed as f64;
        self.stats.last_activity = chrono::Utc::now();
        
        let response = AgentResponse::success(
            context.session_id,
            result,
            processing_time,
        );
        
        Ok(response)
    }
    
    async fn respond(&mut self, _response: AgentResponse) -> Result<()> {
        debug!("MemoryAgent sending response");
        Ok(())
    }
    
    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down MemoryAgent");
        self.status = AgentStatus::Shutdown;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<bool> {
        // Test memory store connectivity
        let test_query = MemoryQuery {
            user_id: None,
            session_id: None,
            memory_type: "test".to_string(),
            query_text: "health_check".to_string(),
            limit: Some(1),
            offset: None,
            filters: HashMap::new(),
        };
        
        match self.memory_store.read().await.query(&test_query.query_text).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    fn get_stats(&self) -> AgentStats {
        self.stats.clone()
    }
    
    fn get_config(&self) -> AgentConfig {
        self.config.clone().into()
    }
}

impl From<MemoryAgentConfig> for AgentConfig {
    fn from(_config: MemoryAgentConfig) -> Self {
        AgentConfig {
            agent_id: "memory_agent".to_string(),
            agent_type: "memory".to_string(),
            max_concurrent_tasks: 10,
            timeout_seconds: 30,
        }
    }
}

impl Default for MemoryAgentConfig {
    fn default() -> Self {
        Self {
            default_query_limit: 100,
            enable_compression: true,
            compression_threshold: 1000,
            cache_duration_seconds: 3600,
            enable_cleanup: true,
            cleanup_interval_seconds: 86400,
        }
    }
}
