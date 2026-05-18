//! Context Agent
//! 
//! Agent untuk merge context dari memory, prompt, dan session.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use uuid::Uuid;
use serde_json::{Value, json};
use tracing::{debug, info, warn};

use crate::{
    Agent, AgentError, Result, AgentMessage, AgentResponse, AgentStatus,
    AgentContext, AgentStats, AgentConfig
};
use nexora_memory::MemoryLayers;

/// Memory query untuk context agent
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub user_id: Option<Uuid>,
    pub session_id: Option<Uuid>,
    pub memory_type: String,
    pub query_text: String,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub filters: HashMap<String, String>,
}

/// Memory result type alias
pub type MemoryResult<T> = std::result::Result<T, crate::AgentError>;

/// Context agent untuk menggabungkan berbagai sumber context
pub struct ContextAgent {
    /// Unique ID
    id: Uuid,
    /// Agent name
    name: String,
    /// Current status
    status: AgentStatus,
    /// Memory store
    memory_store: Arc<MemoryLayers>,
    /// Statistics
    stats: AgentStats,
    /// Configuration
    config: ContextAgentConfig,
}

/// Configuration untuk context agent
#[derive(Debug, Clone)]
pub struct ContextAgentConfig {
    /// Maximum context length (tokens)
    pub max_context_length: usize,
    /// Context retention policy
    pub retention_policy: ContextRetentionPolicy,
    /// Context sources
    pub context_sources: Vec<ContextSource>,
    /// Merge strategy
    pub merge_strategy: ContextMergeStrategy,
}

/// Retention policy untuk context
#[derive(Debug, Clone)]
pub enum ContextRetentionPolicy {
    /// Keep all context
    KeepAll,
    /// Keep last N items
    KeepLast(usize),
    /// Keep based on time window (hours)
    TimeWindow(u64),
    /// Keep based on relevance score
    RelevanceThreshold(f64),
}

/// Source untuk context
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize)]
pub enum ContextSource {
    /// Session context
    Session,
    /// User memory
    UserMemory,
    /// Episodic memory
    EpisodicMemory,
    /// Semantic memory
    SemanticMemory,
    /// Working memory
    WorkingMemory,
    /// External context
    External(String),
}

/// Strategy untuk merge context
#[derive(Debug, Clone)]
pub enum ContextMergeStrategy {
    /// Simple concatenation
    Concatenate,
    /// Weighted merge
    Weighted(HashMap<ContextSource, f64>),
    /// Priority-based
    Priority(Vec<ContextSource>),
    /// Semantic merge
    Semantic,
}

/// Merged context result
#[derive(Debug, Clone)]
pub struct MergedContext {
    /// Merged context data
    pub context: Value,
    /// Source contributions
    pub contributions: HashMap<ContextSource, f64>,
    /// Total context size (tokens)
    pub size_tokens: usize,
    /// Metadata
    pub metadata: HashMap<String, Value>,
}

impl ContextAgent {
    /// Create new context agent
    pub fn new(
        memory_store: Arc<MemoryLayers>,
        config: ContextAgentConfig,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "ContextAgent".to_string(),
            status: AgentStatus::Initializing,
            memory_store,
            stats: AgentStats::default(),
            config,
        }
    }
    
    /// Merge context dari berbagai sumber
    pub async fn merge_context(
        &self,
        session_id: Uuid,
        user_id: Option<Uuid>,
        prompt_context: &AgentContext,
    ) -> Result<MergedContext> {
        debug!("Merging context for session: {}", session_id);
        
        let mut context_data = HashMap::new();
        let mut contributions = HashMap::new();
        let mut total_size = 0;
        
        // Collect context dari setiap source
        for source in &self.config.context_sources {
            let (data, size) = self.collect_context_from_source(source, session_id, user_id, prompt_context).await?;
            
            if !data.is_null() {
                context_data.insert(format!("{:?}", source), data.clone());
                contributions.insert(source.clone(), size as f64);
                total_size += size;
            }
        }
        
        // Apply merge strategy
        let merged_context = self.apply_merge_strategy(&context_data, &contributions).await?;
        
        // Apply retention policy
        let final_context = self.apply_retention_policy(&merged_context).await?;
        
        let result = MergedContext {
            context: final_context,
            contributions: contributions.clone(),
            size_tokens: total_size,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("session_id".to_string(), Value::String(session_id.to_string()));
                meta.insert("merge_strategy".to_string(), Value::String(format!("{:?}", self.config.merge_strategy)));
                meta.insert("sources_count".to_string(), Value::Number(contributions.len().into()));
                meta
            },
        };
        
        debug!("Context merged successfully, size: {} tokens", result.size_tokens);
        Ok(result)
    }
    
    /// Collect context dari specific source
    async fn collect_context_from_source(
        &self,
        source: &ContextSource,
        session_id: Uuid,
        user_id: Option<Uuid>,
        prompt_context: &AgentContext,
    ) -> Result<(Value, usize)> {
        match source {
            ContextSource::Session => {
                let session_data = Value::Object(
                    prompt_context.session_state.iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                );
                Ok((session_data.clone(), self.estimate_token_count(&session_data)))
            }
            
            ContextSource::UserMemory => {
                if let Some(user_id) = user_id {
                    let query = MemoryQuery {
                        user_id: Some(user_id),
                        session_id: None,
                        memory_type: "user_memory".to_string(),
                        query_text: "user_context".to_string(),
                        limit: Some(100),
                        offset: None,
                        filters: HashMap::new(),
                    };
                    
                    match self.memory_store.query(&query.query_text).await {
                        Ok(results) => {
                            let converted_results = self.convert_layers_to_memory_results(results);
                            let memory_data = self.convert_memory_results(&converted_results);
                            Ok((memory_data.clone(), self.estimate_token_count(&memory_data)))
                        }
                        Err(e) => {
                            warn!("Failed to query user memory: {}", e);
                            Ok((Value::Null, 0))
                        }
                    }
                } else {
                    Ok((Value::Null, 0))
                }
            }
            
            ContextSource::EpisodicMemory => {
                let query = MemoryQuery {
                    user_id,
                    session_id: Some(session_id),
                    memory_type: "episodic".to_string(),
                    query_text: "episodic_context".to_string(),
                    limit: Some(50),
                    offset: None,
                    filters: HashMap::new(),
                };
                
                match self.memory_store.query(&query.query_text).await {
                    Ok(results) => {
                        let converted_results = self.convert_layers_to_memory_results(results);
                        let memory_data = self.convert_memory_results(&converted_results);
                        Ok((memory_data.clone(), self.estimate_token_count(&memory_data)))
                    }
                    Err(e) => {
                        warn!("Failed to query episodic memory: {}", e);
                        Ok((Value::Null, 0))
                    }
                }
            }
            
            ContextSource::SemanticMemory => {
                let query = MemoryQuery {
                    user_id,
                    session_id: None,
                    memory_type: "semantic".to_string(),
                    query_text: "semantic_context".to_string(),
                    limit: Some(100),
                    offset: None,
                    filters: HashMap::new(),
                };
                
                match self.memory_store.query(&query.query_text).await {
                    Ok(results) => {
                        let converted_results = self.convert_layers_to_memory_results(results);
                        let memory_data = self.convert_memory_results(&converted_results);
                        Ok((memory_data.clone(), self.estimate_token_count(&memory_data)))
                    }
                    Err(e) => {
                        warn!("Failed to query semantic memory: {}", e);
                        Ok((Value::Null, 0))
                    }
                }
            }
            
            ContextSource::WorkingMemory => {
                let query = MemoryQuery {
                    user_id,
                    session_id: Some(session_id),
                    memory_type: "working".to_string(),
                    query_text: "working_context".to_string(),
                    limit: Some(20),
                    offset: None,
                    filters: HashMap::new(),
                };
                
                match self.memory_store.query(&query.query_text).await {
                    Ok(results) => {
                        let converted_results = self.convert_layers_to_memory_results(results);
                        let memory_data = self.convert_memory_results(&converted_results);
                        Ok((memory_data.clone(), self.estimate_token_count(&memory_data)))
                    }
                    Err(e) => {
                        warn!("Failed to query working memory: {}", e);
                        Ok((Value::Null, 0))
                    }
                }
            }
            
            ContextSource::External(source_name) => {
                // Implement external context fetching
                debug!("Fetching external context from source: {}", source_name);
                self.fetch_external_context(source_name).await
            }
        }
    }
    
    /// Fetch external context from various sources
    async fn fetch_external_context(&self, source_name: &str) -> Result<(Value, usize)> {
        match source_name {
            // Web API sources
            "wikipedia" => self.fetch_wikipedia_context().await,
            "news" => self.fetch_news_context().await,
            "weather" => self.fetch_weather_context().await,
            "stock" => self.fetch_stock_context().await,
            
            // Database sources  
            "user_profile" => self.fetch_user_profile_context().await,
            "knowledge_base" => self.fetch_knowledge_base_context().await,
            
            // File system sources
            "documents" => self.fetch_document_context().await,
            "logs" => self.fetch_log_context().await,
            
            // Default fallback
            _ => {
                warn!("Unknown external context source: {}", source_name);
                Ok((Value::Null, 0))
            }
        }
    }
    
    /// Fetch context from Wikipedia API
    async fn fetch_wikipedia_context(&self) -> Result<(Value, usize)> {
        debug!("Fetching Wikipedia context via API");
        
        let response = reqwest::get("https://en.wikipedia.org/api/rest_v1/page/random/summary")
            .await
            .map_err(|e| AgentError::ProcessingError(
                format!("External source wikipedia requires reqwest dependency: {}", e)
            ))?;
        
        if !response.status().is_success() {
            return Err(AgentError::ProcessingError(
                format!("Wikipedia API returned unexpected status: {}", response.status())
            ));
        }
        
        let data: Value = response.json().await?;
        
        let result = json!({
            "source": "wikipedia",
            "title": data["title"],
            "content": data["extract"],
            "page_url": data["content_urls"]["desktop"]["page"],
            "last_updated": chrono::Utc::now().to_rfc3339(),
            "confidence": 0.9
        });
        
        let token_count = self.estimate_token_count(&result);
        Ok((result, token_count))
    }
    
    /// Fetch context from news API
    async fn fetch_news_context(&self) -> Result<(Value, usize)> {
        debug!("Fetching news context via NewsAPI");
        
        let api_key = std::env::var("NEWSAPI_KEY")
            .map_err(|_| AgentError::ProcessingError(
                "External source news requires NEWSAPI_KEY environment variable".to_string()
            ))?;
        
        let response = reqwest::Client::new()
            .get("https://newsapi.org/v2/top-headlines")
            .query(&[("country", "us"), ("pageSize", "5")])
            .header("X-Api-Key", &api_key)
            .send()
            .await
            .map_err(|e| AgentError::ProcessingError(
                format!("External source news requires reqwest dependency: {}", e)
            ))?;
        
        if !response.status().is_success() {
            return Err(AgentError::ProcessingError(
                format!("News API returned status: {}", response.status())
            ));
        }
        
        let data: Value = response.json().await?;
        
        let headlines: Vec<Value> = data["articles"].as_array()
            .map(|articles| {
                articles.iter().map(|a| json!({
                    "title": a["title"],
                    "source": a["source"]["name"],
                    "url": a["url"],
                })).collect()
            })
            .unwrap_or_default();
        
        let result = json!({
            "source": "news",
            "headlines": headlines,
            "last_updated": chrono::Utc::now().to_rfc3339(),
            "confidence": 0.8
        });
        
        let token_count = self.estimate_token_count(&result);
        Ok((result, token_count))
    }
    
    /// Fetch context from weather API via Open-Meteo (free, no API key required)
    async fn fetch_weather_context(&self) -> Result<(Value, usize)> {
        debug!("Fetching weather context via Open-Meteo API");
        
        let response = reqwest::get(
            "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&current_weather=true"
        )
            .await
            .map_err(|e| AgentError::ProcessingError(
                format!("External source weather requires reqwest dependency: {}", e)
            ))?;
        
        if !response.status().is_success() {
            return Err(AgentError::ProcessingError(
                format!("Weather API returned status: {}", response.status())
            ));
        }
        
        let data: Value = response.json().await?;
        let current = &data["current_weather"];
        
        let result = json!({
            "source": "weather",
            "temperature": format!("{}°C", current["temperature"].as_f64().unwrap_or(0.0)),
            "wind_speed": format!("{} km/h", current["windspeed"].as_f64().unwrap_or(0.0)),
            "wind_direction": format!("{}°", current["winddirection"].as_f64().unwrap_or(0.0)),
            "weather_code": current["weathercode"].as_i64().unwrap_or(0),
            "location": "40.7128°N, 74.0060°W",
            "last_updated": chrono::Utc::now().to_rfc3339(),
            "confidence": 0.7
        });
        
        let token_count = self.estimate_token_count(&result);
        Ok((result, token_count))
    }
    
    /// Fetch context from stock market API via Finnhub
    async fn fetch_stock_context(&self) -> Result<(Value, usize)> {
        debug!("Fetching stock context via Finnhub API");
        
        let api_key = std::env::var("FINNHUB_API_KEY")
            .map_err(|_| AgentError::ProcessingError(
                "External source stock requires FINNHUB_API_KEY environment variable".to_string()
            ))?;
        
        let response = reqwest::Client::new()
            .get("https://finnhub.io/api/v1/quote")
            .query(&[("symbol", "SPY")])
            .header("X-Finnhub-Token", &api_key)
            .send()
            .await
            .map_err(|e| AgentError::ProcessingError(
                format!("External source stock requires reqwest dependency: {}", e)
            ))?;
        
        if !response.status().is_success() {
            return Err(AgentError::ProcessingError(
                format!("Stock API returned status: {}", response.status())
            ));
        }
        
        let data: Value = response.json().await?;
        
        let current_price = data["c"].as_f64().unwrap_or(0.0);
        let previous_close = data["pc"].as_f64().unwrap_or(0.0);
        let change_pct = if previous_close > 0.0 {
            ((current_price - previous_close) / previous_close * 100.0 * 100.0).round() / 100.0
        } else {
            0.0
        };
        
        let result = json!({
            "source": "stock",
            "market_status": "Open",
            "major_indices": {
                "S&P_500": format!("{:.2} ({:+.2}%)", current_price, change_pct),
            },
            "last_updated": chrono::Utc::now().to_rfc3339(),
            "confidence": 0.6
        });
        
        let token_count = self.estimate_token_count(&result);
        Ok((result, token_count))
    }
    
    /// Fetch user profile context from database
    async fn fetch_user_profile_context(&self) -> Result<(Value, usize)> {
        Err(AgentError::ProcessingError(
            "External source user_profile requires a configured user profile database service".to_string()
        ))
    }
    
    /// Fetch context from knowledge base
    async fn fetch_knowledge_base_context(&self) -> Result<(Value, usize)> {
        Err(AgentError::ProcessingError(
            "External source knowledge_base requires a configured knowledge base service".to_string()
        ))
    }
    
    /// Fetch context from document storage via filesystem
    async fn fetch_document_context(&self) -> Result<(Value, usize)> {
        debug!("Fetching document context from filesystem");
        
        let docs_dir = std::path::Path::new("documents");
        if !docs_dir.exists() {
            return Err(AgentError::ProcessingError(
                "External source documents requires a 'documents/' directory with indexed files".to_string()
            ));
        }
        
        let entries = std::fs::read_dir(docs_dir)
            .map_err(|e| AgentError::ProcessingError(
                format!("Failed to read documents directory: {}", e)
            ))?;
        
        let mut recent_files = Vec::with_capacity(10);
        let mut total_files = 0u64;
        
        for entry in entries {
            let entry = entry.map_err(|e| AgentError::ProcessingError(
                format!("Failed to read document entry: {}", e)
            ))?;
            if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                total_files += 1;
                if recent_files.len() < 10 {
                    if let Some(name) = entry.file_name().to_str() {
                        recent_files.push(Value::String(name.to_string()));
                    }
                }
            }
        }
        
        let result = json!({
            "source": "documents",
            "recent_files": recent_files,
            "total_files": total_files,
            "last_updated": chrono::Utc::now().to_rfc3339(),
            "confidence": 0.7
        });
        
        let token_count = self.estimate_token_count(&result);
        Ok((result, token_count))
    }
    
    /// Fetch context from log files via filesystem
    async fn fetch_log_context(&self) -> Result<(Value, usize)> {
        debug!("Fetching log context from filesystem");
        
        let log_dir = std::path::Path::new("logs");
        if !log_dir.exists() {
            return Err(AgentError::ProcessingError(
                "External source logs requires a 'logs/' directory with application logs".to_string()
            ));
        }
        
        let entries = std::fs::read_dir(log_dir)
            .map_err(|e| AgentError::ProcessingError(
                format!("Failed to read logs directory: {}", e)
            ))?;
        
        let mut recent_errors = Vec::with_capacity(10);
        let mut system_status = "Unknown".to_string();
        let mut latest_time = std::time::SystemTime::UNIX_EPOCH;
        
        for entry in entries {
            let entry = entry.map_err(|e| AgentError::ProcessingError(
                format!("Failed to read log entry: {}", e)
            ))?;
            
            if let Ok(meta) = entry.metadata() {
                let modified = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                if modified > latest_time && meta.is_file() {
                    latest_time = modified;
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        for line in content.lines().rev().take(20) {
                            if line.contains("error") || line.contains("ERROR") || line.contains("Error") {
                                if recent_errors.len() < 10 {
                                    recent_errors.push(Value::String(line.to_string()));
                                }
                            }
                        }
                        system_status = if recent_errors.is_empty() { "Healthy".to_string() } else { "Degraded".to_string() };
                    }
                }
            }
        }
        
        let result = json!({
            "source": "logs",
            "recent_errors": recent_errors,
            "system_status": system_status,
            "last_updated": chrono::Utc::now().to_rfc3339(),
            "confidence": 0.6
        });
        
        let token_count = self.estimate_token_count(&result);
        Ok((result, token_count))
    }
    
    /// Apply merge strategy
    async fn apply_merge_strategy(
        &self,
        context_data: &HashMap<String, Value>,
        contributions: &HashMap<ContextSource, f64>,
    ) -> Result<Value> {
        match &self.config.merge_strategy {
            ContextMergeStrategy::Concatenate => {
                let mut merged = serde_json::Map::new();
                for (key, value) in context_data {
                    merged.insert(key.clone(), value.clone());
                }
                Ok(Value::Object(merged))
            }
            
            ContextMergeStrategy::Weighted(weights) => {
                let mut weighted_data = serde_json::Map::new();
                
                for (source_name, data) in context_data {
                    if let Ok(source_enum) = source_name.parse::<ContextSource>() {
                        if let Some(&weight) = weights.get(&source_enum) {
                            let weighted_value = json!({
                                "data": data,
                                "weight": weight,
                                "contribution": contributions.get(&source_enum).unwrap_or(&0.0)
                            });
                            weighted_data.insert(source_name.clone(), weighted_value);
                        }
                    }
                }
                
                Ok(Value::Object(weighted_data))
            }
            
            ContextMergeStrategy::Priority(priority_order) => {
                let mut prioritized_data = serde_json::Map::new();
                
                for source in priority_order {
                    let source_key = format!("{:?}", source);
                    if let Some(data) = context_data.get(&source_key) {
                        prioritized_data.insert(source_key, data.clone());
                    }
                }
                
                // Add remaining sources
                for (key, value) in context_data {
                    if !prioritized_data.contains_key(key) {
                        prioritized_data.insert(key.clone(), value.clone());
                    }
                }
                
                Ok(Value::Object(prioritized_data))
            }
            
            ContextMergeStrategy::Semantic => {
                // Implement semantic merge with similarity scoring
                debug!("Performing semantic context merge");
                
                let mut merged = serde_json::Map::new();
                let mut context_items: Vec<(String, Value, f64)> = Vec::with_capacity(contributions.len());
                
                // Extract items with their contribution scores
                for (source, contribution) in contributions {
                    let source_key = format!("{:?}", source);
                    if let Some(data_tuple) = context_data.get(&source_key) {
                        if let Value::Array(arr) = data_tuple {
                            if let Some(data) = arr.get(0) {
                                context_items.push((source_key, data.clone(), *contribution));
                            }
                        } else {
                            context_items.push((source_key, data_tuple.clone(), *contribution));
                        }
                    }
                }
                
                // Sort by contribution score (highest first)
                context_items.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                
                // Merge with semantic deduplication
                let mut merged_keys = std::collections::HashSet::new();
                for (source, data, score) in context_items {
                    if let Value::Object(obj) = &data {
                        for (key, value) in obj {
                            let merged_key = format!("{}_{}", source, key);
                            
                            // Check for semantic conflicts
                            if !merged_keys.contains(&merged_key) {
                                // Apply semantic weighting based on contribution score
                                let weighted_value = if score > 0.8 {
                                    // High confidence - keep as is
                                    value.clone()
                                } else if score > 0.5 {
                                    // Medium confidence - add confidence metadata
                                    json!({
                                        "value": value,
                                        "confidence": score,
                                        "source": source
                                    })
                                } else {
                                    // Low confidence - add warning
                                    json!({
                                        "value": value,
                                        "confidence": score,
                                        "source": source,
                                        "warning": "Low confidence"
                                    })
                                };
                                
                                merged.insert(merged_key.clone(), weighted_value);
                                merged_keys.insert(merged_key);
                            } else {
                                // Handle semantic conflicts - merge with priority
                                if let Some(existing) = merged.get_mut(&merged_key) {
                                    if let Some(existing_confidence) = existing.get("confidence") {
                                        let existing_conf = existing_confidence.as_f64().unwrap_or(0.0);
                                        if score > existing_conf {
                                            // Replace with higher confidence value
                                            *existing = json!({
                                                "value": value,
                                                "confidence": score,
                                                "source": source,
                                                "replaced": true
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Non-object data, just add with source prefix
                        merged.insert(format!("{}_data", source), data.clone());
                    }
                }
                
                // Add merge metadata
                merged.insert("_merge_strategy".to_string(), json!("semantic"));
                merged.insert("_merge_timestamp".to_string(), json!(chrono::Utc::now().to_rfc3339()));
                merged.insert("_items_merged".to_string(), json!(merged_keys.len()));
                
                Ok(Value::Object(merged))
            }
        }
    }
    
    /// Apply retention policy
    async fn apply_retention_policy(&self, context: &Value) -> Result<Value> {
        match &self.config.retention_policy {
            ContextRetentionPolicy::KeepAll => Ok(context.clone()),
            
            ContextRetentionPolicy::KeepLast(n) => {
                // Implement keep last N items
                debug!("Keeping last {} items from context", n);
                
                if let Value::Object(obj) = &context {
                    let mut filtered = serde_json::Map::new();
                    let mut items_kept = 0;
                    
                    // Keep items in order of insertion (last N)
                    for (key, value) in obj.iter().rev() {
                        if items_kept < *n {
                            filtered.insert(key.clone(), value.clone());
                            items_kept += 1;
                        }
                    }
                    
                    Ok(Value::Object(filtered))
                } else {
                    Ok(context.clone())
                }
            }
            
            ContextRetentionPolicy::TimeWindow(hours) => {
                // Implement time window filtering
                debug!("Filtering context within {} hour time window", hours);
                
                let cutoff_time = chrono::Utc::now() - chrono::Duration::hours(*hours as i64);
                
                if let Value::Object(obj) = &context {
                    let mut filtered = serde_json::Map::new();
                    
                    for (key, value) in obj {
                        // Check if value has timestamp
                        if let Some(timestamp_str) = value.get("timestamp").and_then(|v| v.as_str()) {
                            if let Ok(timestamp) = chrono::DateTime::parse_from_rfc3339(timestamp_str) {
                                if timestamp > chrono::DateTime::<chrono::Utc>::from(cutoff_time) {
                                    filtered.insert(key.clone(), value.clone());
                                }
                            } else {
                                // If timestamp parsing fails, keep the item
                                filtered.insert(key.clone(), value.clone());
                            }
                        } else {
                            // If no timestamp, keep the item
                            filtered.insert(key.clone(), value.clone());
                        }
                    }
                    
                    Ok(Value::Object(filtered))
                } else {
                    Ok(context.clone())
                }
            }
            
            ContextRetentionPolicy::RelevanceThreshold(threshold) => {
                // Implement relevance filtering
                debug!("Filtering context with relevance threshold: {}", threshold);
                
                if let Value::Object(obj) = &context {
                    let mut filtered = serde_json::Map::new();
                    
                    for (key, value) in obj {
                        // Check if value has relevance score
                        if let Some(relevance) = value.get("relevance").and_then(|v| v.as_f64()) {
                            if relevance >= *threshold {
                                filtered.insert(key.clone(), value.clone());
                            }
                        } else {
                            // If no relevance score, keep the item
                            filtered.insert(key.clone(), value.clone());
                        }
                    }
                    
                    Ok(Value::Object(filtered))
                } else {
                    Ok(context.clone())
                }
            }
        }
    }
    
    /// Helper function to convert Vec<layers::MemoryEntry> to Vec<MemoryResult<MemoryEntry>>
    fn convert_layers_to_memory_results(&self, layers_results: Vec<nexora_memory::layers::MemoryEntry>) -> Vec<MemoryResult<nexora_memory::MemoryEntry>> {
        layers_results
            .into_iter()
            .enumerate()
            .map(|(i, entry)| {
                // Use simple sequential ID — hash was immediately overwritten
                let memory_id = i as u32;
                
                // Determine memory type from entry content and context
                let memory_type = if entry.value.contains("context") || 
                                 entry.value.contains("session") {
                    nexora_memory::MemoryType::Working
                } else if entry.value.contains("semantic") || 
                         entry.value.contains("fact") {
                    nexora_memory::MemoryType::Semantic
                } else if entry.value.contains("episodic") || 
                         entry.value.contains("experience") {
                    nexora_memory::MemoryType::Episodic
                } else {
                    nexora_memory::MemoryType::Working // Default for context agent
                };
                
                let model_entry = nexora_memory::MemoryEntry {
                    memory_id,
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
                Ok(model_entry)
            })
            .collect()
    }
    
    /// Convert memory results to JSON
    fn convert_memory_results(&self, results: &[MemoryResult<nexora_memory::MemoryEntry>]) -> Value {
        let memories: Vec<Value> = results.iter()
            .filter_map(|result| {
                match result {
                    Ok(entry) => Some(json!({
                        "memory_id": entry.memory_id,
                        "memory_type": entry.memory_type,
                        "content": entry.content,
                        "timestamp": entry.timestamp,
                        "activation": entry.activation,
                        "relevance": entry.relevance,
                        "emotional_salience": entry.emotional_salience,
                        "strength": entry.strength
                    })),
                    Err(_) => None,
                }
            })
            .collect();
        
        Value::Array(memories)
    }
    
    /// Estimate token count (rough approximation)
    fn estimate_token_count(&self, value: &Value) -> usize {
        let json_str = serde_json::to_string(value).unwrap_or_default();
        // Rough approximation: 1 token ≈ 4 characters
        json_str.len() / 4
    }
}

#[async_trait]
impl Agent for ContextAgent {
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn agent_type(&self) -> &str {
        "context"
    }
    
    fn status(&self) -> AgentStatus {
        self.status.clone()
    }
    
    async fn initialize(&mut self, _config: AgentConfig) -> Result<()> {
        info!("Initializing ContextAgent");
        self.status = AgentStatus::Ready;
        Ok(())
    }
    
    async fn receive(&mut self, message: AgentMessage) -> Result<()> {
        debug!("ContextAgent received message: {}", message.message_type);
        // Store message for processing
        Ok(())
    }
    
    async fn process(&mut self, context: AgentContext) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();
        
        debug!("ContextAgent processing context for session: {}", context.session_id);
        
        // Extract session info from context
        let session_id = context.session_id;
        let user_id = context.user_id;
        
        // Merge context
        let merged_context = self.merge_context(session_id, user_id, &context).await?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Update stats
        self.stats.messages_processed += 1;
        self.stats.avg_processing_time_ms = 
            (self.stats.avg_processing_time_ms * (self.stats.messages_processed - 1) as f64 + 
             processing_time as f64) / self.stats.messages_processed as f64;
        self.stats.last_activity = chrono::Utc::now();
        
        let response = AgentResponse::success(
            context.session_id, // Using session_id as request_id for now
            json!({
                "merged_context": merged_context.context,
                "contributions": merged_context.contributions,
                "size_tokens": merged_context.size_tokens,
                "metadata": merged_context.metadata
            }),
            processing_time,
        );
        
        Ok(response)
    }
    
    async fn respond(&mut self, _response: AgentResponse) -> Result<()> {
        debug!("ContextAgent sending response");
        Ok(())
    }
    
    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down ContextAgent");
        self.status = AgentStatus::Shutdown;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<bool> {
        // Check if memory store is accessible
        let test_query = MemoryQuery {
            user_id: None,
            session_id: None,
            memory_type: "test".to_string(),
            query_text: "health_check".to_string(),
            limit: Some(1),
            offset: None,
            filters: HashMap::new(),
        };
        
        match self.memory_store.query(&test_query.query_text).await {
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

impl From<ContextAgentConfig> for AgentConfig {
    fn from(_config: ContextAgentConfig) -> Self {
        AgentConfig {
            agent_id: "context_agent".to_string(),
            agent_type: "context".to_string(),
            max_concurrent_tasks: 6,
            timeout_seconds: 20,
        }
    }
}

impl Default for ContextAgentConfig {
    fn default() -> Self {
        Self {
            max_context_length: 4096,
            retention_policy: ContextRetentionPolicy::KeepAll,
            context_sources: vec![
                ContextSource::Session,
                ContextSource::WorkingMemory,
                ContextSource::EpisodicMemory,
                ContextSource::SemanticMemory,
            ],
            merge_strategy: ContextMergeStrategy::Concatenate,
        }
    }
}

// Implement parsing for ContextSource
impl std::str::FromStr for ContextSource {
    type Err = String;
    
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Session" => Ok(ContextSource::Session),
            "UserMemory" => Ok(ContextSource::UserMemory),
            "EpisodicMemory" => Ok(ContextSource::EpisodicMemory),
            "SemanticMemory" => Ok(ContextSource::SemanticMemory),
            "WorkingMemory" => Ok(ContextSource::WorkingMemory),
            _ => Err(format!("Unknown ContextSource: {}", s)),
        }
    }
}
