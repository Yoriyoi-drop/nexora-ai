//! Response Agent
//! 
//! Agent untuk final formatting dan response generation.

use std::collections::HashMap;
use async_trait::async_trait;
use uuid::Uuid;
use serde_json::{Value, json};
use tracing::{debug, info};

use crate::{
    Agent, AgentError, Result, AgentMessage, AgentResponse, AgentStatus,
    AgentContext, AgentStats, AgentConfig
};

/// Response agent untuk final formatting
pub struct ResponseAgent {
    /// Unique ID
    id: Uuid,
    /// Agent name
    name: String,
    /// Current status
    status: AgentStatus,
    /// Response formatters
    formatters: HashMap<String, Box<dyn ResponseFormatter>>,
    /// Statistics
    stats: AgentStats,
    /// Configuration
    config: ResponseAgentConfig,
}

/// Configuration untuk response agent
#[derive(Debug, Clone)]
pub struct ResponseAgentConfig {
    /// Default response format
    pub default_format: String,
    /// Enable response caching
    pub enable_caching: bool,
    /// Cache duration (seconds)
    pub cache_duration_seconds: u64,
    /// Enable response compression
    pub enable_compression: bool,
    /// Maximum response size (bytes)
    pub max_response_size_bytes: usize,
}

/// Formatted response
#[derive(Debug, Clone)]
pub struct FormattedResponse {
    /// Formatted content
    pub content: String,
    /// Response format
    pub format: String,
    /// Metadata
    pub metadata: HashMap<String, Value>,
    /// Response size (bytes)
    pub size_bytes: usize,
    /// Generation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Trait untuk response formatters
pub trait ResponseFormatter: Send + Sync {
    /// Format name
    fn name(&self) -> &str;
    
    /// Format response
    fn format(&self, data: &Value, context: &Value) -> Result<FormattedResponse>;
    
    /// Can handle this format?
    fn can_handle(&self, format: &str) -> bool;
}

impl ResponseAgent {
    /// Create new response agent
    pub fn new(config: ResponseAgentConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "ResponseAgent".to_string(),
            status: AgentStatus::Initializing,
            formatters: HashMap::new(),
            stats: AgentStats::default(),
            config,
        }
    }
    
    /// Add response formatter
    pub fn add_formatter(&mut self, formatter: Box<dyn ResponseFormatter>) {
        self.formatters.insert(formatter.name().to_string(), formatter);
    }
    
    /// Format response
    pub fn format_response(
        &self,
        data: &Value,
        format: Option<&str>,
        context: &Value,
    ) -> Result<FormattedResponse> {
        debug!("Formatting response with format: {:?}", format);
        
        let format_name = format.unwrap_or(&self.config.default_format);
        
        // Find appropriate formatter
        let formatter = self.formatters.get(format_name)
            .or_else(|| self.formatters.get(&self.config.default_format))
            .ok_or_else(|| AgentError::ProcessingError(
                format!("No formatter found for format: {}", format_name)
            ))?;
        
        // Format response
        let formatted = formatter.format(data, context)?;
        
        // Check size limit
        if formatted.size_bytes > self.config.max_response_size_bytes {
            return Err(AgentError::ProcessingError(
                format!("Response size ({}) exceeds maximum ({})", 
                       formatted.size_bytes, self.config.max_response_size_bytes)
            ));
        }
        
        debug!("Response formatted successfully, size: {} bytes", formatted.size_bytes);
        Ok(formatted)
    }
    
    /// Create simple text response
    pub fn create_text_response(&self, content: String, metadata: Option<HashMap<String, Value>>) -> FormattedResponse {
        let size_bytes = content.len();
        
        FormattedResponse {
            content,
            format: "text".to_string(),
            metadata: metadata.unwrap_or_default(),
            size_bytes,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Create JSON response
    pub fn create_json_response(&self, data: Value, metadata: Option<HashMap<String, Value>>) -> Result<FormattedResponse> {
        let content = serde_json::to_string_pretty(&data)
            .map_err(|e| AgentError::ProcessingError(format!("JSON serialization error: {}", e)))?;
        
        let size_bytes = content.len();
        
        Ok(FormattedResponse {
            content,
            format: "json".to_string(),
            metadata: metadata.unwrap_or_default(),
            size_bytes,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Create markdown response
    pub fn create_markdown_response(&self, content: String, metadata: Option<HashMap<String, Value>>) -> FormattedResponse {
        let size_bytes = content.len();
        
        FormattedResponse {
            content,
            format: "markdown".to_string(),
            metadata: metadata.unwrap_or_default(),
            size_bytes,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Create HTML response
    pub fn create_html_response(&self, content: String, metadata: Option<HashMap<String, Value>>) -> FormattedResponse {
        let html_content = format!(
            r#"<html>
<head>
    <title>Nexora Response</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .response {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="response">
        {}
    </div>
</body>
</html>"#,
            content
        );
        
        let size_bytes = html_content.len();
        
        FormattedResponse {
            content: html_content,
            format: "html".to_string(),
            metadata: metadata.unwrap_or_default(),
            size_bytes,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Create error response
    pub fn create_error_response(&self, error: String, error_code: Option<u32>) -> FormattedResponse {
        let content = if let Some(code) = error_code {
            format!("Error {}: {}", code, error)
        } else {
            format!("Error: {}", error)
        };
        
        let mut metadata = HashMap::new();
        metadata.insert("error".to_string(), Value::String(error));
        if let Some(code) = error_code {
            metadata.insert("error_code".to_string(), Value::Number(code.into()));
        }
        
        self.create_text_response(content, Some(metadata))
    }
    
    /// Create streaming response
    pub fn create_streaming_response(&self, chunks: Vec<String>) -> Result<FormattedResponse> {
        let combined_content = chunks.join("");
        let size_bytes = combined_content.len();
        
        let mut metadata = HashMap::new();
        metadata.insert("chunk_count".to_string(), Value::Number(chunks.len().into()));
        metadata.insert("streaming".to_string(), Value::Bool(true));
        
        Ok(FormattedResponse {
            content: combined_content,
            format: "streaming".to_string(),
            metadata,
            size_bytes,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Validate response format
    pub fn validate_response(&self, response: &FormattedResponse) -> Result<bool> {
        // Check if format is supported
        if !self.formatters.contains_key(&response.format) && response.format != self.config.default_format {
            return Ok(false);
        }
        
        // Check size limit
        if response.size_bytes > self.config.max_response_size_bytes {
            return Ok(false);
        }
        
        // Basic content validation
        if response.content.trim().is_empty() {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Get available formats
    pub fn get_available_formats(&self) -> Vec<String> {
        self.formatters.keys().cloned().collect()
    }
    
    /// Get response statistics
    pub fn get_response_stats(&self) -> ResponseStats {
        ResponseStats {
            available_formats: self.formatters.len(),
            default_format: self.config.default_format.clone(),
            caching_enabled: self.config.enable_caching,
            compression_enabled: self.config.enable_compression,
            max_response_size_bytes: self.config.max_response_size_bytes,
        }
    }
}

/// Response statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResponseStats {
    pub available_formats: usize,
    pub default_format: String,
    pub caching_enabled: bool,
    pub compression_enabled: bool,
    pub max_response_size_bytes: usize,
}

#[async_trait]
impl Agent for ResponseAgent {
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn agent_type(&self) -> &str {
        "response"
    }
    
    fn status(&self) -> AgentStatus {
        self.status.clone()
    }
    
    async fn initialize(&mut self, _config: AgentConfig) -> Result<()> {
        info!("Initializing ResponseAgent");
        
        // Add default formatters
        self.add_default_formatters();
        
        self.status = AgentStatus::Ready;
        Ok(())
    }
    
    async fn receive(&mut self, message: AgentMessage) -> Result<()> {
        debug!("ResponseAgent received message: {}", message.message_type);
        Ok(())
    }
    
    async fn process(&mut self, context: AgentContext) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();
        
        debug!("ResponseAgent processing request for session: {}", context.session_id);
        
        // Extract action from context
        let action = context.parameters.get("action").and_then(|v| v.as_str()).unwrap_or("format");
        
        let result = match action {
            "format" => {
                let data = context.parameters.get("data")
                    .cloned()
                    .unwrap_or(Value::Null);
                
                let format = context.parameters.get("format")
                    .and_then(|v| v.as_str());
                
                let metadata_value = serde_json::to_value(&context.metadata)?;
                let formatted = self.format_response(&data, format, &metadata_value)?;
                
                json!({
                    "action": "format",
                    "formatted_response": {
                        "content": formatted.content,
                        "format": formatted.format,
                        "size_bytes": formatted.size_bytes,
                        "timestamp": formatted.timestamp,
                        "metadata": formatted.metadata
                    }
                })
            }
            
            "create_text" => {
                let content = context.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("content required".to_string()))?;
                
                let formatted = self.create_text_response(content.to_string(), None);
                
                json!({
                    "action": "create_text",
                    "formatted_response": {
                        "content": formatted.content,
                        "format": formatted.format,
                        "size_bytes": formatted.size_bytes,
                        "timestamp": formatted.timestamp,
                        "metadata": formatted.metadata
                    }
                })
            }
            
            "create_json" => {
                let data = context.parameters.get("data")
                    .cloned()
                    .unwrap_or(Value::Null);
                
                let formatted = self.create_json_response(data, None)?;
                
                json!({
                    "action": "create_json",
                    "formatted_response": {
                        "content": formatted.content,
                        "format": formatted.format,
                        "size_bytes": formatted.size_bytes,
                        "timestamp": formatted.timestamp,
                        "metadata": formatted.metadata
                    }
                })
            }
            
            "create_markdown" => {
                let content = context.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("content required".to_string()))?;
                
                let formatted = self.create_markdown_response(content.to_string(), None);
                
                json!({
                    "action": "create_markdown",
                    "formatted_response": {
                        "content": formatted.content,
                        "format": formatted.format,
                        "size_bytes": formatted.size_bytes,
                        "timestamp": formatted.timestamp,
                        "metadata": formatted.metadata
                    }
                })
            }
            
            "create_html" => {
                let content = context.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("content required".to_string()))?;
                
                let formatted = self.create_html_response(content.to_string(), None);
                
                json!({
                    "action": "create_html",
                    "formatted_response": {
                        "content": formatted.content,
                        "format": formatted.format,
                        "size_bytes": formatted.size_bytes,
                        "timestamp": formatted.timestamp,
                        "metadata": formatted.metadata
                    }
                })
            }
            
            "create_error" => {
                let error = context.parameters.get("error")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("error required".to_string()))?;
                
                let error_code = context.parameters.get("error_code")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                
                let formatted = self.create_error_response(error.to_string(), error_code);
                
                json!({
                    "action": "create_error",
                    "formatted_response": {
                        "content": formatted.content,
                        "format": formatted.format,
                        "size_bytes": formatted.size_bytes,
                        "timestamp": formatted.timestamp,
                        "metadata": formatted.metadata
                    }
                })
            }
            
            "create_streaming" => {
                let chunks = context.parameters.get("chunks")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| AgentError::ProcessingError("chunks required".to_string()))?;
                
                let chunk_strings: std::result::Result<Vec<String>, _> = chunks.iter()
                    .map(|v| v.as_str().map(|s| s.to_string())
                         .ok_or_else(|| AgentError::ProcessingError("Invalid chunk".to_string())))
                    .collect();
                
                let formatted = self.create_streaming_response(chunk_strings?)?;
                
                json!({
                    "action": "create_streaming",
                    "formatted_response": {
                        "content": formatted.content,
                        "format": formatted.format,
                        "size_bytes": formatted.size_bytes,
                        "timestamp": formatted.timestamp,
                        "metadata": formatted.metadata
                    }
                })
            }
            
            "validate" => {
                let content = context.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("content required".to_string()))?;
                
                let format = context.parameters.get("format")
                    .and_then(|v| v.as_str())
                    .unwrap_or("text");
                
                let size_bytes = content.len();
                let response = FormattedResponse {
                    content: content.to_string(),
                    format: format.to_string(),
                    metadata: HashMap::new(),
                    size_bytes,
                    timestamp: chrono::Utc::now(),
                };
                
                let is_valid = self.validate_response(&response)?;
                
                json!({
                    "action": "validate",
                    "is_valid": is_valid,
                    "format": format,
                    "size_bytes": size_bytes
                })
            }
            
            "formats" => {
                let formats = self.get_available_formats();
                json!({
                    "action": "formats",
                    "available_formats": formats,
                    "default_format": self.config.default_format
                })
            }
            
            "stats" => {
                let stats = self.get_response_stats();
                json!({
                    "action": "stats",
                    "stats": stats
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
        debug!("ResponseAgent sending response");
        Ok(())
    }
    
    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down ResponseAgent");
        self.status = AgentStatus::Shutdown;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<bool> {
        // Check if we have formatters available
        Ok(!self.formatters.is_empty())
    }
    
    fn get_stats(&self) -> AgentStats {
        self.stats.clone()
    }
    
    fn get_config(&self) -> AgentConfig {
        self.config.clone().into()
    }
}

impl ResponseAgent {
    /// Add default formatters
    fn add_default_formatters(&mut self) {
        // Add JSON formatter
        self.add_formatter(Box::new(JsonFormatter));
        
        // Add text formatter
        self.add_formatter(Box::new(TextFormatter));
        
        // Add markdown formatter
        self.add_formatter(Box::new(MarkdownFormatter));
        
        // Add HTML formatter
        self.add_formatter(Box::new(HtmlFormatter));
    }
}

/// JSON formatter
struct JsonFormatter;

impl ResponseFormatter for JsonFormatter {
    fn name(&self) -> &str {
        "json"
    }
    
    fn format(&self, data: &Value, _context: &Value) -> Result<FormattedResponse> {
        let content = serde_json::to_string_pretty(data)
            .map_err(|e| AgentError::ProcessingError(format!("JSON formatting error: {}", e)))?;
        
        let size_bytes = content.len();
        
        Ok(FormattedResponse {
            content,
            format: "json".to_string(),
            metadata: HashMap::new(),
            size_bytes,
            timestamp: chrono::Utc::now(),
        })
    }
    
    fn can_handle(&self, format: &str) -> bool {
        format == "json"
    }
}

/// Text formatter
struct TextFormatter;

impl ResponseFormatter for TextFormatter {
    fn name(&self) -> &str {
        "text"
    }
    
    fn format(&self, data: &Value, _context: &Value) -> Result<FormattedResponse> {
        let content = match data {
            Value::String(s) => s.clone(),
            Value::Null => "No content".to_string(),
            _ => serde_json::to_string(data)
                .unwrap_or_else(|_| "Invalid data".to_string()),
        };
        
        let size_bytes = content.len();
        
        Ok(FormattedResponse {
            content,
            format: "text".to_string(),
            metadata: HashMap::new(),
            size_bytes,
            timestamp: chrono::Utc::now(),
        })
    }
    
    fn can_handle(&self, format: &str) -> bool {
        format == "text"
    }
}

/// Markdown formatter
struct MarkdownFormatter;

impl ResponseFormatter for MarkdownFormatter {
    fn name(&self) -> &str {
        "markdown"
    }
    
    fn format(&self, data: &Value, _context: &Value) -> Result<FormattedResponse> {
        let content = match data {
            Value::String(s) => s.clone(),
            Value::Object(obj) => {
                let mut md_content = String::new();
                for (key, value) in obj {
                    md_content.push_str(&format!("**{}**: {}\n\n", key, value));
                }
                md_content
            }
            Value::Array(arr) => {
                let mut md_content = String::new();
                for (i, item) in arr.iter().enumerate() {
                    md_content.push_str(&format!("{}. {}\n\n", i + 1, item));
                }
                md_content
            }
            _ => serde_json::to_string(data)
                .unwrap_or_else(|_| "```json\nInvalid data\n```".to_string()),
        };
        
        let size_bytes = content.len();
        
        Ok(FormattedResponse {
            content,
            format: "markdown".to_string(),
            metadata: HashMap::new(),
            size_bytes,
            timestamp: chrono::Utc::now(),
        })
    }
    
    fn can_handle(&self, format: &str) -> bool {
        format == "markdown"
    }
}

/// HTML formatter
struct HtmlFormatter;

impl ResponseFormatter for HtmlFormatter {
    fn name(&self) -> &str {
        "html"
    }
    
    fn format(&self, data: &Value, _context: &Value) -> Result<FormattedResponse> {
        let content = match data {
            Value::String(s) => s.clone(),
            Value::Object(obj) => {
                let mut html_content = String::new();
                html_content.push_str("<table>\n");
                for (key, value) in obj {
                    html_content.push_str(&format!(
                        "  <tr><td><strong>{}</strong></td><td>{}</td></tr>\n",
                        key, value
                    ));
                }
                html_content.push_str("</table>");
                html_content
            }
            Value::Array(arr) => {
                let mut html_content = String::new();
                html_content.push_str("<ul>\n");
                for item in arr {
                    html_content.push_str(&format!("  <li>{}</li>\n", item));
                }
                html_content.push_str("</ul>");
                html_content
            }
            _ => format!("<pre>{}</pre>", 
                serde_json::to_string(data).unwrap_or_else(|_| "Invalid data".to_string())
            ),
        };
        
        let html_content = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Nexora Response</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .response {{ background: #f9f9f9; padding: 20px; border-radius: 8px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="response">
        {}
    </div>
</body>
</html>"#,
            content
        );
        
        let size_bytes = html_content.len();
        
        Ok(FormattedResponse {
            content: html_content,
            format: "html".to_string(),
            metadata: HashMap::new(),
            size_bytes,
            timestamp: chrono::Utc::now(),
        })
    }
    
    fn can_handle(&self, format: &str) -> bool {
        format == "html"
    }
}

impl From<ResponseAgentConfig> for AgentConfig {
    fn from(_config: ResponseAgentConfig) -> Self {
        AgentConfig {
            agent_id: "response_agent".to_string(),
            agent_type: "response".to_string(),
            max_concurrent_tasks: 8,
            timeout_seconds: 10,
        }
    }
}

impl Default for ResponseAgentConfig {
    fn default() -> Self {
        Self {
            default_format: "json".to_string(),
            enable_caching: true,
            cache_duration_seconds: 3600,
            enable_compression: false,
            max_response_size_bytes: 10000,
        }
    }
}
