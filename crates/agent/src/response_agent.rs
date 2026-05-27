//! Response Agent
//!
//! Agent untuk final formatting dan response generation.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use tracing::{debug, info};
use uuid::Uuid;

use crate::{
    Agent, AgentConfig, AgentContext, AgentError, AgentMessage, AgentResponse, AgentStats,
    AgentStatus, Result,
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
        self.formatters
            .insert(formatter.name().to_string(), formatter);
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
        let formatter = self
            .formatters
            .get(format_name)
            .or_else(|| self.formatters.get(&self.config.default_format))
            .ok_or_else(|| AgentError::ProcessingError {
                operation: "format".to_string(),
                reason: format!("No formatter found for format: {}", format_name),
            })?;

        // Format response
        let formatted = formatter.format(data, context)?;

        // Check size limit
        if formatted.size_bytes > self.config.max_response_size_bytes {
            return Err(AgentError::ProcessingError {
                operation: "format".to_string(),
                reason: format!(
                    "Response size ({}) exceeds maximum ({})",
                    formatted.size_bytes, self.config.max_response_size_bytes
                ),
            });
        }

        debug!(
            "Response formatted successfully, size: {} bytes",
            formatted.size_bytes
        );
        Ok(formatted)
    }

    /// Create simple text response
    pub fn create_text_response(
        &self,
        content: String,
        metadata: Option<HashMap<String, Value>>,
    ) -> FormattedResponse {
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
    pub fn create_json_response(
        &self,
        data: Value,
        metadata: Option<HashMap<String, Value>>,
    ) -> Result<FormattedResponse> {
        let content =
            serde_json::to_string_pretty(&data).map_err(|e| AgentError::ProcessingError {
                operation: "serialize".to_string(),
                reason: format!("JSON serialization error: {}", e),
            })?;

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
    pub fn create_markdown_response(
        &self,
        content: String,
        metadata: Option<HashMap<String, Value>>,
    ) -> FormattedResponse {
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
    pub fn create_html_response(
        &self,
        content: String,
        metadata: Option<HashMap<String, Value>>,
    ) -> FormattedResponse {
        let html_content = format!(
            r#"<!DOCTYPE html>
<html>
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
    pub fn create_error_response(
        &self,
        error: String,
        error_code: Option<u32>,
    ) -> FormattedResponse {
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
        metadata.insert(
            "chunk_count".to_string(),
            Value::Number(chunks.len().into()),
        );
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
        if !self.formatters.contains_key(&response.format)
            && response.format != self.config.default_format
        {
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

        debug!(
            "ResponseAgent processing request for session: {}",
            context.session_id
        );

        // Extract action from context
        let action = context
            .parameters
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("format");

        let result = match action {
            "format" => {
                let data = context
                    .parameters
                    .get("data")
                    .cloned()
                    .unwrap_or(Value::Null);

                let format = context.parameters.get("format").and_then(|v| v.as_str());

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
                let content = context
                    .parameters
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError {
                        operation: "validate".to_string(),
                        reason: "content required".to_string(),
                    })?;

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
                let data = context
                    .parameters
                    .get("data")
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
                let content = context
                    .parameters
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError {
                        operation: "validate".to_string(),
                        reason: "content required".to_string(),
                    })?;

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
                let content = context
                    .parameters
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError {
                        operation: "validate".to_string(),
                        reason: "content required".to_string(),
                    })?;

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
                let error = context
                    .parameters
                    .get("error")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError {
                        operation: "validate".to_string(),
                        reason: "error required".to_string(),
                    })?;

                let error_code = context
                    .parameters
                    .get("error_code")
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
                let chunks = context
                    .parameters
                    .get("chunks")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| AgentError::ProcessingError {
                        operation: "validate".to_string(),
                        reason: "chunks required".to_string(),
                    })?;

                let chunk_strings: std::result::Result<Vec<String>, _> = chunks
                    .iter()
                    .map(|v| {
                        v.as_str().map(|s| s.to_string()).ok_or_else(|| {
                            AgentError::ProcessingError {
                                operation: "parse".to_string(),
                                reason: "Invalid chunk".to_string(),
                            }
                        })
                    })
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
                let content = context
                    .parameters
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError {
                        operation: "validate".to_string(),
                        reason: "content required".to_string(),
                    })?;

                let format = context
                    .parameters
                    .get("format")
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
                return Err(AgentError::ProcessingError {
                    operation: "execute_action".to_string(),
                    reason: format!("Unknown action: {}", action),
                });
            }
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        // Update stats
        self.stats.messages_processed += 1;
        self.stats.avg_processing_time_ms = (self.stats.avg_processing_time_ms
            * (self.stats.messages_processed - 1) as f64
            + processing_time as f64)
            / self.stats.messages_processed as f64;
        self.stats.last_activity = chrono::Utc::now();

        let response = AgentResponse::success(context.session_id, result, processing_time);

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
        let content =
            serde_json::to_string_pretty(data).map_err(|e| AgentError::ProcessingError {
                operation: "serialize".to_string(),
                reason: format!("JSON formatting error: {}", e),
            })?;

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
            _ => serde_json::to_string(data).unwrap_or_else(|_| "Invalid data".to_string()),
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
                    md_content.push_str("**");
                    md_content.push_str(key);
                    md_content.push_str("**: ");
                    md_content.push_str(&value.to_string());
                    md_content.push_str("\n\n");
                }
                md_content
            }
            Value::Array(arr) => {
                let mut md_content = String::new();
                for (i, item) in arr.iter().enumerate() {
                    md_content.push_str(&(i + 1).to_string());
                    md_content.push_str(". ");
                    md_content.push_str(&item.to_string());
                    md_content.push_str("\n\n");
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
                    html_content.push_str("  <tr><td><strong>");
                    html_content.push_str(key);
                    html_content.push_str("</strong></td><td>");
                    html_content.push_str(&value.to_string());
                    html_content.push_str("</td></tr>\n");
                }
                html_content.push_str("</table>");
                html_content
            }
            Value::Array(arr) => {
                let mut html_content = String::new();
                html_content.push_str("<ul>\n");
                for item in arr {
                    html_content.push_str("  <li>");
                    html_content.push_str(&item.to_string());
                    html_content.push_str("</li>\n");
                }
                html_content.push_str("</ul>");
                html_content
            }
            _ => format!(
                "<pre>{}</pre>",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_agent_config_default() {
        let config = ResponseAgentConfig::default();
        assert_eq!(config.default_format, "json");
        assert!(config.enable_caching);
        assert_eq!(config.cache_duration_seconds, 3600);
        assert!(!config.enable_compression);
        assert_eq!(config.max_response_size_bytes, 10000);
    }

    #[test]
    fn test_response_agent_config_clone_debug() {
        let config = ResponseAgentConfig::default();
        let cloned = config.clone();
        assert_eq!(format!("{:?}", config), format!("{:?}", cloned));
    }

    #[test]
    fn test_formatted_response_creation() {
        let resp = FormattedResponse {
            content: "Hello".into(),
            format: "text".into(),
            metadata: HashMap::new(),
            size_bytes: 5,
            timestamp: chrono::Utc::now(),
        };
        assert_eq!(resp.content, "Hello");
        assert_eq!(resp.format, "text");
        assert_eq!(resp.size_bytes, 5);
    }

    #[test]
    fn test_response_agent_new() {
        let config = ResponseAgentConfig::default();
        let agent = ResponseAgent::new(config);
        assert_eq!(agent.name(), "ResponseAgent");
        assert_eq!(agent.agent_type(), "response");
        assert_eq!(agent.status(), AgentStatus::Initializing);
    }

    #[test]
    fn test_create_text_response() {
        let agent = ResponseAgent::new(ResponseAgentConfig::default());
        let resp = agent.create_text_response("test content".into(), None);
        assert_eq!(resp.format, "text");
        assert_eq!(resp.content, "test content");
        assert_eq!(resp.size_bytes, 12);
    }

    #[test]
    fn test_create_json_response() {
        let agent = ResponseAgent::new(ResponseAgentConfig::default());
        let data = serde_json::json!({"key": "value"});
        let resp = agent.create_json_response(data, None).unwrap();
        assert_eq!(resp.format, "json");
        assert!(resp.size_bytes > 0);
    }

    #[test]
    fn test_create_markdown_response() {
        let agent = ResponseAgent::new(ResponseAgentConfig::default());
        let resp = agent.create_markdown_response("# Header".into(), None);
        assert_eq!(resp.format, "markdown");
        assert_eq!(resp.content, "# Header");
    }

    #[test]
    fn test_create_html_response() {
        let agent = ResponseAgent::new(ResponseAgentConfig::default());
        let resp = agent.create_html_response("<p>Hello</p>".into(), None);
        assert_eq!(resp.format, "html");
        assert!(resp.content.contains("<!DOCTYPE html>"));
        assert!(resp.content.contains("<p>Hello</p>"));
    }

    #[test]
    fn test_create_error_response() {
        let agent = ResponseAgent::new(ResponseAgentConfig::default());
        let resp = agent.create_error_response("not found".into(), Some(404));
        assert_eq!(resp.format, "text");
        assert!(resp.content.contains("404"));
        assert!(resp.content.contains("not found"));
    }

    #[test]
    fn test_create_error_response_no_code() {
        let agent = ResponseAgent::new(ResponseAgentConfig::default());
        let resp = agent.create_error_response("fail".into(), None);
        assert!(resp.content.contains("fail"));
        assert!(!resp.content.contains("Error: fail") == false);
    }

    #[test]
    fn test_create_streaming_response() {
        let agent = ResponseAgent::new(ResponseAgentConfig::default());
        let chunks = vec!["Hello ".into(), "World".into()];
        let resp = agent.create_streaming_response(chunks).unwrap();
        assert_eq!(resp.content, "Hello World");
        assert_eq!(
            resp.metadata.get("chunk_count"),
            Some(&serde_json::json!(2))
        );
    }

    #[test]
    fn test_validate_response_empty_content() {
        let agent = ResponseAgent::new(ResponseAgentConfig::default());
        let resp = FormattedResponse {
            content: "   ".into(),
            format: "json".into(),
            metadata: HashMap::new(),
            size_bytes: 3,
            timestamp: chrono::Utc::now(),
        };
        let valid = agent.validate_response(&resp).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_response_stats() {
        let agent = ResponseAgent::new(ResponseAgentConfig::default());
        let stats = agent.get_response_stats();
        assert_eq!(stats.default_format, "json");
        assert!(stats.caching_enabled);
    }

    #[test]
    fn test_get_available_formats_empty() {
        let agent = ResponseAgent::new(ResponseAgentConfig::default());
        let formats = agent.get_available_formats();
        assert!(formats.is_empty());
    }
}
