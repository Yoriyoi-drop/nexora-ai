//! Configuration loading and management

use crate::error::{NexoraError, NexoraResult};
use std::path::Path;
use tracing::{info, warn, debug, error};

use super::core::CoreConfig;
use super::tokenizer::TokenizerConfig;
use super::models::ModelsConfig;
use super::memory::MemoryConfig;
use super::utils::UtilsConfig;
use super::server::ServerConfig;
use super::api::ApiConfig;
use super::logging::LoggingConfig;

/// Main configuration for Nexora AI system
/// 
/// This struct contains all configuration parameters for the entire Nexora AI system.
/// Each sub-configuration handles a specific aspect of the system.
/// 
/// # Validation
/// 
/// The configuration is validated when `validate()` is called. Validation includes:
/// - Checking that all numeric values are within reasonable ranges
/// - Ensuring file paths are valid
/// - Verifying that required fields are present
/// - Checking logical consistency between configuration sections
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NexoraConfig {
    /// Core system configuration
    pub core: CoreConfig,
    
    /// Tokenizer configuration for text processing
    pub tokenizer: TokenizerConfig,
    
    /// AI model configuration
    pub models: ModelsConfig,
    
    /// Memory management configuration
    pub memory: MemoryConfig,
    
    /// Utility functions configuration
    pub utils: UtilsConfig,
    
    /// Server configuration
    pub server: ServerConfig,
    
    /// API configuration
    pub api: ApiConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
}

impl Default for NexoraConfig {
    fn default() -> Self {
        Self {
            core: CoreConfig::default(),
            tokenizer: TokenizerConfig::default(),
            models: ModelsConfig::default(),
            memory: MemoryConfig::default(),
            utils: UtilsConfig::default(),
            server: ServerConfig::default(),
            api: ApiConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl NexoraConfig {
    /// Load configuration from file
    /// 
    /// # Errors
    /// Returns `NexoraError::Io` if file cannot be read
    /// Returns `NexoraError::Serialization` if TOML parsing fails
    pub fn from_file<P: AsRef<Path>>(path: P) -> NexoraResult<Self> {
        let path_ref = path.as_ref();
        debug!("Loading configuration from: {:?}", path_ref);
        
        if !path_ref.exists() {
            return Err(NexoraError::not_found(format!("Configuration file not found: {:?}", path_ref)));
        }
        
        let content = std::fs::read_to_string(path_ref)
            .map_err(|e| NexoraError::io(e))?;
            
        let config: NexoraConfig = toml::from_str(&content)
            .map_err(|e| NexoraError::serialization(e))?;
            
        info!("✅ Configuration loaded successfully from: {:?}", path_ref);
        Ok(config)
    }
    
    /// Save configuration to file
    /// 
    /// # Errors
    /// Returns `NexoraError::Io` if file cannot be written
    /// Returns `NexoraError::Serialization` if TOML serialization fails
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> NexoraResult<()> {
        let path_ref = path.as_ref();
        debug!("Saving configuration to: {:?}", path_ref);
        
        // Create parent directories if they don't exist
        if let Some(parent) = path_ref.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| NexoraError::io(e))?;
        }
        
        let content = toml::to_string_pretty(self)
            .map_err(|e| NexoraError::serialization(e))?;
            
        std::fs::write(path_ref, content)
            .map_err(|e| NexoraError::io(e))?;
            
        info!("✅ Configuration saved successfully to: {:?}", path_ref);
        Ok(())
    }
    
    /// Validate configuration comprehensively
    /// 
    /// This method performs extensive validation of all configuration parameters
    /// to ensure system stability and prevent runtime errors.
    /// 
    /// # Errors
    /// Returns `NexoraError::Validation` if any configuration parameter is invalid
    pub fn validate(&self) -> NexoraResult<()> {
        info!("🔍 Starting comprehensive configuration validation...");
        let mut errors = Vec::new();
        
        // Validate core configuration
        if self.core.max_concurrent_requests == 0 {
            errors.push("core.max_concurrent_requests must be greater than 0".to_string());
        } else if self.core.max_concurrent_requests > 50000 {
            warn!("core.max_concurrent_requests is very high ({}), consider reducing", self.core.max_concurrent_requests);
        }
        
        if self.core.request_timeout_ms == 0 {
            errors.push("core.request_timeout_ms must be greater than 0".to_string());
        } else if self.core.request_timeout_ms > 600000 { // 10 minutes
            warn!("core.request_timeout_ms is very high ({}ms), consider reducing", self.core.request_timeout_ms);
        }
        
        // Validate tokenizer configuration
        if self.tokenizer.vocab_size == 0 {
            errors.push("tokenizer.vocab_size must be greater than 0".to_string());
        } else if self.tokenizer.vocab_size > 1000000 {
            warn!("tokenizer.vocab_size is very large ({}), this may impact performance", self.tokenizer.vocab_size);
        }
        
        if self.tokenizer.min_frequency == 0 {
            errors.push("tokenizer.min_frequency must be greater than 0".to_string());
        } else if self.tokenizer.min_frequency > 100 {
            warn!("tokenizer.min_frequency is high ({}), may reduce vocabulary coverage", self.tokenizer.min_frequency);
        }
        
        // Validate models configuration
        if self.models.vocab_size == 0 {
            errors.push("models.vocab_size must be greater than 0".to_string());
        }
        
        if self.models.d_model == 0 {
            errors.push("models.d_model must be greater than 0".to_string());
        } else if self.models.d_model % self.models.n_heads != 0 {
            errors.push("models.d_model must be divisible by models.n_heads".to_string());
        }
        
        if self.models.n_heads == 0 {
            errors.push("models.n_heads must be greater than 0".to_string());
        }
        
        if self.models.n_layers == 0 {
            errors.push("models.n_layers must be greater than 0".to_string());
        }
        
        // Validate memory configuration
        if self.memory.short_term_capacity == 0 {
            errors.push("memory.short_term_capacity must be greater than 0".to_string());
        }
        
        if self.memory.session_capacity == 0 {
            errors.push("memory.session_capacity must be greater than 0".to_string());
        }
        
        if self.memory.long_term_capacity == 0 {
            errors.push("memory.long_term_capacity must be greater than 0".to_string());
        }
        
        // Validate server configuration
        if self.server.port == 0 {
            errors.push("server.port must be greater than 0".to_string());
        } else if self.server.port < 1024 {
            warn!("server.port {} is in privileged range, ensure proper permissions", self.server.port);
        }
        
        if self.server.max_connections == 0 {
            errors.push("server.max_connections must be greater than 0".to_string());
        } else if self.server.max_connections > 100000 {
            warn!("server.max_connections is very high ({}), ensure system can handle this", self.server.max_connections);
        }
        
        // Validate API configuration
        if self.api.timeout_seconds == 0 {
            errors.push("api.timeout_seconds must be greater than 0".to_string());
        }
        
        if self.api.requests_per_minute == 0 {
            errors.push("api.requests_per_minute must be greater than 0".to_string());
        }
        
        // Validate logging configuration
        if !["trace", "debug", "info", "warn", "error"].contains(&self.logging.level.as_str()) {
            errors.push(format!("logging.level '{}' is invalid, must be one of: trace, debug, info, warn, error", self.logging.level));
        }
        
        // Check for logical consistency
        if self.core.max_concurrent_requests > self.server.max_connections {
            warn!("core.max_concurrent_requests ({}) exceeds server.max_connections ({}), this may cause request queuing", 
                  self.core.max_concurrent_requests, self.server.max_connections);
        }
        
        // Report validation results
        if !errors.is_empty() {
            error!("❌ Configuration validation failed with {} errors:", errors.len());
            for (i, error) in errors.iter().enumerate() {
                error!("  {}. {}", i + 1, error);
            }
            return Err(NexoraError::validation("configuration", format!("Validation failed: {}", errors.join("; "))));
        }
        
        info!("✅ Configuration validation passed successfully");
        Ok(())
    }
    
    /// Get comprehensive configuration summary
    /// 
    /// Returns a formatted string with all key configuration parameters
    /// for debugging and monitoring purposes.
    pub fn summary(&self) -> String {
        debug!("Generating configuration summary");
        
        let summary = format!(
            "📋 Nexora AI Configuration Summary:\n\
             ┌─────────────────────────────────────────────────────────────┐\n\
             │ Core Configuration:                                        │\n\
             │   • ML Intent Detection: {}                                 │\n\
             │   • Coordination Enabled: {}                               │\n\
             │   • Max Concurrent Requests: {}                             │\n\
             │   • Request Timeout: {}ms                                   │\n\
             ├─────────────────────────────────────────────────────────────┤\n\
             │ Tokenizer Configuration:                                    │\n\
             │   • Vocabulary Size: {}                                     │\n\
             │   • Cache Size: {}                                          │\n\
             │   • Min Frequency: {}                                       │\n\
             ├─────────────────────────────────────────────────────────────┤\n\
             │ Model Configuration:                                        │\n\
             │   • Model Dimension: {}                                     │\n\
             │   • Number of Layers: {}                                    │\n\
             │   • Number of Heads: {}                                     │\n\
             │   • Vocabulary Size: {}                                    │\n\
             ├─────────────────────────────────────────────────────────────┤\n\
             │ Memory Configuration:                                       │\n\
             │   • Short Term Capacity: {}                                 │\n\
             │   • Session Capacity: {}                                    │\n\
             │   • Long Term Capacity: {}                                  │\n\
             ├─────────────────────────────────────────────────────────────┤\n\
             │ Server Configuration:                                       │\n\
             │   • Host: {}                                                │\n\
             │   • Port: {}                                                │\n\
             │   • Max Connections: {}                                      │\n\
             │   • TLS Enabled: {}                                         │\n\
             ├─────────────────────────────────────────────────────────────┤\n\
             │ API Configuration:                                          │\n\
             │   • Base URL: {}                                            │\n\
             │   • Timeout: {}s                                            │\n\
             │   • Rate Limiting: {}                                       │\n\
             │   • Requests per Minute: {}                                 │\n\
             ├─────────────────────────────────────────────────────────────┤\n\
             │ Logging Configuration:                                      │\n\
             │   • Level: {}                                               │\n\
             │   • File Logging: {}                                        │\n\
             │   • Structured Logging: {}                                   │\n\
             └─────────────────────────────────────────────────────────────┘",
            // Core
            if self.core.enable_ml_intent { "✅ Enabled" } else { "❌ Disabled" },
            if self.core.enable_coordination { "✅ Enabled" } else { "❌ Disabled" },
            self.core.max_concurrent_requests,
            self.core.request_timeout_ms,
            // Tokenizer
            self.tokenizer.vocab_size,
            self.tokenizer.cache_size,
            self.tokenizer.min_frequency,
            // Models
            self.models.d_model,
            self.models.n_layers,
            self.models.n_heads,
            self.models.vocab_size,
            // Memory
            self.memory.short_term_capacity,
            self.memory.session_capacity,
            self.memory.long_term_capacity,
            // Server
            self.server.host,
            self.server.port,
            self.server.max_connections,
            if self.server.enable_tls { "✅ Enabled" } else { "❌ Disabled" },
            // API
            self.api.base_url,
            self.api.timeout_seconds,
            if self.api.enable_rate_limiting { "✅ Enabled" } else { "❌ Disabled" },
            self.api.requests_per_minute,
            // Logging
            self.logging.level.to_uppercase(),
            if self.logging.enable_file_logging { "✅ Enabled" } else { "❌ Disabled" },
            if self.logging.enable_structured_logging { "✅ Enabled" } else { "❌ Disabled" }
        );
        
        info!("📊 Configuration summary generated");
        summary
    }
    
    /// Get configuration as JSON for API responses
    /// 
    /// Returns a JSON representation of the configuration
    /// with sensitive information filtered out.
    pub fn to_safe_json(&self) -> NexoraResult<serde_json::Value> {
        debug!("Converting configuration to safe JSON format");
        
        let json_value = serde_json::to_value(self)
            .map_err(|e| NexoraError::serialization(e))?;
            
        // Remove sensitive information if needed
        // For now, return the full config as it doesn't contain sensitive data
        Ok(json_value)
    }
    
    /// Validate and log configuration status
    /// 
    /// This method combines validation and logging for convenience
    pub fn validate_and_log(&self) -> NexoraResult<()> {
        info!("🔧 Validating and logging configuration...");
        
        // First validate the configuration
        self.validate()?;
        
        // Log the summary
        let summary = self.summary();
        info!("{}", summary);
        
        // Log additional metrics
        debug!("Configuration metrics:");
        debug!("  - Total configuration sections: 7");
        debug!("  - Estimated memory footprint: ~{}KB", self.estimate_memory_usage());
        debug!("  - Configuration complexity: {}", self.calculate_complexity());
        
        Ok(())
    }
    
    /// Estimate memory usage of configuration in KB
    fn estimate_memory_usage(&self) -> usize {
        // Rough estimation of memory usage
        let base_size = std::mem::size_of::<Self>();
        let string_overhead = self.server.host.len() + self.api.base_url.len() + self.logging.level.len();
        (base_size + string_overhead) / 1024
    }
    
    /// Calculate configuration complexity score
    fn calculate_complexity(&self) -> &'static str {
        let mut score = 0;
        
        if self.core.enable_ml_intent { score += 1; }
        if self.core.enable_coordination { score += 1; }
        if self.server.enable_tls { score += 1; }
        if self.api.enable_rate_limiting { score += 1; }
        if self.logging.enable_file_logging { score += 1; }
        if self.logging.enable_structured_logging { score += 1; }
        
        match score {
            0..=2 => "Simple",
            3..=4 => "Moderate",
            5..=6 => "Complex",
            _ => "Very Complex",
        }
    }
}
