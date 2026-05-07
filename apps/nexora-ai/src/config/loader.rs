//! Configuration loading and management

use anyhow::Result;
use std::path::Path;
use tracing::info;

use super::core::CoreConfig;
use super::tokenizer::TokenizerConfig;
use super::models::ModelsConfig;
use super::memory::MemoryConfig;
use super::utils::UtilsConfig;
use super::server::ServerConfig;
use super::api::ApiConfig;
use super::logging::LoggingConfig;

/// Main configuration for Nexora AI system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NexoraConfig {
    pub core: CoreConfig,
    pub tokenizer: TokenizerConfig,
    pub models: ModelsConfig,
    pub memory: MemoryConfig,
    pub utils: UtilsConfig,
    pub server: ServerConfig,
    pub api: ApiConfig,
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
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: NexoraConfig = toml::from_str(&content)?;
        info!("Configuration loaded from file");
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        info!("Configuration saved to file");
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate core configuration
        if self.core.max_concurrent_requests == 0 {
            return Err(anyhow::anyhow!("max_concurrent_requests must be greater than 0"));
        }
        
        if self.core.request_timeout_ms == 0 {
            return Err(anyhow::anyhow!("request_timeout_ms must be greater than 0"));
        }
        
        // Validate tokenizer configuration
        if self.tokenizer.vocab_size == 0 {
            return Err(anyhow::anyhow!("vocab_size must be greater than 0"));
        }
        
        if self.tokenizer.min_frequency == 0 {
            return Err(anyhow::anyhow!("min_frequency must be greater than 0"));
        }
        
        // Validate models configuration
        if self.models.vocab_size == 0 {
            return Err(anyhow::anyhow!("models.vocab_size must be greater than 0"));
        }
        
        if self.models.d_model == 0 {
            return Err(anyhow::anyhow!("d_model must be greater than 0"));
        }
        
        if self.models.n_heads == 0 {
            return Err(anyhow::anyhow!("n_heads must be greater than 0"));
        }
        
        if self.models.n_layers == 0 {
            return Err(anyhow::anyhow!("n_layers must be greater than 0"));
        }
        
        // Validate memory configuration
        if self.memory.short_term_capacity == 0 {
            return Err(anyhow::anyhow!("short_term_capacity must be greater than 0"));
        }
        
        if self.memory.session_capacity == 0 {
            return Err(anyhow::anyhow!("session_capacity must be greater than 0"));
        }
        
        // Validate server configuration
        if self.server.port == 0 {
            return Err(anyhow::anyhow!("port must be greater than 0"));
        }
        
        if self.server.max_connections == 0 {
            return Err(anyhow::anyhow!("max_connections must be greater than 0"));
        }
        
        // Validate API configuration
        if self.api.timeout_seconds == 0 {
            return Err(anyhow::anyhow!("timeout_seconds must be greater than 0"));
        }
        
        if self.api.requests_per_minute == 0 {
            return Err(anyhow::anyhow!("requests_per_minute must be greater than 0"));
        }
        
        info!("Configuration validation passed");
        Ok(())
    }
    
    /// Get configuration summary
    pub fn summary(&self) -> String {
        format!(
            "Nexora Configuration Summary:\n\
             - Core: ML intent={}, Coordination={}, Max requests={}\n\
             - Tokenizer: Vocab size={}, Cache size={}\n\
             - Models: d_model={}, n_layers={}, n_heads={}\n\
             - Memory: Short term={}, Session={}, Long term={}\n\
             - Server: {}:{}, TLS={}\n\
             - API: {}, Rate limiting={}\n\
             - Logging: Level={}, File={}",
            self.core.enable_ml_intent,
            self.core.enable_coordination,
            self.core.max_concurrent_requests,
            self.tokenizer.vocab_size,
            self.tokenizer.cache_size,
            self.models.d_model,
            self.models.n_layers,
            self.models.n_heads,
            self.memory.short_term_capacity,
            self.memory.session_capacity,
            self.memory.long_term_capacity,
            self.server.host,
            self.server.port,
            self.server.enable_tls,
            self.api.base_url,
            self.api.enable_rate_limiting,
            self.logging.level,
            self.logging.enable_file_logging
        )
    }
}
