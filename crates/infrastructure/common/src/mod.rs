//! Modular Logging System for Nexora AI
//! 
//! Split dari common/src/logging.rs (988 baris) menjadi modular components.

pub mod config;
pub mod console;
pub mod file;
pub mod formatter;
pub mod filter;
pub mod init;

// Re-export main components
pub use config::*;
pub use console::*;
pub use file::*;
pub use formatter::*;
pub use filter::*;
pub use init::*;

use anyhow::Result;
use tracing::{Level, Subscriber};
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Registry,
};

/// Main logging system
pub struct LoggingSystem {
    config: LoggingConfig,
    subscriber: Option<Box<dyn Subscriber + Send + Sync>>,
}

impl LoggingSystem {
    /// Create new logging system
    pub fn new(config: LoggingConfig) -> Self {
        Self {
            config,
            subscriber: None,
        }
    }
    
    /// Initialize logging system
    pub fn init(&mut self) -> Result<()> {
        let subscriber = self.build_subscriber()?;
        subscriber.init();
        self.subscriber = Some(Box::new(subscriber));
        Ok(())
    }
    
    /// Build subscriber from configuration
    fn build_subscriber(&self) -> Result<impl Subscriber> {
        let env_filter = self.build_env_filter()?;
        let subscriber = Registry::default().with(env_filter);
        
        let subscriber = if self.config.console {
            let console_layer = self.build_console_layer()?;
            subscriber.with(console_layer)
        } else {
            Box::new(subscriber)
        };
        
        let subscriber = if self.config.file {
            let file_layer = self.build_file_layer()?;
            subscriber.with(file_layer)
        } else {
            subscriber
        };
        
        Ok(subscriber)
    }
    
    /// Build environment filter
    fn build_env_filter(&self) -> Result<EnvFilter> {
        let level = self.parse_log_level(&self.config.level)?;
        let mut filter = EnvFilter::from_default_env()
            .add_directive(level.into());
        
        // Add custom filters
        for filter_pattern in &self.config.filters {
            filter = filter.add_directive(filter_pattern.parse()?);
        }
        
        Ok(filter)
    }
    
    /// Parse log level string
    fn parse_log_level(&self, level: &str) -> Result<Level> {
        match level.to_lowercase().as_str() {
            "trace" => Ok(Level::TRACE),
            "debug" => Ok(Level::DEBUG),
            "info" => Ok(Level::INFO),
            "warn" => Ok(Level::WARN),
            "error" => Ok(Level::ERROR),
            _ => Err(anyhow::anyhow!("Invalid log level: {}", level)),
        }
    }
    
    /// Build console layer
    fn build_console_layer(&self) -> Result<Box<dyn tracing_subscriber::Layer<Registry> + Send + Sync>> {
        let console_layer = tracing_subscriber::fmt::layer()
            .with_target(self.config.target)
            .with_thread_ids(self.config.thread_ids)
            .with_span_events(if self.config.span_events { 
                tracing_subscriber::fmt::format::FmtSpan::CLOSE 
            } else { 
                tracing_subscriber::fmt::format::FmtSpan::NONE 
            })
            .with_ansi(cfg!(feature = "color"))
            .with_writer(std::io::stdout);
        
        if self.config.json_format {
            Ok(Box::new(console_layer.json()))
        } else {
            Ok(Box::new(console_layer))
        }
    }
    
    /// Build file layer
    fn build_file_layer(&self) -> Result<Box<dyn tracing_subscriber::Layer<Registry> + Send + Sync>> {
        use tracing_appender::{non_blocking, rolling};
        
        let file_dir = self.config.file_dir.clone()
            .unwrap_or_else(|| std::path::PathBuf::from("logs"));
        
        let file_appender = rolling::daily(&file_dir, &self.config.file_name_prefix);
        let (non_blocking, _guard) = non_blocking(file_appender);
        
        let file_layer = tracing_subscriber::fmt::layer()
            .with_target(self.config.target)
            .with_thread_ids(self.config.thread_ids)
            .with_span_events(if self.config.span_events { 
                tracing_subscriber::fmt::format::FmtSpan::CLOSE 
            } else { 
                tracing_subscriber::fmt::format::FmtSpan::NONE 
            })
            .with_ansi(false)
            .with_writer(non_blocking);
        
        if self.config.json_format {
            Ok(Box::new(file_layer.json()))
        } else {
            Ok(Box::new(file_layer))
        }
    }
    
    /// Get current configuration
    pub fn config(&self) -> &LoggingConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: LoggingConfig) -> Result<()> {
        self.config = config;
        self.init()
    }
}

impl Default for LoggingSystem {
    fn default() -> Self {
        Self::new(LoggingConfig::default())
    }
}
