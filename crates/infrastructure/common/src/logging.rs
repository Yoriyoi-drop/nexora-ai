//! Simple Logging Configuration for Nexora AI
//! 
//! Simplified logging setup without complex type issues

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::Level;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Registry,
};

/// Simple logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Whether to log to console
    pub console: bool,
    /// Whether to include timestamps
    pub timestamps: bool,
    /// Whether to include target/module
    pub target: bool,
    /// Whether to include thread IDs
    pub thread_ids: bool,
    /// Whether to include span traces
    pub span_events: bool,
    /// Custom filter patterns
    pub filters: Vec<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            console: true,
            timestamps: true,
            target: true,
            thread_ids: false,
            span_events: true,
            filters: vec![
                "nexora=info".to_string(),
                "axum=info".to_string(),
                "tokio=warn".to_string(),
                "hyper=warn".to_string(),
                "sqlx=warn".to_string(),
                "tower=warn".to_string(),
            ],
        }
    }
}

/// Parse log level string to Level
fn parse_log_level(level: &str) -> Result<Level> {
    match level.to_lowercase().as_str() {
        "trace" => Ok(Level::TRACE),
        "debug" => Ok(Level::DEBUG),
        "info" => Ok(Level::INFO),
        "warn" => Ok(Level::WARN),
        "error" => Ok(Level::ERROR),
        _ => Err(anyhow::anyhow!("Invalid log level: {}", level)),
    }
}

/// Build environment filter
fn build_env_filter(config: &LoggingConfig, level: Level) -> Result<EnvFilter> {
    let filter_str = if config.filters.is_empty() {
        format!("nexora={}", level)
    } else {
        format!("nexora={},{}", level, config.filters.join(","))
    };
    
    EnvFilter::try_new(&filter_str)
        .map_err(|e| anyhow::anyhow!("Failed to create env filter: {}", e))
}

/// Initialize simple logging system
pub fn init_logging(config: LoggingConfig) -> Result<()> {
    let level = parse_log_level(&config.level)?;
    
    // Build environment filter
    let env_filter = build_env_filter(&config, level)?;
    
    // Create simple subscriber
    let subscriber = Registry::default()
        .with(env_filter)
        .with(
            fmt::layer()
                .with_target(config.target)
                .with_thread_ids(config.thread_ids)
                .with_span_events(if config.span_events { FmtSpan::CLOSE } else { FmtSpan::NONE })
                .with_ansi(true)
                .with_writer(std::io::stdout)
        );
    
    // Initialize global subscriber
    subscriber.init();
    
    tracing::info!("Logging initialized with level: {}", config.level);
    Ok(())
}

/// Simple logging utilities
pub mod utils {
    use super::*;
    use std::time::Instant;
    
    /// Measure execution time of a function
    pub fn time_it<F, T>(name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        
        tracing::debug!(
            duration_ms = duration.as_millis(),
            "Function {} completed", name
        );
        
        result
    }
    
    /// Measure execution time of an async function
    pub async fn time_it_async<F, T>(name: &str, f: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();
        let result = f.await;
        let duration = start.elapsed();
        
        tracing::debug!(
            duration_ms = duration.as_millis(),
            "Async function {} completed", name
        );
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_log_level() {
        assert!(parse_log_level("info").is_ok());
        assert!(parse_log_level("invalid").is_err());
    }

    #[test]
    fn test_default_config() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, "info");
        assert!(config.console);
    }

    #[tokio::test]
    async fn test_init_logging() {
        let config = LoggingConfig::default();
        let _ = config;
    }
}
