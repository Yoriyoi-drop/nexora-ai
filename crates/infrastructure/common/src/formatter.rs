//! Logging Formatter Utilities
//! 
//! Custom formatting for log messages.

use tracing_subscriber::fmt::format;
use std::fmt::Write;

/// Custom formatter configuration
#[derive(Debug, Clone)]
pub struct FormatterConfig {
    /// Whether to include timestamps
    pub timestamps: bool,
    /// Timestamp format
    pub timestamp_format: TimestampFormat,
    /// Whether to include target/module
    pub target: bool,
    /// Whether to include level
    pub level: bool,
    /// Whether to include thread IDs
    pub thread_ids: bool,
    /// Whether to include span names
    pub span_names: bool,
    /// Whether to include file and line
    pub file_line: bool,
}

/// Timestamp format options
#[derive(Debug, Clone, Copy)]
pub enum TimestampFormat {
    /// RFC3339 format (ISO 8601)
    Rfc3339,
    /// Unix timestamp
    Unix,
    /// Local time format
    Local,
    /// Custom format string (uses local time as fallback)
    Custom,
}

impl Default for FormatterConfig {
    fn default() -> Self {
        Self {
            timestamps: true,
            timestamp_format: TimestampFormat::Rfc3339,
            target: true,
            level: true,
            thread_ids: false,
            span_names: false,
            file_line: false,
        }
    }
}

/// Custom formatter for tracing
pub struct CustomFormatter {
    config: FormatterConfig,
}

impl CustomFormatter {
    /// Create new custom formatter
    pub fn new(config: FormatterConfig) -> Self {
        Self { config }
    }
    
    /// Format timestamp
    fn format_timestamp(&self, buf: &mut String) -> Result<(), std::fmt::Error> {
        if !self.config.timestamps {
            return Ok(());
        }
        
        let now = std::time::SystemTime::now();
        let timestamp = match self.config.timestamp_format {
            TimestampFormat::Rfc3339 => {
                chrono::DateTime::<chrono::Utc>::from(now)
                    .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                    .to_string()
            }
            TimestampFormat::Unix => {
                now.duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
                    .to_string()
            }
            TimestampFormat::Local => {
                chrono::DateTime::<chrono::Local>::from(now)
                    .format("%Y-%m-%d %H:%M:%S%.3f")
                    .to_string()
            }
            TimestampFormat::Custom => {
                let custom_format = "%Y-%m-%d %H:%M:%S%.3f";
                chrono::DateTime::<chrono::Local>::from(now)
                    .format(custom_format)
                    .to_string()
            }
        };
        
        write!(buf, "{}", timestamp)
    }
    
    /// Format log level
    fn format_level(&self, level: &tracing::Level, buf: &mut String) -> Result<(), std::fmt::Error> {
        if !self.config.level {
            return Ok(());
        }
        
        let level_str = match level {
            &tracing::Level::TRACE => "TRACE",
            &tracing::Level::DEBUG => "DEBUG",
            &tracing::Level::INFO  => "INFO ",
            &tracing::Level::WARN  => "WARN ",
            &tracing::Level::ERROR => "ERROR",
        };
        
        write!(buf, "{}", level_str)
    }
    
    /// Format target/module
    fn format_target(&self, target: &str, buf: &mut String) -> Result<(), std::fmt::Error> {
        if !self.config.target {
            return Ok(());
        }
        
        write!(buf, "{}", target)
    }
    
    /// Format thread ID
    fn format_thread_id(&self, buf: &mut String) -> Result<(), std::fmt::Error> {
        if !self.config.thread_ids {
            return Ok(());
        }
        
        let thread_id = std::thread::current().id();
        write!(buf, "{:?}", thread_id)
    }
    
    /// Format file and line
    fn format_file_line(&self, file: Option<&str>, line: Option<u32>, buf: &mut String) -> Result<(), std::fmt::Error> {
        if !self.config.file_line {
            return Ok(());
        }
        
        match (file, line) {
            (Some(f), Some(l)) => write!(buf, "{}:{}", f, l),
            (Some(f), None) => write!(buf, "{}", f),
            (None, Some(l)) => write!(buf, "line:{}", l),
            (None, None) => Ok(()),
        }
    }
}

impl Default for CustomFormatter {
    fn default() -> Self {
        Self::new(FormatterConfig::default())
    }
}

/// Create custom formatter with default config
pub fn custom_formatter() -> CustomFormatter {
    CustomFormatter::default()
}

/// Create custom formatter with custom config
pub fn custom_formatter_with_config(config: FormatterConfig) -> CustomFormatter {
    CustomFormatter::new(config)
}

/// Compact formatter for production
pub fn compact_formatter() -> CustomFormatter {
    CustomFormatter::new(FormatterConfig {
        timestamps: false,
        timestamp_format: TimestampFormat::Unix,
        target: true,
        level: true,
        thread_ids: false,
        span_names: false,
        file_line: false,
    })
}

/// Detailed formatter for development
pub fn detailed_formatter() -> CustomFormatter {
    CustomFormatter::new(FormatterConfig {
        timestamps: true,
        timestamp_format: TimestampFormat::Rfc3339,
        target: true,
        level: true,
        thread_ids: true,
        span_names: true,
        file_line: true,
    })
}
