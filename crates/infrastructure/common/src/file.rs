//! File Logging Utilities
//! 
//! File-specific logging functionality.

use std::path::PathBuf;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing::Subscriber;

/// File logging configuration
#[derive(Debug, Clone)]
pub struct FileConfig {
    /// File output directory
    pub file_dir: PathBuf,
    /// File name prefix
    pub file_name_prefix: String,
    /// Maximum file size before rotation (in MB)
    pub max_file_size_mb: u64,
    /// Maximum number of rotated files to keep
    pub max_files: usize,
    /// Whether to use JSON format
    pub json_format: bool,
    /// Whether to show timestamps
    pub show_timestamps: bool,
    /// Whether to show target/module
    pub show_target: bool,
    /// Whether to show thread IDs
    pub show_thread_ids: bool,
    /// Whether to show span events
    pub show_span_events: bool,
}

impl Default for FileConfig {
    fn default() -> Self {
        Self {
            file_dir: std::path::PathBuf::from("logs"),
            file_name_prefix: "nexora".to_string(),
            max_file_size_mb: 100,
            max_files: 10,
            json_format: false,
            show_timestamps: true,
            show_target: true,
            show_thread_ids: false,
            show_span_events: true,
        }
    }
}

/// File layer builder
pub struct FileLayerBuilder {
    config: FileConfig,
}

impl FileLayerBuilder {
    /// Create new file layer builder
    pub fn new() -> Self {
        Self {
            config: FileConfig::default(),
        }
    }
    
    /// Set file directory
    pub fn with_dir<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.config.file_dir = dir.into();
        self
    }
    
    /// Set file name prefix
    pub fn with_prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.config.file_name_prefix = prefix.into();
        self
    }
    
    /// Set maximum file size
    pub fn with_max_size(mut self, size_mb: u64) -> Self {
        self.config.max_file_size_mb = size_mb;
        self
    }
    
    /// Set maximum number of files
    pub fn with_max_files(mut self, max_files: usize) -> Self {
        self.config.max_files = max_files;
        self
    }
    
    /// Set JSON format
    pub fn with_json(mut self, json_format: bool) -> Self {
        self.config.json_format = json_format;
        self
    }
    
    /// Set timestamps visibility
    pub fn with_timestamps(mut self, show: bool) -> Self {
        self.config.show_timestamps = show;
        self
    }
    
    /// Set target visibility
    pub fn with_target(mut self, show: bool) -> Self {
        self.config.show_target = show;
        self
    }
    
    /// Set thread IDs visibility
    pub fn with_thread_ids(mut self, show: bool) -> Self {
        self.config.show_thread_ids = show;
        self
    }
    
    /// Set span events visibility
    pub fn with_span_events(mut self, show: bool) -> Self {
        self.config.show_span_events = show;
        self
    }
    
    /// Build file layer
    pub fn build(self) -> Result<Box<dyn tracing_subscriber::Layer<tracing_subscriber::Registry> + Send + Sync>, std::io::Error> {
        use tracing_appender::{non_blocking, rolling};
        
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&self.config.file_dir)?;
        
        let file_appender = rolling::daily(&self.config.file_dir, &self.config.file_name_prefix);
        let (non_blocking, _guard) = non_blocking(file_appender);
        
        let layer = fmt::layer()
            .with_target(self.config.show_target)
            .with_thread_ids(self.config.show_thread_ids)
            .with_span_events(if self.config.show_span_events { 
                fmt::format::FmtSpan::CLOSE 
            } else { 
                fmt::format::FmtSpan::NONE 
            })
            .with_ansi(false)
            .with_writer(non_blocking);
        
        if self.config.json_format {
            Ok(Box::new(layer.json()))
        } else {
            Ok(Box::new(layer))
        }
    }
}

impl Default for FileLayerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Create file layer with default config
pub fn file_layer() -> Result<Box<dyn tracing_subscriber::Layer<tracing_subscriber::Registry> + Send + Sync>, std::io::Error> {
    FileLayerBuilder::default().build()
}

/// Create file layer with custom config
pub fn file_layer_with_config(config: FileConfig) -> Result<Box<dyn tracing_subscriber::Layer<tracing_subscriber::Registry> + Send + Sync>, std::io::Error> {
    FileLayerBuilder::new()
        .with_dir(config.file_dir)
        .with_prefix(config.file_name_prefix)
        .with_max_size(config.max_file_size_mb)
        .with_max_files(config.max_files)
        .with_json(config.json_format)
        .with_timestamps(config.show_timestamps)
        .with_target(config.show_target)
        .with_thread_ids(config.show_thread_ids)
        .with_span_events(config.show_span_events)
        .build()
}
