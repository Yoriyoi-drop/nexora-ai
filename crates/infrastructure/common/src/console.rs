//! Console Logging Utilities
//! 
//! Console-specific logging functionality.

use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing::Subscriber;

/// Console logging configuration
#[derive(Debug, Clone)]
pub struct ConsoleConfig {
    /// Whether to use colors
    pub use_colors: bool,
    /// Whether to show timestamps
    pub show_timestamps: bool,
    /// Whether to show target/module
    pub show_target: bool,
    /// Whether to show thread IDs
    pub show_thread_ids: bool,
    /// Whether to show span events
    pub show_span_events: bool,
    /// Whether to use JSON format
    pub json_format: bool,
}

impl Default for ConsoleConfig {
    fn default() -> Self {
        Self {
            use_colors: cfg!(feature = "color"),
            show_timestamps: true,
            show_target: true,
            show_thread_ids: false,
            show_span_events: true,
            json_format: false,
        }
    }
}

/// Console layer builder
pub struct ConsoleLayerBuilder {
    config: ConsoleConfig,
}

impl ConsoleLayerBuilder {
    /// Create new console layer builder
    pub fn new() -> Self {
        Self {
            config: ConsoleConfig::default(),
        }
    }
    
    /// Set color usage
    pub fn with_colors(mut self, use_colors: bool) -> Self {
        self.config.use_colors = use_colors;
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
    
    /// Set JSON format
    pub fn with_json(mut self, json_format: bool) -> Self {
        self.config.json_format = json_format;
        self
    }
    
    /// Build console layer
    pub fn build(self) -> Box<dyn tracing_subscriber::Layer<tracing_subscriber::Registry> + Send + Sync> {
        let layer = fmt::layer()
            .with_target(self.config.show_target)
            .with_thread_ids(self.config.show_thread_ids)
            .with_span_events(if self.config.show_span_events { 
                fmt::format::FmtSpan::CLOSE 
            } else { 
                fmt::format::FmtSpan::NONE 
            })
            .with_ansi(self.config.use_colors)
            .with_writer(std::io::stdout);
        
        if self.config.json_format {
            Box::new(layer.json())
        } else {
            Box::new(layer)
        }
    }
}

impl Default for ConsoleLayerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Create console layer with default config
pub fn console_layer() -> Box<dyn tracing_subscriber::Layer<tracing_subscriber::Registry> + Send + Sync> {
    ConsoleLayerBuilder::default().build()
}

/// Create console layer with custom config
pub fn console_layer_with_config(config: ConsoleConfig) -> Box<dyn tracing_subscriber::Layer<tracing_subscriber::Registry> + Send + Sync> {
    ConsoleLayerBuilder::new()
        .with_colors(config.use_colors)
        .with_timestamps(config.show_timestamps)
        .with_target(config.show_target)
        .with_thread_ids(config.show_thread_ids)
        .with_span_events(config.show_span_events)
        .with_json(config.json_format)
        .build()
}
