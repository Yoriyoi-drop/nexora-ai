use serde::{Deserialize, Serialize};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub level: TracingLevel,
    pub format: TracingFormat,
    pub enable_file_sink: bool,
    pub file_path: Option<String>,
    pub enable_json: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            level: TracingLevel::Info,
            format: TracingFormat::Compact,
            enable_file_sink: false,
            file_path: None,
            enable_json: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TracingLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for TracingLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TracingLevel::Trace => write!(f, "trace"),
            TracingLevel::Debug => write!(f, "debug"),
            TracingLevel::Info => write!(f, "info"),
            TracingLevel::Warn => write!(f, "warn"),
            TracingLevel::Error => write!(f, "error"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TracingFormat {
    Compact,
    Pretty,
    Json,
}

pub fn init_tracing(config: &TracingConfig) -> Result<(), String> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("nexora={},info", config.level)));

    let subscriber = Registry::default().with(env_filter);

    match config.format {
        TracingFormat::Json => {
            let fmt_layer = tracing_subscriber::fmt::layer().json();
            if config.enable_file_sink {
                let appender = create_file_appender(config)?;
                let file_layer = tracing_subscriber::fmt::layer().json().with_writer(appender);
                subscriber.with(fmt_layer).with(file_layer).init();
            } else {
                subscriber.with(fmt_layer).init();
            }
        }
        TracingFormat::Pretty => {
            let fmt_layer = tracing_subscriber::fmt::layer().pretty();
            if config.enable_file_sink {
                let appender = create_file_appender(config)?;
                let file_layer = tracing_subscriber::fmt::layer().pretty().with_writer(appender);
                subscriber.with(fmt_layer).with(file_layer).init();
            } else {
                subscriber.with(fmt_layer).init();
            }
        }
        TracingFormat::Compact => {
            let fmt_layer = tracing_subscriber::fmt::layer().compact();
            if config.enable_file_sink {
                let appender = create_file_appender(config)?;
                let file_layer = tracing_subscriber::fmt::layer().compact().with_writer(appender);
                subscriber.with(fmt_layer).with(file_layer).init();
            } else {
                subscriber.with(fmt_layer).init();
            }
        }
    }

    tracing::info!("Tracing initialized at level {}", config.level);
    Ok(())
}

fn create_file_appender(config: &TracingConfig) -> Result<tracing_appender::rolling::RollingFileAppender, String> {
    let path = config.file_path.as_deref().unwrap_or("logs/nexora-trace.log");
    let parent = std::path::Path::new(path)
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or(".");
    let file_stem = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("nexora-trace");

    std::fs::create_dir_all(parent)
        .map_err(|e| format!("Failed to create log directory {}: {}", parent, e))?;

    Ok(tracing_appender::rolling::daily(parent, file_stem))
}
