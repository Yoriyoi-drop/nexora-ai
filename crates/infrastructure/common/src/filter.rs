//! Logging Filter Utilities
//! 
//! Custom filtering for log messages.

use tracing_subscriber::filter::{EnvFilter, Filter};
use tracing::Level;

/// Filter configuration
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// Global log level
    pub level: Level,
    /// Module-specific filters
    pub module_filters: Vec<(String, Level)>,
    /// Target-specific filters
    pub target_filters: Vec<(String, Level)>,
    /// Custom filter directives
    pub directives: Vec<String>,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            level: Level::INFO,
            module_filters: vec![
                ("nexora".to_string(), Level::INFO),
                ("axum".to_string(), Level::WARN),
                ("tokio".to_string(), Level::WARN),
                ("hyper".to_string(), Level::WARN),
                ("sqlx".to_string(), Level::WARN),
                ("tower".to_string(), Level::WARN),
            ],
            target_filters: vec![],
            directives: vec![],
        }
    }
}

/// Filter builder
pub struct FilterBuilder {
    config: FilterConfig,
}

impl FilterBuilder {
    /// Create new filter builder
    pub fn new() -> Self {
        Self {
            config: FilterConfig::default(),
        }
    }
    
    /// Set global level
    pub fn with_level(mut self, level: Level) -> Self {
        self.config.level = level;
        self
    }
    
    /// Add module filter
    pub fn with_module_filter<S: Into<String>>(mut self, module: S, level: Level) -> Self {
        self.config.module_filters.push((module.into(), level));
        self
    }
    
    /// Add target filter
    pub fn with_target_filter<S: Into<String>>(mut self, target: S, level: Level) -> Self {
        self.config.target_filters.push((target.into(), level));
        self
    }
    
    /// Add custom directive
    pub fn with_directive<S: Into<String>>(mut self, directive: S) -> Self {
        self.config.directives.push(directive.into());
        self
    }
    
    /// Build env filter
    pub fn build(self) -> Result<EnvFilter, Box<dyn std::error::Error>> {
        let mut filter = EnvFilter::from_default_env()
            .add_directive(self.config.level.into());
        
        // Add module filters
        for (module, level) in self.config.module_filters {
            let directive = format!("{}={}", module, level);
            filter = filter.add_directive(directive.parse()?);
        }
        
        // Add target filters
        for (target, level) in self.config.target_filters {
            let directive = format!("{}={}", target, level);
            filter = filter.add_directive(directive.parse()?);
        }
        
        // Add custom directives
        for directive in self.config.directives {
            filter = filter.add_directive(directive.parse()?);
        }
        
        Ok(filter)
    }
}

impl Default for FilterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Create filter with default config
pub fn default_filter() -> Result<EnvFilter, Box<dyn std::error::Error>> {
    FilterBuilder::default().build()
}

/// Create filter with custom config
pub fn filter_with_config(config: FilterConfig) -> Result<EnvFilter, Box<dyn std::error::Error>> {
    FilterBuilder::new()
        .with_level(config.level)
        .with_module_filters(config.module_filters)
        .with_target_filters(config.target_filters)
        .with_directives(config.directives)
        .build()
}

/// Development filter (more verbose)
pub fn development_filter() -> Result<EnvFilter, Box<dyn std::error::Error>> {
    FilterBuilder::new()
        .with_level(Level::DEBUG)
        .with_module_filter("nexora", Level::DEBUG)
        .with_module_filter("axum", Level::DEBUG)
        .with_module_filter("tokio", Level::INFO)
        .with_module_filter("hyper", Level::INFO)
        .with_module_filter("sqlx", Level::INFO)
        .with_module_filter("tower", Level::INFO)
        .build()
}

/// Production filter (less verbose)
pub fn production_filter() -> Result<EnvFilter, Box<dyn std::error::Error>> {
    FilterBuilder::new()
        .with_level(Level::INFO)
        .with_module_filter("nexora", Level::INFO)
        .with_module_filter("axum", Level::WARN)
        .with_module_filter("tokio", Level::ERROR)
        .with_module_filter("hyper", Level::ERROR)
        .with_module_filter("sqlx", Level::WARN)
        .with_module_filter("tower", Level::WARN)
        .build()
}

/// Test filter (minimal output)
pub fn test_filter() -> Result<EnvFilter, Box<dyn std::error::Error>> {
    FilterBuilder::new()
        .with_level(Level::ERROR)
        .with_module_filter("nexora", Level::ERROR)
        .build()
}

/// Custom filter for AI operations
pub fn ai_operations_filter() -> Result<EnvFilter, Box<dyn std::error::Error>> {
    FilterBuilder::new()
        .with_level(Level::INFO)
        .with_module_filter("nexora", Level::DEBUG)
        .with_module_filter("nexora::inference", Level::INFO)
        .with_module_filter("nexora::models", Level::DEBUG)
        .with_module_filter("nexora::memory", Level::INFO)
        .with_module_filter("nexora::agent", Level::INFO)
        .with_module_filter("nexora::tensor", Level::TRACE)
        .with_directive("nexora::foundation=debug")
        .build()
}

impl FilterBuilder {
    /// Helper method to add multiple module filters
    fn with_module_filters(mut self, filters: Vec<(String, Level)>) -> Self {
        for (module, level) in filters {
            self.config.module_filters.push((module, level));
        }
        self
    }
    
    /// Helper method to add multiple target filters
    fn with_target_filters(mut self, filters: Vec<(String, Level)>) -> Self {
        for (target, level) in filters {
            self.config.target_filters.push((target, level));
        }
        self
    }
}
