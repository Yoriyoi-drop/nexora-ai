//! Nexora-AI Main Library - Pure Rust Entry Point
//! 
//! Main entry point for Nexora AI system - delegates to specialized crates

use anyhow::Result;
use std::sync::{Arc, RwLock};
use tracing::{info, debug};
use chrono::Utc;

pub mod cli;
pub mod config;
pub mod server;
pub mod api;
pub mod core;

pub use cli::Cli;
pub use config::NexoraConfig;
pub use server::{NexoraServer, ServerConfig};
pub use api::{NexoraApi, ApiConfig, ApiResponse};
pub use core::*;

/// Nexora AI System - Delegates to specialized crates
/// 
/// This is the main entry point that coordinates all Nexora components
/// but delegates actual implementation to specialized crates.

#[derive(Debug, Clone)]
pub struct NexoraAI {
    models: Arc<RwLock<Vec<String>>>,
    config: NexoraConfig,
    memory_manager: Arc<RwLock<Option<String>>>,
    start_time: chrono::DateTime<Utc>,
    system_info_cache: Arc<RwLock<Option<SystemInfo>>>,
    request_count: Arc<RwLock<u64>>,
    system_monitor: SystemMonitor,
    text_generator: TextGenerator,
    chat_engine: ChatEngine,
    request_processor: RequestProcessor,
}

impl NexoraAI {
    /// Create new Nexora AI instance
    pub async fn new(config: NexoraConfig) -> Result<Self> {
        info!("Initializing Nexora AI system with delegated components...");
        
        // Validate configuration
        config.validate()?;
        
        // Initialize system monitoring
        let mut system = sysinfo::System::new_all();
        system.refresh_all();
        
        debug!("System initialized with {} cores, {}MB total memory", 
               system.cpus().len(), 
               system.total_memory() / (1024 * 1024));
        
        let models = Arc::new(RwLock::new(vec!["default".to_string()]));
        let system_info_cache = Arc::new(RwLock::new(None));
        let request_count = Arc::new(RwLock::new(0));
        
        let system_monitor = SystemMonitor::new(
            models.clone(),
            config.clone(),
            Utc::now(),
            system_info_cache.clone(),
            request_count.clone(),
        );
        
        Ok(Self {
            models,
            config,
            memory_manager: Arc::new(RwLock::new(None)),
            start_time: Utc::now(),
            system_info_cache,
            request_count: request_count.clone(),
            system_monitor,
            text_generator: TextGenerator::new(),
            chat_engine: ChatEngine::new(),
            request_processor: RequestProcessor::new(request_count.clone()),
        })
    }
    
    /// Get system information - delegates to monitoring crate
    pub async fn get_system_info(&self) -> Result<SystemInfo> {
        self.system_monitor.get_system_info().await
    }
    
    /// Health check - delegates to monitoring crate
    pub async fn health_check(&self) -> Result<HealthStatus> {
        self.system_monitor.health_check().await
    }
    
    /// Get performance metrics - delegates to monitoring crate
    pub async fn get_performance_metrics(&self) -> Result<serde_json::Value> {
        // Simple performance metrics implementation
        let system_info = self.get_system_info().await?;
        Ok(serde_json::json!({
            "cpu_usage": system_info.cpu_usage,
            "memory_usage": system_info.memory_usage,
            "uptime": system_info.uptime,
            "active_models": system_info.active_models.len(),
            "timestamp": system_info.last_updated
        }))
    }
    
    /// Process a request - delegates to inference crate
    pub async fn process_request(&self, input: &str) -> Result<String> {
        self.request_processor.process_request(input).await
    }
    
    /// Generate text - delegates to inference crate
    pub async fn generate_text(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        self.text_generator.generate_text(prompt, max_tokens, temperature).await
    }
    
    /// Chat conversation - delegates to inference crate
    pub async fn chat(&self, message: &str, conversation_id: Option<String>) -> Result<String> {
        self.chat_engine.chat(message, conversation_id).await
    }
    
    /// Analyze code - delegates to models crate
    pub async fn analyze_code(&self, code: &str, language: &str) -> Result<String> {
        info!("Analyzing {} code ({} chars)", language, code.len());
        
        let analysis = self.request_processor.analyze_code(code).await?;
        Ok(format!("Language: {}, Lines: {}, Functions: {}, Classes: {}", 
                  analysis.language, analysis.line_count, analysis.functions.len(), analysis.classes.len()))
    }
    
    /// Generate code - delegates to models crate
    pub async fn generate_code(&self, description: &str, language: &str) -> Result<String> {
        info!("Generating {} code from description: {}", language, description);
        
        // Simple code generation - would delegate to models crate
        let code = match language.to_lowercase().as_str() {
            "rust" => format!("// Generated Rust code\nfn {}() {{\n    // Implementation\n}}", description.replace(" ", "_")),
            "python" => format!("# Generated Python code\ndef {}():\n    # Implementation\n    pass", description.replace(" ", "_")),
            "javascript" => format!("// Generated JavaScript code\nfunction {}() {{\n    // Implementation\n}}", description.replace(" ", "_")),
            _ => format!("// Generated {} code\n// {} implementation", language, description),
        };
        
        Ok(code)
    }
    
    /// Get training model - placeholder implementation
    pub async fn get_training_model(&self) -> Result<String> {
        // Placeholder - would return actual training model
        Ok("placeholder_model".to_string())
    }
}
