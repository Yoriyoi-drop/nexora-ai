//! Nexora-AI Main Library - Pure Rust Entry Point
//! 
//! Main entry point for Nexora AI system - delegates to specialized crates

use crate::error::{NexoraError, NexoraResult};
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{info, debug, warn, error};
use chrono::Utc;

pub mod cli;
pub mod config;
pub mod server;
pub mod api;
pub mod core;
pub mod error;
pub mod security;

pub use cli::Cli;
pub use config::NexoraConfig;
pub use server::{NexoraServer, ServerConfig};
pub use api::{NexoraApi, ApiConfig, ApiResponse};
pub use core::*;
pub use security::*;

/// Nexora AI System - Main orchestrator for all AI components
/// 
/// This is the central coordinator that manages all Nexora AI components including:
/// - Model management and loading
/// - Request processing and routing
/// - System monitoring and health checks
/// - Text generation and chat functionality
/// - Code analysis and generation
/// 
/// The system follows a delegation pattern where this main struct coordinates
/// but delegates actual implementation to specialized crates for better modularity.
/// 
/// # Examples
/// 
/// ```rust,no_run
/// use nexora_ai::{NexoraAI, NexoraConfig};
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = NexoraConfig::default();
///     let ai = NexoraAI::new(config).await?;
///     
///     let response = ai.chat("Hello, world!", None).await?;
///     println!("{}", response);
///     Ok(())
/// }
/// ```
/// 
/// # Thread Safety
/// 
/// This struct is thread-safe and can be shared across multiple threads using `Arc`.
/// All internal state is protected by `Arc<RwLock<T>>` for safe concurrent access.
#[derive(Debug, Clone)]
pub struct NexoraAI {
    /// Available AI models for inference
    models: Arc<RwLock<Vec<String>>>,
    
    /// System configuration
    config: NexoraConfig,
    
    /// Memory manager for handling different memory layers
    memory_manager: Arc<RwLock<Option<String>>>,
    
    /// System startup timestamp for uptime tracking
    start_time: chrono::DateTime<Utc>,
    
    /// Cached system information to avoid frequent system calls
    system_info_cache: Arc<RwLock<Option<SystemInfo>>>,
    
    /// Total request counter for metrics
    request_count: Arc<AtomicU64>,
    
    /// System monitoring and health check component
    system_monitor: SystemMonitor,
    
    /// Text generation engine for various text tasks
    text_generator: TextGenerator,
    
    /// Chat engine for conversational AI interactions
    chat_engine: ChatEngine,
    
    /// Request processor for handling incoming requests
    request_processor: RequestProcessor,
}

impl NexoraAI {
    /// Create new Nexora AI instance
    /// 
    /// # Errors
    /// Returns `NexoraError::Config` if configuration validation fails
    /// Returns `NexoraError::Initialization` if system initialization fails
    pub async fn new(config: NexoraConfig) -> NexoraResult<Self> {
        info!("🚀 Initializing Nexora AI system with delegated components...");
        
        // Validate configuration
        config.validate().map_err(|e| NexoraError::config(format!("Configuration validation failed: {}", e)))?;
        
        // Initialize system monitoring
        let mut system = sysinfo::System::new_all();
        system.refresh_all();
        
        debug!("System initialized with {} cores, {}MB total memory", 
               system.cpus().len(), 
               system.total_memory() / (1024 * 1024));
        
        let models = Arc::new(RwLock::new(vec!["default".to_string()]));
        let system_info_cache = Arc::new(RwLock::new(None));
        let request_count = Arc::new(AtomicU64::new(0));
        
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
    /// 
    /// # Errors
    /// Returns `NexoraError::System` if system information retrieval fails
    pub async fn get_system_info(&self) -> NexoraResult<SystemInfo> {
        self.system_monitor.get_system_info().await
            .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))
    }
    
    /// Health check - delegates to monitoring crate
    /// 
    /// # Errors
    /// Returns `NexoraError::System` if health check fails
    pub async fn health_check(&self) -> NexoraResult<HealthStatus> {
        self.system_monitor.health_check().await
            .map_err(|e| NexoraError::system(format!("Health check failed: {}", e)))
    }
    
    /// Get performance metrics - delegates to monitoring crate
    /// 
    /// # Errors
    /// Returns `NexoraError::System` if metrics collection fails
    pub async fn get_performance_metrics(&self) -> NexoraResult<serde_json::Value> {
        let system_info = self.get_system_info().await?;
        Ok(serde_json::json!({
            "cpu_usage": system_info.cpu_usage,
            "memory_usage": system_info.memory_usage,
            "uptime": system_info.uptime,
            "active_models": system_info.active_models.len(),
            "timestamp": system_info.last_updated,
            "request_count": self.request_count.load(Ordering::Relaxed),
            "uptime_seconds": (Utc::now() - self.start_time).num_seconds()
        }))
    }
    
    /// Process a request - delegates to inference crate
    /// 
    /// # Errors
    /// Returns `NexoraError::Processing` if request processing fails
    pub async fn process_request(&self, input: &str) -> NexoraResult<String> {
        if input.trim().is_empty() {
            return Err(NexoraError::validation("input", "Input cannot be empty"));
        }
        
        info!("Processing request: {} chars", input.len());
        self.request_processor.process_request(input).await
            .map_err(|e| NexoraError::processing(format!("Request processing failed: {}", e)))
    }
    
    /// Generate text - delegates to inference crate
    /// 
    /// # Errors
    /// Returns `NexoraError::Model` if text generation fails
    pub async fn generate_text(&self, prompt: &str, max_tokens: usize, temperature: f32) -> NexoraResult<String> {
        if prompt.trim().is_empty() {
            return Err(NexoraError::validation("prompt", "Prompt cannot be empty"));
        }
        
        if max_tokens == 0 {
            return Err(NexoraError::validation("max_tokens", "Max tokens must be greater than 0"));
        }
        
        if !(0.0..=2.0).contains(&temperature) {
            return Err(NexoraError::validation("temperature", "Temperature must be between 0.0 and 2.0"));
        }
        
        info!("Generating text: prompt={}, max_tokens={}, temperature={}", 
              prompt.len(), max_tokens, temperature);
              
        self.text_generator.generate_text(prompt, max_tokens, temperature).await
            .map_err(|e| NexoraError::model(format!("Text generation failed: {}", e)))
    }
    
    /// Chat conversation - delegates to inference crate
    /// 
    /// # Errors
    /// Returns `NexoraError::Model` if chat processing fails
    pub async fn chat(&self, message: &str, conversation_id: Option<String>) -> NexoraResult<String> {
        if message.trim().is_empty() {
            return Err(NexoraError::validation("message", "Message cannot be empty"));
        }
        
        info!("Chat message: {} chars, conversation_id: {:?}", 
              message.len(), conversation_id);
              
        self.chat_engine.chat(message, conversation_id).await
            .map_err(|e| NexoraError::model(format!("Chat processing failed: {}", e)))
    }
    
    /// Analyze code - delegates to models crate
    /// 
    /// # Errors
    /// Returns `NexoraError::Model` if code analysis fails
    pub async fn analyze_code(&self, code: &str, language: &str) -> NexoraResult<String> {
        if code.trim().is_empty() {
            return Err(NexoraError::validation("code", "Code cannot be empty"));
        }
        
        if language.trim().is_empty() {
            return Err(NexoraError::validation("language", "Language cannot be empty"));
        }
        
        info!("Analyzing {} code ({} chars)", language, code.len());
        
        let analysis = self.request_processor.analyze_code(code).await
            .map_err(|e| NexoraError::model(format!("Code analysis failed: {}", e)))?;
            
        Ok(format!("Language: {}, Lines: {}, Functions: {}, Classes: {}", 
                  analysis.language, analysis.line_count, analysis.functions.len(), analysis.classes.len()))
    }
    
    /// Generate code - delegates to models crate
    /// 
    /// # Errors
    /// Returns `NexoraError::Model` if code generation fails
    pub async fn generate_code(&self, description: &str, language: &str) -> NexoraResult<String> {
        if description.trim().is_empty() {
            return Err(NexoraError::validation("description", "Description cannot be empty"));
        }
        
        if language.trim().is_empty() {
            return Err(NexoraError::validation("language", "Language cannot be empty"));
        }
        
        info!("Generating {} code from description: {}", language, description);
        
        // Enhanced code generation with better templates
        let safe_description = description.replace(|c: char| !c.is_alphanumeric() && c != ' ', "_");
        let code = match language.to_lowercase().as_str() {
            "rust" => format!(
                "/// Generated Rust code\n/// Description: {}\npub fn {}() -> Result<(), Box<dyn std::error::Error>> {{\n    println!(\"Generated function called for: {}\");\n    Ok(())\n}}",
                description, safe_description, description
            ),
            "python" => format!(
                "# Generated Python code\n# Description: {}\ndef {}():\n    \"\"\"Generated function.\"\"\"\n    print(\"Generated function called for: {}\")\n    return True",
                description, safe_description, description
            ),
            "javascript" => format!(
                "// Generated JavaScript code\n// Description: {}\nfunction {}() {{\n    console.log('Generated function called for: {}');\n    return true;\n}}",
                description, safe_description, description
            ),
            "typescript" => format!(
                "// Generated TypeScript code\n// Description: {}\nfunction {}(): boolean {{\n    console.log('Generated function called for: {}');\n    return true;\n}}",
                description, safe_description, description
            ),
            "go" => format!(
                "// Generated Go code\n// Description: {}\npackage main\n\nimport \"fmt\"\n\nfunc {}() {{\n    fmt.Println(\"Generated function called for: {}\")\n}}",
                description, safe_description, description
            ),
            _ => format!(
                "// Generated {} code\n// Description: {}\nfunction {}() {{\n    console.log('Generated function called');\n    return true;\n}}",
                language, description, safe_description
            ),
        };
        
        Ok(code)
    }
    
    /// Get training model information
    /// 
    /// Retrieves the currently configured training model for the system.
    /// This method checks the available models and returns the first one found.
    /// 
    /// # Returns
    /// 
    /// Returns the name of the training model as a `String`.
    /// 
    /// # Errors
    /// 
    /// - `NexoraError::Model` - If no training models are configured
    /// - `NexoraError::System` - If there's an error accessing the models list
    /// 
    /// # Examples
    /// 
    /// ```rust,no_run
    /// use nexora_ai::{NexoraAI, NexoraConfig};
    /// 
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let config = NexoraConfig::default();
    ///     let ai = NexoraAI::new(config).await?;
    ///     
    ///     match ai.get_training_model().await {
    ///         Ok(model) => println!("Training model: {}", model),
    ///         Err(e) => eprintln!("Error getting training model: {}", e),
    ///     }
    ///     
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_training_model(&self) -> NexoraResult<String> {
        info!("Retrieving training model information...");
        
        // Check if we have any models configured
        let models = self.models.read()
            .map_err(|e| NexoraError::system(format!("Failed to acquire read lock for models: {}", e)))?;
            
        if models.is_empty() {
            return Err(NexoraError::not_found("No training models configured"));
        }
        
        let model_name = models.first().unwrap().clone();
        info!("Found training model: {}", model_name);
        Ok(model_name)
    }
}
