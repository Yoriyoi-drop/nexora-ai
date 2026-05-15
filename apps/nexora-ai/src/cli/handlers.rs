//! Command handlers for CLI operations

use crate::error::{NexoraError, NexoraResult};
use std::path::PathBuf;
use tracing::{info, warn, error};

use crate::{NexoraAI, NexoraConfig};
use super::commands::{Cli, Commands, ConfigAction, TokenizerAction, MemoryAction};

impl Cli {
    /// Run CLI application
    pub async fn run(&self) -> NexoraResult<()> {
        // Initialize logging
        self.init_logging();
        
        // Load configuration
        let mut config = if self.config.exists() {
            match NexoraConfig::from_file(&self.config) {
                Ok(config) => config,
                Err(e) => {
                    warn!("Failed to load configuration: {}", e);
                    return Err(NexoraError::config(format!("Configuration load failed: {}", e)));
                }
            }
        } else {
            warn!("Configuration file not found, using defaults");
            NexoraConfig::default()
        };
        
        // Override config with CLI arguments
        config = self.override_config(config);
        
        // Validate configuration
        if let Err(e) = config.validate() {
            return Err(NexoraError::config(format!("Configuration validation failed: {}", e)));
        }
        
        // Initialize Nexora AI
        let nexora = match NexoraAI::new(config).await {
            Ok(ai) => ai,
            Err(e) => {
                error!("Failed to initialize Nexora AI: {}", e);
                return Err(NexoraError::initialization(format!("Nexora AI initialization failed: {}", e)));
            }
        };
        
        // Execute command
        match &self.command {
            Commands::Start { host, port, tls, cert_path, key_path } => {
                self.run_server(&nexora, host, *port, *tls, cert_path, key_path).await
            }
            Commands::Process { input, format, output } => {
                self.run_process(&nexora, input, format, output).await
            }
            Commands::Generate { prompt, max_tokens, temperature, output } => {
                self.run_generate(&nexora, prompt, *max_tokens, *temperature, output).await
            }
            Commands::Chat { interactive, message, conversation_id, history_file } => {
                self.run_chat(&nexora, *interactive, message, conversation_id, history_file).await
                    .map_err(|e| NexoraError::processing(format!("Chat command failed: {}", e)))
            }
            Commands::Analyze { file, language, format, output } => {
                self.run_analyze(&nexora, file, language, format, output).await
            }
            Commands::Codegen { description, language, output } => {
                self.run_codegen(&nexora, description, language, output).await
            }
            Commands::Train { data, output, epochs, batch_size, learning_rate, gpu } => {
                self.run_train(&nexora, data, output, *epochs, *batch_size, *learning_rate, *gpu).await
                    .map_err(|e| NexoraError::processing(format!("Train command failed: {}", e)))
            }
            Commands::Evaluate { model, test_data, output } => {
                self.run_evaluate(&nexora, model, test_data, output).await
                    .map_err(|e| NexoraError::processing(format!("Evaluate command failed: {}", e)))
            }
            Commands::Info { performance, memory, models, format } => {
                self.run_info(&nexora, *performance, *memory, *models, format).await
            }
            Commands::Health { detailed } => {
                self.run_health(&nexora, *detailed).await
            }
            Commands::Config { action } => {
                self.run_config(action).await
            }
            Commands::Tokenizer { action } => {
                self.run_tokenizer(action).await
            }
            Commands::Memory { action } => {
                self.run_memory(&nexora, action).await
            }
        }
    }
    
    /// Initialize logging
    fn init_logging(&self) {
        use tracing_subscriber::util::SubscriberInitExt;
        
        let level = match self.log_level.to_lowercase().as_str() {
            "trace" => tracing::Level::TRACE,
            "debug" => tracing::Level::DEBUG,
            "info" => tracing::Level::INFO,
            "warn" => tracing::Level::WARN,
            "error" => tracing::Level::ERROR,
            _ => tracing::Level::INFO,
        };
        
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(level)
            .with_ansi(true)
            .finish();
        
        subscriber.init();
        
        if self.verbose {
            info!("Verbose logging enabled");
        }
    }
    
    /// Override configuration with CLI arguments
    fn override_config(&self, mut config: NexoraConfig) -> NexoraConfig {
        match &self.command {
            Commands::Start { host, port, tls, cert_path, key_path } => {
                config.server.host = host.clone();
                config.server.port = *port;
                config.server.enable_tls = *tls;
                if let Some(cert) = cert_path {
                    config.server.cert_path = Some(cert.to_string_lossy().to_string());
                }
                if let Some(key) = key_path {
                    config.server.key_path = Some(key.to_string_lossy().to_string());
                }
            }
            _ => {}
        }
        
        config
    }
    
    /// Write output to file or stdout
    fn write_output(&self, content: &str, output: &Option<PathBuf>) -> NexoraResult<()> {
        match output {
            Some(path) => {
                std::fs::write(path, content)
                    .map_err(|e| NexoraError::io(e))?;
                info!("Output written to: {:?}", path);
            }
            None => {
                println!("{}", content);
            }
        }
        Ok(())
    }
    
    /// Run server command
    async fn run_server(
        &self,
        _nexora: &NexoraAI,
        host: &str,
        port: u16,
        tls: bool,
        cert_path: &Option<PathBuf>,
        key_path: &Option<PathBuf>,
    ) -> NexoraResult<()> {
        info!("Starting Nexora AI server on {}:{}", host, port);
        
        let config = crate::ServerConfig {
            host: host.to_string(),
            port,
            enable_tls: tls,
            cert_path: cert_path.as_ref().map(|p| p.to_string_lossy().to_string()),
            key_path: key_path.as_ref().map(|p| p.to_string_lossy().to_string()),
            max_connections: 1000,
            request_timeout_seconds: 30,
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
        };
        
        let server = crate::NexoraServer::new(config);
        
        server.start().await
            .map_err(|e| NexoraError::system(format!("Server start failed: {}", e)))
    }
    
    /// Run process command
    async fn run_process(
        &self,
        nexora: &NexoraAI,
        input: &str,
        format: &str,
        output: &Option<PathBuf>,
    ) -> NexoraResult<()> {
        info!("Processing input: {}", input);
        
        let response = nexora.process_request(input).await
            .map_err(|e| NexoraError::processing(format!("Request processing failed: {}", e)))?;
        
        match format {
            "json" => {
                let json_output = serde_json::json!({
                    "input": input,
                    "response": response,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                });
                let output_str = serde_json::to_string_pretty(&json_output)
                    .map_err(|e| NexoraError::serialization(e))?;
                self.write_output(&output_str, output)?;
            }
            "text" => {
                self.write_output(&response, output)?;
            }
            _ => {
                return Err(NexoraError::validation("format", format!("Unsupported output format: {}", format)));
            }
        }
        
        Ok(())
    }
    
    /// Run generate command
    async fn run_generate(
        &self,
        nexora: &NexoraAI,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        output: &Option<PathBuf>,
    ) -> NexoraResult<()> {
        info!("Generating text from prompt: {}", prompt);
        
        let generated = nexora.generate_text(prompt, max_tokens, temperature).await
            .map_err(|e| NexoraError::model(format!("Text generation failed: {}", e)))?;
        self.write_output(&generated, output)?;
        
        Ok(())
    }
    
    /// Run analyze command
    async fn run_analyze(
        &self,
        nexora: &NexoraAI,
        file: &PathBuf,
        language: &str,
        format: &str,
        output: &Option<PathBuf>,
    ) -> NexoraResult<()> {
        info!("Analyzing {} code in file: {:?}", language, file);
        
        let code = std::fs::read_to_string(file)
            .map_err(|e| NexoraError::Io { source: e })?;
        let analysis = nexora.analyze_code(&code, language).await?;
        
        match format {
            "json" => {
                let json_output = serde_json::json!({
                    "file": file.to_string_lossy(),
                    "language": language,
                    "analysis": analysis,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                });
                let output_str = serde_json::to_string_pretty(&json_output)
                    .map_err(|e| NexoraError::serialization(e))?;
                self.write_output(&output_str, output)?;
            }
            "text" => {
                self.write_output(&analysis, output)?;
            }
            _ => {
                return Err(NexoraError::validation("format", format!("Unsupported output format: {}", format)));
            }
        }
        
        Ok(())
    }
    
    /// Run codegen command
    async fn run_codegen(
        &self,
        nexora: &NexoraAI,
        description: &str,
        language: &str,
        output: &Option<PathBuf>,
    ) -> NexoraResult<()> {
        info!("Generating {} code from description: {}", language, description);
        
        let code = nexora.generate_code(description, language).await?;
        self.write_output(&code, output)?;
        
        Ok(())
    }
    
    /// Run info command
    async fn run_info(
        &self,
        nexora: &NexoraAI,
        performance: bool,
        memory: bool,
        models: bool,
        format: &str,
    ) -> NexoraResult<()> {
        let mut info_text = String::new();
        
        if performance {
            let system_info = nexora.get_system_info().await
                    .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))?;
            info_text.push_str(&format!("Performance Metrics:\n"));
            info_text.push_str(&format!("  CPU Usage: {:.1}%\n", system_info.cpu_usage));
            info_text.push_str(&format!("  Memory Usage: {:.1}%\n", system_info.memory_usage));
            info_text.push_str(&format!("  Process Count: {}\n", system_info.process_count));
            info_text.push_str(&format!("  Thread Count: {}\n", system_info.thread_count));
            info_text.push_str(&format!("  Load Average: {:.2}, {:.2}, {:.2}\n\n", 
                system_info.load_average.0, system_info.load_average.1, system_info.load_average.2));
        }
        
        if memory {
            let system_info = nexora.get_system_info().await
                    .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))?;
            info_text.push_str(&format!("Memory Statistics:\n"));
            info_text.push_str(&format!("  Total Memory: {} MB\n", 
                system_info.memory_stats.total_memory / (1024 * 1024)));
            info_text.push_str(&format!("  Used Memory: {} MB\n", 
                system_info.memory_stats.used_memory / (1024 * 1024)));
            info_text.push_str(&format!("  Available Memory: {} MB\n", 
                system_info.memory_stats.available_memory / (1024 * 1024)));
            info_text.push_str(&format!("  Cache Size: {} MB\n\n", 
                system_info.memory_stats.cache_size / (1024 * 1024)));
        }
        
        if models {
            let system_info = nexora.get_system_info().await
                    .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))?;
            info_text.push_str(&format!("Model Information:\n"));
            info_text.push_str(&format!("  Active Models: {}\n", system_info.active_models.join(", ")));
            info_text.push_str(&format!("  Version: {}\n\n", system_info.version));
        }
        
        if info_text.is_empty() {
            info_text = "No specific information requested. Use --performance, --memory, or --models flags.".to_string();
        }
        
        match format {
            "json" => {
                let json_output = serde_json::json!({
                    "info": info_text,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                });
                let output_str = serde_json::to_string_pretty(&json_output)
                    .map_err(|e| NexoraError::serialization(e))?;
                println!("{}", output_str);
            }
            "text" => {
                println!("{}", info_text);
            }
            _ => {
                return Err(NexoraError::validation("format", format!("Unsupported output format: {}", format)));
            }
        }
        
        Ok(())
    }
    
    /// Run health command
    async fn run_health(&self, nexora: &NexoraAI, detailed: bool) -> NexoraResult<()> {
        let health = nexora.health_check().await
            .map_err(|e| NexoraError::system(format!("Health check failed: {}", e)))?;
        
        println!("System Health: {}", if health.healthy { "✓ Healthy" } else { "✗ Unhealthy" });
        println!("Performance Score: {:.1}/100", health.performance_score);
        println!("Uptime: {} seconds", health.uptime_seconds);
        println!("Total Operations: {}", health.total_operations);
        println!("Average Response Time: {:.2} ms", health.average_response_time);
        println!("Error Rate: {:.2}%", health.error_rate * 100.0);
        println!("Active Connections: {}", health.active_connections);
        println!("Last Check: {}", health.last_check.to_rfc3339());
        
        if detailed {
            println!("\nComponent Health:");
            for (component, healthy) in &health.component_health {
                println!("  {}: {}", component, if *healthy { "✓" } else { "✗" });
            }
        }
        
        Ok(())
    }
    
    /// Run config command
    async fn run_config(&self, action: &ConfigAction) -> NexoraResult<()> {
        match action {
            ConfigAction::Show => {
                if self.config.exists() {
                    let content = std::fs::read_to_string(&self.config)
                    .map_err(|e| NexoraError::Io { source: e })?;
                    println!("{}", content);
                } else {
                    println!("Configuration file not found: {:?}", self.config);
                }
            }
            ConfigAction::Validate => {
                if self.config.exists() {
                    let config = NexoraConfig::from_file(&self.config)
                    .map_err(|e| NexoraError::config(format!("Failed to load config for validation: {}", e)))?;
                    config.validate()?;
                    println!("Configuration is valid ✓");
                } else {
                    println!("Configuration file not found: {:?}", self.config);
                }
            }
            ConfigAction::Generate { output } => {
                let default_config = NexoraConfig::default();
                let config_str = toml::to_string_pretty(&default_config)
                    .map_err(|e| NexoraError::serialization(e))?;
                std::fs::write(output, config_str)
                    .map_err(|e| NexoraError::Io { source: e })?;
                println!("Default configuration generated: {:?}", output);
            }
            ConfigAction::Update { key, value } => {
                println!("Config update not implemented yet: {} = {}", key, value);
            }
        }
        Ok(())
    }
    
    /// Run tokenizer command
    async fn run_tokenizer(&self, action: &TokenizerAction) -> NexoraResult<()> {
        match action {
            TokenizerAction::Train { data, output, vocab_size, min_frequency } => {
                println!("Tokenizer training not implemented yet:");
                println!("  Data: {:?}", data);
                println!("  Output: {:?}", output);
                println!("  Vocab size: {}", vocab_size);
                println!("  Min frequency: {}", min_frequency);
            }
            TokenizerAction::Test { text, detailed } => {
                println!("Tokenizer testing not implemented yet:");
                println!("  Text: {}", text);
                println!("  Detailed: {}", detailed);
            }
            TokenizerAction::Info { model } => {
                println!("Tokenizer info not implemented yet:");
                println!("  Model: {:?}", model);
            }
        }
        Ok(())
    }
    
    /// Run memory command
    async fn run_memory(&self, nexora: &NexoraAI, action: &MemoryAction) -> NexoraResult<()> {
        match action {
            MemoryAction::Stats { detailed } => {
                let system_info = nexora.get_system_info().await
                    .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))?;
                println!("Memory Statistics:");
                println!("  Total Memory: {} MB", 
                    system_info.memory_stats.total_memory / (1024 * 1024));
                println!("  Used Memory: {} MB", 
                    system_info.memory_stats.used_memory / (1024 * 1024));
                println!("  Available Memory: {} MB", 
                    system_info.memory_stats.available_memory / (1024 * 1024));
                println!("  Usage: {:.1}%", system_info.memory_usage);
                
                if *detailed {
                    println!("  Cache Size: {} MB", 
                        system_info.memory_stats.cache_size / (1024 * 1024));
                    println!("  Component Status: {}", system_info.components.memory);
                }
            }
            MemoryAction::Clear { layer } => {
                println!("Memory clear not implemented yet for layer: {}", layer);
            }
            MemoryAction::Export { output, format } => {
                println!("Memory export not implemented yet:");
                println!("  Output: {:?}", output);
                println!("  Format: {}", format);
            }
            MemoryAction::Import { input, format } => {
                println!("Memory import not implemented yet:");
                println!("  Input: {:?}", input);
                println!("  Format: {}", format);
            }
        }
        Ok(())
    }
}
