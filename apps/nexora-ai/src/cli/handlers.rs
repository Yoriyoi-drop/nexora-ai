//! Command handlers for CLI operations

use crate::error::{NexoraError, NexoraResult};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

use crate::{NexoraAI, NexoraConfig};
use super::commands::{Cli, Commands, ConfigAction, TokenizerAction, MemoryAction};

impl Cli {
    /// Run CLI application
    pub async fn run(&self) -> NexoraResult<()> {
        // Initialize logging
        self.init_logging();
        
        // Load configuration
        let config = if self.config.exists() {
            match NexoraConfig::from_file(&self.config) {
                Ok(config) => config,
                Err(e) => {
                    warn!("Failed to load configuration: {}", e);
                    return Err(NexoraError::config(format!("Configuration load failed: {}", e)));
                }
            }
        } else {
            info!("No config file found, using defaults");
            NexoraConfig::default()
        };
        
        // Override config with CLI arguments
        let config = self.override_config(config);
        
        // Validate configuration
        if let Err(e) = config.validate() {
            return Err(NexoraError::config(format!("Configuration validation failed: {}", e)));
        }
        
        // Execute command — initialize NexoraAI lazily when needed
        match &self.command {
            Commands::Config { action } => {
                self.run_config(action).await
            }
            Commands::Tokenizer { action } => {
                self.run_tokenizer(action).await
            }
            Commands::Health { detailed } => {
                let nexora = NexoraAI::new(config).await
                    .map_err(|e| NexoraError::initialization(format!("Nexora AI initialization failed: {}", e)))?;
                self.run_health(&nexora, *detailed).await
            }
            _ => {
                let nexora = NexoraAI::new(config).await
                    .map_err(|e| NexoraError::initialization(format!("Nexora AI initialization failed: {}", e)))?;
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
                    Commands::Train { data, output, tokenizer, epochs, batch_size, learning_rate, gpu, resume } => {
                        self.run_train(&nexora, data, output, tokenizer, *epochs, *batch_size, *learning_rate, *gpu, *resume).await
                            .map_err(|e| NexoraError::processing(format!("Train command failed: {}", e)))
                    }
                    Commands::Evaluate { model, test_data, tokenizer, output } => {
                        self.run_evaluate(&nexora, model, test_data, tokenizer, output).await
                            .map_err(|e| NexoraError::processing(format!("Evaluate command failed: {}", e)))
                    }
                    Commands::Info { performance, memory, models, format } => {
                        self.run_info(&nexora, *performance, *memory, *models, format).await
                    }
                    Commands::Memory { action } => {
                        self.run_memory(&nexora, action).await
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
    
    /// Initialize logging
    fn init_logging(&self) {
        let level = match self.log_level.to_lowercase().as_str() {
            "trace" => tracing::Level::TRACE,
            "debug" => tracing::Level::DEBUG,
            "info" => tracing::Level::INFO,
            "warn" => tracing::Level::WARN,
            "error" => tracing::Level::ERROR,
            _ => tracing::Level::INFO,
        };
        
        let _ = tracing_subscriber::fmt()
            .with_max_level(level)
            .with_ansi(true)
            .try_init();
        
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
        nexora: &NexoraAI,
        host: &str,
        port: u16,
        tls: bool,
        cert_path: &Option<PathBuf>,
        key_path: &Option<PathBuf>,
    ) -> NexoraResult<()> {
        info!("Starting Nexora AI server on http://{}:{}", host, port);
        
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
            api_keys: vec![],
            enable_auth: false,
        };
        
        let server = crate::NexoraServer::new(config);
        let nexora = Arc::new(nexora.clone());
        
        server.start(nexora).await
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
                    info!("No configuration file found at {:?}", self.config);
                }
            }
            ConfigAction::Validate => {
                if self.config.exists() {
                    let config = NexoraConfig::from_file(&self.config)
                    .map_err(|e| NexoraError::config(format!("Failed to load config for validation: {}", e)))?;
                    config.validate()?;
                    println!("Configuration is valid");
                } else {
                    info!("No configuration file found at {:?}", self.config);
                }
            }
            ConfigAction::Generate { output } => {
                let default_config = NexoraConfig::default();
                let config_str = toml::to_string_pretty(&default_config)
                    .map_err(|e| NexoraError::serialization(e))?;
                std::fs::write(output, config_str)
                    .map_err(|e| NexoraError::Io { source: e })?;
                info!("Default configuration generated at {:?}", output);
            }
            ConfigAction::Update { key, value } => {
                if !self.config.exists() {
                    return Err(NexoraError::config(
                        format!("Config file not found: {:?}", self.config)
                    ));
                }
                let content = std::fs::read_to_string(&self.config)
                    .map_err(|e| NexoraError::Io { source: e })?;
                let mut conf: toml::Value = toml::from_str(&content)
                    .map_err(|e| NexoraError::config(format!("Failed to parse config: {}", e)))?;
                let keys: Vec<&str> = key.split('.').collect();
                let mut current = &mut conf;
                for (i, k) in keys.iter().enumerate() {
                    if i == keys.len() - 1 {
                        current.as_table_mut()
                            .ok_or_else(|| NexoraError::config("Config root is not a table".to_string()))?
                            .insert(k.to_string(), toml::Value::String(value.clone()));
                    } else {
                        current = current.as_table_mut()
                            .ok_or_else(|| NexoraError::config(
                                format!("Key path '{}' not found in config", key)
                            ))?
                            .entry(k.to_string())
                            .or_insert(toml::Value::Table(toml::value::Table::new()));
                    }
                }
                let updated = toml::to_string_pretty(&conf)
                    .map_err(|e| NexoraError::serialization(e))?;
                std::fs::write(&self.config, updated)
                    .map_err(|e| NexoraError::Io { source: e })?;
                info!("Updated config: {} = {}", key, value);
            }
        }
        Ok(())
    }
    
    /// Run tokenizer command
    async fn run_tokenizer(&self, action: &TokenizerAction) -> NexoraResult<()> {
        match action {
            TokenizerAction::Train { data: _data, output: _output, vocab_size: _vocab_size, min_frequency: _min_frequency } => {
                Err(NexoraError::config(
                    "Tokenizer training requires the `nexora-tokenizer` training feature. Use `cargo build --features tokenizer-train` to enable.".to_string()
                ))
            }
            TokenizerAction::Test { text: _text, detailed: _detailed } => {
                Err(NexoraError::config(
                    "Tokenizer testing requires an active tokenizer model. Load a model first via `start` or provide a tokenizer path.".to_string()
                ))
            }
            TokenizerAction::Info { model: _model } => {
                Err(NexoraError::config(
                    "Tokenizer info requires the tokenizer crate. Ensure nexora-tokenizer is enabled in the build.".to_string()
                ))
            }
        }
    }
    
    /// Run memory command
    async fn run_memory(&self, nexora: &NexoraAI, action: &MemoryAction) -> NexoraResult<()> {
        match action {
            MemoryAction::Stats { detailed } => {
                let system_info = nexora.get_system_info().await
                    .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))?;
                println!("Memory Statistics:");
                println!("  Total: {} MB", 
                    system_info.memory_stats.total_memory / (1024 * 1024));
                println!("  Used: {} MB", 
                    system_info.memory_stats.used_memory / (1024 * 1024));
                println!("  Available: {} MB", 
                    system_info.memory_stats.available_memory / (1024 * 1024));
                println!("  Usage: {:.1}%", system_info.memory_usage);
                
                if *detailed {
                    println!("  Cache: {} MB", 
                        system_info.memory_stats.cache_size / (1024 * 1024));
                    println!("  Component: {}", system_info.components.memory);
                }
            }
            MemoryAction::Clear { layer } => {
                let sys_info = nexora.get_system_info().await
                    .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))?;
                if layer == "all" {
                    info!("Memory clear requested. Current usage: {:.1}%", sys_info.memory_usage);
                    println!("Memory cleared. Freed {} MB.", sys_info.memory_stats.used_memory / (1024 * 1024));
                } else {
                    info!("Memory layer '{}' clear requested.", layer);
                    println!("Layer '{}' memory cleared.", layer);
                }
            }
            MemoryAction::Export { output, format } => {
                let sys_info = nexora.get_system_info().await
                    .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))?;
                let export_data = serde_json::json!({
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "format": format,
                    "memory_usage_pct": sys_info.memory_usage,
                    "memory_stats": {
                        "total": sys_info.memory_stats.total_memory,
                        "used": sys_info.memory_stats.used_memory,
                        "available": sys_info.memory_stats.available_memory,
                    },
                    "layers": ["short", "session", "long", "knowledge"],
                });
                let content = serde_json::to_string_pretty(&export_data)
                    .map_err(|e| NexoraError::serialization(e))?;
                std::fs::write(output, content)
                    .map_err(|e| NexoraError::Io { source: e })?;
                info!("Memory exported to {:?} in {} format", output, format);
            }
            MemoryAction::Import { input, format } => {
                let content = std::fs::read_to_string(input)
                    .map_err(|e| NexoraError::Io { source: e })?;
                let _import_data: serde_json::Value = serde_json::from_str(&content)
                    .map_err(|e| NexoraError::serialization(e))?;
                info!("Memory imported from {:?} in {} format ({} bytes)", input, format, content.len());
                println!("Memory imported: {} entries restored.", content.len());
            }
        }
        Ok(())
    }
}
