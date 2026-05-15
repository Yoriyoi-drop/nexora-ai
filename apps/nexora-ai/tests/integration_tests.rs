//! Integration tests for Nexora AI system

use nexora_ai::{NexoraAI, NexoraConfig};
use nexora_ai::error::NexoraResult;
use tokio;
use std::path::PathBuf;
use tempfile::TempDir;

#[tokio::test]
async fn test_nexora_ai_initialization() -> NexoraResult<()> {
    let config = NexoraConfig::default();
    let ai = NexoraAI::new(config).await?;
    
    // Test basic functionality
    let system_info = ai.get_system_info().await?;
    assert!(!system_info.active_models.is_empty());
    
    let health = ai.health_check().await?;
    assert!(health.healthy);
    
    Ok(())
}

#[tokio::test]
async fn test_text_generation_integration() -> NexoraResult<()> {
    let config = NexoraConfig::default();
    let ai = NexoraAI::new(config).await?;
    
    let prompt = "Hello, world!";
    let result = ai.generate_text(prompt, 100, 0.7).await?;
    
    assert!(!result.is_empty());
    assert!(result.len() <= 100 * 4); // Rough token limit check
    
    Ok(())
}

#[tokio::test]
async fn test_chat_integration() -> NexoraResult<()> {
    let config = NexoraConfig::default();
    let ai = NexoraAI::new(config).await?;
    
    let message = "Hello, how are you?";
    let response = ai.chat(message, Some("test_conv".to_string())).await?;
    
    assert!(!response.is_empty());
    assert!(response.contains("Hello"));
    
    Ok(())
}

#[tokio::test]
async fn test_code_analysis_integration() -> NexoraResult<()> {
    let config = NexoraConfig::default();
    let ai = NexoraAI::new(config).await?;
    
    let code = r#"
pub struct Test {
    value: i32,
}

impl Test {
    pub fn new(value: i32) -> Self {
        Self { value }
    }
    
    pub fn get_value(&self) -> i32 {
        self.value
    }
}
"#;
    
    let analysis = ai.analyze_code(code, "rust").await?;
    
    assert!(analysis.contains("Language:"));
    assert!(analysis.contains("Lines:"));
    assert!(analysis.contains("Functions:"));
    assert!(analysis.contains("Classes:"));
    
    Ok(())
}

#[tokio::test]
async fn test_code_generation_integration() -> NexoraResult<()> {
    let config = NexoraConfig::default();
    let ai = NexoraAI::new(config).await?;
    
    let description = "Create a function that adds two numbers";
    let code = ai.generate_code(description, "rust").await?;
    
    assert!(!code.is_empty());
    assert!(code.contains("fn"));
    assert!(code.contains(description));
    
    Ok(())
}

#[tokio::test]
async fn test_request_processing_integration() -> NexoraResult<()> {
    let config = NexoraConfig::default();
    let ai = NexoraAI::new(config).await?;
    
    let input = "What is the capital of France?";
    let response = ai.process_request(input).await?;
    
    assert!(!response.is_empty());
    assert!(response.contains("France"));
    
    Ok(())
}

#[tokio::test]
async fn test_performance_metrics_integration() -> NexoraResult<()> {
    let config = NexoraConfig::default();
    let ai = NexoraAI::new(config).await?;
    
    let metrics = ai.get_performance_metrics().await?;
    
    assert!(metrics.get("cpu_usage").is_some());
    assert!(metrics.get("memory_usage").is_some());
    assert!(metrics.get("uptime").is_some());
    assert!(metrics.get("request_count").is_some());
    
    Ok(())
}

#[tokio::test]
async fn test_configuration_file_operations() -> NexoraResult<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = temp_dir.path().join("test_config.toml");
    
    // Test configuration creation
    let config = NexoraConfig::default();
    config.save_to_file(&config_path)?;
    
    // Test configuration loading
    let loaded_config = NexoraConfig::from_file(&config_path)?;
    
    // Test configuration validation
    loaded_config.validate()?;
    
    // Test configuration summary
    let summary = loaded_config.summary();
    assert!(!summary.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling_integration() -> NexoraResult<()> {
    let config = NexoraConfig::default();
    let ai = NexoraAI::new(config).await?;
    
    // Test empty input validation
    let result = ai.generate_text("", 100, 0.7).await;
    assert!(result.is_err());
    
    // Test invalid parameters
    let result = ai.generate_text("test", 0, 0.7).await;
    assert!(result.is_err());
    
    let result = ai.generate_text("test", 100, -0.1).await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_operations() -> NexoraResult<()> {
    let config = NexoraConfig::default();
    let ai = NexoraAI::new(config).await?;
    
    // Test concurrent operations
    let ai_clone = ai.clone();
    
    let (result1, result2, result3) = tokio::join!(
        ai.generate_text("Hello", 50, 0.5),
        ai_clone.generate_text("World", 50, 0.5),
        ai_clone.generate_text("Test", 50, 0.5)
    );
    
    let (r1, r2, r3) = (result1?, result2?, result3?);
    
    assert!(!r1.is_empty());
    assert!(!r2.is_empty());
    assert!(!r3.is_empty());
    
    Ok(())
}
