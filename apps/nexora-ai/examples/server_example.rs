//! Server example for Nexora AI
//! 
//! This example demonstrates how to run the Nexora AI API server

use nexora_ai::{NexoraAI, NexoraConfig, NexoraApi, ApiConfig};
use tokio;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Nexora AI
    let config = NexoraConfig::default();
    let _nexora = Arc::new(NexoraAI::new(config).await?);
    
    // Configure API server
    let api_config = ApiConfig {
        base_url: "http://127.0.0.1:8080".to_string(),
        api_key: None,
        timeout_seconds: 30,
        max_retries: 3,
        enable_rate_limiting: true,
        requests_per_minute: 1000,
    };
    
    println!("🌐 Nexora AI API Server Example");
    println!("===============================");
    println!("🚀 Starting server on {}", api_config.base_url);
    
    // Create API client
    let client = NexoraApi::new(api_config)?;
    
    println!("✅ API client created successfully!");
    println!("📊 Available API methods:");
    println!("  - process_text()    - Process text request");
    
    println!("\n🔧 Example API usage:");
    println!("let response = client.process_text(\"Hello, world!\").await?;");
    
    // Example API call
    println!("\n🚀 Making example API call...");
    match client.process_text("Hello, world!").await {
        Ok(response) => {
            println!("✅ API call successful!");
            println!("📝 Response: {}", response.data.unwrap_or("No data".to_string()));
        }
        Err(e) => println!("❌ API call failed: {}", e),
    }
    
    println!("👋 Example completed");
    Ok(())
}
