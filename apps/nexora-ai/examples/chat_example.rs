//! Chat example for Nexora AI
//! 
//! This example demonstrates how to use the chat functionality

use nexora_ai::{NexoraAI, NexoraConfig};
use tokio;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Nexora AI
    let config = NexoraConfig::default();
    let nexora = NexoraAI::new(config).await?;
    
    println!("💬 Nexora AI Chat Example");
    println!("========================");
    
    // Create a conversation ID
    let conversation_id = Some(Uuid::new_v4().to_string());
    
    // Simulate a chat conversation
    let messages = vec![
        "Hi, I'm interested in learning about machine learning.",
        "What are the basic concepts I should start with?",
        "Can you recommend some resources for beginners?",
        "Thank you for the help!",
    ];
    
    for (i, message) in messages.iter().enumerate() {
        println!("\n👤 User (Message {}): {}", i + 1, message);
        
        match nexora.chat(message, conversation_id.clone()).await {
            Ok(response) => {
                println!("🤖 Assistant: {}", response);
            }
            Err(e) => {
                println!("❌ Chat failed: {}", e);
            }
        }
    }
    
    println!("\n🎉 Chat example completed!");
    Ok(())
}
