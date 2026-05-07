//! Basic usage example for Nexora AI
//! 
//! This example demonstrates how to use the core functionality of Nexora AI

use nexora_ai::{NexoraAI, NexoraConfig};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Nexora AI with default configuration
    let config = NexoraConfig::default();
    let nexora = NexoraAI::new(config).await?;
    
    println!("🚀 Nexora AI Basic Usage Example");
    println!("================================");
    
    // 1. Health check
    println!("\n📊 Performing health check...");
    match nexora.health_check().await {
        Ok(health) => {
            println!("✅ System healthy: {}", health.healthy);
            println!("📈 Performance score: {:.2}", health.performance_score);
        }
        Err(e) => println!("❌ Health check failed: {}", e),
    }
    
    // 2. Get system information
    println!("\n💻 Getting system information...");
    match nexora.get_system_info().await {
        Ok(info) => {
            println!("🖥️  CPU usage: {:.1}%", info.cpu_usage * 100.0);
            println!("🧠 Total memory: {} MB", info.memory_stats.total_memory / (1024 * 1024));
            println!("⏱️  Uptime: {} seconds", info.uptime);
        }
        Err(e) => println!("❌ Failed to get system info: {}", e),
    }
    
    // 3. Process a simple request
    println!("\n🤖 Processing text request...");
    let input = "Hello, how are you today?";
    match nexora.process_request(input).await {
        Ok(response) => {
            println!("📝 Input: {}", input);
            println!("💬 Response: {}", response);
        }
        Err(e) => println!("❌ Request processing failed: {}", e),
    }
    
    // 4. Generate text
    println!("\n✍️  Generating text...");
    match nexora.generate_text("The future of AI is", 50, 0.7).await {
        Ok(generated_text) => {
            println!("📄 Generated text: {}", generated_text);
        }
        Err(e) => println!("❌ Text generation failed: {}", e),
    }
    
    // 5. Code analysis
    println!("\n🔍 Analyzing code...");
    let code = r#"
fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
"#;
    match nexora.analyze_code(code, "rust").await {
        Ok(analysis) => {
            println!("🔧 Code analysis: {}", analysis);
        }
        Err(e) => println!("❌ Code analysis failed: {}", e),
    }
    
    // 6. Code generation
    println!("\n⚙️  Generating code...");
    match nexora.generate_code("Create a function that calculates factorial", "rust").await {
        Ok(generated_code) => {
            println!("📦 Generated code:\n{}", generated_code);
        }
        Err(e) => println!("❌ Code generation failed: {}", e),
    }
    
    println!("\n🎉 Basic usage example completed!");
    Ok(())
}
