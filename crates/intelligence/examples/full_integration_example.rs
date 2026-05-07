//! Full Integration Example
//! 
//! Demonstrates complete usage of all integrated models:
//! - SACA (Systematic Adaptive Code Architecture)
//! - ATQS (Adaptive Tensor Quantization & Sparsification)
//! - CAFFEINE (Contrastive-Aware Fusion Framework)
//! - HAS-MoE-FFN (Hybrid Adaptive Structured MoE-FFN)

use nexora_model::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 Nexora AI Full Integration Example");
    println!("=====================================\n");
    
    // Example 1: Basic SACA
    basic_saca_example().await?;
    
    println!("\n{}\n", "=".repeat(50));
    
    // Example 2: SACA + ATQS (Compression)
    saca_atqs_example().await?;
    
    println!("\n{}\n", "=".repeat(50));
    
    // Example 3: SACA + CAFFEINE (Multimodal)
    saca_caffeine_example().await?;
    
    println!("\n{}\n", "=".repeat(50));
    
    // Example 4: SACA + HAS-MoE-FFN (Expert Routing)
    saca_has_moe_example().await?;
    
    println!("\n{}\n", "=".repeat(50));
    
    // Example 5: Full Integration (All Models)
    full_integration_example().await?;
    
    println!("\n{}\n", "=".repeat(50));
    
    // Example 6: Performance Comparison
    performance_comparison().await?;
    
    println!("\n✅ All integration examples completed successfully!");
    Ok(())
}

/// Example 1: Basic SACA
async fn basic_saca_example() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("📝 Example 1: Basic SACA");
    println!("------------------------");
    
    let model = UnifiedModelFactory::create_basic_coder().await.unwrap();
    
    let task = CodingTask {
        description: "Create a function that implements bubble sort algorithm in Rust".to_string(),
        requirements: vec![
            "Use Rust programming language".to_string(),
            "Include proper error handling".to_string(),
            "Add documentation comments".to_string(),
            "Write unit tests".to_string(),
        ],
        constraints: vec![
            "Time complexity O(n²)".to_string(),
            "Space complexity O(1)".to_string(),
            "No external dependencies".to_string(),
        ],
        context: None,
    };
    
    let solution = model.generate_code(&task).await.unwrap();
    
    println!("📊 Results:");
    println!("  Quality Score: {:.3}", solution.quality_score);
    println!("  Execution Time: {:?}", solution.execution_time);
    println!("  Integration Mode: {:?}", solution.integration_mode);
    println!("  Models Used: Basic SACA only");
    
    Ok(())
}

/// Example 2: SACA + ATQS (Compression)
async fn saca_atqs_example() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🗜️  Example 2: SACA + ATQS Compression");
    println!("-------------------------------------");
    
    let model = UnifiedModelFactory::create_compressed_coder().await.unwrap();
    
    let task = CodingTask {
        description: "Implement a large-scale data processing pipeline with compression optimization".to_string(),
        requirements: vec![
            "Process large datasets efficiently".to_string(),
            "Implement data compression".to_string(),
            "Optimize memory usage".to_string(),
            "Include performance benchmarks".to_string(),
        ],
        constraints: vec![
            "Memory usage < 100MB".to_string(),
            "Processing time < 10 seconds".to_string(),
            "Compression ratio > 2:1".to_string(),
        ],
        context: None,
    };
    
    let solution = model.generate_code(&task).await.unwrap();
    
    println!("📊 Results:");
    println!("  Quality Score: {:.3}", solution.quality_score);
    println!("  Execution Time: {:?}", solution.execution_time);
    println!("  ATQS Compression Applied: {}", solution.atqs_compression_applied);
    println!("  Compression Ratio: {:.2}:1", solution.compression_ratio);
    println!("  Integration Mode: {:?}", solution.integration_mode);
    
    Ok(())
}

/// Example 3: SACA + CAFFEINE (Multimodal)
async fn saca_caffeine_example() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Example 3: SACA + CAFFEINE Multimodal");
    println!("--------------------------------------");
    
    let model = UnifiedModelFactory::create_multimodal_coder().await.unwrap();
    
    let task = CodingTask {
        description: "Create a computer vision application that processes images and generates analysis reports".to_string(),
        requirements: vec![
            "Image processing capabilities".to_string(),
            "Visual analysis features".to_string(),
            "Report generation".to_string(),
            "User interface components".to_string(),
        ],
        constraints: vec![
            "Support multiple image formats".to_string(),
            "Real-time processing".to_string(),
            "Accurate analysis results".to_string(),
        ],
        context: Some(nexora_model::saca::TaskContext {
            repository_path: Some("./vision_app".to_string()),
            existing_files: vec![
                "src/image_processor.rs".to_string(),
                "src/analysis.rs".to_string(),
                "src/ui.rs".to_string(),
            ],
            dependencies: vec![
                "image".to_string(),
                "opencv".to_string(),
                "egui".to_string(),
            ],
            coding_standards: {
                let mut standards = HashMap::new();
                standards.insert("naming".to_string(), "snake_case".to_string());
                standards.insert("documentation".to_string(), "rustdoc".to_string());
                standards
            },
        }),
    };
    
    let solution = model.generate_code(&task).await.unwrap();
    
    println!("📊 Results:");
    println!("  Quality Score: {:.3}", solution.quality_score);
    println!("  Execution Time: {:?}", solution.execution_time);
    println!("  CAFFEINE Multimodal Enhanced: {}", solution.caffeine_multimodal_enhanced);
    println!("  Multimodal Features: {}", solution.multimodal_features.len());
    println!("  Integration Mode: {:?}", solution.integration_mode);
    
    // Test multimodal processing
    let multimodal_inputs = MultiModalInputs {
        text: Some(TextInput {
            text: "Analyze this image and generate a report".to_string(),
            tokens: None,
            language: "en".to_string(),
        }),
        image: None, // Would contain actual image data in real usage
        audio: None,
        video: None,
        context: Some(ContextInfo {
            task_type: nexora_model::caffeine::types::TaskType::Reasoning,
            instruction: Some("Generate detailed analysis report".to_string()),
            previous_actions: vec![],
            environment_state: None,
        }),
    };
    
    let _multimodal_output = model.process_multimodal(&multimodal_inputs).await.unwrap();
    println!("  Multimodal Processing: ✅ Success");
    
    Ok(())
}

/// Example 4: SACA + HAS-MoE-FFN (Expert Routing)
async fn saca_has_moe_example() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Example 4: SACA + HAS-MoE-FFN Expert Routing");
    println!("---------------------------------------------");
    
    let model = UnifiedModelFactory::create_expert_coder().await.unwrap();
    
    let task = CodingTask {
        description: "Implement a complex mathematical algorithm with multiple optimization strategies".to_string(),
        requirements: vec![
            "Advanced mathematical computations".to_string(),
            "Multiple optimization approaches".to_string(),
            "Performance analysis".to_string(),
            "Comparative benchmarks".to_string(),
        ],
        constraints: vec![
            "High precision calculations".to_string(),
            "Optimized for speed".to_string(),
            "Memory efficient".to_string(),
        ],
        context: Some(nexora_model::saca::TaskContext {
            repository_path: Some("./math_optimization".to_string()),
            existing_files: vec![
                "src/algorithms.rs".to_string(),
                "src/optimization.rs".to_string(),
                "src/benchmarks.rs".to_string(),
            ],
            dependencies: vec![
                "nalgebra".to_string(),
                "ndarray".to_string(),
                "criterion".to_string(),
            ],
            coding_standards: {
                let mut standards = HashMap::new();
                standards.insert("naming".to_string(), "snake_case".to_string());
                standards.insert("documentation".to_string(), "rustdoc".to_string());
                standards
            },
        }),
    };
    
    let solution = model.generate_code(&task).await.unwrap();
    
    println!("📊 Results:");
    println!("  Quality Score: {:.3}", solution.quality_score);
    println!("  Execution Time: {:?}", solution.execution_time);
    println!("  HAS-MoE Routing Applied: {}", solution.has_moe_routing_applied);
    println!("  Routing Efficiency: {:.3}", solution.routing_efficiency);
    println!("  Integration Mode: {:?}", solution.integration_mode);
    
    Ok(())
}

/// Example 5: Full Integration (All Models)
async fn full_integration_example() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🌟 Example 5: Full Integration (All Models)");
    println!("----------------------------------------");
    
    let model = UnifiedModelFactory::create_full_integration().await.unwrap();
    
    let stats = model.get_statistics();
    println!("📈 Model Statistics:");
    println!("  Integration Mode: {:?}", stats.integration_mode);
    println!("  Models Enabled: {}", stats.models_enabled);
    println!("  ATQS Enabled: {}", stats.atqs_enabled);
    println!("  CAFFEINE Enabled: {}", stats.caffeine_enabled);
    println!("  HAS-MoE Enabled: {}", stats.has_moe_enabled);
    
    let task = CodingTask {
        description: "Create an advanced AI-powered development assistant that integrates multiple AI models for optimal code generation, compression, multimodal processing, and expert routing".to_string(),
        requirements: vec![
            "Integrate multiple AI models".to_string(),
            "Optimize for performance and efficiency".to_string(),
            "Support multimodal inputs".to_string(),
            "Provide expert-level code generation".to_string(),
            "Include comprehensive testing".to_string(),
        ],
        constraints: vec![
            "High quality code generation".to_string(),
            "Efficient resource utilization".to_string(),
            "Scalable architecture".to_string(),
            "Robust error handling".to_string(),
        ],
        context: Some(nexora_model::saca::TaskContext {
            repository_path: Some("./ai_assistant".to_string()),
            existing_files: vec![
                "src/models.rs".to_string(),
                "src/integration.rs".to_string(),
                "src/processing.rs".to_string(),
                "src/optimization.rs".to_string(),
            ],
            dependencies: vec![
                "tokio".to_string(),
                "serde".to_string(),
                "ndarray".to_string(),
                "tracing".to_string(),
            ],
            coding_standards: {
                let mut standards = HashMap::new();
                standards.insert("naming".to_string(), "snake_case".to_string());
                standards.insert("documentation".to_string(), "rustdoc".to_string());
                standards.insert("testing".to_string(), "comprehensive".to_string());
                standards
            },
        }),
    };
    
    let solution = model.generate_code(&task).await.unwrap();
    
    println!("\n📊 Full Integration Results:");
    println!("  Quality Score: {:.3}", solution.quality_score);
    println!("  Execution Time: {:?}", solution.execution_time);
    println!("  ATQS Compression Applied: {}", solution.atqs_compression_applied);
    println!("  Compression Ratio: {:.2}:1", solution.compression_ratio);
    println!("  CAFFEINE Multimodal Enhanced: {}", solution.caffeine_multimodal_enhanced);
    println!("  Multimodal Features: {}", solution.multimodal_features.len());
    println!("  HAS-MoE Routing Applied: {}", solution.has_moe_routing_applied);
    println!("  Routing Efficiency: {:.3}", solution.routing_efficiency);
    println!("  Integration Mode: {:?}", solution.integration_mode);
    
    // Test multimodal processing
    let multimodal_inputs = MultiModalInputs {
        text: Some(TextInput {
            text: "Generate optimized code with multimodal understanding".to_string(),
            tokens: None,
            language: "en".to_string(),
        }),
        image: None,
        audio: None,
        video: None,
        context: Some(ContextInfo {
            task_type: nexora_model::caffeine::types::TaskType::Generation,
            instruction: Some("Create high-quality, optimized code".to_string()),
            previous_actions: vec![],
            environment_state: None,
        }),
    };
    
    let _multimodal_output = model.process_multimodal(&multimodal_inputs).await.unwrap();
    println!("  Multimodal Processing: ✅ Success");
    
    Ok(())
}

/// Example 6: Performance Comparison
async fn performance_comparison() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Example 6: Performance Comparison");
    println!("------------------------------------");
    
    let models = vec![
        ("Basic SACA", UnifiedModelFactory::create_basic_coder().await.unwrap()),
        ("SACA + ATQS", UnifiedModelFactory::create_compressed_coder().await.unwrap()),
        ("SACA + CAFFEINE", UnifiedModelFactory::create_multimodal_coder().await.unwrap()),
        ("SACA + HAS-MoE", UnifiedModelFactory::create_expert_coder().await.unwrap()),
        ("Full Integration", UnifiedModelFactory::create_full_integration().await.unwrap()),
    ];
    
    let task = CodingTask {
        description: "Implement a generic binary search tree with insertion, deletion, and search operations".to_string(),
        requirements: vec![
            "Generic type support".to_string(),
            "Efficient operations".to_string(),
            "Memory management".to_string(),
            "Unit tests included".to_string(),
        ],
        constraints: vec![
            "O(log n) average operations".to_string(),
            "No memory leaks".to_string(),
            "Thread-safe if possible".to_string(),
        ],
        context: None,
    };
    
    println!("🏃 Running performance comparison...");
    
    let mut results = Vec::new();
    
    for (name, model) in models {
        print!("  {}: ", name);
        
        let start_time = std::time::Instant::now();
        let solution = model.generate_code(&task.clone()).await.unwrap();
        let execution_time = start_time.elapsed();
        
        let stats = model.get_statistics();
        
        println!("✅ {:.3}s (Quality: {:.3}, Models: {})", 
                execution_time.as_secs_f32(), 
                solution.quality_score,
                stats.models_enabled);
        
        results.push((name, execution_time, solution.quality_score, stats.models_enabled));
    }
    
    println!("\n📊 Performance Summary:");
    println!("  {:<20} {:<10} {:<10} {:<10}", "Model", "Time(s)", "Quality", "Models");
    println!("{}", "-".repeat(55));
    
    for (name, time, quality, models) in &results {
        println!("  {:<20} {:<10.3} {:<10.3} {:<10}", name, time.as_secs_f32(), quality, models);
    }
    
    // Find best performers
    let fastest = results.iter().min_by_key(|(_, time, _, _)| *time).unwrap();
    let highest_quality = results.iter().max_by(|(_, _, quality, _), (_, _, quality2, _)| quality.partial_cmp(quality2).unwrap()).unwrap();
    let most_models = results.iter().max_by(|(_, _, _, models), (_, _, _, models2)| models.partial_cmp(models2).unwrap()).unwrap();
    
    println!("\n🏆 Awards:");
    println!("  Fastest: {} ({:.3}s)", fastest.0, fastest.1.as_secs_f32());
    println!("  Highest Quality: {} ({:.3})", highest_quality.0, highest_quality.2);
    println!("  Most Models: {} ({} models)", most_models.0, most_models.3);
    
    Ok(())
}
