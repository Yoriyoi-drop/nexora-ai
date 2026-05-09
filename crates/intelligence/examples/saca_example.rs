//! SACA Framework Example
//! 
//! Demonstrates complete usage of SACA framework for automated code generation

use nexora_foundation::reasoning::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 SACA Framework Example");
    println!("==========================\n");
    
    // Example 1: Basic SACA Usage
    basic_saca_example().await?;
    
    println!("\n{}\n", "=".repeat(50));
    
    // Example 2: Advanced SACA with Configuration
    advanced_saca_example().await?;
    
    println!("\n{}\n", "=".repeat(50));
    
    // Example 3: SACA with Model Integration
    integration_example().await?;
    
    println!("\n{}\n", "=".repeat(50));
    
    // Example 4: Performance Benchmarking
    benchmark_example().await?;
    
    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Example 1: Basic SACA Usage
async fn basic_saca_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Example 1: Basic SACA Usage");
    println!("---------------------------------");
    
    // Initialize SACA with default configuration
    let config = SACAConfig::default();
    let saca = SACA::new(config).await?;
    
    // Define a simple coding task
    let task = CodingTask {
        description: "Create a function that sorts an array of integers in ascending order using quicksort algorithm".to_string(),
        requirements: vec![
            "Use quicksort algorithm".to_string(),
            "Handle empty arrays".to_string(),
            "Include error handling".to_string(),
            "Add unit tests".to_string(),
        ],
        constraints: vec![
            "Time complexity O(n log n) average case".to_string(),
            "Space complexity O(log n) due to recursion".to_string(),
            "No external dependencies".to_string(),
        ],
        context: None,
    };
    
    println!("🎯 Task: {}", task.description);
    println!("📋 Requirements: {}", task.requirements.len());
    println!("⚠️  Constraints: {}", task.constraints.len());
    
    // Execute SACA pipeline
    let start_time = std::time::Instant::now();
    let solution = saca.solve(task).await?;
    let execution_time = start_time.elapsed();
    
    // Display results
    println!("\n📊 Results:");
    println!("  Quality Score: {:.3}", solution.quality_score);
    println!("  Modules Generated: {}", solution.modules.len());
    println!("  Test Coverage: {:.1}%", solution.test_coverage * 100.0);
    println!("  Performance Grade: {:?}", solution.performance_grade);
    println!("  Total Iterations: {}", solution.total_iterations);
    println!("  Feedback Loops: {}", solution.total_feedback_loops);
    println!("  Execution Time: {:?}", execution_time);
    
    // Display generated code snippet
    println!("\n💻 Generated Code (first 200 chars):");
    let code_preview = if solution.final_code.len() > 200 {
        &solution.final_code[..200]
    } else {
        &solution.final_code
    };
    println!("{}\n...", code_preview);
    
    Ok(())
}

/// Example 2: Advanced SACA with Custom Configuration
async fn advanced_saca_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Example 2: Advanced SACA Configuration");
    println!("----------------------------------------");
    
    // Create custom SACA configuration
    let mut config = SACAConfig::default();
    
    // Configure Chain-of-Thought reasoning
    config.cot_config.max_reasoning_steps = 15;
    config.cot_config.reasoning_depth = ReasoningDepth::Deep;
    config.cot_config.include_edge_cases = true;
    config.cot_config.include_assumptions = true;
    config.cot_config.include_risks = true;
    
    // Configure sampling
    config.sampling_config.num_candidates = 8;
    config.sampling_config.sampling_strategy = SamplingStrategy::Diverse;
    config.sampling_config.diversity_threshold = 0.4;
    config.sampling_config.algorithm_variety = true;
    
    // Configure execution
    config.execute_config.max_fix_attempts = 5;
    config.execute_config.error_analysis_depth = ErrorAnalysisDepth::Deep;
    config.execute_config.parallel_execution = true;
    
    // Configure global settings
    config.quality_threshold = 0.85;
    config.max_feedback_loops = 7;
    config.parallel_execution = true;
    config.enable_caching = true;
    config.log_level = "info".to_string();
    
    let saca = SACA::new(config).await?;
    
    // Define a more complex task
    let task = CodingTask {
        description: "Implement a thread-safe binary search tree with concurrent insert, delete, and search operations".to_string(),
        requirements: vec![
            "Thread-safe operations".to_string(),
            "Lock-free or fine-grained locking".to_string(),
            "Concurrent readers and writers".to_string(),
            "Memory management".to_string(),
            "Performance optimization".to_string(),
            "Comprehensive testing".to_string(),
        ],
        constraints: vec![
            "No unsafe code".to_string(),
            "Handle concurrent access safely".to_string(),
            "O(log n) operations".to_string(),
            "Memory efficient".to_string(),
        ],
        context: Some(TaskContext {
            repository_path: Some("./examples".to_string()),
            existing_files: vec![
                "utils.rs".to_string(),
                "concurrent.rs".to_string(),
                "testing.rs".to_string(),
            ],
            dependencies: vec![
                "tokio".to_string(),
                "parking_lot".to_string(),
                "crossbeam".to_string(),
            ],
            coding_standards: {
                let mut standards = HashMap::new();
                standards.insert("naming".to_string(), "snake_case".to_string());
                standards.insert("documentation".to_string(), "rustdoc".to_string());
                standards
            },
        }),
    };
    
    println!("🎯 Complex Task: Concurrent BST Implementation");
    println!("⚙️  Custom Configuration Applied");
    
    // Execute with advanced settings
    let start_time = std::time::Instant::now();
    let solution = saca.solve(task).await?;
    let execution_time = start_time.elapsed();
    
    println!("\n📊 Advanced Results:");
    println!("  Quality Score: {:.3}", solution.quality_score);
    println!("  Modules: {}", solution.modules.len());
    println!("  Execution Time: {:?}", execution_time);
    println!("  Iterations: {}", solution.total_iterations);
    println!("  Feedback Loops: {}", solution.total_feedback_loops);
    
    // Show phase metrics if available
    let metrics = saca.get_metrics().await;
    println!("  Total Tasks Processed: {}", metrics.total_tasks_processed);
    println!("  Average Quality: {:.3}", metrics.average_quality_score);
    println!("  Success Rate: {:.1}%", metrics.success_rate * 100.0);
    
    Ok(())
}

/// Example 3: SACA with Model Integration
async fn integration_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Example 3: SACA with Model Integration");
    println!("----------------------------------------");
    
    // Configure SACA for integration
    let mut saca_config = SACAConfig::default();
    saca_config.quality_threshold = 0.9;
    saca_config.enable_caching = true;
    
    // Create integrated SACA instance
    let integration = SACAIntegration::new(saca_config).await?;
    
    let stats = integration.get_integration_stats();
    println!("📈 Integration Status:");
    println!("  ATQS Compression: {}", stats.atqs_enabled);
    println!("  Caffeine Multimodal: {}", stats.caffeine_enabled);
    println!("  HAS MoE Routing: {}", stats.has_moe_enabled);
    println!("  Total Models: {}", stats.total_models_enabled);
    
    // Define a task suitable for multimodal processing
    let task = CodingTask {
        description: "Create a multimodal data processing system that can handle text, images, and structured data with AI-enhanced analysis".to_string(),
        requirements: vec![
            "Multimodal input processing".to_string(),
            "AI-enhanced analysis".to_string(),
            "Efficient data compression".to_string(),
            "Expert routing for different data types".to_string(),
            "Real-time processing capabilities".to_string(),
        ],
        constraints: vec![
            "Handle large datasets efficiently".to_string(),
            "Maintain data integrity".to_string(),
            "Optimize for throughput".to_string(),
        ],
        context: Some(TaskContext {
            repository_path: Some("./multimodal_project".to_string()),
            existing_files: vec![
                "processors/text.rs".to_string(),
                "processors/image.rs".to_string(),
                "processors/structured.rs".to_string(),
            ],
            dependencies: vec![
                "serde".to_string(),
                "image".to_string(),
                "ndarray".to_string(),
                "tokio".to_string(),
            ],
            coding_standards: HashMap::new(),
        }),
    };
    
    println!("\n🎯 Multimodal Task: AI-Enhanced Data Processing");
    
    // Solve with integration (note: models would need to be actually configured)
    let start_time = std::time::Instant::now();
    
    // For demonstration, we'll use basic SACA solve
    // In real usage with models, you'd call solve_with_models()
    let solution = integration.saca().solve(task).await?;
    let execution_time = start_time.elapsed();
    
    println!("\n📊 Integration Results:");
    println!("  Quality Score: {:.3}", solution.quality_score);
    println!("  Execution Time: {:?}", execution_time);
    println!("  Test Coverage: {:.1}%", solution.test_coverage * 100.0);
    
    // Note: In a real implementation with models enabled, you would see:
    // - Compression ratios from ATQS
    // - Multimodal features from Caffeine  
    // - Routing efficiency from HAS MoE FFN
    
    println!("\n💡 Note: Full model integration requires actual model configurations");
    
    Ok(())
}

/// Example 4: Performance Benchmarking
async fn benchmark_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Example 4: Performance Benchmarking");
    println!("------------------------------------");
    
    let config = SACAConfig::default();
    let saca = SACA::new(config).await?;
    
    // Define multiple tasks for benchmarking
    let tasks = vec![
        CodingTask {
            description: "Implement a hash table with chaining collision resolution".to_string(),
            requirements: vec!["O(1) average operations".to_string()],
            constraints: vec!["Memory efficient".to_string()],
            context: None,
        },
        CodingTask {
            description: "Create a priority queue using binary heap".to_string(),
            requirements: vec!["Efficient insert and extract".to_string()],
            constraints: vec!["O(log n) operations".to_string()],
            context: None,
        },
        CodingTask {
            description: "Implement a graph traversal algorithm (BFS and DFS)".to_string(),
            requirements: vec!["Handle disconnected graphs".to_string()],
            constraints: vec!["Linear time complexity".to_string()],
            context: None,
        },
        CodingTask {
            description: "Create a string pattern matching algorithm".to_string(),
            requirements: vec!["Multiple pattern support".to_string()],
            constraints: vec!["Efficient searching".to_string()],
            context: None,
        },
        CodingTask {
            description: "Implement a binary search with variations".to_string(),
            requirements: vec!["Handle edge cases".to_string()],
            constraints: vec!["O(log n) complexity".to_string()],
            context: None,
        },
    ];
    
    let tasks_count = tasks.len();
    println!("🏃 Running benchmark with {} tasks...", tasks_count);
    
    let mut total_time = std::time::Duration::ZERO;
    let mut total_quality = 0.0f32;
    let mut successful_tasks = 0;
    
    for (i, task) in tasks.into_iter().enumerate() {
        print!("  Task {}: {}... ", i + 1, task.description.split('.').next().unwrap_or("Unknown"));
        
        let start_time = std::time::Instant::now();
        match saca.solve(task).await {
            Ok(solution) => {
                let task_time = start_time.elapsed();
                total_time += task_time;
                total_quality += solution.quality_score;
                successful_tasks += 1;
                
                println!("✅ {:.3}s (Quality: {:.3})", task_time.as_secs_f32(), solution.quality_score);
            },
            Err(e) => {
                println!("❌ Failed: {}", e);
            }
        }
    }
    
    if successful_tasks > 0 {
        let avg_time = total_time / successful_tasks as u32;
        let avg_quality = total_quality / successful_tasks as f32;
        let throughput = successful_tasks as f64 / total_time.as_secs_f64();
        
        println!("\n📊 Benchmark Results:");
        println!("  Successful Tasks: {}/{}", successful_tasks, tasks_count);
        println!("  Average Time per Task: {:?}", avg_time);
        println!("  Average Quality Score: {:.3}", avg_quality);
        println!("  Throughput: {:.2} tasks/second", throughput);
        println!("  Total Execution Time: {:?}", total_time);
        
        // Get SACA metrics
        let metrics = saca.get_metrics().await;
        println!("  SACA Success Rate: {:.1}%", metrics.success_rate * 100.0);
        println!("  Average Iterations: {:.1}", metrics.average_iterations_per_task);
    } else {
        println!("\n❌ No tasks completed successfully");
    }
    
    Ok(())
}
