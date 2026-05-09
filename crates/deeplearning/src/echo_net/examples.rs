//! ECHO-Net Ω Examples
//!
//! Contoh penggunaan ECHO-Net Ω untuk berbagai skenario:
//! - Basic usage
//! - Streaming inference
//! - Training simulation
//! - Performance benchmarking
//! - Custom configuration

use crate::echo_net::*;
use ndarray::{Array1, ArrayD};
use std::time::Instant;

/// Basic ECHO-Net Ω usage example
pub fn basic_usage_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ECHO-Net Ω Basic Usage Example ===");
    
    // Create configuration
    let config = EchoNetConfig {
        vocab_size: 1000,
        embedding_dim: 256,
        output_size: 128,
        amplitude_dim: 128,
        phase_dim: 64,
        resonance_dim: 64,
        num_bands: 4,
        band_frequencies: vec![1.0, 2.0, 4.0, 8.0],
        kernel_size: 32,
        compression_levels: 3,
        compression_ratio: 0.5,
        memory_size: 1000,
        decay_alpha: 0.01,
        write_threshold: 0.1,
        reasoning_steps: 2,
        reasoning_alpha: 0.1,
        energy_weight: 1.0,
        entropy_weight: 0.5,
        coherence_weight: 0.3,
        top_k: 32,
        routing_threshold: 0.01,
        phase_lr: 0.01,
        energy_clip: 5.0,
        streaming_window: 512,
        update_frequency: 50,
        ..Default::default()
    };
    
    // Create model
    let mut model = EchoNetModel::new(config)?;
    println!("✓ Model created successfully");
    
    // Process some tokens
    let tokens = vec![1, 5, 12, 8, 23, 42, 7, 19, 3, 15];
    println!("Processing {} tokens...", tokens.len());
    
    let start_time = Instant::now();
    let output = model.forward(&tokens)?;
    let duration = start_time.elapsed();
    
    println!("✓ Forward pass completed in {:?}", duration);
    println!("Output shape: {:?}", output.shape());
    println!("Output range: [{:.4}, {:.4}]", 
             output.iter().fold(f32::INFINITY, |a, &b| a.min(*b)),
             output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(*b)));
    
    // Get model metrics
    let metrics = model.get_metrics();
    println!("Throughput: {:.1} tokens/sec", metrics.throughput_tokens_per_second);
    println!("Memory utilization: {:.2}%", metrics.memory_utilization * 100.0);
    println!("Semantic coherence: {:.4}", metrics.semantic_coherence);
    
    Ok(())
}

/// Streaming inference example
pub fn streaming_inference_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== ECHO-Net Ω Streaming Inference Example ===");
    
    // Create streaming-optimized configuration
    let config = EchoNetConfig {
        vocab_size: 5000,
        embedding_dim: 512,
        output_size: 256,
        streaming_window: 1024,
        update_frequency: 25,
        top_k: 64,
        ..Default::default()
    };
    
    let mut model = EchoNetModel::new(config)?;
    println!("✓ Streaming model created");
    
    // Simulate streaming data
    let stream_data = vec![
        vec![1, 2, 3, 4, 5],
        vec![6, 7, 8, 9, 10],
        vec![11, 12, 13, 14, 15],
        vec![16, 17, 18, 19, 20],
        vec![21, 22, 23, 24, 25],
    ];
    
    println!("Processing {} streaming chunks...", stream_data.len());
    
    let total_start = Instant::now();
    let mut chunk_times = Vec::new();
    
    for (chunk_idx, chunk) in stream_data.iter().enumerate() {
        let chunk_start = Instant::now();
        
        let output = model.forward(chunk)?;
        let chunk_duration = chunk_start.elapsed();
        chunk_times.push(chunk_duration);
        
        println!("Chunk {}: {:?} (output sum: {:.4})", 
                chunk_idx + 1, chunk_duration, output.iter().sum::<f32>());
    }
    
    let total_duration = total_start.elapsed();
    let avg_chunk_time = chunk_times.iter().sum::<std::time::Duration>() / chunk_times.len() as u32;
    
    println!("✓ Streaming completed in {:?}", total_duration);
    println!("Average chunk time: {:?}", avg_chunk_time);
    println!("Streaming efficiency: {:.2}%", 
             (avg_chunk_time.as_secs_f32() / total_duration.as_secs_f32()) * 100.0);
    
    Ok(())
}

/// Training simulation example
pub fn training_simulation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== ECHO-Net Ω Training Simulation Example ===");
    
    // Create training configuration
    let config = EchoNetConfig {
        vocab_size: 10000,
        embedding_dim: 512,
        output_size: 256,
        reasoning_steps: 3,
        phase_lr: 0.001,
        energy_clip: 2.0,
        ..Default::default()
    };
    
    let mut model = EchoNetModel::new(config)?;
    println!("✓ Training model created");
    
    // Simulate training data
    let training_data = vec![
        (vec![1, 2, 3, 4, 5], vec![0.1, 0.2, 0.3, 0.4, 0.5]),
        (vec![6, 7, 8, 9, 10], vec![0.2, 0.3, 0.4, 0.5, 0.6]),
        (vec![11, 12, 13, 14, 15], vec![0.3, 0.4, 0.5, 0.6, 0.7]),
        (vec![16, 17, 18, 19, 20], vec![0.4, 0.5, 0.6, 0.7, 0.8]),
    ];
    
    println!("Simulating training on {} samples...", training_data.len());
    
    let mut total_loss = 0.0;
    let training_start = Instant::now();
    
    for epoch in 0..3 {
        println!("Epoch {}:", epoch + 1);
        let mut epoch_loss = 0.0;
        
        for (batch_idx, (input_tokens, target)) in training_data.iter().enumerate() {
            // Forward pass
            let output = model.forward(input_tokens)?;
            
            // Simulate loss calculation (MSE)
            let mut loss = 0.0;
            for (i, &target_val) in target.iter().enumerate() {
                if i < output.len() {
                    let diff = output[i] - target_val;
                    loss += diff * diff;
                }
            }
            loss /= target.len() as f32;
            
            epoch_loss += loss;
            
            // Simulate gradient update (in real training, this would involve backpropagation)
            // For this example, we just track the loss
            
            if batch_idx % 2 == 0 {
                println!("  Batch {}: loss = {:.6}", batch_idx + 1, loss);
            }
        }
        
        let avg_epoch_loss = epoch_loss / training_data.len() as f32;
        total_loss += avg_epoch_loss;
        println!("  Average epoch loss: {:.6}", avg_epoch_loss);
    }
    
    let training_duration = training_start.elapsed();
    let avg_loss = total_loss / 3.0;
    
    println!("✓ Training simulation completed in {:?}", training_duration);
    println!("Average loss across epochs: {:.6}", avg_loss);
    
    // Get training statistics
    let component_stats = model.get_component_statistics();
    println!("Training efficiency: {:.2}%", component_stats.training_stats.stabilization_efficiency * 100.0);
    
    Ok(())
}

/// Performance benchmarking example
pub fn performance_benchmark_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== ECHO-Net Ω Performance Benchmark Example ===");
    
    // Test different model sizes
    let test_configs = vec![
        ("Small", EchoNetConfig {
            vocab_size: 1000,
            embedding_dim: 128,
            output_size: 64,
            ..Default::default()
        }),
        ("Medium", EchoNetConfig {
            vocab_size: 5000,
            embedding_dim: 256,
            output_size: 128,
            ..Default::default()
        }),
        ("Large", EchoNetConfig {
            vocab_size: 10000,
            embedding_dim: 512,
            output_size: 256,
            ..Default::default()
        }),
    ];
    
    for (size_name, config) in test_configs {
        println!("\nBenchmarking {} model:", size_name);
        
        let mut model = EchoNetModel::new(config)?;
        
        // Benchmark different sequence lengths
        let sequence_lengths = vec![32, 64, 128, 256];
        
        for &seq_len in &sequence_lengths {
            // Generate test sequence
            let test_sequence: Vec<usize> = (0..seq_len)
                .map(|i| i % 1000)
                .collect();
            
            // Warm up
            let _ = model.forward(&test_sequence[..seq_len.min(16)]);
            
            // Benchmark
            let iterations = 10;
            let start_time = Instant::now();
            
            for _ in 0..iterations {
                let _ = model.forward(&test_sequence);
            }
            
            let duration = start_time.elapsed();
            let avg_time = duration.as_secs_f32() / iterations as f32;
            let tokens_per_sec = seq_len as f32 / avg_time;
            
            println!("  Seq length {}: {:.3}s avg, {:.1} tokens/sec", 
                    seq_len, avg_time, tokens_per_sec);
        }
        
        // Get memory statistics
        let metrics = model.get_metrics();
        println!("  Memory utilization: {:.2}%", metrics.memory_utilization * 100.0);
        println!("  Sparsity ratio: {:.4}", metrics.sparsity_ratio);
    }
    
    Ok(())
}

/// Custom configuration example
pub fn custom_configuration_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== ECHO-Net Ω Custom Configuration Example ===");
    
    // Create highly customized configuration
    let config = EchoNetConfig {
        // Model architecture
        vocab_size: 20000,
        embedding_dim: 768,
        output_size: 384,
        amplitude_dim: 384,
        phase_dim: 192,
        resonance_dim: 192,
        
        // Multi-band holographic writer
        num_bands: 6,
        band_frequencies: vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        kernel_size: 48,
        
        // Compression
        compression_levels: 4,
        compression_ratio: 0.4,
        
        // Memory
        memory_size: 5000,
        decay_alpha: 0.005,
        write_threshold: 0.15,
        
        // Reasoning
        reasoning_steps: 4,
        reasoning_alpha: 0.05,
        
        // Retrieval
        energy_weight: 1.2,
        entropy_weight: 0.4,
        coherence_weight: 0.4,
        top_k: 48,
        routing_threshold: 0.005,
        
        // Training
        phase_lr: 0.005,
        energy_clip: 3.0,
        
        // Streaming
        streaming_window: 2048,
        update_frequency: 75,
        
        ..Default::default()
    };
    
    println!("Creating model with custom configuration...");
    let mut model = EchoNetModel::new(config)?;
    println!("✓ Custom model created");
    
    // Test with challenging input
    let challenging_sequence = vec![
        42, 7, 19, 3, 15, 23, 8, 31, 12, 5,
        17, 29, 11, 33, 4, 21, 9, 25, 13, 37,
        6, 18, 2, 14, 26, 10, 22, 8, 30, 16,
    ];
    
    println!("Processing challenging sequence of {} tokens...", challenging_sequence.len());
    
    let start_time = Instant::now();
    let output = model.forward(&challenging_sequence)?;
    let duration = start_time.elapsed();
    
    println!("✓ Processing completed in {:?}", duration);
    
    // Analyze output characteristics
    let output_stats = analyze_output(&output);
    println!("Output statistics:");
    println!("  Mean: {:.6}", output_stats.mean);
    println!("  Std: {:.6}", output_stats.std);
    println!("  Min: {:.6}", output_stats.min);
    println!("  Max: {:.6}", output_stats.max);
    println!("  Entropy: {:.6}", output_stats.entropy);
    
    // Get detailed component statistics
    let component_stats = model.get_component_statistics();
    println!("\nComponent performance:");
    println!("  IRR reasoning depth: {:.2}", component_stats.irr_stats.reasoning_depth);
    println!("  TKRR routing efficiency: {:.4}", component_stats.tkrr_stats.routing_efficiency);
    println!("  ISC output quality: {:.4}", component_stats.isc_stats.output_quality);
    println!("  Training stabilization: {:.2}%", component_stats.training_stats.stabilization_efficiency * 100.0);
    
    Ok(())
}

/// Output statistics
#[derive(Debug)]
struct OutputStats {
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
    entropy: f32,
}

fn analyze_output(output: &Array1<f32>) -> OutputStats {
    let mean = output.iter().sum::<f32>() / output.len() as f32;
    let variance = output.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / output.len() as f32;
    let std = variance.sqrt();
    let min = output.iter().fold(f32::INFINITY, |a, &b| a.min(*b));
    let max = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(*b));
    
    // Calculate entropy
    let mut entropy = 0.0;
    for &val in output.iter() {
        if val > 0.0 {
            entropy -= val * val.ln();
        }
    }
    
    OutputStats {
        mean,
        std,
        min,
        max,
        entropy,
    }
}

/// Run all examples
pub fn run_all_examples() -> Result<(), Box<dyn std::error::Error>> {
    println!("ECHO-Net Ω Examples\n");
    
    basic_usage_example()?;
    streaming_inference_example()?;
    training_simulation_example()?;
    performance_benchmark_example()?;
    custom_configuration_example()?;
    
    println!("\n✓ All examples completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_usage_example() {
        let result = basic_usage_example();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_streaming_inference_example() {
        let result = streaming_inference_example();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_training_simulation_example() {
        let result = training_simulation_example();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_performance_benchmark_example() {
        let result = performance_benchmark_example();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_custom_configuration_example() {
        let result = custom_configuration_example();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_output_analysis() {
        let output = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let stats = analyze_output(&output);
        
        assert!((stats.mean - 0.3).abs() < 1e-6);
        assert!(stats.min == 0.1);
        assert!(stats.max == 0.5);
        assert!(stats.std > 0.0);
    }
}
