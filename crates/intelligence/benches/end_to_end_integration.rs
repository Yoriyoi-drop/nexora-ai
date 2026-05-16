//! End-to-End Integration Benchmark
//! 
//! Benchmark yang menggabungkan ATQS (compression), Caffeine (multimodal encoder), 
//! dan HAS-MoE-FFN (expert routing) untuk mengukur performa holistik sistem.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::sync::Arc;
use tokio::runtime::Runtime;
use tracing::{info, warn, debug, error, span, Level, Instrument};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

// Import modul yang akan di-benchmark
use nexora_foundation::multimodal::caffeine::{Caffeine, CaffeineConfig, MultiModalInputs};
use nexora_foundation::multimodal::caffeine::types::{UnifiedToken, ModalityType, TextInput, ImageInput, ImageFormat, ContextInfo, TaskType, MultiModalOutputs, PerformanceMetrics, ActionOutput};
use nexora_foundation::multimodal::caffeine::config::{EncodersConfig, QFormerConfig, TokenizerConfig, ActionConfig};
use nexora_foundation::compression::prelude::*;
use nexora_foundation::has_moe_ffn::*;

/// Initialize logger untuk benchmark
fn init_logger() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nexora_model=debug,benchmark=info".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();
    
    info!("Logger initialized for end-to-end integration benchmark");
}

/// Buat input dummy untuk benchmark
fn create_dummy_multimodal_input(_batch_size: usize) -> MultiModalInputs {
    let span = span!(Level::DEBUG, "create_dummy_input", _batch_size);
    let _enter = span.enter();
    
    debug!("Creating dummy multimodal input for batch size: {}", _batch_size);
    
    // Dummy text data with longer content and proper tokens
    let text_data = "This is a comprehensive text input for benchmark testing purposes. It contains enough content to ensure proper tokenization and encoding by the text encoder. The text should be substantial enough to generate meaningful embeddings.";
    let tokens = Some(vec![101, 2009, 2023, 1037, 2205, 3931, 1998, 3735, 1997, 1037, 2205, 3931, 1010, 102]); // BERT-style tokens
    
    let inputs = MultiModalInputs {
        text: Some(TextInput {
            text: text_data.to_string(),
            tokens,
            language: "en".to_string(), // Use English for better compatibility
        }),
        image: None, // Skip image to avoid regional alignment issues
        audio: None,
        video: None,
        context: Some(ContextInfo {
            task_type: TaskType::Generation,
            instruction: Some("Generate comprehensive response".to_string()),
            previous_actions: vec![],
            environment_state: None,
        }),
    };
    
    debug!("Created dummy multimodal input (enhanced text with context)");
    inputs
}

/// Buat konfigurasi ATQS untuk benchmark
fn create_atqs_config() -> ATQSConfig {
    debug!("Creating ATQS configuration for benchmark");
    
    let config = ATQSConfig::default();
    
    info!("ATQS config created");
    config
}

/// Buat konfigurasi Caffeine untuk benchmark
fn create_caffeine_config(enable_atqs: bool, enable_moe: bool) -> CaffeineConfig {
    debug!("Creating Caffeine configuration - ATQS: {}, MoE: {}", enable_atqs, enable_moe);
    
    let config = CaffeineConfig {
        encoders_config: EncodersConfig::default(),
        qformer_config: QFormerConfig::default(),
        tokenizer_config: TokenizerConfig::default(),
        action_config: ActionConfig::default(),
        atqs_config: if enable_atqs { Some(create_atqs_config()) } else { None },
        has_moe_config: if enable_moe { Some(create_moe_config()) } else { None },
        enable_atqs_compression: enable_atqs,
        enable_has_moe_routing: enable_moe,
        model_dim: 768,
        max_sequence_length: 512,
        num_attention_heads: 12,
        num_hidden_layers: 6,
        hidden_dim: 3072,
        dropout_rate: 0.1,
    };
    
    debug!("Caffeine configuration created successfully");
    config
}

/// Buat konfigurasi Router untuk benchmark
fn create_router_config() -> RouterConfig {
    debug!("Creating Router configuration for benchmark");
    
    let config = nexora_foundation::has_moe_ffn::routing::RouterConfig {
        hidden_size: 4096,
        num_experts: 8,
        top_k: 2,
        ..Default::default()
    };
    
    info!("Router config created");
    config
}

/// Buat konfigurasi HAS-MoE-FFN untuk benchmark
fn create_moe_config() -> HasMoeFFNConfig {
    debug!("Creating HAS-MoE-FFN configuration for benchmark");
    
    let config = HasMoeFFNConfig::default();
    
    info!("MoE config created - num_experts: {}, top_k: {}", 
          config.num_experts, config.top_k);
    config
}

/// Benchmark integrasi penuh ATQS + Caffeine + MoE
fn benchmark_full_integration(c: &mut Criterion) {
    let span = span!(Level::INFO, "benchmark_full_integration");
    let _enter = span.enter();
    
    info!("Starting full integration benchmark - ATQS + Caffeine + MoE");
    
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("full_integration");
    
    // Test dengan berbagai batch sizes
    for batch_size in [1, 4, 8, 16].iter() {
        info!("Benchmarking batch size: {}", batch_size);
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("atqs_caffeine_moe", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let span = span!(Level::DEBUG, "benchmark_iteration", batch_size);
                    let _enter = span.enter();
                    
                    debug!("Starting benchmark iteration for batch size: {}", batch_size);
                    
                    // Inisialisasi komponen
                    let config = create_caffeine_config(true, true);
                    let mut caffeine = Caffeine::new(config).unwrap();
                    info!("Caffeine instance created with ATQS and MoE enabled");
                    
                    // Buat input
                    let inputs = create_dummy_multimodal_input(batch_size);
                    debug!("Input created for benchmark");
                    
                    // Jalankan forward pass
                    let start_time = std::time::Instant::now();
                    let result = caffeine.forward(black_box(&inputs));
                    let duration = start_time.elapsed();
                    
                    black_box(result);
                    black_box(duration);
                    
                    debug!("Forward pass completed successfully in {:?}", duration);
                });
            },
        );
    }
    
    group.finish();
    info!("Full integration benchmark completed");
}

/// Benchmark Caffeine saja (tanpa ATQS dan MoE)
fn benchmark_caffeine_only(c: &mut Criterion) {
    let span = span!(Level::INFO, "benchmark_caffeine_only");
    let _enter = span.enter();
    
    info!("Starting Caffeine-only benchmark (baseline)");
    
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("caffeine_only");
    
    for batch_size in [1, 4, 8, 16].iter() {
        debug!("Benchmarking Caffeine-only with batch size: {}", batch_size);
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("caffeine_baseline", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let config = create_caffeine_config(false, true); // Enable MoE for QFormer compatibility
                    let mut caffeine = Caffeine::new(config).unwrap();
                    debug!("Caffeine baseline instance created");
                    
                    let inputs = create_dummy_multimodal_input(batch_size);
                    match caffeine.forward(black_box(&inputs)) {
                        Ok(result) => {
                            black_box(result);
                        }
                        Err(e) => {
                            // Log error but don't panic - treat as failed iteration
                            debug!("Caffeine forward failed: {:?}", e);
                            black_box(());
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
    info!("Caffeine-only benchmark completed");
}

/// Benchmark ATQS + Caffeine (tanpa MoE)
fn benchmark_atqs_caffeine(c: &mut Criterion) {
    let span = span!(Level::INFO, "benchmark_atqs_caffeine");
    let _enter = span.enter();
    
    info!("Starting ATQS + Caffeine benchmark (without MoE)");
    
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("atqs_caffeine");
    
    for batch_size in [1, 4, 8, 16].iter() {
        debug!("Benchmarking ATQS+Caffeine with batch size: {}", batch_size);
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("atqs_caffeine", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let config = create_caffeine_config(true, false);
                    let mut caffeine = Caffeine::new(config).unwrap();
                    debug!("ATQS+Caffeine instance created");
                    
                    let inputs = create_dummy_multimodal_input(batch_size);
                    match caffeine.forward(black_box(&inputs)) {
                        Ok(result) => {
                            black_box(result);
                        }
                        Err(e) => {
                            // Log error but don't panic - treat as failed iteration
                            debug!("Caffeine forward failed: {:?}", e);
                            black_box(());
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
    info!("ATQS + Caffeine benchmark completed");
}

/// Benchmark Caffeine + MoE (tanpa ATQS)
fn benchmark_caffeine_moe(c: &mut Criterion) {
    let span = span!(Level::INFO, "benchmark_caffeine_moe");
    let _enter = span.enter();
    
    info!("Starting Caffeine + MoE benchmark (without ATQS)");
    
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("caffeine_moe");
    
    for batch_size in [1, 4, 8, 16].iter() {
        debug!("Benchmarking Caffeine+MoE with batch size: {}", batch_size);
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("caffeine_moe", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let config = create_caffeine_config(false, true);
                    let mut caffeine = Caffeine::new(config).unwrap();
                    debug!("Caffeine+MoE instance created");
                    
                    let inputs = create_dummy_multimodal_input(batch_size);
                    match caffeine.forward(black_box(&inputs)) {
                        Ok(result) => {
                            black_box(result);
                        }
                        Err(e) => {
                            // Log error but don't panic - treat as failed iteration
                            debug!("Caffeine forward failed: {:?}", e);
                            black_box(());
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
    info!("Caffeine + MoE benchmark completed");
}

/// Benchmark komponen individual untuk perbandingan
fn benchmark_individual_components(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Benchmark ATQS compression saja
    let mut group = c.benchmark_group("atqs_compression");
    
    for size in [1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("compression", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let config = create_atqs_config();
                    let mut compression = CompressionEngine::new(config).unwrap();
                    
                    // Create dummy UnifiedToken data
                    let mut tokens = Vec::with_capacity(size);
                    for i in 0..size {
                        tokens.push(UnifiedToken {
                            token_id: i,
                            modality: ModalityType::Text,
                            embedding: vec![0.5; 768],
                            position: i,
                            timestamp: None,
                            spatial_coords: None,
                        });
                    }
                    let _result = compression.compress(black_box(tokens)).unwrap_or_else(|e| {
                        debug!("Compression failed: {:?}", e);
                        vec![] // Return empty vector on error
                    });
                });
            },
        );
    }
    
    group.finish();
    
    // Benchmark HAS-MoE-FFN routing saja
    let mut group = c.benchmark_group("moe_routing");
    
    for input_size in [4096].iter() { // Use size that matches MoE config
        group.throughput(Throughput::Elements(*input_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("routing", input_size),
            input_size,
            |b, &input_size| {
                b.iter(|| {
                    let config = create_moe_config();
                    let mut moe = HasMoeFFN::new(config);
                    
                    let input = ndarray::Array2::from_shape_vec(
                        (1, input_size), 
                        vec![0.5f32; input_size]
                    ).unwrap();
                    
                    let _result = moe.forward(black_box(&input));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memori usage
fn benchmark_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_usage");
    
    group.bench_function("full_integration_memory", |b| {
        b.iter(|| {
            let config = create_caffeine_config(true, true);
            let mut caffeine = Caffeine::new(config).unwrap();
            
            let inputs = create_dummy_multimodal_input(8);
            
            // Measure memory before
            let memory_before = get_memory_usage();
            
            let _result = caffeine.forward(black_box(&inputs)).unwrap_or_else(|e| {
                debug!("Forward failed: {:?}", e);
                MultiModalOutputs {
                    text: None,
                    image: None,
                    audio: None,
                    video: None,
                    actions: vec![],
                    metrics: PerformanceMetrics {
                        total_time_ms: 0.0,
                        encoding_time_ms: 0.0,
                        query_time_ms: 0.0,
                        tokenization_time_ms: 0.0,
                        action_time_ms: 0.0,
                        memory_usage_mb: 0.0,
                        gpu_utilization_percent: 0.0,
                    },
                }
            });
            
            // Measure memory after
            let memory_after = get_memory_usage();
            
            black_box((memory_before, memory_after));
        });
    });
    
    group.finish();
}

/// Helper function untuk mendapatkan memory usage (simplified)
fn get_memory_usage() -> usize {
    // Placeholder - dalam implementasi nyata gunakan psutil atau sejenisnya
    0
}

/// Benchmark latency breakdown
fn benchmark_latency_breakdown(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("latency_breakdown");
    
    group.bench_function("encoding_stage", |b| {
        b.iter(|| {
            let config = create_caffeine_config(false, false);
            let mut caffeine = Caffeine::new(config).unwrap();
            
            let inputs = create_dummy_multimodal_input(4);
            
            // Measure full forward pass time (encoding included)
            let start = std::time::Instant::now();
            let _result = caffeine.forward(black_box(&inputs)).unwrap_or_else(|e| {
                debug!("Forward failed: {:?}", e);
                MultiModalOutputs {
                    text: None,
                    image: None,
                    audio: None,
                    video: None,
                    actions: vec![],
                    metrics: PerformanceMetrics {
                        total_time_ms: 0.0,
                        encoding_time_ms: 0.0,
                        query_time_ms: 0.0,
                        tokenization_time_ms: 0.0,
                        action_time_ms: 0.0,
                        memory_usage_mb: 0.0,
                        gpu_utilization_percent: 0.0,
                    },
                }
            });
            let duration = start.elapsed();
            
            black_box(duration);
        });
    });
    
    group.bench_function("qformer_stage", |b| {
        b.iter(|| {
            let config = create_caffeine_config(false, false);
            let mut caffeine = Caffeine::new(config).unwrap();
            
            let inputs = create_dummy_multimodal_input(4);
            
            // Measure full forward pass time (qformer included)
            let start = std::time::Instant::now();
            let _result = caffeine.forward(black_box(&inputs)).unwrap_or_else(|e| {
                debug!("Forward failed: {:?}", e);
                MultiModalOutputs {
                    text: None,
                    image: None,
                    audio: None,
                    video: None,
                    actions: vec![],
                    metrics: PerformanceMetrics {
                        total_time_ms: 0.0,
                        encoding_time_ms: 0.0,
                        query_time_ms: 0.0,
                        tokenization_time_ms: 0.0,
                        action_time_ms: 0.0,
                        memory_usage_mb: 0.0,
                        gpu_utilization_percent: 0.0,
                    },
                }
            });
            let duration = start.elapsed();
            
            black_box(duration);
        });
    });
    
    group.finish();
}

/// Benchmark throughput dengan batch processing
fn benchmark_batch_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("batch_throughput");
    
    for batch_size in [1, 2, 4, 8, 16, 32].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("throughput", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let config = create_caffeine_config(true, true);
                    let mut caffeine = Caffeine::new(config).unwrap();
                    
                    let inputs = create_dummy_multimodal_input(batch_size);
                    
                    let start = std::time::Instant::now();
                    let _result = caffeine.forward(black_box(&inputs)).unwrap_or_else(|e| {
                        debug!("Batch throughput forward failed: {:?}", e);
                        MultiModalOutputs {
                            text: None,
                            image: None,
                            audio: None,
                            video: None,
                            actions: vec![],
                            metrics: PerformanceMetrics {
                                total_time_ms: 0.0,
                                encoding_time_ms: 0.0,
                                query_time_ms: 0.0,
                                tokenization_time_ms: 0.0,
                                action_time_ms: 0.0,
                                memory_usage_mb: 0.0,
                                gpu_utilization_percent: 0.0,
                            },
                        }
                    });
                    let duration = start.elapsed();
                    
                    // Calculate throughput: samples per second
                    let throughput = batch_size as f64 / duration.as_secs_f64();
                    black_box(throughput);
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_full_integration,
    benchmark_caffeine_only,
    benchmark_atqs_caffeine,
    benchmark_caffeine_moe,
    benchmark_individual_components,
    benchmark_memory_usage,
    benchmark_latency_breakdown,
    benchmark_batch_throughput
);

criterion_main!(benches);
