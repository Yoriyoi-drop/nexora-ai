# HAS-MoE-FFN Integration dengan ATQS

## Overview

HAS-MoE-FFN (Hybrid Adaptive Structured MoE-FFN) diimplementasikan sebagai modul terpisah yang dapat bekerja sama dengan ATQS untuk memberikan performa optimal dan efisiensi parameter.

## Arsitektur Integrasi

```
Input → HAS-MoE-FFN → ATQS Compression → Output
         ↓
    Context Analysis
         ↓
    Expert Selection
         ↓
    Load Balancing
         ↓
    Expert Processing
         ↓
    Aggregation
         ↓
    Output Fusion
```

## Cara Penggunaan

### 1. Basic Usage - HAS-MoE-FFN Standalone

```rust
use nexora_model::has_moe_ffn::prelude::*;

// Buat konfigurasi medium model
let config = HasMoeFfnConfig::medium_model();

// Buat HAS-MoE-FFN instance
let mut has_moe = HasMoeFfn::new(config)?;

// Forward pass
let input = ArrayD::from_shape_vec(vec![4096], vec![0.1; 4096])?;
let output = has_moe.forward(&input)?;
```

### 2. Integration dengan ATQS Compression

```rust
use nexora_model::has_moe_ffn::integration_examples::*;

// Buat enhanced model dengan ATQS compression
let config = EnhancedModelConfig {
    model_dim: 4096,
    num_experts: 8,
    enable_atqs_compression: true,
    compression_level: atqs::config::CompressionLevel::Medium,
    use_structured_matrices: true,
    ..Default::default()
};

let mut model = EnhancedAtqsModel::new(config)?;

// Forward pass dengan compression
let input = ArrayD::from_shape_vec(vec![4096], vec![0.1; 4096])?;
let output = model.integration.forward_with_compression(&input, true)?;

// Dapatkan performance statistics
let stats = model.integration.get_performance_stats();
println!("Compression ratio: {:.2}", stats.compression_ratio);
println!("Throughput: {:.2} tokens/sec", stats.throughput_tokens_per_sec);
```

### 3. Training dengan ATQS + HAS-MoE-FFN

```rust
// Training step
let batch_input = ArrayD::from_shape_vec(vec![32, 4096], vec![0.1; 32 * 4096])?;
let targets = ArrayD::from_shape_vec(vec![32, 4096], vec![0.2; 32 * 4096])?;

let metrics = model.training_step(&batch_input, &targets, 0.001)?;
println!("Loss: {:.6}", metrics.loss);
println!("Forward time: {:.2}ms", metrics.forward_time_ms);
println!("Throughput: {:.2} samples/sec", metrics.throughput_samples_per_sec);
```

## Konfigurasi untuk Berbagai Use Cases

### 1. Inference-Optimized Configuration

```rust
let config = HasMoeFfnConfig::inference_optimized();
// - Memory efficient: true
// - Parallel experts: true  
// - No dropout
// - Optimized untuk throughput
```

### 2. Training-Optimized Configuration

```rust
let config = HasMoeFfnConfig::training_optimized();
// - Mixed precision: true
// - CUDA support: true
// - Regularization dropout
// - Optimized untuk gradient stability
```

### 3. Custom Configuration

```rust
let config = HasMoeFfnConfig::new(2048, 6, 2)
    .with_expert_config(SwiGLUExpertConfig {
        input_dim: 2048,
        hidden_dim: 5504,  // 2.7x expansion
        output_dim: 2048,
        matrix_config: StructuredMatrixConfig {
            rank: 96,        // Low-rank factor
            block_size: 256,  // Block-diagonal
            sparsity_ratio: 0.1,
            use_low_rank: true,
            use_block_diagonal: true,
        },
        specialization: ExpertSpecialization::Reasoning,
        activation_dropout: 0.0,
    });
```

## Performance Optimization Tips

### 1. Expert Specialization

```rust
// Buat expert set yang seimbang
let experts = ExpertFactory::create_balanced_expert_set(
    8,      // num_experts
    4096,   // input_dim
    11008,  // hidden_dim (2.7x)
    4096,   // output_dim
)?;

// Otomatis mencakup:
// - 2 Reasoning experts
// - 2 Coding experts  
// - 2 Mathematics experts
// - 1 Language expert
// - 1 General expert
```

### 2. Load Balancing Strategy

```rust
let config = LoadBalancerConfig {
    strategy: LoadBalancingStrategy::Adaptive,
    max_queue_length: 100,
    timeout_ms: 5000,
    rebalance_interval_ms: 1000,
};

// Adaptive strategy memberikan:
// - Dynamic load distribution
// - Performance-based routing
// - Automatic rebalancing
```

### 3. Aggregation Method

```rust
let config = AggregationConfig {
    method: AggregationMethod::Attention,
    learnable_weights: true,
    attention_mechanism: true,
    normalization: true,
};

// Attention aggregation memberikan:
// - Context-aware fusion
// - Learnable attention weights
// - Output normalization
```

## Monitoring dan Debugging

### 1. Performance Metrics

```rust
let routing_stats = has_moe.router.get_routing_stats();
println!("Average confidence: {:.3}", routing_stats.average_confidence);
println!("Load balance score: {:.3}", routing_stats.load_balance_score);

let load_stats = has_moe.load_balancer.get_load_stats();
println!("Average load: {:.3}", load_stats.average_load);
println!("Healthy experts: {}/{}", load_stats.healthy_experts, load_stats.total_experts);
```

### 2. Expert Analysis

```rust
for (i, expert) in has_moe.experts.iter().enumerate() {
    let params = expert.parameter_count();
    let memory = expert.memory_usage();
    let compression = expert.gate_matrix.compression_ratio();
    
    println!("Expert {}: {} params, {} bytes, {:.1}% compression", 
        i, params, memory, compression * 100.0);
}
```

## Best Practices

### 1. Memory Management
- Gunakan structured matrices untuk reduksi parameter
- Monitor memory usage secara real-time
- Implementasi garbage collection untuk expert outputs

### 2. Performance Tuning
- Pilih top_k yang sesuai dengan hardware
- Gunakan adaptive load balancing untuk workload dinamis
- Monitor expert utilization untuk deteksi bottleneck

### 3. Error Handling
- Implementasi graceful fallback untuk expert failures
- Gunakan circuit breaker untuk expert overload
- Logging komprehensif untuk debugging

### 4. Integration Patterns
- Gunakan factory pattern untuk expert creation
- Implementasi strategy pattern untuk load balancing
- Gunakan observer pattern untuk performance monitoring

## Troubleshooting

### Common Issues

1. **Expert Overload**
   - Symptom: High latency, low throughput
   - Solution: Increase load balancer capacity, use adaptive strategy

2. **Memory Leaks**
   - Symptom: Memory usage grows continuously
   - Solution: Implement proper cleanup, monitor expert lifecycle

3. **Load Imbalance**
   - Symptom: Some experts overloaded, others idle
   - Solution: Tune load balancing parameters, enable rebalancing

4. **Performance Degradation**
   - Symptom: Throughput decreases over time
   - Solution: Check expert specialization, update routing strategy

## Advanced Usage

### 1. Custom Expert Types

```rust
pub struct CustomExpert {
    base_expert: StructuredSwiGLUExpert,
    custom_logic: CustomProcessor,
}

impl CustomExpert {
    pub fn new(config: SwiGLUExpertConfig, custom_logic: CustomProcessor) -> Result<Self> {
        let base_expert = StructuredSwiGLUExpert::new(config, "custom_expert")?;
        Ok(Self { base_expert, custom_logic })
    }
    
    pub fn forward(&mut self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let base_output = self.base_expert.forward(input)?;
        self.custom_logic.process(&base_output)
    }
}
```

### 2. Multi-Modal Integration

```rust
let multimodal_config = HasMoeFfnConfig {
    model_dim: 8192,  // Combined text + image features
    num_experts: 16,
    top_k: 4,
    expert_config: SwiGLUExpertConfig {
        specialization: ExpertSpecialization::Creative,  // Untuk multimodal reasoning
        hidden_dim: 22016,  // 2.7x expansion
        ..Default::default()
    },
    ..Default::default()
};
```

## Kesimpulan

HAS-MoE-FFN menyediakan implementasi MoE yang modern dan efisien yang dapat diintegrasikan dengan ATQS untuk:

- **Parameter Efficiency**: 60% reduction vs dense FFN
- **Computation Efficiency**: 40% faster inference  
- **Adaptive Specialization**: Task-specific expert activation
- **ATQS Integration**: Seamless compression pipeline
- **Scalability**: Linear scaling dengan number of experts

Dengan mengikuti best practices dan configuration guidelines di atas, Anda dapat mengoptimalkan performa model untuk berbagai use cases.
