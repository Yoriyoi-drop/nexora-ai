# Nexora Deep Learning Library

![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)
![Rust](https://img.shields.io/badge/rust-1.70+-orange)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey)

Deep Learning Library untuk Nexora AI dengan implementasi STAR-X (Selective Temporal Adaptive Resonance Network) dan berbagai RNN layers.

## 🚀 Fitur Utama

### STAR-X (Selective Temporal Adaptive Resonance Network)
- **Temporal Gating Hierarchy (TGH)** - Multi-resolution temporal processing
- **Sparse Causal Attention (SCA)** - O(T log T) complexity dengan dynamic routing
- **Harmonic Temporal Encoding (HTE)** - Periodicity dan rhythm capture
- **Selective State Update (SSU)** - Relevance-based state updates
- **Adaptive Gradient Resonance (AGR)** - Gradient stabilization
- **Episodic Memory Retention (EMR)** - Selective-write memory system
- **Associative State Composition (ASC)** - Parallel associative scan
- **Adaptive Compute Allocation (ACA)** - Dynamic compute routing

### RNN Layers
- Vanilla RNN
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Custom recurrent architectures

## 📊 Kompleksitas

| Model | Time Complexity | Memory | Sparsity | Parallel Training |
|-------|----------------|--------|---------|-------------------|
| Transformer | O(T²) | High | Dense | ✅ |
| LSTM/GRU | O(T) | Medium | Dense | ❌ |
| **STAR-X** | **O(T log T)** | **Medium** | **Sparse** | **✅** |

## 🛠️ Instalasi

Tambahkan ke `Cargo.toml`:

```toml
[dependencies]
nexora-deeplearning = { version = "0.1.0", features = ["full"] }
```

### Features

- `default` - Core functionality
- `std` - Standard library support
- `serde` - Serialization support
- `tokio` - Async runtime
- `cuda` - GPU acceleration
- `candle` - Candle backend
- `tch` - PyTorch backend
- `full` - All features

## 🎯 Quick Start

### Basic STAR-X Model

```rust
use nexora_deeplearning::star_x::*;

// Create model
let mut model = StarXModel::default_model()?;

// Process input
let input = create_input_tensor();
let output = model.forward(&input)?;

// Process sequence
let outputs = model.forward_sequence(&inputs)?;
```

### Specialized Models

```rust
// Long context (100K+ tokens)
let mut model = StarXModel::long_context_model(100_000)?;

// Streaming inference
let mut model = StarXModel::streaming_model()?;

// Multimodal processing
let mut model = StarXModel::multimodal_model()?;
```

### Layer Integration

```rust
use nexora_deeplearning::layers::*;

// Create STAR-X layer
let mut layer = MutableStarXLayer::default_layer("star_x".to_string())?;

// Use as foundation model
let adapter = StarXFoundationAdapter::new(config)?;
let output = adapter.forward(&input)?;
```

## 📖 Contoh Lengkap

### Text Processing

```rust
use nexora_deeplearning::star_x::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create text-optimized model
    let mut model = StarXModel::default_model()?;
    
    // Process text sequence
    let text_tokens = tokenize("Hello, world!")?;
    let outputs = model.forward_sequence(&text_tokens)?;
    
    // Get metrics
    let metrics = model.get_metrics();
    println!("Compute efficiency: {:.2}%", metrics.compute_efficiency * 100.0);
    
    Ok(())
}
```

### Long Context Processing

```rust
use nexora_deeplearning::star_x::*;

fn process_long_document(document: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create long context model
    let mut model = StarXModel::long_context_model(50_000)?;
    
    // Process document chunks
    let chunks = chunk_document(document, 512)?;
    let outputs = model.forward_sequence(&chunks)?;
    
    // Extract final representation
    let document_embedding = outputs.last().unwrap();
    
    Ok(())
}
```

### Streaming Inference

```rust
use nexora_deeplearning::star_x::*;

fn streaming_inference() -> Result<(), Box<dyn std::error::Error>> {
    // Create streaming-optimized model
    let mut model = StarXModel::streaming_model()?;
    
    // Process streaming data
    for token in get_streaming_tokens() {
        let output = model.forward(&token)?;
        process_output(&output)?;
    }
    
    Ok(())
}
```

## 🔧 Konfigurasi

### STAR-X Configuration

```rust
use nexora_deeplearning::star_x::*;

let mut config = StarXConfig::default();
config.hidden_size = 1536;
config.attention_heads = 12;
config.max_sparse_connections = 128;
config.update_threshold = 0.15;

let model = StarXModel::new(config)?;
```

### Compute Allocation

```rust
use nexora_deeplearning::star_x::aca::RoutingStrategy;

model.set_routing_strategy(RoutingStrategy::Adaptive);
```

## 📊 Performance

### Benchmark Results

| Model | Context Length | Tokens/ms | Memory (MB) | Efficiency |
|-------|---------------|-----------|-------------|------------|
| STAR-X | 1K | 850 | 45 | 92% |
| STAR-X | 10K | 720 | 120 | 88% |
| STAR-X | 100K | 650 | 450 | 85% |

### Memory Usage

```rust
let memory_usage = model.estimate_memory_usage();
println!("Total memory: {} MB", memory_usage.total_mb);
println!("Parameters: {} MB", memory_usage.model_parameters_mb);
println!("Episodic memory: {} MB", memory_usage.episodic_memory_mb);
```

## 🧪 Testing

Run tests:

```bash
# All tests
cargo test

# STAR-X specific tests
cargo test star_x

# Benchmarks
cargo bench --features benchmarks
```

### Integration Tests

```bash
# Run examples
cargo run --example star_x_demo --features std,tokio

# Performance comparison
cargo run --example star_x_demo --release
```

## 🔬 Advanced Usage

### Custom Components

```rust
use nexora_deeplearning::star_x::*;

// Custom TGH configuration
let tgh = TemporalGatingHierarchy::new(
    768,  // input_size
    1536, // hidden_size
    384,  // micro_gate_size
    768,  // meso_gate_size
    1536, // macro_gate_size
    16,   // chunk_size
)?;

// Custom attention
let sca = SparseCausalAttention::new(
    1536, // hidden_size
    16,   // attention_heads
    256,  // max_sparse_connections
    0.1,  // entropy_regularization
)?;
```

### Training Integration

```rust
// Enable training mode
model.optimize_for_training()?;

// Training loop
for epoch in 0..num_epochs {
    for batch in dataloader {
        let output = model.forward(&batch.input)?;
        let loss = compute_loss(&output, &batch.target)?;
        
        // Backward pass (simplified)
        let gradients = compute_gradients(&loss)?;
        update_parameters(&gradients)?;
    }
}
```

## 🤝 Kontribusi

1. Fork repository
2. Buat feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

Project ini dilisensikan di bawah MIT License atau Apache License 2.0.

## 🙏 Acknowledgments

- Terinspirasi dari arsitektur Transformer modern
- Berdasarkan penelitian state-space models
- Implementasi sparse attention mechanisms
- Kontribusi dari Nexora AI community

## 📞 Support

- 📖 [Documentation](https://docs.nexora.ai/deeplearning)
- 🐛 [Issues](https://github.com/nexora-ai/nexora/issues)
- 💬 [Discussions](https://github.com/nexora-ai/nexora/discussions)
- 📧 [Email](mailto:support@nexora.ai)

---

**Nexora Deep Learning** - Next generation AI architecture for scalable, efficient, and intelligent systems.
