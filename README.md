# Nexora-AI

LLM (Large Language Model) implementation from scratch in Rust.

## Features

### Core Components
- **BPE Tokenizer**: Byte Pair Encoding with vocabulary save/load
- **Tensor Operations**: 1D, 2D, 3D tensors with element-wise ops, matmul, transpose, softmax
- **Transformer Layers**:
  - RMSNorm (Root Mean Square Normalization)
  - RoPE (Rotary Positional Embedding)
  - GQA (Grouped Query Attention)
  - SwiGLU FFN (Feed-Forward Network)
  - Full Transformer Block with residual connections

### Inference
- **Causal Mask**: Prevents tokens from seeing future positions
- **KV Cache**: Efficient autoregressive inference with key-value caching
- **Logits Projection**: Hidden state to vocabulary logits
- **Sampling**: Greedy, temperature, top-k, top-p sampling strategies
- **Autoregressive Generation**: Complete text generation loop
- **Model Save/Load**: Binary format for model weights

### Training (Foundation)
- **Cross-Entropy Loss**: Standard language modeling loss (verified with random logits ~ln(vocab_size))
- **AdamW Optimizer**: Adam with weight decay
- **Gradient Checking**: Numerical gradient verification using finite differences
- **Training Loop**: Forward pass, loss computation, backward pass (placeholder)

### Database Layer
- **PostgreSQL Support**: Full PostgreSQL driver implementation
- **SQLite Support**: SQLite database driver
- **Connection Pooling**: Database connection pool management
- **Query Builder**: Type-safe query construction

### API Layer
- **REST API**: HTTP API server with authentication
- **Middleware**: CORS, compression, rate limiting
- **Metrics**: Performance monitoring and statistics
- **WebSocket**: Real-time communication support

### Memory Management
- **LRU Cache**: Least Recently Used memory caching
- **Episodic Memory**: Context-aware memory storage
- **Memory Compression**: Efficient memory utilization

### Model Orchestration
- **Task Decomposition**: Automatic task breakdown
- **Model Selection**: Intelligent model routing
- **Parallel Execution**: Concurrent task processing
- **Performance Tracking**: Model performance analytics

## Project Structure

```
nexora-ai/
├── core/                 # Core controller and types
│   ├── controller.rs     # Main system controller
│   ├── types.rs          # Core data structures
│   ├── context.rs        # Context management
│   └── memory.rs         # Memory management
├── tokenizer/            # BPE tokenizer implementation
│   ├── bpe_tokenizer.rs  # BPE algorithm
│   ├── pretokenizer.rs   # Text preprocessing
│   └── vocab_builder.rs  # Vocabulary management
├── model/                # Model implementations
│   ├── tensor_impl.rs    # Tensor operations
│   ├── transformer_impl.rs # Transformer layers
│   └── orchestrator_impl.rs # Model orchestration
├── models/               # Specialized models
│   ├── attention.rs      # Attention mechanisms
│   ├── layers.rs         # Neural network layers
│   └── specialist.rs     # Domain-specific models
├── database/             # Database layer
│   ├── postgres.rs        # PostgreSQL driver
│   ├── sqlite.rs          # SQLite driver
│   └── connection_pool.rs # Connection management
├── api/                   # API layer
│   ├── server.rs          # HTTP server
│   ├── handlers.rs        # Request handlers
│   └── middleware.rs      # API middleware
├── memory/                # Memory systems
│   ├── cache.rs           # Caching mechanisms
│   ├── episodic.rs        # Episodic memory
│   └── compression.rs    # Memory compression
├── nexora-ai/             # Main library
│   ├── lib.rs             # Library entry point
│   └── cli.rs             # Command line interface
└── Cargo.toml             # Workspace configuration
```

## Building

```bash
# Build the entire project
cargo build

# Build with optimizations
cargo build --release

# Run tests
cargo test

# Check compilation without building
cargo check

# Format code
cargo fmt

# Run linter
cargo clippy
```

## Running Tests

```bash
cargo test
```

Test coverage includes:
- Tokenizer tests (BPE encoding/decoding, save/load)
- Tensor tests (operations, matmul, softmax, initialization)
- Layer tests (RMSNorm, RoPE, GQA, SwiGLU, TransformerBlock)
- Database tests (PostgreSQL, SQLite, connection pooling)
- API tests (REST endpoints, authentication, middleware)
- Memory tests (LRU cache, episodic memory, compression)

## Usage

### Basic Usage

```rust
use nexora_ai::{NexoraAI, ModelFactory};

// Create a new Nexora AI instance
let ai = NexoraAI::new()?;

// Process text
let result = ai.process_request("Hello, world!")?;
println!("{}", result);
```

### API Server

```bash
# Start the API server
cargo run --bin nexora-ai -- --server

# Server will be available at http://localhost:8080
```

### Model Training

```rust
use nexora_model::{ModelFactory, TransformerConfig};

// Create a transformer model
let config = TransformerConfig {
    vocab_size: 50257,
    hidden_size: 768,
    num_heads: 12,
    num_layers: 12,
    max_seq_len: 2048,
};

let model = ModelFactory::create_transformer(&config)?;
```

## Model Architecture

```
Input IDs → Token Embedding → Position Embedding
              ↓
        Transformer Blocks (N layers)
              ↓
              RMSNorm
              ↓
           LM Head
              ↓
            Logits
              ↓
           Sampling
              ↓
          Output Text
```

Each Transformer Block:
```
Input → RMSNorm → GQA Attention → Residual → RMSNorm → SwiGLU FFN → Residual → Output
```

## Training Note

The training components are implemented as foundations:
- **Cross-Entropy Loss**: Verified to produce ~ln(vocab_size) for random models
- **AdamW Optimizer**: Full implementation with bias correction and weight decay
- **Gradient Checking**: Numerical gradient verification utility for validating backward passes

Full backpropagation through all layers requires implementing backward passes for:
- Logits projection backward
- RMSNorm backward
- GQA Attention backward
- SwiGLU FFN backward
- Embedding backward

The gradient check utility is ready to validate each backward pass as they are implemented.

## Configuration

### Default Model Configuration
```rust
TransformerConfig {
    vocab_size: 50257,
    hidden_size: 768,
    num_heads: 12,
    num_kv_heads: 4,  // GQA ratio 3:1
    num_layers: 12,
    max_seq_len: 2048,
    intermediate_size: 3072,
    rope_theta: 10000.0,
    use_cache: true,
}
```

### Database Configuration
```rust
DatabaseConfig {
    host: "localhost".to_string(),
    port: 5432,
    database: "nexora".to_string(),
    username: "nexora".to_string(),
    password: "password".to_string(),
    ssl_mode: SslMode::Prefer,
    max_connections: 10,
    min_connections: 1,
    connection_timeout: Duration::from_secs(30),
}
```

### API Configuration
```rust
ApiConfig {
    host: "0.0.0.0".to_string(),
    port: 8080,
    workers: 4,
    enable_cors: true,
    enable_metrics: true,
    enable_compression: true,
    rate_limit: Some(RateLimit {
        requests_per_minute: 60,
        burst_size: 10,
    }),
}
```

## Performance

The pure Rust implementation provides:
- **Memory Safety**: No memory leaks or buffer overflows
- **Thread Safety**: Safe concurrent operations
- **Zero-Copy**: Efficient data sharing where possible
- **Async Support**: Non-blocking I/O operations
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Development

### Code Quality
- **100% Rust**: No C/C++ dependencies
- **Memory Safe**: Guaranteed by Rust's ownership system
- **Thread Safe**: Safe concurrent programming
- **Type Safe**: Strong type system prevents bugs
- **Test Coverage**: Comprehensive test suite

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run `cargo fmt` and `cargo clippy`
6. Submit a pull request

## License

MIT License - Feel free to use for learning and research.

## Acknowledgments

- Rust community for excellent tooling and ecosystem
- Transformer paper authors for the foundational architecture
- OpenAI for popularizing the transformer architecture
- All contributors who helped make this project possible
