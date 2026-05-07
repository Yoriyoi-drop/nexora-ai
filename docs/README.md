# Nexora AI - Comprehensive AI System

Nexora AI is a comprehensive artificial intelligence system built in Rust that provides advanced capabilities for natural language processing, machine learning, inference, and data management.

## 🚀 Features

### Core Components
- **Multi-Layer Memory System**: Advanced memory management with 4 layers (Short-term, Session, Long-term, Knowledge)
- **Inference Engine**: High-performance neural network inference with GPU acceleration support
- **Training System**: Complete training pipeline with backward passes and gradient optimization
- **Database Layer**: Advanced connection pooling and query optimization
- **API Layer**: RESTful API with comprehensive security and rate limiting

### Advanced Features
- **Error Handling & Recovery**: Comprehensive error management with automatic recovery mechanisms
- **Logging & Monitoring**: Structured logging with real-time metrics and alerting
- **Memory Management**: Advanced memory pooling and leak detection
- **Security**: Multi-layer security with authentication, authorization, and input validation
- **Configuration Management**: Environment-aware configuration with validation and migration

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [API Documentation](#api-documentation)
6. [Architecture](#architecture)
7. [Development](#development)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Contributing](#contributing)

## 🚀 Quick Start

### Prerequisites

- Rust 1.70 or higher
- PostgreSQL (for database features)
- CUDA Toolkit (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/nexora-ai/nexora-ai.git
cd nexora-ai

# Build the project
cargo build --release

# Run tests
cargo test

# Start the server
cargo run --bin nexora-ai
```

### Basic Usage

```rust
use nexora_ai::{NexoraAI, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = Config::from_file("config.toml")?;
    
    // Initialize Nexora AI
    let nexora = NexoraAI::new(config).await?;
    
    // Process a request
    let response = nexora.process_request("Hello, how can you help me?").await?;
    
    println!("Response: {}", response);
    
    Ok(())
}
```

## ⚙️ Configuration

### Environment Variables

```bash
# Core configuration
export NEXORA_ENV=development
export NEXORA_HOST=127.0.0.1
export NEXORA_PORT=8080

# Database configuration
export DATABASE_URL=postgresql://user:password@localhost/nexora

# Security
export NEXORA_SECRET_KEY=your-secret-key-here

# Logging
export RUST_LOG=info
```

### Configuration File (config.toml)

```toml
[server]
host = "127.0.0.1"
port = 8080
max_connections = 1000
enable_tls = false

[database]
url = "postgresql://localhost/nexora"
max_connections = 20
connection_timeout = 30

[models]
model_path = "./models"
cache_size = 1024
enable_gpu = false

[logging]
level = "info"
console = true
file = true
file_dir = "./logs"

[security]
enable_auth = true
rate_limit = 100
enable_cors = true
```

## 📚 Usage Examples

### 1. Basic Inference

```rust
use nexora_ai::inference::InferenceEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = InferenceEngine::new("models/gpt2").await?;
    
    let prompt = "The future of artificial intelligence is";
    let response = engine.generate(prompt, 100).await?;
    
    println!("Generated: {}", response);
    Ok(())
}
```

### 2. Training a Model

```rust
use nexora_ai::models::training::{SimpleNeuralModel, TrainingConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 32,
        epochs: 100,
        ..Default::default()
    };
    
    let mut model = SimpleNeuralModel::new(768, 3072, 768);
    
    // Training loop
    for epoch in 0..config.epochs {
        let loss = model.train_step(&training_data, &labels, config.learning_rate).await?;
        println!("Epoch {}: Loss = {:.4}", epoch, loss);
    }
    
    // Save the trained model
    model.save_model("trained_model.json").await?;
    
    Ok(())
}
```

### 3. Memory Management

```rust
use nexora_ai::memory::{MemoryManager, MemoryLayer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let memory = MemoryManager::new();
    
    // Store different types of memories
    memory.store(MemoryLayer::Short, "user_input", "Hello world").await?;
    memory.store(MemoryLayer::Session, "conversation", "User said hello").await?;
    memory.store(MemoryLayer::Long, "knowledge", "Hello is a greeting").await?;
    
    // Retrieve memories
    if let Some(response) = memory.retrieve(MemoryLayer::Short, "user_input").await? {
        println!("Retrieved: {}", response);
    }
    
    // Search across all memory layers
    let results = memory.search("hello").await?;
    for result in results {
        println!("Found in {:?}: {}", result.layer, result.value);
    }
    
    Ok(())
}
```

### 4. Database Operations

```rust
use nexora_ai::database::{DatabasePool, DatabaseConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = DatabaseConfig {
        database_url: "postgresql://localhost/nexora".to_string(),
        max_connections: 20,
        ..Default::default()
    };
    
    let pool = DatabasePool::new(config).await?;
    
    // Execute a query
    let results = pool.execute_query(
        "SELECT * FROM users WHERE active = $1",
        Some(&[&true])
    ).await?;
    
    // Execute a transaction
    pool.execute_transaction(|tx| {
        Box::pin(async move {
            tx.execute("INSERT INTO logs (message) VALUES ($1)", &[&"Test"])?;
            tx.execute("UPDATE stats SET count = count + 1")?;
            Ok(())
        })
    }).await?;
    
    Ok(())
}
```

### 5. API Server

```rust
use nexora_ai::api::{ApiServer, ApiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ApiConfig {
        host: "127.0.0.1".to_string(),
        port: 8080,
        enable_cors: true,
        enable_metrics: true,
        ..Default::default()
    };
    
    let server = ApiServer::new(config).await?;
    
    // Start the server
    server.start().await?;
    
    Ok(())
}
```

### 6. Error Handling

```rust
use nexora_ai::common::error::{NexoraError, ErrorRecoveryManager};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut recovery_manager = ErrorRecoveryManager::new();
    
    // Handle an error
    let error = NexoraError::NetworkError {
        code: NetworkErrorCode::Timeout,
        message: "Request timeout".to_string(),
        endpoint: "api.example.com".to_string(),
        retry_count: 0,
    };
    
    let action = recovery_manager.handle_error(&error, "api_client").await?;
    
    match action {
        RecoveryAction::RetrySuccess => println!("Retry successful"),
        RecoveryAction::FallbackUsed(fallback) => println!("Using fallback: {}", fallback),
        RecoveryAction::EmergencyShutdown => println!("Emergency shutdown triggered"),
        _ => println!("No action taken"),
    }
    
    Ok(())
}
```

### 7. Monitoring and Logging

```rust
use nexora_ai::common::logging::{LoggingConfig, MonitoringConfig, init_logging_and_monitoring};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let logging_config = LoggingConfig {
        level: "info".to_string(),
        console: true,
        file: true,
        file_dir: Some("./logs".into()),
        ..Default::default()
    };
    
    let monitoring_config = MonitoringConfig {
        enable_metrics: true,
        enable_health_checks: true,
        enable_alerting: true,
        ..Default::default()
    };
    
    let metrics_collector = init_logging_and_monitoring(logging_config, monitoring_config)?;
    
    // Initialize monitoring
    metrics_collector.initialize().await?;
    
    // Add custom metrics
    metrics_collector.add_custom_metric(
        "requests_processed".to_string(),
        42.0,
        "count".to_string(),
        std::collections::HashMap::new()
    ).await;
    
    // Get current metrics
    let metrics = metrics_collector.get_metrics().await;
    println!("CPU Usage: {:.1}%", metrics.cpu_usage_percent);
    
    Ok(())
}
```

## 🔌 API Documentation

### REST API Endpoints

#### Health Check
```http
GET /health
```

#### Inference
```http
POST /api/v1/inference
Content-Type: application/json

{
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7
}
```

#### Training
```http
POST /api/v1/training/start
Content-Type: application/json

{
  "model_type": "simple_neural",
  "config": {
    "learning_rate": 0.001,
    "batch_size": 32
  }
}
```

#### Memory Operations
```http
POST /api/v1/memory/store
Content-Type: application/json

{
  "layer": "session",
  "key": "conversation_1",
  "value": "User said hello"
}
```

```http
GET /api/v1/memory/retrieve/{layer}/{key}
```

```http
GET /api/v1/memory/search?q=hello
```

### WebSocket API

Real-time inference and streaming responses:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/inference');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'generate',
        prompt: 'The future of AI is',
        max_tokens: 100
    }));
};

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    console.log('Generated:', response.text);
};
```

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer                                 │
├─────────────────────────────────────────────────────────────┤
│                    Core Layer                                │
├─────────────────────────────────────────────────────────────┤
│  Inference  │  Training  │  Memory  │  Database  │  Utils    │
├─────────────────────────────────────────────────────────────┤
│                    Common Layer                              │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure                            │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **API Layer** (`nexora-ai`)
   - HTTP server and WebSocket support
   - Request routing and middleware
   - Authentication and authorization
   - Rate limiting and security

2. **Core Layer** (`core`, `inference`, `models`, `memory`, `database`)
   - Business logic and algorithms
   - Neural network inference and training
   - Memory management and storage
   - Database operations and optimization

3. **Common Layer** (`common`)
   - Shared utilities and types
   - Error handling and logging
   - Configuration management
   - Monitoring and metrics

### Data Flow

```
User Request → API Layer → Core Layer → Inference Engine → Response
                ↓           ↓            ↓
            Middleware → Memory → Database → Cache
```

## 🔧 Development

### Project Structure

```
nexora-ai/
├── nexora-ai/          # Main application and API server
├── core/               # Core business logic
├── inference/          # Neural network inference engine
├── models/             # Model implementations and training
├── memory/             # Memory management system
├── database/           # Database layer and connection pooling
├── api/                # HTTP API and middleware
├── common/             # Shared utilities and types
├── tokenizer/          # Text tokenization
└── utils/              # Additional utilities
```

### Building from Source

```bash
# Build all components
cargo build --workspace

# Build with optimizations
cargo build --release

# Run with specific features
cargo run --features gpu,mysql

# Build documentation
cargo doc --open
```

### Code Style

We use the following tools for code quality:

```bash
# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings

# Run tests with coverage
cargo tarpaulin --out Html

# Check for security vulnerabilities
cargo audit
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p nexora-core

# Run tests with specific pattern
cargo test inference

# Run integration tests
cargo test --test integration_tests

# Run benchmarks
cargo bench
```

### Test Coverage

```bash
# Generate coverage report
cargo tarpaulin --out Html

# View coverage report
open tarpaulin-report.html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality
- **Performance Tests**: Benchmarks and load testing
- **Security Tests**: Vulnerability scanning and penetration testing

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=builder /app/target/release/nexora-ai /usr/local/bin/
EXPOSE 8080
CMD ["nexora-ai"]
```

```bash
# Build Docker image
docker build -t nexora-ai .

# Run container
docker run -p 8080:8080 -e DATABASE_URL=postgresql://... nexora-ai
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexora-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nexora-ai
  template:
    metadata:
      labels:
        app: nexora-ai
    spec:
      containers:
      - name: nexora-ai
        image: nexora-ai:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

### Environment Setup

```bash
# Production environment
export NEXORA_ENV=production
export RUST_LOG=warn
export DATABASE_URL=postgresql://prod-db:5432/nexora
export NEXORA_SECRET_KEY=$(openssl rand -hex 32)

# Development environment
export NEXORA_ENV=development
export RUST_LOG=debug
export DATABASE_URL=postgresql://localhost:5432/nexora_dev
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Review Process

- All changes require code review
- Tests must pass for all changes
- Documentation must be updated for API changes
- Security review required for sensitive changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 [Documentation](https://docs.nexora-ai.com)
- 🐛 [Issue Tracker](https://github.com/nexora-ai/nexora-ai/issues)
- 💬 [Discord Community](https://discord.gg/nexora-ai)
- 📧 [Email Support](mailto:support@nexora-ai.com)

## 🙏 Acknowledgments

- The Rust community for excellent tooling and libraries
- Contributors who have helped improve this project
- Open source projects that make this possible

## 📊 Performance Benchmarks

| Component | Operation | Latency | Throughput |
|-----------|-----------|---------|------------|
| Inference | GPT-2 Small | 50ms | 20 req/s |
| Memory | Store/Retrieve | 1ms | 1000 ops/s |
| Database | Query | 5ms | 200 qps |
| API | Request | 10ms | 100 req/s |

## 🔮 Roadmap

### Version 1.0 (Current)
- ✅ Core inference engine
- ✅ Basic training capabilities
- ✅ Memory management system
- ✅ REST API
- ✅ Database integration

### Version 1.1 (Planned)
- 🔄 GPU acceleration
- 🔄 Model fine-tuning
- 🔄 Advanced caching
- 🔄 Performance optimizations

### Version 2.0 (Future)
- 📋 Distributed training
- 📋 Multi-modal support
- 📋 Advanced security features
- 📋 Cloud deployment tools

---

**Built with ❤️ by the Nexora AI Team**
