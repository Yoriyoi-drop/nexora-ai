# SPARO Framework Documentation

## Overview

**SPARO (Self-Play Aligned Reasoning via Prospect-Theoretic Stepwise Optimization)** adalah sebuah paradigma alignment AI yang menggabungkan 6 teknik inovatif untuk pelatihan model bahasa yang lebih efektif dan efisien.

## Komponen Utama

### 1. DPO (Direct Preference Optimization)
- Menghilangkan reward model terpisah
- Mengubah pelatihan preferensi menjadi klasifikasi biner
- Efisien dan scalable

### 2. KTO (Kahneman-Tversky Optimization)
- Menerima label "baik/buruk" secara independen
- Terinspirasi oleh Prospect Theory
- Fleksibel untuk data tidak berpasangan

### 3. IPO (Identity Preference Optimization)
- Regularisasi anti-overfitting
- Menjaga kemampuan generalisasi
- Mencegah model menghafal data preferensi

### 4. RLVF (Reinforcement Learning from Verifiable Feedback)
- Evaluasi kualitas per langkah penalaran
- Feedback dapat diverifikasi otomatis
- Presisi perbaikan yang tinggi

### 5. SPIN (Self-Play with Instruction Following)
- Model berlatih tanding melawan dirinya sendiri
- Peningkatan iteratif tanpa supervisi
- Konvergensi yang stabil

### 6. RLAIF (Reinforcement Learning from AI Feedback)
- AI cerdas sebagai pemberi umpan balik
- Menggantikan peran manusia yang mahal
- Skalabilitas tinggi

## Struktur Modul

```
models/src/sparo/
├── mod.rs              # Main module exports
├── core.rs             # Core components and types
├── dpo.rs              # DPO implementation
├── kto.rs              # KTO implementation
├── ipo.rs              # IPO implementation
├── rlvf.rs             # RLVF implementation
├── spin.rs             # SPIN implementation
├── rlaif.rs            # RLAIF implementation
├── trainer.rs           # Main training orchestration
├── data.rs             # Data handling utilities
├── tests.rs            # Unit tests
└── examples/
    ├── basic_sparo.rs   # Basic usage example
    └── sparo_example.rs # Complete example
```

## Quick Start

### Basic Usage

```rust
use nexora_model::sparo::prelude::*;

fn main() -> anyhow::Result<()> {
    // Create configuration
    let config = SparoConfig::default();
    
    // Create models
    let student = PolicyModel::new(uuid::Uuid::new_v4(), (512, 512));
    let teacher = PolicyModel::new(uuid::Uuid::new_v4(), (512, 512));
    
    // Initialize trainer
    let mut trainer = SparoTrainer::new(config, student, teacher)?;
    
    // Generate training prompts
    let prompts = trainer::utils::generate_training_prompts(100);
    
    // Run training
    let result = trainer.train(&prompts)?;
    
    println!("Training completed! Final loss: {:.6}", result.final_state.current_loss);
    
    Ok(())
}
```

### Component-Specific Usage

```rust
// DPO Example
let model = PolicyModel::new(uuid::Uuid::new_v4(), (100, 100));
let dpo_trainer = DpoTrainer::new(model, DpoConfig::default());

// KTO Example
let kto_trainer = KtoTrainer::new(model, KtoConfig::default());

// RLVF Example
let rlvf_manager = RlvfManager::new(RlvfConfig::default());
let verification = rlvf_manager.verify_trace(&trace)?;

// SPIN Example
let spin_trainer = SpinTrainer::new(config, student, teacher);
let tournament = spin_trainer.run_tournament(&prompts)?;

// RLAIF Example
let rlaif_manager = RlaifManager::new(RlaifConfig::default());
let feedback = rlaif_manager.generate_feedback(&trace)?;
```

## Konfigurasi

### SparoConfig

```rust
pub struct SparoConfig {
    pub alpha: f32,           // DPO weight (default: 0.4)
    pub beta: f32,            // KTO weight (default: 0.3)
    pub gamma: f32,           // IPO weight (default: 0.3)
    pub learning_rate: f32,    // Learning rate (default: 1e-4)
    pub batch_size: usize,     // Batch size (default: 32)
    pub max_iterations: usize,  // Max iterations (default: 1000)
    pub convergence_threshold: f32, // Convergence threshold (default: 1e-6)
}
```

### Component Configurations

Setiap komponen memiliki konfigurasi spesifik:

- **DpoConfig**: beta, regularization_strength, label_smoothing
- **KtoConfig**: reference_point, loss_aversion, probability_weighting
- **IpoConfig**: tau, kl_weight, identity_strength, max_kl
- **RlvfConfig**: step_weight, final_weight, verification_threshold
- **SpinConfig**: rounds_per_iteration, temperature, top_p
- **RlaifConfig**: judge_model, confidence_threshold, max_judges

## Data Flow

1. **Data Collection**: Generate reasoning traces dari model
2. **AI Feedback**: RLAIF generates feedback menggunakan AI judge
3. **Verification**: RLVF verifies steps secara otomatis
4. **Training**: DPO, KTO, dan IPO train model dengan feedback
5. **Self-Play**: SPIN improves model melalui self-play
6. **Iteration**: Proses berulang untuk konvergensi

## Metrics dan Monitoring

### Training Metrics

- **Total Loss**: Weighted combination dari DPO, KTO, dan IPO losses
- **Component Losses**: Individual loss untuk setiap komponen
- **Convergence Rate**: Seberapa cepat model konvergen
- **Improvement Rate**: Persentase peningkatan dari awal training

### Component-Specific Metrics

- **DPO**: Preference accuracy, calibration
- **KTO**: Label distribution, prospect value
- **IPO**: KL divergence, similarity scores
- **RLVF**: Verification accuracy, step quality
- **SPIN**: Win rate, improvement score
- **RLAIF**: Judge confidence, consensus rate

## Best Practices

### 1. Configuration Tuning

- **Alpha/Beta/Gamma**: Sesuaikan berdasarkan data availability
- **Learning Rate**: Mulai dengan 1e-4, adjust berdasarkan convergence
- **Batch Size**: Balance antara memory dan stability

### 2. Data Quality

- Gunakan RLVF untuk filter low-quality traces
- Implementasi data augmentation dengan dataset
- Validasi feedback consistency

### 3. Training Strategy

- Mulai dengan fewer iterations untuk debugging
- Monitor component contributions
- Gunakan checkpointing untuk long training runs

### 4. Monitoring

- Track individual component losses
- Monitor convergence patterns
- Analyze feedback distribution

## Examples

### Complete Training Example

Lihat `models/examples/sparo_example.rs` untuk contoh lengkap:

- Setup konfigurasi
- Inisialisasi models dan trainer
- Data preparation dan augmentation
- Training loop dengan monitoring
- Component demonstrations
- Checkpoint saving

### Basic Usage Example

Lihat `models/src/sparo/examples/basic_sparo.rs` untuk contoh sederhana:

- Minimal setup
- Basic training loop
- Simple metrics

## Testing

Run tests dengan:

```bash
cargo test --package nexora-model sparo
```

Run examples dengan:

```bash
cargo run --package nexora-model --example sparo_example
cargo run --package nexora-model --example basic_sparo
```

## Performance Considerations

### Memory Usage

- Model parameters: O(model_size)
- Training data: O(batch_size × sequence_length)
- Feedback storage: O(num_traces × num_steps)

### Computational Complexity

- DPO: O(batch_size × model_size)
- KTO: O(batch_size × model_size)
- IPO: O(constraints × model_size)
- RLVF: O(steps × verification_complexity)
- SPIN: O(games × model_size)
- RLAIF: O(traces × judge_complexity)

### Optimization Tips

1. **Batch Processing**: Process multiple traces simultaneously
2. **Lazy Evaluation**: Generate feedback on-demand
3. **Memory Pooling**: Reuse allocations
4. **Parallel Processing**: Parallelize independent components

## Troubleshooting

### Common Issues

1. **High Loss**: Check learning rate, data quality
2. **No Convergence**: Verify configuration, increase iterations
3. **Memory Issues**: Reduce batch size, model size
4. **Slow Training**: Optimize data pipeline, use GPU

### Debug Mode

Enable debug logging:

```rust
use log::debug;
env_logger::init();
```

### Validation

Validate configuration:

```rust
trainer::utils::validate_config(&config)?;
data::utils::validate_dataset(&traces, &feedback)?;
```

## Future Extensions

### Planned Features

- Multi-modal support
- Distributed training
- Advanced verifiers
- Custom judge models
- Real-time monitoring dashboard

### Research Directions

- Adaptive weight tuning
- Meta-learning for configuration
- Cross-domain transfer
- Interpretability tools

## References

- DPO: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- KTO: [Kahneman-Tversky Optimization](https://arxiv.org/abs/2402.01306)
- IPO: [Identity Preference Optimization](https://arxiv.org/abs/2310.13612)
- RLVF: [Reinforcement Learning from Verifiable Feedback](https://arxiv.org/abs/2309.17406)
- SPIN: [Self-Play with Instruction Following](https://arxiv.org/abs/2402.03153)
- RLAIF: [Reinforcement Learning from AI Feedback](https://arxiv.org/abs/2309.00267)

## Contributing

1. Fork repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## License

SPARO framework adalah bagian dari Nexora AI project dan dilisensikan under MIT License.
