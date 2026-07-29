use nexora_deeplearning::quantization::QFormat;
use nexora_training::{Trainer, TrainerConfig};
use nexora_transformer::{CausalLM, TransformerConfig};
use std::path::PathBuf;

fn tiny_model() -> CausalLM {
    CausalLM::new(TransformerConfig {
        vocab_size: 64,
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        max_seq_len: 64,
        intermediate_size: 64,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
        use_cache: false,
        num_experts: 0,
        top_k_experts: 0,
        expert_intermediate_size: 0,
        shared_expert: 0,
        quantization: QFormat::F16,
        use_half_precision: true,
        use_domain_experts: false,
        shard: Default::default(),
    })
}

fn tmp_path() -> PathBuf {
    std::env::temp_dir().join(format!("nexora_test_{}", std::process::id()))
}

#[test]
fn test_training_visual_demo() {
    let model = tiny_model();
    let config = TrainerConfig {
        learning_rate: 0.01,
        max_steps: 20,
        seq_length: 16,
        vocab_size: 64,
        batch_size: 1,
        save_path: None,
        save_every: 1000,
        report_every: 1000,
        weight_decay: 0.0,
        max_grad_norm: None,
        warmup_steps: 3,
        val_every_steps: 1000,
        early_stop_patience: 10,
        use_gpu: true,
        num_replicas: 1,
    };

    let mut trainer = Trainer::with_model(model, config);
    trainer.prepare();

    // Train on a simple counting pattern: 1,2,3,4,5,6,7,8,9,10,11
    let pattern: Vec<u32> = (1..12).collect();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TRAINING DEMO — 20 steps, 32-hidden, 2-layer");
    println!("  Pattern: 1 2 3 4 5 6 7 8 9 10 11");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Step   |   Loss   |  Avg Loss  |  Δ");
    println!("────────┼──────────┼────────────┼──────");

    let mut losses: Vec<f32> = Vec::new();
    for step in 0..20 {
        let (input, target) = trainer.prepare_batch(&pattern);
        if let Some(loss) = trainer.train_batch(&input, &target) {
            losses.push(loss);
            let avg = trainer.avg_loss();
            let delta = if losses.len() > 1 {
                losses[losses.len() - 2] - loss
            } else {
                0.0
            };
            println!(
                "  {:>4}   |  {:.4}  |  {:.4}   | {:.4}",
                step + 1,
                loss,
                avg,
                delta
            );
        }
    }

    println!("────────┴──────────┴────────────┴──────");
    let first = losses.first().copied().unwrap_or(0.0);
    let last = losses.last().copied().unwrap_or(0.0);
    println!(
        "  First: {:.4} → Last: {:.4}  (Δ: {:.4})",
        first,
        last,
        first - last
    );

    assert!(
        last < first,
        "Loss should decrease: first={:.4} last={:.4}",
        first,
        last
    );
    println!("  ✅ Loss decreased! Training works!\n");
}

#[test]
fn test_training_loss_decreases() {
    let model = tiny_model();
    let config = TrainerConfig {
        learning_rate: 0.01,
        max_steps: 50,
        seq_length: 16,
        vocab_size: 64,
        batch_size: 2,
        save_path: None,
        save_every: 1000,
        report_every: 1000,
        weight_decay: 0.0,
        max_grad_norm: None,
        warmup_steps: 5,
        val_every_steps: 1000,
        early_stop_patience: 10,
        use_gpu: true,
        num_replicas: 1,
    };

    let mut trainer = Trainer::with_model(model, config);
    trainer.prepare();

    let pattern: Vec<u32> = (1..12).collect();
    let mut losses: Vec<f32> = Vec::new();

    for _ in 0..50 {
        let (input, target) = trainer.prepare_batch(&pattern);
        if let Some(loss) = trainer.train_batch(&input, &target) {
            losses.push(loss);
        }
    }

    assert!(
        losses.len() >= 5,
        "Should have multiple loss values, got {}",
        losses.len()
    );

    let first_avg: f32 = losses.iter().take(5).sum::<f32>() / 5.0;
    let last_avg: f32 = losses.iter().rev().take(5).sum::<f32>() / 5.0;

    assert!(
        last_avg < first_avg,
        "Loss should decrease: first_avg={:.6}, last_avg={:.6}",
        first_avg,
        last_avg
    );

    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < 2.0,
        "Final loss should be reasonable, got {:.6}",
        final_loss
    );
}

#[test]
fn test_training_save_and_load() {
    let model = tiny_model();
    let config = TrainerConfig {
        learning_rate: 0.01,
        max_steps: 10,
        seq_length: 16,
        vocab_size: 64,
        batch_size: 1,
        save_path: None,
        save_every: 1000,
        report_every: 1000,
        weight_decay: 0.0,
        max_grad_norm: None,
        warmup_steps: 0,
        val_every_steps: 1000,
        early_stop_patience: 10,
        use_gpu: true,
        num_replicas: 1,
    };

    let mut trainer = Trainer::with_model(model, config);
    trainer.prepare();

    let pattern: Vec<u32> = (1..12).collect();
    let (input, target) = trainer.prepare_batch(&pattern);
    let before_loss = trainer.train_batch(&input, &target).unwrap();

    let save_path = tmp_path();
    trainer.save(save_path.to_str().unwrap()).unwrap();

    let loaded_config = TrainerConfig {
        max_steps: 20,
        ..trainer.config.clone()
    };
    let mut loaded_model = tiny_model();
    Trainer::load(&mut loaded_model, save_path.to_str().unwrap()).unwrap();
    let mut loaded = Trainer::with_model(loaded_model, loaded_config);
    loaded.prepare();
    let (input2, target2) = loaded.prepare_batch(&pattern);
    let after_loss = loaded.train_batch(&input2, &target2).unwrap();

    assert!(
        after_loss < before_loss || (after_loss - before_loss).abs() < 0.5,
        "Loaded model should train: before={:.6}, after={:.6}",
        before_loss,
        after_loss
    );

    let _ = std::fs::remove_file(&save_path);
}
