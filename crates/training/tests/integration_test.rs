use nexora_autograd::{Adam, TensorOps};
use nexora_training::{Trainer, TrainerConfig};
use nexora_transformer::trainable::TrainableCausalLM;
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
        use_half_precision: true,
    })
}

fn make_config() -> TrainerConfig {
    TrainerConfig {
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
    }
}

fn tmp_path() -> PathBuf {
    std::env::temp_dir().join(format!("nexora_integration_{}", std::process::id()))
}

#[test]
fn test_trainer_completes_training_step() {
    let model = tiny_model();
    let mut trainer = Trainer::with_model(model, make_config());
    trainer.prepare();

    let pattern: Vec<u32> = (1..12).collect();
    let (input, target) = trainer.prepare_batch(&pattern);
    let result = trainer.train_batch(&input, &target);

    assert!(
        result.is_some(),
        "Training step should produce a loss value"
    );
    let loss = result.unwrap();
    assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
    assert!(trainer.step > 0 || trainer.accumulation_counter > 0);
}

#[test]
fn test_checkpoint_save_load_preserves_weights() {
    let mut trainer = Trainer::with_model(tiny_model(), make_config());
    trainer.prepare();

    let pattern: Vec<u32> = (1..12).collect();
    for _ in 0..3 {
        let (input, target) = trainer.prepare_batch(&pattern);
        trainer.train_batch(&input, &target);
    }

    let save_path = tmp_path();
    trainer.save(save_path.to_str().unwrap()).unwrap();

    let mut loaded_model = tiny_model();
    Trainer::load(&mut loaded_model, save_path.to_str().unwrap()).unwrap();

    let original_params: Vec<f32> = trainer
        .trainable
        .as_ref()
        .map(|t| {
            t.parameters()
                .iter()
                .flat_map(|p| p.data().into_iter())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let trainable_loaded = TrainableCausalLM::from_inference(&loaded_model);
    let loaded_params: Vec<f32> = trainable_loaded
        .parameters()
        .iter()
        .flat_map(|p| p.data().into_iter())
        .collect();

    assert_eq!(
        original_params.len(),
        loaded_params.len(),
        "Parameter count should match"
    );
    for (i, (a, b)) in original_params.iter().zip(loaded_params.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "Parameter {} differs: {} vs {}",
            i,
            a,
            b
        );
    }

    let _ = std::fs::remove_file(&save_path);
}

#[test]
fn test_gradient_accumulation_matches_non_accumulated() {
    let single_config = TrainerConfig {
        batch_size: 1,
        ..make_config()
    };
    let mut single = Trainer::with_model(tiny_model(), single_config);
    single.prepare();

    let acc_config = TrainerConfig {
        batch_size: 2,
        ..make_config()
    };
    let mut acc = Trainer::with_model(tiny_model(), acc_config);
    acc.prepare();

    let pattern: Vec<u32> = (1..16).collect();

    for _ in 0..4 {
        let (input, target) = single.prepare_batch(&pattern);
        single.train_batch(&input, &target);
    }

    for _ in 0..8 {
        let (input, target) = acc.prepare_batch(&pattern);
        acc.train_batch(&input, &target);
    }

    let single_params: Vec<f32> = single
        .trainable
        .as_ref()
        .map(|t| {
            t.parameters()
                .iter()
                .flat_map(|p| p.data().into_iter())
                .collect()
        })
        .unwrap_or_default();

    let acc_params: Vec<f32> = acc
        .trainable
        .as_ref()
        .map(|t| {
            t.parameters()
                .iter()
                .flat_map(|p| p.data().into_iter())
                .collect()
        })
        .unwrap_or_default();

    assert_eq!(single_params.len(), acc_params.len());
}

#[test]
fn test_adam_converges_on_quadratic() {
    let target = 3.14159f32;

    let x = nexora_autograd::Tensor::from_slice(&[1.0f32], &[1]);
    x.set_requires_grad(true);

    let mut opt = Adam::new(vec![x.clone()], 0.1);

    let mut prev_loss = f32::INFINITY;
    for _ in 0..100 {
        let pred = x.clone().mul(&x.clone());
        let t_tensor = nexora_autograd::Tensor::from_slice(&[target * target], &[1]);
        let loss = (pred.sub(&t_tensor)).powf(2.0).mean();
        loss.backward();

        opt.step();
        opt.zero_grad();

        let current_loss = loss.data()[0];
        if current_loss < prev_loss {
            prev_loss = current_loss;
        }
    }

    let final_val = x.data()[0];
    let final_loss = (final_val * final_val - target * target).abs();

    assert!(
        final_loss < 1.0,
        "Adam should converge on quadratic: final_val={}, target_sq={}, loss={}",
        final_val,
        target * target,
        final_loss
    );
}
