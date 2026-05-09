//! SPARO Framework Example
//! 
//! Contoh penggunaan lengkap SPARO untuk pelatihan model bahasa

use nexora_foundation::alignment::sparo::prelude::*;
use nexora_foundation::alignment::sparo::trainer;
use nexora_foundation::alignment::sparo::data;
use nexora_foundation::alignment::sparo::kto;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    println!("🚀 SPARO Framework Example");
    println!("==========================\n");

    // 1. Setup konfigurasi
    println!("📋 Setting up configuration...");
    let sparo_config = SparoConfig {
        alpha: 0.4,  // DPO weight
        beta: 0.3,   // KTO weight
        gamma: 0.3,   // IPO weight
        learning_rate: 1e-4,
        batch_size: 16,
        max_iterations: 10, // Kecil untuk demo
        convergence_threshold: 1e-6,
    };

    // Validasi konfigurasi
    nexora_foundation::alignment::sparo::trainer::utils::validate_config(&sparo_config)?;
    println!("✅ Configuration validated");

    // 2. Buat model student dan teacher
    println!("\n🤖 Creating models...");
    let student_model = PolicyModel::new(
        uuid::Uuid::new_v4(),
        (512, 512) // Dimensi model
    );
    
    let teacher_model = PolicyModel::new(
        uuid::Uuid::new_v4(),
        (512, 512)
    );
    
    println!("✅ Models created");

    // 3. Inisialisasi SPARO trainer
    println!("\n🎯 Initializing SPARO trainer...");
    let mut trainer = SparoTrainer::new(sparo_config, student_model, teacher_model)?;
    println!("✅ Trainer initialized");

    // 4. Generate training prompts
    println!("\n📝 Generating training prompts...");
    let prompts = nexora_foundation::alignment::sparo::trainer::utils::generate_training_prompts(20);
    println!("Generated {} prompts", prompts.len());
    
    // Tampilkan beberapa contoh prompt
    for (i, prompt) in prompts.iter().take(3).enumerate() {
        println!("  {}: {}", i + 1, prompt);
    }

    // 5. Setup dataset
    println!("\n📚 Setting up dataset...");
    let dataset_config = DatasetConfig {
        max_traces_per_batch: 50,
        min_quality_score: 0.3,
        enable_augmentation: true,
        shuffle_data: true,
        validation_split: 0.2,
    };
    
    let mut dataset = SparoDataset::new(dataset_config);
    
    // Generate sample data
    let (sample_traces, sample_feedback) = nexora_foundation::alignment::sparo::data::utils::generate_sample_data(50);
    dataset.add_data(sample_traces, sample_feedback)?;
    
    // Augmentasi data
    dataset.augment_dataset()?;
    
    let stats = dataset.get_stats();
    println!("Dataset stats:");
    println!("  Total traces: {}", stats.total_traces);
    println!("  Total feedback: {}", stats.total_feedback);
    println!("  Validation traces: {}", stats.validation_traces);
    println!("  Average trace length: {:.2}", stats.avg_trace_length);

    // 6. Jalankan training
    println!("\n🏃 Starting SPARO training...");
    println!("=========================\n");
    
    let training_result = trainer.train(&prompts)?;
    
    println!("\n🎉 Training completed!");
    println!("==================");
    println!("Final iteration: {}", training_result.final_state.iteration);
    println!("Final loss: {:.6}", training_result.final_state.current_loss);
    println!("Best loss: {:.6}", training_result.final_state.best_loss);
    println!("Converged: {}", training_result.final_state.converged);

    // 7. Tampilkan training statistics
    println!("\n📊 Training Statistics");
    println!("=====================");
    let training_stats = trainer.get_training_stats();
    println!("Total iterations: {}", training_stats.iterations);
    println!("Average loss: {:.6}", training_stats.avg_loss);
    println!("Loss trend: {:.6}", training_stats.loss_trend);
    println!("Total samples processed: {}", training_stats.total_samples);

    // 8. Analisis metrics
    println!("\n📈 Metrics Analysis");
    println!("==================");
    let metrics_analysis = nexora_foundation::alignment::sparo::trainer::utils::analyze_metrics(&training_result.metrics_history);
    println!("Final loss: {:.6}", metrics_analysis.final_loss);
    println!("Best loss: {:.6}", metrics_analysis.best_loss);
    println!("Average loss: {:.6}", metrics_analysis.avg_loss);
    println!("Convergence rate: {:.2}%", metrics_analysis.convergence_rate * 100.0);
    println!("Improvement rate: {:.2}%", metrics_analysis.improvement_rate * 100.0);
    
    println!("\nComponent balance:");
    println!("  DPO contribution: {:.1}%", metrics_analysis.component_balance.dpo_contribution * 100.0);
    println!("  KTO contribution: {:.1}%", metrics_analysis.component_balance.kto_contribution * 100.0);
    println!("  IPO contribution: {:.1}%", metrics_analysis.component_balance.ipo_contribution * 100.0);

    // 9. Demo komponen individual
    println!("\n🔧 Component Demonstrations");
    println!("==========================");
    
    demo_dpo_component()?;
    demo_kto_component()?;
    demo_ipo_component()?;
    demo_rlvf_component()?;
    demo_spin_component()?;
    demo_rlaif_component()?;

    // 10. Save checkpoint
    println!("\n💾 Saving checkpoint...");
    let checkpoint_path = "sparo_checkpoint.json";
    trainer.save_checkpoint(checkpoint_path)?;
    println!("✅ Checkpoint saved to {}", checkpoint_path);

    println!("\n🎊 SPARO example completed successfully!");
    Ok(())
}

fn demo_dpo_component() -> anyhow::Result<()> {
    println!("\n📌 DPO (Direct Preference Optimization) Demo");
    println!("============================================");
    
    let model = PolicyModel::new(uuid::Uuid::new_v4(), (100, 100));
    let config = DpoConfig::default();
    let mut trainer = DpoTrainer::new(model, config);
    
    // Buat preference pairs
    let pairs = vec![
        PreferencePair {
            id: uuid::Uuid::new_v4(),
            prompt: "What is 2+2?".to_string(),
            chosen: "2+2 = 4".to_string(),
            rejected: "2+2 = 5".to_string(),
            chosen_logprob: -1.0,
            rejected_logprob: -2.0,
            reference_chosen_logprob: -1.1,
            reference_rejected_logprob: -2.1,
        }
    ];
    
    let loss = trainer.training_step(&pairs)?;
    println!("DPO loss: {:.6}", loss);
    println!("✅ DPO demo completed");
    
    Ok(())
}

fn demo_kto_component() -> anyhow::Result<()> {
    println!("\n📌 KTO (Kahneman-Tversky Optimization) Demo");
    println!("===========================================");
    
    let model = PolicyModel::new(uuid::Uuid::new_v4(), (100, 100));
    let config = KtoConfig::default();
    let mut trainer = KtoTrainer::new(model, config);
    
    // Buat independent labels
    let labels = vec![
        IndependentLabel {
            id: uuid::Uuid::new_v4(),
            prompt: "Explain gravity".to_string(),
            response: "Gravity is a force that attracts objects toward each other".to_string(),
            is_good: true,
            confidence: 0.9,
            log_probability: -1.0,
            reference_log_probability: -1.2,
        }
    ];
    
    let loss = trainer.training_step(&labels)?;
    println!("KTO loss: {:.6}", loss);
    
    // Tampilkan distribusi
    let stats = nexora_foundation::alignment::sparo::kto::utils::analyze_distribution(&labels);
    println!("Label distribution:");
    println!("  Total: {}", stats.total);
    println!("  Good: {} ({:.1}%)", stats.good_count, stats.good_ratio * 100.0);
    println!("  Bad: {} ({:.1}%)", stats.bad_count, stats.bad_ratio * 100.0);
    
    println!("✅ KTO demo completed");
    
    Ok(())
}

fn demo_ipo_component() -> anyhow::Result<()> {
    println!("\n📌 IPO (Identity Preference Optimization) Demo");
    println!("==============================================");
    
    let model = PolicyModel::new(uuid::Uuid::new_v4(), (100, 100));
    let config = IpoConfig::default();
    let mut trainer = IpoTrainer::new(model, config);
    
    // Generate constraints
    let prompts = vec![
        "What is AI?".to_string(),
        "Explain machine learning".to_string(),
    ];
    
    trainer.generate_constraints(&prompts)?;
    let loss = trainer.training_step()?;
    
    println!("IPO loss: {:.6}", loss);
    
    let stats = trainer.get_regularization_stats();
    println!("Regularization stats:");
    println!("  Constraints: {}", stats.num_constraints);
    println!("  Avg KL divergence: {:.6}", stats.avg_kl_divergence);
    println!("  Avg similarity: {:.6}", stats.avg_similarity);
    
    println!("✅ IPO demo completed");
    
    Ok(())
}

fn demo_rlvf_component() -> anyhow::Result<()> {
    println!("\n📌 RLVF (Reinforcement Learning from Verifiable Feedback) Demo");
    println!("================================================================");
    
    let config = RlvfConfig::default();
    let manager = RlvfManager::new(config);
    
    // Buat reasoning trace
    let trace = ReasoningTrace {
        id: uuid::Uuid::new_v4(),
        prompt: "Solve: 2 + 2 = ?".to_string(),
        steps: vec![
            ReasoningStep {
                id: uuid::Uuid::new_v4(),
                content: "Step 1: I need to add 2 and 2".to_string(),
                step_number: 1,
                timestamp: chrono::Utc::now(),
            },
            ReasoningStep {
                id: uuid::Uuid::new_v4(),
                content: "Step 2: 2 + 2 = 4".to_string(),
                step_number: 2,
                timestamp: chrono::Utc::now(),
            },
        ],
        final_answer: "The answer is 4".to_string(),
        created_at: chrono::Utc::now(),
    };
    
    // Verifikasi trace
    let step_feedbacks = manager.verify_trace(&trace)?;
    let judge_feedbacks = manager.feedback_to_judge_feedback(&step_feedbacks);
    
    println!("Verification results:");
    for (i, feedback) in step_feedbacks.iter().enumerate() {
        println!("  Step {}: Score = {:.3}, Verifiable = {}", 
            i + 1, feedback.overall_score, feedback.is_verifiable);
    }
    
    let stats = manager.get_verification_stats(&step_feedbacks);
    println!("Verification stats:");
    println!("  Total steps: {}", stats.total_steps);
    println!("  Verifiable steps: {}", stats.verifiable_steps);
    println!("  Correct steps: {}", stats.correct_steps);
    println!("  Accuracy: {:.1}%", stats.accuracy * 100.0);
    
    println!("✅ RLVF demo completed");
    
    Ok(())
}

fn demo_spin_component() -> anyhow::Result<()> {
    println!("\n📌 SPIN (Self-Play with Instruction Following) Demo");
    println!("==================================================");
    
    let student = PolicyModel::new(uuid::Uuid::new_v4(), (100, 100));
    let teacher = PolicyModel::new(uuid::Uuid::new_v4(), (100, 100));
    let config = SpinConfig::default();
    let mut trainer = SpinTrainer::new(config, student, teacher);
    
    let prompts = vec![
        "What is the capital of France?".to_string(),
        "Explain photosynthesis".to_string(),
    ];
    
    // Jalankan tournament
    let tournament = trainer.run_tournament(&prompts)?;
    let loss = trainer.update_models(&tournament)?;
    
    println!("Tournament results:");
    println!("  Total games: {}", tournament.games.len());
    println!("  Student wins: {}", tournament.student_wins);
    println!("  Teacher wins: {}", tournament.teacher_wins);
    println!("  Draws: {}", tournament.draws);
    println!("  Student win rate: {:.1}%", tournament.student_win_rate * 100.0);
    println!("  Improvement score: {:.3}", tournament.improvement_score);
    println!("  SPIN loss: {:.6}", loss);
    
    let stats = trainer.get_training_stats(&tournament);
    println!("Training stats:");
    println!("  Avg student score: {:.3}", stats.avg_student_score);
    println!("  Avg teacher score: {:.3}", stats.avg_teacher_score);
    println!("  Convergence rate: {:.3}", stats.convergence_rate);
    
    println!("✅ SPIN demo completed");
    
    Ok(())
}

fn demo_rlaif_component() -> anyhow::Result<()> {
    println!("\n📌 RLAIF (Reinforcement Learning from AI Feedback) Demo");
    println!("======================================================");
    
    let config = RlaifConfig::default();
    let manager = RlaifManager::new(config);
    
    // Buat reasoning trace
    let trace = ReasoningTrace {
        id: uuid::Uuid::new_v4(),
        prompt: "Explain the concept of machine learning".to_string(),
        steps: vec![
            ReasoningStep {
                id: uuid::Uuid::new_v4(),
                content: "Machine learning is a subset of AI".to_string(),
                step_number: 1,
                timestamp: chrono::Utc::now(),
            },
            ReasoningStep {
                id: uuid::Uuid::new_v4(),
                content: "It involves training algorithms on data".to_string(),
                step_number: 2,
                timestamp: chrono::Utc::now(),
            },
        ],
        final_answer: "Machine learning enables computers to learn from experience".to_string(),
        created_at: chrono::Utc::now(),
    };
    
    // Generate AI feedback
    let feedback = manager.generate_feedback(&trace)?;
    
    println!("AI Feedback results:");
    for (i, fb) in feedback.iter().enumerate() {
        match &fb.feedback_type {
            FeedbackType::Independent { step_id, is_good, confidence } => {
                println!("  Feedback {}: Independent - Step {} = {}, Confidence = {:.2}", 
                    i + 1, step_id, is_good, confidence);
            },
            FeedbackType::Pairwise { preferred, rejected, confidence } => {
                println!("  Feedback {}: Pairwise - Preferred {} over {}, Confidence = {:.2}", 
                    i + 1, preferred, rejected, confidence);
            },
        }
    }
    
    let stats = manager.get_judge_stats();
    println!("Judge stats:");
    println!("  Number of judges: {}", stats.num_judges);
    println!("  Judge models: {:?}", stats.judge_names);
    println!("  Average confidence: {:.2}", stats.avg_confidence);
    
    println!("✅ RLAIF demo completed");
    
    Ok(())
}
