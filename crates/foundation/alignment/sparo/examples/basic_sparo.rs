//! Basic SPARO Example
//! 
//! Contoh sederhana penggunaan SPARO framework

use nexora_model::sparo::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("🚀 Basic SPARO Example");
    println!("=====================\n");

    // 1. Create simple configuration
    let config = SparoConfig::default();
    println!("✅ Configuration created");

    // 2. Create models
    let student_model = PolicyModel::new(uuid::Uuid::new_v4(), (100, 100));
    let teacher_model = PolicyModel::new(uuid::Uuid::new_v4(), (100, 100));
    println!("✅ Models created");

    // 3. Create trainer
    let mut trainer = SparoTrainer::new(config, student_model, teacher_model)?;
    println!("✅ Trainer initialized");

    // 4. Create sample prompts
    let prompts = vec![
        "What is 2+2?".to_string(),
        "Explain gravity".to_string(),
        "What is AI?".to_string(),
    ];
    println!("✅ {} prompts created", prompts.len());

    // 5. Generate traces
    println!("\n📝 Generating reasoning traces...");
    let traces = trainer.generate_traces(&prompts)?;
    println!("Generated {} traces", traces.len());

    // 6. Generate AI feedback
    println!("\n🤖 Generating AI feedback...");
    let ai_feedback = trainer.generate_ai_feedback(&traces)?;
    println!("Generated {} AI feedback entries", ai_feedback.len());

    // 7. Verify steps
    println!("\n🔍 Verifying steps...");
    let verified_feedback = trainer.verify_steps(&traces)?;
    println!("Generated {} verified feedback entries", verified_feedback.len());

    // 8. Create training batch
    let combined_feedback = trainer.combine_feedback(&ai_feedback, &verified_feedback);
    let batch = TrainingBatch::new(traces, combined_feedback, 0);
    println!("✅ Training batch created with {} traces", batch.size());

    // 9. Train components
    println!("\n🏋️ Training components...");
    let losses = trainer.train_components(&batch)?;
    println!("Training losses:");
    println!("  DPO: {:.6}", losses.dpo_loss);
    println!("  KTO: {:.6}", losses.kto_loss);
    println!("  IPO: {:.6}", losses.ipo_loss);
    println!("  Total: {:.6}", losses.total_loss);

    // 10. Get training stats
    let stats = trainer.get_training_stats();
    println!("\n📊 Training Statistics:");
    println!("  Iterations: {}", stats.iterations);
    println!("  Current loss: {:.6}", stats.current_loss);
    println!("  Best loss: {:.6}", stats.best_loss);
    println!("  Converged: {}", stats.converged);

    println!("\n🎉 Basic SPARO example completed!");
    Ok(())
}
