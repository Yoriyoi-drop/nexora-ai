//! ORACLE Trainer - Main Training Orchestration
//! 
//! Koordinator utama yang menggabungkan semua komponen ORACLE:
//! Sparse MoE + MLA backbone, Extended RoPE, FIM + Contrastive pretraining,
//! dan Code DPO alignment untuk pelatihan end-to-end.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::backbone::{OracleBackbone, OracleBackboneConfig};
use super::rope::{ExtendedRope, ExtendedRopeConfig, CrossFilePositionTracker};
use super::pretraining::{OraclePretrainer, OraclePretrainingConfig, TrainingBatch, TrainingExample};
use super::alignment::{CodeDpoTrainer, CodeDpoConfig, CodePreferencePair};
use super::code_utils::CodeTokenizer;
use super::verifiers::CodeVerifier;

/// Konfigurasi lengkap ORACLE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleConfig {
    /// Backbone configuration
    pub backbone: OracleBackboneConfig,
    /// RoPE configuration
    pub rope: ExtendedRopeConfig,
    /// Pretraining configuration
    pub pretraining: OraclePretrainingConfig,
    /// DPO configuration
    pub dpo: CodeDpoConfig,
    /// Training parameters
    pub training: TrainingConfig,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            backbone: OracleBackboneConfig::default(),
            rope: ExtendedRopeConfig::default(),
            pretraining: OraclePretrainingConfig::default(),
            dpo: CodeDpoConfig::default(),
            training: TrainingConfig::default(),
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of pretraining epochs
    pub pretraining_epochs: usize,
    /// Number of alignment epochs
    pub alignment_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Warmup steps
    pub warmup_steps: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Gradient clipping norm
    pub grad_clip_norm: f32,
    /// Checkpoint interval
    pub checkpoint_interval: usize,
    /// Evaluation interval
    pub eval_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            pretraining_epochs: 10,
            alignment_epochs: 5,
            batch_size: 32,
            learning_rate: 1e-4,
            warmup_steps: 1000,
            max_seq_len: 8192,
            grad_clip_norm: 1.0,
            checkpoint_interval: 1000,
            eval_interval: 500,
        }
    }
}

/// ORACLE Training State
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleTrainingState {
    pub current_epoch: usize,
    pub current_step: usize,
    pub phase: TrainingPhase,
    pub pretraining_loss: f32,
    pub alignment_loss: f32,
    pub best_validation_loss: f32,
    pub convergence_patience: usize,
    pub training_metrics: Vec<TrainingMetrics>,
}

impl Default for OracleTrainingState {
    fn default() -> Self {
        Self {
            current_epoch: 0,
            current_step: 0,
            phase: TrainingPhase::Pretraining,
            pretraining_loss: f32::INFINITY,
            alignment_loss: f32::INFINITY,
            best_validation_loss: f32::INFINITY,
            convergence_patience: 0,
            training_metrics: Vec::new(),
        }
    }
}

/// Training phases
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrainingPhase {
    Pretraining,
    Alignment,
    Evaluation,
    Complete,
}

/// Training metrics for tracking progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
    pub learning_rate: f32,
    pub throughput: f32,
    pub timestamp: u64,
}

/// Main ORACLE Trainer
pub struct OracleTrainer {
    config: OracleConfig,
    backbone: OracleBackbone,
    rope: ExtendedRope,
    pretrainer: OraclePretrainer,
    dpo_trainer: CodeDpoTrainer,
    tokenizer: CodeTokenizer,
    verifier: crate::oracle::verifiers::CodeVerifierManager,
    position_tracker: CrossFilePositionTracker,
    training_state: OracleTrainingState,
}

/// Lightweight code analysis result used by the ORACLE facade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAnalysisResult {
    pub quality_score: f32,
    pub findings: Vec<String>,
}

impl OracleTrainer {
    /// Create new ORACLE trainer
    pub fn new(config: OracleConfig, vocab_size: usize) -> Result<Self> {
        // Initialize components
        let backbone = OracleBackbone::new(config.backbone.clone(), vocab_size);
        let rope = ExtendedRope::new(config.rope.clone());
        let pretrainer = OraclePretrainer::new(config.pretraining.clone());
        
        // Create reference model for DPO
        let reference_model = crate::oracle::alignment::CodeModel::new(vocab_size, config.training.max_seq_len);
        let current_model = crate::oracle::alignment::CodeModel::new(vocab_size, config.training.max_seq_len);
        let dpo_trainer = CodeDpoTrainer::new(config.dpo.clone(), current_model, reference_model);
        
        let tokenizer = CodeTokenizer::new();
        let verifier = crate::oracle::verifiers::CodeVerifierManager::new();
        let position_tracker = CrossFilePositionTracker::new();
        
        Ok(Self {
            config,
            backbone,
            rope,
            pretrainer,
            dpo_trainer,
            tokenizer,
            verifier,
            position_tracker,
            training_state: OracleTrainingState::default(),
        })
    }

    /// Analyze code without running the full training pipeline.
    pub async fn analyze_code(&self, code: &str) -> Result<CodeAnalysisResult> {
        Ok(CodeAnalysisResult {
            quality_score: if code.trim().is_empty() { 0.0 } else { 1.0 },
            findings: Vec::new(),
        })
    }
    
    /// Main training loop
    pub fn train(&mut self, training_data: &[TrainingExample]) -> Result<TrainingResult> {
        println!("🚀 Starting ORACLE Training");
        println!("=============================");
        
        let mut result = TrainingResult::new();
        
        // Phase 1: Pretraining with FIM + Contrastive
        println!("\n📚 Phase 1: Pretraining (FIM + Contrastive)");
        println!("===============================================");
        
        self.training_state.phase = TrainingPhase::Pretraining;
        let pretraining_result = self.run_pretraining(training_data)?;
        result.pretraining_result = Some(pretraining_result);
        
        // Phase 2: Alignment with Code DPO
        println!("\n🎯 Phase 2: Alignment (Code DPO)");
        println!("=================================");
        
        self.training_state.phase = TrainingPhase::Alignment;
        let alignment_result = self.run_alignment(training_data)?;
        result.alignment_result = Some(alignment_result);
        
        // Phase 3: Final Evaluation
        println!("\n📊 Phase 3: Final Evaluation");
        println!("==============================");
        
        self.training_state.phase = TrainingPhase::Evaluation;
        let final_metrics = self.final_evaluation(training_data)?;
        result.final_metrics = Some(final_metrics);
        
        self.training_state.phase = TrainingPhase::Complete;
        result.final_state = self.training_state.clone();
        
        println!("\n🎉 ORACLE Training Completed!");
        println!("============================");
        
        Ok(result)
    }
    
    /// Run pretraining phase
    fn run_pretraining(&mut self, training_data: &[TrainingExample]) -> Result<PretrainingResult> {
        let mut result = PretrainingResult::new();
        
        for epoch in 0..self.config.training.pretraining_epochs {
            println!("Pretraining Epoch {}/{}", epoch + 1, self.config.training.pretraining_epochs);
            
            // Create batches
            let batches = self.create_pretraining_batches(training_data)?;
            
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;
            
            for (batch_idx, batch) in batches.iter().enumerate() {
                // Training step
                let loss = self.pretrainer.training_step(batch)?;
                epoch_loss += loss.total_loss;
                batch_count += 1;
                
                // Update position tracker
                self.update_position_tracker(&batch.examples)?;
                
                // Logging
                if batch_idx % 10 == 0 {
                    println!("  Batch {}/{}: Loss = {:.6}", 
                        batch_idx + 1, batches.len(), loss.total_loss);
                }
                
                // Checkpoint
                if self.training_state.current_step % self.config.training.checkpoint_interval == 0 {
                    self.save_checkpoint(format!("pretraining_epoch_{}_step_{}", 
                        epoch, self.training_state.current_step))?;
                }
                
                self.training_state.current_step += 1;
            }
            
            let avg_epoch_loss = epoch_loss / batch_count.max(1) as f32;
            self.training_state.pretraining_loss = avg_epoch_loss;
            
            // Evaluation
            if epoch % self.config.training.eval_interval == 0 {
                let eval_metrics = self.final_evaluation(training_data)?;
                result.epoch_metrics.push(PretrainingMetrics {
                    epoch,
                    avg_loss: eval_metrics.pretraining_loss,
                    fim_loss: 0.0,
                    contrastive_loss: eval_metrics.alignment_loss,
                });
                
                println!("  Eval Loss: {:.6}", eval_metrics.pretraining_loss);
                
                // Early stopping
                if self.check_early_stopping(eval_metrics.pretraining_loss) {
                    println!("  Early stopping triggered");
                    break;
                }
            }
            
            self.training_state.current_epoch = epoch;
        }
        
        result.final_loss = self.training_state.pretraining_loss;
        result.total_epochs = self.training_state.current_epoch + 1;
        
        Ok(result)
    }
    
    /// Run alignment phase
    fn run_alignment(&mut self, training_data: &[TrainingExample]) -> Result<AlignmentResult> {
        let mut result = AlignmentResult::new();
        
        // Generate code preference pairs
        println!("Generating code preference pairs...");
        let preference_pairs = self.generate_preference_pairs(training_data)?;
        println!("Generated {} preference pairs", preference_pairs.len());
        
        for epoch in 0..self.config.training.alignment_epochs {
            println!("Alignment Epoch {}/{}", epoch + 1, self.config.training.alignment_epochs);
            
            // Create batches
            let batches = self.create_alignment_batches(&preference_pairs)?;
            
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;
            
            for (batch_idx, batch) in batches.iter().enumerate() {
                // Training step
                let loss = self.dpo_trainer.training_step(batch)?;
                epoch_loss += loss.total_loss;
                batch_count += 1;
                
                // Logging
                if batch_idx % 10 == 0 {
                    println!("  Batch {}/{}: Loss = {:.6}", 
                        batch_idx + 1, batches.len(), loss.total_loss);
                }
                
                // Checkpoint
                if self.training_state.current_step % self.config.training.checkpoint_interval == 0 {
                    self.save_checkpoint(format!("alignment_epoch_{}_step_{}", 
                        epoch, self.training_state.current_step))?;
                }
                
                self.training_state.current_step += 1;
            }
            
            let avg_epoch_loss = epoch_loss / batch_count.max(1) as f32;
            self.training_state.alignment_loss = avg_epoch_loss;
            
            // Get alignment statistics
            let alignment_stats = self.dpo_trainer.get_alignment_stats(&preference_pairs);
            result.alignment_stats.push(alignment_stats.clone());
            
            println!("  Avg Loss: {:.6}", avg_epoch_loss);
            println!("  Security Score: {:.3}", alignment_stats.avg_security_score);
            println!("  Efficiency Score: {:.3}", alignment_stats.avg_efficiency_score);
            println!("  Quality Score: {:.3}", alignment_stats.avg_code_quality_score);
            
            self.training_state.current_epoch = epoch;
        }
        
        result.final_loss = self.training_state.alignment_loss;
        result.total_epochs = self.training_state.current_epoch + 1;
        
        Ok(result)
    }
    
    /// Final evaluation
    pub fn final_evaluation(&mut self, training_data: &[TrainingExample]) -> Result<EvaluationMetrics> {
        println!("Running comprehensive evaluation...");
        
        // Pretraining evaluation
        let pretraining_batches = self.create_pretraining_batches(training_data)?;
        let mut pretraining_losses = Vec::new();
        
        for batch in pretraining_batches.iter().take(10) { // Sample 10 batches
            let loss = self.pretrainer.training_step(batch)?;
            pretraining_losses.push(loss.total_loss);
        }
        
        // Alignment evaluation
        let preference_pairs = self.generate_preference_pairs(training_data)?;
        let alignment_batches = self.create_alignment_batches(&preference_pairs)?;
        let mut alignment_losses = Vec::new();
        
        for batch in alignment_batches.iter().take(10) { // Sample 10 batches
            let loss = self.dpo_trainer.training_step(batch)?;
            alignment_losses.push(loss.total_loss);
        }
        
        // Code verification evaluation
        let mut verification_scores = Vec::new();
        for example in training_data.iter().take(100) { // Sample 100 examples
            let code = self.tokenizer.decode(&example.tokens)?;
            let result = self.verifier.verify_code(&code, "python")?;
            verification_scores.push(result);
        }
        
        let avg_pretraining_loss = pretraining_losses.iter().sum::<f32>() / pretraining_losses.len().max(1) as f32;
        let avg_alignment_loss = alignment_losses.iter().sum::<f32>() / alignment_losses.len().max(1) as f32;
        let avg_verification_score = verification_scores.iter().sum::<f32>() / verification_scores.len().max(1) as f32;
        
        Ok(EvaluationMetrics {
            pretraining_loss: avg_pretraining_loss,
            alignment_loss: avg_alignment_loss,
            verification_score: avg_verification_score,
            overall_score: (avg_pretraining_loss + avg_alignment_loss) / 2.0 * avg_verification_score,
        })
    }
    
    /// Create pretraining batches
    fn create_pretraining_batches(&self, training_data: &[TrainingExample]) -> Result<Vec<TrainingBatch>> {
        let mut batches = Vec::new();
        
        for chunk in training_data.chunks(self.config.training.batch_size) {
            let batch = TrainingBatch {
                examples: chunk.to_vec(),
            };
            batches.push(batch);
        }
        
        Ok(batches)
    }
    
    /// Create alignment batches
    fn create_alignment_batches(&self, preference_pairs: &[CodePreferencePair]) -> Result<Vec<Vec<CodePreferencePair>>> {
        let mut batches = Vec::new();
        
        for chunk in preference_pairs.chunks(self.config.training.batch_size) {
            batches.push(chunk.to_vec());
        }
        
        Ok(batches)
    }
    
    /// Generate preference pairs for alignment
    fn generate_preference_pairs(&self, training_data: &[TrainingExample]) -> Result<Vec<CodePreferencePair>> {
        let mut pairs = Vec::new();
        
        // Generate code samples from examples
        for example in training_data {
            let prompt = format!("Complete this code: {}", 
                self.tokenizer.decode(&example.tokens)?);
            
            let generated_pairs = self.dpo_trainer.generate_preferences(&prompt, 2)?;
            pairs.extend(generated_pairs);
        }
        
        Ok(pairs)
    }
    
    /// Update position tracker
    fn update_position_tracker(&mut self, examples: &[TrainingExample]) -> Result<()> {
        for example in examples {
            // Extract file ID from metadata (simplified)
            let file_id = example.metadata
                .get("file_id")
                .map(|s| s.as_str())
                .unwrap_or("0")
                .parse::<usize>()
                .unwrap_or(0);
            
            // Add position for each token
            for _ in &example.tokens {
                self.position_tracker.add_position(file_id);
            }
        }
        
        Ok(())
    }
    
    /// Check early stopping
    fn check_early_stopping(&mut self, current_loss: f32) -> bool {
        if current_loss < self.training_state.best_validation_loss {
            self.training_state.best_validation_loss = current_loss;
            self.training_state.convergence_patience = 0;
            false
        } else {
            self.training_state.convergence_patience += 1;
            self.training_state.convergence_patience > 5 // Patience of 5 epochs
        }
    }
    
    /// Save checkpoint
    pub fn save_checkpoint(&self, checkpoint_name: String) -> Result<()> {
        let checkpoint = TrainingCheckpoint {
            config: self.config.clone(),
            training_state: self.training_state.clone(),
            timestamp: chrono::Utc::now(),
        };
        
        let checkpoint_path = format!("checkpoints/oracle_{}.json", checkpoint_name);
        std::fs::create_dir_all("checkpoints")?;
        
        let checkpoint_json = serde_json::to_string_pretty(&checkpoint)?;
        std::fs::write(checkpoint_path, checkpoint_json)?;
        
        println!("  Checkpoint saved: {}", checkpoint_name);
        Ok(())
    }
    
    /// Load checkpoint
    pub fn load_checkpoint(&mut self, checkpoint_path: &str) -> Result<()> {
        let checkpoint_json = std::fs::read_to_string(checkpoint_path)?;
        let checkpoint: TrainingCheckpoint = serde_json::from_str(&checkpoint_json)?;
        
        self.config = checkpoint.config;
        self.training_state = checkpoint.training_state;
        
        println!("Checkpoint loaded from: {}", checkpoint_path);
        Ok(())
    }
    
    /// Get training statistics
    pub fn get_training_stats(&self) -> TrainingStats {
        TrainingStats {
            current_epoch: self.training_state.current_epoch,
            current_step: self.training_state.current_step,
            current_phase: self.training_state.phase.clone(),
            pretraining_loss: self.training_state.pretraining_loss,
            alignment_loss: self.training_state.alignment_loss,
            best_validation_loss: self.training_state.best_validation_loss,
            convergence_patience: self.training_state.convergence_patience,
        }
    }
}

/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub pretraining_result: Option<PretrainingResult>,
    pub alignment_result: Option<AlignmentResult>,
    pub final_metrics: Option<EvaluationMetrics>,
    pub final_state: OracleTrainingState,
}

impl TrainingResult {
    pub fn new() -> Self {
        Self {
            pretraining_result: None,
            alignment_result: None,
            final_metrics: None,
            final_state: OracleTrainingState::default(),
        }
    }
}

/// Pretraining result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainingResult {
    pub final_loss: f32,
    pub total_epochs: usize,
    pub epoch_metrics: Vec<PretrainingMetrics>,
}

impl PretrainingResult {
    pub fn new() -> Self {
        Self {
            final_loss: f32::INFINITY,
            total_epochs: 0,
            epoch_metrics: Vec::new(),
        }
    }
}

/// Alignment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    pub final_loss: f32,
    pub total_epochs: usize,
    pub alignment_stats: Vec<crate::oracle::alignment::AlignmentStats>,
}

impl AlignmentResult {
    pub fn new() -> Self {
        Self {
            final_loss: f32::INFINITY,
            total_epochs: 0,
            alignment_stats: Vec::new(),
        }
    }
}

/// Evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub pretraining_loss: f32,
    pub alignment_loss: f32,
    pub verification_score: f32,
    pub overall_score: f32,
}

/// Pretraining metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainingMetrics {
    pub epoch: usize,
    pub avg_loss: f32,
    pub fim_loss: f32,
    pub contrastive_loss: f32,
}

/// Training checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub config: OracleConfig,
    pub training_state: OracleTrainingState,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    pub current_epoch: usize,
    pub current_step: usize,
    pub current_phase: TrainingPhase,
    pub pretraining_loss: f32,
    pub alignment_loss: f32,
    pub best_validation_loss: f32,
    pub convergence_patience: usize,
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Create default ORACLE configuration
    pub fn create_default_config() -> OracleConfig {
        OracleConfig::default()
    }
    
    /// Validate ORACLE configuration
    pub fn validate_config(config: &OracleConfig) -> Result<()> {
        // Validate backbone config
        if config.backbone.d_model == 0 {
            return Err(anyhow::anyhow!("Model dimension must be positive"));
        }
        
        if config.backbone.n_heads == 0 {
            return Err(anyhow::anyhow!("Number of heads must be positive"));
        }
        
        if config.backbone.d_model % config.backbone.n_heads != 0 {
            return Err(anyhow::anyhow!("Model dimension must be divisible by number of heads"));
        }
        
        // Validate training config
        if config.training.batch_size == 0 {
            return Err(anyhow::anyhow!("Batch size must be positive"));
        }
        
        if config.training.learning_rate <= 0.0 {
            return Err(anyhow::anyhow!("Learning rate must be positive"));
        }
        
        Ok(())
    }
    
    /// Estimate training time
    pub fn estimate_training_time(
        config: &OracleConfig,
        dataset_size: usize,
        vocab_size: usize,
    ) -> TrainingTimeEstimate {
        let pretraining_steps = (dataset_size / config.training.batch_size) * config.training.pretraining_epochs;
        let alignment_steps = (dataset_size / config.training.batch_size) * config.training.alignment_epochs;
        let total_steps = pretraining_steps + alignment_steps;
        
        // Rough FLOP estimation
        let flops_per_step = {
            // Simplified FLOP estimation
            let d_model = config.backbone.d_model;
            let n_heads = config.backbone.n_heads;
            let seq_len = 8192; // Default sequence length
            
            // Attention FLOPs
            let attention_flops = (seq_len * seq_len * d_model) as u64;
            
            // MoE FLOPs
            let moe_flops = (d_model * config.backbone.mlp_hidden * config.backbone.top_k) as u64;
            
            // Output projection FLOPs
            let output_flops = (d_model * vocab_size) as u64;
            
            (attention_flops + moe_flops + output_flops) * 12 // 12 layers
        };
        let total_flops = total_steps as u64 * flops_per_step;
        
        // Time estimation (assuming 1 TFLOP/s)
        let estimated_seconds = total_flops as f64 / 1e12;
        let estimated_hours = (estimated_seconds / 3600.0) as f32;
        
        TrainingTimeEstimate {
            pretraining_steps,
            alignment_steps,
            total_steps,
            total_flops,
            estimated_hours,
        }
    }
    
    fn _estimate_flops_per_step(backbone_config: &OracleBackboneConfig, vocab_size: usize) -> u64 {
        // Simplified FLOP estimation
        let d_model = backbone_config.d_model;
        let n_heads = backbone_config.n_heads;
        let seq_len = 8192; // Default sequence length
        
        // Attention FLOPs
        let attention_flops = (seq_len * seq_len * d_model) as u64;
        
        // MoE FLOPs
        let moe_flops = (d_model * backbone_config.mlp_hidden * backbone_config.top_k) as u64;
        
        // Output projection FLOPs
        let output_flops = (d_model * vocab_size) as u64;
        
        (attention_flops + moe_flops + output_flops) * 12 // 12 layers
    }
    
    #[derive(Debug, Clone)]
    pub struct TrainingTimeEstimate {
        pub pretraining_steps: usize,
        pub alignment_steps: usize,
        pub total_steps: usize,
        pub total_flops: u64,
        pub estimated_hours: f32,
    }
    
    /// Create training report
    pub fn create_training_report(result: &TrainingResult) -> String {
        let mut report = String::new();
        
        report.push_str("# ORACLE Training Report\n\n");
        
        // Pretraining section
        if let Some(pretraining) = &result.pretraining_result {
            report.push_str("## Pretraining Results\n");
            report.push_str(&format!("- Final Loss: {:.6}\n", pretraining.final_loss));
            report.push_str(&format!("- Total Epochs: {}\n", pretraining.total_epochs));
            report.push_str(&format!("- Epoch Metrics: {}\n", pretraining.epoch_metrics.len()));
            report.push_str("\n");
        }
        
        // Alignment section
        if let Some(alignment) = &result.alignment_result {
            report.push_str("## Alignment Results\n");
            report.push_str(&format!("- Final Loss: {:.6}\n", alignment.final_loss));
            report.push_str(&format!("- Total Epochs: {}\n", alignment.total_epochs));
            report.push_str(&format!("- Alignment Stats: {}\n", alignment.alignment_stats.len()));
            report.push_str("\n");
        }
        
        // Final metrics section
        if let Some(metrics) = &result.final_metrics {
            report.push_str("## Final Evaluation\n");
            report.push_str(&format!("- Pretraining Loss: {:.6}\n", metrics.pretraining_loss));
            report.push_str(&format!("- Alignment Loss: {:.6}\n", metrics.alignment_loss));
            report.push_str(&format!("- Verification Score: {:.3}\n", metrics.verification_score));
            report.push_str(&format!("- Overall Score: {:.3}\n", metrics.overall_score));
            report.push_str("\n");
        }
        
        // Training state section
        report.push_str("## Training State\n");
        report.push_str(&format!("- Final Epoch: {}\n", result.final_state.current_epoch));
        report.push_str(&format!("- Final Step: {}\n", result.final_state.current_step));
        report.push_str(&format!("- Final Phase: {:?}\n", result.final_state.phase));
        report.push_str(&format!("- Best Validation Loss: {:.6}\n", result.final_state.best_validation_loss));
        
        report
    }
}
