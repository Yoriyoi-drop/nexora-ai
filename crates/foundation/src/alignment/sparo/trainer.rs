//! SPARO Trainer - Main training orchestration
//! 
//! Mengkoordinasikan semua komponen SPARO untuk pelatihan end-to-end

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use rand::seq::SliceRandom;

use super::core::{
    SparoConfig, PolicyModel, ReasoningTrace, JudgeFeedback,
    TrainingMetrics, ModelState, SparoLoss
};
use super::data::TrainingBatch;
use super::dpo::{DpoTrainer, DpoConfig};
use super::kto::{KtoTrainer, KtoConfig};
use super::ipo::{IpoTrainer, IpoConfig};
use super::rlvf::{RlvfManager, RlvfConfig};
use super::spin::{SpinTrainer, SpinConfig, SelfPlayTournament};
use super::rlaif::{RlaifManager, RlaifConfig};

/// Main SPARO Trainer
pub struct SparoTrainer {
    config: SparoConfig,
    
    // Component trainers
    dpo_trainer: DpoTrainer,
    kto_trainer: KtoTrainer,
    ipo_trainer: IpoTrainer,
    
    // Managers
    rlvf_manager: RlvfManager,
    rlaif_manager: RlaifManager,
    spin_trainer: SpinTrainer,
    
    // Training state
    student_model: PolicyModel,
    teacher_model: PolicyModel,
    training_state: ModelState,
    metrics_history: Vec<TrainingMetrics>,
}

impl SparoTrainer {
    /// Create new SPARO trainer
    pub fn new(
        config: SparoConfig,
        student_model: PolicyModel,
        teacher_model: PolicyModel,
    ) -> Result<Self> {
        // Initialize component trainers
        let dpo_config = DpoConfig::default();
        let kto_config = KtoConfig::default();
        let ipo_config = IpoConfig::default();
        let rlvf_config = RlvfConfig::default();
        let rlaif_config = RlaifConfig::default();
        let spin_config = SpinConfig::default();
        
        let dpo_trainer = DpoTrainer::new(student_model.clone(), dpo_config);
        let kto_trainer = KtoTrainer::new(student_model.clone(), kto_config);
        let ipo_trainer = IpoTrainer::new(student_model.clone(), ipo_config);
        let rlvf_manager = RlvfManager::new(rlvf_config);
        let rlaif_manager = RlaifManager::new(rlaif_config);
        let spin_trainer = SpinTrainer::new(spin_config, student_model.clone(), teacher_model.clone());
        
        Ok(Self {
            config,
            dpo_trainer,
            kto_trainer,
            ipo_trainer,
            rlvf_manager,
            rlaif_manager,
            spin_trainer,
            student_model,
            teacher_model,
            training_state: ModelState {
                iteration: 0,
                loss_history: Vec::new(),
                current_loss: 0.0,
                best_loss: f32::INFINITY,
                converged: false,
            },
            metrics_history: Vec::new(),
        })
    }
    
    /// Main training loop
    pub fn train(&mut self, prompts: &[String]) -> Result<TrainingResult> {
        let mut training_result = TrainingResult::new();
        
        for iteration in 0..self.config.max_iterations {
            self.training_state.iteration = iteration;
            
            // Generate reasoning traces
            let traces = self.generate_traces(prompts)?;
            
            // Step 1: RLAIF - Generate AI feedback
            let ai_feedback = self.generate_ai_feedback(&traces)?;
            
            // Step 2: RLVF - Verify steps automatically
            let verified_feedback = self.verify_steps(&traces)?;
            
            // Combine feedback
            let combined_feedback = self.combine_feedback(&ai_feedback, &verified_feedback);
            
            // Step 3: Extract training data for each component
            let training_batch = TrainingBatch::new(traces, combined_feedback, iteration);
            
            // Step 4: Train each component
            let component_losses = self.train_components(&training_batch)?;
            
            // Step 5: SPIN - Self-play improvement
            if iteration % 5 == 0 { // Every 5 iterations
                let tournament = self.run_self_play(prompts)?;
                let spin_loss = self.spin_trainer.update_models(&tournament)?;
                
                // Check for student promotion
                if self.spin_trainer.promote_student_to_teacher(&tournament)? {
                    self.teacher_model = self.student_model.clone();
                    self.teacher_model.set_as_teacher();
                }
            }
            
            // Calculate total loss
            let total_loss = self.calculate_total_loss(&component_losses)?;
            self.training_state.current_loss = total_loss;
            self.training_state.loss_history.push(total_loss);
            
            // Update best loss
            if total_loss < self.training_state.best_loss {
                self.training_state.best_loss = total_loss;
            }
            
            // Create metrics
            let metrics = TrainingMetrics::new(iteration, component_losses.clone());
            self.metrics_history.push(metrics.clone());
            
            // Check convergence
            if self.check_convergence() {
                self.training_state.converged = true;
                break;
            }
            
            // Log progress
            self.log_progress(iteration, &component_losses, total_loss)?;
        }
        
        training_result.final_state = self.training_state.clone();
        training_result.metrics_history = self.metrics_history.clone();
        training_result.final_model = self.student_model.clone();
        
        Ok(training_result)
    }
    
    /// Generate reasoning traces from current model
    fn generate_traces(&self, prompts: &[String]) -> Result<Vec<ReasoningTrace>> {
        let mut traces = Vec::new();
        
        for prompt in prompts {
            let trace = self.generate_single_trace(prompt)?;
            traces.push(trace);
        }
        
        Ok(traces)
    }
    
    /// Generate single reasoning trace
    fn generate_single_trace(&self, prompt: &str) -> Result<ReasoningTrace> {
        let mut steps = Vec::new();
        let step_count = (prompt.len() / 50).max(1).min(10);
        
        for i in 1..=step_count {
            let step = super::core::ReasoningStep {
                id: Uuid::new_v4(),
                content: format!("Step {}: Reasoning about {}", i, prompt),
                step_number: i,
                timestamp: chrono::Utc::now(),
            };
            steps.push(step);
        }
        
        let final_answer = format!("Final answer based on {} steps", steps.len());
        
        Ok(ReasoningTrace {
            id: Uuid::new_v4(),
            prompt: prompt.to_string(),
            steps,
            final_answer,
            created_at: chrono::Utc::now(),
        })
    }
    
    /// Generate AI feedback using RLAIF
    fn generate_ai_feedback(&self, traces: &[ReasoningTrace]) -> Result<Vec<JudgeFeedback>> {
        let mut all_feedback = Vec::new();
        
        for trace in traces {
            let feedback = self.rlaif_manager.generate_feedback(trace)?;
            all_feedback.extend(feedback);
        }
        
        Ok(all_feedback)
    }
    
    /// Verify steps using RLVF
    fn verify_steps(&self, traces: &[ReasoningTrace]) -> Result<Vec<JudgeFeedback>> {
        let mut all_feedback = Vec::new();
        
        for trace in traces {
            let step_feedbacks = self.rlvf_manager.verify_trace(trace)?;
            let judge_feedbacks = self.rlvf_manager.feedback_to_judge_feedback(&step_feedbacks);
            all_feedback.extend(judge_feedbacks);
        }
        
        Ok(all_feedback)
    }
    
    /// Combine feedback from different sources
    fn combine_feedback(&self, ai_feedback: &[JudgeFeedback], verified_feedback: &[JudgeFeedback]) -> Vec<JudgeFeedback> {
        let mut combined = Vec::new();
        combined.extend_from_slice(ai_feedback);
        combined.extend_from_slice(verified_feedback);
        combined
    }
    
    /// Train all components
    fn train_components(&mut self, batch: &TrainingBatch) -> Result<SparoLoss> {
        // DPO training
        let dpo_pairs = self.dpo_trainer.extract_preference_pairs(&batch.traces, &batch.feedback)?;
        let dpo_loss = self.dpo_trainer.training_step(&dpo_pairs)?;
        
        // KTO training
        let kto_labels = self.kto_trainer.extract_independent_labels(&batch.traces, &batch.feedback)?;
        let kto_loss = self.kto_trainer.training_step(&kto_labels)?;
        
        // IPO training
        let ipo_loss = self.ipo_trainer.training_step()?;
        
        // Update IPO constraints
        self.ipo_trainer.update_constraints()?;
        
        Ok(SparoLoss {
            total_loss: self.config.alpha * dpo_loss + self.config.beta * kto_loss + self.config.gamma * ipo_loss,
            dpo_loss,
            kto_loss,
            ipo_loss,
        })
    }
    
    /// Run self-play tournament
    fn run_self_play(&mut self, prompts: &[String]) -> Result<SelfPlayTournament> {
        self.spin_trainer.run_tournament(prompts)
    }
    
    /// Calculate total weighted loss
    fn calculate_total_loss(&self, losses: &SparoLoss) -> Result<f32> {
        Ok(self.config.alpha * losses.dpo_loss + 
           self.config.beta * losses.kto_loss + 
           self.config.gamma * losses.ipo_loss)
    }
    
    /// Check if training has converged
    fn check_convergence(&self) -> bool {
        if self.training_state.loss_history.len() < 10 {
            return false;
        }
        
        let recent_losses: Vec<f32> = self.training_state.loss_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        
        let avg_recent = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let variance = recent_losses.iter()
            .map(|&x| (x - avg_recent).powi(2))
            .sum::<f32>() / recent_losses.len() as f32;
        
        variance < self.config.convergence_threshold
    }
    
    /// Log training progress
    fn log_progress(&self, iteration: usize, losses: &SparoLoss, total_loss: f32) -> Result<()> {
        println!("Iteration {}: Total Loss = {:.6}", iteration, total_loss);
        println!("  DPO Loss = {:.6} (weight: {:.2})", losses.dpo_loss, self.config.alpha);
        println!("  KTO Loss = {:.6} (weight: {:.2})", losses.kto_loss, self.config.beta);
        println!("  IPO Loss = {:.6} (weight: {:.2})", losses.ipo_loss, self.config.gamma);
        println!("  Best Loss = {:.6}", self.training_state.best_loss);
        println!("  Converged = {}", self.training_state.converged);
        println!();
        
        Ok(())
    }
    
    /// Get current training statistics
    pub fn get_training_stats(&self) -> TrainingStats {
        let current_loss = self.training_state.current_loss;
        let best_loss = self.training_state.best_loss;
        let iterations = self.training_state.iteration;
        let converged = self.training_state.converged;
        
        let avg_loss = if self.training_state.loss_history.is_empty() {
            0.0
        } else {
            self.training_state.loss_history.iter().sum::<f32>() / 
            self.training_state.loss_history.len() as f32
        };
        
        let loss_trend = if self.training_state.loss_history.len() >= 2 {
            let recent = self.training_state.loss_history.iter().rev().take(5).sum::<f32>() / 5.0;
            let earlier = self.training_state.loss_history.iter().rev().skip(5).take(5).sum::<f32>() / (5.0_f32).max(1.0);
            recent - earlier
        } else {
            0.0
        };
        
        TrainingStats {
            iterations,
            current_loss,
            best_loss,
            avg_loss,
            loss_trend,
            converged,
            total_samples: self.metrics_history.len(),
        }
    }
    
    /// Save training checkpoint (binary format)
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        let checkpoint = TrainingCheckpoint {
            config: self.config.clone(),
            training_state: self.training_state.clone(),
            metrics_history: self.metrics_history.clone(),
            student_model: self.student_model.clone(),
            teacher_model: self.teacher_model.clone(),
        };
        
        let encoded = bincode::serialize(&checkpoint)?;
        std::fs::write(path, encoded)?;
        
        Ok(())
    }
    
    /// Load training checkpoint (binary format)
    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        let encoded = std::fs::read(path)?;
        let checkpoint: TrainingCheckpoint = bincode::deserialize(&encoded)?;
        
        self.config = checkpoint.config;
        self.training_state = checkpoint.training_state;
        self.metrics_history = checkpoint.metrics_history;
        self.student_model = checkpoint.student_model;
        self.teacher_model = checkpoint.teacher_model;
        
        Ok(())
    }
}

/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub final_state: ModelState,
    pub metrics_history: Vec<TrainingMetrics>,
    pub final_model: PolicyModel,
}

impl TrainingResult {
    fn new() -> Self {
        Self {
            final_state: ModelState {
                iteration: 0,
                loss_history: Vec::new(),
                current_loss: 0.0,
                best_loss: f32::INFINITY,
                converged: false,
            },
            metrics_history: Vec::new(),
            final_model: PolicyModel::new(Uuid::new_v4(), (100, 100)),
        }
    }
}

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    pub iterations: usize,
    pub current_loss: f32,
    pub best_loss: f32,
    pub avg_loss: f32,
    pub loss_trend: f32,
    pub converged: bool,
    pub total_samples: usize,
}

/// Training checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub config: SparoConfig,
    pub training_state: ModelState,
    pub metrics_history: Vec<TrainingMetrics>,
    pub student_model: PolicyModel,
    pub teacher_model: PolicyModel,
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Create default SPARO trainer
    pub fn create_default_trainer(
        student_model: PolicyModel,
        teacher_model: PolicyModel,
    ) -> Result<SparoTrainer> {
        let config = SparoConfig::default();
        SparoTrainer::new(config, student_model, teacher_model)
    }
    
    /// Generate training prompts
    pub fn generate_training_prompts(count: usize) -> Vec<String> {
        let base_prompts = vec![
            "Solve this math problem: {}",
            "Explain the concept of {}",
            "Write a short story about {}",
            "What is the capital of {}?",
            "How do you {}?",
            "Compare and contrast {} and {}",
            "Analyze the following: {}",
            "Create a plan for {}",
        ];
        
        let mut prompts = Vec::new();
        let topics = vec![
            "gravity", "photosynthesis", "democracy", "machine learning",
            "climate change", "quantum physics", "economics", "psychology",
            "artificial intelligence", "blockchain", "genetics", "astronomy",
        ];
        
        for i in 0..count {
            let template = base_prompts[i % base_prompts.len()];
            let topic = topics[i % topics.len()];
            prompts.push(template.replace("{}", topic));
        }
        
        prompts
    }
    
    /// Validate training configuration
    pub fn validate_config(config: &SparoConfig) -> Result<()> {
        if config.alpha < 0.0 || config.alpha > 1.0 {
            return Err(anyhow::anyhow!("Invalid alpha value: {}", config.alpha));
        }
        
        if config.beta < 0.0 || config.beta > 1.0 {
            return Err(anyhow::anyhow!("Invalid beta value: {}", config.beta));
        }
        
        if config.gamma < 0.0 || config.gamma > 1.0 {
            return Err(anyhow::anyhow!("Invalid gamma value: {}", config.gamma));
        }
        
        let total_weight = config.alpha + config.beta + config.gamma;
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(anyhow::anyhow!("Weights must sum to 1.0, got {}", total_weight));
        }
        
        if config.learning_rate <= 0.0 {
            return Err(anyhow::anyhow!("Learning rate must be positive"));
        }
        
        if config.batch_size == 0 {
            return Err(anyhow::anyhow!("Batch size must be positive"));
        }
        
        Ok(())
    }
    
    /// Analyze training metrics
    pub fn analyze_metrics(metrics: &[TrainingMetrics]) -> MetricsAnalysis {
        if metrics.is_empty() {
            return MetricsAnalysis::default();
        }
        
        let total_losses: Vec<f32> = metrics.iter().map(|m| m.loss.total_loss).collect();
        let dpo_losses: Vec<f32> = metrics.iter().map(|m| m.loss.dpo_loss).collect();
        let kto_losses: Vec<f32> = metrics.iter().map(|m| m.loss.kto_loss).collect();
        let ipo_losses: Vec<f32> = metrics.iter().map(|m| m.loss.ipo_loss).collect();
        
        let final_loss = total_losses.last().unwrap_or(&0.0);
        let best_loss = total_losses.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&0.0);
        let avg_loss = total_losses.iter().sum::<f32>() / total_losses.len() as f32;
        
        let convergence_rate = calculate_convergence_rate(&total_losses);
        let improvement_rate = calculate_improvement_rate(&total_losses);
        
        MetricsAnalysis {
            final_loss: *final_loss,
            best_loss: *best_loss,
            avg_loss,
            convergence_rate,
            improvement_rate,
            total_iterations: metrics.len(),
            component_balance: ComponentBalance {
                dpo_contribution: dpo_losses.iter().sum::<f32>() / total_losses.iter().sum::<f32>(),
                kto_contribution: kto_losses.iter().sum::<f32>() / total_losses.iter().sum::<f32>(),
                ipo_contribution: ipo_losses.iter().sum::<f32>() / total_losses.iter().sum::<f32>(),
            },
        }
    }
    
    fn calculate_convergence_rate(losses: &[f32]) -> f32 {
        if losses.len() < 10 {
            return 0.0;
        }
        
        let recent: Vec<f32> = losses.iter().rev().take(5).cloned().collect();
        let earlier: Vec<f32> = losses.iter().rev().skip(5).take(5).cloned().collect();
        
        let recent_var = recent.iter().map(|x| (x - recent.iter().sum::<f32>() / recent.len() as f32).powi(2)).sum::<f32>() / recent.len() as f32;
        let earlier_var = earlier.iter().map(|x| (x - earlier.iter().sum::<f32>() / earlier.len() as f32).powi(2)).sum::<f32>() / earlier.len() as f32;
        
        if earlier_var == 0.0 {
            1.0
        } else {
            1.0 - (recent_var / earlier_var)
        }
    }
    
    fn calculate_improvement_rate(losses: &[f32]) -> f32 {
        if losses.len() < 2 {
            return 0.0;
        }
        
        let initial = losses[0];
        let final_loss = losses[losses.len() - 1];
        
        if initial == 0.0 {
            0.0
        } else {
            (initial - final_loss) / initial
        }
    }
    
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct MetricsAnalysis {
        pub final_loss: f32,
        pub best_loss: f32,
        pub avg_loss: f32,
        pub convergence_rate: f32,
        pub improvement_rate: f32,
        pub total_iterations: usize,
        pub component_balance: ComponentBalance,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct ComponentBalance {
        pub dpo_contribution: f32,
        pub kto_contribution: f32,
        pub ipo_contribution: f32,
    }
}
