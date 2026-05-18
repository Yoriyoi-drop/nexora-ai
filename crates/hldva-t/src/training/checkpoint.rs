//! Checkpoint management for HLDVA-T training

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use super::curriculum::TrainingMetrics;

/// Training checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
    pub metrics: TrainingMetrics,
    pub model_state: ModelState,
    pub optimizer_state: OptimizerState,
    pub timestamp: String,
}

/// Model state for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    pub parameters: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
    pub config: ModelConfig,
}

/// Optimizer state for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub learning_rate: f32,
    pub step: usize,
    pub momentum: Option<Vec<Vec<f32>>>,
    pub moments: Option<Vec<Vec<f32>>>,
    pub variance: Option<Vec<Vec<f32>>>,
}

/// Checkpoint manager
pub struct CheckpointManager {
    config: CheckpointConfig,
    checkpoint_dir: String,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(config: CheckpointConfig, checkpoint_dir: String) -> Self {
        Self {
            config,
            checkpoint_dir,
        }
    }
    
    /// Save checkpoint
    pub fn save_checkpoint(&self, checkpoint: &TrainingCheckpoint) -> HLDVAResult<()> {
        let filename = format!("checkpoint_epoch_{}_step_{}.bin", 
                              checkpoint.epoch, checkpoint.step);
        let filepath = Path::new(&self.checkpoint_dir).join(filename);
        
        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(&self.checkpoint_dir)?;
        
        // Serialize checkpoint to JSON
        let serialized = serde_json::to_string_pretty(checkpoint)?;
        fs::write(&filepath, serialized)?;
        
        println!("Saved checkpoint to: {:?}", filepath);
        
        // Keep only the most recent checkpoints
        self.cleanup_old_checkpoints()?;
        
        Ok(())
    }
    
    /// Load checkpoint
    pub fn load_checkpoint(&self, epoch: usize, step: usize) -> HLDVAResult<TrainingCheckpoint> {
        let filename = format!("checkpoint_epoch_{}_step_{}.bin", epoch, step);
        let filepath = Path::new(&self.checkpoint_dir).join(filename);
        
        let serialized = fs::read_to_string(&filepath)?;
        let checkpoint: TrainingCheckpoint = serde_json::from_str(&serialized)?;
        
        Ok(checkpoint)
    }
    
    /// Load latest checkpoint
    pub fn load_latest_checkpoint(&self) -> HLDVAResult<Option<TrainingCheckpoint>> {
        let checkpoints = self.list_checkpoints()?;
        
        let latest = checkpoints.into_iter()
            .max_by(|a, b| a.epoch.cmp(&b.epoch).then(a.step.cmp(&b.step)));
        
        match latest {
            Some(info) => self.load_checkpoint(info.epoch, info.step).map(Some),
            None => Ok(None),
        }
    }
    
    /// List available checkpoints
    pub fn list_checkpoints(&self) -> HLDVAResult<Vec<CheckpointInfo>> {
        let dir = Path::new(&self.checkpoint_dir);
        if !dir.exists() {
            return Ok(vec![]);
        }
        
        let mut checkpoints = Vec::new();
        let entries = fs::read_dir(dir)?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                let filename = path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                
                if let Ok(serialized) = fs::read_to_string(&path) {
                    if let Ok(ckpt) = serde_json::from_str::<TrainingCheckpoint>(&serialized) {
                        let size_bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                        checkpoints.push(CheckpointInfo {
                            filename,
                            epoch: ckpt.epoch,
                            step: ckpt.step,
                            loss: ckpt.loss,
                            timestamp: ckpt.timestamp,
                            size_bytes,
                        });
                    }
                }
            }
        }
        
        Ok(checkpoints)
    }
    
    /// Clean up old checkpoints
    fn cleanup_old_checkpoints(&self) -> HLDVAResult<()> {
        let mut checkpoints = self.list_checkpoints()?;
        
        if checkpoints.len() > self.config.max_checkpoints {
            checkpoints.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
            
            let to_remove = checkpoints.len() - self.config.max_checkpoints;
            for checkpoint in checkpoints.iter().take(to_remove) {
                let filepath = Path::new(&self.checkpoint_dir).join(&checkpoint.filename);
                if filepath.exists() {
                    fs::remove_file(&filepath)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate checkpoint integrity
    pub fn validate_checkpoint(&self, checkpoint: &TrainingCheckpoint) -> HLDVAResult<bool> {
        // Check if checkpoint data is consistent
        if checkpoint.epoch == 0 && checkpoint.step == 0 {
            return Ok(false); // Invalid checkpoint
        }
        
        // Check if model state is present
        if checkpoint.model_state.parameters.is_empty() {
            return Ok(false); // Invalid model state
        }
        
        Ok(true)
    }
    
    /// Export checkpoint to different format
    pub fn export_checkpoint(&self, checkpoint: &TrainingCheckpoint, format: ExportFormat) -> HLDVAResult<()> {
        match format {
            ExportFormat::PyTorch => {
                println!("Exporting checkpoint to PyTorch format");
                // Implementation for PyTorch export
            }
            ExportFormat::ONNX => {
                println!("Exporting checkpoint to ONNX format");
                // Implementation for ONNX export
            }
            ExportFormat::TensorFlow => {
                println!("Exporting checkpoint to TensorFlow format");
                // Implementation for TensorFlow export
            }
        }
        
        Ok(())
    }
}

/// Checkpoint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub filename: String,
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
    pub timestamp: String,
    pub size_bytes: u64,
}

/// Checkpoint configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub save_interval: usize,
    pub max_checkpoints: usize,
    pub save_optimizer_state: bool,
    pub save_training_state: bool,
    pub compression: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            save_interval: 1000,
            max_checkpoints: 5,
            save_optimizer_state: true,
            save_training_state: true,
            compression: false,
        }
    }
}

/// Export formats
#[derive(Debug, Clone)]
pub enum ExportFormat {
    PyTorch,
    ONNX,
    TensorFlow,
}

/// Checkpoint utilities
pub struct CheckpointUtils;

impl CheckpointUtils {
    /// Create checkpoint from model and optimizer
    pub fn create_checkpoint(
        epoch: usize,
        step: usize,
        loss: f32,
        metrics: TrainingMetrics,
        model: &dyn Model,
        optimizer: &dyn Optimizer,
    ) -> HLDVAResult<TrainingCheckpoint> {
        let model_state = ModelState {
            parameters: model.get_parameters()?,
            shapes: model.get_parameter_shapes()?,
            config: model.get_config(),
        };
        
        let optimizer_state = OptimizerState {
            learning_rate: optimizer.get_learning_rate(),
            step,
            momentum: optimizer.get_momentum(),
            moments: optimizer.get_moments(),
            variance: optimizer.get_variance(),
        };
        
        Ok(TrainingCheckpoint {
            epoch,
            step,
            loss,
            metrics,
            model_state,
            optimizer_state,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }
    
    /// Restore model and optimizer from checkpoint
    pub fn restore_from_checkpoint(
        checkpoint: &TrainingCheckpoint,
        model: &mut dyn Model,
        optimizer: &mut dyn Optimizer,
    ) -> HLDVAResult<()> {
        model.load_parameters(&checkpoint.model_state.parameters, &checkpoint.model_state.shapes)?;
        optimizer.load_state(&checkpoint.optimizer_state)?;
        Ok(())
    }
    
    /// Compare two checkpoints
    pub fn compare_checkpoints(
        checkpoint1: &TrainingCheckpoint,
        checkpoint2: &TrainingCheckpoint,
    ) -> CheckpointComparison {
        CheckpointComparison {
            epoch_diff: checkpoint2.epoch as isize - checkpoint1.epoch as isize,
            step_diff: checkpoint2.step as isize - checkpoint1.step as isize,
            loss_diff: checkpoint2.loss - checkpoint1.loss,
            accuracy_diff: checkpoint2.metrics.accuracy - checkpoint1.metrics.accuracy,
            learning_rate_diff: checkpoint2.metrics.learning_rate - checkpoint1.metrics.learning_rate,
        }
    }
}

/// Checkpoint comparison result
#[derive(Debug, Clone)]
pub struct CheckpointComparison {
    pub epoch_diff: isize,
    pub step_diff: isize,
    pub loss_diff: f32,
    pub accuracy_diff: f32,
    pub learning_rate_diff: f32,
}

// Mock traits for the checkpoint utilities
pub trait Model {
    fn get_parameters(&self) -> HLDVAResult<HashMap<String, Vec<f32>>>;
    fn get_parameter_shapes(&self) -> HLDVAResult<HashMap<String, Vec<usize>>>;
    fn get_config(&self) -> ModelConfig;
    fn load_parameters(&mut self, params: &HashMap<String, Vec<f32>>, shapes: &HashMap<String, Vec<usize>>) -> HLDVAResult<()>;
}

pub trait Optimizer {
    fn get_learning_rate(&self) -> f32;
    fn get_momentum(&self) -> Option<Vec<Vec<f32>>>;
    fn get_moments(&self) -> Option<Vec<Vec<f32>>>;
    fn get_variance(&self) -> Option<Vec<Vec<f32>>>;
    fn load_state(&mut self, state: &OptimizerState) -> HLDVAResult<()>;
}

// Mock model config
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
}
