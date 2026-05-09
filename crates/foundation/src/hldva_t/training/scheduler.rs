//! Training Scheduler module for HLDVA-T

use crate::hldva_t::types::*;
use crate::atqs::Tensor;

/// Training Scheduler
pub struct TrainingScheduler {
    config: SchedulerConfig,
    current_epoch: usize,
    current_step: usize,
    total_steps: usize,
}

impl TrainingScheduler {
    /// Create new training scheduler
    pub fn new(config: SchedulerConfig, dataset_size: usize) -> Self {
        let steps_per_epoch = dataset_size / config.batch_size;
        let total_steps = steps_per_epoch * config.num_epochs;
        
        Self {
            config,
            current_epoch: 0,
            current_step: 0,
            total_steps,
        }
    }
    
    /// Check if should continue training
    pub fn should_continue(&self) -> bool {
        self.current_epoch < self.config.num_epochs
    }
    
    /// Advance to next step
    pub fn step(&mut self) -> bool {
        self.current_step += 1;
        
        // Check if epoch completed
        let steps_per_epoch = self.total_steps / self.config.num_epochs;
        if self.current_step % steps_per_epoch == 0 {
            self.current_epoch += 1;
            return true; // Epoch completed
        }
        
        false
    }
    
    /// Get current learning rate
    pub fn get_learning_rate(&self) -> f32 {
        match &self.config.lr_scheduler {
            LRSchedulerType::Constant => self.config.initial_lr,
            LRSchedulerType::Step { step_size, gamma } => {
                let decay_steps = self.current_step / step_size;
                self.config.initial_lr * gamma.powi(decay_steps as i32)
            }
            LRSchedulerType::Exponential { gamma } => {
                self.config.initial_lr * gamma.powi(self.current_step as i32)
            }
            LRSchedulerType::Cosine { t_max, eta_min } => {
                let progress = (self.current_step % t_max) as f32 / *t_max as f32;
                eta_min + (self.config.initial_lr - eta_min) * 
                    (0.5 * (1.0 + (std::f32::consts::PI * progress).cos()))
            }
            LRSchedulerType::Warmup { warmup_steps, scheduler } => {
                if self.current_step < *warmup_steps {
                    self.config.initial_lr * (self.current_step as f32 / *warmup_steps as f32)
                } else {
                    // Use underlying scheduler after warmup
                    self.config.initial_lr // Simplified
                }
            }
        }
    }
    
    /// Get current epoch
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }
    
    /// Get current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }
    
    /// Get progress percentage
    pub fn progress(&self) -> f32 {
        self.current_step as f32 / self.total_steps as f32
    }
    
    /// Check if should save checkpoint
    pub fn should_save_checkpoint(&self) -> bool {
        self.current_step % self.config.checkpoint_interval == 0
    }
    
    /// Check if should validate
    pub fn should_validate(&self) -> bool {
        self.current_step % self.config.validation_interval == 0
    }
    
    /// Check if should log
    pub fn should_log(&self) -> bool {
        self.current_step % self.config.log_interval == 0
    }
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub initial_lr: f32,
    pub lr_scheduler: LRSchedulerType,
    pub checkpoint_interval: usize,
    pub validation_interval: usize,
    pub log_interval: usize,
}

#[derive(Debug, Clone)]
pub enum LRSchedulerType {
    Constant,
    Step { step_size: usize, gamma: f32 },
    Exponential { gamma: f32 },
    Cosine { t_max: usize, eta_min: f32 },
    Warmup { warmup_steps: usize, scheduler: Box<LRSchedulerType> },
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_epochs: 100,
            batch_size: 32,
            initial_lr: 0.001,
            lr_scheduler: LRSchedulerType::Constant,
            checkpoint_interval: 1000,
            validation_interval: 100,
            log_interval: 10,
        }
    }
}

/// Batch scheduler for dynamic batch sizing
pub struct BatchScheduler {
    min_batch_size: usize,
    max_batch_size: usize,
    current_batch_size: usize,
    memory_threshold: f32,
    performance_window: Vec<f32>,
}

impl BatchScheduler {
    /// Create new batch scheduler
    pub fn new(min_batch_size: usize, max_batch_size: usize, memory_threshold: f32) -> Self {
        Self {
            min_batch_size,
            max_batch_size,
            current_batch_size: min_batch_size,
            memory_threshold,
            performance_window: Vec::new(),
        }
    }
    
    /// Update batch size based on performance
    pub fn update_batch_size(&mut self, throughput: f32, memory_usage: f32) {
        self.performance_window.push(throughput);
        
        // Keep only recent performance metrics
        if self.performance_window.len() > 10 {
            self.performance_window.remove(0);
        }
        
        // Adjust batch size based on memory usage and performance
        if memory_usage < self.memory_threshold && self.performance_window.len() >= 5 {
            let avg_throughput = self.performance_window.iter().sum::<f32>() / self.performance_window.len() as f32;
            
            // Try to increase batch size if performance is improving
            if self.current_batch_size < self.max_batch_size {
                self.current_batch_size = (self.current_batch_size * 2).min(self.max_batch_size);
            }
        } else if memory_usage > self.memory_threshold {
            // Decrease batch size if memory usage is high
            self.current_batch_size = (self.current_batch_size / 2).max(self.min_batch_size);
        }
    }
    
    /// Get current batch size
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size
    }
}

/// Gradient accumulation scheduler
pub struct GradientAccumulationScheduler {
    base_batch_size: usize,
    target_batch_size: usize,
    current_accumulation_steps: usize,
    memory_efficient: bool,
}

impl GradientAccumulationScheduler {
    /// Create new gradient accumulation scheduler
    pub fn new(base_batch_size: usize, target_batch_size: usize, memory_efficient: bool) -> Self {
        let accumulation_steps = target_batch_size / base_batch_size;
        
        Self {
            base_batch_size,
            target_batch_size,
            current_accumulation_steps: accumulation_steps.max(1),
            memory_efficient,
        }
    }
    
    /// Update accumulation steps based on memory constraints
    pub fn update_accumulation_steps(&mut self, available_memory: f32) {
        if self.memory_efficient && available_memory < 0.5 {
            // Reduce accumulation steps under memory pressure
            self.current_accumulation_steps = (self.current_accumulation_steps / 2).max(1);
        } else if available_memory > 0.8 {
            // Increase accumulation steps when memory is available
            let max_steps = self.target_batch_size / self.base_batch_size;
            self.current_accumulation_steps = (self.current_accumulation_steps * 2).min(max_steps);
        }
    }
    
    /// Get current accumulation steps
    pub fn accumulation_steps(&self) -> usize {
        self.current_accumulation_steps
    }
    
    /// Check if should update parameters
    pub fn should_update(&self, step: usize) -> bool {
        step % self.current_accumulation_steps == 0
    }
}

/// Mixed precision training scheduler
pub struct MixedPrecisionScheduler {
    enabled: bool,
    loss_scale: f32,
    overflow_counter: usize,
    max_overflow_count: usize,
}

impl MixedPrecisionScheduler {
    /// Create new mixed precision scheduler
    pub fn new(enabled: bool, initial_loss_scale: f32) -> Self {
        Self {
            enabled,
            loss_scale: initial_loss_scale,
            overflow_counter: 0,
            max_overflow_count: 3,
        }
    }
    
    /// Handle overflow in mixed precision training
    pub fn handle_overflow(&mut self) -> bool {
        if !self.enabled {
            return false;
        }
        
        self.overflow_counter += 1;
        
        if self.overflow_counter >= self.max_overflow_count {
            // Reduce loss scale
            self.loss_scale /= 2.0;
            self.overflow_counter = 0;
            return true; // Loss scale was reduced
        }
        
        false
    }
    
    /// Update loss scale after successful step
    pub fn update_loss_scale(&mut self) {
        if self.enabled && self.overflow_counter == 0 {
            // Gradually increase loss scale
            self.loss_scale *= 1.1;
        }
    }
    
    /// Get current loss scale
    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }
    
    /// Check if mixed precision is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}
