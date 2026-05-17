//! Optimizer module for HLDVA-T training

use crate::hldva_t::types::*;
use crate::atqs::Tensor;

/// Optimizer trait
pub trait Optimizer {
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) -> HLDVAResult<()>;
    fn zero_grad(&mut self, gradients: &mut [Tensor]) -> HLDVAResult<()>;
    fn set_learning_rate(&mut self, lr: f32);
}

/// Adam Optimizer
pub struct AdamOptimizer {
    _config: AdamConfig,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step: usize,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamOptimizer {
    /// Create new Adam optimizer
    pub fn new(config: AdamConfig) -> Self {
        let learning_rate = config.learning_rate;
        let beta1 = config.beta1;
        let beta2 = config.beta2;
        let epsilon = config.epsilon;
        
        Self {
            _config: config,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            step: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }
    
    /// Initialize moment vectors
    fn initialize_moments(&mut self, parameters: &[Tensor]) {
        self.m.clear();
        self.v.clear();
        
        for param in parameters {
            let param_data = param.data();
            self.m.push(vec![0.0; param_data.len()]);
            self.v.push(vec![0.0; param_data.len()]);
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) -> HLDVAResult<()> {
        if self.m.is_empty() {
            self.initialize_moments(parameters);
        }
        
        self.step += 1;
        
        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let param_data = param.data_mut();
            let grad_data = grad.data();
            
            if param_data.len() != grad_data.len() {
                return Err(HLDVAError::Training("Parameter and gradient dimensions must match".to_string()));
            }
            
            // Update biased first moment estimate
            for (m_val, &grad_val) in self.m[i].iter_mut().zip(grad_data.iter()) {
                *m_val = self.beta1 * *m_val + (1.0 - self.beta1) * grad_val;
            }
            
            // Update biased second moment estimate
            for (v_val, &grad_val) in self.v[i].iter_mut().zip(grad_data.iter()) {
                *v_val = self.beta2 * *v_val + (1.0 - self.beta2) * grad_val * grad_val;
            }
            
            // Compute bias-corrected estimates
            let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);
            
            // Update parameters
            for (param_val, (m_val, v_val)) in param_data.iter_mut()
                .zip(self.m[i].iter().zip(self.v[i].iter())) {
                let m_hat = *m_val / bias_correction1;
                let v_hat = *v_val / bias_correction2;
                *param_val -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self, gradients: &mut [Tensor]) -> HLDVAResult<()> {
        for grad in gradients {
            let grad_data = grad.data_mut();
            for val in grad_data.iter_mut() {
                *val = 0.0;
            }
        }
        Ok(())
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

/// SGD Optimizer
pub struct SGDOptimizer {
    _config: SGDConfig,
    learning_rate: f32,
    momentum: f32,
    velocity: Vec<Vec<f32>>,
}

impl SGDOptimizer {
    /// Create new SGD optimizer
    pub fn new(config: SGDConfig) -> Self {
        let learning_rate = config.learning_rate;
        let momentum = config.momentum;
        
        Self {
            _config: config,
            learning_rate,
            momentum,
            velocity: Vec::new(),
        }
    }
    
    /// Initialize velocity vectors
    fn initialize_velocity(&mut self, parameters: &[Tensor]) {
        self.velocity.clear();
        for param in parameters {
            let param_data = param.data();
            self.velocity.push(vec![0.0; param_data.len()]);
        }
    }
}

impl Optimizer for SGDOptimizer {
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) -> HLDVAResult<()> {
        if self.velocity.is_empty() {
            self.initialize_velocity(parameters);
        }
        
        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let param_data = param.data_mut();
            let grad_data = grad.data();
            
            if param_data.len() != grad_data.len() {
                return Err(HLDVAError::Training("Parameter and gradient dimensions must match".to_string()));
            }
            
            // Update velocity
            for (vel_val, &grad_val) in self.velocity[i].iter_mut().zip(grad_data.iter()) {
                *vel_val = self.momentum * *vel_val + grad_val;
            }
            
            // Update parameters
            for (param_val, &vel_val) in param_data.iter_mut().zip(self.velocity[i].iter()) {
                *param_val -= self.learning_rate * vel_val;
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self, gradients: &mut [Tensor]) -> HLDVAResult<()> {
        for grad in gradients {
            let grad_data = grad.data_mut();
            for val in grad_data.iter_mut() {
                *val = 0.0;
            }
        }
        Ok(())
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

/// Adam configuration
#[derive(Debug, Clone)]
pub struct AdamConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// SGD configuration
#[derive(Debug, Clone)]
pub struct SGDConfig {
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }
}

/// Learning rate scheduler
pub trait LearningRateScheduler {
    fn get_lr(&self, step: usize) -> f32;
    fn step(&mut self, step: usize);
}

/// Step learning rate scheduler
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
    current_step: usize,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
}

impl LearningRateScheduler for StepLR {
    fn get_lr(&self, step: usize) -> f32 {
        let decay_steps = step / self.step_size;
        self.initial_lr * self.gamma.powi(decay_steps as i32)
    }
    
    fn step(&mut self, step: usize) {
        self.current_step = step;
    }
}

/// Exponential learning rate scheduler
pub struct ExponentialLR {
    initial_lr: f32,
    gamma: f32,
    current_step: usize,
}

impl ExponentialLR {
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self {
            initial_lr,
            gamma,
            current_step: 0,
        }
    }
}

impl LearningRateScheduler for ExponentialLR {
    fn get_lr(&self, step: usize) -> f32 {
        self.initial_lr * self.gamma.powi(step as i32)
    }
    
    fn step(&mut self, step: usize) {
        self.current_step = step;
    }
}

/// Cosine annealing learning rate scheduler
pub struct CosineAnnealingLR {
    initial_lr: f32,
    t_max: usize,
    eta_min: f32,
    current_step: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, t_max: usize, eta_min: f32) -> Self {
        Self {
            initial_lr,
            t_max,
            eta_min,
            current_step: 0,
        }
    }
}

impl LearningRateScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        let progress = (step % self.t_max) as f32 / self.t_max as f32;
        self.eta_min + (self.initial_lr - self.eta_min) * (0.5 * (1.0 + (std::f32::consts::PI * progress).cos()))
    }
    
    fn step(&mut self, step: usize) {
        self.current_step = step;
    }
}
