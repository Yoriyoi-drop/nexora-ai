use uuid::Uuid;

/// Adaptive Scheduler Node — mengatur learning rate schedule
#[derive(Debug, Clone)]
pub struct AdaptiveSchedulerNode {
    pub id: Uuid,
    pub name: String,
    pub base_lr: f64,
    pub current_lr: f64,
    pub schedule_type: SchedulerType,
    pub metrics: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SchedulerType {
    CosineAnnealing,
    StepDecay { step_size: usize, gamma: f64 },
    ExponentialDecay { gamma: f64 },
    ReduceOnPlateau { patience: usize, factor: f64 },
    OneCycle,
}

impl AdaptiveSchedulerNode {
    pub fn new(name: &str, base_lr: f64, schedule_type: SchedulerType) -> Self {
        AdaptiveSchedulerNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            base_lr,
            current_lr: base_lr,
            schedule_type,
            metrics: Vec::new(),
        }
    }

    /// Update learning rate berdasarkan schedule
    pub fn step(&mut self, epoch: usize, loss: Option<f64>) -> f64 {
        if let Some(l) = loss {
            self.metrics.push(l);
        }

        self.current_lr = match self.schedule_type {
            SchedulerType::CosineAnnealing => {
                let cos = (std::f64::consts::PI * epoch as f64 / 100.0).cos();
                self.base_lr * 0.5 * (1.0 + cos)
            }
            SchedulerType::ExponentialDecay { gamma } => {
                self.base_lr * gamma.powf(epoch as f64)
            }
            SchedulerType::StepDecay { step_size, gamma } => {
                let factor = gamma.powi((epoch / step_size) as i32);
                self.base_lr * factor
            }
            SchedulerType::ReduceOnPlateau { patience, factor } => {
                if self.metrics.len() > patience {
                    let recent = &self.metrics[self.metrics.len() - patience..];
                    let improving = recent.windows(2).any(|w| w[1] < w[0]);
                    if !improving {
                        self.current_lr * factor
                    } else {
                        self.current_lr
                    }
                } else {
                    self.current_lr
                }
            }
            SchedulerType::OneCycle => {
                let half = 50.0;
                if (epoch as f64) < half {
                    self.base_lr * (epoch as f64 / half)
                } else {
                    self.base_lr * (2.0 - epoch as f64 / half).max(0.01)
                }
            }
        };

        self.current_lr
    }
}
