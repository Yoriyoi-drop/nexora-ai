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
            SchedulerType::ExponentialDecay { gamma } => self.base_lr * gamma.powf(epoch as f64),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_new() {
        let s = AdaptiveSchedulerNode::new("sched", 0.01, SchedulerType::CosineAnnealing);
        assert_eq!(s.base_lr, 0.01);
        assert_eq!(s.current_lr, 0.01);
    }

    #[test]
    fn test_cosine_annealing() {
        let mut s = AdaptiveSchedulerNode::new("s", 1.0, SchedulerType::CosineAnnealing);
        let lr = s.step(0, None);
        assert!((lr - 1.0).abs() < 1e-5);
        let lr2 = s.step(50, None);
        assert!(lr2 < 1.0);
    }

    #[test]
    fn test_exponential_decay() {
        let mut s = AdaptiveSchedulerNode::new(
            "s",
            1.0,
            SchedulerType::ExponentialDecay { gamma: 0.5 },
        );
        let lr = s.step(1, None);
        assert!((lr - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_step_decay() {
        let mut s = AdaptiveSchedulerNode::new(
            "s",
            1.0,
            SchedulerType::StepDecay {
                step_size: 2,
                gamma: 0.1,
            },
        );
        assert!((s.step(0, None) - 1.0).abs() < 1e-5);
        assert!((s.step(2, None) - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut s = AdaptiveSchedulerNode::new(
            "s",
            1.0,
            SchedulerType::ReduceOnPlateau {
                patience: 2,
                factor: 0.5,
            },
        );
        s.step(0, Some(1.0));
        s.step(1, Some(0.9));
        s.step(2, Some(0.95)); // not improving
        let lr = s.step(3, Some(0.93)); // still not improving
        assert!((lr - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_one_cycle() {
        let mut s = AdaptiveSchedulerNode::new("s", 1.0, SchedulerType::OneCycle);
        let lr = s.step(0, None);
        assert!((lr - 0.0).abs() < 1e-5);
        let lr2 = s.step(50, None);
        assert!((lr2 - 1.0).abs() < 1e-5);
    }
}
