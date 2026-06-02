/// Configuration for EchoNet injection into the transformer pipeline.
#[derive(Debug, Clone)]
pub struct EchoNetInjectionConfig {
    pub inject_after_layer: usize,
    pub phase_separation_strength: f32,
    pub max_window: usize,
    pub alpha: f32,
}

impl Default for EchoNetInjectionConfig {
    fn default() -> Self {
        Self {
            inject_after_layer: 2,
            phase_separation_strength: 0.1,
            max_window: 64,
            alpha: 0.01,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub steps: usize,
    pub final_loss: f64,
    pub total_tokens: usize,
    pub duration_secs: f64,
    pub val_loss: Option<f64>,
    pub val_perplexity: Option<f64>,
}
