use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone)]
pub struct ProgressTracker {
    pub current_epoch: usize,
    pub total_epochs: usize,
    pub samples_processed: u64,
    pub total_samples: u64,
    pub batches_processed: u64,
    pub total_batches: u64,
    start_time: Instant,
    epoch_start_time: Instant,
    last_report_time: Instant,
    last_report_samples: u64,
    running_avg_speed: f64,
}

impl ProgressTracker {
    pub fn new(total_samples: u64, total_epochs: usize) -> Self {
        let now = Instant::now();
        Self {
            current_epoch: 0,
            total_epochs,
            samples_processed: 0,
            total_samples,
            batches_processed: 0,
            total_batches: 0,
            start_time: now,
            epoch_start_time: now,
            last_report_time: now,
            last_report_samples: 0,
            running_avg_speed: 0.0,
        }
    }

    pub fn start_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
        self.epoch_start_time = Instant::now();
        info!("Epoch {}/{} dimulai", epoch, self.total_epochs);
    }

    pub fn add_samples(&mut self, count: u64, batches: u64) {
        self.samples_processed += count;
        self.batches_processed += batches;
        self.update_speed();
    }

    fn update_speed(&mut self) {
        let now = Instant::now();
        let dt = now - self.last_report_time;
        if dt >= Duration::from_secs(2) {
            let ds = self.samples_processed - self.last_report_samples;
            let speed = ds as f64 / dt.as_secs_f64();
            self.running_avg_speed = if self.running_avg_speed == 0.0 {
                speed
            } else {
                0.7 * self.running_avg_speed + 0.3 * speed
            };
            self.last_report_time = now;
            self.last_report_samples = self.samples_processed;
        }
    }

    pub fn speed(&self) -> f64 {
        self.running_avg_speed
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn epoch_elapsed(&self) -> Duration {
        self.epoch_start_time.elapsed()
    }

    pub fn eta(&self) -> Duration {
        let speed = self.speed();
        if speed <= 0.0 {
            return Duration::from_secs(0);
        }
        let remaining = (self.total_samples - self.samples_processed) as f64
            + (self.total_epochs - self.current_epoch).saturating_sub(1) as f64 * self.total_samples as f64;
        Duration::from_secs_f64(remaining / speed)
    }

    pub fn report(&self) {
        let speed = self.speed();
        let elapsed = self.elapsed();
        let eta = self.eta();
        let pct = if self.total_samples > 0 {
            self.samples_processed as f64 / self.total_samples as f64 * 100.0
        } else {
            0.0
        };

        info!(
            "Epoch {}/{} | {}/{} samples ({:.1}%) | {:.0} samples/s | elapsed: {:?} | ETA: {:?}",
            self.current_epoch, self.total_epochs,
            self.samples_processed, self.total_samples, pct,
            speed, elapsed, eta,
        );
    }
}

#[derive(Debug, Clone)]
pub struct StreamingStats {
    pub read_speed: f64,
    pub decompress_speed: f64,
    pub queue_depth: usize,
    pub gpu_starvation: bool,
    pub memory_mb: u64,
}

impl StreamingStats {
    pub fn new() -> Self {
        Self {
            read_speed: 0.0,
            decompress_speed: 0.0,
            queue_depth: 0,
            gpu_starvation: false,
            memory_mb: 0,
        }
    }

    pub fn detect_bottleneck(&self) -> Option<String> {
        if self.gpu_starvation {
            return Some("GPU STARVATION: data pipeline too slow".into());
        }
        if self.queue_depth == 0 && self.read_speed < 1000.0 {
            return Some("DATA BOTTLENECK: read speed too low".into());
        }
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeState {
    pub epoch: usize,
    pub shard_index: usize,
    pub sample_offset: u64,
    pub optimizer_state: Option<Vec<u8>>,
    pub best_val_loss: Option<f64>,
}

impl ResumeState {
    pub fn save(&self, path: &std::path::Path) -> Result<(), ResumeError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| ResumeError::Serialize(e.to_string()))?;
        std::fs::write(path, content)
            .map_err(|e| ResumeError::Io(e.to_string()))?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, ResumeError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ResumeError::Io(e.to_string()))?;
        serde_json::from_str(&content)
            .map_err(|e| ResumeError::Parse(e.to_string()))
    }
}

#[derive(Debug)]
pub enum ResumeError {
    Io(String),
    Parse(String),
    Serialize(String),
}

impl std::fmt::Display for ResumeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResumeError::Io(msg) => write!(f, "Resume IO: {}", msg),
            ResumeError::Parse(msg) => write!(f, "Resume parse: {}", msg),
            ResumeError::Serialize(msg) => write!(f, "Resume serialize: {}", msg),
        }
    }
}

impl std::error::Error for ResumeError {}
