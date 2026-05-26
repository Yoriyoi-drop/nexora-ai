use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
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
            + (self.total_epochs - self.current_epoch).saturating_sub(1) as f64
                * self.total_samples as f64;
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
            self.current_epoch,
            self.total_epochs,
            self.samples_processed,
            self.total_samples,
            pct,
            speed,
            elapsed,
            eta,
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
    /// Ratio of time GPU spends waiting for data vs total time (0.0-1.0).
    /// Updated periodically by the streaming pipeline.
    pub gpu_wait_ratio: f64,
    /// How many samples have been checked for GPU starvation
    starvation_check_count: u64,
    /// Accumulated GPU wait time in seconds
    total_gpu_wait_secs: f64,
    /// Accumulated total observation time in seconds
    total_observe_secs: f64,
}

impl StreamingStats {
    pub fn new() -> Self {
        Self {
            read_speed: 0.0,
            decompress_speed: 0.0,
            queue_depth: 0,
            gpu_starvation: false,
            memory_mb: 0,
            gpu_wait_ratio: 0.0,
            starvation_check_count: 0,
            total_gpu_wait_secs: 0.0,
            total_observe_secs: 0.0,
        }
    }

    /// Report a GPU wait observation.
    /// `wait_secs`: time GPU spent waiting for data in this interval.
    /// `total_secs`: total wall-clock time for this interval.
    ///
    /// If GPU waits more than 30% of the time over the observation window,
    /// `gpu_starvation` is set to true; once set, it stays set until reset.
    /// Call this periodically from the streaming stats update cycle.
    pub fn report_gpu_wait(&mut self, wait_secs: f64, total_secs: f64) {
        self.starvation_check_count += 1;
        self.total_gpu_wait_secs += wait_secs;
        self.total_observe_secs += total_secs;

        if self.total_observe_secs > 0.0 {
            self.gpu_wait_ratio = self.total_gpu_wait_secs / self.total_observe_secs;
        }

        const GPU_STARVATION_THRESHOLD: f64 = 0.30;
        if self.gpu_wait_ratio >= GPU_STARVATION_THRESHOLD {
            self.gpu_starvation = true;
        }
    }

    /// Get the observed GPU wait ratio (0.0-1.0).
    pub fn gpu_wait_ratio(&self) -> f64 {
        self.gpu_wait_ratio
    }

    /// Reset starvation tracking counters (keeps `gpu_starvation` flag as-is).
    pub fn reset_starvation_counters(&mut self) {
        self.starvation_check_count = 0;
        self.total_gpu_wait_secs = 0.0;
        self.total_observe_secs = 0.0;
    }

    pub fn detect_bottleneck(&self) -> Option<String> {
        if self.gpu_starvation {
            return Some(
                format!(
                    "GPU STARVATION: data pipeline too slow (wait ratio {:.1}%)",
                    self.gpu_wait_ratio * 100.0
                ),
            );
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
        std::fs::write(path, content).map_err(|e| ResumeError::Io(e.to_string()))?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, ResumeError> {
        let content = std::fs::read_to_string(path).map_err(|e| ResumeError::Io(e.to_string()))?;
        serde_json::from_str(&content).map_err(|e| ResumeError::Parse(e.to_string()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_progress_tracker_new() {
        let p = ProgressTracker::new(1000, 5);
        assert_eq!(p.total_samples, 1000);
        assert_eq!(p.total_epochs, 5);
        assert_eq!(p.current_epoch, 0);
        assert_eq!(p.samples_processed, 0);
    }

    #[test]
    fn test_start_epoch() {
        let mut p = ProgressTracker::new(1000, 5);
        p.start_epoch(3);
        assert_eq!(p.current_epoch, 3);
    }

    #[test]
    fn test_add_samples() {
        let mut p = ProgressTracker::new(1000, 1);
        p.add_samples(100, 5);
        assert_eq!(p.samples_processed, 100);
        assert_eq!(p.batches_processed, 5);
    }

    #[test]
    fn test_eta_zero_speed() {
        let p = ProgressTracker::new(1000, 1);
        assert_eq!(p.eta(), Duration::from_secs(0));
    }

    #[test]
    fn test_streaming_stats_new() {
        let s = StreamingStats::new();
        assert_eq!(s.read_speed, 0.0);
        assert!(!s.gpu_starvation);
        assert_eq!(s.gpu_wait_ratio, 0.0);
    }

    #[test]
    fn test_gpu_wait_below_threshold() {
        let mut s = StreamingStats::new();
        s.report_gpu_wait(0.1, 1.0);
        assert!(!s.gpu_starvation);
        assert!((s.gpu_wait_ratio() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_gpu_starvation_detected() {
        let mut s = StreamingStats::new();
        s.report_gpu_wait(0.4, 1.0);
        assert!(s.gpu_starvation);
    }

    #[test]
    fn test_reset_starvation_counters() {
        let mut s = StreamingStats::new();
        s.report_gpu_wait(0.5, 1.0);
        assert!(s.gpu_starvation);
        s.reset_starvation_counters();
        assert_eq!(s.gpu_wait_ratio(), 0.5);
    }

    #[test]
    fn test_detect_bottleneck_gpu_starvation() {
        let mut s = StreamingStats::new();
        s.report_gpu_wait(0.5, 1.0);
        let bottleneck = s.detect_bottleneck();
        assert!(bottleneck.unwrap().contains("GPU STARVATION"));
    }

    #[test]
    fn test_detect_bottleneck_low_speed() {
        let s = StreamingStats::new();
        let bottleneck = s.detect_bottleneck();
        assert!(bottleneck.unwrap().contains("BOTTLENECK"));
    }

    #[test]
    fn test_resume_state_json_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("resume.json");
        let state = ResumeState {
            epoch: 2,
            shard_index: 5,
            sample_offset: 1000,
            optimizer_state: None,
            best_val_loss: Some(0.5),
        };
        state.save(&path).unwrap();
        let loaded = ResumeState::load(&path).unwrap();
        assert_eq!(loaded.epoch, 2);
        assert_eq!(loaded.shard_index, 5);
        assert_eq!(loaded.sample_offset, 1000);
        assert_eq!(loaded.best_val_loss, Some(0.5));
    }

    #[test]
    fn test_resume_state_load_nonexistent() {
        let result = ResumeState::load(std::path::Path::new("/nonexistent.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_resume_error_display() {
        let e = ResumeError::Io("file not found".into());
        assert_eq!(format!("{}", e), "Resume IO: file not found");
    }
}
