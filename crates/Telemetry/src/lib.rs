pub mod store;
pub mod rate_limiter;
pub mod recorder;

use std::sync::Arc;
use std::path::PathBuf;
use tokio::time::{interval, Duration};
use store::TelemetryStore;
use rate_limiter::RateLimiter;
use recorder::TelemetryRecorder;

pub struct TelemetrySystem {
    pub store: Arc<TelemetryStore>,
    pub rate_limiter: RateLimiter,
    pub recorder: TelemetryRecorder,
    persistence_path: Option<PathBuf>,
}

impl TelemetrySystem {
    pub fn new(requests_per_minute: Option<u32>) -> Self {
        let store = Arc::new(TelemetryStore::new(10000, 5000, 1000));
        let rps = requests_per_minute.unwrap_or(60) as f64;
        let rate_limiter = RateLimiter::new(rps, rps * 2.0);
        let recorder = TelemetryRecorder::new(store.clone());
        Self { store, rate_limiter, recorder, persistence_path: None }
    }

    pub fn new_with_store(requests_per_minute: Option<u32>, store: Arc<TelemetryStore>) -> Self {
        let rps = requests_per_minute.unwrap_or(60) as f64;
        let rate_limiter = RateLimiter::new(rps, rps * 2.0);
        let recorder = TelemetryRecorder::new(store.clone());
        Self { store, rate_limiter, recorder, persistence_path: None }
    }

    pub fn with_persistence(mut self, path: impl Into<PathBuf>) -> Self {
        self.persistence_path = Some(path.into());
        self
    }

    pub async fn save(&self) {
        if let Some(ref path) = self.persistence_path {
            let _ = self.store.save_to_file(path.to_str().unwrap_or("telemetry.json")).await;
        }
    }

    pub async fn load(path: &str) -> Self {
        let store = Arc::new(TelemetryStore::load_from_file(path).await);
        let rate_limiter = RateLimiter::new(60.0, 120.0);
        let recorder = TelemetryRecorder::new(store.clone());
        Self { store, rate_limiter, recorder, persistence_path: Some(PathBuf::from(path)) }
    }

    pub async fn prune(&self) {
        self.store.prune().await;
    }

    pub fn spawn_background_tasks(self: &Arc<Self>) {
        let this = self.clone();
        tokio::spawn(async move {
            let mut timer = interval(Duration::from_secs(30));
            loop {
                timer.tick().await;
                this.prune().await;
                this.save().await;
            }
        });
    }
}

impl Clone for TelemetrySystem {
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            rate_limiter: self.rate_limiter.clone(),
            recorder: self.recorder.clone(),
            persistence_path: self.persistence_path.clone(),
        }
    }
}
