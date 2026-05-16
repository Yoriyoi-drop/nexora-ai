use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    pub enabled: bool,
    pub sample_interval_ms: u64,
    pub max_samples: usize,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_interval_ms: 100,
            max_samples: 10_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CpuSample {
    pub timestamp_ms: u64,
    pub usage_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct MemorySample {
    pub timestamp_ms: u64,
    pub usage_bytes: f64,
}

#[derive(Debug, Clone)]
pub struct Profiler {
    config: ProfilingConfig,
    cpu_samples: Vec<CpuSample>,
    memory_samples: Vec<MemorySample>,
    start_time: Option<Instant>,
}

impl Profiler {
    pub fn new(config: ProfilingConfig) -> Self {
        Self {
            config,
            cpu_samples: Vec::new(),
            memory_samples: Vec::new(),
            start_time: None,
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.cpu_samples.clear();
        self.memory_samples.clear();
    }

    pub fn record_cpu(&mut self, usage_ratio: f64) {
        if !self.config.enabled {
            return;
        }
        if self.cpu_samples.len() >= self.config.max_samples {
            return;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.cpu_samples.push(CpuSample {
            timestamp_ms: now,
            usage_ratio,
        });
    }

    pub fn record_memory(&mut self, usage_bytes: f64) {
        if !self.config.enabled {
            return;
        }
        if self.memory_samples.len() >= self.config.max_samples {
            return;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.memory_samples.push(MemorySample {
            timestamp_ms: now,
            usage_bytes,
        });
    }

    pub fn cpu_samples(&self) -> &[CpuSample] {
        &self.cpu_samples
    }

    pub fn memory_samples(&self) -> &[MemorySample] {
        &self.memory_samples
    }

    pub fn elapsed_seconds(&self) -> Option<f64> {
        self.start_time
            .map(|t| t.elapsed().as_secs_f64())
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new(ProfilingConfig::default())
    }
}
