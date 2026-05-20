use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct GpuKernelStats {
    pub calls: u64,
    pub total_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl Default for GpuKernelStats {
    fn default() -> Self {
        Self {
            calls: 0,
            total_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
        }
    }
}

#[derive(Debug)]
pub enum ProfileScope {
    Kernel(String),
    Phase(String),
}

#[derive(Debug)]
pub struct GpuProfiler {
    enabled: bool,
    start_times: Vec<(ProfileScope, Instant)>,
    kernel_stats: HashMap<String, GpuKernelStats>,
    phase_log: Vec<(String, Duration)>,
}

impl GpuProfiler {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            start_times: Vec::new(),
            kernel_stats: HashMap::new(),
            phase_log: Vec::new(),
        }
    }

    pub fn begin_kernel(&mut self, name: &str) {
        if !self.enabled {
            return;
        }
        self.start_times
            .push((ProfileScope::Kernel(name.to_string()), Instant::now()));
    }

    pub fn end_kernel(&mut self, name: &str) {
        if !self.enabled {
            return;
        }
        let elapsed = self.pop_start(&ProfileScope::Kernel(name.to_string()));
        if let Some(d) = elapsed {
            let stats = self.kernel_stats.entry(name.to_string()).or_default();
            stats.calls += 1;
            stats.total_time += d;
            if d < stats.min_time {
                stats.min_time = d;
            }
            if d > stats.max_time {
                stats.max_time = d;
            }
        }
    }

    pub fn begin_phase(&mut self, label: &str) {
        if !self.enabled {
            return;
        }
        self.start_times
            .push((ProfileScope::Phase(label.to_string()), Instant::now()));
    }

    pub fn end_phase(&mut self, label: &str) {
        if !self.enabled {
            return;
        }
        let elapsed = self.pop_start(&ProfileScope::Phase(label.to_string()));
        if let Some(d) = elapsed {
            self.phase_log.push((label.to_string(), d));
        }
    }

    fn pop_start(&mut self, scope: &ProfileScope) -> Option<Duration> {
        let idx = self.start_times.iter().rposition(|(s, _)| {
            match (s, scope) {
                (ProfileScope::Kernel(a), ProfileScope::Kernel(b)) => a == b,
                (ProfileScope::Phase(a), ProfileScope::Phase(b)) => a == b,
                _ => false,
            }
        })?;
        let (_, start) = self.start_times.remove(idx);
        Some(start.elapsed())
    }

    pub fn report(&self) -> String {
        if !self.enabled {
            return "GPU profiler disabled".to_string();
        }

        let mut lines = Vec::new();
        lines.push("GPU Profiler Report".to_string());
        lines.push("===================".to_string());

        if !self.kernel_stats.is_empty() {
            lines.push("Kernels:".to_string());
            let mut sorted: Vec<_> = self.kernel_stats.iter().collect();
            sorted.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));
            for (name, stats) in &sorted {
                let avg = stats.total_time.as_secs_f64() / stats.calls as f64;
                lines.push(format!(
                    "  {:25} x{:4}  total:{:.3}s  avg:{:.3}ms  min:{:.3}ms  max:{:.3}ms",
                    name,
                    stats.calls,
                    stats.total_time.as_secs_f64(),
                    avg * 1000.0,
                    stats.min_time.as_secs_f64() * 1000.0,
                    stats.max_time.as_secs_f64() * 1000.0,
                ));
            }
        }

        if !self.phase_log.is_empty() {
            lines.push("Phases:".to_string());
            for (label, duration) in &self.phase_log {
                lines.push(format!(
                    "  {:25}  {:.3}s",
                    label,
                    duration.as_secs_f64()
                ));
            }
        }

        lines.join("\n")
    }

    pub fn clear(&mut self) {
        self.start_times.clear();
        self.kernel_stats.clear();
        self.phase_log.clear();
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}
