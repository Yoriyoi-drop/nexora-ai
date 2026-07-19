use chrono::Local;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use sysinfo::System;
use nexora_monitoring::{HealthChecker, HealthStatus, MetricsCollector, SystemMetrics};

// ─── Constants ────────────────────────────────────────────────────────────────
pub const CPU_WARN: f32 = 50.0;
pub const CPU_CRIT: f32 = 80.0;
pub const MEM_WARN: f32 = 70.0;
pub const MEM_CRIT: f32 = 90.0;
pub const UPDATE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);
pub const LOG_CAPACITY: usize = 200;
pub const LOG_DISPLAY: usize = 20;

// ─── Tab / Page ───────────────────────────────────────────────────────────────
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTab {
    Overview  = 0,
    Inference = 1,
    Models    = 2,
    Training  = 3,
    Tests     = 4,
    Logs      = 5,
}

impl ActiveTab {
    pub fn next(self) -> Self {
        match self {
            Self::Overview  => Self::Inference,
            Self::Inference => Self::Models,
            Self::Models    => Self::Training,
            Self::Training  => Self::Tests,
            Self::Tests     => Self::Logs,
            Self::Logs      => Self::Overview,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Overview  => Self::Logs,
            Self::Inference => Self::Overview,
            Self::Models    => Self::Inference,
            Self::Training  => Self::Models,
            Self::Tests     => Self::Training,
            Self::Logs      => Self::Tests,
        }
    }

    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Overview),
            1 => Some(Self::Inference),
            2 => Some(Self::Models),
            3 => Some(Self::Training),
            4 => Some(Self::Tests),
            5 => Some(Self::Logs),
            _ => None,
        }
    }

    pub fn as_usize(self) -> usize { self as usize }

    pub fn titles() -> [&'static str; 6] {
        [" Overview ", " Inference ", " Models ", " Training ", " Tests ", " Logs "]
    }
}

// ─── Data Structures ──────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub total_memory: u64,
    pub used_memory: u64,
    pub processes: usize,
    pub uptime: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,
    pub status: String,
    pub duration: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct NextestOutput {
    #[serde(rename = "test_run")]
    pub test_run: TestRun,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct TestRun {
    #[serde(rename = "test_list")]
    pub test_list: Vec<TestCase>,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct TestCase {
    #[serde(rename = "test_name")]
    pub test_name: String,
    pub status: String,
    #[serde(rename = "exec_time")]
    pub exec_time: Option<f64>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
}

// ─── Threshold Alert ──────────────────────────────────────────────────────────
pub enum ThresholdAlert {
    None,
    Warning,
    Critical,
}

impl ThresholdAlert {
    pub fn from_cpu(val: f32) -> Self {
        if val >= CPU_CRIT { Self::Critical }
        else if val >= CPU_WARN { Self::Warning }
        else { Self::None }
    }

    pub fn from_mem(val: f32) -> Self {
        if val >= MEM_CRIT { Self::Critical }
        else if val >= MEM_WARN { Self::Warning }
        else { Self::None }
    }
}

// ─── Monitoring Bridge ────────────────────────────────────────────────────────
pub struct MonitoringBridge {
    pub collector: MetricsCollector,
    pub checker: HealthChecker,
    pub last_cpu_pct: f32,
    pub last_mem_pct: f32,
    pub last_health: HealthStatus,
    pub gpu_detected: bool,
    pub last_gpu_poll: Instant,
    pub gpu_poll_interval: std::time::Duration,
    pub gpu_name: String,
}

impl MonitoringBridge {
    pub fn new() -> Self {
        Self {
            collector: MetricsCollector::default(),
            checker: HealthChecker::new(),
            last_cpu_pct: 0.0,
            last_mem_pct: 0.0,
            last_health: HealthStatus::Healthy,
            gpu_detected: false,
            last_gpu_poll: Instant::now(),
            gpu_poll_interval: std::time::Duration::from_secs(10),
            gpu_name: String::new(),
        }
    }

    pub fn update(&mut self, sys: &SystemInfo) -> Vec<LogEntry> {
        let mut events = Vec::new();

        let cpu_ratio = sys.cpu_usage as f64 / 100.0;
        let mem_bytes = sys.used_memory as f64;

        self.collector.set_cpu_usage(cpu_ratio);
        self.collector.set_memory_usage(mem_bytes);

        let cpu_alert = ThresholdAlert::from_cpu(sys.cpu_usage);
        let mem_alert = ThresholdAlert::from_mem(sys.memory_usage);

        if matches!(cpu_alert, ThresholdAlert::Critical)
            && !matches!(ThresholdAlert::from_cpu(self.last_cpu_pct), ThresholdAlert::Critical)
        {
            events.push(log_entry("CRIT", format!("CPU usage critical: {:.1}%", sys.cpu_usage)));
        } else if matches!(cpu_alert, ThresholdAlert::Warning)
            && matches!(ThresholdAlert::from_cpu(self.last_cpu_pct), ThresholdAlert::None)
        {
            events.push(log_entry("WARN", format!("CPU usage high: {:.1}%", sys.cpu_usage)));
        }

        if matches!(mem_alert, ThresholdAlert::Critical)
            && !matches!(ThresholdAlert::from_mem(self.last_mem_pct), ThresholdAlert::Critical)
        {
            events.push(log_entry("CRIT", format!("Memory usage critical: {:.1}%", sys.memory_usage)));
        } else if matches!(mem_alert, ThresholdAlert::Warning)
            && matches!(ThresholdAlert::from_mem(self.last_mem_pct), ThresholdAlert::None)
        {
            events.push(log_entry("WARN", format!("Memory usage high: {:.1}%", sys.memory_usage)));
        }

        self.last_cpu_pct = sys.cpu_usage;
        self.last_mem_pct = sys.memory_usage;

        self.checker.update_metrics(SystemMetrics {
            cpu_usage_ratio: cpu_ratio,
            memory_usage_bytes: mem_bytes,
            active_connections: 0,
            queue_depth: 0,
            error_rate: 0.0,
            average_latency_ms: 0.0,
        });

        let health = self.checker.check_health();
        if health.status != self.last_health {
            events.push(log_entry("INFO", format!("Health status changed: {}", health.status)));
            self.last_health = health.status.clone();
        }

        events
    }

    pub fn poll_gpu(&mut self) -> Vec<LogEntry> {
        let mut events = Vec::new();
        let now = Instant::now();
        if self.gpu_detected && now.duration_since(self.last_gpu_poll) < self.gpu_poll_interval {
            return events;
        }
        self.last_gpu_poll = now;

        let detected = self.try_nvidia_smi(&mut events)
            || self.try_rocm_smi(&mut events)
            || self.try_drm_sysfs(&mut events);

        if !detected {
            if self.gpu_detected {
                self.gpu_detected = false;
                self.collector.set_gpu_alive(false);
                events.push(log_entry("WARN", "GPU lost — all detection methods failed".to_string()));
            } else {
                self.collector.set_gpu_alive(false);
            }
        }

        events
    }

    fn try_nvidia_smi(&mut self, events: &mut Vec<LogEntry>) -> bool {
        let Ok(out) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                   "--format=csv,noheader,nounits"])
            .output() else { return false; };
        if !out.status.success() { return false; }

        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = match stdout.lines().next() {
            Some(l) if !l.trim().is_empty() => l,
            _ => return false,
        };
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 5 { return false; }

        let gpu_util  = parts[2].parse::<f64>().unwrap_or(0.0);
        let mem_used  = parts[3].parse::<f64>().unwrap_or(0.0);
        let mem_total = parts[4].parse::<f64>().unwrap_or(1.0);
        let mem_pct   = (mem_used / mem_total) * 100.0;
        let mem_bytes = (mem_total * 1_048_576.0) as u64;

        self.apply_gpu_metrics(gpu_util, mem_pct, mem_bytes, events);
        if !self.gpu_detected {
            self.gpu_detected = true;
            self.gpu_name = format!("NVIDIA {}", parts[1]);
            events.push(log_entry("INFO", format!("GPU detected: {} ({}%)", self.gpu_name, gpu_util)));
        }
        true
    }

    fn try_rocm_smi(&mut self, events: &mut Vec<LogEntry>) -> bool {
        let Ok(out) = std::process::Command::new("rocm-smi")
            .args(["--showuse", "--showmeminfo", "vram", "--json"])
            .output() else { return false; };
        if !out.status.success() { return false; }

        let stdout = String::from_utf8_lossy(&out.stdout);
        let json: serde_json::Value = match serde_json::from_str(&stdout) {
            Ok(v) => v, Err(_) => return false,
        };
        let card = match json.as_object().and_then(|m| m.values().next()) {
            Some(v) => v, None => return false,
        };
        let gpu_util = card.get("GPU use (%)")
            .or_else(|| card.get("GPU use"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.trim_end_matches('%').parse::<f64>().ok())
            .unwrap_or(0.0);
        let mem_used_mb  = card.get("VRAM Used Memory (MiB)").and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
        let mem_total_mb = card.get("VRAM Total Memory (MiB)").and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
        let mem_pct   = if mem_total_mb > 0.0 { (mem_used_mb / mem_total_mb) * 100.0 } else { 0.0 };
        let mem_bytes = (mem_total_mb * 1_048_576.0) as u64;

        self.apply_gpu_metrics(gpu_util, mem_pct, mem_bytes, events);
        if !self.gpu_detected {
            self.gpu_detected = true;
            self.gpu_name = "AMD GPU".to_string();
            events.push(log_entry("INFO", format!("GPU detected: AMD (rocm-smi, {}%)", gpu_util)));
        }
        true
    }

    fn try_drm_sysfs(&mut self, events: &mut Vec<LogEntry>) -> bool {
        let drm = std::path::Path::new("/sys/class/drm/");
        if !drm.is_dir() { return false; }
        let mut found = false;
        let mut name  = String::new();
        if let Ok(entries) = std::fs::read_dir(drm) {
            for e in entries.flatten() {
                let n = e.file_name();
                let s = n.to_string_lossy();
                if !s.starts_with("renderD") && !s.starts_with("card") { continue; }
                let vendor = std::fs::read_to_string(e.path().join("device/vendor")).unwrap_or_default();
                let vid = vendor.trim();
                if !vid.is_empty() {
                    found = true;
                    name = format!("{} GPU ({})", match vid {
                        "0x1002"|"1002" => "AMD", "0x10de"|"10de" => "NVIDIA",
                        "0x8086"|"8086" => "Intel", _ => "Unknown",
                    }, s);
                }
            }
        }
        if found {
            self.collector.set_gpu_alive(true);
            if !self.gpu_detected {
                self.gpu_detected = true;
                self.gpu_name = name;
                events.push(log_entry("INFO", format!("GPU detected: {} (driver only)", self.gpu_name)));
            }
            true
        } else { false }
    }

    fn apply_gpu_metrics(&self, util_pct: f64, mem_pct: f64, mem_bytes: u64, _events: &mut Vec<LogEntry>) {
        self.collector.set_gpu_alive(true);
        self.collector.set_gpu_utilization_pct(util_pct);
        self.collector.set_gpu_memory_percent(mem_pct);
        self.collector.set_gpu_memory_bytes(mem_bytes);
    }

    pub fn health_report(&self) -> String {
        let r = self.checker.check_health();
        let checks: Vec<String> = r.checks.iter()
            .map(|c| format!("{} {} ({}ms)", if c.healthy { "✓" } else { "✗" }, c.name, c.latency_ms))
            .collect();
        format!("Status: {}\nUptime: {}s\n{}", r.status, r.uptime_seconds, checks.join("\n"))
    }
}

// ─── App State ────────────────────────────────────────────────────────────────
pub struct App {
    pub system_info: SystemInfo,
    pub test_results: Vec<TestResult>,
    pub logs: Vec<LogEntry>,
    pub selected_test: usize,
    pub log_scroll: usize,
    pub should_quit: bool,
    pub last_update: Instant,
    pub is_running_tests: bool,
    pub sys: System,
    pub monitor: MonitoringBridge,
    pub dirty: bool,
    pub started: Instant,
    pub show_health_detail: bool,
    pub active_tab: ActiveTab,
}

impl App {
    pub fn new() -> Self {
        let mut logs = Vec::new();
        logs.push(log_entry("INFO",
            "Nexora Dashboard — Tab/Shift+Tab: navigate pages | 1-6: jump page | q: quit".to_string()));

        Self {
            system_info: SystemInfo {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                total_memory: 0,
                used_memory: 0,
                processes: 0,
                uptime: "00:00:00".to_string(),
            },
            test_results: vec![],
            logs,
            selected_test: 0,
            log_scroll: 0,
            should_quit: false,
            last_update: Instant::now(),
            is_running_tests: false,
            sys: System::new_all(),
            monitor: MonitoringBridge::new(),
            dirty: true,
            started: Instant::now(),
            show_health_detail: false,
            active_tab: ActiveTab::Overview,
        }
    }

    pub fn update_system_info(&mut self) {
        self.sys.refresh_all();
        self.system_info.cpu_usage     = self.sys.global_cpu_info().cpu_usage();
        self.system_info.total_memory  = self.sys.total_memory();
        self.system_info.used_memory   = self.sys.used_memory();
        self.system_info.memory_usage  =
            (self.system_info.used_memory as f32 / self.system_info.total_memory as f32) * 100.0;
        self.system_info.processes     = self.sys.processes().len();
        let u = System::uptime();
        self.system_info.uptime = format!("{:02}:{:02}:{:02}", u / 3600, (u % 3600) / 60, u % 60);

        for ev in self.monitor.update(&self.system_info) { self.push_log(ev); }
        for ev in self.monitor.poll_gpu()                { self.push_log(ev); }
        self.dirty = true;
    }

    pub fn add_log(&mut self, level: &str, message: &str) {
        self.push_log(log_entry(level, message.to_string()));
    }

    fn push_log(&mut self, entry: LogEntry) {
        self.logs.push(entry);
        while self.logs.len() > LOG_CAPACITY { self.logs.remove(0); }
        self.dirty = true;
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
pub fn log_entry(level: &str, message: impl Into<String>) -> LogEntry {
    LogEntry {
        timestamp: Local::now().format("%H:%M:%S").to_string(),
        level: level.to_string(),
        message: message.into(),
    }
}

pub fn gauge_color(val: f32, warn: f32, crit: f32) -> ratatui::style::Color {
    use ratatui::style::Color;
    if val >= crit { Color::Red }
    else if val >= warn { Color::Yellow }
    else { Color::Green }
}

pub fn format_memory(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{} B", bytes)
    }
}
