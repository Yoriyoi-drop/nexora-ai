use chrono::Local;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use nexora_monitoring::{HealthChecker, HealthStatus, MetricsCollector, SystemMetrics};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::process::Command;

const CPU_WARN: f32 = 50.0;
const CPU_CRIT: f32 = 80.0;
const MEM_WARN: f32 = 70.0;
const MEM_CRIT: f32 = 90.0;
const UPDATE_INTERVAL: Duration = Duration::from_secs(2);
const LOG_CAPACITY: usize = 100;
const LOG_DISPLAY: usize = 12;

#[derive(Debug, Clone)]
struct SystemInfo {
    cpu_usage: f32,
    memory_usage: f32,
    total_memory: u64,
    used_memory: u64,
    processes: usize,
    uptime: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestResult {
    name: String,
    status: String,
    duration: String,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct LogEntry {
    timestamp: String,
    level: String,
    message: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
struct NextestOutput {
    #[serde(rename = "test_run")]
    test_run: TestRun,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
struct TestRun {
    #[serde(rename = "test_list")]
    test_list: Vec<TestCase>,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
struct TestCase {
    #[serde(rename = "test_name")]
    test_name: String,
    status: String,
    #[serde(rename = "exec_time")]
    exec_time: Option<f64>,
    stdout: Option<String>,
    stderr: Option<String>,
}

enum ThresholdAlert {
    None,
    Warning,
    Critical,
}

impl ThresholdAlert {
    fn from_cpu(val: f32) -> Self {
        if val >= CPU_CRIT { Self::Critical }
        else if val >= CPU_WARN { Self::Warning }
        else { Self::None }
    }

    fn from_mem(val: f32) -> Self {
        if val >= MEM_CRIT { Self::Critical }
        else if val >= MEM_WARN { Self::Warning }
        else { Self::None }
    }
}

struct MonitoringBridge {
    collector: MetricsCollector,
    checker: HealthChecker,
    last_cpu_pct: f32,
    last_mem_pct: f32,
    last_health: HealthStatus,
    gpu_detected: bool,
    last_gpu_poll: Instant,
    gpu_poll_interval: Duration,
    gpu_name: String,
}

impl MonitoringBridge {
    fn new() -> Self {
        let collector = MetricsCollector::default();
        let checker = HealthChecker::new();
        Self {
            collector,
            checker,
            last_cpu_pct: 0.0,
            last_mem_pct: 0.0,
            last_health: HealthStatus::Healthy,
            gpu_detected: false,
            last_gpu_poll: Instant::now(),
            gpu_poll_interval: Duration::from_secs(10),
            gpu_name: String::new(),
        }
    }

    fn update(&mut self, sys: &SystemInfo) -> Vec<LogEntry> {
        let mut events = Vec::new();

        let cpu_ratio = (sys.cpu_usage as f64) / 100.0;
        let mem_bytes = sys.used_memory as f64;
        let _mem_ratio = (sys.memory_usage as f64) / 100.0;

        self.collector.set_cpu_usage(cpu_ratio);
        self.collector.set_memory_usage(mem_bytes);

        let cpu_alert = ThresholdAlert::from_cpu(sys.cpu_usage);
        let mem_alert = ThresholdAlert::from_mem(sys.memory_usage);

        if matches!(cpu_alert, ThresholdAlert::Critical)
            && !matches!(ThresholdAlert::from_cpu(self.last_cpu_pct), ThresholdAlert::Critical)
        {
            events.push(LogEntry {
                timestamp: Local::now().format("%H:%M:%S").to_string(),
                level: "CRIT".to_string(),
                message: format!("CPU usage critical: {:.1}%", sys.cpu_usage),
            });
        } else if matches!(cpu_alert, ThresholdAlert::Warning)
            && matches!(ThresholdAlert::from_cpu(self.last_cpu_pct), ThresholdAlert::None)
        {
            events.push(LogEntry {
                timestamp: Local::now().format("%H:%M:%S").to_string(),
                level: "WARN".to_string(),
                message: format!("CPU usage high: {:.1}%", sys.cpu_usage),
            });
        }

        if matches!(mem_alert, ThresholdAlert::Critical)
            && !matches!(ThresholdAlert::from_mem(self.last_mem_pct), ThresholdAlert::Critical)
        {
            events.push(LogEntry {
                timestamp: Local::now().format("%H:%M:%S").to_string(),
                level: "CRIT".to_string(),
                message: format!("Memory usage critical: {:.1}%", sys.memory_usage),
            });
        } else if matches!(mem_alert, ThresholdAlert::Warning)
            && matches!(ThresholdAlert::from_mem(self.last_mem_pct), ThresholdAlert::None)
        {
            events.push(LogEntry {
                timestamp: Local::now().format("%H:%M:%S").to_string(),
                level: "WARN".to_string(),
                message: format!("Memory usage high: {:.1}%", sys.memory_usage),
            });
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
            events.push(LogEntry {
                timestamp: Local::now().format("%H:%M:%S").to_string(),
                level: "INFO".to_string(),
                message: format!("Health status: {}", health.status),
            });
            self.last_health = health.status.clone();
        }

        events
    }

    fn poll_gpu(&mut self) -> Vec<LogEntry> {
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
                events.push(LogEntry {
                    timestamp: Local::now().format("%H:%M:%S").to_string(),
                    level: "WARN".to_string(),
                    message: "GPU lost — all detection methods failed".to_string(),
                });
            } else {
                self.collector.set_gpu_alive(false);
            }
        }

        events
    }

    fn try_nvidia_smi(&mut self, events: &mut Vec<LogEntry>) -> bool {
        let Ok(out) = std::process::Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output() else { return false; };

        if !out.status.success() { return false; }

        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = match stdout.lines().next() {
            Some(l) if !l.trim().is_empty() => l,
            _ => return false,
        };

        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 5 { return false; }

        let gpu_util = parts[2].parse::<f64>().unwrap_or(0.0);
        let mem_used = parts[3].parse::<f64>().unwrap_or(0.0);
        let mem_total = parts[4].parse::<f64>().unwrap_or(1.0);
        let mem_pct = (mem_used / mem_total) * 100.0;
        let mem_bytes = (mem_total * 1_048_576.0) as u64;

        self.apply_gpu_metrics(gpu_util, mem_pct, mem_bytes, events);
        if !self.gpu_detected {
            self.gpu_detected = true;
            self.gpu_name = format!("NVIDIA {}", parts[1]);
            events.push(LogEntry {
                timestamp: Local::now().format("%H:%M:%S").to_string(),
                level: "INFO".to_string(),
                message: format!("GPU detected: {} ({}%)", self.gpu_name, gpu_util),
            });
        }
        true
    }

    fn try_rocm_smi(&mut self, events: &mut Vec<LogEntry>) -> bool {
        let Ok(out) = std::process::Command::new("rocm-smi")
            .args(&["--showuse", "--showmeminfo", "vram", "--json"])
            .output() else { return false; };

        if !out.status.success() { return false; }

        let stdout = String::from_utf8_lossy(&out.stdout);
        let json: serde_json::Value = match serde_json::from_str(&stdout) {
            Ok(v) => v,
            Err(_) => return false,
        };

        let card = match json.as_object().and_then(|m| m.values().next()) {
            Some(v) => v,
            None => return false,
        };

        let gpu_util = card.get("GPU use (%)")
            .or_else(|| card.get("GPU use"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.trim_end_matches('%').parse::<f64>().ok())
            .unwrap_or(0.0);

        let mem_used_mb = card.get("VRAM Total Memory (MiB)")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        let mem_total_mb = card.get("VRAM Total Memory (MiB)")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1.0);

        let mem_pct = if mem_total_mb > 0.0 { (mem_used_mb / mem_total_mb) * 100.0 } else { 0.0 };
        let mem_bytes = (mem_total_mb * 1_048_576.0) as u64;

        self.apply_gpu_metrics(gpu_util, mem_pct, mem_bytes, events);
        if !self.gpu_detected {
            self.gpu_detected = true;
            self.gpu_name = "AMD GPU".to_string();
            events.push(LogEntry {
                timestamp: Local::now().format("%H:%M:%S").to_string(),
                level: "INFO".to_string(),
                message: format!("GPU detected: AMD (rocm-smi, {}%)", gpu_util),
            });
        }
        true
    }

    fn try_drm_sysfs(&mut self, events: &mut Vec<LogEntry>) -> bool {
        let drm_path = std::path::Path::new("/sys/class/drm/");
        if !drm_path.is_dir() { return false; }

        let mut gpu_found = false;
        let mut gpu_name = String::new();

        if let Ok(entries) = std::fs::read_dir(drm_path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();

                if !name_str.starts_with("renderD") && !name_str.starts_with("card") {
                    continue;
                }

                let vendor_path = entry.path().join("device/vendor");

                let vendor = std::fs::read_to_string(&vendor_path).unwrap_or_default();
                let vendor_id = vendor.trim();

                if !vendor_id.is_empty() {
                    gpu_found = true;
                    let vendor_name = match vendor_id {
                        "0x1002" | "1002" => "AMD",
                        "0x10de" | "10de" => "NVIDIA",
                        "0x8086" | "8086" => "Intel",
                        "0x1a03" | "1a03" => "ASPEED",
                        _ => "Unknown",
                    };
                    gpu_name = format!("{} GPU ({})", vendor_name, name_str);
                }
            }
        }

        if gpu_found {
            self.collector.set_gpu_alive(true);
            if !self.gpu_detected {
                self.gpu_detected = true;
                self.gpu_name = gpu_name;
                events.push(LogEntry {
                    timestamp: Local::now().format("%H:%M:%S").to_string(),
                    level: "INFO".to_string(),
                    message: format!("GPU detected: {} (driver only, no usage data)", self.gpu_name),
                });
            }
            true
        } else {
            false
        }
    }

    fn apply_gpu_metrics(&self, util_pct: f64, mem_pct: f64, mem_bytes: u64, _events: &mut Vec<LogEntry>) {
        self.collector.set_gpu_alive(true);
        self.collector.set_gpu_utilization_pct(util_pct);
        self.collector.set_gpu_memory_percent(mem_pct);
        self.collector.set_gpu_memory_bytes(mem_bytes);
    }

    fn health_report(&self) -> String {
        let report = self.checker.check_health();
        let checks: Vec<String> = report
            .checks
            .iter()
            .map(|c| {
                let icon = if c.healthy { "✓" } else { "✗" };
                format!("{} {} ({}ms)", icon, c.name, c.latency_ms)
            })
            .collect();
        format!("Status: {}\nUptime: {}s\n{}", report.status, report.uptime_seconds, checks.join("\n"))
    }
}

struct App {
    system_info: SystemInfo,
    test_results: Vec<TestResult>,
    logs: Vec<LogEntry>,
    selected_test: usize,
    should_quit: bool,
    last_update: Instant,
    last_test_run: Instant,
    is_running_tests: bool,
    sys: System,
    monitor: MonitoringBridge,
    dirty: bool,
    started: Instant,
    show_health_detail: bool,
}

impl App {
    fn new() -> Self {
        let mut logs = Vec::new();
        logs.push(LogEntry {
            timestamp: Local::now().format("%H:%M:%S").to_string(),
            level: "INFO".to_string(),
            message: "Dashboard initialized — press 't' to run tests, 'h' for health detail".to_string(),
        });

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
            should_quit: false,
            last_update: Instant::now(),
            last_test_run: Instant::now(),
            is_running_tests: false,
            sys: System::new_all(),
            monitor: MonitoringBridge::new(),
            dirty: true,
            started: Instant::now(),
            show_health_detail: false,
        }
    }

    async fn run_tests(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.is_running_tests = true;
        self.add_log("INFO", "Starting cargo-nextest run...");

        let output = Command::new("cargo")
            .args(&["nextest", "run", "--message-format=json"])
            .current_dir(std::env::current_dir()?)
            .output()
            .await?;

        if output.status.success() {
            let stdout = String::from_utf8(output.stdout)?;
            self.parse_nextest_output(&stdout)?;
            self.add_log("INFO", &format!("Tests completed: {} total", self.test_results.len()));
        } else {
            let stderr = String::from_utf8(output.stderr)?;
            self.add_log("ERROR", &format!("Test execution failed: {}", stderr));
        }

        self.is_running_tests = false;
        self.last_test_run = Instant::now();
        Ok(())
    }

    fn parse_nextest_output(&mut self, output: &str) -> Result<(), Box<dyn std::error::Error>> {
        let lines: Vec<&str> = output.lines().collect();
        let mut new_test_results = Vec::with_capacity(lines.len());

        for line in lines {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<serde_json::Value>(line) {
                Ok(json) => {
                    if let Some(test_name) = json.get("test_name").and_then(|v| v.as_str()) {
                        let status = json
                            .get("status")
                            .and_then(|v| v.as_str())
                            .unwrap_or("UNKNOWN");
                        let exec_time = json.get("exec_time").and_then(|v| v.as_f64());
                        let stdout = json.get("stdout").and_then(|v| v.as_str());
                        let stderr = json.get("stderr").and_then(|v| v.as_str());

                        let status_icon = match status {
                            "passed" => "✓ PASSED",
                            "failed" => "✗ FAILED",
                            "skipped" => "⚠ SKIPPED",
                            _ => "? UNKNOWN",
                        };

                        let duration = exec_time.map_or_else(|| "N/A".to_string(), |t| format!("{:.2}s", t));

                        let error = if status == "failed" {
                            stderr.or(stdout).map(|s| s.to_string())
                        } else {
                            None
                        };

                        new_test_results.push(TestResult {
                            name: test_name.to_string(),
                            status: status_icon.to_string(),
                            duration,
                            error,
                        });

                        if status == "failed" {
                            self.add_log("ERROR", &format!("Test FAILED: {}", test_name));
                        }
                    }
                }
                Err(_) => {
                    let lower = line.to_lowercase();
                    if lower.contains("error") || lower.contains("failed") {
                        self.add_log("ERROR", line);
                    } else if lower.contains("warning") {
                        self.add_log("WARN", line);
                    } else {
                        self.add_log("INFO", line);
                    }
                }
            }
        }

        if !new_test_results.is_empty() {
            let passed = new_test_results.iter().filter(|t| t.status.starts_with('✓')).count();
            let failed = new_test_results.iter().filter(|t| t.status.starts_with('✗')).count();
            self.test_results = new_test_results;
            self.add_log("INFO", &format!("Results: {} passed, {} failed", passed, failed));
            self.dirty = true;
        }

        Ok(())
    }

    fn update_system_info(&mut self) {
        self.sys.refresh_all();

        self.system_info.cpu_usage = self.sys.global_cpu_info().cpu_usage();
        self.system_info.total_memory = self.sys.total_memory();
        self.system_info.used_memory = self.sys.used_memory();
        self.system_info.memory_usage =
            (self.system_info.used_memory as f32 / self.system_info.total_memory as f32) * 100.0;
        self.system_info.processes = self.sys.processes().len();

        let uptime_secs = System::uptime();
        let hours = uptime_secs / 3600;
        let minutes = (uptime_secs % 3600) / 60;
        let seconds = uptime_secs % 60;
        self.system_info.uptime = format!("{:02}:{:02}:{:02}", hours, minutes, seconds);

        let events = self.monitor.update(&self.system_info);
        for ev in events {
            self.logs.push(ev);
            self.trim_logs();
        }

        let gpu_events = self.monitor.poll_gpu();
        for ev in gpu_events {
            self.logs.push(ev);
            self.trim_logs();
        }

        self.dirty = true;
    }

    fn add_log(&mut self, level: &str, message: &str) {
        self.logs.push(LogEntry {
            timestamp: Local::now().format("%H:%M:%S").to_string(),
            level: level.to_string(),
            message: message.to_string(),
        });
        self.trim_logs();
        self.dirty = true;
    }

    fn trim_logs(&mut self) {
        while self.logs.len() > LOG_CAPACITY {
            self.logs.remove(0);
        }
    }

    fn on_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Up => {
                if self.selected_test > 0 {
                    self.selected_test -= 1;
                    self.dirty = true;
                }
            }
            KeyCode::Down => {
                if self.selected_test < self.test_results.len().saturating_sub(1) {
                    self.selected_test += 1;
                    self.dirty = true;
                }
            }
            KeyCode::Char('r') => {
                self.add_log("INFO", "Manual refresh triggered");
                self.update_system_info();
            }
            KeyCode::Char('h') => {
                self.show_health_detail = !self.show_health_detail;
                self.dirty = true;
            }
            KeyCode::Char('t') => {}
            _ => {}
        }
    }
}

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Min(10),
            Constraint::Length(LOG_DISPLAY as u16 + 2),
        ])
        .split(f.size());

    render_header(f, chunks[0], app);
    render_main(f, chunks[1], app);
    render_logs(f, chunks[2], &app.logs);
}

fn gauge_color(val: f32, warn: f32, crit: f32) -> Color {
    if val >= crit { Color::Red }
    else if val >= warn { Color::Yellow }
    else { Color::Green }
}

fn format_memory(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{} B", bytes)
    }
}

fn render_header(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ])
        .split(area);

    let cpu_color = gauge_color(app.system_info.cpu_usage, CPU_WARN, CPU_CRIT);
    let cpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" CPU "))
        .gauge_style(Style::default().fg(cpu_color))
        .percent((app.system_info.cpu_usage as u16).min(100))
        .label(format!("{:.1}%", app.system_info.cpu_usage));
    f.render_widget(cpu_gauge, chunks[0]);

    let mem_color = gauge_color(app.system_info.memory_usage, MEM_WARN, MEM_CRIT);
    let mem_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Memory "))
        .gauge_style(Style::default().fg(mem_color))
        .percent((app.system_info.memory_usage as u16).min(100))
        .label(format!("{}/{}", format_memory(app.system_info.used_memory), format_memory(app.system_info.total_memory)));
    f.render_widget(mem_gauge, chunks[1]);

    let gpu_alive = app.monitor.collector.gpu_alive.get() as u8;
    let gpu_pct = app.monitor.collector.gpu_utilization_pct.get() as u16;
    let gpu_mem = app.monitor.collector.gpu_memory_percent.get() as u16;
    let gpu_mem_bytes = app.monitor.collector.gpu_memory_bytes.get() as u64;

    let gpu_label = if gpu_alive == 0 {
        "N/A (no GPU)".to_string()
    } else if gpu_mem_bytes > 0 {
        format!("{}%  mem:{:.0}%", gpu_pct, gpu_mem)
    } else {
        format!("{}%", gpu_pct)
    };
    let gpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(format!(" GPU{} ", if app.monitor.gpu_name.is_empty() { String::new() } else { format!(" ({})", &app.monitor.gpu_name) })))
        .gauge_style(Style::default().fg(if gpu_alive == 0 { Color::DarkGray } else { Color::Magenta }))
        .percent(gpu_pct.min(100))
        .label(gpu_label);
    f.render_widget(gpu_gauge, chunks[2]);

    let health_report = app.monitor.checker.check_health();
    let (health_color, health_icon) = match &health_report.status {
        HealthStatus::Healthy => (Color::Green, "✓ HEALTHY"),
        HealthStatus::Degraded => (Color::Yellow, "⚠ DEGRADED"),
        HealthStatus::Unhealthy => (Color::Red, "✗ UNHEALTHY"),
    };
    let health_text = if app.show_health_detail {
        let detail: Vec<String> = health_report.checks.iter().map(|c| {
            format!("{}: {} ({}ms)", c.name, if c.healthy { "ok" } else { "FAIL" }, c.latency_ms)
        }).collect();
        vec![
            Line::from(Span::styled(health_icon, Style::default().fg(health_color).add_modifier(Modifier::BOLD))),
            Line::from(Span::raw(format!("uptime: {}s", health_report.uptime_seconds))),
            Line::from(Span::raw(detail.join(" | "))),
        ]
    } else {
        vec![
            Line::from(Span::styled(health_icon, Style::default().fg(health_color).add_modifier(Modifier::BOLD))),
            Line::from(Span::raw("Press 'h' for details")),
            Line::from(Span::raw(format!("{} checks", health_report.checks.len()))),
        ]
    };
    let health_block = Paragraph::new(health_text)
        .block(Block::default().borders(Borders::ALL).title(" Health "))
        .style(Style::default().fg(health_color));
    f.render_widget(health_block, chunks[3]);

    let sys_lines = vec![
        Line::from(Span::raw(format!("Processes: {}", app.system_info.processes))),
        Line::from(Span::raw(format!("Uptime: {}", app.system_info.uptime))),
        Line::from(Span::raw(format!("Runtime: {}s", app.started.elapsed().as_secs()))),
    ];
    let sys_block = Paragraph::new(sys_lines)
        .block(Block::default().borders(Borders::ALL).title(" System "));
    f.render_widget(sys_block, chunks[4]);
}

fn render_main(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(area);

    render_test_panel(f, chunks[0], app);
    render_metrics_panel(f, chunks[1], app);
}

fn render_test_panel(f: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = app
        .test_results
        .iter()
        .enumerate()
        .map(|(i, test)| {
            let style = if i == app.selected_test {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            let icon = if test.status.starts_with('✓') { "✓" }
                       else if test.status.starts_with('✗') { "✗" }
                       else if test.status.starts_with('⚠') { "⚠" }
                       else { "?" };

            let content = format!(" {} {} ({})", icon, test.name, test.duration);

            ListItem::new(content).style(style)
        })
        .collect();

    let title = if app.is_running_tests {
        " Tests (running...) "
    } else {
        " Tests "
    };

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD));

    f.render_widget(list, area);
}

fn render_metrics_panel(f: &mut Frame, area: Rect, app: &App) {
    let m = &app.monitor.collector;

    let dash = m.tokens_per_sec.get();
    let throughput = m.throughput_tokens.get();
    let latency_p50 = 0.0;
    let latency_p95 = 0.0;
    let cache_hit = m.cache_hit_ratio.get();
    let kv_pressure = m.kv_cache_pressure.get();
    let kv_int_frag = m.kv_internal_frag_ratio.get();
    let kv_ext_frag = m.kv_external_frag_ratio.get();
    let gpu_mem_bytes = m.gpu_memory_bytes.get() as u64;
    let gpu_mem_pct = m.gpu_memory_percent.get();
    let gpu_alive = m.gpu_alive.get() as u8;
    let gpu_tokens = m.gpu_tokens.get();
    let cpu_tok = m.cpu_tokens.get();
    let gpu_fallbacks = m.gpu_cpu_fallbacks.get();
    let req_total = m.request_counter.get();
    let req_fail = m.request_failures.get();
    let queue = m.queue_depth.get() as u64;
    let batching = m.batching_efficiency_pct.get();
    let mem_frag = m.memory_fragmentation.get();
    let train_loss = m.training_loss.get();
    let train_lr = m.training_learning_rate.get();
    let train_grad = m.training_grad_norm.get();

    let mut lines = Vec::new();

    lines.push(Line::from(Span::styled("INFERENCE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
    lines.push(Line::from(vec![
        Span::raw(format!("  Requests: {:.0}  ", req_total)),
        Span::styled(format!("fail: {:.0}", req_fail), Style::default().fg(if req_fail > 0.0 { Color::Red } else { Color::DarkGray })),
        Span::raw(format!("  queue: {}", queue)),
    ]));
    lines.push(Line::from(vec![
        Span::raw(format!("  Tokens/sec: {:.1}  ", dash)),
        Span::raw(format!("  Throughput: {:.0}", throughput)),
    ]));
    if latency_p50 > 0.0 {
        lines.push(Line::from(format!("  Latency: P50={:.0}ms  P95={:.0}ms", latency_p50, latency_p95)));
    } else {
        lines.push(Line::from(Span::styled("  Latency: N/A (no server data)", Style::default().fg(Color::DarkGray))));
    }
    lines.push(Line::from(Span::raw("")));

    lines.push(Line::from(Span::styled("KV CACHE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
    let cache_color = if cache_hit > 0.8 { Color::Green } else if cache_hit > 0.5 { Color::Yellow } else { Color::Red };
    lines.push(Line::from(vec![
        Span::styled(format!("  Hit ratio: {:.1}%", cache_hit * 100.0), Style::default().fg(cache_color)),
        Span::raw(format!("  pressure: {:.1}%", kv_pressure * 100.0)),
    ]));
    lines.push(Line::from(vec![
        Span::raw(format!("  Internal frag: {:.1}%  ", kv_int_frag * 100.0)),
        Span::raw(format!("  External frag: {:.1}%", kv_ext_frag * 100.0)),
    ]));
    lines.push(Line::from(Span::raw("")));

    lines.push(Line::from(Span::styled("GPU", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
    if gpu_alive == 1 {
        lines.push(Line::from(vec![
            Span::raw(format!("  Status: {}  ", "ALIVE")),
            Span::raw(format!("  Memory: {}/{}", format_memory(gpu_mem_bytes), format_memory(if gpu_mem_pct > 0.0 { (gpu_mem_bytes as f64 / (gpu_mem_pct / 100.0)) as u64 } else { 0 }))),
        ]));
        lines.push(Line::from(vec![
            Span::raw(format!("  Tokens (GPU): {:.0}  ", gpu_tokens)),
            Span::raw(format!("  Tokens (CPU): {:.0}  ", cpu_tok)),
            Span::styled(format!("  Fallbacks: {:.0}", gpu_fallbacks), Style::default().fg(if gpu_fallbacks > 0.0 { Color::Yellow } else { Color::DarkGray })),
        ]));
        lines.push(Line::from(vec![
            Span::raw(format!("  Batching eff: {:.1}%  ", batching)),
            Span::raw(format!("  Mem frag: {:.1}%", mem_frag * 100.0)),
        ]));
    } else {
        lines.push(Line::from(Span::styled("  No GPU detected", Style::default().fg(Color::DarkGray))));
    }
    lines.push(Line::from(Span::raw("")));

    lines.push(Line::from(Span::styled("TRAINING", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
    if train_loss > 0.0 || train_lr > 0.0 {
        lines.push(Line::from(vec![
            Span::raw(format!("  Loss: {:.4}  ", train_loss)),
            Span::raw(format!("  LR: {:.8}  ", train_lr)),
            Span::raw(format!("  Grad norm: {:.4}", train_grad)),
        ]));
    } else {
        lines.push(Line::from(Span::styled("  No active training", Style::default().fg(Color::DarkGray))));
    }
    lines.push(Line::from(Span::raw("")));

    if let Some(test) = app.test_results.get(app.selected_test) {
        lines.push(Line::from(Span::styled("SELECTED TEST", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
        lines.push(Line::from(vec![
            Span::styled("  Name: ", Style::default().fg(Color::White)),
            Span::raw(&test.name),
        ]));
        let status_color = if test.status.starts_with('✓') { Color::Green }
                           else if test.status.starts_with('✗') { Color::Red }
                           else { Color::Yellow };
        lines.push(Line::from(vec![
            Span::styled("  Status: ", Style::default().fg(Color::White)),
            Span::styled(&test.status, Style::default().fg(status_color)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  Duration: ", Style::default().fg(Color::White)),
            Span::raw(&test.duration),
        ]));
        if let Some(err) = &test.error {
            lines.push(Line::from(Span::styled("  Error:", Style::default().fg(Color::Red))));
            lines.push(Line::from(Span::styled(format!("    {}", err), Style::default().fg(Color::LightRed))));
        }
    } else {
        lines.push(Line::from(Span::styled("Select a test result for details", Style::default().fg(Color::DarkGray))));
    }

    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" Metrics "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, area);
}

fn render_logs(f: &mut Frame, area: Rect, logs: &[LogEntry]) {
    let start = logs.len().saturating_sub(LOG_DISPLAY);
    let displayed = &logs[start..];

    let items: Vec<ListItem> = displayed
        .iter()
        .map(|log| {
            let style = match log.level.as_str() {
                "CRIT" => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                "ERROR" => Style::default().fg(Color::Red),
                "WARN" => Style::default().fg(Color::Yellow),
                "INFO" => Style::default().fg(Color::Blue),
                _ => Style::default(),
            };
            let content = format!("[{}] {} {}", log.timestamp, log.level, log.message);
            ListItem::new(content).style(style)
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Events (q:quit t:tests r:refresh h:health) "),
    );

    f.render_widget(list, area);
}

async fn run_app(mut terminal: Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    let mut app = App::new();
    app.update_system_info();
    app.add_log("INFO", &format!("System: {} CPU cores, {} total memory",
        app.sys.cpus().len(),
        format_memory(app.system_info.total_memory)));

    let mut last_log_time = Instant::now();

    loop {
        if app.last_update.elapsed() >= UPDATE_INTERVAL {
            app.update_system_info();
            app.last_update = Instant::now();

            if last_log_time.elapsed() >= Duration::from_secs(30) {
                app.add_log("INFO", &format!("CPU: {:.1}%  Mem: {:.1}%  Processes: {}  Health: {}",
                    app.system_info.cpu_usage,
                    app.system_info.memory_usage,
                    app.system_info.processes,
                    app.monitor.health_report().lines().next().unwrap_or("unknown")));
                last_log_time = Instant::now();
            }
        }

        if app.dirty {
            terminal.draw(|f| ui(f, &app))?;
            app.dirty = false;
        }

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                let manual_test_trigger = key.code == KeyCode::Char('t') && !app.is_running_tests;
                app.on_key(key.code);

                if manual_test_trigger {
                    if let Err(e) = app.run_tests().await {
                        app.add_log("ERROR", &format!("Failed to run tests: {}", e));
                    }
                }

                if app.should_quit {
                    break;
                }
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;

    let res = run_app(terminal).await;

    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    io::stdout().flush()?;
    res
}

#[cfg(test)]
mod tests;
