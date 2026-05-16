#![allow(dead_code, unused_imports, unused_variables, unused_mut)]

use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io::{self, Write};
use sysinfo::System;
use chrono::Local;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::time::{Duration, Instant};
use tokio::process::Command;
use serde::{Deserialize, Serialize};

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

// Struct untuk parsing cargo-nextest JSON output
#[derive(Debug, Serialize, Deserialize)]
struct NextestOutput {
    #[serde(rename = "test_run")]
    test_run: TestRun,
}

#[derive(Debug, Serialize, Deserialize)]
struct TestRun {
    #[serde(rename = "test_list")]
    test_list: Vec<TestCase>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TestCase {
    #[serde(rename = "test_name")]
    test_name: String,
    #[serde(rename = "status")]
    status: String,
    #[serde(rename = "exec_time")]
    exec_time: Option<f64>,
    #[serde(rename = "stdout")]
    stdout: Option<String>,
    #[serde(rename = "stderr")]
    stderr: Option<String>,
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
}

impl App {
    fn new() -> Self {
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
            logs: vec![
                LogEntry {
                    timestamp: "15:23:45".to_string(),
                    level: "INFO".to_string(),
                    message: "Dashboard initialized".to_string(),
                },
            ],
            selected_test: 0,
            should_quit: false,
            last_update: Instant::now(),
            last_test_run: Instant::now(),
            is_running_tests: false,
        }
    }

    async fn run_tests(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.is_running_tests = true;
        self.add_log("INFO", "Starting cargo-nextest run...");
        
        // Run cargo-nextest with JSON output
        let output = Command::new("cargo")
            .args(&["nextest", "run", "--message-format=json"])
            .current_dir(std::env::current_dir()?)
            .output()
            .await?;

        if output.status.success() {
            let stdout = String::from_utf8(output.stdout)?;
            self.parse_nextest_output(&stdout)?;
            self.add_log("INFO", "Tests completed successfully");
        } else {
            let stderr = String::from_utf8(output.stderr)?;
            self.add_log("ERROR", &format!("Test execution failed: {}", stderr));
        }
        
        self.is_running_tests = false;
        self.last_test_run = Instant::now();
        Ok(())
    }

    fn parse_nextest_output(&mut self, output: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Parse each line as potential JSON (nextest outputs multiple JSON objects)
        let mut new_test_results = Vec::new();
        
        for line in output.lines() {
            if line.trim().is_empty() {
                continue;
            }
            
            // Try to parse as JSON
            match serde_json::from_str::<serde_json::Value>(line) {
                Ok(json) => {
                    if let Some(test_name) = json.get("test_name").and_then(|v| v.as_str()) {
                        let status = json.get("status").and_then(|v| v.as_str()).unwrap_or("UNKNOWN");
                        let exec_time = json.get("exec_time").and_then(|v| v.as_f64());
                        let stdout = json.get("stdout").and_then(|v| v.as_str());
                        let stderr = json.get("stderr").and_then(|v| v.as_str());
                        
                        let status_icon = match status {
                            "passed" => "✓ PASSED",
                            "failed" => "✗ FAILED",
                            "skipped" => "⚠ SKIPPED",
                            _ => "? UNKNOWN",
                        };
                        
                        let duration = if let Some(time) = exec_time {
                            format!("{:.2}s", time)
                        } else {
                            "N/A".to_string()
                        };
                        
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
                        
                        self.add_log("INFO", &format!("Test: {} - {}", test_name, status));
                    }
                }
                Err(_) => {
                    // Line is not JSON, might be regular output
                    if line.contains("error") || line.contains("failed") {
                        self.add_log("ERROR", line);
                    } else if line.contains("warning") {
                        self.add_log("WARN", line);
                    } else {
                        self.add_log("INFO", line);
                    }
                }
            }
        }
        
        if !new_test_results.is_empty() {
            self.test_results = new_test_results;
            self.add_log("INFO", &format!("Updated {} test results", self.test_results.len()));
        }
        
        Ok(())
    }

    fn update_system_info(&mut self) {
        let mut system = System::new_all();
        system.refresh_all();
        
        self.system_info.cpu_usage = system.global_cpu_info().cpu_usage();
        self.system_info.total_memory = system.total_memory();
        self.system_info.used_memory = system.used_memory();
        self.system_info.memory_usage = (self.system_info.used_memory as f32 / self.system_info.total_memory as f32) * 100.0;
        self.system_info.processes = system.processes().len();
        
        let uptime = System::uptime();
        let hours = uptime / 3600;
        let minutes = (uptime % 3600) / 60;
        let seconds = uptime % 60;
        self.system_info.uptime = format!("{:02}:{:02}:{:02}", hours, minutes, seconds);
    }

    fn add_log(&mut self, level: &str, message: &str) {
        let now = Local::now();
        let timestamp = now.format("%H:%M:%S").to_string();
        
        self.logs.push(LogEntry {
            timestamp,
            level: level.to_string(),
            message: message.to_string(),
        });
        
        // Keep only last 100 logs
        if self.logs.len() > 100 {
            self.logs.remove(0);
        }
    }

    fn on_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Up => {
                if self.selected_test > 0 {
                    self.selected_test -= 1;
                }
            }
            KeyCode::Down => {
                if self.selected_test < self.test_results.len().saturating_sub(1) {
                    self.selected_test += 1;
                }
            }
            KeyCode::Char('r') => {
                self.add_log("INFO", "Manual refresh triggered");
                self.update_system_info();
            }
            KeyCode::Char('t') => {
                if !self.is_running_tests {
                    self.add_log("INFO", "Manual test run triggered");
                    // Note: This will be handled in the main loop
                }
            }
            _ => {}
        }
    }
}

fn ui(f: &mut Frame, app: &App) {
    // Create main layout with 3 panels
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(10),  // Header
            Constraint::Percentage(75),  // Main content
            Constraint::Percentage(15),  // Logs
        ])
        .split(f.size());

    // Header
    render_header(f, chunks[0], app);

    // Main content split into left (tests) and right (details)
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(chunks[1]);

    render_test_panel(f, main_chunks[0], app);
    render_details_panel(f, main_chunks[1], app);

    // Logs panel
    render_logs(f, chunks[2], &app.logs);
}

fn render_header(f: &mut Frame, area: Rect, app: &App) {
    let header_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    // CPU Usage
    let cpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("CPU Usage"))
        .gauge_style(Style::default().fg(Color::Green))
        .percent(app.system_info.cpu_usage as u16)
        .label(format!("{:.1}%", app.system_info.cpu_usage));

    f.render_widget(cpu_gauge, header_chunks[0]);

    // Memory Usage
    let mem_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Memory Usage"))
        .gauge_style(Style::default().fg(Color::Blue))
        .percent(app.system_info.memory_usage as u16)
        .label(format!("{:.1}%", app.system_info.memory_usage));

    f.render_widget(mem_gauge, header_chunks[1]);

    // System Info
    let sys_info = Paragraph::new(vec![
        Line::from(format!("Processes: {}", app.system_info.processes)),
        Line::from(format!("Uptime: {}", app.system_info.uptime)),
    ])
    .block(Block::default().borders(Borders::ALL).title("System"));

    f.render_widget(sys_info, header_chunks[2]);

    // Status
    let status_text = if app.is_running_tests {
        vec![
            Line::from("Nexora AI Dashboard"),
            Line::from("Status: Running Tests"),
        ]
    } else {
        vec![
            Line::from("Nexora AI Dashboard"),
            Line::from("Status: Ready"),
        ]
    };

    let status_color = if app.is_running_tests {
        Color::Yellow
    } else {
        Color::Green
    };

    let status = Paragraph::new(status_text)
    .block(Block::default().borders(Borders::ALL).title("Status"))
    .style(Style::default().fg(status_color));

    f.render_widget(status, header_chunks[3]);
}

fn render_test_panel(f: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = app.test_results
        .iter()
        .enumerate()
        .map(|(i, test)| {
            let style = if i == app.selected_test {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            let content = format!("{} {} ({})", 
                test.name, 
                match test.status.as_str() {
                    "✓ PASSED" => "✓",
                    "✗ FAILED" => "✗", 
                    "⚠ RUNNING" => "⚠",
                    _ => "?",
                },
                test.duration
            );

            ListItem::new(content).style(style)
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Test Results"))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD));

    f.render_widget(list, area);
}

fn render_details_panel(f: &mut Frame, area: Rect, app: &App) {
    if let Some(test) = app.test_results.get(app.selected_test) {
        let details = vec![
            Line::from(vec![
                Span::styled("Test: ", Style::default().fg(Color::Cyan)),
                Span::raw(&test.name),
            ]),
            Line::from(vec![
                Span::styled("Status: ", Style::default().fg(Color::Cyan)),
                Span::styled(&test.status, match test.status.as_str() {
                    "✓ PASSED" => Style::default().fg(Color::Green),
                    "✗ FAILED" => Style::default().fg(Color::Red),
                    "⚠ RUNNING" => Style::default().fg(Color::Yellow),
                    _ => Style::default(),
                }),
            ]),
            Line::from(vec![
                Span::styled("Duration: ", Style::default().fg(Color::Cyan)),
                Span::raw(&test.duration),
            ]),
        ];

        let mut error_details = details.clone();
        if let Some(error) = &test.error {
            error_details.push(Line::from(""));
            error_details.push(Line::from(vec![
                Span::styled("Error: ", Style::default().fg(Color::Red)),
                Span::raw(error),
            ]));
        }

        let paragraph = Paragraph::new(error_details)
            .block(Block::default().borders(Borders::ALL).title("Test Details"))
            .wrap(Wrap { trim: true });

        f.render_widget(paragraph, area);
    }
}

fn render_logs(f: &mut Frame, area: Rect, logs: &[LogEntry]) {
    let items: Vec<ListItem> = logs.iter().rev().take(10).rev().map(|log| {
        let style = match log.level.as_str() {
            "ERROR" => Style::default().fg(Color::Red),
            "WARN" => Style::default().fg(Color::Yellow),
            "INFO" => Style::default().fg(Color::Blue),
            _ => Style::default(),
        };

        let content = format!("[{}] {}: {}", log.timestamp, log.level, log.message);
        ListItem::new(content).style(style)
    }).collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Realtime Logs"));

    f.render_widget(list, area);
}

async fn run_app(mut terminal: Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    let mut app = App::new();
    app.update_system_info();
    
    let mut last_log_time = Instant::now();
    let mut should_run_tests = true; // Auto-run tests on startup

    loop {
        // Auto-run tests on startup
        if should_run_tests && !app.is_running_tests {
            should_run_tests = false;
            if let Err(e) = app.run_tests().await {
                app.add_log("ERROR", &format!("Failed to run tests: {}", e));
            }
        }

        // Update system info every 2 seconds
        if app.last_update.elapsed() >= Duration::from_secs(2) {
            app.update_system_info();
            app.last_update = Instant::now();
        }

        // Auto-run tests every 30 seconds
        if app.last_test_run.elapsed() >= Duration::from_secs(30) && !app.is_running_tests {
            if let Err(e) = app.run_tests().await {
                app.add_log("ERROR", &format!("Failed to run tests: {}", e));
            }
        }

        // Add periodic logs when not running tests
        if last_log_time.elapsed() >= Duration::from_secs(10) && !app.is_running_tests {
            let messages = vec![
                "Dashboard monitoring active",
                "System resources normal",
                "Auto-test refresh enabled",
                "Ready for manual test run (press 't')",
            ];
            
            let message = messages[fastrand::usize(..messages.len())];
            app.add_log("INFO", message);
            last_log_time = Instant::now();
        }

        terminal.draw(|f| ui(f, &app))?;

        // Handle input
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                // Check for manual test run trigger
                let manual_test_trigger = key.code == KeyCode::Char('t') && !app.is_running_tests;
                
                app.on_key(key.code);
                
                // Run tests manually if triggered
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
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run app
    let res = run_app(terminal).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        io::stdout(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    io::stdout().flush()?;

    res
}
