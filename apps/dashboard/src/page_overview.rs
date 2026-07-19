/// Halaman 1 — Overview
/// Menampilkan ringkasan sistem: CPU, Memori, GPU, Health, info proses,
/// dan ringkasan singkat semua subsistem Nexora.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::types::{gauge_color, format_memory, App, CPU_WARN, CPU_CRIT, MEM_WARN, MEM_CRIT};
use nexora_monitoring::HealthStatus;

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // gauge row
            Constraint::Length(5),  // gauge row 2
            Constraint::Min(8),     // subsystem + health
        ])
        .split(area);

    render_gauges_row1(f, rows[0], app);
    render_gauges_row2(f, rows[1], app);
    render_bottom(f, rows[2], app);
}

fn render_gauges_row1(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // CPU gauge
    let cpu_color = gauge_color(app.system_info.cpu_usage, CPU_WARN, CPU_CRIT);
    let cpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" ⚙  CPU Usage "))
        .gauge_style(Style::default().fg(cpu_color))
        .percent((app.system_info.cpu_usage as u16).min(100))
        .label(format!("{:.1}%  ({} cores)", app.system_info.cpu_usage, {
            app.sys.cpus().len()
        }));
    f.render_widget(cpu_gauge, chunks[0]);

    // Memory gauge
    let mem_color = gauge_color(app.system_info.memory_usage, MEM_WARN, MEM_CRIT);
    let mem_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 🧠 Memory Usage "))
        .gauge_style(Style::default().fg(mem_color))
        .percent((app.system_info.memory_usage as u16).min(100))
        .label(format!(
            "{} / {}",
            format_memory(app.system_info.used_memory),
            format_memory(app.system_info.total_memory)
        ));
    f.render_widget(mem_gauge, chunks[1]);
}

fn render_gauges_row2(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(area);

    // GPU gauge
    let m = &app.monitor.collector;
    let gpu_alive = m.gpu_alive.get() as u8;
    let gpu_pct   = m.gpu_utilization_pct.get() as u16;
    let gpu_mem   = m.gpu_memory_percent.get();
    let gpu_bytes = m.gpu_memory_bytes.get() as u64;

    let gpu_title = if app.monitor.gpu_name.is_empty() {
        " 🎮 GPU ".to_string()
    } else {
        format!(" 🎮 GPU — {} ", app.monitor.gpu_name)
    };
    let gpu_label = if gpu_alive == 0 {
        "No GPU detected".to_string()
    } else {
        format!("util:{gpu_pct}%  mem:{:.0}%  {}", gpu_mem, format_memory(gpu_bytes))
    };
    let gpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(gpu_title))
        .gauge_style(Style::default().fg(
            if gpu_alive == 0 { Color::DarkGray } else { Color::Magenta }
        ))
        .percent(gpu_pct.min(100))
        .label(gpu_label);
    f.render_widget(gpu_gauge, chunks[0]);

    // Health widget
    let report = app.monitor.checker.check_health();
    let (hcolor, hicon) = match &report.status {
        HealthStatus::Healthy   => (Color::Green,  "✓ HEALTHY"),
        HealthStatus::Degraded  => (Color::Yellow, "⚠ DEGRADED"),
        HealthStatus::Unhealthy => (Color::Red,    "✗ UNHEALTHY"),
    };
    let health_lines = if app.show_health_detail {
        let detail: Vec<String> = report.checks.iter().map(|c| {
            format!("  {} {}  {}ms", if c.healthy { "✓" } else { "✗" }, c.name, c.latency_ms)
        }).collect();
        let mut lines = vec![
            Line::from(Span::styled(hicon, Style::default().fg(hcolor).add_modifier(Modifier::BOLD))),
            Line::from(format!("  Uptime: {}s  | {} checks", report.uptime_seconds, report.checks.len())),
        ];
        for d in detail { lines.push(Line::from(d)); }
        lines
    } else {
        vec![
            Line::from(Span::styled(hicon, Style::default().fg(hcolor).add_modifier(Modifier::BOLD))),
            Line::from(format!("  {} checks  uptime {}s", report.checks.len(), report.uptime_seconds)),
            Line::from(Span::styled("  Press 'h' to expand", Style::default().fg(Color::DarkGray))),
        ]
    };
    let health_block = Paragraph::new(health_lines)
        .block(Block::default().borders(Borders::ALL).title(" 💓 Health "))
        .style(Style::default().fg(hcolor));
    f.render_widget(health_block, chunks[1]);

    // System info
    let sys_lines = vec![
        Line::from(format!("  Processes : {}", app.system_info.processes)),
        Line::from(format!("  Sys Uptime: {}", app.system_info.uptime)),
        Line::from(format!("  Dashboard : {}s", app.started.elapsed().as_secs())),
    ];
    let sys_block = Paragraph::new(sys_lines)
        .block(Block::default().borders(Borders::ALL).title(" 🖥  System "));
    f.render_widget(sys_block, chunks[2]);
}

fn render_bottom(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    render_subsystem_status(f, chunks[0], app);
    render_quick_stats(f, chunks[1], app);
}

fn render_subsystem_status(f: &mut Frame, area: Rect, _app: &App) {
    let items: Vec<ListItem> = vec![
        ("Foundation", "NXR CausalLM backbone — 10 models ready",    Color::Green),
        ("Inference",  "KV Cache + Paged Blocks + Beam Search",        Color::Green),
        ("MoE FFN",    "32 experts, top-2 gating (CUDA/wgpu/CPU)",     Color::Green),
        ("SACA",       "6-phase reasoning pipeline (CoT → Rerank)",    Color::Green),
        ("CAFFEINE",   "Multimodal fusion: image/audio/video/text",    Color::Green),
        ("ORACLE",     "Code verifiers: security/perf/correctness",    Color::Green),
        ("ATQS",       "Adaptive quantization + AWQ calibration",      Color::Green),
        ("Distributed","Gossip cluster + 5-strategy router",           Color::Green),
        ("Agent Sys",  "Planner-Worker hierarchy + plan dispatch",     Color::Green),
        ("Distillation","KD teacher→student (Swift/Aether/Omnis Lite)",Color::Yellow),
    ].into_iter().map(|(name, desc, color)| {
        ListItem::new(Line::from(vec![
            Span::styled(format!("  ● {:<14}", name), Style::default().fg(color).add_modifier(Modifier::BOLD)),
            Span::raw(desc),
        ]))
    }).collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" 🔧 Subsystem Status "));
    f.render_widget(list, area);
}

fn render_quick_stats(f: &mut Frame, area: Rect, app: &App) {
    let m = &app.monitor.collector;
    let req_total   = m.request_counter.get();
    let req_fail    = m.request_failures.get();
    let tok_sec     = m.tokens_per_sec.get();
    let cache_hit   = m.cache_hit_ratio.get() * 100.0;
    let train_loss  = m.training_loss.get();
    let train_lr    = m.training_learning_rate.get();
    let queue       = m.queue_depth.get();
    let gpu_fallback= m.gpu_cpu_fallbacks.get();

    let passed = app.test_results.iter().filter(|t| t.status.starts_with('✓')).count();
    let failed = app.test_results.iter().filter(|t| t.status.starts_with('✗')).count();
    let total  = app.test_results.len();

    let lines = vec![
        Line::from(Span::styled("  QUICK STATS", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(format!("  Requests  : {:.0} total  ({:.0} fail)", req_total, req_fail)),
        Line::from(format!("  Tokens/sec: {:.1}", tok_sec)),
        Line::from(format!("  Cache hit : {:.1}%", cache_hit)),
        Line::from(format!("  Queue     : {:.0}", queue)),
        Line::from(format!("  GPU falls : {:.0}", gpu_fallback)),
        Line::from(Span::raw("")),
        Line::from(Span::styled("  TRAINING", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        if train_loss > 0.0 {
            Line::from(format!("  Loss: {:.4}  LR: {:.2e}", train_loss, train_lr))
        } else {
            Line::from(Span::styled("  No active training session", Style::default().fg(Color::DarkGray)))
        },
        Line::from(Span::raw("")),
        Line::from(Span::styled("  TESTS", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        if total == 0 {
            Line::from(Span::styled("  No test results yet — press 't'", Style::default().fg(Color::DarkGray)))
        } else {
            Line::from(vec![
                Span::styled(format!("  ✓ {passed} passed  "), Style::default().fg(Color::Green)),
                Span::styled(format!("✗ {failed} failed  "), Style::default().fg(if failed > 0 { Color::Red } else { Color::DarkGray })),
                Span::raw(format!("({total} total)")),
            ])
        },
    ];

    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" 📊 Quick Stats "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, area);
}
