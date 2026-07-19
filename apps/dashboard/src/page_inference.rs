/// Halaman 2 — Inference
/// Detail metrics inferensi: throughput, latency, KV cache, batching,
/// GPU utilization, dan distributed routing.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Wrap},
    Frame,
};

use crate::types::{format_memory, gauge_color, App};

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),   // throughput gauges
            Constraint::Min(10),     // detail panels
        ])
        .split(area);

    render_throughput_gauges(f, rows[0], app);
    render_detail_panels(f, rows[1], app);
}

fn render_throughput_gauges(f: &mut Frame, area: Rect, app: &App) {
    let m = &app.monitor.collector;
    let tok_sec    = m.tokens_per_sec.get() as f32;
    let cache_hit  = (m.cache_hit_ratio.get() * 100.0) as f32;
    let kv_pres    = (m.kv_cache_pressure.get() * 100.0) as f32;
    let batch_eff  = m.batching_efficiency_pct.get() as f32;

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    // Tokens/sec gauge (capped at 1000 for display)
    let tok_pct = (tok_sec / 1000.0 * 100.0).min(100.0) as u16;
    let tok_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" ⚡ Tokens/sec "))
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent(tok_pct)
        .label(format!("{:.1} tok/s", tok_sec));
    f.render_widget(tok_gauge, chunks[0]);

    // Cache hit ratio
    let cache_color = gauge_color(100.0 - cache_hit, 30.0, 60.0); // invert: low hit = bad
    let cache_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 💾 Cache Hit "))
        .gauge_style(Style::default().fg(cache_color))
        .percent((cache_hit as u16).min(100))
        .label(format!("{:.1}%", cache_hit));
    f.render_widget(cache_gauge, chunks[1]);

    // KV cache pressure (high = bad)
    let kv_color = gauge_color(kv_pres, 50.0, 80.0);
    let kv_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 🗜  KV Pressure "))
        .gauge_style(Style::default().fg(kv_color))
        .percent((kv_pres as u16).min(100))
        .label(format!("{:.1}%", kv_pres));
    f.render_widget(kv_gauge, chunks[2]);

    // Batching efficiency
    let batch_color = gauge_color(100.0 - batch_eff, 30.0, 60.0);
    let batch_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 📦 Batch Eff. "))
        .gauge_style(Style::default().fg(batch_color))
        .percent((batch_eff as u16).min(100))
        .label(format!("{:.1}%", batch_eff));
    f.render_widget(batch_gauge, chunks[3]);
}

fn render_detail_panels(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(area);

    render_request_stats(f, cols[0], app);
    render_kv_cache_detail(f, cols[1], app);
    render_gpu_inference(f, cols[2], app);
}

fn render_request_stats(f: &mut Frame, area: Rect, app: &App) {
    let m = &app.monitor.collector;
    let req_total   = m.request_counter.get();
    let req_fail    = m.request_failures.get();
    let throughput  = m.throughput_tokens.get();
    let queue       = m.queue_depth.get() as u64;
    let mem_frag    = m.memory_fragmentation.get() * 100.0;

    let success_rate = if req_total > 0.0 {
        ((req_total - req_fail) / req_total) * 100.0
    } else { 100.0 };

    let fail_color = if req_fail > 0.0 { Color::Red } else { Color::DarkGray };
    let sr_color   = if success_rate >= 99.0 { Color::Green }
                     else if success_rate >= 95.0 { Color::Yellow }
                     else { Color::Red };

    let lines = vec![
        Line::from(Span::styled("  REQUEST METRICS", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from(format!("  Total requests : {:.0}", req_total)),
        Line::from(vec![
            Span::raw("  Failures       : "),
            Span::styled(format!("{:.0}", req_fail), Style::default().fg(fail_color)),
        ]),
        Line::from(vec![
            Span::raw("  Success rate   : "),
            Span::styled(format!("{:.1}%", success_rate), Style::default().fg(sr_color)),
        ]),
        Line::from(format!("  Queue depth    : {}", queue)),
        Line::from(format!("  Throughput     : {:.0} tok/run", throughput)),
        Line::from(Span::raw("")),
        Line::from(Span::styled("  MEMORY", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from(format!("  Fragmentation  : {:.1}%", mem_frag)),
        Line::from(format!("  RAM used       : {}", format_memory(app.system_info.used_memory))),
        Line::from(format!("  RAM total      : {}", format_memory(app.system_info.total_memory))),
        Line::from(Span::raw("")),
        Line::from(Span::styled("  LATENCY", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from(Span::styled("  P50 / P95 : N/A (no server data)", Style::default().fg(Color::DarkGray))),
        Line::from(Span::styled("  Start server to enable live metrics", Style::default().fg(Color::DarkGray))),
    ];

    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" 📈 Requests "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, area);
}

fn render_kv_cache_detail(f: &mut Frame, area: Rect, app: &App) {
    let m = &app.monitor.collector;
    let cache_hit   = m.cache_hit_ratio.get() * 100.0;
    let kv_pres     = m.kv_cache_pressure.get() * 100.0;
    let int_frag    = m.kv_internal_frag_ratio.get() * 100.0;
    let ext_frag    = m.kv_external_frag_ratio.get() * 100.0;

    let hit_color = if cache_hit >= 80.0 { Color::Green }
                    else if cache_hit >= 50.0 { Color::Yellow }
                    else { Color::Red };

    let lines = vec![
        Line::from(Span::styled("  KV CACHE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::raw("  Hit ratio      : "),
            Span::styled(format!("{:.1}%", cache_hit), Style::default().fg(hit_color)),
        ]),
        Line::from(format!("  Pressure       : {:.1}%", kv_pres)),
        Line::from(format!("  Internal frag  : {:.1}%", int_frag)),
        Line::from(format!("  External frag  : {:.1}%", ext_frag)),
        Line::from(Span::raw("")),
        Line::from(Span::styled("  PAGED CACHE CONFIG", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from("  Block size     : 64 tokens"),
        Line::from("  Max blocks     : 65 536"),
        Line::from("  Watermark      : 70%"),
        Line::from("  Hot→Warm       : 15s"),
        Line::from("  Warm→Cold      : 60s"),
        Line::from(Span::raw("")),
        Line::from(Span::styled("  PREFIX SHARING", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from("  PrefixTrie DAG: enabled"),
        Line::from("  CoW on diverge: enabled"),
        Line::from("  Shared blocks : tracked via ref_count"),
    ];

    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" 🗄  KV Cache "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, area);
}

fn render_gpu_inference(f: &mut Frame, area: Rect, app: &App) {
    let m = &app.monitor.collector;
    let gpu_alive    = m.gpu_alive.get() as u8;
    let gpu_pct      = m.gpu_utilization_pct.get();
    let gpu_mem_pct  = m.gpu_memory_percent.get();
    let gpu_mem_bytes= m.gpu_memory_bytes.get() as u64;
    let gpu_tokens   = m.gpu_tokens.get();
    let cpu_tokens   = m.cpu_tokens.get();
    let fallbacks    = m.gpu_cpu_fallbacks.get();

    let fb_color = if fallbacks > 0.0 { Color::Yellow } else { Color::DarkGray };

    let mut lines = vec![
        Line::from(Span::styled("  GPU INFERENCE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
    ];

    if gpu_alive == 0 {
        lines.push(Line::from(Span::styled("  No GPU detected", Style::default().fg(Color::DarkGray))));
        lines.push(Line::from(Span::styled("  Running CPU-only mode", Style::default().fg(Color::DarkGray))));
    } else {
        lines.push(Line::from(vec![
            Span::raw("  GPU            : "),
            Span::styled(&app.monitor.gpu_name, Style::default().fg(Color::Green)),
        ]));
        lines.push(Line::from(format!("  Utilization    : {:.1}%", gpu_pct)));
        lines.push(Line::from(format!("  VRAM used      : {} ({:.1}%)", format_memory(gpu_mem_bytes), gpu_mem_pct)));
        lines.push(Line::from(format!("  Tokens (GPU)   : {:.0}", gpu_tokens)));
        lines.push(Line::from(format!("  Tokens (CPU)   : {:.0}", cpu_tokens)));
        lines.push(Line::from(vec![
            Span::raw("  CPU fallbacks  : "),
            Span::styled(format!("{:.0}", fallbacks), Style::default().fg(fb_color)),
        ]));
    }

    lines.push(Line::from(Span::raw("")));
    lines.push(Line::from(Span::styled("  BACKEND CHAIN", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
    lines.push(Line::from(Span::raw("")));
    lines.push(Line::from("  1. CUDA  (cuBLAS + NVRTC JIT)"));
    lines.push(Line::from("  2. wgpu  (WGSL compute)"));
    lines.push(Line::from("  3. CPU   (ndarray fallback)"));
    lines.push(Line::from(Span::raw("")));
    lines.push(Line::from(Span::styled("  CUDA OPS AVAILABLE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
    lines.push(Line::from(Span::raw("")));
    lines.push(Line::from("  matmul · add · sub · mul · div"));
    lines.push(Line::from("  relu · gelu · silu · sigmoid · tanh"));
    lines.push(Line::from("  softmax · transpose · FlashAttention"));

    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" 🎮 GPU "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, area);
}
