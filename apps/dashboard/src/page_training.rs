/// Halaman 4 — Training
/// Metrics training aktif, konfigurasi checkpoint, pipeline data,
/// info distillation, dan cara pakai CLI training.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::types::App;

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // live metric gauges
            Constraint::Min(10),    // detail panels
        ])
        .split(area);

    render_live_gauges(f, rows[0], app);
    render_detail_panels(f, rows[1], app);
}

fn render_live_gauges(f: &mut Frame, area: Rect, app: &App) {
    let m = &app.monitor.collector;
    let loss      = m.training_loss.get() as f32;
    let lr        = m.training_learning_rate.get() as f32;
    let grad_norm = m.training_grad_norm.get() as f32;

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(area);

    // Loss gauge (0-10 capped)
    let loss_pct   = ((loss / 10.0) * 100.0).min(100.0) as u16;
    let loss_color = if loss == 0.0 { Color::DarkGray }
                     else if loss < 1.0 { Color::Green }
                     else if loss < 3.0 { Color::Yellow }
                     else { Color::Red };
    let loss_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 📉 Training Loss "))
        .gauge_style(Style::default().fg(loss_color))
        .percent(loss_pct)
        .label(if loss == 0.0 { "no session".to_string() } else { format!("{:.4}", loss) });
    f.render_widget(loss_gauge, chunks[0]);

    // LR gauge (0 - 1e-3 range, scaled)
    let lr_pct     = ((lr / 0.001) * 100.0).min(100.0) as u16;
    let lr_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 📐 Learning Rate "))
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent(lr_pct)
        .label(if lr == 0.0 { "n/a".to_string() } else { format!("{:.2e}", lr) });
    f.render_widget(lr_gauge, chunks[1]);

    // Grad norm gauge (0-5 capped)
    let gn_pct   = ((grad_norm / 5.0) * 100.0).min(100.0) as u16;
    let gn_color = if grad_norm == 0.0 { Color::DarkGray }
                   else if grad_norm <= 1.0 { Color::Green }
                   else if grad_norm <= 2.0 { Color::Yellow }
                   else { Color::Red };
    let gn_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 📏 Grad Norm "))
        .gauge_style(Style::default().fg(gn_color))
        .percent(gn_pct)
        .label(if grad_norm == 0.0 { "n/a".to_string() } else { format!("{:.4}", grad_norm) });
    f.render_widget(gn_gauge, chunks[2]);
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

    render_trainer_config(f, cols[0], app);
    render_checkpoint_info(f, cols[1]);
    render_distillation_info(f, cols[2]);
}

fn render_trainer_config(f: &mut Frame, area: Rect, app: &App) {
    let m = &app.monitor.collector;
    let loss     = m.training_loss.get();
    let lr       = m.training_learning_rate.get();
    let grad     = m.training_grad_norm.get();
    let active   = loss > 0.0 || lr > 0.0;

    let status_line = if active {
        Line::from(vec![
            Span::raw("  Status  : "),
            Span::styled("● ACTIVE", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ])
    } else {
        Line::from(vec![
            Span::raw("  Status  : "),
            Span::styled("○ IDLE", Style::default().fg(Color::DarkGray)),
        ])
    };

    let lines = vec![
        Line::from(Span::styled("  LIVE METRICS", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        status_line,
        Line::from(format!("  Loss     : {:.4}", loss)),
        Line::from(format!("  LR       : {:.2e}", lr)),
        Line::from(format!("  Grad norm: {:.4}", grad)),
        Line::from(format!("  Uptime   : {}s", app.started.elapsed().as_secs())),
        Line::from(Span::raw("")),
        Line::from(Span::styled("  TRAINER FEATURES", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from("  ✓ Gradient accumulation (batch_size)"),
        Line::from("  ✓ LR warmup + cosine decay"),
        Line::from("  ✓ AdamW — decoupled weight decay"),
        Line::from("  ✓ Gradient clipping (max_grad_norm)"),
        Line::from("  ✓ Per-epoch data shuffling"),
        Line::from("  ✓ Graceful shutdown (Ctrl+C)"),
        Line::from("  ✓ BLAS acceleration (optional)"),
        Line::from("  ✓ HuggingFace dataset live fetch"),
        Line::from(Span::raw("")),
        Line::from(Span::styled("  DATASTREAM FILTERS", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from("  Length filter · Quality filter"),
        Line::from("  Dedup · Toxicity · Prompt-injection"),
    ];

    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" ⚙  Trainer "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, area);
}

fn render_checkpoint_info(f: &mut Frame, area: Rect) {
    let items: Vec<ListItem> = vec![
        ("Small ckpt",  "every 100 steps", "{base}.step-{N}.safetensors",     Color::Green),
        ("Big ckpt",    "every 1000 steps", "{base}.big-{N}.safetensors",     Color::Yellow),
        ("Final ckpt",  "end of training", "{base}.final.safetensors",         Color::Cyan),
    ].into_iter().map(|(kind, freq, pattern, color)| {
        ListItem::new(vec![
            Line::from(Span::styled(
                format!("  {} — {}", kind, freq),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )),
            Line::from(format!("    {}", pattern)),
        ])
    }).collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" 💾 Checkpoints "));
    f.render_widget(list, area);
}

fn render_distillation_info(f: &mut Frame, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Distillation students
    let student_items: Vec<ListItem> = vec![
        ("Swift Lite",  "swift",  128, 3, "~14M",  "443×", Color::Green),
        ("Aether Lite", "aether", 128, 3, "~14M",  "443×", Color::Yellow),
        ("Omnis Lite",  "omnis",  512, 16, "~162M", "38×", Color::Red),
    ].into_iter().map(|(name, model, hidden, layers, params, comp, color)| {
        ListItem::new(Line::from(vec![
            Span::styled(format!("  {:<12}", name), Style::default().fg(color).add_modifier(Modifier::BOLD)),
            Span::raw(format!("h={} l={}  {}  comp {}", hidden, layers, params, comp)),
        ]))
    }).collect();

    let student_list = List::new(student_items)
        .block(Block::default().borders(Borders::ALL).title(" 🎓 KD Students "));
    f.render_widget(student_list, rows[0]);

    // Distillation config
    let lines = vec![
        Line::from(Span::styled("  KD FORMULA", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from("  L = α·T²·KL(teacher||student)"),
        Line::from("      + (1-α)·CE(student, targets)"),
        Line::from(Span::raw("")),
        Line::from("  α   = 0.7  (KD weight)"),
        Line::from("  T   = 4.0  (temperature)"),
        Line::from(Span::raw("")),
        Line::from(Span::styled("  CLI", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from(Span::styled(
            "  nexora distill --student swift-lite",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            "    --hf-dataset wikitext --steps 500",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" 📖 Distillation "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, rows[1]);
}
