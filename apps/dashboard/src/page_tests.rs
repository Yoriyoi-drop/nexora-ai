/// Halaman 5 — Tests
/// Test runner terintegrasi: list hasil, detail selected test,
/// dan summary statistik. Tekan 't' untuk jalankan.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::types::App;

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(area);

    render_test_list(f, cols[0], app);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(cols[1]);

    render_test_detail(f, right[0], app);
    render_test_summary(f, right[1], app);
}

fn render_test_list(f: &mut Frame, area: Rect, app: &App) {
    let title = if app.is_running_tests {
        " 🔄 Tests (running…) — wait "
    } else if app.test_results.is_empty() {
        " 🧪 Tests — press 't' to run "
    } else {
        " 🧪 Tests "
    };

    let items: Vec<ListItem> = app.test_results.iter().enumerate().map(|(i, t)| {
        let is_sel = i == app.selected_test;
        let (icon, color) = if t.status.starts_with('✓') {
            ("✓", Color::Green)
        } else if t.status.starts_with('✗') {
            ("✗", Color::Red)
        } else if t.status.starts_with('⚠') {
            ("⚠", Color::Yellow)
        } else {
            ("?", Color::DarkGray)
        };

        let style = if is_sel {
            Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(color)
        };

        // Shorten long test names for display
        let display_name = if t.name.len() > 48 {
            format!("…{}", &t.name[t.name.len()-47..])
        } else {
            t.name.clone()
        };

        ListItem::new(format!(" {} {}  ({})", icon, display_name, t.duration))
            .style(style)
    }).collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD));
    f.render_widget(list, area);
}

fn render_test_detail(f: &mut Frame, area: Rect, app: &App) {
    let lines = if let Some(test) = app.test_results.get(app.selected_test) {
        let (status_color, status_icon) = if test.status.starts_with('✓') {
            (Color::Green,  "PASSED ✓")
        } else if test.status.starts_with('✗') {
            (Color::Red,    "FAILED ✗")
        } else {
            (Color::Yellow, "SKIPPED ⚠")
        };

        let mut lines = vec![
            Line::from(Span::styled("  TEST DETAIL", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::styled("  Name     : ", Style::default().fg(Color::White)),
                Span::raw(&test.name),
            ]),
            Line::from(vec![
                Span::styled("  Status   : ", Style::default().fg(Color::White)),
                Span::styled(status_icon, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("  Duration : ", Style::default().fg(Color::White)),
                Span::raw(&test.duration),
            ]),
        ];

        if let Some(err) = &test.error {
            lines.push(Line::from(Span::raw("")));
            lines.push(Line::from(Span::styled("  ERROR OUTPUT:", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))));
            for (i, l) in err.lines().take(12).enumerate() {
                let prefix = if i == 0 { "  " } else { "    " };
                lines.push(Line::from(Span::styled(
                    format!("{}{}", prefix, l),
                    Style::default().fg(Color::LightRed),
                )));
            }
            if err.lines().count() > 12 {
                lines.push(Line::from(Span::styled(
                    "  … (truncated)",
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }

        lines
    } else {
        vec![
            Line::from(Span::raw("")),
            Line::from(Span::styled(
                "  No test selected.",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "  ↑/↓ to select a test from the list.",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "  Press 't' to run all tests.",
                Style::default().fg(Color::DarkGray),
            )),
        ]
    };

    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" 🔍 Selected Test "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, area);
}

fn render_test_summary(f: &mut Frame, area: Rect, app: &App) {
    let total   = app.test_results.len();
    let passed  = app.test_results.iter().filter(|t| t.status.starts_with('✓')).count();
    let failed  = app.test_results.iter().filter(|t| t.status.starts_with('✗')).count();
    let skipped = app.test_results.iter().filter(|t| t.status.starts_with('⚠')).count();
    let pass_rate = if total > 0 { (passed as f64 / total as f64) * 100.0 } else { 0.0 };

    let rate_color = if pass_rate >= 99.0 { Color::Green }
                     else if pass_rate >= 90.0 { Color::Yellow }
                     else { Color::Red };

    let lines = vec![
        Line::from(Span::styled("  SUMMARY", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::raw("  Total   : "),
            Span::styled(format!("{total}"), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("  Passed  : "),
            Span::styled(format!("{passed}"), Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("  Failed  : "),
            Span::styled(format!("{failed}"), Style::default().fg(if failed > 0 { Color::Red } else { Color::DarkGray })),
        ]),
        Line::from(vec![
            Span::raw("  Skipped : "),
            Span::styled(format!("{skipped}"), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("  Pass %  : "),
            Span::styled(format!("{:.1}%", pass_rate), Style::default().fg(rate_color).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(Span::raw("")),
        Line::from(Span::styled("  KEYBINDS", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from("  t       Run all tests"),
        Line::from("  ↑ / ↓  Navigate results"),
        Line::from("  r       Refresh system info"),
        Line::from("  h       Toggle health detail"),
    ];

    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" 📊 Stats "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, area);
}
