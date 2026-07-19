/// Halaman 6 — Logs
/// Full log viewer dengan scroll, filter level, dan statistik log.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::types::{App, LOG_CAPACITY};

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(10), Constraint::Length(28)])
        .split(area);

    render_log_list(f, cols[0], app);
    render_log_sidebar(f, cols[1], app);
}

fn render_log_list(f: &mut Frame, area: Rect, app: &App) {
    // How many lines fit inside the bordered box
    let visible = (area.height as usize).saturating_sub(2);
    let total   = app.logs.len();
    // scroll_offset: 0 = bottom (newest), increase to scroll up
    let max_scroll = total.saturating_sub(visible);
    let scroll      = app.log_scroll.min(max_scroll);
    let start       = total.saturating_sub(visible + scroll);
    let end         = total.saturating_sub(scroll);

    let items: Vec<ListItem> = app.logs[start..end]
        .iter()
        .map(|log| {
            let (style, prefix) = match log.level.as_str() {
                "CRIT"  => (Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),   "CRIT "),
                "ERROR" => (Style::default().fg(Color::Red),                                "ERR  "),
                "WARN"  => (Style::default().fg(Color::Yellow),                             "WARN "),
                "INFO"  => (Style::default().fg(Color::Blue),                               "INFO "),
                _       => (Style::default().fg(Color::White),                              "DBG  "),
            };
            let content = format!("[{}] {} {}", log.timestamp, prefix, log.message);
            ListItem::new(content).style(style)
        })
        .collect();

    let scroll_hint = if scroll > 0 {
        format!(" Logs [↑{} more above | PgUp/PgDn/End scroll] ", scroll)
    } else {
        format!(" Logs [{}/{}] (PgUp to scroll up) ", total.min(visible), total)
    };

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(scroll_hint));
    f.render_widget(list, area);
}

fn render_log_sidebar(f: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    render_log_stats(f, rows[0], app);
    render_log_help(f, rows[1]);
}

fn render_log_stats(f: &mut Frame, area: Rect, app: &App) {
    let total   = app.logs.len();
    let crits   = app.logs.iter().filter(|l| l.level == "CRIT").count();
    let errors  = app.logs.iter().filter(|l| l.level == "ERROR").count();
    let warns   = app.logs.iter().filter(|l| l.level == "WARN").count();
    let infos   = app.logs.iter().filter(|l| l.level == "INFO").count();
    let others  = total - crits - errors - warns - infos;

    let newest_ts = app.logs.last().map(|l| l.timestamp.as_str()).unwrap_or("—");
    let oldest_ts = app.logs.first().map(|l| l.timestamp.as_str()).unwrap_or("—");

    let lines = vec![
        Line::from(Span::styled(" STATISTICS", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::raw(" Total : "),
            Span::styled(format!("{}/{}", total, LOG_CAPACITY), Style::default().fg(Color::White)),
        ]),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled(format!(" CRIT  : {}", crits), Style::default().fg(
                if crits > 0 { Color::Red } else { Color::DarkGray }
            ).add_modifier(if crits > 0 { Modifier::BOLD } else { Modifier::empty() })),
        ]),
        Line::from(vec![
            Span::styled(format!(" ERROR : {}", errors), Style::default().fg(
                if errors > 0 { Color::Red } else { Color::DarkGray }
            )),
        ]),
        Line::from(vec![
            Span::styled(format!(" WARN  : {}", warns), Style::default().fg(
                if warns > 0 { Color::Yellow } else { Color::DarkGray }
            )),
        ]),
        Line::from(vec![
            Span::styled(format!(" INFO  : {}", infos), Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::styled(format!(" OTHER : {}", others), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(Span::raw("")),
        Line::from(Span::styled(" RANGE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from(format!(" Oldest: {}", oldest_ts)),
        Line::from(format!(" Newest: {}", newest_ts)),
    ];

    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" 📋 Log Stats "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, area);
}

fn render_log_help(f: &mut Frame, area: Rect) {
    let lines = vec![
        Line::from(Span::styled(" NAVIGATION", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from(" PgUp   Scroll up"),
        Line::from(" PgDn   Scroll down"),
        Line::from(" End    Jump to bottom"),
        Line::from(" Home   Jump to top"),
        Line::from(Span::raw("")),
        Line::from(Span::styled(" LOG LEVELS", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled(" ● CRIT ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::raw("Critical alert"),
        ]),
        Line::from(vec![
            Span::styled(" ● ERR  ", Style::default().fg(Color::Red)),
            Span::raw("Error event"),
        ]),
        Line::from(vec![
            Span::styled(" ● WARN ", Style::default().fg(Color::Yellow)),
            Span::raw("Warning"),
        ]),
        Line::from(vec![
            Span::styled(" ● INFO ", Style::default().fg(Color::Blue)),
            Span::raw("Informational"),
        ]),
    ];

    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" ⌨  Help "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, area);
}
