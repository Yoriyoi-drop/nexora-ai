/// Halaman 3 — Models
/// Info lengkap 10 model NXR series: tier, arsitektur, spesialisasi,
/// subsystem yang di-wire, dan pipeline delegation masing-masing.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::types::App;

struct ModelInfo {
    name:       &'static str,
    tier:       &'static str,
    hidden:     u16,
    heads:      u16,
    layers:     u16,
    specialty:  &'static str,
    classifier: &'static str,
    wired:      &'static str,
    tier_color: Color,
}

const MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "Omnis",  tier: "Ultra", hidden: 768, heads: 12, layers: 8,
        specialty:  "Expert routing — 7 domain (math/science/code/creative/reasoning/factual/general)",
        classifier: "2-layer MLP → 7 domains (MoE top-2 gating, Xavier init)",
        wired:      "nexora_has_moe_ffn::Router  (real learned gating)",
        tier_color: Color::Red,
    },
    ModelInfo {
        name: "Axiom",  tier: "Ultra", hidden: 768, heads: 12, layers: 8,
        specialty:  "Structured reasoning — deductive/inductive/abductive/analogical/causal",
        classifier: "2-layer MLP → 6 reasoning types",
        wired:      "SacaEngine::reason()  — full 6-phase CoT pipeline",
        tier_color: Color::Red,
    },
    ModelInfo {
        name: "Genesis",tier: "Ultra", hidden: 768, heads: 12, layers: 8,
        specialty:  "Iterative self-refinement — quality loop (clarity/depth/accuracy/structure)",
        classifier: "2-layer MLP → 6 quality dims, threshold 0.6, max 3 iterations",
        wired:      "SacaEngine::reason() + quality classifier feedback",
        tier_color: Color::Red,
    },
    ModelInfo {
        name: "Vortex", tier: "Apex",  hidden: 512, heads: 8,  layers: 6,
        specialty:  "Code review — bugs/security/performance/style/architecture",
        classifier: "2-layer MLP → 6 categories + language detection (8 langs)",
        wired:      "CodeVerifierManager::verify_detailed()  — 4 rule-based verifiers",
        tier_color: Color::Yellow,
    },
    ModelInfo {
        name: "Aether", tier: "Apex",  hidden: 512, heads: 8,  layers: 6,
        specialty:  "Emotion understanding — joy/sadness/anger/fear/surprise/disgust/trust/neutral",
        classifier: "2-layer MLP → 8 emotions, average pooled token embeddings",
        wired:      "CaffeineProcessor::process_multimodal()  — text pipeline",
        tier_color: Color::Yellow,
    },
    ModelInfo {
        name: "Nexum",  tier: "Apex",  hidden: 512, heads: 8,  layers: 6,
        specialty:  "Task decomposition — simple/moderate/complex/multi_domain",
        classifier: "2-layer MLP → 4 complexity levels",
        wired:      "SacaEngine + CodeVerifierManager quality scoring",
        tier_color: Color::Yellow,
    },
    ModelInfo {
        name: "Spectra", tier: "Pro",  hidden: 384, heads: 6,  layers: 4,
        specialty:  "Creative generation — narrative/poetic/persuasive/technical/dialogue/descriptive",
        classifier: "2-layer MLP → 6 styles, per-style temperature adjustment",
        wired:      "CaffeineProcessor::process_multimodal()  — vision/audio/fusion",
        tier_color: Color::Magenta,
    },
    ModelInfo {
        name: "Cipher", tier: "Pro",   hidden: 384, heads: 6,  layers: 4,
        specialty:  "Security analysis — injection/xss/auth/crypto/config/network",
        classifier: "2-layer MLP → 6 threat categories",
        wired:      "CodeVerifierManager::verify_detailed()  — security verifier",
        tier_color: Color::Magenta,
    },
    ModelInfo {
        name: "Kronos", tier: "Core",  hidden: 256, heads: 4,  layers: 3,
        specialty:  "Temporal reasoning — urgent/scheduled/historical/realtime/evergreen",
        classifier: "2-layer MLP → 5 temporal modes + chrono::Utc::now() injection",
        wired:      "SacaEngine::reason()  — temporal context phases 1-4+6",
        tier_color: Color::Cyan,
    },
    ModelInfo {
        name: "Swift",  tier: "Edge",  hidden: 128, heads: 4,  layers: 2,
        specialty:  "Fast routing — qa/summarize/translate/generate/analyze",
        classifier: "2-layer MLP → 5 task types, adjusts max_tokens + temperature",
        wired:      "nexora_has_moe_ffn::Router  — latency-aware dispatch",
        tier_color: Color::Green,
    },
];

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(area);

    render_model_list(f, cols[0], app);
    render_model_detail(f, cols[1]);
}

fn render_model_list(f: &mut Frame, area: Rect, _app: &App) {
    let items: Vec<ListItem> = MODELS.iter().map(|m| {
        ListItem::new(Line::from(vec![
            Span::styled(
                format!("  {:<8}", m.name),
                Style::default().fg(m.tier_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("[{:<5}]", m.tier),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(format!(" h={} l={}", m.hidden, m.layers)),
        ]))
    }).collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" 🤖 NXR Series "));
    f.render_widget(list, area);
}

fn render_model_detail(f: &mut Frame, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    render_model_table(f, rows[0]);
    render_backbone_info(f, rows[1]);
}

fn render_model_table(f: &mut Frame, area: Rect) {
    let mut lines = vec![
        Line::from(vec![
            Span::styled(
                format!("  {:<8} {:<6} {:<7} {:<6} {:<6}  {}", "Model", "Tier", "Hidden", "Heads", "Layers", "Specialty"),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::styled(
            format!("  {}", "─".repeat(82)),
            Style::default().fg(Color::DarkGray),
        )),
    ];

    for m in MODELS {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<8}", m.name),
                Style::default().fg(m.tier_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{:<6}", m.tier),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(format!("{:<7}", m.hidden)),
            Span::raw(format!("{:<6}", m.heads)),
            Span::raw(format!("{:<6}", m.layers)),
            Span::raw(format!("  {}", shorten(m.specialty, 50))),
        ]));
    }

    lines.push(Line::from(Span::raw("")));
    lines.push(Line::from(Span::styled(
        "  All models use CausalLM backbone — defined via define_foundation_model!() macro",
        Style::default().fg(Color::DarkGray),
    )));

    let block = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" 📋 Model Table "))
        .wrap(Wrap { trim: false });
    f.render_widget(block, area);
}

fn render_backbone_info(f: &mut Frame, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Wiring info
    let mut wiring_lines = vec![
        Line::from(Span::styled("  PHASE 4 WIRING", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
    ];
    for m in MODELS {
        wiring_lines.push(Line::from(vec![
            Span::styled(format!("  {:<8}", m.name), Style::default().fg(m.tier_color)),
            Span::raw(shorten(m.wired, 45).to_string()),
        ]));
    }

    let wiring_block = Paragraph::new(wiring_lines)
        .block(Block::default().borders(Borders::ALL).title(" 🔌 Subsystem Wiring "))
        .wrap(Wrap { trim: false });
    f.render_widget(wiring_block, cols[0]);

    // Classifier info
    let mut cls_lines = vec![
        Line::from(Span::styled("  CLASSIFIERS (MLP)", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw("")),
    ];
    for m in MODELS {
        cls_lines.push(Line::from(vec![
            Span::styled(format!("  {:<8}", m.name), Style::default().fg(m.tier_color)),
            Span::raw(shorten(m.classifier, 45).to_string()),
        ]));
    }
    cls_lines.push(Line::from(Span::raw("")));
    cls_lines.push(Line::from(Span::styled(
        "  All classifiers: Xavier init weights, OnceLock singleton",
        Style::default().fg(Color::DarkGray),
    )));

    let cls_block = Paragraph::new(cls_lines)
        .block(Block::default().borders(Borders::ALL).title(" 🧬 Classifiers "))
        .wrap(Wrap { trim: false });
    f.render_widget(cls_block, cols[1]);
}

fn shorten(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..max] }
}

// Expose model info for potential future use
pub fn model_count() -> usize { MODELS.len() }

pub fn models_by_tier(tier: &str) -> Vec<&'static str> {
    MODELS.iter()
        .filter(|m| m.tier == tier)
        .map(|m| m.name)
        .collect()
}
