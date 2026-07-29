//! Mesin eksekusi untuk CAFFEINE
//!
//! Mengeksekusi tindakan yang direncanakan dan menangani hasilnya

use crate::multimodal::error::Result;
use crate::multimodal::types::*;
use async_trait::async_trait;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;
use tokio::process::Command;
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};

static RE_URL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s"'<>(){}|\\^`[\]]+"#).expect("valid URL regex"));
static RE_EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"#).expect("valid email regex")
});
static RE_PHONE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"\+?[\d\-\(\)\s]{7,20}"#).expect("valid phone regex"));
static RE_URL_SIMPLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s]+"#).expect("valid URL simple regex"));

/// Validates that a string is a safe URL for xdg-open
fn is_safe_url(destination: &str) -> bool {
    if destination.is_empty() || destination.len() > 2048 {
        return false;
    }
    // Only allow http, https, mailto, and file protocols
    let lower = destination.to_lowercase();
    if lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("mailto:")
        || lower.starts_with("file://")
    {
        !destination.contains(';')
            && !destination.contains('|')
            && !destination.contains('`')
            && !destination.contains('$')
            && !destination.contains('>')
            && !destination.contains('<')
            && !destination.contains('&')
    } else {
        false
    }
}

/// Sanitize text for xdotool type - block control characters and shell metacharacters
fn sanitize_xdotool_text(text: &str) -> String {
    text.chars()
        .filter(|&c| {
            // Allow printable ASCII and Unicode, block control chars except \n, \t
            c == '\n' || c == '\t' || (c >= ' ' && c != '\x7f')
        })
        .collect()
}

/// Mesin eksekusi untuk menjalankan tindakan
pub struct ExecutionEngine {
    _action_config: crate::multimodal::config::ActionConfig,
    action_handlers: HashMap<ActionType, Box<dyn ActionHandler>>,
    execution_history: Vec<ExecutionRecord>,
}

impl ExecutionEngine {
    /// Membuat mesin eksekusi baru dengan konfigurasi yang diberikan
    pub fn new(config: crate::multimodal::config::ActionConfig) -> Result<Self> {
        let mut action_handlers: HashMap<ActionType, Box<dyn ActionHandler>> = HashMap::new();

        action_handlers.insert(ActionType::Click, Box::new(ClickHandler::new()));
        action_handlers.insert(ActionType::Type, Box::new(TypeHandler::new()));
        action_handlers.insert(ActionType::Scroll, Box::new(ScrollHandler::new()));
        action_handlers.insert(ActionType::Drag, Box::new(DragHandler::new()));
        action_handlers.insert(ActionType::Wait, Box::new(WaitHandler::new()));
        action_handlers.insert(ActionType::Navigate, Box::new(NavigateHandler::new()));
        action_handlers.insert(ActionType::Extract, Box::new(ExtractHandler::new()));
        action_handlers.insert(ActionType::Analyze, Box::new(AnalyzeHandler::new()));

        Ok(Self {
            _action_config: config,
            action_handlers,
            execution_history: Vec::new(),
        })
    }

    /// Mengeksekusi satu tindakan dan mencatat hasilnya
    pub async fn execute(&mut self, action: &Action) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();

        if let Some(handler) = self.action_handlers.get(&action.action_type) {
            let result = handler.execute(action).await?;

            let execution_time = start_time.elapsed().as_millis() as f32;
            let record = ExecutionRecord {
                action: action.clone(),
                result,
                execution_time_ms: execution_time,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|e| {
                        crate::multimodal::error::CaffeineError::output_generation(&format!(
                            "Gagal mendapatkan timestamp: {}",
                            e
                        ))
                    })?
                    .as_secs_f32(),
            };

            self.execution_history.push(record);
            info!(
                "Tindakan {:?} selesai dalam {:.2}ms",
                action.action_type, execution_time
            );

            Ok(result)
        } else {
            Err(crate::multimodal::error::CaffeineError::action_head(
                &format!(
                    "Tidak ada handler ditemukan untuk tipe tindakan: {:?}",
                    action.action_type
                ),
            ))
        }
    }

    /// Mengeksekusi kumpulan tindakan secara berurutan
    pub async fn execute_batch(&mut self, actions: &[Action]) -> Result<Vec<ExecutionResult>> {
        let mut results = Vec::new();

        for action in actions {
            let result = self.execute(action).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Mengembalikan statistik eksekusi untuk analisis performa
    pub fn get_execution_stats(&self) -> ExecutionStats {
        let total_executions = self.execution_history.len();
        let successful_executions = self
            .execution_history
            .iter()
            .filter(|record| matches!(record.result, ExecutionResult::Success))
            .count();
        let failed_executions = total_executions - successful_executions;

        let average_execution_time = if total_executions > 0 {
            self.execution_history
                .iter()
                .map(|record| record.execution_time_ms)
                .sum::<f32>()
                / total_executions as f32
        } else {
            0.0
        };

        ExecutionStats {
            total_executions,
            successful_executions,
            failed_executions,
            success_rate: if total_executions > 0 {
                successful_executions as f32 / total_executions as f32
            } else {
                0.0
            },
            average_execution_time_ms: average_execution_time,
        }
    }

    /// Menghapus seluruh riwayat eksekusi
    pub fn clear_history(&mut self) {
        self.execution_history.clear();
    }

    /// Mengembalikan referensi ke riwayat eksekusi
    pub fn get_history(&self) -> &[ExecutionRecord] {
        &self.execution_history
    }
}

/// Catatan eksekusi untuk satu tindakan
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    pub action: Action,
    pub result: ExecutionResult,
    pub execution_time_ms: f32,
    pub timestamp: f32,
}

/// Statistik eksekusi untuk analisis performa
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub total_executions: usize,
    pub successful_executions: usize,
    pub failed_executions: usize,
    pub success_rate: f32,
    pub average_execution_time_ms: f32,
}

/// Trait untuk handler tindakan
#[async_trait]
pub trait ActionHandler: Send {
    /// Mengeksekusi tindakan dan mengembalikan hasil
    async fn execute(&self, action: &Action) -> Result<ExecutionResult>;
    /// Mengembalikan nama handler
    fn get_handler_name(&self) -> &str;
}

/// Handler untuk tindakan klik
///
/// Menggunakan `xdotool` untuk klik kiri mouse, dengan fallback simulasi
/// jika `xdotool` tidak tersedia di sistem.
pub struct ClickHandler {
    click_delay_ms: u64,
}

impl ClickHandler {
    /// Membuat handler klik baru
    pub fn new() -> Self {
        Self {
            click_delay_ms: 100,
        }
    }
}

#[async_trait]
impl ActionHandler for ClickHandler {
    async fn execute(&self, action: &Action) -> Result<ExecutionResult> {
        let x = action
            .parameters
            .get("x")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let y = action
            .parameters
            .get("y")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        sleep(Duration::from_millis(self.click_delay_ms)).await;

        if !(x >= 0.0 && y >= 0.0 && x <= 1.0 && y <= 1.0) {
            warn!("Koordinat klik tidak valid: ({:.2}, {:.2})", x, y);
            return Ok(ExecutionResult::Failure);
        }

        match Command::new("xdotool")
            .arg("mousemove")
            .arg(x.to_string())
            .arg(y.to_string())
            .arg("click")
            .arg("1")
            .output()
            .await
        {
            Ok(o) if o.status.success() => {
                info!("Klik dieksekusi di ({:.2}, {:.2}) melalui xdotool", x, y);
                Ok(ExecutionResult::Success)
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                error!("xdotool gagal untuk klik: {}", stderr);
                Err(crate::multimodal::error::CaffeineError::action_head(
                    &format!("xdotool gagal: {}", stderr),
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                warn!(
                    "xdotool tidak ditemukan, tidak dapat klik di ({:.2}, {:.2})",
                    x, y
                );
                Err(crate::multimodal::error::CaffeineError::action_head(
                    "Click action requires xdotool which is not installed on this system",
                ))
            }
            Err(e) => {
                error!("Gagal menjalankan xdotool: {}", e);
                Err(crate::multimodal::error::CaffeineError::Io(e))
            }
        }
    }

    fn get_handler_name(&self) -> &str {
        "ClickHandler"
    }
}

/// Handler untuk tindakan mengetik teks
///
/// Menggunakan `xdotool type` untuk mengetik teks, dengan fallback simulasi
/// jika `xdotool` tidak tersedia di sistem.
pub struct TypeHandler {
    _typing_delay_ms: u64,
}

impl TypeHandler {
    /// Membuat handler ketik baru
    pub fn new() -> Self {
        Self {
            _typing_delay_ms: 50,
        }
    }
}

#[async_trait]
impl ActionHandler for TypeHandler {
    async fn execute(&self, action: &Action) -> Result<ExecutionResult> {
        let text = action
            .parameters
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if text.is_empty() {
            warn!("Teks kosong untuk tindakan ketik");
            return Ok(ExecutionResult::Failure);
        }

        let text = sanitize_xdotool_text(&text);
        if text.is_empty() {
            warn!("Text was empty after sanitization (all characters were blocked)");
            return Ok(ExecutionResult::Failure);
        }

        match Command::new("xdotool")
            .arg("type")
            .arg(&text)
            .output()
            .await
        {
            Ok(o) if o.status.success() => {
                info!("Teks diketik melalui xdotool ({} karakter)", text.len());
                Ok(ExecutionResult::Success)
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                error!("xdotool gagal untuk mengetik: {}", stderr);
                Err(crate::multimodal::error::CaffeineError::action_head(
                    &format!("xdotool gagal: {}", stderr),
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                warn!(
                    "xdotool tidak ditemukan, tidak dapat mengetik teks ({} karakter)",
                    text.len()
                );
                Err(crate::multimodal::error::CaffeineError::action_head(
                    "Type action requires xdotool which is not installed on this system",
                ))
            }
            Err(e) => {
                error!("Gagal menjalankan xdotool: {}", e);
                Err(crate::multimodal::error::CaffeineError::Io(e))
            }
        }
    }

    fn get_handler_name(&self) -> &str {
        "TypeHandler"
    }
}

/// Handler untuk tindakan scroll
///
/// Menggunakan `xdotool click 4/5` untuk scroll atas/bawah, dengan fallback
/// simulasi jika `xdotool` tidak tersedia di sistem.
pub struct ScrollHandler {
    scroll_speed: f32,
}

impl ScrollHandler {
    /// Membuat handler scroll baru
    pub fn new() -> Self {
        Self { scroll_speed: 1.0 }
    }
}

#[async_trait]
impl ActionHandler for ScrollHandler {
    async fn execute(&self, action: &Action) -> Result<ExecutionResult> {
        let direction = action
            .parameters
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("down");

        let amount = action
            .parameters
            .get("amount")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let scroll_distance = amount * self.scroll_speed;
        let button = match direction {
            "up" => "4",
            _ => "5",
        };

        match Command::new("xdotool")
            .arg("click")
            .arg("--repeat")
            .arg((scroll_distance as u64).to_string())
            .arg(button)
            .output()
            .await
        {
            Ok(o) if o.status.success() => {
                info!(
                    "Scroll {} sejauh {:.2} unit melalui xdotool",
                    direction, scroll_distance
                );
                Ok(ExecutionResult::Success)
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                error!("xdotool gagal untuk scroll: {}", stderr);
                Err(crate::multimodal::error::CaffeineError::action_head(
                    &format!("xdotool gagal: {}", stderr),
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                warn!(
                    "xdotool tidak ditemukan, tidak dapat scroll {} sejauh {:.2} unit",
                    direction, scroll_distance
                );
                Err(crate::multimodal::error::CaffeineError::action_head(
                    "Scroll action requires xdotool which is not installed on this system",
                ))
            }
            Err(e) => {
                error!("Gagal menjalankan xdotool: {}", e);
                Err(crate::multimodal::error::CaffeineError::Io(e))
            }
        }
    }

    fn get_handler_name(&self) -> &str {
        "ScrollHandler"
    }
}

/// Handler untuk tindakan drag
///
/// Menggunakan `xdotool` untuk drag mouse dari posisi awal ke posisi akhir,
/// dengan fallback simulasi jika `xdotool` tidak tersedia di sistem.
pub struct DragHandler {
    drag_duration_ms: u64,
}

impl DragHandler {
    /// Membuat handler drag baru
    pub fn new() -> Self {
        Self {
            drag_duration_ms: 500,
        }
    }
}

#[async_trait]
impl ActionHandler for DragHandler {
    async fn execute(&self, action: &Action) -> Result<ExecutionResult> {
        let start_x = action
            .parameters
            .get("start_x")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let start_y = action
            .parameters
            .get("start_y")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let end_x = action
            .parameters
            .get("end_x")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let end_y = action
            .parameters
            .get("end_y")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        match Command::new("xdotool")
            .arg("mousemove")
            .arg(start_x.to_string())
            .arg(start_y.to_string())
            .arg("mousedown")
            .arg("1")
            .output()
            .await
        {
            Ok(o) if o.status.success() => {
                sleep(Duration::from_millis(self.drag_duration_ms)).await;

                match Command::new("xdotool")
                    .arg("mousemove")
                    .arg(end_x.to_string())
                    .arg(end_y.to_string())
                    .arg("mouseup")
                    .arg("1")
                    .output()
                    .await
                {
                    Ok(o2) if o2.status.success() => {
                        info!(
                            "Drag dieksekusi dari ({:.2}, {:.2}) ke ({:.2}, {:.2}) melalui xdotool",
                            start_x, start_y, end_x, end_y
                        );
                        Ok(ExecutionResult::Success)
                    }
                    Ok(o2) => {
                        let stderr = String::from_utf8_lossy(&o2.stderr);
                        error!("xdotool gagal saat mouseup/mousemove: {}", stderr);
                        Err(crate::multimodal::error::CaffeineError::action_head(
                            &format!("xdotool gagal saat drag: {}", stderr),
                        ))
                    }
                    Err(e2) if e2.kind() == std::io::ErrorKind::NotFound => {
                        warn!(
                            "xdotool tidak ditemukan, tidak dapat drag dari ({:.2}, {:.2}) ke ({:.2}, {:.2})",
                            start_x, start_y, end_x, end_y
                        );
                        Err(crate::multimodal::error::CaffeineError::action_head(
                            "Drag action requires xdotool which is not installed on this system",
                        ))
                    }
                    Err(e2) => {
                        error!("Gagal menjalankan xdotool: {}", e2);
                        Err(crate::multimodal::error::CaffeineError::Io(e2))
                    }
                }
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                error!("xdotool gagal saat mousedown: {}", stderr);
                Err(crate::multimodal::error::CaffeineError::action_head(
                    &format!("xdotool gagal: {}", stderr),
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                warn!(
                    "xdotool tidak ditemukan, tidak dapat drag dari ({:.2}, {:.2}) ke ({:.2}, {:.2})",
                    start_x, start_y, end_x, end_y
                );
                Err(crate::multimodal::error::CaffeineError::action_head(
                    "Drag action requires xdotool which is not installed on this system",
                ))
            }
            Err(e) => {
                error!("Gagal menjalankan xdotool: {}", e);
                Err(crate::multimodal::error::CaffeineError::Io(e))
            }
        }
    }

    fn get_handler_name(&self) -> &str {
        "DragHandler"
    }
}

/// Handler untuk tindakan menunggu
///
/// Menunda eksekusi selama durasi yang ditentukan menggunakan `tokio::time::sleep`.
pub struct WaitHandler {
    default_wait_ms: u64,
}

impl WaitHandler {
    /// Membuat handler tunggu baru
    pub fn new() -> Self {
        Self {
            default_wait_ms: 1000,
        }
    }
}

#[async_trait]
impl ActionHandler for WaitHandler {
    async fn execute(&self, action: &Action) -> Result<ExecutionResult> {
        let duration_ms = action
            .parameters
            .get("duration_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.default_wait_ms);

        info!("Menunggu selama {} ms", duration_ms);
        sleep(Duration::from_millis(duration_ms)).await;

        Ok(ExecutionResult::Success)
    }

    fn get_handler_name(&self) -> &str {
        "WaitHandler"
    }
}

/// Handler untuk tindakan navigasi
///
/// Menggunakan `xdg-open` untuk membuka URL di browser default. Mengembalikan
/// error jika `xdg-open` tidak tersedia atau gagal.
pub struct NavigateHandler {
    _navigation_timeout_ms: u64,
}

impl NavigateHandler {
    /// Membuat handler navigasi baru
    pub fn new() -> Self {
        Self {
            _navigation_timeout_ms: 5000,
        }
    }
}

#[async_trait]
impl ActionHandler for NavigateHandler {
    async fn execute(&self, action: &Action) -> Result<ExecutionResult> {
        let destination = action
            .parameters
            .get("destination")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if destination.is_empty() || destination == "unknown" {
            warn!("Tujuan navigasi tidak valid: '{}'", destination);
            return Ok(ExecutionResult::Failure);
        }

        if !is_safe_url(destination) {
            warn!("Dangerous navigation destination blocked: '{}'", destination);
            return Err(crate::multimodal::error::CaffeineError::action_head(
                &format!("Navigation blocked: invalid or dangerous URL: {}", destination),
            ));
        }

        info!("Mencoba navigasi ke '{}'", destination);

        match Command::new("xdg-open").arg(destination).output().await {
            Ok(o) if o.status.success() => {
                info!("Navigasi ke '{}' berhasil", destination);
                Ok(ExecutionResult::Success)
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                error!("xdg-open gagal: {}", stderr);
                Err(crate::multimodal::error::CaffeineError::action_head(
                    &format!(
                        "Navigasi membutuhkan integrasi browser (xdg-open gagal): {}",
                        stderr
                    ),
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                error!("xdg-open tidak ditemukan di sistem");
                Err(crate::multimodal::error::CaffeineError::action_head(
                    "Navigasi membutuhkan integrasi browser: xdg-open tidak tersedia",
                ))
            }
            Err(e) => {
                error!("Gagal menjalankan xdg-open: {}", e);
                Err(crate::multimodal::error::CaffeineError::Io(e))
            }
        }
    }

    fn get_handler_name(&self) -> &str {
        "NavigateHandler"
    }
}

/// Handler untuk tindakan ekstraksi
/// Mengekstrak URL, email, nomor telepon, teks, atau pola kustom dari parameter.
pub struct ExtractHandler {
    _extraction_timeout_ms: u64,
}

impl ExtractHandler {
    pub fn new() -> Self {
        Self {
            _extraction_timeout_ms: 3000,
        }
    }
}

#[async_trait]
impl ActionHandler for ExtractHandler {
    async fn execute(&self, action: &Action) -> Result<ExecutionResult> {
        let target = action
            .parameters
            .get("target")
            .and_then(|v| v.as_str())
            .unwrap_or("text");

        let method = action
            .parameters
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("semantic");

        let content = action
            .parameters
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let extracted = match target {
            "url" | "URL" => RE_URL
                .find_iter(content)
                .map(|m| m.as_str().to_string())
                .collect::<Vec<_>>()
                .join("\n"),
            "email" => RE_EMAIL
                .find_iter(content)
                .map(|m| m.as_str().to_string())
                .collect::<Vec<_>>()
                .join("\n"),
            "phone" => RE_PHONE
                .find_iter(content)
                .map(|m| m.as_str().to_string())
                .collect::<Vec<_>>()
                .join("\n"),
            "custom" => {
                let pattern = action
                    .parameters
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if pattern.is_empty() {
                    return Err(crate::multimodal::error::CaffeineError::action_head(
                        "Custom extraction requires a 'pattern' parameter",
                    ));
                }
                let re = match Regex::new(pattern) {
                    Ok(r) => r,
                    Err(e) => {
                        return Err(crate::multimodal::error::CaffeineError::action_head(
                            &format!("Invalid regex pattern: {}", e),
                        ))
                    }
                };
                re.find_iter(content)
                    .map(|m| m.as_str().to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            _ => content.to_string(),
        };

        info!(
            "Extracted {} bytes using target '{}' method '{}'",
            extracted.len(),
            target,
            method
        );
        Ok(ExecutionResult::Success)
    }

    fn get_handler_name(&self) -> &str {
        "ExtractHandler"
    }
}

/// Handler untuk tindakan analisis
/// Melakukan analisis sentimen, entity recognition, length counting,
/// keyword extraction, atau summarization pada teks.
pub struct AnalyzeHandler {
    _analysis_timeout_ms: u64,
}

impl AnalyzeHandler {
    pub fn new() -> Self {
        Self {
            _analysis_timeout_ms: 2000,
        }
    }
}

#[async_trait]
impl ActionHandler for AnalyzeHandler {
    async fn execute(&self, action: &Action) -> Result<ExecutionResult> {
        let analysis_type = action
            .parameters
            .get("analysis_type")
            .and_then(|v| v.as_str())
            .unwrap_or("general");

        let text = action
            .parameters
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let result = match analysis_type {
            "sentiment" => {
                let positive_words = [
                    "good",
                    "great",
                    "excellent",
                    "amazing",
                    "wonderful",
                    "love",
                    "beautiful",
                    "fantastic",
                    "happy",
                    "awesome",
                ];
                let negative_words = [
                    "bad", "terrible", "awful", "horrible", "hate", "ugly", "poor", "worst", "sad",
                    "angry",
                ];
                let lower = text.to_lowercase();
                let pos_count = positive_words.iter().filter(|w| lower.contains(*w)).count();
                let neg_count = negative_words.iter().filter(|w| lower.contains(*w)).count();
                let sentiment = if pos_count > neg_count {
                    "positive"
                } else if neg_count > pos_count {
                    "negative"
                } else {
                    "neutral"
                };
                format!(
                    "sentiment: {} (positive: {}, negative: {})",
                    sentiment, pos_count, neg_count
                )
            }
            "entities" => {
                let mut entities = Vec::new();
                for m in RE_URL_SIMPLE.find_iter(text) {
                    entities.push(format!("URL: {}", m.as_str()));
                }
                for m in RE_EMAIL.find_iter(text) {
                    entities.push(format!("Email: {}", m.as_str()));
                }
                if entities.is_empty() {
                    "entities: none found".to_string()
                } else {
                    entities.join("\n")
                }
            }
            "length" => {
                let char_count = text.chars().count();
                let word_count = text.split_whitespace().count();
                format!("length: {} chars, {} words", char_count, word_count)
            }
            "keywords" => {
                let mut freq: std::collections::HashMap<String, usize> =
                    std::collections::HashMap::new();
                for word in text.split_whitespace() {
                    let clean = word
                        .trim_matches(|c: char| !c.is_alphanumeric())
                        .to_lowercase();
                    if clean.len() > 2 {
                        *freq.entry(clean).or_default() += 1;
                    }
                }
                let mut words: Vec<(String, usize)> = freq.into_iter().collect();
                words.sort_by(|a, b| b.1.cmp(&a.1));
                let top5: Vec<String> = words
                    .into_iter()
                    .take(5)
                    .map(|(w, c)| format!("{}:{}", w, c))
                    .collect();
                format!("keywords: {}", top5.join(", "))
            }
            "summary" => {
                let sentences: Vec<&str> = text
                    .split(|c: char| c == '.' || c == '!' || c == '?')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();
                let summary = if sentences.len() <= 2 {
                    sentences.join(". ")
                } else {
                    let take = (sentences.len() / 3).max(1);
                    sentences[..take].join(". ") + "."
                };
                if summary.len() > text.len() / 2 {
                    text.chars().take(200).collect::<String>() + "..."
                } else {
                    summary
                }
            }
            _ => {
                let char_count = text.chars().count();
                let word_count = text.split_whitespace().count();
                format!(
                    "general analysis: {} chars, {} words",
                    char_count, word_count
                )
            }
        };

        info!("Analysis '{}' completed: {}", analysis_type, result);
        Ok(ExecutionResult::Success)
    }

    fn get_handler_name(&self) -> &str {
        "AnalyzeHandler"
    }
}
