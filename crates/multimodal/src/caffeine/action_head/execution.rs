//! Mesin eksekusi untuk CAFFEINE
//!
//! Mengeksekusi tindakan yang direncanakan dan menangani hasilnya

use crate::caffeine::error::Result;
use crate::caffeine::types::*;
use async_trait::async_trait;
use std::collections::HashMap;
use tokio::process::Command;
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};

/// Mesin eksekusi untuk menjalankan tindakan
pub struct ExecutionEngine {
    action_config: crate::caffeine::config::ActionConfig,
    action_handlers: HashMap<ActionType, Box<dyn ActionHandler>>,
    execution_history: Vec<ExecutionRecord>,
}

impl ExecutionEngine {
    /// Membuat mesin eksekusi baru dengan konfigurasi yang diberikan
    pub fn new(config: crate::caffeine::config::ActionConfig) -> Result<Self> {
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
            action_config: config,
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
                        crate::caffeine::error::CaffeineError::output_generation(&format!(
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
            Err(crate::caffeine::error::CaffeineError::action_head(
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
                Err(crate::caffeine::error::CaffeineError::action_head(
                    &format!("xdotool gagal: {}", stderr),
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                warn!(
                    "xdotool tidak ditemukan, simulasi klik di ({:.2}, {:.2})",
                    x, y
                );
                Ok(ExecutionResult::Success)
            }
            Err(e) => {
                error!("Gagal menjalankan xdotool: {}", e);
                Err(crate::caffeine::error::CaffeineError::Io(e))
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
    typing_delay_ms: u64,
}

impl TypeHandler {
    /// Membuat handler ketik baru
    pub fn new() -> Self {
        Self {
            typing_delay_ms: 50,
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

        match Command::new("xdotool")
            .arg("type")
            .arg(&text)
            .output()
            .await
        {
            Ok(o) if o.status.success() => {
                info!(
                    "Teks diketik melalui xdotool ({} karakter)",
                    text.len()
                );
                Ok(ExecutionResult::Success)
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                error!("xdotool gagal untuk mengetik: {}", stderr);
                Err(crate::caffeine::error::CaffeineError::action_head(
                    &format!("xdotool gagal: {}", stderr),
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                warn!(
                    "xdotool tidak ditemukan, simulasi ketik teks ({} karakter)",
                    text.len()
                );
                sleep(Duration::from_millis(
                    self.typing_delay_ms * text.len() as u64,
                ))
                .await;
                Ok(ExecutionResult::Success)
            }
            Err(e) => {
                error!("Gagal menjalankan xdotool: {}", e);
                Err(crate::caffeine::error::CaffeineError::Io(e))
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
                Err(crate::caffeine::error::CaffeineError::action_head(
                    &format!("xdotool gagal: {}", stderr),
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                warn!(
                    "xdotool tidak ditemukan, simulasi scroll {} sejauh {:.2} unit",
                    direction, scroll_distance
                );
                Ok(ExecutionResult::Success)
            }
            Err(e) => {
                error!("Gagal menjalankan xdotool: {}", e);
                Err(crate::caffeine::error::CaffeineError::Io(e))
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
                        error!(
                            "xdotool gagal saat mouseup/mousemove: {}",
                            stderr
                        );
                        Err(crate::caffeine::error::CaffeineError::action_head(
                            &format!("xdotool gagal saat drag: {}", stderr),
                        ))
                    }
                    Err(e2) if e2.kind() == std::io::ErrorKind::NotFound => {
                        warn!(
                            "xdotool tidak ditemukan, simulasi drag dari ({:.2}, {:.2}) ke ({:.2}, {:.2})",
                            start_x, start_y, end_x, end_y
                        );
                        Ok(ExecutionResult::Success)
                    }
                    Err(e2) => {
                        error!("Gagal menjalankan xdotool: {}", e2);
                        Err(crate::caffeine::error::CaffeineError::Io(e2))
                    }
                }
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                error!("xdotool gagal saat mousedown: {}", stderr);
                Err(crate::caffeine::error::CaffeineError::action_head(
                    &format!("xdotool gagal: {}", stderr),
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                warn!(
                    "xdotool tidak ditemukan, simulasi drag dari ({:.2}, {:.2}) ke ({:.2}, {:.2})",
                    start_x, start_y, end_x, end_y
                );
                sleep(Duration::from_millis(self.drag_duration_ms)).await;
                Ok(ExecutionResult::Success)
            }
            Err(e) => {
                error!("Gagal menjalankan xdotool: {}", e);
                Err(crate::caffeine::error::CaffeineError::Io(e))
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
    navigation_timeout_ms: u64,
}

impl NavigateHandler {
    /// Membuat handler navigasi baru
    pub fn new() -> Self {
        Self {
            navigation_timeout_ms: 5000,
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

        info!("Mencoba navigasi ke '{}'", destination);

        match Command::new("xdg-open")
            .arg(destination)
            .output()
            .await
        {
            Ok(o) if o.status.success() => {
                info!("Navigasi ke '{}' berhasil", destination);
                Ok(ExecutionResult::Success)
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                error!("xdg-open gagal: {}", stderr);
                Err(crate::caffeine::error::CaffeineError::action_head(
                    &format!(
                        "Navigasi membutuhkan integrasi browser (xdg-open gagal): {}",
                        stderr
                    ),
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                error!("xdg-open tidak ditemukan di sistem");
                Err(crate::caffeine::error::CaffeineError::action_head(
                    "Navigasi membutuhkan integrasi browser: xdg-open tidak tersedia",
                ))
            }
            Err(e) => {
                error!("Gagal menjalankan xdg-open: {}", e);
                Err(crate::caffeine::error::CaffeineError::Io(e))
            }
        }
    }

    fn get_handler_name(&self) -> &str {
        "NavigateHandler"
    }
}

/// Handler untuk tindakan ekstraksi
///
/// Saat ini menggunakan logika ekstraksi simulasi. Membutuhkan backend ekstraksi
/// nyata untuk produksi.
pub struct ExtractHandler {
    extraction_timeout_ms: u64,
}

impl ExtractHandler {
    /// Membuat handler ekstraksi baru
    pub fn new() -> Self {
        Self {
            extraction_timeout_ms: 3000,
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

        warn!(
            "Ekstraksi '{}' menggunakan metode '{}' — ini adalah simulasi, butuh backend ekstraksi nyata",
            target, method
        );

        sleep(Duration::from_millis(500)).await;

        let extracted_content = match target {
            "text" => "Sample extracted text content",
            "image" => "Sample extracted image description",
            "data" => "Sample extracted data",
            _ => "Sample extracted content",
        };

        info!("Ekstraksi selesai: {}", extracted_content);

        Ok(ExecutionResult::Success)
    }

    fn get_handler_name(&self) -> &str {
        "ExtractHandler"
    }
}

/// Handler untuk tindakan analisis
///
/// Saat ini menggunakan logika analisis simulasi. Membutuhkan backend analisis
/// nyata untuk produksi.
pub struct AnalyzeHandler {
    analysis_timeout_ms: u64,
}

impl AnalyzeHandler {
    /// Membuat handler analisis baru
    pub fn new() -> Self {
        Self {
            analysis_timeout_ms: 2000,
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

        let _context = action
            .parameters
            .get("context")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        warn!(
            "Analisis '{}' — ini adalah simulasi, butuh backend analisis nyata",
            analysis_type
        );

        sleep(Duration::from_millis(800)).await;

        let analysis_result = match analysis_type {
            "classification" => "Classification: Positive",
            "sentiment" => "Sentiment: Neutral",
            "semantic" => "Semantic analysis completed",
            _ => "General analysis completed",
        };

        info!("Hasil analisis: {}", analysis_result);

        Ok(ExecutionResult::Success)
    }

    fn get_handler_name(&self) -> &str {
        "AnalyzeHandler"
    }
}
