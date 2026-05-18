use std::sync::Mutex;
use sysinfo::{Disks, Networks, System};
use tauri::{Emitter, State};

mod telemetry;

pub use telemetry::*;

#[derive(Debug, Default)]
pub struct AppState {
    pub nexora_ai_url: Mutex<Option<String>>,
    pub system: Mutex<System>,
    pub telemetry_client: Mutex<Option<TelemetryClient>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub ram_used_gb: f64,
    pub ram_total_gb: f64,
    pub ram_percent: f64,
    pub disk_used_gb: f64,
    pub disk_total_gb: f64,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
    pub processes: usize,
    pub uptime_secs: u64,
    pub cpu_cores: usize,
    pub cpu_per_core: Vec<f64>,
    pub gpu_usage: Option<f64>,
    pub gpu_vram_used_gb: Option<f64>,
    pub gpu_vram_total_gb: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NexoraAIMetrics {
    pub connected: bool,
    pub url: String,
    pub health: Option<AiHealthTelemetry>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConnectedStatus {
    pub connected: bool,
    pub url: String,
    pub system_metrics: bool,
    pub ai_metrics: bool,
}

/// Aggregated telemetry for the frontend
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AggregatedTelemetry {
    pub system: Option<SystemMetrics>,
    pub ai_health: Option<AiHealthTelemetry>,
    pub inference: Option<InferenceTelemetry>,
    pub agents: Vec<AgentTelemetry>,
    pub memory_nodes: Vec<MemoryTelemetry>,
    pub memory_summary: Option<MemorySummary>,
    pub pipelines: Vec<PipelineTelemetry>,
    pub hallucinations: Option<HallucinationTelemetry>,
    pub training: Option<TrainingTelemetry>,
    pub models: Vec<ModelTelemetry>,
    pub token_flows: Vec<TokenFlowTelemetry>,
}

// ===== Tauri Commands =====

#[tauri::command]
fn get_system_metrics(state: State<AppState>) -> SystemMetrics {
    let mut sys = state.system.lock().unwrap();
    sys.refresh_all();

    let cpu_usage = sys.global_cpu_usage() as f64;
    let cpu_per_core: Vec<f64> = sys.cpus().iter().map(|c| c.cpu_usage() as f64).collect();
    let cpu_cores = sys.cpus().len();

    let ram_used = sys.used_memory();
    let ram_total = sys.total_memory();
    let ram_used_gb = ram_used as f64 / 1024.0 / 1024.0 / 1024.0;
    let ram_total_gb = ram_total as f64 / 1024.0 / 1024.0 / 1024.0;
    let ram_percent = if ram_total > 0 { (ram_used as f64 / ram_total as f64) * 100.0 } else { 0.0 };

    let disks = Disks::new_with_refreshed_list();
    let (disk_total, disk_used, disk_read, disk_write) = {
        let mut total = 0u64; let mut used = 0u64; let mut read = 0u64; let mut write = 0u64;
        for d in &disks {
            total += d.total_space();
            used += d.total_space() - d.available_space();
            read += d.usage().total_read_bytes;
            write += d.usage().total_written_bytes;
        }
        (total, used, read, write)
    };

    let networks = Networks::new_with_refreshed_list();
    let (net_rx, net_tx) = {
        let mut rx = 0u64; let mut tx = 0u64;
        for (_, data) in &networks { rx += data.total_received(); tx += data.total_transmitted(); }
        (rx, tx)
    };

    let uptime = System::uptime();

    SystemMetrics {
        cpu_usage,
        ram_used_gb, ram_total_gb, ram_percent,
        disk_used_gb: disk_used as f64 / 1024.0 / 1024.0 / 1024.0,
        disk_total_gb: disk_total as f64 / 1024.0 / 1024.0 / 1024.0,
        disk_read_bytes: disk_read, disk_write_bytes: disk_write,
        network_rx_bytes: net_rx, network_tx_bytes: net_tx,
        processes: sys.processes().len(),
        uptime_secs: uptime, cpu_cores, cpu_per_core,
        gpu_usage: None, gpu_vram_used_gb: None, gpu_vram_total_gb: None,
    }
}

#[tauri::command]
async fn connect_nexora_ai(state: State<'_, AppState>, url: String) -> Result<NexoraAIMetrics, String> {
    let base_url = url.trim_end_matches('/').to_string();
    let client = TelemetryClient::new(&base_url);

    let health = client.fetch_health().await;

    match &health {
        Ok(h) => {
            *state.nexora_ai_url.lock().unwrap() = Some(base_url.clone());
            *state.telemetry_client.lock().unwrap() = Some(TelemetryClient::new(&base_url));
            Ok(NexoraAIMetrics {
                connected: true,
                url: base_url,
                health: Some(h.clone()),
                error: None,
            })
        }
        Err(e) => {
            *state.nexora_ai_url.lock().unwrap() = None;
            Ok(NexoraAIMetrics {
                connected: false, url: base_url,
                health: None,
                error: Some(format!("Connection failed: {}", e)),
            })
        }
    }
}

#[tauri::command]
fn get_connection_status(state: State<AppState>) -> ConnectedStatus {
    let url_guard = state.nexora_ai_url.lock().unwrap();
    let connected = url_guard.is_some();
    ConnectedStatus {
        connected,
        url: url_guard.clone().unwrap_or_default(),
        system_metrics: true,
        ai_metrics: connected,
    }
}

#[tauri::command]
fn disconnect_nexora_ai(state: State<AppState>) {
    *state.nexora_ai_url.lock().unwrap() = None;
    *state.telemetry_client.lock().unwrap() = None;
}

// ===== New Telemetry Commands =====

fn get_client(state: &AppState) -> Option<TelemetryClient> {
    state.telemetry_client.lock().unwrap().clone()
}

#[tauri::command]
async fn get_ai_health(state: State<'_, AppState>) -> Result<Option<AiHealthTelemetry>, String> {
    match get_client(&state) {
        Some(c) => c.fetch_health().await.map(Some),
        None => Ok(None),
    }
}

#[tauri::command]
async fn get_inference_telemetry(state: State<'_, AppState>) -> Result<Option<InferenceTelemetry>, String> {
    match get_client(&state) {
        Some(c) => c.fetch_inference().await.map(Some),
        None => Ok(None),
    }
}

#[tauri::command]
async fn get_agent_telemetry(state: State<'_, AppState>) -> Result<Vec<AgentTelemetry>, String> {
    match get_client(&state) {
        Some(c) => c.fetch_agents().await,
        None => Ok(vec![]),
    }
}

#[tauri::command]
async fn get_memory_telemetry(state: State<'_, AppState>) -> Result<telemetry::MemoryResponse, String> {
    match get_client(&state) {
        Some(c) => {
            let (nodes, summary) = c.fetch_memory().await?;
            Ok(telemetry::MemoryResponse { nodes, summary })
        }
        None => Ok(telemetry::MemoryResponse { nodes: vec![], summary: None }),
    }
}

#[tauri::command]
async fn get_pipeline_telemetry(state: State<'_, AppState>) -> Result<Vec<PipelineTelemetry>, String> {
    match get_client(&state) {
        Some(c) => c.fetch_pipelines().await,
        None => Ok(vec![]),
    }
}

#[tauri::command]
async fn get_hallucination_telemetry(state: State<'_, AppState>) -> Result<Option<HallucinationTelemetry>, String> {
    match get_client(&state) {
        Some(c) => c.fetch_hallucinations().await.map(Some),
        None => Ok(None),
    }
}

#[tauri::command]
async fn get_training_telemetry(state: State<'_, AppState>) -> Result<Option<TrainingTelemetry>, String> {
    match get_client(&state) {
        Some(c) => c.fetch_training().await.map(Some),
        None => Ok(None),
    }
}

#[tauri::command]
async fn get_model_telemetry(state: State<'_, AppState>) -> Result<Vec<ModelTelemetry>, String> {
    match get_client(&state) {
        Some(c) => c.fetch_models().await,
        None => Ok(vec![]),
    }
}

#[tauri::command]
async fn get_token_flow_telemetry(state: State<'_, AppState>) -> Result<Vec<TokenFlowTelemetry>, String> {
    match get_client(&state) {
        Some(c) => c.fetch_token_flows().await,
        None => Ok(vec![]),
    }
}

#[tauri::command]
async fn get_aggregated_telemetry(state: State<'_, AppState>) -> Result<AggregatedTelemetry, String> {
    let sys = Some(get_system_metrics(state.clone()));

    match get_client(&state) {
        Some(c) => {
            let snapshot = c.fetch_snapshot().await.unwrap_or(TelemetrySnapshot {
                timestamp: 0, system: None, ai_health: None, inference: None,
                agents: vec![], memory_nodes: vec![], memory_summary: None,
                pipelines: vec![], hallucinations: None, training: None,
                models: vec![], token_flows: vec![],
            });
            Ok(AggregatedTelemetry {
                system: sys,
                ai_health: snapshot.ai_health,
                inference: snapshot.inference,
                agents: snapshot.agents,
                memory_nodes: snapshot.memory_nodes,
                memory_summary: snapshot.memory_summary,
                pipelines: snapshot.pipelines,
                hallucinations: snapshot.hallucinations,
                training: snapshot.training,
                models: snapshot.models,
                token_flows: snapshot.token_flows,
            })
        }
        None => Ok(AggregatedTelemetry {
            system: sys,
            ai_health: None, inference: None,
            agents: vec![], memory_nodes: vec![], memory_summary: None,
            pipelines: vec![], hallucinations: None, training: None,
            models: vec![], token_flows: vec![],
        }),
    }
}

// ===== Telemetry Event Stream =====

fn start_telemetry_stream(app_handle: tauri::AppHandle) {
    tokio::spawn(async move {
        let mut system = System::new();
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
        let mut counter: u64 = 0;

        loop {
            interval.tick().await;
            counter += 1;

            system.refresh_all();

            let cpu = system.global_cpu_usage() as f64;
            let ram_used = system.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
            let ram_total = system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
            let ram_pct = if system.total_memory() > 0 {
                (system.used_memory() as f64 / system.total_memory() as f64) * 100.0
            } else { 0.0 };

            let sys_event = TelemetryEvent {
                timestamp: chrono::Utc::now().timestamp() as u64,
                source: "system".into(),
                event_type: "system_metrics".into(),
                value: cpu,
                severity: if cpu > 90.0 { "critical".into() } else if cpu > 70.0 { "warning".into() } else { "normal".into() },
                label: format!("CPU: {:.1}% | RAM: {:.1}/{:.1}GB ({:.0}%)", cpu, ram_used, ram_total, ram_pct),
                metadata: None,
            };
            let _ = app_handle.emit("telemetry:system", &sys_event);

            if counter % 5 == 0 {
                let uptime = System::uptime();
                let summary = serde_json::json!({
                    "cpu_usage": cpu,
                    "ram_used_gb": ram_used,
                    "ram_total_gb": ram_total,
                    "ram_percent": ram_pct,
                    "cpu_cores": system.cpus().len(),
                    "processes": system.processes().len(),
                    "uptime_secs": uptime,
                    "timestamp": chrono::Utc::now().timestamp(),
                });
                let _ = app_handle.emit("telemetry:summary", &summary);
            }
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            start_telemetry_stream(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_system_metrics,
            connect_nexora_ai,
            get_connection_status,
            disconnect_nexora_ai,
            get_ai_health,
            get_inference_telemetry,
            get_agent_telemetry,
            get_memory_telemetry,
            get_pipeline_telemetry,
            get_hallucination_telemetry,
            get_training_telemetry,
            get_model_telemetry,
            get_token_flow_telemetry,
            get_aggregated_telemetry,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
