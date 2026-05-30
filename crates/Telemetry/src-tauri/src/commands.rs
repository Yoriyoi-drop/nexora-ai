use crate::telemetry::{
    AgentTelemetry, AiHealthTelemetry, ConnectionStatus, HallucinationTelemetry,
    InferenceTelemetry, MemoryResponse, ModelTelemetry, NexoraAIMetrics, PipelineTelemetry,
    SystemMetrics, TelemetryClient, TokenFlowTelemetry, TrainingTelemetry, TelemetrySnapshot,
};
use std::sync::Arc;
use std::sync::Mutex;
use tauri::State;
use sysinfo::System;

pub struct AppState {
    client: Arc<Mutex<Option<TelemetryClient>>>,
    sys: Arc<Mutex<System>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            client: Arc::new(Mutex::new(None)),
            sys: Arc::new(Mutex::new(System::new_all())),
        }
    }
}

#[tauri::command]
fn get_system_metrics(state: State<'_, AppState>) -> Result<SystemMetrics, String> {
    let mut sys = state.sys.lock().map_err(|e| e.to_string())?;
    sys.refresh_all();

    let cpus = sys.cpus();
    let cpu_cores = cpus.len();
    let cpu_per_core: Vec<f64> = cpus.iter().map(|c| c.cpu_usage() as f64).collect();
    let cpu_usage = cpu_per_core.iter().sum::<f64>() / cpu_cores.max(1) as f64;

    let total_mem = sys.total_memory();
    let used_mem = sys.used_memory();
    let ram_total_gb = total_mem as f64 / 1_073_741_824.0;
    let ram_used_gb = used_mem as f64 / 1_073_741_824.0;
    let ram_percent = if total_mem > 0 {
        (used_mem as f64 / total_mem as f64) * 100.0
    } else {
        0.0
    };

    Ok(SystemMetrics {
        cpu_usage,
        ram_used_gb,
        ram_total_gb,
        ram_percent,
        disk_used_gb: 0.0,
        disk_total_gb: 0.0,
        disk_read_bytes: 0,
        disk_write_bytes: 0,
        network_rx_bytes: 0,
        network_tx_bytes: 0,
        processes: sys.processes().len(),
        uptime_secs: System::uptime(),
        cpu_cores,
        cpu_per_core,
        gpu_usage: None,
        gpu_vram_used_gb: None,
        gpu_vram_total_gb: None,
    })
}

#[tauri::command]
async fn connect_nexora_ai(state: State<'_, AppState>, url: String) -> Result<NexoraAIMetrics, String> {
    let client = TelemetryClient::new(&url);
    let health = client.fetch_health().await.map_err(|e| e.to_string()).ok();

    let mut guard = state.client.lock().map_err(|e| e.to_string())?;
    *guard = Some(client);

    let connected = health.is_some();
    let err_msg = if health.is_none() { Some("Failed to connect".into()) } else { None };
    Ok(NexoraAIMetrics {
        connected,
        url,
        health,
        error: err_msg,
    })
}

#[tauri::command]
async fn disconnect_nexora_ai(state: State<'_, AppState>) -> Result<(), String> {
    let mut guard = state.client.lock().map_err(|e| e.to_string())?;
    *guard = None;
    Ok(())
}

#[tauri::command]
async fn get_connection_status(state: State<'_, AppState>) -> Result<ConnectionStatus, String> {
    let guard = state.client.lock().map_err(|e| e.to_string())?;
    match guard.as_ref() {
        Some(client) => Ok(ConnectionStatus {
            connected: true,
            url: client.base_url.clone(),
            system_metrics: true,
            ai_metrics: true,
        }),
        None => Ok(ConnectionStatus {
            connected: false,
            url: String::new(),
            system_metrics: false,
            ai_metrics: false,
        }),
    }
}

#[tauri::command]
async fn get_ai_health(state: State<'_, AppState>) -> Result<Option<AiHealthTelemetry>, String> {
    let client = match state.client.lock().map_err(|e| e.to_string())?.clone() {
        Some(c) => c,
        None => return Ok(None),
    };
    client.fetch_health().await.map_err(|e| e.to_string()).map(Some)
}

#[tauri::command]
async fn get_inference_telemetry(state: State<'_, AppState>) -> Result<Option<InferenceTelemetry>, String> {
    let client = match state.client.lock().map_err(|e| e.to_string())?.clone() {
        Some(c) => c,
        None => return Ok(None),
    };
    client.fetch_inference().await.map_err(|e| e.to_string()).map(Some)
}

#[tauri::command]
async fn get_agent_telemetry(state: State<'_, AppState>) -> Result<Vec<AgentTelemetry>, String> {
    let client = match state.client.lock().map_err(|e| e.to_string())?.clone() {
        Some(c) => c,
        None => return Ok(vec![]),
    };
    client.fetch_agents().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_memory_telemetry(state: State<'_, AppState>) -> Result<MemoryResponse, String> {
    let client = match state.client.lock().map_err(|e| e.to_string())?.clone() {
        Some(c) => c,
        None => return Ok(MemoryResponse { nodes: vec![], summary: None }),
    };
    client.fetch_memory().await
        .map(|(nodes, summary)| MemoryResponse { nodes, summary })
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_pipeline_telemetry(state: State<'_, AppState>) -> Result<Vec<PipelineTelemetry>, String> {
    let client = match state.client.lock().map_err(|e| e.to_string())?.clone() {
        Some(c) => c,
        None => return Ok(vec![]),
    };
    client.fetch_pipelines().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_hallucination_telemetry(state: State<'_, AppState>) -> Result<Option<HallucinationTelemetry>, String> {
    let client = match state.client.lock().map_err(|e| e.to_string())?.clone() {
        Some(c) => c,
        None => return Ok(None),
    };
    client.fetch_hallucinations().await.map_err(|e| e.to_string()).map(Some)
}

#[tauri::command]
async fn get_training_telemetry(state: State<'_, AppState>) -> Result<Option<TrainingTelemetry>, String> {
    let client = match state.client.lock().map_err(|e| e.to_string())?.clone() {
        Some(c) => c,
        None => return Ok(None),
    };
    client.fetch_training().await.map_err(|e| e.to_string()).map(Some)
}

#[tauri::command]
async fn get_model_telemetry(state: State<'_, AppState>) -> Result<Vec<ModelTelemetry>, String> {
    let client = match state.client.lock().map_err(|e| e.to_string())?.clone() {
        Some(c) => c,
        None => return Ok(vec![]),
    };
    client.fetch_models().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_token_flow_telemetry(state: State<'_, AppState>) -> Result<Vec<TokenFlowTelemetry>, String> {
    let client = match state.client.lock().map_err(|e| e.to_string())?.clone() {
        Some(c) => c,
        None => return Ok(vec![]),
    };
    client.fetch_token_flows().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_aggregated_telemetry(state: State<'_, AppState>) -> Result<Option<TelemetrySnapshot>, String> {
    let client = match state.client.lock().map_err(|e| e.to_string())?.clone() {
        Some(c) => c,
        None => return Ok(None),
    };
    client.fetch_snapshot().await.map_err(|e| e.to_string()).map(Some)
}
