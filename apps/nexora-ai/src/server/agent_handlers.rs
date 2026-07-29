use axum::{extract::Path, http::StatusCode, response::IntoResponse, Extension, Json};
use nexora_agent::agent_manager::ManagerCommand;
use nexora_alignment::isolation::killswitch::{KillTarget, KillTrigger};
use nexora_alignment::isolation::layer1_mode::ModeId;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::oneshot;

use crate::NexoraAI;

#[derive(Deserialize)]
pub struct GenerateViaAgentRequest {
    pub prompt: String,
    #[serde(default)]
    pub context: Option<Value>,
}

/// Convenience endpoint: prompt → create plan → dispatch → wait for completion → return result
pub async fn generate_via_agent(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(req): Json<GenerateViaAgentRequest>,
) -> impl IntoResponse {
    let agent_manager = nexora.agent_manager();
    let cmd_tx = agent_manager.command_sender();

    // 1. Find planner agent
    let (tx, rx) = oneshot::channel();
    if cmd_tx
        .send(ManagerCommand::ListAgentIds { response_tx: tx })
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to list agents"})),
        );
    }
    let grouped = rx.await.unwrap_or_default();
    let planner_ids = grouped.get("planner").cloned().unwrap_or_default();
    let planner_id = match planner_ids.first() {
        Some(id) => *id,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "No planner agent available"})),
            );
        }
    };

    // 2. Create plan
    let mut payload = json!({"task": req.prompt});
    if let Some(ctx) = req.context {
        payload["context"] = ctx;
    }
    let msg = nexora_agent::AgentMessage::new("create_plan", payload);
    let (tx2, rx2) = oneshot::channel();
    if cmd_tx
        .send(ManagerCommand::SendMessage {
            agent_id: planner_id,
            message: msg,
            response_tx: tx2,
        })
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to send create_plan to planner"})),
        );
    }
    let create_resp = match rx2.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("Planner error: {}", e)})),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("Channel error: {}", e)})),
            );
        }
    };
    let plan_id = match create_resp.payload.get("plan_id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Planner did not return plan_id", "payload": create_resp.payload})),
            );
        }
    };
    let plan_uuid = match uuid::Uuid::parse_str(&plan_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Invalid plan_id from planner"})),
            );
        }
    };

    // 3. Dispatch plan (blocks until all steps complete)
    let (tx3, rx3) = oneshot::channel();
    if cmd_tx
        .send(ManagerCommand::DispatchPlan {
            plan_id: plan_uuid,
            response_tx: tx3,
        })
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to dispatch plan"})),
        );
    }
    if let Err(e) = rx3.await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Dispatch channel error: {}", e)})),
        );
    }

    // 4. Get final plan status
    let (tx4, rx4) = oneshot::channel();
    if cmd_tx
        .send(ManagerCommand::PlanStatus {
            plan_id: plan_uuid,
            response_tx: tx4,
        })
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to query plan status"})),
        );
    }
    match rx4.await {
        Ok(Ok(payload)) => (StatusCode::OK, Json(payload)),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Channel error: {}", e)})),
        ),
    }
}

#[derive(Serialize)]
pub struct AgentListResponse {
    pub agents: Vec<AgentSummary>,
    pub count: usize,
}

#[derive(Serialize)]
pub struct AgentSummary {
    pub id: String,
    pub agent_type: String,
    pub status: String,
}

#[derive(Deserialize)]
pub struct CreatePlanRequest {
    pub task: String,
    #[serde(default)]
    pub context: Option<Value>,
}

#[derive(Deserialize)]
pub struct DispatchRequest {
    pub plan_id: String,
}

pub async fn list_agents(Extension(nexora): Extension<Arc<NexoraAI>>) -> Json<AgentListResponse> {
    let agent_manager = nexora.agent_manager();
    let cmd_tx = agent_manager.command_sender();
    let (tx, rx) = oneshot::channel();

    if cmd_tx
        .send(ManagerCommand::ListAgentIds { response_tx: tx })
        .await
        .is_err()
    {
        return Json(AgentListResponse {
            agents: vec![],
            count: 0,
        });
    }

    let grouped = rx.await.unwrap_or_default();
    let mut agents = Vec::new();

    for (agent_type, ids) in &grouped {
        for id in ids {
            agents.push(AgentSummary {
                id: id.to_string(),
                agent_type: agent_type.clone(),
                status: "active".to_string(),
            });
        }
    }

    let count = agents.len();
    Json(AgentListResponse { agents, count })
}

pub async fn create_plan(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(req): Json<CreatePlanRequest>,
) -> impl IntoResponse {
    let agent_manager = nexora.agent_manager();
    let cmd_tx = agent_manager.command_sender();

    // Find planner agent
    let (tx, rx) = oneshot::channel();
    if cmd_tx
        .send(ManagerCommand::ListAgentIds { response_tx: tx })
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to list agents"})),
        );
    }

    let grouped = rx.await.unwrap_or_default();
    let planner_ids = grouped.get("planner").cloned().unwrap_or_default();
    let planner_id = match planner_ids.first() {
        Some(id) => *id,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "No planner agent available"})),
            );
        }
    };

    let mut payload = json!({"task": req.task});
    if let Some(ctx) = req.context {
        payload["context"] = ctx;
    }

    let msg = nexora_agent::AgentMessage::new("create_plan", payload);
    let (tx2, rx2) = oneshot::channel();
    if cmd_tx
        .send(ManagerCommand::SendMessage {
            agent_id: planner_id,
            message: msg,
            response_tx: tx2,
        })
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to send message to planner"})),
        );
    }

    match rx2.await {
        Ok(Ok(resp)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "created",
                "plan_id": resp.payload.get("plan_id"),
                "details": resp.payload,
            })),
        ),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Channel error: {}", e)})),
        ),
    }
}

pub async fn get_plan(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Path(plan_id): Path<String>,
) -> impl IntoResponse {
    let agent_manager = nexora.agent_manager();
    let cmd_tx = agent_manager.command_sender();

    let (tx, rx) = oneshot::channel();
    let plan_uuid = match uuid::Uuid::parse_str(&plan_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid plan_id format"})),
            );
        }
    };

    if cmd_tx
        .send(ManagerCommand::PlanStatus {
            plan_id: plan_uuid,
            response_tx: tx,
        })
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to query plan status"})),
        );
    }

    match rx.await {
        Ok(Ok(payload)) => (StatusCode::OK, Json(payload)),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Channel error: {}", e)})),
        ),
    }
}

pub async fn dispatch_plan(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(req): Json<DispatchRequest>,
) -> impl IntoResponse {
    let agent_manager = nexora.agent_manager();
    let cmd_tx = agent_manager.command_sender();

    let plan_uuid = match uuid::Uuid::parse_str(&req.plan_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid plan_id format"})),
            );
        }
    };

    let (tx, rx) = oneshot::channel();
    if cmd_tx
        .send(ManagerCommand::DispatchPlan {
            plan_id: plan_uuid,
            response_tx: tx,
        })
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to dispatch plan"})),
        );
    }

    match rx.await {
        Ok(Ok(())) => (
            StatusCode::OK,
            Json(json!({"status": "dispatched", "plan_id": req.plan_id})),
        ),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Channel error: {}", e)})),
        ),
    }
}

pub async fn list_plans(
    Extension(nexora): Extension<Arc<NexoraAI>>,
) -> impl IntoResponse {
    let agent_manager = nexora.agent_manager();
    let cmd_tx = agent_manager.command_sender();

    // Find planner agent
    let (tx, rx) = oneshot::channel();
    if cmd_tx
        .send(ManagerCommand::ListAgentIds { response_tx: tx })
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to list agents"})),
        );
    }

    let grouped = rx.await.unwrap_or_default();
    let planner_ids = grouped.get("planner").cloned().unwrap_or_default();
    let planner_id = match planner_ids.first() {
        Some(id) => *id,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "No planner agent available"})),
            );
        }
    };

    let msg = nexora_agent::AgentMessage::new("list_plans", json!({}));
    let (tx2, rx2) = oneshot::channel();
    if cmd_tx
        .send(ManagerCommand::SendMessage {
            agent_id: planner_id,
            message: msg,
            response_tx: tx2,
        })
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to query plans"})),
        );
    }

    match rx2.await {
        Ok(Ok(resp)) => (StatusCode::OK, Json(resp.payload)),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Channel error: {}", e)})),
        ),
    }
}

#[derive(Deserialize)]
pub struct KillSwitchRequest {
    pub target_type: String,
    pub target_id: Option<String>,
    pub reason: String,
    pub trigger_type: String,
}

pub async fn trigger_kill_switch(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(req): Json<KillSwitchRequest>,
) -> impl IntoResponse {
    let target = match req.target_type.as_str() {
        "agent" => {
            let id = match &req.target_id {
                Some(id) => match uuid::Uuid::parse_str(id) {
                    Ok(uid) => uid,
                    Err(_) => return (StatusCode::BAD_REQUEST, Json(json!({"error": "Invalid agent UUID"}))),
                },
                None => return (StatusCode::BAD_REQUEST, Json(json!({"error": "target_id required for agent kill"}))),
            };
            KillTarget::Agent(id)
        }
        "mode" => {
            let mode_id = req.target_id.unwrap_or_default();
            KillTarget::Mode(ModeId::new(&mode_id))
        }
        "cluster" => KillTarget::Cluster,
        _ => return (StatusCode::BAD_REQUEST, Json(json!({"error": "Invalid target_type, use: agent, mode, cluster"}))),
    };

    let trigger = match req.trigger_type.as_str() {
        "manual" => KillTrigger::Manual { user: "api-user".to_string() },
        "automated" => KillTrigger::AutoQuarantine { anomaly_score: 0.95 },
        "emergency" => KillTrigger::AutoQuarantine { anomaly_score: 1.0 },
        _ => return (StatusCode::BAD_REQUEST, Json(json!({"error": "Invalid trigger_type, use: manual, automated, emergency"}))),
    };

    match nexora.trigger_kill_switch(target, &req.reason, trigger) {
        Ok(event) => (StatusCode::OK, Json(json!({
            "status": "kill_triggered",
            "event_id": event.id,
            "target": format!("{:?}", event.target),
            "reason": event.reason,
            "trigger": format!("{:?}", event.triggered_by),
            "timestamp": event.timestamp,
        }))),
        Err(e) => (StatusCode::FORBIDDEN, Json(json!({
            "error": e.to_string(),
        }))),
    }
}
