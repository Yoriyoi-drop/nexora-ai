use axum::http::StatusCode;
use axum::{Extension, Json};
use std::sync::Arc;

use nexora_runtime::cluster::GossipMessage;

use crate::NexoraAI;

pub async fn gossip_push_handler(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(msg): Json<GossipMessage>,
) -> Result<Json<GossipMessage>, (StatusCode, String)> {
    let gp = nexora
        .gossip_protocol
        .as_ref()
        .ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Distributed mode not enabled".to_string(),
        ))?;
    let response = gp
        .handle_push(msg)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Gossip push failed: {}", e)))?;
    Ok(Json(response))
}

pub async fn gossip_pull_handler(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(msg): Json<GossipMessage>,
) -> Result<Json<GossipMessage>, (StatusCode, String)> {
    let gp = nexora
        .gossip_protocol
        .as_ref()
        .ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Distributed mode not enabled".to_string(),
        ))?;
    let response = gp
        .handle_pull(msg)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Gossip pull failed: {}", e)))?;
    Ok(Json(response))
}
