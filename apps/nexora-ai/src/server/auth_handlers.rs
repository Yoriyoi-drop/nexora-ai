use axum::{
    extract::Path,
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use std::collections::HashMap;

use super::router::ApiKey;
use crate::auth::user::{LoginRequest, RegisterRequest, UserProfile};
use crate::auth::apikey::CreateApiKeyRequest;
use crate::auth::AuthSystem;
use crate::NexoraAI;

type JsonResp = Json<serde_json::Value>;

fn ok_json(data: serde_json::Value) -> JsonResp {
    Json(serde_json::json!({ "success": true, "data": data }))
}

fn err_json(msg: &str) -> JsonResp {
    Json(serde_json::json!({ "success": false, "error": msg }))
}

pub async fn register_handler(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Extension(auth): Extension<Arc<AuthSystem>>,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    if req.email.trim().is_empty() || req.password.trim().is_empty() || req.username.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, err_json("email, username, and password are required")).into_response();
    }
    let result = auth.register(&req).await;
    let mut meta = HashMap::new();
    meta.insert("email".to_string(), req.email.clone());
    match result {
        Ok(ref value) => {
            nexora.telemetry.recorder.record_event("auth.register", Some(req.email.clone()), meta).await;
            (StatusCode::CREATED, ok_json(serde_json::to_value(value).unwrap_or_default())).into_response()
        }
        Err(e) => {
            meta.insert("error".to_string(), e.to_string());
            nexora.telemetry.recorder.record_event("auth.register.failed", Some(req.email), meta).await;
            let status = match &e {
                crate::auth::user::AuthError::EmailAlreadyTaken => StatusCode::CONFLICT,
                crate::auth::user::AuthError::UsernameAlreadyTaken => StatusCode::CONFLICT,
                crate::auth::user::AuthError::PasswordHashError(_) => StatusCode::INTERNAL_SERVER_ERROR,
                _ => StatusCode::BAD_REQUEST,
            };
            (status, err_json(&e.to_string())).into_response()
        }
    }
}

pub async fn login_handler(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Extension(auth): Extension<Arc<AuthSystem>>,
    Json(req): Json<LoginRequest>,
) -> impl IntoResponse {
    if req.email.trim().is_empty() || req.password.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, err_json("email and password are required")).into_response();
    }
    match auth.login(&req.email, &req.password).await {
        Ok(resp) => {
            nexora.telemetry.recorder.record_login(&req.email, true).await;
            (StatusCode::OK, ok_json(serde_json::to_value(resp).unwrap_or_default())).into_response()
        }
        Err(_) => {
            nexora.telemetry.recorder.record_login(&req.email, false).await;
            (StatusCode::UNAUTHORIZED, err_json("Invalid email or password")).into_response()
        }
    }
}

pub async fn get_profile_handler(
    Extension(auth): Extension<Arc<AuthSystem>>,
    Extension(api_key): Extension<ApiKey>,
) -> impl IntoResponse {
    let user = auth.authenticate_api_key(&api_key.0).await;
    match user {
        Some(profile) => (StatusCode::OK, ok_json(serde_json::to_value(profile).unwrap_or_default())).into_response(),
        None => (StatusCode::UNAUTHORIZED, err_json("Invalid API key")).into_response(),
    }
}

pub async fn update_profile_handler(
    Extension(auth): Extension<Arc<AuthSystem>>,
    Extension(api_key): Extension<ApiKey>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let user = match auth.authenticate_api_key(&api_key.0).await {
        Some(u) => u,
        None => return (StatusCode::UNAUTHORIZED, err_json("Invalid API key")).into_response(),
    };
    if let Some(tier) = body.get("tier").and_then(|v| v.as_str()) {
        if auth.users.update_tier(&user.id, tier).await.is_err() {
            return (StatusCode::NOT_FOUND, err_json("User not found")).into_response();
        }
    }
    let updated = auth.users.get_user_by_id(&user.id).await.map(UserProfile::from);
    match updated {
        Some(profile) => (StatusCode::OK, ok_json(serde_json::to_value(profile).unwrap_or_default())).into_response(),
        None => (StatusCode::NOT_FOUND, err_json("User not found")).into_response(),
    }
}

pub async fn list_api_keys_handler(
    Extension(auth): Extension<Arc<AuthSystem>>,
    Extension(api_key): Extension<ApiKey>,
) -> impl IntoResponse {
    let user = match auth.authenticate_api_key(&api_key.0).await {
        Some(u) => u,
        None => return (StatusCode::UNAUTHORIZED, err_json("Invalid API key")).into_response(),
    };
    let keys = auth.api_keys.list_keys(&user.id).await;
    (StatusCode::OK, ok_json(serde_json::to_value(keys).unwrap_or_default())).into_response()
}

pub async fn create_api_key_handler(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Extension(auth): Extension<Arc<AuthSystem>>,
    Extension(api_key): Extension<ApiKey>,
    Json(req): Json<CreateApiKeyRequest>,
) -> impl IntoResponse {
    let user = match auth.authenticate_api_key(&api_key.0).await {
        Some(u) => u,
        None => return (StatusCode::UNAUTHORIZED, err_json("Invalid API key")).into_response(),
    };
    if req.name.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, err_json("name is required")).into_response();
    }
    let record = auth.api_keys.create_key(&user.id, &req.name).await;
    let mut meta = HashMap::new();
    meta.insert("key_name".to_string(), req.name);
    nexora.telemetry.recorder.record_event("auth.apikey.created", Some(user.id.clone()), meta).await;
    (StatusCode::CREATED, ok_json(serde_json::to_value(record).unwrap_or_default())).into_response()
}

pub async fn revoke_api_key_handler(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Extension(auth): Extension<Arc<AuthSystem>>,
    Extension(api_key): Extension<ApiKey>,
    Path(key_id): Path<String>,
) -> impl IntoResponse {
    let user = match auth.authenticate_api_key(&api_key.0).await {
        Some(u) => u,
        None => return (StatusCode::UNAUTHORIZED, err_json("Invalid API key")).into_response(),
    };
    if auth.api_keys.revoke_key(&key_id, &user.id).await {
        let mut meta = HashMap::new();
        meta.insert("key_id".to_string(), key_id);
        nexora.telemetry.recorder.record_event("auth.apikey.revoked", Some(user.id), meta).await;
        (StatusCode::OK, ok_json(serde_json::json!("Key revoked"))).into_response()
    } else {
        (StatusCode::NOT_FOUND, err_json("Key not found")).into_response()
    }
}

pub async fn rotate_api_key_handler(
    Extension(auth): Extension<Arc<AuthSystem>>,
    Extension(api_key): Extension<ApiKey>,
    Path(key_id): Path<String>,
) -> impl IntoResponse {
    let user = match auth.authenticate_api_key(&api_key.0).await {
        Some(u) => u,
        None => return (StatusCode::UNAUTHORIZED, err_json("Invalid API key")).into_response(),
    };
    match auth.api_keys.rotate_key(&key_id, &user.id).await {
        Some(record) => (StatusCode::OK, ok_json(serde_json::to_value(record).unwrap_or_default())).into_response(),
        None => (StatusCode::NOT_FOUND, err_json("Key not found")).into_response(),
    }
}
