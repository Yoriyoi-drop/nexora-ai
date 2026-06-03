use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::security::SecurityUtils;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyRecord {
    pub id: String,
    pub user_id: String,
    pub key: String,
    pub name: String,
    pub is_active: bool,
    pub created_at: u64,
    pub last_used_at: Option<u64>,
    pub expires_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyResponse {
    pub id: String,
    pub name: String,
    pub key_preview: String,
    pub is_active: bool,
    pub created_at: u64,
    pub last_used_at: Option<u64>,
    pub expires_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedStore {
    keys_by_user: HashMap<String, Vec<ApiKeyRecord>>,
    keys_by_value: HashMap<String, String>,
}

#[async_trait::async_trait]
pub trait ApiKeyStore: Send + Sync {
    async fn create_key(&self, user_id: &str, name: &str) -> ApiKeyRecord;
    async fn get_key(&self, key: &str) -> Option<ApiKeyRecord>;
    async fn list_keys(&self, user_id: &str) -> Vec<ApiKeyResponse>;
    async fn revoke_key(&self, key_id: &str, user_id: &str) -> bool;
    async fn rotate_key(&self, key_id: &str, user_id: &str) -> Option<ApiKeyRecord>;
    async fn touch_key(&self, key: &str);
}

pub struct InMemoryApiKeyStore {
    keys_by_user: Arc<RwLock<HashMap<String, Vec<ApiKeyRecord>>>>,
    keys_by_value: Arc<RwLock<HashMap<String, String>>>,
    persist_path: Option<PathBuf>,
}

impl InMemoryApiKeyStore {
    pub fn new() -> Self {
        Self {
            keys_by_user: Arc::new(RwLock::new(HashMap::new())),
            keys_by_value: Arc::new(RwLock::new(HashMap::new())),
            persist_path: None,
        }
    }

    pub fn with_persistence(path: PathBuf) -> Self {
        let store = Self::new();
        let _ = store.load_from_file(&path);
        Self {
            keys_by_user: store.keys_by_user,
            keys_by_value: store.keys_by_value,
            persist_path: Some(path),
        }
    }

    fn persist_path(&self) -> Option<&PathBuf> {
        self.persist_path.as_ref()
    }

    async fn persist(&self) {
        let path = match self.persist_path() {
            Some(p) => p.clone(),
            None => return,
        };
        let keys_by_user = self.keys_by_user.read().await;
        let keys_by_value = self.keys_by_value.read().await;
        let persisted = PersistedStore {
            keys_by_user: keys_by_user.clone(),
            keys_by_value: keys_by_value.clone(),
        };
        if let Ok(json) = serde_json::to_string_pretty(&persisted) {
            let _ = std::fs::write(&path, &json);
        }
    }

    fn load_from_file(&self, path: &PathBuf) {
        if !path.exists() {
            return;
        }
        if let Ok(json) = std::fs::read_to_string(path) {
            if let Ok(persisted) = serde_json::from_str::<PersistedStore>(&json) {
                if let Ok(mut u) = self.keys_by_user.try_write() {
                    *u = persisted.keys_by_user;
                }
                if let Ok(mut v) = self.keys_by_value.try_write() {
                    *v = persisted.keys_by_value;
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl ApiKeyStore for InMemoryApiKeyStore {
    async fn create_key(&self, user_id: &str, name: &str) -> ApiKeyRecord {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let raw_key = format!("nxr_{}", SecurityUtils::generate_secure_token(44));
        let id = Uuid::new_v4().to_string();
        let record = ApiKeyRecord {
            id: id.clone(),
            user_id: user_id.to_string(),
            key: raw_key.clone(),
            name: name.to_string(),
            is_active: true,
            created_at: now,
            last_used_at: None,
            expires_at: None,
        };
        self.keys_by_user
            .write()
            .await
            .entry(user_id.to_string())
            .or_default()
            .push(record.clone());
        self.keys_by_value
            .write()
            .await
            .insert(raw_key, id);
        self.persist().await;
        record
    }

    async fn get_key(&self, key: &str) -> Option<ApiKeyRecord> {
        let id = {
            let key_map = self.keys_by_value.read().await;
            key_map.get(key).cloned()?
        };
        let users = self.keys_by_user.read().await;
        for records in users.values() {
            for r in records {
                if r.id == id {
                    return Some(r.clone());
                }
            }
        }
        None
    }

    async fn list_keys(&self, user_id: &str) -> Vec<ApiKeyResponse> {
        self.keys_by_user
            .read()
            .await
            .get(user_id)
            .map(|keys| {
                keys.iter()
                    .map(|k| ApiKeyResponse {
                        id: k.id.clone(),
                        name: k.name.clone(),
                        key_preview: format!("{}...{}", &k.key[..8], &k.key[k.key.len()-4..]),
                        is_active: k.is_active,
                        created_at: k.created_at,
                        last_used_at: k.last_used_at,
                        expires_at: k.expires_at,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    async fn revoke_key(&self, key_id: &str, user_id: &str) -> bool {
        let mut users = self.keys_by_user.write().await;
        if let Some(keys) = users.get_mut(user_id) {
            if let Some(record) = keys.iter_mut().find(|k| k.id == key_id) {
                record.is_active = false;
                self.persist().await;
                return true;
            }
        }
        false
    }

    async fn rotate_key(&self, key_id: &str, user_id: &str) -> Option<ApiKeyRecord> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let new_raw = format!("nxr_{}", SecurityUtils::generate_secure_token(44));

        let mut users = self.keys_by_user.write().await;
        let keys = users.get_mut(user_id)?;
        let record = keys.iter_mut().find(|k| k.id == key_id)?;
        record.key = new_raw.clone();
        record.last_used_at = Some(now);

        let mut key_map = self.keys_by_value.write().await;
        key_map.insert(new_raw, record.id.clone());
        self.persist().await;

        Some(record.clone())
    }

    async fn touch_key(&self, key: &str) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let id = {
            let key_map = self.keys_by_value.read().await;
            key_map.get(key).cloned()
        };
        let id = match id {
            Some(id) => id,
            None => return,
        };
        let mut users = self.keys_by_user.write().await;
        for records in users.values_mut() {
            for r in records.iter_mut() {
                if r.id == id {
                    r.last_used_at = Some(now);
                    return;
                }
            }
        }
    }
}
