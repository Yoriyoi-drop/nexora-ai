use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::security::SecurityUtils;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub email: String,
    pub username: String,
    #[serde(skip_serializing)]
    pub password_hash: String,
    pub tier: String,
    pub is_active: bool,
    pub is_verified: bool,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub id: String,
    pub email: String,
    pub username: String,
    pub tier: String,
    pub is_active: bool,
    pub is_verified: bool,
    pub created_at: u64,
}

impl From<User> for UserProfile {
    fn from(u: User) -> Self {
        UserProfile {
            id: u.id,
            email: u.email,
            username: u.username,
            tier: u.tier,
            is_active: u.is_active,
            is_verified: u.is_verified,
            created_at: u.created_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub email: String,
    pub username: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginResponse {
    pub token: String,
    pub user: UserProfile,
    pub api_key: String,
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("User already exists: {0}")]
    UserAlreadyExists(String),
    #[error("User not found")]
    UserNotFound,
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("Account is disabled")]
    AccountDisabled,
    #[error("Email already taken")]
    EmailAlreadyTaken,
    #[error("Username already taken")]
    UsernameAlreadyTaken,
    #[error("Password hashing failed: {0}")]
    PasswordHashError(String),
}

#[async_trait::async_trait]
pub trait UserStore: Send + Sync {
    async fn create_user(&self, req: &RegisterRequest) -> Result<User, AuthError>;
    async fn get_user_by_id(&self, id: &str) -> Option<User>;
    async fn get_user_by_email(&self, email: &str) -> Option<User>;
    async fn get_user_by_api_key(&self, api_key: &str) -> Option<User>;
    async fn verify_password(&self, email: &str, password: &str) -> Result<User, AuthError>;
    async fn update_tier(&self, user_id: &str, tier: &str) -> Result<(), AuthError>;
    async fn list_users(&self) -> Vec<UserProfile>;
    async fn seed_default_admin(&self) -> Result<(), AuthError> { Ok(()) }
}

pub struct InMemoryUserStore {
    users_by_id: Arc<RwLock<HashMap<String, User>>>,
    users_by_email: Arc<RwLock<HashMap<String, String>>>,
    api_key_to_user: Arc<RwLock<HashMap<String, String>>>,
}

impl InMemoryUserStore {
    pub fn new() -> Self {
        Self {
            users_by_id: Arc::new(RwLock::new(HashMap::new())),
            users_by_email: Arc::new(RwLock::new(HashMap::new())),
            api_key_to_user: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn seed_default_admin(&self) -> Result<(), AuthError> {
        if self.users_by_email.read().await.contains_key("admin@nexora.ai") {
            return Ok(());
        }
        let hash = SecurityUtils::hash_password("admin")
            .map_err(|e| AuthError::PasswordHashError(e.to_string()))?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let user = User {
            id: Uuid::new_v4().to_string(),
            email: "admin@nexora.ai".to_string(),
            username: "admin".to_string(),
            password_hash: hash,
            tier: "enterprise".to_string(),
            is_active: true,
            is_verified: true,
            created_at: now,
            updated_at: now,
        };
        let api_key = SecurityUtils::generate_secure_token(48);
        let id = user.id.clone();
        let email = user.email.clone();
        self.users_by_id.write().await.insert(id.clone(), user);
        self.users_by_email.write().await.insert(email, id.clone());
        self.api_key_to_user.write().await.insert(api_key, id);
        Ok(())
    }
}

#[async_trait::async_trait]
impl UserStore for InMemoryUserStore {
    async fn create_user(&self, req: &RegisterRequest) -> Result<User, AuthError> {
        let users_by_email = self.users_by_email.read().await;
        if users_by_email.contains_key(&req.email.to_lowercase()) {
            return Err(AuthError::EmailAlreadyTaken);
        }
        drop(users_by_email);

        let hash = SecurityUtils::hash_password(&req.password)
            .map_err(|e| AuthError::PasswordHashError(e.to_string()))?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let user = User {
            id: Uuid::new_v4().to_string(),
            email: req.email.to_lowercase(),
            username: req.username.clone(),
            password_hash: hash,
            tier: "free".to_string(),
            is_active: true,
            is_verified: false,
            created_at: now,
            updated_at: now,
        };
        let api_key = SecurityUtils::generate_secure_token(48);
        let id = user.id.clone();
        let email = user.email.clone();
        let lookup_id = id.clone();
        self.users_by_id.write().await.insert(id.clone(), user);
        self.users_by_email.write().await.insert(email, id);
        self.api_key_to_user.write().await.insert(api_key, lookup_id.clone());
        Ok(self.users_by_id.read().await.get(&lookup_id).cloned().unwrap())
    }

    async fn get_user_by_id(&self, id: &str) -> Option<User> {
        self.users_by_id.read().await.get(id).cloned()
    }

    async fn get_user_by_email(&self, email: &str) -> Option<User> {
        let users_by_email = self.users_by_email.read().await;
        let id = users_by_email.get(&email.to_lowercase())?;
        self.users_by_id.read().await.get(id).cloned()
    }

    async fn get_user_by_api_key(&self, api_key: &str) -> Option<User> {
        let map = self.api_key_to_user.read().await;
        let id = map.get(api_key)?;
        self.users_by_id.read().await.get(id).cloned()
    }

    async fn verify_password(&self, email: &str, password: &str) -> Result<User, AuthError> {
        let user = self.get_user_by_email(email).await.ok_or(AuthError::InvalidCredentials)?;
        if !user.is_active {
            return Err(AuthError::AccountDisabled);
        }
        let ok = SecurityUtils::verify_password(password, &user.password_hash)
            .map_err(|e| AuthError::PasswordHashError(e.to_string()))?;
        if !ok {
            return Err(AuthError::InvalidCredentials);
        }
        Ok(user)
    }

    async fn update_tier(&self, user_id: &str, tier: &str) -> Result<(), AuthError> {
        let mut users = self.users_by_id.write().await;
        let user = users.get_mut(user_id).ok_or(AuthError::UserNotFound)?;
        user.tier = tier.to_string();
        Ok(())
    }

    async fn list_users(&self) -> Vec<UserProfile> {
        self.users_by_id.read().await.values().cloned().map(|u| UserProfile::from(u)).collect()
    }
}
