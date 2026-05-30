pub mod apikey;
pub mod jwt;
pub mod user;

use std::sync::Arc;

use apikey::{ApiKeyStore, InMemoryApiKeyStore};
use jwt::JwtManager;
use user::{InMemoryUserStore, LoginResponse, RegisterRequest, UserProfile, UserStore};

pub struct AuthSystem {
    pub users: Arc<dyn UserStore>,
    pub api_keys: Arc<dyn ApiKeyStore>,
    pub jwt: JwtManager,
}

impl AuthSystem {
    pub fn new(jwt_secret: &str) -> Self {
        Self {
            users: Arc::new(InMemoryUserStore::new()),
            api_keys: Arc::new(InMemoryApiKeyStore::new()),
            jwt: JwtManager::new(jwt_secret),
        }
    }

    pub async fn register(&self, req: &RegisterRequest) -> Result<LoginResponse, user::AuthError> {
        let user = self.users.create_user(req).await?;
        let api_key_record = self.api_keys.create_key(&user.id, "default").await;
        let token = self.jwt.create_token(&user.id, &user.email, &user.tier, 86400)
            .map_err(|e| user::AuthError::PasswordHashError(e))?;
        Ok(LoginResponse {
            token,
            user: UserProfile::from(user),
            api_key: api_key_record.key,
        })
    }

    pub async fn login(&self, email: &str, password: &str) -> Result<LoginResponse, user::AuthError> {
        let user = self.users.verify_password(email, password).await?;
        let token = self.jwt.create_token(&user.id, &user.email, &user.tier, 86400)
            .map_err(|e| user::AuthError::PasswordHashError(e))?;
        let record = self.api_keys.create_key(&user.id, "login-session").await;
        Ok(LoginResponse {
            token,
            user: UserProfile::from(user),
            api_key: record.key,
        })
    }

    pub async fn authenticate_bearer(&self, token: &str) -> Option<UserProfile> {
        let claims = self.jwt.validate_token(token).ok()?;
        let user = self.users.get_user_by_id(&claims.sub).await?;
        if !user.is_active {
            return None;
        }
        Some(UserProfile::from(user))
    }

    pub async fn authenticate_api_key(&self, api_key: &str) -> Option<UserProfile> {
        // Check new key store
        if let Some(record) = self.api_keys.get_key(api_key).await {
            if !record.is_active {
                return None;
            }
            if let Some(exp) = record.expires_at {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if now > exp {
                    return None;
                }
            }
            self.api_keys.touch_key(api_key).await;
            let user = self.users.get_user_by_id(&record.user_id).await?;
            if !user.is_active {
                return None;
            }
            return Some(UserProfile::from(user));
        }
        // Fallback: check legacy user.api_key field
        let user = self.users.get_user_by_api_key(api_key).await?;
        if !user.is_active {
            return None;
        }
        Some(UserProfile::from(user))
    }
}
