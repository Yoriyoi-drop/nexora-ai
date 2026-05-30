use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub email: String,
    pub tier: String,
    pub exp: u64,
    pub iat: u64,
}

pub struct JwtManager {
    secret: String,
}

impl JwtManager {
    pub fn new(secret: &str) -> Self {
        Self { secret: secret.to_string() }
    }

    pub fn create_token(&self, user_id: &str, email: &str, tier: &str, ttl_secs: u64) -> Result<String, String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs();
        let claims = Claims {
            sub: user_id.to_string(),
            email: email.to_string(),
            tier: tier.to_string(),
            exp: now + ttl_secs,
            iat: now,
        };
        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.secret.as_bytes()),
        )
        .map_err(|e| e.to_string())
    }

    pub fn validate_token(&self, token: &str) -> Result<Claims, String> {
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.secret.as_bytes()),
            &Validation::default(),
        )
        .map_err(|e| e.to_string())?;
        Ok(token_data.claims)
    }
}
