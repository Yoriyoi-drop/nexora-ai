use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub email: String,
    pub tier: String,
    pub jti: String,
    pub exp: u64,
    pub iat: u64,
}

pub struct JwtManager {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    algorithm: Algorithm,
}

impl JwtManager {
    pub fn new_rs256(private_key_pem: &str, public_key_pem: &str) -> Result<Self, String> {
        let encoding_key = EncodingKey::from_rsa_pem(private_key_pem.as_bytes())
            .map_err(|e| format!("Failed to load RSA private key: {}", e))?;
        let decoding_key = DecodingKey::from_rsa_pem(public_key_pem.as_bytes())
            .map_err(|e| format!("Failed to load RSA public key: {}", e))?;
        Ok(Self { encoding_key, decoding_key, algorithm: Algorithm::RS256 })
    }

    pub fn new_hs256(secret: &str) -> Self {
        Self {
            encoding_key: EncodingKey::from_secret(secret.as_bytes()),
            decoding_key: DecodingKey::from_secret(secret.as_bytes()),
            algorithm: Algorithm::HS256,
        }
    }

    pub fn algorithm_name(&self) -> &str {
        match self.algorithm {
            Algorithm::RS256 => "RS256",
            Algorithm::HS256 => "HS256",
            _ => "unknown",
        }
    }

    pub fn create_token(&self, user_id: &str, email: &str, tier: &str, ttl_secs: u64) -> Result<String, String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs();
        let jti = uuid::Uuid::new_v4().to_string();
        let claims = Claims {
            sub: user_id.to_string(),
            email: email.to_string(),
            tier: tier.to_string(),
            jti,
            exp: now + ttl_secs,
            iat: now,
        };
        let mut header = Header::new(self.algorithm);
        header.kid = Some(self.algorithm_name().to_string());
        encode(&header, &claims, &self.encoding_key)
            .map_err(|e| e.to_string())
    }

    pub fn validate_token(&self, token: &str) -> Result<Claims, String> {
        let mut validation = Validation::new(self.algorithm);
        validation.validate_exp = true;
        let token_data = decode::<Claims>(token, &self.decoding_key, &validation)
            .map_err(|e| e.to_string())?;
        Ok(token_data.claims)
    }
}
