//! Cryptographic utilities untuk Nexora

use sha2::{Sha256, Sha512, Digest};
use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use uuid::Uuid;

pub struct CryptoUtils;

impl CryptoUtils {
    /// Generate SHA-256 hash
    pub fn sha256(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        hex::encode(result)
    }
    
    /// Generate SHA-512 hash
    pub fn sha512(data: &[u8]) -> String {
        let mut hasher = Sha512::new();
        hasher.update(data);
        let result = hasher.finalize();
        hex::encode(result)
    }
    
    /// Generate hash (alias for SHA-256)
    pub fn hash(data: &[u8]) -> Result<String> {
        Ok(Self::sha256(data))
    }
    
    /// Generate hash from string
    pub fn hash_string(text: &str) -> Result<String> {
        Ok(Self::sha256(text.as_bytes()))
    }
    
    /// Generate HMAC-SHA256
    pub fn hmac_sha256(data: &[u8], key: &[u8]) -> Result<String> {
        use hmac::{Hmac, Mac};
        type HmacSha256 = Hmac<Sha256>;
        
        let mut mac = HmacSha256::new_from_slice(key)?;
        mac.update(data);
        let result = mac.finalize();
        Ok(hex::encode(result.into_bytes()))
    }
    
    /// Generate HMAC-SHA512
    pub fn hmac_sha512(data: &[u8], key: &[u8]) -> Result<String> {
        use hmac::{Hmac, Mac};
        type HmacSha512 = Hmac<Sha512>;
        
        let mut mac = HmacSha512::new_from_slice(key)?;
        mac.update(data);
        let result = mac.finalize();
        Ok(hex::encode(result.into_bytes()))
    }
    
    /// Generate random UUID v4
    pub fn generate_uuid() -> String {
        Uuid::new_v4().to_string()
    }
    
    /// Generate UUID v5 (namespace + name)
    pub fn generate_uuid_v5(_namespace: &str, _name: &str) -> Result<String> {
        // For simplicity, just generate a v4 UUID for now
        // In a real implementation, you'd use proper v5 generation
        let uuid = Uuid::new_v4();
        Ok(uuid.to_string())
    }
    
    /// Generate random bytes
    pub fn random_bytes(length: usize) -> Vec<u8> {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        let mut bytes = vec![0u8; length];
        rng.fill_bytes(&mut bytes);
        bytes
    }
    
    /// Generate random hex string
    pub fn random_hex(length: usize) -> String {
        let bytes = Self::random_bytes(length);
        hex::encode(bytes)
    }
    
    /// Generate random base64 string
    pub fn random_base64(length: usize) -> String {
        let bytes = Self::random_bytes(length);
        general_purpose::STANDARD.encode(bytes)
    }
    
    /// Generate random alphanumeric string
    pub fn random_alphanumeric(length: usize) -> String {
        use rand::Rng;
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                 abcdefghijklmnopqrstuvwxyz\
                                 0123456789";
        let mut rng = rand::thread_rng();
        (0..length)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }
    
    /// Encode data to base64
    pub fn base64_encode(data: &[u8]) -> String {
        general_purpose::STANDARD.encode(data)
    }
    
    /// Decode base64 string
    pub fn base64_decode(encoded: &str) -> Result<Vec<u8>> {
        general_purpose::STANDARD.decode(encoded)
            .map_err(|e| anyhow::anyhow!("Base64 decode error: {}", e))
    }
    
    /// Encode URL-safe base64
    pub fn base64_url_encode(data: &[u8]) -> String {
        general_purpose::URL_SAFE.encode(data)
    }
    
    /// Decode URL-safe base64 string
    pub fn base64_url_decode(encoded: &str) -> Result<Vec<u8>> {
        general_purpose::URL_SAFE.decode(encoded)
            .map_err(|e| anyhow::anyhow!("Base64 URL decode error: {}", e))
    }
    
    /// Generate password hash with salt
    /// 
    /// WARNING: Uses SHA-256 which is NOT suitable for password hashing.
    /// Use Argon2id from `apps/nexora-ai/src/security/mod.rs` instead.
    #[deprecated(note = "Use Argon2id from nexora-ai security module instead")]
    pub fn hash_password(password: &str, salt: Option<&str>) -> Result<String> {
        tracing::warn!("hash_password uses weak SHA-256 scheme. Migrate to Argon2id.");
        let default_salt = Self::generate_uuid();
        let salt = salt.unwrap_or(&default_salt);
        let combined = format!("{}:{}", password, salt);
        let hash = Self::sha256(combined.as_bytes());
        Ok(format!("{}:{}", hash, salt))
    }
    
    /// Verify password hash
    #[deprecated(note = "Use Argon2id from nexora-ai security module instead")]
    pub fn verify_password(password: &str, hash_with_salt: &str) -> Result<bool> {
        tracing::warn!("verify_password uses weak SHA-256 scheme. Migrate to Argon2id.");
        let parts: Vec<&str> = hash_with_salt.split(':').collect();
        if parts.len() != 2 {
            return Ok(false);
        }
        
        let stored_hash = parts[0];
        let salt = parts[1];
        let computed_hash = Self::hash_password(password, Some(salt))?;
        let computed_parts: Vec<&str> = computed_hash.split(':').collect();
        
        Ok(computed_parts[0] == stored_hash)
    }
    
    /// Generate API key
    pub fn generate_api_key(prefix: Option<&str>) -> String {
        let random_part = Self::random_alphanumeric(32);
        match prefix {
            Some(p) => format!("{}_{}", p, random_part),
            None => random_part,
        }
    }
    
    /// Generate token
    pub fn generate_token(length: usize) -> String {
        Self::random_alphanumeric(length)
    }
    
    /// Generate secure session ID
    pub fn generate_session_id() -> String {
        Self::random_alphanumeric(64)
    }
    
    /// Generate CSRF token
    pub fn generate_csrf_token() -> String {
        Self::random_alphanumeric(32)
    }
    
    /// Generate nonce
    pub fn generate_nonce(length: usize) -> String {
        Self::random_alphanumeric(length)
    }
    
    /// Generate fingerprint for data
    pub fn fingerprint(data: &[u8]) -> String {
        Self::sha256(data)
    }
    
    /// Generate fingerprint for string
    pub fn fingerprint_string(text: &str) -> String {
        Self::fingerprint(text.as_bytes())
    }
    
    /// Compare two hashes securely (timing attack resistant)
    pub fn secure_compare(hash1: &str, hash2: &str) -> bool {
        if hash1.len() != hash2.len() {
            return false;
        }
        
        let bytes1 = hash1.as_bytes();
        let bytes2 = hash2.as_bytes();
        
        let mut result = 0u8;
        for (a, b) in bytes1.iter().zip(bytes2.iter()) {
            result |= a ^ b;
        }
        
        result == 0
    }
    
    /// Generate pepper for additional security
    pub fn generate_pepper() -> String {
        Self::random_alphanumeric(64)
    }
    
    /// Derive key from password using PBKDF2
    pub fn derive_key_pbkdf2(password: &str, salt: &[u8], iterations: u32, key_len: usize) -> Result<Vec<u8>> {
        use pbkdf2::pbkdf2_hmac;
        
        
        let mut key = vec![0u8; key_len];
        pbkdf2_hmac::<Sha256>(
            password.as_bytes(),
            salt,
            iterations,
            &mut key,
        );
        Ok(key)
    }
    
    /// Generate key from master key
    pub fn derive_key(master_key: &[u8], context: &str) -> Result<Vec<u8>> {
        let combined = format!("{}:{}", hex::encode(master_key), context);
        Ok(Self::sha256(combined.as_bytes()).into_bytes())
    }
    
    /// Generate deterministic ID from data
    pub fn deterministic_id(data: &[u8]) -> String {
        let hash = Self::sha256(data);
        format!("id_{}", &hash[..16])
    }
    
    /// Generate deterministic ID from string
    pub fn deterministic_id_string(text: &str) -> String {
        Self::deterministic_id(text.as_bytes())
    }
    
    /// Generate checksum for data
    pub fn checksum(data: &[u8]) -> String {
        Self::sha256(data)[..8].to_string()
    }
    
    /// Verify checksum
    pub fn verify_checksum(data: &[u8], expected_checksum: &str) -> bool {
        let computed_checksum = Self::checksum(data);
        Self::secure_compare(&computed_checksum, expected_checksum)
    }
    
    /// Generate salt
    pub fn generate_salt(length: usize) -> String {
        Self::random_alphanumeric(length)
    }
    
    /// Generate pepper
    pub fn generate_pepper_fixed() -> String {
        Self::random_alphanumeric(128)
    }
    
    /// Generate secure random number in range
    pub fn secure_random_range(min: u64, max: u64) -> u64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(min..=max)
    }
    
    /// Generate secure random boolean
    pub fn secure_random_bool() -> bool {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_bool(0.5)
    }
    
    /// Generate secure random choice from slice
    pub fn secure_random_choice<T: Clone>(choices: &[T]) -> Option<T> {
        if choices.is_empty() {
            None
        } else {
            let index = Self::secure_random_range(0, (choices.len() - 1) as u64) as usize;
            Some(choices[index].clone())
        }
    }
    
    /// Generate secure random permutation
    pub fn secure_random_permutation<T: Clone>(items: &[T]) -> Vec<T> {
        let mut result = items.to_vec();
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        result.shuffle(&mut rng);
        result
    }
    
    /// Generate secure random sample without replacement
    pub fn secure_random_sample<T: Clone>(items: &[T], sample_size: usize) -> Vec<T> {
        if sample_size >= items.len() {
            return Self::secure_random_permutation(items);
        }
        
        let mut indices: Vec<usize> = (0..items.len()).collect();
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        
        indices[..sample_size]
            .iter()
            .map(|&i| items[i].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crypto_utils() {
        // Test hash functions
        let data = b"hello world";
        let hash = CryptoUtils::sha256(data);
        assert_eq!(hash.len(), 64); // SHA-256 produces 64 hex chars
        
        // Test HMAC
        let key = b"secret_key";
        let hmac = CryptoUtils::hmac_sha256(data, key).unwrap();
        assert!(!hmac.is_empty());
        
        // Test UUID generation
        let uuid = CryptoUtils::generate_uuid();
        assert_eq!(uuid.len(), 36); // Standard UUID format
        
        // Test base64 encoding/decoding
        let encoded = CryptoUtils::base64_encode(data);
        let decoded = CryptoUtils::base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
        
        // Test password hashing
        let password = "test_password";
        let hash = CryptoUtils::hash_password(password, None).unwrap();
        let verified = CryptoUtils::verify_password(password, &hash).unwrap();
        assert!(verified);
        
        // Test secure compare
        assert!(CryptoUtils::secure_compare("hello", "hello"));
        assert!(!CryptoUtils::secure_compare("hello", "world"));
        
        // Test random generation
        let random_bytes = CryptoUtils::random_bytes(32);
        assert_eq!(random_bytes.len(), 32);
        
        let random_hex = CryptoUtils::random_hex(32);
        assert_eq!(random_hex.len(), 64); // 32 bytes = 64 hex chars
        
        // Test fingerprint
        let fingerprint1 = CryptoUtils::fingerprint(data);
        let fingerprint2 = CryptoUtils::fingerprint(data);
        assert_eq!(fingerprint1, fingerprint2);
        
        // Test checksum
        let checksum = CryptoUtils::checksum(data);
        assert_eq!(checksum.len(), 8);
        
        let verified = CryptoUtils::verify_checksum(data, &checksum);
        assert!(verified);
    }
}
