//! Server configuration and setup

use anyhow::Result;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub enable_tls: bool,
    pub cert_path: Option<String>,
    pub key_path: Option<String>,
    pub max_connections: usize,
    pub request_timeout_seconds: u64,
    pub enable_cors: bool,
    pub cors_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            enable_tls: false,
            cert_path: None,
            key_path: None,
            max_connections: 1000,
            request_timeout_seconds: 30,
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
        }
    }
}

/// Load Rustls PEM file for TLS certificate
pub fn load_rustls_pem_file(cert_path: &str) -> Result<rustls::Certificate> {
    let cert_file = std::fs::File::open(cert_path)?;
    let mut reader = std::io::BufReader::new(cert_file);
    let certs: Vec<rustls::Certificate> = rustls_pemfile::certs(&mut reader)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|cert| rustls::Certificate(cert.to_owned().to_vec()))
        .collect();
    
    if certs.is_empty() {
        return Err(anyhow::anyhow!("No certificates found in file: {}", cert_path));
    }
    
    certs.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("Failed to extract certificate from parsed certificates"))
}

/// Load Rustls private key for TLS
pub fn load_rustls_private_key(key_path: &str) -> Result<rustls::PrivateKey> {
    let key_file = std::fs::File::open(key_path)?;
    let mut reader = std::io::BufReader::new(key_file);
    
    // Try to load as PKCS8 first
    let keys = rustls_pemfile::pkcs8_private_keys(&mut reader)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(key) = keys.into_iter().next() {
        return Ok(rustls::PrivateKey(key.secret_pkcs8_der().to_vec()));
    }
    
    // Reset reader and try as RSA key
    let key_file = std::fs::File::open(key_path)?;
    let mut reader = std::io::BufReader::new(key_file);
    
    let keys = rustls_pemfile::rsa_private_keys(&mut reader)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(key) = keys.into_iter().next() {
        return Ok(rustls::PrivateKey(key.secret_pkcs1_der().to_vec()));
    }
    
    Err(anyhow::anyhow!("No valid private key found in file: {}", key_path))
}
