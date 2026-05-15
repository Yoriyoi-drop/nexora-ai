//! TLS server functionality

use std::net::SocketAddr;
use std::sync::Arc;
use anyhow::Result;
use tokio::net::TcpListener;
use axum::Router;
use tracing::{info, error, debug};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use axum::body::Body;
use axum::body::to_bytes;

use super::config::{ServerConfig, load_rustls_pem_file, load_rustls_private_key};

/// Start TLS server with secure connections
pub async fn start_tls_server(
    config: &ServerConfig,
    listener: TcpListener,
    app: Router,
) -> Result<()> {
    if let (Some(cert_path), Some(key_path)) = (&config.cert_path, &config.key_path) {
        let tls_config = rustls::ServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(
                vec![load_rustls_pem_file(cert_path)?],
                load_rustls_private_key(key_path)?,
            )?;
        
        let tls_acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(tls_config));
        
        let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
        info!("Starting TLS server on {}", addr);
        
        loop {
            match listener.accept().await {
                Ok((stream, remote_addr)) => {
                    info!("New TLS connection from: {}", remote_addr);
                    
                    let tls_acceptor = tls_acceptor.clone();
                    let app = app.clone();
                    
                    tokio::spawn(async move {
                        match tls_acceptor.accept(stream).await {
                            Ok(tls_stream) => {
                                if let Err(e) = handle_tls_stream(tls_stream, app).await {
                                    error!("Error handling TLS stream from {}: {}", remote_addr, e);
                                }
                            }
                            Err(e) => {
                                error!("TLS handshake failed from {}: {}", remote_addr, e);
                            }
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                    continue;
                }
            }
        }
    } else {
        Err(anyhow::anyhow!("TLS enabled but certificate or key path not provided"))
    }
}

/// Handle individual TLS stream
async fn handle_tls_stream(
    mut tls_stream: tokio_rustls::server::TlsStream<tokio::net::TcpStream>,
    _app: Router,
) -> Result<()> {
    // In a real implementation, this would handle HTTP over TLS
    // For now, we'll just log the connection
    let peer_addr = tls_stream.get_ref().0.peer_addr()?;
    info!("Handling TLS connection from: {}", peer_addr);
    
    // Simple TLS connection handling - just log and close
    info!("TLS connection established successfully");
    
    // Read a simple request and send response
    let mut buffer = [0u8; 1024];
    match tls_stream.read(&mut buffer).await {
        Ok(0) => {
            info!("Client closed connection");
        }
        Ok(n) => {
            let request = String::from_utf8_lossy(&buffer[..n]);
            info!("Received TLS request: {}", request.trim());
            
            // Send a simple HTTP response
            let response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nTLS Connection Successful";
            if let Err(e) = tls_stream.write_all(response.as_bytes()).await {
                error!("Failed to send TLS response: {}", e);
            }
        }
        Err(e) => {
            error!("Failed to read from TLS stream: {}", e);
        }
    }
    
    Ok(())
}

/// Handle individual HTTP requests over TLS
#[allow(dead_code)]
async fn handle_http_request(
    req: hyper::Request<Body>
) -> Result<hyper::Response<Body>, hyper::Error> {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();
    
    info!("HTTPS {} {} from TLS connection", method, uri);
    
    // Log request headers for debugging
    for (name, value) in headers.iter() {
        debug!("Header: {}: {:?}", name, value);
    }
    
    // Handle different endpoints
    match (method.as_str(), uri.path()) {
        ("GET", "/") => {
            let response_body = include_str!("../../static/index.html");
            Ok(hyper::Response::builder()
                .status(200)
                .header("content-type", "text/html")
                .body(Body::from(response_body))
                .unwrap())
        },
        ("GET", "/health") => {
            let health_response = serde_json::json!({
                "status": "healthy",
                "tls": "enabled",
                "timestamp": chrono::Utc::now().to_rfc3339()
            });
            Ok(hyper::Response::builder()
                .status(200)
                .header("content-type", "application/json")
                .body(Body::from(health_response.to_string()))
                .unwrap())
        },
        ("POST", "/process") => {
            // Read request body
            let body_bytes = match to_bytes(req.into_body(), 1024 * 1024).await {
                Ok(bytes) => bytes,
                Err(_) => return Ok(hyper::Response::builder()
                    .status(400)
                    .body(Body::from("Request body too large"))
                    .unwrap()),
            };
            let body_str = String::from_utf8_lossy(&body_bytes);
            
            // Parse JSON request
            let request_json: serde_json::Value = serde_json::from_str(&body_str)
                .unwrap_or_else(|_| serde_json::json!({"error": "Invalid JSON"}));
            
            // Process the request
            let input = request_json.get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("No input provided");
            
            let response = serde_json::json!({
                "success": true,
                "response": format!("Processed (TLS): {}", input),
                "timestamp": chrono::Utc::now().to_rfc3339()
            });
            
            Ok(hyper::Response::builder()
                .status(200)
                .header("content-type", "application/json")
                .body(Body::from(response.to_string()))
                .unwrap())
        },
        _ => {
            let not_found = serde_json::json!({
                "error": "Not Found",
                "message": format!("Endpoint {} {} not found", method, uri.path()),
                "timestamp": chrono::Utc::now().to_rfc3339()
            });
            Ok(hyper::Response::builder()
                .status(404)
                .header("content-type", "application/json")
                .body(Body::from(not_found.to_string()))
                .unwrap())
        }
    }
}

/// Generate self-signed certificate for development
pub fn generate_self_signed_cert() -> Result<(String, String)> {
    use rcgen::{Certificate, CertificateParams, DistinguishedName, KeyPair};
    use std::time::SystemTime;
    
    let mut params = CertificateParams::default();
    
    // Set certificate validity (1 year)
    params.not_before = (SystemTime::now() - std::time::Duration::from_secs(86400)).into();
    params.not_after = (SystemTime::now() + std::time::Duration::from_secs(365 * 86400)).into();
    
    // Set distinguished name
    let mut dn = DistinguishedName::new();
    dn.push(rcgen::DnType::CommonName, "localhost");
    dn.push(rcgen::DnType::OrganizationName, "Nexora AI Development");
    dn.push(rcgen::DnType::CountryName, "US");
    params.distinguished_name = dn;
    
    // Add subject alternative names
    params.subject_alt_names = vec![
        rcgen::SanType::DnsName("localhost".to_string()),
        rcgen::SanType::DnsName("127.0.0.1".to_string()),
        rcgen::SanType::IpAddress("127.0.0.1".parse().unwrap()),
    ];
    
    // Generate key pair
    let key_pair = KeyPair::generate(&rcgen::PKCS_ECDSA_P256_SHA256)?;
    params.key_pair = Some(key_pair);
    let cert = Certificate::from_params(params)?;
    
    // Serialize certificate and private key
    let cert_pem = cert.serialize_pem()?;
    let key_pem = cert.serialize_private_key_pem();
    
    info!("Generated self-signed certificate for development");
    
    Ok((cert_pem, key_pem))
}

/// Create TLS acceptor from certificate and key
pub fn create_tls_acceptor(cert_pem: &str, key_pem: &str) -> Result<tokio_rustls::TlsAcceptor> {
    use rustls::ServerConfig;
    use std::io::Cursor;
    
    // Parse certificate
    let cert_der = rustls_pemfile::certs(&mut Cursor::new(cert_pem.as_bytes()))
        .next()
        .ok_or_else(|| anyhow::anyhow!("No certificate found"))??;
    
    // Parse private key
    let mut cursor = Cursor::new(key_pem.as_bytes());
    let mut keys = rustls_pemfile::pkcs8_private_keys(&mut cursor);
    let key_der = keys.next()
        .ok_or_else(|| anyhow::anyhow!("No private key found"))??;
    
    // Create TLS config
    let config = ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(vec![rustls::Certificate(cert_der.to_vec())], rustls::PrivateKey(key_der.secret_pkcs8_der().to_vec()))?;
    
    Ok(tokio_rustls::TlsAcceptor::from(Arc::new(config)))
}
