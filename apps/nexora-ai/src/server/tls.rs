//! TLS server functionality

use std::net::SocketAddr;
use std::sync::Arc;
use anyhow::Result;
use tokio::net::TcpListener;
use axum::Router;
use tracing::{info, error};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tower::Service;

use crate::config::server::ServerConfig;

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

/// Handle individual TLS stream — routes through the axum Router
async fn handle_tls_stream(
    tls_stream: tokio_rustls::server::TlsStream<tokio::net::TcpStream>,
    app: Router,
) -> Result<()> {
    let peer_addr = tls_stream.get_ref().0.peer_addr()?;
    info!("Handling TLS connection from: {}", peer_addr);

    let (reader, mut writer) = tokio::io::split(tls_stream);
    let mut reader = BufReader::new(reader);

    // Parse request line: "METHOD URI HTTP/1.1\r\n"
    let mut request_line = String::new();
    reader.read_line(&mut request_line).await
        .map_err(|e| anyhow::anyhow!("Failed to read request line: {}", e))?;
    let request_line = request_line.trim_end_matches("\r\n");
    let parts: Vec<&str> = request_line.splitn(3, ' ').collect();
    if parts.len() != 3 {
        writer.write_all(b"HTTP/1.1 400 Bad Request\r\ncontent-length: 11\r\n\r\nBad Request\r\n").await?;
        return Ok(());
    }

    let method: axum::http::Method = parts[0].parse()?;
    let uri: axum::http::Uri = parts[1].parse()?;

    // Parse headers
    let mut header_map = axum::http::HeaderMap::new();
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        let trimmed = line.trim_end_matches("\r\n");
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            let key = k.trim();
            let value = v.trim();
            if key.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().ok();
            }
            if let Ok(name) = axum::http::HeaderName::from_bytes(key.as_bytes()) {
                if let Ok(val) = axum::http::HeaderValue::from_str(value) {
                    header_map.append(name, val);
                }
            }
        }
    }

    // Read body if present
    let body = if let Some(len) = content_length {
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf).await?;
        axum::body::Bytes::from(buf)
    } else {
        axum::body::Bytes::new()
    };
    
    let req = axum::http::Request::builder()
        .method(method)
        .uri(uri)
        .body(axum::body::Body::from(body))?;

    // Route through the axum Router
    let mut app = app;
    let res = Service::call(&mut app, req).await.unwrap_or_else(|e| match e {});
    let (parts, res_body) = res.into_parts();
    let status = parts.status;
    let body = axum::body::to_bytes(res_body, 10_000_000)
        .await
        .unwrap_or_default();

    let mut response = format!(
        "HTTP/1.1 {} {}\r\n",
        status.as_u16(),
        status.canonical_reason().unwrap_or("OK")
    );
    for (name, value) in parts.headers.iter() {
        response.push_str(&format!("{}: {}\r\n", name, value.to_str().unwrap_or("")));
    }
    response.push_str(&format!("content-length: {}\r\n", body.len()));
    response.push_str("\r\n");

    writer.write_all(response.as_bytes()).await?;
    writer.write_all(&body).await?;

    info!("TLS connection closed: {}", peer_addr);
    Ok(())
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
