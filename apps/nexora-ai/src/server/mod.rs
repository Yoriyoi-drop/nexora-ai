use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::info;

use crate::NexoraAI;

pub mod config;
pub mod handlers;
pub mod router;
pub mod tls;

pub struct NexoraServer {
    config: ServerConfig,
}

impl NexoraServer {
    pub fn new(config: ServerConfig) -> Self {
        Self { config }
    }

    pub async fn start(&self, nexora: Arc<NexoraAI>) -> Result<(), anyhow::Error> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port).parse()?;
        let listener = TcpListener::bind(addr).await?;
        info!("Nexora server listening on http://{}", addr);

        let app = router::create_router(nexora, &self.config).await?;

        if self.config.enable_tls {
            tls::start_tls_server(&self.config, listener, app).await
        } else {
            axum::serve(listener, app).await
                .map_err(|e| anyhow::anyhow!("Server error: {}", e))
        }
    }
}

pub use config::{ServerConfig, load_rustls_pem_file, load_rustls_private_key};
pub use handlers::*;
pub use router::create_router;
pub use tls::start_tls_server;
